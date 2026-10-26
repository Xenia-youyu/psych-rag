"""生成侧评估：测「答案对不对」，补齐只测检索的缺口。

三个指标都不需要 LLM 判官，纯计算、零额外成本、可复现：

1. 要点命中率
   金标里的 key_points 是清单（「撤销厌恶刺激」「行为频率增加」）。
   把答案和要点都去掉空白与标点后做子串匹配，命中 >= 2/3 算过。
   这是最粗的口径——同义改写会漏判，但它能抓住「答非所问」和「只答一半」。

2. 引用完整性
   - citations：模型拿到几段上下文
   - dangling：答案里引用了不存在的编号（generate() 已经算好）
   - marked：答案里实际出现了 [S1] 之类标记的条数
   真实用户会点引用跳原文，编号错了就是死链。

3. 拒答正确率
   拒答题该输出「未覆盖」；知识题输出「未覆盖」就是漏答。
   两个方向都算，只报一个会掩盖问题。

用法：
    python -m scripts.eval_generation                # 全部可用题
    python -m scripts.eval_generation --limit 6      # 冒烟
    python -m scripts.eval_generation --types concept
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select  # noqa: E402

from app.clients.embed_client import build_embedder  # noqa: E402
from app.clients.llm_client import LLMClient  # noqa: E402
from app.clients.milvus_client import MilvusIndex  # noqa: E402
from app.clients.rerank_client import BGEReranker  # noqa: E402
from app.config import settings  # noqa: E402
from app.core.logging import setup_logging  # noqa: E402
from app.db.base import SessionLocal  # noqa: E402
from app.db.models import KnowledgeBase  # noqa: E402
from app.generate.answer import generate  # noqa: E402
from app.retrieve.service import retrieve  # noqa: E402

GOLDEN = Path(__file__).resolve().parent.parent / "data" / "golden" / "questions.jsonl"

# 拒答题的判据：答案里有没有**声明**这块内容不在库里。
# 不能只认「纯模板拒答」——实测 15 条里 3 条给模板、13 条是「当前知识库未覆盖 X」
# 之后又补了相邻可用的背景，后者是更好的回答，却被旧口径算成失败。
DECLINE_MARKERS = (
    "未覆盖", "无法", "没有", "未给出", "查不到", "不包含",
    "未提供", "不能列出", "无相关资料", "未涉及", "不具备",
)
OUT = Path(__file__).resolve().parent.parent / "data" / "golden" / "generation_baseline.json"

# 去掉空白与常见中英文标点，只留字。要点匹配和引用统计都在这上面做
_PUNCT = re.compile(r"[\s，。；：、！？（）()《》「」『』“”‘’,.;:!?\-—…·\[\]【】<>/\\|~`\"']")


def norm(text: str) -> str:
    return _PUNCT.sub("", text or "")


def longest_common_block(a: str, b: str) -> int:
    """最长公共连续子串长度。用于容忍同义改写：
    要点「学得更快」 vs 答案「学得快」→ 公共块「学得快」= 3 字。
    """
    if not a or not b:
        return 0
    from difflib import SequenceMatcher

    return SequenceMatcher(None, a, b, autojunk=False).find_longest_match(
        0, len(a), 0, len(b)
    ).size


def point_hits(answer: str, points: list[str]) -> tuple[int, list[str]]:
    """返回 (命中数, 没命中的要点)。

    第一版用严格子串匹配，实测 0% —— 答案写「学得快」「促进生理发展」，
    要点写「学得更快」「促进生理发育」，一字之差全判错。
    改成「最长公共连续子串 / 要点长度 >= 0.65」：
    允许同义改写，但仍要求核心词块出现，换成完全不同的说法依然判错。
    """
    flat = norm(answer)
    missed: list[str] = []
    hit = 0
    for point in points:
        p = norm(point)
        if not p:
            continue
        if p in flat:
            hit += 1
            continue
        ratio = longest_common_block(p, flat) / len(p)
        if ratio >= 0.65:
            hit += 1
        else:
            missed.append(point)
    return hit, missed


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


async def semantic_point_hits(emb, answer: str, points: list[str], threshold: float):
    """要点与整段答案的向量相似度。

    词面匹配对同义改写完全失效：实测严格子串 0%，连"学得快 vs 学得更快"都判错。
    所以加这一路：要点和答案各编码一次，余弦 >= 阈值算命中。
    阈值不是拍的——标定时真要点 0.48~0.68、无关对照 0.17~0.27，取中间偏保守。
    """
    if not points:
        return 0, [], []
    vecs = await emb.encode(list(points) + [answer])
    ans = vecs[-1].dense
    sims, hit, missed = [], 0, []
    for point, vec in zip(points, vecs[:-1]):
        sim = cosine(vec.dense, ans)
        sims.append(round(sim, 3))
        if sim >= threshold:
            hit += 1
        else:
            missed.append(point)
    return hit, missed, sims


async def eval_one(item, kb_id, emb, idx, rr, llm, sem, *, sim_threshold: float) -> dict:
    async with sem:
        q = item["question"]
        outcome = await retrieve(
            q, kb_ids=[kb_id], embedder=emb, index=idx, reranker=rr, llm=llm
        )
        result = await generate(llm, question=q, hits=outcome.hits, intent=outcome.intent)

        points = item.get("key_points") or []
        # 先走词面（便宜、精确），没命中的再交给语义通道
        hit, missed = point_hits(result.text, points) if points else (0, [])
        sims: list[float] = []
        if missed:
            extra_hit, still_missed, sims = await semantic_point_hits(
                emb, result.text, missed, sim_threshold
            )
            hit += extra_hit
            missed = still_missed
        markers = set(re.findall(r"\[S(\d+)\]", result.text or ""))

        return {
            "id": item["id"],
            "type": item["type"],
            "question": q,
            "answered": not result.no_context,
            "declined": any(m in (result.text or "")[:300] for m in DECLINE_MARKERS),
            "chars": len(result.text or ""),
            "citations": len(result.citations or []),
            "marked": len(markers),
            "dangling": len(result.dangling_citations or []),
            "points_total": len(points),
            "points_hit": hit,
            "points_missed": missed,
            "point_sims": sims,
            "tokens_out": result.tokens_out,
            "answer": result.text or "",
            "answer_head": (result.text or "")[:160].replace("\n", " "),
        }


async def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--types", default="concept,compare,multihop,refuse")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--show", type=int, default=3)
    parser.add_argument("--sim-threshold", type=float, default=0.42,
                        help="语义命中阈值（标定：真要点 0.48-0.68 / 无关 0.17-0.27）")
    args = parser.parse_args()

    wanted = {t.strip() for t in args.types.split(",") if t.strip()}
    items = [
        json.loads(l)
        for l in GOLDEN.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    items = [
        i for i in items
        if i.get("status") in ("draft", "approved") and i["type"] in wanted
    ]
    if args.limit:
        items = items[: args.limit]
    if not items:
        print("没有可评估的题")
        return

    async with SessionLocal() as s:
        kb_id = await s.scalar(
            select(KnowledgeBase.id).order_by(KnowledgeBase.doc_count.desc()).limit(1)
        )
    emb, idx, rr = build_embedder(), MilvusIndex(), BGEReranker()
    llm = LLMClient(settings.llm.active())
    sem = asyncio.Semaphore(args.workers)

    print(f"评估 {len(items)} 条（阈值 {settings.retrieval.rerank_score_threshold}，"
          f"重排窗口 {settings.retrieval.rerank_max_candidates}，并发 {args.workers}）")

    tasks = [
        eval_one(i, kb_id, emb, idx, rr, llm, sem, sim_threshold=args.sim_threshold)
        for i in items
    ]
    results = []
    done = 0
    for coro in asyncio.as_completed(tasks):
        results.append(await coro)
        done += 1
        if done % 5 == 0:
            print(f"  …{done}/{len(items)}", flush=True)

    results.sort(key=lambda r: r["id"])
    by_type: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_type[r["type"]].append(r)

    print("\n" + "=" * 78)
    print(f"{'类型':10} {'条数':>4} {'作答率':>7} {'要点命中':>9} {'要点过关':>9} {'悬空引用':>9} {'均引用':>7}")
    print("-" * 78)
    for t, rows in by_type.items():
        n = len(rows)
        answered = sum(1 for r in rows if r["answered"])
        with_points = [r for r in rows if r["points_total"]]
        if with_points:
            rate = sum(r["points_hit"] / r["points_total"] for r in with_points) / len(with_points)
            passed = sum(1 for r in with_points if r["points_hit"] / r["points_total"] >= 2 / 3)
            passed_s = f"{passed}/{len(with_points)}"
        else:
            rate, passed_s = 0.0, "—"
        dangling = sum(r["dangling"] for r in rows)
        avg_cit = sum(r["citations"] for r in rows) / n
        print(f"{t:10} {n:>4} {answered / n:>7.0%} {rate:>9.0%} {passed_s:>9} {dangling:>9} {avg_cit:>7.1f}")

    refuse = by_type.get("refuse", [])
    if refuse:
        declined = sum(1 for r in refuse if r["declined"])
        template = sum(1 for r in refuse if not r["answered"])
        print(f"\n拒答题 {len(refuse)} 条：")
        print(f"  声明未覆盖（主指标）  {declined}/{len(refuse)} = {declined / len(refuse):.0%}")
        print(f"  纯模板拒答（次要）    {template}/{len(refuse)} = {template / len(refuse):.0%}")
        for r in refuse:
            if not r["declined"]:
                print(f"  ✗ 未声明 [{r['id']}] {r['question'][:34]}")
                print(f"     答: {r['answer_head'][:90]}")

    knowledge = [r for t, rows in by_type.items() if t != "refuse" for r in rows]
    missed_ctx = [r for r in knowledge if not r["answered"]]
    if missed_ctx:
        print(f"\n该答未答（检索有结果却输出「未覆盖」）：{len(missed_ctx)} 条")
        for r in missed_ctx[:5]:
            print(f"  ✗ [{r['id']}] {r['question'][:40]}")

    low = sorted(
        [r for r in knowledge if r["points_total"] and r["points_hit"] / r["points_total"] < 2 / 3],
        key=lambda r: r["points_hit"] / r["points_total"],
    )
    if low and args.show:
        print(f"\n要点命中不足 2/3 的样例（前 {min(args.show, len(low))} 条）：")
        for r in low[: args.show]:
            print(f"  [{r['id']}] {r['question']}")
            print(f"     命中 {r['points_hit']}/{r['points_total']} | 漏: {' / '.join(r['points_missed'][:3])}")
            print(f"     答: {r['answer_head'][:110]}")

    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n明细已存 → {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
