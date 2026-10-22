"""跑黄金集，算检索与生成的评估指标。

为什么要自己写而不是直接用 Ragas：
    1. 我们的 ground truth 是「要点清单 + gold_chunk_ids」，天然可算，
       不需要 LLM 判官——零成本、结果可复现，跑一百次数字一样。
    2. Ragas 的 faithfulness 需要每条多调两次模型，80 条就是 160 次。
       等有预算了再加，先用免费指标把闭环跑起来。

指标：
    Recall@10   gold 块有几个进了前 5（多 gold 时按比例）
    MRR        第一个正确来源排第几的倒数
    Hit@5      前 5 里至少命中一个的比例
    Refuse     拒答题里「正确拒答」的比例（--with-answer 才算）

用法：
    # 只算检索（快，不花钱）
    python -m scripts.eval_run

    # 连生成一起测（每条一次问答）
    python -m scripts.eval_run --with-answer --limit 10

    # 跑完存基线；之后每次改动都跟基线比
    python -m scripts.eval_run --save-baseline
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
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
from app.db.models import Chunk, KnowledgeBase  # noqa: E402
from app.retrieve.service import retrieve  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS = ROOT / "data" / "golden" / "questions.jsonl"
BASELINE = ROOT / "data" / "golden" / "baseline.json"

REFUSE_HINTS = ("未覆盖", "没有提供", "知识库中没有", "无法回答", "资料不足", "不包含")


def load_questions(types: set[str] | None, limit: int) -> list[dict]:
    items = [json.loads(line) for line in QUESTIONS.read_text(encoding="utf-8").splitlines() if line.strip()]
    items = [item for item in items if item.get("status") in ("approved", "draft")]
    if types:
        items = [item for item in items if item["type"] in types]
    return items[:limit] if limit else items


async def expand_gold(session, gold_ids: list[str]) -> set[str]:
    """把 gold 扩展到同一父块下的所有兄弟块。

    一个知识点常被切成 2-3 个块，只标其中一个会让 Recall 天然偏低——
    检索命中了"另一个块"其实也是对的，只是标的时候没标上。
    """
    rows = (
        await session.execute(
            select(Chunk.id, Chunk.parent_id).where(Chunk.id.in_(gold_ids))
        )
    ).all()
    expanded = {chunk_id for chunk_id, _ in rows}
    parents = {parent_id for _, parent_id in rows if parent_id}
    if parents:
        siblings = (
            await session.scalars(
                select(Chunk.id).where(Chunk.parent_id.in_(parents))
            )
        ).all()
        expanded.update(siblings)
    return expanded


async def pick_kb(session) -> str:
    """取文档最多的库。库里可能还有"入库调试库"这类空库，
    按 limit(1) 拿到它会导致所有检索都命中不了。"""
    return await session.scalar(
        select(KnowledgeBase.id).order_by(KnowledgeBase.doc_count.desc()).limit(1)
    )


async def main() -> int:
    parser = argparse.ArgumentParser(description="黄金集评估")
    parser.add_argument("--types", default="concept,compare,multihop")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--with-answer", action="store_true", help="连生成一起测（花钱）")
    parser.add_argument("--save-baseline", action="store_true")
    parser.add_argument("--threshold", type=float, default=None,
                        help="覆盖 rerank_score_threshold，用于 A/B 对比（不改配置文件）")
    args = parser.parse_args()

    setup_logging()
    if args.threshold is not None:
        settings.retrieval.rerank_score_threshold = args.threshold
        print(f"[A/B] 阈值临时覆盖为 {args.threshold}（仅本次运行）")
    types = {t.strip() for t in args.types.split(",") if t.strip()}
    items = load_questions(types | {"refuse"}, args.limit)
    if not items:
        print("没有可评估的题目")
        return 1

    async with SessionLocal() as session:
        kb_id = await pick_kb(session)
    if not kb_id:
        print("没有知识库")
        return 1

    index = MilvusIndex()
    embedder = build_embedder()
    reranker = BGEReranker()
    llm = LLMClient(settings.llm.active()) if args.with_answer else None

    stats: dict[str, dict[str, float]] = defaultdict(
        lambda: {"n": 0, "recall": 0.0, "mrr": 0.0, "hit": 0.0}
    )
    misses: list[dict] = []
    refuse_total = refuse_ok = 0
    started = time.perf_counter()

    for order, item in enumerate(items, 1):
        kind = item["type"]
        question = item["question"]

        if kind == "refuse":
            if not args.with_answer:
                continue
            refuse_total += 1
            try:
                from app.generate.answer import answer_question

                result = await answer_question(
                    question, kb_ids=[kb_id], embedder=embedder, index=index,
                    reranker=reranker, llm=llm,
                )
                text = result.answer or ""
                # 拒答的判据：说的是"没覆盖"，且没有编出引用
                if any(hint in text for hint in REFUSE_HINTS) and not result.citations:
                    refuse_ok += 1
                else:
                    misses.append(
                        {"id": item["id"], "type": kind, "question": question,
                         "reason": f"未拒答（引用 {len(result.citations)} 条）"}
                    )
            except Exception as exc:
                misses.append({"id": item["id"], "type": kind, "question": question,
                               "reason": f"异常 {type(exc).__name__}"})
            continue

        raw_gold = list(item.get("gold_chunk_ids") or [])
        if not raw_gold:
            continue
        async with SessionLocal() as session:
            gold = await expand_gold(session, raw_gold)

        try:
            outcome = await retrieve(
                question, kb_ids=[kb_id], embedder=embedder, index=index, reranker=reranker, llm=llm
            )
        except Exception as exc:
            misses.append({"id": item["id"], "type": kind, "question": question,
                           "reason": f"检索异常 {type(exc).__name__}"})
            continue

        ranked = [hit.chunk_id for hit in outcome.hits[:10]]
        found = gold & set(ranked)
        recall = len(found) / len(gold)
        rank = next((i + 1 for i, cid in enumerate(ranked) if cid in gold), 0)
        mrr = 1.0 / rank if rank else 0.0

        bucket = stats[kind]
        bucket["n"] += 1
        bucket["recall"] += recall
        bucket["mrr"] += mrr
        bucket["hit"] += 1.0 if found else 0.0

        if not found:
            misses.append({"id": item["id"], "type": kind, "question": question,
                           "reason": f"前 10 未命中（gold {len(gold)} 块）"})

        if order % 10 == 0:
            print(f"  …{order}/{len(items)}")

    elapsed = time.perf_counter() - started
    print("\n" + "=" * 62)
    print(f"{'类型':10} {'条数':>5} {'Recall@10':>10} {'MRR':>8} {'Hit@5':>8}")
    print("-" * 62)
    report = {}
    for kind in ("concept", "compare", "multihop"):
        bucket = stats.get(kind)
        if not bucket or not bucket["n"]:
            continue
        n = bucket["n"]
        row = {
            "n": n,
            "recall@10": round(bucket["recall"] / n, 3),
            "mrr": round(bucket["mrr"] / n, 3),
            "hit@5": round(bucket["hit"] / n, 3),
        }
        report[kind] = row
        print(f"{kind:10} {n:>5} {row['recall@10']:>10.3f} {row['mrr']:>8.3f} {row['hit@5']:>8.3f}")
    if refuse_total:
        report["refuse"] = {"n": refuse_total, "accuracy": round(refuse_ok / refuse_total, 3)}
        print(f"{'refuse':10} {refuse_total:>5} {'拒答正确率':>10} {report['refuse']['accuracy']:>14.3f}")
    print("-" * 62)
    print(f"耗时 {elapsed:.1f}s | 未命中 {len(misses)} 条")

    if misses:
        print("\n未命中明细（前 8 条）：")
        for miss in misses[:8]:
            print(f"  [{miss['id']}] {miss['question'][:32]} → {miss['reason']}")

    if BASELINE.exists() and not args.save_baseline:
        previous = json.loads(BASELINE.read_text(encoding="utf-8"))
        print("\n与基线对比：")
        for kind, row in report.items():
            old = previous.get("metrics", {}).get(kind, {})
            for key in ("recall@10", "mrr", "hit@5", "accuracy"):
                if key in row and key in old:
                    delta = row[key] - old[key]
                    flag = "↑" if delta > 0.005 else ("↓" if delta < -0.005 else "=")
                    print(f"  {kind:9} {key:9} {old[key]:.3f} → {row[key]:.3f} {flag}")

    if args.save_baseline:
        payload = {
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "kb_id": kb_id,
            "collection": settings.milvus.collection,
            "embedding": settings.embedding.provider,
            "metrics": report,
            "misses": len(misses),
        }
        BASELINE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n基线已保存 → {BASELINE}")

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
