"""从现有语料反向生成黄金集候选问题。

为什么是"反向"：先读块、再写题。反过来（先想问题再找出处）写出来的题
有一半在库里找不到 ground truth，只能删——等于白写。

四类问题的配额来自真实提问分布，不是平均分：

| 类型 | 条数 | 测什么 | 判据 |
|---|---|---|---|
| concept | 30 | 单点检索 + 忠实复述 | gold 块进 top5 + 要点命中 ≥ 2/3 |
| compare | 20 | 多路召回的覆盖 | 两个概念的块都进 top5 |
| multihop | 15 | 跨块/跨文档串联 | 至少命中 2 个不同块 |
| refuse | 15 | 不编造 | 拒答话术 + 引用数为 0 |

拒答题是手工写死的：必须是「看起来该有答案，但库里确实没有」的问题。
库里没有变态心理学、没有 DSM、没有认知神经科学——这些领域的问题
正好用来测它会不会硬编。

生成的只是**候选**，`status: draft`。要用之前得人过一遍：
LLM 出的题经常"贴着原文问"，测出来虚高。

用法：
    python -m scripts.build_golden --concept 30 --compare 20 --multihop 15
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select  # noqa: E402

from app.clients.llm_client import LLMClient  # noqa: E402
from app.clients.milvus_client import MilvusIndex  # noqa: E402
from app.config import settings  # noqa: E402
from app.core.logging import setup_logging  # noqa: E402
from app.db.base import SessionLocal  # noqa: E402
from app.db.models import Chunk, Document  # noqa: E402

OUT_DEFAULT = Path(__file__).resolve().parent.parent / "data" / "golden" / "questions.jsonl"

# 只有教材类文档才有"知识点"。哲学著作和写作规范里抽出来的问题
# 会是"某人哪年发了什么文章""期刊多久出一期"——那不是知识问答。
DOC_INCLUDE = ("实验心理学", "教育心理学", "研究方法", "儿童心理学", "心理咨询", "心理学导论")
DOC_EXCLUDE = ("论文写作规范", "人的条件", "人类与大地母亲", "论人的天性", "逻辑图")

# 文献引用、期刊信息、版权声明——都不该变成题目
NOISE = re.compile(
    r"年第\s*\d+\s*期|第\s*\d+\s*卷|出版社|责任编辑|收稿日期|"
    r"ISBN|期刊|投稿|征稿|订阅|邮编|印刷|开本|定价"
)

CONCEPT_PROMPT = """下面是心理学教材的一段原文。请基于它出一道问答题。

原文：
{content}

要求：
1. 用学生真实会问的自然问法，8-30 字；不要用「根据上文」「这段材料中」这类措辞
2. 不能照抄原文里连续 8 个字以上的表述（否则靠关键词就能命中，测不出语义检索）
3. 这个问题要能且仅能由上面这段原文回答
4. key_points 写 2-4 条答案要点，每条不超过 25 字，是短语不是整句

只输出 JSON：{{"question": "...", "key_points": ["...", "..."]}}"""

COMPARE_PROMPT = """下面是同一本教材中的两段原文，分别讲两个相关概念。
请出一道辨析题，要求同时用到两段内容。

原文 A：
{content_a}

原文 B：
{content_b}

要求：
1. 问法自然是「A 和 B 有什么区别」「A 与 B 的关系是什么」这类
2. 不能照抄原文里连续 8 个字以上的表述
3. key_points 写 3-4 条，覆盖两者的关键差异

只输出 JSON：{{"question": "...", "key_points": ["...", "..."]}}"""

MULTIHOP_PROMPT = """下面是两本不同教材的段落。请出一道需要综合两者才能回答的题。

材料一（{doc_a}）：
{content_a}

材料二（{doc_b}）：
{content_b}

要求：
1. 问题必须同时依赖两份材料，只读一份答不全
2. 不能照抄原文里连续 8 个字以上的表述
3. key_points 写 2-4 条，每条注明来自哪一份材料更好

只输出 JSON：{{"question": "...", "key_points": ["...", "..."]}}"""


# 库里确实没有这些领域的内容（没有变态心理学、认知神经科学、量表手册）
REFUSE_CASES = [
    "DSM-5 中孤独症谱系障碍的诊断标准有哪些条目",
    "认知行为疗法治疗惊恐障碍的标准疗程是几周",
    "韦氏成人智力量表第四版的四个指数分别是什么",
    "儿童多动症的诊断需要满足哪几条 DSM 标准",
    "fMRI 研究里血氧水平依赖信号的延迟是多少毫秒",
    "眼动仪采样率对注视点识别精度有什么影响",
    "正念减压疗法 MBSR 的八周课程具体怎么安排",
    "焦虑自评量表 SAS 的划界分是多少",
    "精神分裂症前驱期筛查用哪个量表",
    "深度学习模型在面部表情识别上的准确率是多少",
    "ERP 成分 P300 的潜伏期正常范围是多少",
    "老年人轻度认知障碍的蒙特利尔量表评分标准",
    "创伤后应激障碍的暴露疗法分几个阶段",
    "孤独症早期筛查工具 M-CHAT 的条目数",
    "抗抑郁药物 SSRI 的起效时间通常是多久",
]


def longest_common(a: str, b: str) -> int:
    """最长公共子串长度，用来防止问题直接抄原文。"""
    if not a or not b:
        return 0
    best = 0
    prev = [0] * (len(b) + 1)
    for ch in a:
        cur = [0] * (len(b) + 1)
        for idx, other in enumerate(b, 1):
            if ch == other:
                cur[idx] = prev[idx - 1] + 1
                best = max(best, cur[idx])
        prev = cur
    return best


def clean_question(text: str) -> str:
    return re.sub(r"\s+", "", (text or "").strip().strip("？?。")) + "？"


async def generate_one(
    client: LLMClient, prompt: str, content_a: str, content_b: str = ""
) -> dict | None:
    data = None
    # 并发高了会撞超时（client 的 timeout=30s），返回空串而不是报错。
    # 这类失败重试一次基本就过，比整体重跑便宜。
    for attempt in range(2):
        try:
            data = await client.chat_json(
                [{"role": "user", "content": prompt}], temperature=0.7, max_tokens=900
            )
            break
        except Exception:
            if attempt == 0:
                await asyncio.sleep(1.5)
    if data is None:
        return None

    question = clean_question(str(data.get("question", "")))
    points = [str(p).strip() for p in (data.get("key_points") or []) if str(p).strip()]
    if not (8 <= len(question) <= 40) or not (2 <= len(points) <= 5):
        return None
    # 抄袭检查：和原文连续重合超过 10 字，说明这题不用检索、关键词就能答
    if longest_common(question, content_a) > 10:
        return None
    if content_b and longest_common(question, content_b) > 10:
        return None
    return {"question": question, "key_points": points}


async def main() -> int:
    parser = argparse.ArgumentParser(description="生成黄金集候选")
    parser.add_argument("--concept", type=int, default=30)
    parser.add_argument("--compare", type=int, default=20)
    parser.add_argument("--multihop", type=int, default=15)
    parser.add_argument("--refuse", type=int, default=15)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--out", default=str(OUT_DEFAULT))
    args = parser.parse_args()

    setup_logging()

    async with SessionLocal() as session:
        docs = {d.id: d.filename for d in (await session.execute(select(Document.id, Document.filename))).all()}
        rows = (
            await session.execute(
                select(Chunk.id, Chunk.doc_id, Chunk.page_start, Chunk.content, Chunk.kb_id).where(
                    Chunk.char_count.between(220, 500)
                )
            )
        ).all()

    by_doc: dict[str, list] = defaultdict(list)
    kb_ids: set[str] = set()
    for chunk_id, doc_id, page, content, kb_id in rows:
        name = docs.get(doc_id)
        if not name:
            continue
        if any(word in name for word in DOC_EXCLUDE):
            continue
        if DOC_INCLUDE and not any(word in name for word in DOC_INCLUDE):
            continue
        if NOISE.search(content):
            continue
        by_doc[name].append((chunk_id, page, content))
        kb_ids.add(kb_id)

    if not by_doc:
        print("没有可用语料")
        return 1

    random.seed(20261022)
    doc_names = sorted(by_doc)

    # ---- 概念题：每个文档轮流取，保证覆盖面 ----
    concept_jobs = []
    cursor = 0
    while len(concept_jobs) < args.concept:
        name = doc_names[cursor % len(doc_names)]
        pool = by_doc[name]
        if pool:
            concept_jobs.append(random.choice(pool))
        cursor += 1
        if cursor > args.concept * 6:
            break

    # ---- 对比 / 多跳：必须挑内容相关的两块 ----
    # 随机抽两块来"对比"是行不通的：模型看到两段毫不相干的文字，
    # 会直接返回空。用向量近邻找主题相近的块，同一文档内的做对比，
    # 跨文档的做多跳。
    index = MilvusIndex()
    kb_list = sorted(kb_ids)
    lookup = {item[0]: (name, item) for name in by_doc for item in by_doc[name]}

    async def related_pair(seed_id: str, *, same_doc: bool):
        vector = await index.get_vector(seed_id)
        if vector is None:
            return None
        try:
            hits = await index.search(kb_ids=kb_list, dense=vector, topk=30)
        except Exception:
            return None
        seed_name = lookup[seed_id][0]
        for hit in hits:
            other_id = hit.get("chunk_id")
            if other_id == seed_id or other_id not in lookup:
                continue
            other_name = lookup[other_id][0]
            is_same = other_name == seed_name
            if is_same != same_doc or not same_doc and is_same:
                continue
            if same_doc and not is_same:
                continue
            return seed_name, lookup[seed_id][1], other_name, lookup[other_id][1]
        return None

    seeds = [item for name in doc_names for item in by_doc[name]]
    random.shuffle(seeds)

    compare_jobs = []
    for item in seeds:
        if len(compare_jobs) >= args.compare:
            break
        pair = await related_pair(item[0], same_doc=True)
        if pair:
            name, a, _other_name, b = pair
            compare_jobs.append((name, a, b))

    multihop_jobs = []
    for item in seeds:
        if len(multihop_jobs) >= args.multihop:
            break
        pair = await related_pair(item[0], same_doc=False)
        if pair:
            _n1, a, n2, b = pair
            multihop_jobs.append((lookup[item[0]][0], a, n2, b))

    client = LLMClient(settings.llm.active())
    sem = asyncio.Semaphore(args.workers)
    items: list[dict] = []

    async def run(kind: str, prompt: str, a_text: str, b_text: str, meta: dict) -> None:
        async with sem:
            result = await generate_one(client, prompt, a_text, b_text)
        if result:
            items.append({"type": kind, **result, **meta, "status": "draft"})

    tasks = []
    for chunk_id, page, content in concept_jobs:
        tasks.append(
            run(
                "concept",
                CONCEPT_PROMPT.format(content=content[:1200]),
                content,
                "",
                {"gold_chunk_ids": [chunk_id], "source_page": page},
            )
        )
    for name, a, b in compare_jobs:
        tasks.append(
            run(
                "compare",
                COMPARE_PROMPT.format(content_a=a[2][:900], content_b=b[2][:900]),
                a[2],
                b[2],
                {"gold_chunk_ids": [a[0], b[0]], "source_page": a[1], "doc": name},
            )
        )
    for n1, a, n2, b in multihop_jobs:
        tasks.append(
            run(
                "multihop",
                MULTIHOP_PROMPT.format(
                    doc_a=n1, content_a=a[2][:800], doc_b=n2, content_b=b[2][:800]
                ),
                a[2],
                b[2],
                {"gold_chunk_ids": [a[0], b[0]], "source_page": a[1], "doc": f"{n1} / {n2}"},
            )
        )

    print(f"生成中：概念 {len(concept_jobs)} / 对比 {len(compare_jobs)} / 多跳 {len(multihop_jobs)}")
    await asyncio.gather(*tasks)

    # 拒答题不需要生成，直接写死
    for question in REFUSE_CASES[: args.refuse]:
        items.append(
            {
                "type": "refuse",
                "question": question,
                "key_points": ["应说明知识库未覆盖该内容", "不应给出具体条目或数值"],
                "gold_chunk_ids": [],
                "source_page": 0,
                "status": "draft",
            }
        )

    order = {"concept": 0, "compare": 1, "multihop": 2, "refuse": 3}
    items.sort(key=lambda item: (order.get(item["type"], 9), item["question"]))
    for index, item in enumerate(items, 1):
        prefix = {"concept": "c", "compare": "m", "multihop": "h", "refuse": "r"}[item["type"]]
        item["id"] = f"{prefix}{index:03d}"

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as sink:
        for item in items:
            sink.write(json.dumps(item, ensure_ascii=False) + "\n")

    from collections import Counter

    counts = Counter(item["type"] for item in items)
    print(f"\n生成 {len(items)} 条 → {out}")
    for kind in ("concept", "compare", "multihop", "refuse"):
        print(f"  {kind:9} {counts.get(kind, 0):>3}")
    print("\n全部为 draft，需要人工过一遍再改成 approved")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
