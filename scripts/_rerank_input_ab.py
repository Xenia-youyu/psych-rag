"""验证「先回填父块再重排」这个假设：子块 vs 父块，哪个重排更准？

代码注释里写着「拿 1200 字的父块打分，上下文完整，判别质量明显更高」，
但这条结论从没被验证过。父块变长也会稀释相关性——短问句对 1200 字，
命中句只占 1/8，分数可能反而更低。

方法：同一批候选，分别用子块正文、父块正文各打一次分，比金标的名次与分数。
只读，不改代码、不改配置。
"""

from __future__ import annotations

import asyncio
import json
import statistics
from pathlib import Path

from sqlalchemy import select

from app.clients.embed_client import build_embedder
from app.clients.milvus_client import MilvusIndex
from app.clients.rerank_client import BGEReranker
from app.config import settings
from app.core.logging import setup_logging
from app.db.base import SessionLocal
from app.db.models import Chunk, KnowledgeBase
from app.retrieve.fusion import dedupe_by_parent, rrf_fuse
from app.retrieve.service import _backfill_parents, _recall_one

GOLDEN = Path(__file__).resolve().parent.parent / "data" / "golden" / "questions.jsonl"


async def expand_gold(session, ids: list[str]) -> set[str]:
    rows = (await session.execute(select(Chunk.id, Chunk.parent_id).where(Chunk.id.in_(ids)))).all()
    out = {cid for cid, _ in rows}
    parents = {pid for _, pid in rows if pid}
    if parents:
        out.update((await session.scalars(select(Chunk.id).where(Chunk.parent_id.in_(parents)))).all())
    return out


async def rank_of(gold: set[str], hits, scores) -> tuple[float | None, int | None, float]:
    pairs = sorted(zip(hits, scores), key=lambda x: -x[1])
    for rank, (hit, sc) in enumerate(pairs, 1):
        if hit.chunk_id in gold:
            return float(sc), rank, float(pairs[0][1])
    return None, None, float(pairs[0][1])


async def engine_dispose() -> None:
    from app.db.base import engine
    await engine.dispose()


async def main() -> None:
    setup_logging()
    items = [json.loads(l) for l in GOLDEN.read_text(encoding="utf-8").splitlines() if l.strip()]
    items = [i for i in items if i.get("status") in ("approved", "draft") and i["type"] == "concept"]

    session = SessionLocal()
    kb_id = await session.scalar(
        select(KnowledgeBase.id).order_by(KnowledgeBase.doc_count.desc()).limit(1)
    )

    emb, idx, rr = build_embedder(), MilvusIndex(), BGEReranker()
    conf = settings.retrieval

    child_scores: list[float] = []
    parent_scores: list[float] = []
    child_hits = parent_hits = 0
    better_parent = better_child = 0
    rows: list[tuple[str, float | None, int | None, float | None, int | None]] = []

    for n, item in enumerate(items, 1):
        try:
            gold = await expand_gold(session, item.get("gold_chunk_ids") or [])
        except Exception as exc:
            print(f"  [{n:2}] {item['id']} 取 gold 失败：{type(exc).__name__}")
            continue
        embedding = (await emb.encode([item["question"]]))[0]
        raw = await _recall_one(
            embedding, kb_ids=[kb_id], index=idx,
            dense_topk=conf.dense_topk, sparse_topk=conf.sparse_topk, hybrid=conf.hybrid,
        )
        if not raw:
            continue
        hits = dedupe_by_parent(rrf_fuse({"q0": raw}, k=conf.rrf_k))
        if not hits:
            continue

        # A：用子块正文（当前 chunk 自带的内容）打分
        child_texts = [h.content for h in hits]
        s_child = await rr.score(item["question"], child_texts)
        cs, cr, _ = await rank_of(gold, hits, s_child)

        # B：回填父块之后再打分
        backfilled = await _backfill_parents(hits, index=idx)
        s_parent = await rr.score(item["question"], [h.content for h in backfilled])
        ps, pr, _ = await rank_of(gold, backfilled, s_parent)

        rows.append((item["id"], cs, cr, ps, pr))
        if cs is not None:
            child_scores.append(cs)
            child_hits += 1
        if ps is not None:
            parent_scores.append(ps)
            parent_hits += 1
        if cs is not None and ps is not None:
            if ps > cs:
                better_parent += 1
            elif cs > ps:
                better_child += 1

        flag = ""
        if cs is not None and ps is not None and abs(ps - cs) > 0.03:
            flag = "  ← 父块更好" if ps > cs else "  ← 子块更好"
        print(f"  [{n:2}] {item['id']} 子块={cs if cs is None else f'{cs:.4f}'}"
              f"(#{cr}) 父块={ps if ps is None else f'{ps:.4f}'}(#{pr}){flag}")

    print("\n" + "=" * 66)
    if child_scores and parent_scores:
        print(f"金标平均分：子块 {statistics.mean(child_scores):.4f} | 父块 {statistics.mean(parent_scores):.4f}")
        print(f"金标中位分：子块 {statistics.median(child_scores):.4f} | 父块 {statistics.median(parent_scores):.4f}")
    print(f"金标出现在候选里：子块 {child_hits}/{len(items)} | 父块 {parent_hits}/{len(items)}")
    print(f"逐题对比（两者都有分）：父块更高 {better_parent} 题，子块更高 {better_child} 题")
    await session.close()
    await engine_dispose()



if __name__ == "__main__":
    asyncio.run(main())
