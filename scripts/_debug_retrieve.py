"""逐段追踪指定题目的 gold 位次，判断是召回问题还是标注问题。

只调 embedding API + Milvus + 本地重排，不调生成侧 LLM。
用法：python -m scripts._debug_retrieve <题目在 jsonl 里的行号(1-based)>
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import func, select  # noqa: E402

from app.clients.embed_client import build_embedder  # noqa: E402
from app.clients.milvus_client import MilvusIndex  # noqa: E402
from app.clients.rerank_client import BGEReranker  # noqa: E402
from app.config import settings  # noqa: E402
from app.core.logging import setup_logging  # noqa: E402
from app.db.base import SessionLocal  # noqa: E402
from app.db.models import Chunk, Document, KnowledgeBase  # noqa: E402
from app.retrieve.fusion import dedupe_by_parent, filter_by_threshold, Hit  # noqa: E402
from app.retrieve.service import retrieve  # noqa: E402

GOLDEN = Path(__file__).resolve().parent.parent / "data" / "golden" / "questions.jsonl"


def load_item(index: int) -> dict:
    lines = [l for l in GOLDEN.read_text(encoding="utf-8").splitlines() if l.strip()]
    item = json.loads(lines[index - 1])
    return item


async def expand_gold(session, gold_ids: list[str]) -> set[str]:
    rows = (
        await session.execute(select(Chunk.id, Chunk.parent_id).where(Chunk.id.in_(gold_ids)))
    ).all()
    expanded = {cid for cid, _ in rows}
    parents = {pid for _, pid in rows if pid}
    if parents:
        siblings = (
            await session.scalars(select(Chunk.id).where(Chunk.parent_id.in_(parents)))
        ).all()
        expanded.update(siblings)
    return expanded


async def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("line", type=int, help="题目在 jsonl 里的行号（1 基）")
    args = parser.parse_args()

    item = load_item(args.line)
    async with SessionLocal() as s:
        kb = await s.scalar(
            select(KnowledgeBase.id).order_by(KnowledgeBase.doc_count.desc()).limit(1)
        )
        gold = await expand_gold(s, item.get("gold_chunk_ids") or [])
        rows = (
            await s.execute(
                select(Chunk.id, Chunk.parent_id, Chunk.page_start, Chunk.char_count, Document.filename)
                .join(Document, Document.id == Chunk.doc_id)
                .where(Chunk.id.in_(item["gold_chunk_ids"]))
            )
        ).all()
        print("问题:", item["question"])
        print(f"类型 {item['type']} | gold {len(gold)} 块（含同父块兄弟）")
        for cid, pid, page, chars, fname in rows:
            print(f"  gold {cid[:8]} p{page} {chars}字 《{fname[:24]}》")
        if not rows:
            print("  !! gold_chunk_ids 在库里查不到——可能是上一轮清洗删掉了")

    emb, idx, rr = build_embedder(), MilvusIndex(), BGEReranker()
    q = item["question"]
    vec = (await emb.encode([q]))[0]
    raw_hits = await idx.search(kb_ids=[kb], dense=vec.dense, topk=30)
    hits = [
        Hit(
            chunk_id=h["chunk_id"],
            score=float(h.get("score", 0)),
            parent_id=h.get("parent_id"),
            content=h.get("content", ""),
        )
        for h in raw_hits
    ]
    scores = await rr.score(q, [h.content[:1200] for h in hits])
    for h, s in zip(hits, scores):
        h.score = float(s)
    deduped = dedupe_by_parent(hits)
    print(f"\n召回 {len(hits)} → 去重 {len(deduped)} 条（前 8）：")
    gold_ids = set(item["gold_chunk_ids"])
    for i, h in enumerate(deduped[:8], 1):
        star = "★" if h.chunk_id in gold_ids else " "
        flag = "保留" if h.score >= settings.retrieval.rerank_score_threshold else "过滤"
        print(f"  {star}{i}. {h.score:.4f} [{flag}] {h.content[:52].replace(chr(10), ' ')}")

    outcome = await retrieve(q, kb_ids=[kb], embedder=emb, index=idx, reranker=rr)
    kept = filter_by_threshold(deduped, settings.retrieval.rerank_score_threshold)
    print(f"\nretrieve 返回 {len(outcome.hits)} 条 | after_threshold {len(kept)} | "
          f"selected {len(outcome.hits)}")
    print("stats:", {k: v for k, v in outcome.stats.items() if k != "latency_ms"})
    hit_rank = next((i + 1 for i, h in enumerate(outcome.hits)
                     if h.chunk_id in gold_ids), 0)
    print(f"gold 在最终结果里的位次: {hit_rank or '未出现'}")


if __name__ == "__main__":
    asyncio.run(main())
