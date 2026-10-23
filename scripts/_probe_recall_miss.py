"""查两条「压根没召回」的题：gold 在不在库里、向量搜不搜得到。

c004 / c020 在评估里是「前 10 未命中」，之前没查是召回问题还是标注问题。
只用 embedding API + Milvus 检索，不加载本地重排模型（不跟 OCR 抢 CPU）。
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from sqlalchemy import select

from app.clients.embed_client import build_embedder
from app.clients.milvus_client import MilvusIndex
from app.core.logging import setup_logging
from app.db.base import SessionLocal
from app.db.models import Chunk, KnowledgeBase

GOLDEN = Path(__file__).resolve().parent.parent / "data" / "golden" / "questions.jsonl"
TARGETS = {"c004", "c020"}
TOPK = 50


async def expand_gold(session, ids: list[str]) -> set[str]:
    rows = (await session.execute(select(Chunk.id, Chunk.parent_id).where(Chunk.id.in_(ids)))).all()
    out = {cid for cid, _ in rows}
    parents = {pid for _, pid in rows if pid}
    if parents:
        out.update((await session.scalars(select(Chunk.id).where(Chunk.parent_id.in_(parents)))).all())
    return out


async def main() -> None:
    setup_logging()
    items = [json.loads(l) for l in GOLDEN.read_text(encoding="utf-8").splitlines() if l.strip()]
    items = [i for i in items if i["id"] in TARGETS]

    async with SessionLocal() as s:
        kb_id = await s.scalar(
            select(KnowledgeBase.id).order_by(KnowledgeBase.doc_count.desc()).limit(1)
        )

    emb, idx = build_embedder(), MilvusIndex()

    for item in items:
        gold = await expand_gold_cached(item)
        vector = (await emb.encode([item["question"]]))[0]
        hits = await idx.search(kb_ids=[kb_id], dense=vector.dense, topk=TOPK)

        print("=" * 72)
        print(f"{item['id']}  {item['question']}")
        print(f"扩展后 gold {len(gold)} 块：")
        for gid in list(gold)[:4]:
            content = await content_of(gid)
            print(f"   [{gid[:8]}] {content[:90].replace(chr(10), ' / ')}")

        rank = next((n for n, h in enumerate(hits, 1) if h["chunk_id"] in gold), None)
        print(f"\n向量检索 top{TOPK}：{'命中，位次 ' + str(rank) if rank else '未命中'}")
        print("前 5 条召回：")
        for n, h in enumerate(hits[:5], 1):
            print(f"   {n}. {h.get('score', 0):.4f} {h['content'][:64].replace(chr(10), ' ')}")


_content_cache: dict[str, str] = {}
_gold_cache: dict[str, set[str]] = {}


async def content_of(chunk_id: str) -> str:
    if chunk_id not in _content_cache:
        async with SessionLocal() as s:
            _content_cache[chunk_id] = await s.scalar(
                select(Chunk.content).where(Chunk.id == chunk_id)
            ) or ""
    return _content_cache[chunk_id]


async def expand_gold_cached(item) -> set[str]:
    if item["id"] not in _gold_cache:
        async with SessionLocal() as s:
            _gold_cache[item["id"]] = await expand_gold(s, item.get("gold_chunk_ids") or [])
    return _gold_cache[item["id"]]


if __name__ == "__main__":
    asyncio.run(main())
