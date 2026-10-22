"""逐段打点：看 gold 块在检索链路的哪一步掉出去。

用法：python -m scripts._debug_retrieve [行号，默认 1]
读 data/golden/questions.jsonl 的第 N 行（0-based）。
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from sqlalchemy import func, select

from app.clients.embed_client import build_embedder
from app.clients.milvus_client import MilvusIndex
from app.clients.rerank_client import BGEReranker
from app.config import settings
from app.db.base import SessionLocal
from app.db.models import Chunk, KnowledgeBase
from app.retrieve.fusion import dedupe_by_parent, rrf_fuse
from app.retrieve.service import _backfill_parents, _recall_one, retrieve

GOLDEN = Path(__file__).resolve().parent.parent / "data" / "golden" / "questions.jsonl"


async def expand_gold(session, ids: list[str]) -> set[str]:
    rows = (await session.execute(select(Chunk.id, Chunk.parent_id).where(Chunk.id.in_(ids)))).all()
    out = {cid for cid, _ in rows}
    parents = {pid for _, pid in rows if pid}
    if parents:
        out.update((await session.scalars(select(Chunk.id).where(Chunk.parent_id.in_(parents)))).all())
    return out


def mark(ids: list[str], gold: set[str]) -> str:
    return f"gold 命中 {sum(1 for i in ids if i in gold)} / 共 {len(ids)} 条"


async def main() -> None:
    line_no = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    item = json.loads(GOLDEN.read_text(encoding="utf-8").splitlines()[line_no])
    print(f"问题：{item['question']}")
    print(f"原始 gold：{len(item['gold_chunk_ids'])} 块")

    conf = settings.retrieval
    print(
        f"配置：dense_topk={conf.dense_topk} sparse_topk={conf.sparse_topk} "
        f"hybrid={conf.hybrid} rrf_k={conf.rrf_k} "
        f"rerank_max_candidates={conf.rerank_max_candidates} "
        f"rerank_topk={conf.rerank_topk} 阈值={conf.rerank_score_threshold}"
    )

    async with SessionLocal() as s:
        kb_id = await s.scalar(
            select(KnowledgeBase.id).order_by(KnowledgeBase.doc_count.desc()).limit(1)
        )
        gold = await expand_gold(s, item["gold_chunk_ids"])
        rows = (await s.execute(
            select(Chunk.id, Chunk.parent_id, Chunk.page_start, Chunk.char_count)
            .where(Chunk.id.in_(item["gold_chunk_ids"]))
        )).all()
        # 有子块指向它 => 它自己是父块，父块不参与召回（Milvus 里被 is_parent 过滤掉）
        print("\ngold 块属性：")
        for cid, pid, page, chars in rows:
            child_count = await s.scalar(
                select(func.count(Chunk.id)).where(Chunk.parent_id == cid)
            )
            kind = f"父块（{child_count} 个子块）" if child_count else "子块"
            print(f"  {cid[:8]} parent={pid[:8] or '(空)'} {kind} p{page} {chars}字")
        print(f"扩展后 gold 共 {len(gold)} 块（含同父块兄弟）")

    emb, idx, rr = build_embedder(), MilvusIndex(), BGEReranker()

    # ---- 整条链路（与线上一致）----
    out = await retrieve(item["question"], kb_ids=[kb_id], embedder=emb, index=idx, reranker=rr)
    print(f"\n[retrieve] queries={out.queries}")
    print(f"[retrieve] stats={json.dumps({k: v for k, v in out.stats.items() if k != 'latency_ms'}, ensure_ascii=False)}")
    print(f"[retrieve] 返回 {len(out.hits)} 条，{mark([h.chunk_id for h in out.hits], gold)}")

    # ---- 手工重放，逐段看 gold ----
    queries = out.queries or [item["question"]]
    embeddings = await emb.encode(queries)
    ranked: dict[str, list] = {}
    for i, embedding in enumerate(embeddings):
        ranked[f"q{i}"] = await _recall_one(
            embedding, kb_ids=[kb_id], index=idx,
            dense_topk=conf.dense_topk, sparse_topk=conf.sparse_topk, hybrid=conf.hybrid,
        )
        ids = [h.chunk_id for h in ranked[f"q{i}"]]
        pos = next((n + 1 for n, x in enumerate(ids) if x in gold), None)
        print(f"\n[召回 q{i}] {len(ids)} 条，{mark(ids, gold)}，gold 最高位次={pos}")

    fused = rrf_fuse(ranked, k=conf.rrf_k)
    ids = [h.chunk_id for h in fused]
    pos = next((n + 1 for n, x in enumerate(ids) if x in gold), None)
    print(f"[融合] {len(ids)} 条，{mark(ids, gold)}，最高位次={pos}")

    deduped = dedupe_by_parent(fused)
    ids = [h.chunk_id for h in deduped]
    pos = next((n + 1 for n, x in enumerate(ids) if x in gold), None)
    print(f"[去重] {len(ids)} 条，{mark(ids, gold)}，最高位次={pos}")

    cut = conf.rerank_max_candidates
    in_window = ids[:cut] if cut > 0 else ids
    print(f"[重排窗口前 {cut}] {len(in_window)} 条，{mark(in_window, gold)}"
          f"  ← gold 是否被截掉：{'是' if any(i in gold for i in ids) and not any(i in gold for i in in_window) else '否'}")

    backfilled = await _backfill_parents(deduped, index=idx)
    scores = await rr.score(item["question"], [h.content for h in backfilled[:cut]])
    top = sorted(zip(backfilled[:cut], scores), key=lambda x: -x[1])
    print(f"\n[重排后前 8]")
    for h, sc in top[:8]:
        flag = "★gold" if h.chunk_id in gold else "     "
        keep = "过" if sc >= conf.rerank_score_threshold else "砍"
        print(f"  {sc:.4f} [{keep}] {flag} p{h.meta.get('page_start', '?')} {h.content[:40]}")


if __name__ == "__main__":
    asyncio.run(main())
