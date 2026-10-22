"""重排阈值标定：用「概念题的金标分数」对「拒答题的最高分」找分界。

为什么要标定：阈值 0.58 是拍下来的一个数，而 bge-reranker 在这个语料上的
分数整体落在 0.50–0.60，硬切会把第二名之后全砍掉，只剩 1 条上下文。

方法：
- 概念题（concept）：金标块的重排分数 = 应该被保留的那一侧
- 拒答题（refuse）：库内没有答案，最高分 = 应该被拒绝的那一侧
- 在候选阈值上统计「金标保留率」与「拒答误收率」，挑一个平衡点

只读，不改配置、不改数据。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
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


def pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[idx]


async def expand_gold(session, ids: list[str]) -> set[str]:
    rows = (await session.execute(select(Chunk.id, Chunk.parent_id).where(Chunk.id.in_(ids)))).all()
    out = {cid for cid, _ in rows}
    parents = {pid for _, pid in rows if pid}
    if parents:
        out.update((await session.scalars(select(Chunk.id).where(Chunk.parent_id.in_(parents)))).all())
    return out


all_scores: list[list[float]] = []


async def score_question(item, kb_id, emb, idx, rr, conf):
    """返回 (最高分, 金标最高分或 None, 金标名次或 None, 候选数)。"""
    embedding = (await emb.encode([item["question"]]))[0]
    hits = await _recall_one(
        embedding, kb_ids=[kb_id], index=idx,
        dense_topk=conf.dense_topk, sparse_topk=conf.sparse_topk, hybrid=conf.hybrid,
    )
    if not hits:
        return None, None, None, 0

    fused = rrf_fuse({"q0": hits}, k=conf.rrf_k)
    deduped = await _backfill_parents(dedupe_by_parent(fused), index=idx)
    scores = await rr.score(item["question"], [h.content for h in deduped])
    pairs = sorted(zip(deduped, scores), key=lambda x: -x[1])

    all_scores.append([float(x) for x in scores])
    gold = await expand_gold_cached(item, kb_id)
    gold_score, gold_rank = None, None
    for rank, (hit, sc) in enumerate(pairs, 1):
        if hit.chunk_id in gold:
            gold_score, gold_rank = float(sc), rank
            break
    return float(pairs[0][1]), gold_score, gold_rank, len(pairs)


_gold_cache: dict[str, set[str]] = {}


async def expand_gold_cached(item, kb_id):
    key = item["id"]
    if key not in _gold_cache:
        async with SessionLocal() as s:
            _gold_cache[key] = await expand_gold(s, item.get("gold_chunk_ids") or [])
    return _gold_cache[key]


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="0 = 全部")
    args = parser.parse_args()

    setup_logging()
    items = [json.loads(line) for line in GOLDEN.read_text(encoding="utf-8").splitlines() if line.strip()]
    # 与 eval_run 保持一致：被人工判为 reject 的题不参与统计
    items = [i for i in items if i.get("status") in ("approved", "draft")]
    if args.limit:
        items = items[: args.limit]

    concepts = [i for i in items if i["type"] == "concept"]
    refuses = [i for i in items if i["type"] == "refuse"]
    print(f"概念题 {len(concepts)} 条 | 拒答题 {len(refuses)} 条 | 现阈值 {settings.retrieval.rerank_score_threshold}\n")

    async with SessionLocal() as s:
        kb_id = await s.scalar(
            select(KnowledgeBase.id).order_by(KnowledgeBase.doc_count.desc()).limit(1)
        )

    emb, idx, rr = build_embedder(), MilvusIndex(), BGEReranker()
    conf = settings.retrieval

    gold_scores: list[float] = []
    concept_top1: list[float] = []
    miss = 0
    for n, item in enumerate(concepts, 1):
        top1, gs, rank, cand = await score_question(item, kb_id, emb, idx, rr, conf)
        if top1 is None:
            continue
        concept_top1.append(top1)
        if gs is None:
            miss += 1
            print(f"  [{n:2}] top1={top1:.4f} gold 未进候选  {item['question'][:34]}")
        else:
            gold_scores.append(gs)
            print(f"  [{n:2}] top1={top1:.4f} gold={gs:.4f} 名次={rank:2} 候选={cand}  {item['question'][:30]}")

    refuse_top1: list[float] = []
    print()
    for n, item in enumerate(refuses, 1):
        top1, _gs, _rank, cand = await score_question(item, kb_id, emb, idx, rr, conf)
        if top1 is None:
            continue
        refuse_top1.append(top1)
        print(f"  [R{n:2}] top1={top1:.4f} 候选={cand}  {item['question'][:34]}")

    print("\n" + "=" * 74)
    print("金标分数分布（概念题）")
    if gold_scores:
        print(f"  n={len(gold_scores)} min={min(gold_scores):.4f} p10={pct(gold_scores,0.1):.4f} "
              f"中位={statistics.median(gold_scores):.4f} p90={pct(gold_scores,0.9):.4f} max={max(gold_scores):.4f}")
    print(f"  金标不在候选里：{miss} 条")
    print("拒答题最高分分布")
    if refuse_top1:
        print(f"  n={len(refuse_top1)} min={min(refuse_top1):.4f} p10={pct(refuse_top1,0.1):.4f} "
              f"中位={statistics.median(refuse_top1):.4f} p90={pct(refuse_top1,0.9):.4f} max={max(refuse_top1):.4f}")

    if concept_top1:
        print(f"概念题 top1 分布：min={min(concept_top1):.4f} p10={pct(concept_top1,0.1):.4f} "
              f"中位={statistics.median(concept_top1):.4f} max={max(concept_top1):.4f}")

    print("\n阈值扫描（保留率 = 金标分 >= 阈值；误收率 = 拒答最高分 >= 阈值）")
    print(f"{'阈值':>6} {'金标保留':>9} {'拒答误收':>9} {'平均喂给模型':>10}")
    best = None
    for t in [round(x / 100, 2) for x in range(45, 71)]:
        keep = sum(1 for v in gold_scores if v >= t) / len(gold_scores) if gold_scores else 0
        false = sum(1 for v in refuse_top1 if v >= t) / len(refuse_top1) if refuse_top1 else 0
        flag = ""
        score = keep - false
        if best is None or score > best[0]:
            best = (score, t, keep, false)
        avg_kept = statistics.mean(
            sum(1 for v in row if v >= t) for row in all_scores
        ) if all_scores else 0
        print(f"{t:>6.2f} {keep:>8.0%} {false:>8.0%} {avg_kept:>8.1f}")
    if best:
        print(f"\n建议阈值 {best[1]:.2f}：金标保留 {best[2]:.0%}，拒答误收 {best[3]:.0%}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
