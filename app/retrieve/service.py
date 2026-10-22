"""检索编排：把「问题」变成「可喂给模型的上下文」。

链路（v3 文档 5.x）：
    意图 → 改写（可能多 query）→ 逐 query 双路召回 → RRF 合并
    → 父块去重 → 重排（用原始问题打分）→ 阈值过滤 → 父块回填 → 上下文预算裁剪

三个刻意的设计决定：
1. **重排用原始问题，不用改写后的问题**。改写是为了召回，重排是为了判断"这段内容能不能回答用户真正问的东西"。
2. **回填放在重排之前**。子块 300 字常常是半句假设半句结论，拿它去交叉编码，模型只能靠猜；
   父块 1200 字上下文完整，判别质量明显更高。代价是重排文本变长约 4 倍，
   用 rerank_max_candidates（只重排 RRF 前 N 条）+ rerank_max_length（截断到 512 token）控回来。
3. **预算裁剪放在最后**。字符预算要按回填后的真实长度算，否则算出来的 6000 字实际会变成 12000 字。

所有外部依赖都是注入的，因此整条链路可以用替身完整测试，不碰服务器。
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from app.config import settings
from app.core.logging import get_logger
from app.generate.answer import Citation, build_citations
from app.generate.intent import Intent, classify, strategy_for
from app.generate.rewrite import RewriteResult, rewrite
from app.retrieve.fusion import (
    Hit,
    build_context_block,
    dedupe_by_parent,
    filter_by_threshold,
    rrf_fuse,
    select_context,
)

logger = get_logger(__name__)


class EmbedderLike(Protocol):
    async def encode(self, texts: list[str]) -> list[Any]: ...


class IndexLike(Protocol):
    async def search(self, *, kb_ids: list[str], dense=None, sparse=None,
                     topk: int | None = None) -> list[dict[str, Any]]: ...

    async def fetch_chunks(self, chunk_ids: list[str]) -> dict[str, dict[str, Any]]: ...


class RerankerLike(Protocol):
    async def score(self, query: str, documents: list[str]) -> list[float]: ...


# doc_id -> {"title": ..., "edition": ...}
DocMetaProvider = Callable[[str], Awaitable[dict[str, str]]]


@dataclass
class RetrievalOutcome:
    hits: list[Hit] = field(default_factory=list)
    context: str = ""
    citations: list[Citation] = field(default_factory=list)
    intent: Intent = Intent.DEFINE
    strategy: str = "direct"
    queries: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return not self.hits


def _row_to_hit(row: dict[str, Any], *, source: str, title: str = "",
                edition: str = "") -> Hit:
    return Hit(
        chunk_id=str(row.get("chunk_id", "")),
        score=float(row.get("score", 0.0)),
        parent_id=row.get("parent_id") or None,
        source=source,
        content=row.get("content", ""),
        meta={
            "doc_id": row.get("doc_id", ""),
            "source_title": title,
            "edition": edition,
            "page_start": row.get("page_start", 0),
            "page_end": row.get("page_end", 0),
            "domain": row.get("domain", ""),
            "evidence_grade": row.get("evidence_grade", ""),
        },
    )


async def _recall_one(
    query_embedding: Any,
    *,
    kb_ids: list[str],
    index: IndexLike,
    dense_topk: int,
    sparse_topk: int,
    hybrid: bool,
) -> list[Hit]:
    """单条 query 的召回。稠密 + 稀疏一起发；关掉 hybrid 时只走稠密。"""
    dense = getattr(query_embedding, "dense", None)
    sparse = getattr(query_embedding, "sparse", None) if hybrid else None

    rows = await index.search(
        kb_ids=kb_ids,
        dense=dense,
        sparse=sparse,
        topk=max(dense_topk, sparse_topk),
    )
    return [_row_to_hit(row, source="dense" if dense else "sparse") for row in rows]


async def retrieve(
    question: str,
    *,
    kb_ids: list[str],
    embedder: EmbedderLike,
    index: IndexLike,
    reranker: RerankerLike | None = None,
    llm=None,
    doc_meta: DocMetaProvider | None = None,
    intent: Intent | None = None,
) -> RetrievalOutcome:
    """执行完整检索链路。任何环节失败都降级而不是抛错，除非知识库为空。"""
    conf = settings.retrieval
    started = time.perf_counter()
    stats: dict[str, Any] = {"kb_count": len(kb_ids)}

    if not kb_ids:
        logger.info("retrieve.no_kb")
        return RetrievalOutcome(stats={**stats, "reason": "no_kb"})

    # 1. 意图 + 改写
    resolved_intent = intent or await classify(question, llm=llm)
    strategy = strategy_for(resolved_intent)

    if llm is not None:
        rewrite_result: RewriteResult = await rewrite(question, resolved_intent, llm)
    else:
        rewrite_result = RewriteResult("direct", [question])

    queries = rewrite_result.queries or [question]
    stats.update({"intent": resolved_intent.value, "strategy": strategy,
                  "query_count": len(queries)})

    # 2. 编码（一次批量编码所有改写后的问题）
    try:
        embeddings = await embedder.encode(queries)
    except Exception as exc:  # noqa: BLE001
        logger.error("retrieve.embed_failed", extra={"extra_fields": {"error": repr(exc)}})
        return RetrievalOutcome(
            intent=resolved_intent, strategy=strategy, queries=queries,
            stats={**stats, "error": "embed_failed"},
        )

    # 3. 逐 query 召回，按 query 分组做 RRF
    ranked_lists: dict[str, list[Hit]] = {}
    for index_no, embedding in enumerate(embeddings):
        try:
            hits = await _recall_one(
                embedding,
                kb_ids=kb_ids,
                index=index,
                dense_topk=conf.dense_topk,
                sparse_topk=conf.sparse_topk,
                hybrid=conf.hybrid,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "retrieve.search_failed",
                extra={"extra_fields": {"query_no": index_no, "error": repr(exc)}},
            )
            continue
        ranked_lists[f"q{index_no}"] = hits

    recalled = sum(len(hits) for hits in ranked_lists.values())
    fused = rrf_fuse(ranked_lists, k=conf.rrf_k)
    stats.update({"recalled": recalled, "fused": len(fused)})

    # 4. 父块去重（同一个父块下命中多个子块时只留 RRF 分最高的那条）
    deduped = dedupe_by_parent(fused)
    stats["after_dedupe"] = len(deduped)

    # 5. 父块回填。
    # 顺序很关键：**先回填再重排**。交叉编码要判断的是"这段内容能不能回答这个问题"，
    # 拿 300 字的子块去打分，看到的往往是半句假设、半句结论，模型只能靠猜；
    # 拿 1200 字的父块打分，上下文完整，判别质量明显更高。
    # 代价是重排文本变长（约 4 倍计算量），靠 rerank_max_candidates 和
    # rerank_max_length 两个上限控回来。
    deduped = await _backfill_parents(deduped, index=index)

    # 6. 重排（用原始问题，且只重排 RRF 排名靠前的一批）
    rerank_ok = False
    rerank_input = deduped
    if reranker is not None:
        if conf.rerank_max_candidates > 0:
            rerank_input = deduped[: conf.rerank_max_candidates]
            stats["rerank_truncated"] = len(deduped) - len(rerank_input)

        # 只有一条候选也要重排。参考项目在 <2 时直接短路，但那样就跳过了阈值过滤，
        # 一条垃圾结果会被原样送去生成——"检索不到就说未覆盖"这条设计就破了。
        # 单对交叉编码约 0.2 秒，买的是「能拒绝坏结果」，值。
        if rerank_input:
            documents = [hit.content for hit in rerank_input]
            try:
                scores = await reranker.score(question, documents)
                for hit, score in zip(rerank_input, scores, strict=False):
                    hit.score = float(score)
                rerank_input.sort(key=lambda h: h.score, reverse=True)
                rerank_ok = True
            except Exception as exc:  # noqa: BLE001
                # 重排失败保留 RRF 顺序，不阻断
                logger.warning(
                    "retrieve.rerank_failed", extra={"extra_fields": {"error": repr(exc)}}
                )

    # 7. 阈值过滤。
    # 只有重排真的成功时才按阈值筛：RRF 分的量级是 1/(k+rank) ≈ 0.016，
    # 拿 0.15 的重排阈值去筛会全灭。重排挂了却继续筛，等于把结果全丢掉。
    if rerank_ok:
        candidates = filter_by_threshold(rerank_input, conf.rerank_score_threshold)
    else:
        candidates = rerank_input
    stats["rerank_ok"] = rerank_ok
    stats["after_threshold"] = len(candidates)

    # 8. 按预算裁剪
    selected = select_context(
        candidates,
        top_n=conf.rerank_topk,
        threshold=0,
        max_chars=conf.context_max_chars,
    )
    stats["selected"] = len(selected)

    # 9. 补齐文献元信息（书名/版次）
    if doc_meta is not None:
        await _attach_doc_meta(selected, doc_meta)

    context = build_context_block(selected, max_chars=conf.context_max_chars)
    citations = build_citations(selected)

    stats["latency_ms"] = int((time.perf_counter() - started) * 1000)
    logger.info("retrieve.done", extra={"extra_fields": stats})

    return RetrievalOutcome(
        hits=selected,
        context=context,
        citations=citations,
        intent=resolved_intent,
        strategy=strategy,
        queries=queries,
        stats=stats,
    )


async def _backfill_parents(hits: list[Hit], *, index: IndexLike) -> list[Hit]:
    """子块命中后取父块原文。父块缺失时保留子块，不丢结果。"""
    parent_ids = [hit.parent_id for hit in hits if hit.parent_id]
    if not parent_ids:
        return hits

    try:
        parents = await index.fetch_chunks(parent_ids)
    except Exception as exc:  # noqa: BLE001
        logger.warning("retrieve.backfill_failed", extra={"extra_fields": {"error": repr(exc)}})
        return hits

    for hit in hits:
        if not hit.parent_id:
            continue
        parent = parents.get(hit.parent_id)
        if parent and parent.get("content"):
            # 子块页码更精确（指向命中位置），父块内容更完整，两者各取所需
            hit.meta["hit_page"] = hit.meta.get("page_start", 0)
            hit.content = parent["content"]
            if parent.get("page_start"):
                hit.meta["page_start"] = parent["page_start"]
            if parent.get("page_end"):
                hit.meta["page_end"] = parent["page_end"]
    return hits


async def _attach_doc_meta(hits: list[Hit], provider: DocMetaProvider) -> None:
    cache: dict[str, dict[str, str]] = {}
    for hit in hits:
        doc_id = str(hit.meta.get("doc_id", ""))
        if not doc_id:
            continue
        if doc_id not in cache:
            try:
                cache[doc_id] = await provider(doc_id)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "retrieve.doc_meta_failed",
                    extra={"extra_fields": {"doc_id": doc_id, "error": repr(exc)}},
                )
                cache[doc_id] = {}
        meta = cache[doc_id]
        if meta.get("title"):
            hit.meta["source_title"] = meta["title"]
        if meta.get("edition"):
            hit.meta["edition"] = meta["edition"]
