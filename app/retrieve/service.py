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

    # 5. 重排（用原始问题，只重排 RRF 排名靠前的一批）
    #
    # 这里曾经是「先回填父块再重排」，理由写得很顺：拿 300 字子块打分看到的是
    # 半句假设半句结论，拿 1200 字父块打分上下文完整，判别质量更高。
    # **这个假设是错的。** scripts/_rerank_input_ab.py 在 26 道概念题上实测：
    #   金标平均分  子块 0.7043 / 父块 0.6472
    #   逐题对比    子块更好 15 题，父块更好 11 题
    #   灾难案例    c006 #1→#15、c011 #1→#13、c016 #1→#13
    # 父块里只有 1/4 的内容和问题相关，交叉编码器看的是整段相关性，被稀释了；
    # 被稀释到阈值以下，正确答案就直接消失。父块全文只该用于生成，不该用于打分。
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

    # 8. 按预算裁出前 N 条（这一步还是子块正文，条数与预算都按子块算）
    selected = select_context(
        candidates,
        top_n=conf.rerank_topk,
        threshold=0,
        max_chars=conf.context_max_chars,
    )

    # 9. 回填父块：只给生成用，不参与打分。
    # 回填后单条从 ~300 字涨到 ~1200 字，所以要按字符预算再裁一次。
    selected = await _backfill_parents(selected, index=index)

    # 9.5 邻页扩展：父块按页切，跨页小节会被页边界切断，
    #     只拿到前半截时模型只能如实说「未覆盖」——而书里是有的。
    selected, neighbor_added = await _expand_neighbors(
        selected, index=index, query=question, reranker=reranker
    )
    stats["neighbor_added"] = neighbor_added

    selected = select_context(
        selected,
        top_n=conf.rerank_topk + neighbor_added,
        threshold=0,
        max_chars=conf.context_max_chars,
    )
    stats["selected"] = len(selected)

    # 10. 补齐文献元信息（书名/版次）
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


async def _expand_neighbors(
    hits: list[Hit],
    *,
    index: IndexLike,
    query: str = "",
    reranker: RerankerLike | None = None,
    span: int = 1,
    max_seeds: int = 3,
) -> tuple[list[Hit], int]:
    """把命中父块的相邻页一并取回来，紧跟在原命中之后。

    为什么需要这一步：父块按页切（一页 ≈ 一个父块），而教材里不少小节是跨页的。
    「埃里克森人格发展八阶段」在发展心理学里横跨 p48–p50，只命中 p48 时，
    上下文只剩前三个阶段——模型会老实说「本知识库未覆盖第四阶段」，
    而书里明明写着。这不是检索不到，是拿到的东西被页边界切断了。

    邻页按「种子之后」插入，不按分数重排：它们是同一段论述的后半部分，
    跟着种子走比按相关性排序更符合阅读顺序，也不会被预算裁剪挤到队尾。
    """
    fetch = getattr(index, "fetch_neighbor_parents", None)
    if fetch is None or not hits:
        return hits, 0

    known = {hit.chunk_id for hit in hits}
    known |= {hit.parent_id for hit in hits if hit.parent_id}
    added_count = 0
    merged: list[Hit] = []

    for seed in hits[:max_seeds]:
        merged.append(seed)

        doc_id = str(seed.meta.get("doc_id", ""))
        page = int(seed.meta.get("page_start") or 0)
        if not doc_id or not page:
            continue

        # 先看后一页再找前一页：跨页小节多数是往后延续的
        pages = [page + offset for offset in range(1, span + 1)]
        pages += [page - offset for offset in range(1, span + 1)]

        try:
            rows = await fetch(doc_id, pages)
        except Exception as exc:  # noqa: BLE001 - 扩展失败不该让整次检索失败
            logger.warning(
                "retrieve.neighbor_failed",
                extra={"extra_fields": {"doc_id": doc_id, "page": page, "error": repr(exc)}},
            )
            continue

        # 只补一页：跨页小节通常只断一次，补两页反而会把别的来源挤出预算。
        #
        # 同一页会有多个父块，必须**拼接**而不是挑一个。实测教育心理学 p49：
        # 第 8 阶段被切成「…这一阶段个体的发」+「展受前几阶段发展的影响极大…」，
        # 挑任何一块都会丢掉半个阶段——模型只能回答「材料中没有第 8 阶段」。
        # 拼接顺序按 chunk_id，它就是按文档顺序生成的。
        by_page: dict[int, list[dict]] = {}
        for row in rows:
            page_no = int(row.get("page_start") or -1)
            if page_no in pages:
                by_page.setdefault(page_no, []).append(row)

        ordered: list[dict] = []
        for page_no in pages:
            parts = sorted(by_page.get(page_no, []), key=lambda r: r.get("chunk_id", ""))
            if not parts:
                continue
            merged_row = dict(parts[0])
            merged_row["content"] = "".join(str(x.get("content") or "") for x in parts)
            ordered.append(merged_row)

        picked = ordered[:1]
        if picked and reranker is not None and query:
            # 邻页是「同一本书的下一页」，不等于「同一段论述的后续」。
            # 人格心理学 p87 讲繁衍感，翻页 p88 已经在讲学派评价了——
            # 这种邻页补进去只会占预算，还会把真正有用的来源挤到后面。
            # 判据用「不低于种子分的 85%」，而不是绝对阈值：
            # 补全段落天然比命中段落分低，绝对阈值会把它一起砍掉。
            try:
                scores = await reranker.score(query, [(picked[0].get("content") or "")[:2000]])
                keep = scores[0] >= max(0.50, seed.score * 0.85)
            except Exception:  # noqa: BLE001 - 打分失败就当它合格，宁可多补
                keep = True
            if not keep:
                logger.info(
                    "retrieve.neighbor_dropped",
                    extra={"extra_fields": {"page": picked[0].get("page_start")}},
                )
                picked = []

        for row in picked:
            chunk_id = str(row.get("chunk_id", ""))
            if not chunk_id or chunk_id in known:
                continue
            content = str(row.get("content") or "")
            if not content.strip():
                continue
            known.add(chunk_id)
            added_count += 1
            merged.append(
                Hit(
                    chunk_id=chunk_id,
                    # 分数压低一点点：邻页是补全用的，不该盖过真正命中的那一页
                    score=seed.score * 0.97,
                    parent_id=chunk_id,
                    source="neighbor",
                    content=content,
                    meta={
                        "doc_id": doc_id,
                        "page_start": int(row.get("page_start") or 0),
                        "page_end": int(row.get("page_end") or 0),
                        "domain": row.get("domain", ""),
                        "evidence_grade": row.get("evidence_grade", ""),
                    },
                )
            )

    merged.extend(hits[max_seeds:])
    return merged, added_count


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
