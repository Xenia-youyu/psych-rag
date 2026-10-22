"""Milvus 客户端：稠密 / 稀疏双路检索与元数据回填。

关键设计（v3 文档 3.3 节）：
- kb_id 作 partition_key。多知识库隔离靠分区而不是事后 expr 过滤，
  检索空间在进 HNSW 之前就被切小，比全库搜完再筛快一个量级。
- 稠密与稀疏共用同一个 collection，两条 ANN 请求并发发。
- 父块回填：子块命中后按 parent_id 取父块原文，避免 LLM 拿到断章取义的碎片。

pymilvus 是同步库，全部包 asyncio.to_thread，不阻塞事件循环。
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from typing import Any

from app.config import settings
from app.core.errors import AppError, ErrorCode
from app.core.logging import get_logger

logger = get_logger(__name__)

OUTPUT_FIELDS = [
    "chunk_id",
    "doc_id",
    "parent_id",
    "content",
    "page_start",
    "page_end",
    "domain",
    "evidence_grade",
    "is_parent",
]


class MilvusIndex:
    def __init__(self, *, uri: str | None = None, db_name: str | None = None,
                 collection: str | None = None) -> None:
        conf = settings.milvus
        self.uri = uri or conf.uri
        self.db_name = db_name or conf.database
        self.collection_name = collection or conf.collection
        self._alias = "rag_psy"
        self._connected = False
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------ 连接
    def _connect_sync(self):
        from pymilvus import connections

        if self._connected:
            return
        token = (
            f"{settings.milvus.user}:{settings.milvus.password}"
            if settings.milvus.user
            else None
        )
        try:
            connections.connect(
                alias=self._alias, uri=self.uri, token=token, db_name=self.db_name
            )
            self._connected = True
            logger.info(
                "milvus.connected",
                extra={"extra_fields": {"uri": self.uri, "db": self.db_name}},
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("milvus.connect_failed", extra={"extra_fields": {"error": repr(exc)}})
            raise AppError(ErrorCode.INTERNAL, f"Milvus 连接失败：{exc}") from exc

    async def _ensure(self) -> None:
        async with self._lock:
            await asyncio.to_thread(self._connect_sync)

    def _collection_sync(self):
        from pymilvus import Collection

        self._connect_sync()
        return Collection(self.collection_name, using=self._alias)

    # ------------------------------------------------------------ 检索
    def _kb_expr(self, kb_ids: Iterable[str]) -> str:
        ids = [f'"{kb}"' for kb in kb_ids if kb]
        if not ids:
            # 没有指定知识库就不检索，返回永不匹配的表达式
            return 'kb_id in ["__none__"]'
        return f"kb_id in [{', '.join(ids)}]"

    def _search_expr(self, kb_ids: Iterable[str], *, include_parents: bool = False) -> str:
        """检索过滤条件：知识库范围 + 排除父块。

        父块必须留在 collection 里（命中子块后要按 parent_id 回填全文），
        但不能作为召回候选：它比分块长 4 倍，向量是整页的平均语义，
        容易靠"大而全"的相似度挤进 topK，把真正的子块位置顶掉。
        去重逻辑只能事后补救，而 topK 名额一旦被占，真实命中就丢了。
        """
        expr = self._kb_expr(kb_ids)
        if not include_parents:
            expr = f"{expr} and is_parent == false"
        return expr

    def _search_sync(
        self,
        *,
        kb_ids: list[str],
        dense: list[float] | None,
        sparse: dict[str, float] | None,
        topk: int,
        include_parents: bool = False,
    ) -> dict[str, list[dict[str, Any]]]:
        from pymilvus import AnnSearchRequest, RRFRanker

        collection = self._collection_sync()
        collection.load()

        expr = self._search_expr(kb_ids, include_parents=include_parents)
        requests = []
        if dense:
            requests.append(
                AnnSearchRequest(
                    data=[dense],
                    anns_field="dense_vector",
                    param={"metric_type": "COSINE", "params": {"ef": 64}},
                    limit=topk,
                    expr=expr,
                )
            )
        if sparse:
            requests.append(
                AnnSearchRequest(
                    data=[sparse],
                    anns_field="sparse_vector",
                    param={"metric_type": "IP", "params": {}},
                    limit=topk,
                    expr=expr,
                )
            )

        if not requests:
            return {"dense": [], "sparse": []}

        results = collection.hybrid_search(
            reqs=requests,
            rerank=RRFRanker(k=settings.retrieval.rrf_k),
            limit=topk,
            output_fields=OUTPUT_FIELDS,
        )

        output: dict[str, list[dict[str, Any]]] = {"fused": []}
        for hits in results:
            for hit in hits:
                entity = hit.entity
                output["fused"].append(
                    {
                        "chunk_id": entity.get("chunk_id"),
                        "score": float(hit.score),
                        "doc_id": entity.get("doc_id", ""),
                        "parent_id": entity.get("parent_id", ""),
                        "content": entity.get("content", ""),
                        "page_start": entity.get("page_start", 0),
                        "page_end": entity.get("page_end", 0),
                        "domain": entity.get("domain", ""),
                        "evidence_grade": entity.get("evidence_grade", ""),
                        "is_parent": bool(entity.get("is_parent", False)),
                    }
                )
        return output

    async def search(
        self,
        *,
        kb_ids: list[str],
        dense: list[float] | None = None,
        sparse: dict[str, float] | None = None,
        topk: int | None = None,
        include_parents: bool = False,
    ) -> list[dict[str, Any]]:
        limit = topk or settings.retrieval.dense_topk
        if not kb_ids:
            return []
        await self._ensure()
        result = await asyncio.to_thread(
            self._search_sync,
            kb_ids=kb_ids,
            dense=dense,
            sparse=sparse,
            topk=limit,
            include_parents=include_parents,
        )
        return result.get("fused", [])

    # ------------------------------------------------------------ 回填
    def _fetch_sync(self, chunk_ids: list[str]) -> dict[str, dict[str, Any]]:
        if not chunk_ids:
            return {}
        collection = self._collection_sync()
        quoted = ", ".join(f'"{cid}"' for cid in chunk_ids)
        rows = collection.query(expr=f"chunk_id in [{quoted}]", output_fields=OUTPUT_FIELDS)
        return {row["chunk_id"]: row for row in rows}

    async def fetch_chunks(self, chunk_ids: list[str]) -> dict[str, dict[str, Any]]:
        await self._ensure()
        return await asyncio.to_thread(self._fetch_sync, chunk_ids)

    def _get_vector_sync(self, chunk_id: str) -> list[float] | None:
        collection = self._collection_sync()
        collection.load()
        rows = collection.query(
            expr=f'chunk_id == "{chunk_id}"', output_fields=["dense_vector"]
        )
        if not rows:
            return None
        vector = rows[0].get("dense_vector")
        return list(vector) if vector is not None else None

    async def get_vector(self, chunk_id: str) -> list[float] | None:
        """取出库里已有的稠密向量。

        关联推荐要算「这段内容和谁像」，重新编码一次多花 0.4 秒，
        而这条向量本来就在库里躺着。
        """
        await self._ensure()
        return await asyncio.to_thread(self._get_vector_sync, chunk_id)

    # ------------------------------------------------------------ 写入
    def _upsert_sync(self, records: list[dict[str, Any]]) -> int:
        if not records:
            return 0
        collection = self._collection_sync()
        # 这里刻意不 flush。实测一次 flush 约 30 秒，而入库要分几十批——
        # 每批都 flush，一本 500 页的书光刷盘就要一小时。
        # 正确做法是整篇写完刷一次，见 pipeline 收尾处的 index.flush()。
        collection.upsert(records)
        return len(records)

    async def upsert(self, records: list[dict[str, Any]]) -> int:
        await self._ensure()
        count = await asyncio.to_thread(self._upsert_sync, records)
        logger.info("milvus.upsert", extra={"extra_fields": {"count": count}})
        return count

    def _flush_sync(self) -> None:
        self._collection_sync().flush()

    async def flush(self) -> None:
        """把增量段落盘。

        搜索不是非 flush 才命中，但 Bounded 一致性下刚写入的数据可能暂时查不到。
        整篇文档写完后刷一次，兼顾可见性与速度。
        """
        await self._ensure()
        await asyncio.to_thread(self._flush_sync)

    def _delete_sync(self, expr: str) -> None:
        collection = self._collection_sync()
        collection.delete(expr)
        # 必须 flush：Milvus 的删除是逻辑删除，不 flush 的话已加载的 segment
        # 仍会把旧数据返回，用户会看到"文档删了却还能搜到"（实测复现过）。
        # 删除不是高频操作，这点开销值得。
        collection.flush()

    async def delete_by_doc(self, doc_id: str) -> None:
        await self._ensure()
        await asyncio.to_thread(self._delete_sync, f'doc_id == "{doc_id}"')
        logger.info("milvus.delete_doc", extra={"extra_fields": {"doc_id": doc_id}})

    async def delete_by_kb(self, kb_id: str) -> None:
        await self._ensure()
        await asyncio.to_thread(self._delete_sync, f'kb_id == "{kb_id}"')
        logger.info("milvus.delete_kb", extra={"extra_fields": {"kb_id": kb_id}})

    async def count(self) -> int:
        await self._ensure()
        collection = await asyncio.to_thread(self._collection_sync)
        return await asyncio.to_thread(lambda: collection.num_entities)
