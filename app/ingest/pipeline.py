"""入库流水线：文件 → 解析 → 切块 → 向量化 → Milvus / MySQL。

六个阶段与 Document.status 一一对应，失败时停在哪个阶段就记在哪里：
    uploaded → parsing → chunking → embedding → done / failed

四个刻意的设计决定：

1. **MySQL 只存子块，Milvus 存子块 + 父块。**
   用户在「文本块」页面看到的是可编辑的检索单元（子块），父块只是给 LLM
   补上下文的容器，塞进 MySQL 只会让列表页噪声翻倍。
   但父块必须在 Milvus 里——命中子块后要靠 chunk_id 回查父块全文。

2. **失败不抛给调用方，写进 Document.error_msg。**
   这是后台任务，抛出去只有日志能看见。前端靠轮询列表看状态，
   用户需要知道"第几步失败、失败原因是什么"。

3. **重跑幂等。** 先按 doc_id 删向量、按 doc_id 删 chunk 行，再写。
   否则同一本书点两次上传，检索结果会重复出现两份。

4. **每阶段单独提交事务。** 一本 500 页的书要跑十几分钟，
   如果最后统一提交，中途崩了前面全白干，状态也停在 uploaded 让人以为没开始。
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import delete, func, select

from app.config import settings
from app.core.logging import get_logger
from app.db.models import Chunk, Document, KnowledgeBase
from app.ingest.base import ParsedDoc
from app.ingest.chunker import split_parent_child
from app.ingest.cleaner import strip_watermark
from app.ingest.frontmatter import strip_front_matter

logger = get_logger(__name__)

# 清洗页脚广告后，剩余内容不足这些字数就丢弃该块
MIN_CHILD_CHARS = 80
MIN_PARENT_CHARS = 120

EMBED_BATCH = 32


@dataclass
class IngestReport:
    doc_id: str
    status: str
    pages: int = 0
    dropped_pages: int = 0
    parents: int = 0
    children: int = 0
    vectors: int = 0
    elapsed_ms: int = 0
    error: str = ""
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == "done"


@dataclass
class ChunkPlan:
    """切块中间产物，纯数据，方便单测。"""

    parents: list[dict[str, Any]] = field(default_factory=list)
    children: list[dict[str, Any]] = field(default_factory=list)


def clean_watermarks(plan: ChunkPlan) -> tuple[ChunkPlan, int, int]:
    """清洗父子块里的页脚广告，返回 (新计划, 修改数, 丢弃数)。

    实测《实验心理学》2290 块里有 763 块带盗版水印（占 33%），
    每页页脚都是「资料下载：bbs.xxx / 答疑QQ：xxx」。
    这些块会命中「考研」类查询，引用展开时还会把广告一起展示给用户。

    只删水印行，正文保留；清洗后不足 80 字的块才整体丢弃
    （那种块基本只剩水印了）。父块同理，但更少删——
    父块不参与召回，只用于命中后回填全文。
    """
    touched = 0
    dropped = 0

    parents: list[dict[str, Any]] = []
    for item in plan.parents:
        text, removed = strip_watermark(item.get("content", ""))
        if removed:
            item["content"] = text
            touched += 1
        if len(text) >= MIN_PARENT_CHARS:
            parents.append(item)

    children: list[dict[str, Any]] = []
    for item in plan.children:
        text, removed = strip_watermark(item.get("content", ""))
        if removed:
            item["content"] = text
            touched += 1
        if len(text) >= MIN_CHILD_CHARS:
            children.append(item)
        else:
            dropped += 1

    return ChunkPlan(parents=parents, children=children), touched, dropped


def build_chunks(doc_id: str, parsed: ParsedDoc) -> ChunkPlan:
    """把逐页正文切成父子块。

    按页切而不是整本连续切：页码是引用溯源的关键，
    整本切完之后再反推页码要靠字符偏移，容易差一页，
    而引用页码错了比没有页码更糟——读者会去翻一页找不到。
    """
    conf = settings.chunk
    plan = ChunkPlan()

    for page in parsed.pages:
        if not page.text.strip():
            continue
        groups = split_parent_child(
            page.text,
            parent_size=conf.parent_size,
            parent_min_size=conf.parent_min_size,
            child_size=conf.child_size,
            child_min_size=conf.child_min_size,
            child_overlap=int(conf.child_size * conf.overlap_ratio),
            separators=conf.separators or None,
            id_prefix="p",
        )
        for p_index, group in enumerate(groups):
            parent_id = f"{doc_id}_{page.page_no:04d}_p{p_index:02d}"
            plan.parents.append(
                {
                    "chunk_id": parent_id,
                    "content": group.content,
                    "page_start": page.page_no,
                    "page_end": page.page_no,
                }
            )
            for c_index, child in enumerate(group.children):
                plan.children.append(
                    {
                        "chunk_id": f"{parent_id}_c{c_index:02d}",
                        "parent_id": parent_id,
                        "content": child,
                        "page_start": page.page_no,
                        "page_end": page.page_no,
                    }
                )
    return plan


class IngestPipeline:
    """把一份已上传的文档变成可检索的向量。"""

    def __init__(
        self,
        *,
        session_factory=None,
        parser=None,
        embedder=None,
        index=None,
    ) -> None:
        # 全部可注入：测试时给替身，生产时用默认实现
        self._session_factory = session_factory
        self._parser = parser
        self._embedder = embedder
        self._index = index

    # ------------------------------------------------------------ 依赖
    def _get_session_factory(self):
        if self._session_factory is None:
            from app.db.base import SessionLocal

            self._session_factory = SessionLocal
        return self._session_factory

    def _get_index(self):
        if self._index is None:
            from app.clients.milvus_client import MilvusIndex

            self._index = MilvusIndex()
        return self._index

    def _get_embedder(self):
        if self._embedder is None:
            from app.clients.embed_client import build_embedder

            self._embedder = build_embedder()
        return self._embedder

    # ------------------------------------------------------------ 主流程
    async def run(self, doc_id: str, *, max_pages: int | None = None) -> IngestReport:
        started = time.perf_counter()
        report = IngestReport(doc_id=doc_id, status="failed")

        async with self._get_session_factory()() as session:
            document = await session.get(Document, doc_id)
            if document is None:
                report.error = "文档不存在"
                return report
            path = document.storage_path
            kb_id = document.kb_id
            filename = document.filename

        try:
            parsed = await self._stage_parse(doc_id, path, filename, max_pages)
            report.pages = parsed.page_count
            if not parsed.pages:
                raise RuntimeError("解析结果为空：可能是纯图片页或加密 PDF")

            # 目录和版权页要在切块前剔除：它们含全书章节名，
            # 向量距离会贴着任何问题，把真正的正文答案挤掉
            filtered = strip_front_matter(
                parsed,
                drop_toc=settings.parser.drop_toc,
                drop_copyright=settings.parser.drop_copyright,
                drop_preface=settings.parser.drop_preface,
            )
            parsed = filtered.parsed
            report.dropped_pages = filtered.dropped_count
            if not parsed.pages:
                raise RuntimeError("剔除目录/版权页后正文为空")

            plan = await self._stage_chunk(doc_id, kb_id, parsed)
            report.parents = len(plan.parents)
            report.children = len(plan.children)
            if not plan.children:
                raise RuntimeError("切块结果为空：正文可能过短或全是乱码")

            report.vectors = await self._stage_embed(doc_id, kb_id, plan)

            await self._update(doc_id, status="done", page_count=parsed.page_count,
                               error_msg="", last_stage="done")
            report.status = "done"
            report.stats["parser"] = parsed.parser

        except Exception as exc:  # noqa: BLE001 - 后台任务必须兜住一切
            report.error = f"{type(exc).__name__}: {exc}"[:480]
            logger.exception("ingest.failed", extra={"extra_fields": {"doc_id": doc_id}})
            await self._update(doc_id, status="failed", error_msg=report.error,
                               last_stage="failed")

        # 计数必须在这里重算：走 HTTP 上传时 api 层算过一次，但 scripts/ingest_batch.py
        # 直接建文档记录、绕过了接口，不补这一步知识库会一直显示 0 个文档。
        # 失败也要算——文档记录确实存在，列表里得看得见它。
        await self._refresh_kb_counts(kb_id)

        report.elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.info(
            "ingest.done",
            extra={
                "extra_fields": {
                    "doc_id": doc_id,
                    "status": report.status,
                    "pages": report.pages,
                    "parents": report.parents,
                    "children": report.children,
                    "vectors": report.vectors,
                    "elapsed_ms": report.elapsed_ms,
                    "error": report.error,
                }
            },
        )
        return report

    # ------------------------------------------------------------ 各阶段
    async def _stage_parse(
        self, doc_id: str, path: str, filename: str, max_pages: int | None
    ) -> ParsedDoc:
        await self._update(doc_id, status="parsing", last_stage="parsing")
        parser = self._parser
        if parser is None:
            from app.ingest.parsers import pick_parser

            parser = pick_parser(filename)
        # 解析是同步阻塞的（MinerU/CPU 推理），放线程池避免卡事件循环
        return await asyncio.to_thread(parser.parse, path, max_pages=max_pages)

    async def _stage_chunk(self, doc_id: str, kb_id: str, parsed: ParsedDoc) -> ChunkPlan:
        await self._update(doc_id, status="chunking", last_stage="chunking")
        plan = build_chunks(doc_id, parsed)
        plan, touched, dropped = clean_watermarks(plan)
        if touched:
            logger.info(
                "ingest.watermark_cleaned",
                extra={"extra_fields": {"doc_id": doc_id, "blocks": touched, "dropped": dropped}},
            )

        async with self._get_session_factory()() as session:
            # 幂等：先清掉这个文档的旧块，否则重跑会翻倍
            await session.execute(delete(Chunk).where(Chunk.doc_id == doc_id))
            for seq, item in enumerate(plan.children):
                session.add(
                    Chunk(
                        id=item["chunk_id"],
                        doc_id=doc_id,
                        kb_id=kb_id,
                        parent_id=item["parent_id"],
                        seq=seq,
                        content=item["content"],
                        char_count=len(item["content"]),
                        page_start=item["page_start"],
                        page_end=item["page_end"],
                        meta={"indexed": False},
                    )
                )
            await session.commit()
        return plan

    async def _stage_embed(self, doc_id: str, kb_id: str, plan: ChunkPlan) -> int:
        await self._update(doc_id, status="embedding", last_stage="embedding")
        embedder = self._get_embedder()
        index = self._get_index()

        # 父块 + 子块一起向量化：父块必须进 Milvus，命中子块后要回查它的全文。
        # 用 is_parent 标记身份，检索时据此把父块排除在召回之外。
        items = [(item, "", True) for item in plan.parents] + [
            (item, item["parent_id"], False) for item in plan.children
        ]
        total = 0
        for start in range(0, len(items), EMBED_BATCH):
            batch = items[start : start + EMBED_BATCH]
            vectors = await embedder.encode([item["content"] for item, _, _ in batch])
            records = []
            for (item, parent_id, is_parent), vector in zip(batch, vectors, strict=True):
                record = {
                    "chunk_id": item["chunk_id"],
                    "kb_id": kb_id,
                    "doc_id": doc_id,
                    "parent_id": parent_id,
                    "content": item["content"][:8000],
                    "dense_vector": vector.dense,
                    "page_start": item["page_start"],
                    "page_end": item["page_end"],
                    "domain": "",
                    "evidence_grade": "",
                    "is_parent": is_parent,
                }
                # 向量来自 API 时没有稀疏权重，dense-only 集合里也没有这个字段。
                # 无条件塞空 dict 会被 Milvus 拒掉（稀疏向量不允许全空）。
                if vector.sparse:
                    record["sparse_vector"] = {int(k): v for k, v in vector.sparse.items()}
                records.append(record)
            # 先清掉旧向量再写，避免重跑留下孤儿数据
            if start == 0:
                await index.delete_by_doc(doc_id)
            total += await index.upsert(records)

        # 整篇写完刷一次盘。放在每批 upsert 里会让一本 500 页的书多花一小时
        # （实测单次 flush 约 30 秒）。替身对象没有 flush，跳过即可。
        flush = getattr(index, "flush", None)
        if flush is not None:
            await flush()

        async with self._get_session_factory()() as session:
            chunks = (
                await session.scalars(select(Chunk).where(Chunk.doc_id == doc_id))
            ).all()
            for row in chunks:
                row.meta = {**(row.meta or {}), "indexed": True}
            await session.commit()
        return total

    async def _update(self, doc_id: str, **fields: Any) -> None:
        async with self._get_session_factory()() as session:
            document = await session.get(Document, doc_id)
            if document is None:
                return
            for key, value in fields.items():
                setattr(document, key, value)
            await session.commit()

    async def _refresh_kb_counts(self, kb_id: str) -> None:
        """重算知识库的文档数与块数。

        用 count 查询而不是自增：重跑、删除、失败三种情况都会让自增算错，
        而列表页显示的数字必须和实际条数一致，否则用户会以为入库没成功。
        """
        async with self._get_session_factory()() as session:
            kb = await session.get(KnowledgeBase, kb_id)
            if kb is None:
                return
            kb.doc_count = int(
                await session.scalar(
                    select(func.count(Document.id)).where(Document.kb_id == kb_id)
                )
                or 0
            )
            kb.chunk_count = int(
                await session.scalar(
                    select(func.count(Chunk.id)).where(Chunk.kb_id == kb_id)
                )
                or 0
            )
            await session.commit()
