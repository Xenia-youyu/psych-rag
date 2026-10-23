"""入库流水线测试：用 SQLite + 替身跑完整流程，不碰真实模型和 Milvus。

重点验证四件"只在串联时才暴露"的事：
- 父块与子块的 chunk_id/parent_id 关系是否闭合（父块必须能被回查到）
- 重跑是否幂等（同一本书点两次上传，块数不能翻倍）
- 失败是否写进 Document.error_msg（后台任务抛异常没人看得见）
- 空解析是否能被明确拒绝（否则会入库 0 条还显示成功）
"""

from __future__ import annotations

import uuid

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.db.models import Base, Chunk, Document, KnowledgeBase, User
from app.ingest.base import ParsedDoc, ParsedPage
from app.ingest.pipeline import IngestPipeline, build_chunks


# ---------------------------------------------------------------- 替身
class FakeParser:
    name = "fake"

    def __init__(self, pages: list[str] | None = None) -> None:
        self.pages = pages if pages is not None else ["第一页正文。" * 60, "第二页正文。" * 60]
        self.calls: list[str] = []

    def parse(
        self,
        path: str,
        *,
        max_pages: int | None = None,
        start_page: int | None = None,
        end_page: int | None = None,
    ) -> ParsedDoc:
        self.calls.append(path)
        return ParsedDoc(
            source_path=path,
            parser=self.name,
            pages=[ParsedPage(page_no=i + 1, text=t) for i, t in enumerate(self.pages)],
            markdown="\n\n".join(self.pages),
        )


class BrokenParser:
    name = "broken"

    def parse(
        self,
        path: str,
        *,
        max_pages: int | None = None,
        start_page: int | None = None,
        end_page: int | None = None,
    ) -> ParsedDoc:
        raise RuntimeError("解析器炸了")


class FakeEmbedding:
    def __init__(self, dense: list[float], sparse: dict[str, float]) -> None:
        self.dense = dense
        self.sparse = sparse


class FakeEmbedder:
    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    async def encode(self, texts: list[str]) -> list[FakeEmbedding]:
        self.batches.append(list(texts))
        return [FakeEmbedding(dense=[0.1, 0.2, 0.3], sparse={"1": 0.5}) for _ in texts]


class FakeIndex:
    def __init__(self, *, fail_upsert: bool = False) -> None:
        self.upserted: list[dict] = []
        self.deleted_docs: list[str] = []
        self.fail_upsert = fail_upsert

    async def delete_by_doc(self, doc_id: str) -> None:
        self.deleted_docs.append(doc_id)

    async def upsert(self, records: list[dict]) -> int:
        if self.fail_upsert:
            raise RuntimeError("milvus 挂了")
        self.upserted.extend(records)
        return len(records)


# ---------------------------------------------------------------- 夹具
@pytest_asyncio.fixture
async def db():
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:", poolclass=StaticPool
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)

    async with factory() as session:
        user = User(username="tester", password_hash="x")
        session.add(user)
        await session.flush()
        kb = KnowledgeBase(id=uuid.uuid4().hex, owner_id=user.id, name="测试库")
        session.add(kb)
        await session.flush()
        doc = Document(
            id=uuid.uuid4().hex,
            kb_id=kb.id,
            filename="book.pdf",
            file_type="pdf",
            status="uploaded",
            last_stage="uploaded",
            storage_path="D:/fake/book.pdf",
        )
        session.add(doc)
        await session.commit()
        ids = {"user": user.id, "kb": kb.id, "doc": doc.id}

    yield factory, ids
    await engine.dispose()


class TestBuildChunks:
    def test_parent_child_relation_is_closed(self) -> None:
        """每个子块的 parent_id 都必须能在父块里找到，否则回填父块会失败。"""
        parsed = ParsedDoc(
            source_path="x",
            parser="fake",
            pages=[ParsedPage(page_no=7, text="甲" * 3000)],
        )
        plan = build_chunks("doc1", parsed)

        parent_ids = {p["chunk_id"] for p in plan.parents}
        assert parent_ids, "没有切出父块"
        for child in plan.children:
            assert child["parent_id"] in parent_ids

    def test_page_number_carried(self) -> None:
        """页码必须跟着块走，引用溯源全靠它。"""
        parsed = ParsedDoc(
            source_path="x",
            parser="fake",
            pages=[
                ParsedPage(page_no=12, text="乙" * 1500),
                ParsedPage(page_no=13, text="丙" * 1500),
            ],
        )
        plan = build_chunks("doc1", parsed)

        pages = {p["page_start"] for p in plan.parents}
        assert pages == {12, 13}

    def test_empty_pages_skipped(self) -> None:
        parsed = ParsedDoc(
            source_path="x",
            parser="fake",
            pages=[ParsedPage(page_no=1, text="   "), ParsedPage(page_no=2, text="有内容")],
        )
        plan = build_chunks("doc1", parsed)

        assert not plan.parents == []
        assert all(p["page_start"] == 2 for p in plan.parents)


class TestPipelineSuccess:
    async def test_full_run_marks_done(self, db) -> None:
        factory, ids = db
        embedder, index = FakeEmbedder(), FakeIndex()
        pipeline = IngestPipeline(
            session_factory=factory, parser=FakeParser(), embedder=embedder, index=index
        )

        report = await pipeline.run(ids["doc"])

        assert report.ok, report.error
        assert report.parents > 0 and report.children > 0
        assert report.vectors == report.parents + report.children

        async with factory() as session:
            doc = await session.get(Document, ids["doc"])
            assert doc.status == "done"
            assert doc.error_msg == ""

    async def test_mysql_stores_children_only(self, db) -> None:
        """MySQL 只存子块，父块只进 Milvus——否则文本块列表页全是重复内容。"""
        factory, ids = db
        pipeline = IngestPipeline(
            session_factory=factory, parser=FakeParser(), embedder=FakeEmbedder(), index=FakeIndex()
        )
        report = await pipeline.run(ids["doc"])

        async with factory() as session:
            rows = (await session.scalars(Chunk.__table__.select())).all()
        assert len(rows) == report.children

    async def test_vectors_include_parents(self, db) -> None:
        """父块必须进 Milvus：命中子块后要靠它回查全文。"""
        factory, ids = db
        index = FakeIndex()
        pipeline = IngestPipeline(
            session_factory=factory, parser=FakeParser(), embedder=FakeEmbedder(), index=index
        )
        await pipeline.run(ids["doc"])

        parent_records = [r for r in index.upserted if r["parent_id"] == ""]
        child_records = [r for r in index.upserted if r["parent_id"]]
        assert parent_records and child_records
        assert len(parent_records) + len(child_records) == len(index.upserted)

    async def test_rerun_is_idempotent(self, db) -> None:
        """同一本书跑两次，块数不能翻倍。"""
        factory, ids = db
        pipeline = IngestPipeline(
            session_factory=factory, parser=FakeParser(), embedder=FakeEmbedder(), index=FakeIndex()
        )
        first = await pipeline.run(ids["doc"])
        second = await pipeline.run(ids["doc"])

        assert first.children == second.children
        async with factory() as session:
            rows = (await session.scalars(Chunk.__table__.select())).all()
        assert len(rows) == second.children

    async def test_old_vectors_deleted_before_write(self, db) -> None:
        factory, ids = db
        index = FakeIndex()
        pipeline = IngestPipeline(
            session_factory=factory, parser=FakeParser(), embedder=FakeEmbedder(), index=index
        )
        await pipeline.run(ids["doc"])

        assert ids["doc"] in index.deleted_docs


class TestPipelineFailure:
    async def test_parser_error_recorded(self, db) -> None:
        """后台任务抛异常没人看得见，必须落进 error_msg 供前端展示。"""
        factory, ids = db
        pipeline = IngestPipeline(
            session_factory=factory, parser=BrokenParser(), embedder=FakeEmbedder(), index=FakeIndex()
        )
        report = await pipeline.run(ids["doc"])

        assert not report.ok
        assert "解析器炸了" in report.error

        async with factory() as session:
            doc = await session.get(Document, ids["doc"])
            assert doc.status == "failed"
            assert "解析器炸了" in doc.error_msg

    async def test_empty_parse_is_rejected(self, db) -> None:
        """解析出 0 页却报成功，是最容易骗过自己的失败。"""
        factory, ids = db
        pipeline = IngestPipeline(
            session_factory=factory,
            parser=FakeParser(pages=[]),
            embedder=FakeEmbedder(),
            index=FakeIndex(),
        )
        report = await pipeline.run(ids["doc"])

        assert not report.ok
        assert "解析结果为空" in report.error

    async def test_milvus_failure_recorded(self, db) -> None:
        factory, ids = db
        pipeline = IngestPipeline(
            session_factory=factory,
            parser=FakeParser(),
            embedder=FakeEmbedder(),
            index=FakeIndex(fail_upsert=True),
        )
        report = await pipeline.run(ids["doc"])

        assert not report.ok
        assert "milvus 挂了" in report.error

    async def test_missing_document(self, db) -> None:
        factory, _ = db
        pipeline = IngestPipeline(
            session_factory=factory, parser=FakeParser(), embedder=FakeEmbedder(), index=FakeIndex()
        )
        report = await pipeline.run("not-exist")

        assert not report.ok
        assert "文档不存在" in report.error


class TestParentFlag:
    """父块与子块写进同一个 collection，必须靠 is_parent 区分身份。

    没有这个标记，Milvus 无从知道哪些实体不该被召回——
    而父块的向量是整页平均语义，恰恰最容易在 topK 里抢位置。
    """

    async def test_parent_records_flagged(self, db) -> None:
        factory, ids = db
        index = FakeIndex()
        pipeline = IngestPipeline(
            session_factory=factory, parser=FakeParser(), embedder=FakeEmbedder(), index=index
        )
        await pipeline.run(ids["doc"])

        parents = [r for r in index.upserted if r["is_parent"]]
        children = [r for r in index.upserted if not r["is_parent"]]

        assert parents and children, "父块或子块没写进 Milvus"
        assert len(parents) + len(children) == len(index.upserted)
        # 父块没有 parent_id，子块必须指向父块
        assert all(r["parent_id"] == "" for r in parents)
        assert all(r["parent_id"] for r in children)

    async def test_child_parent_id_resolves(self, db) -> None:
        factory, ids = db
        index = FakeIndex()
        pipeline = IngestPipeline(
            session_factory=factory, parser=FakeParser(), embedder=FakeEmbedder(), index=index
        )
        await pipeline.run(ids["doc"])

        parent_ids = {r["chunk_id"] for r in index.upserted if r["is_parent"]}
        referenced = {r["parent_id"] for r in index.upserted if not r["is_parent"]}
        assert referenced <= parent_ids, "子块指向了不存在的父块，回填会拿不到全文"
