"""每日推荐测试。

最该钉住的是「当天固定」这条：如果每次刷新都换一批，用户看到一条想细看，
回头就找不到了。这比"随机够不够随机"重要得多。
"""

from __future__ import annotations

import pytest_asyncio
from httpx import AsyncClient

from app.api import recommend as recommend_api
from app.db.models import Chunk, Document
from tests.api.test_kb import create_kb, make_user  # noqa: F401

LONG = (
    "自变量是实验者主动操纵的刺激或条件，因变量是由自变量变化引起的现象变化。"
    "二者的区分要点在于谁被操纵、谁被测量。实验设计时必须明确这一点，"
    "否则无法判断结果究竟由哪个因素导致。"
)


class FakeIndex:
    def __init__(self) -> None:
        self.search_calls = 0

    async def get_vector(self, chunk_id: str):
        return [0.1, 0.2, 0.3]

    async def fetch_chunks(self, chunk_ids: list[str]):
        return {
            chunk_id: {"content": f"父块全文：{LONG}" * 3, "page_start": 1}
            for chunk_id in chunk_ids
        }

    async def search(self, **_: object):
        self.search_calls += 1
        return [
            {"chunk_id": "c0", "doc_id": "doc1", "parent_id": "p0", "content": LONG, "page_start": 1},
            {"chunk_id": "c1", "doc_id": "doc1", "parent_id": "p1", "content": "关联的因变量测量方法。" * 4, "page_start": 2},
            {"chunk_id": "c2", "doc_id": "doc1", "parent_id": "p2", "content": "另一条关联内容。" * 4, "page_start": 3},
        ]


class FakeRuntime:
    def __init__(self) -> None:
        self.index = FakeIndex()
        self.embedder = object()
        self.reranker = object()
        self.llm = object()


@pytest_asyncio.fixture
async def rec_env(sqlite_client: AsyncClient, monkeypatch):
    runtime = FakeRuntime()
    monkeypatch.setattr(recommend_api, "get_runtime", lambda: runtime)
    return sqlite_client, runtime


async def seed(client: AsyncClient, kb_id: str, count: int = 8) -> None:
    async with client.session_factory() as session:
        session.add(
            Document(
                id="doc1",
                kb_id=kb_id,
                filename="普通心理学（第3版）.pdf",
                file_type="pdf",
                status="done",
                last_stage="done",
                storage_path="D:/x.pdf",
            )
        )
        for index in range(count):
            content = f"{index}、{LONG}"
            session.add(
                Chunk(
                    id=f"c{index}",
                    doc_id="doc1",
                    kb_id=kb_id,
                    parent_id=f"p{index}",
                    seq=index,
                    content=content,
                    char_count=len(content),
                    page_start=index + 1,
                    page_end=index + 1,
                )
            )
        await session.commit()


class TestDaily:
    async def test_same_day_is_stable(self, rec_env) -> None:
        client, _ = rec_env
        _, headers = await make_user(client)
        kb_id = await create_kb(client, headers)
        await seed(client, kb_id)

        first = (await client.get(f"/api/recommend/daily?kb_id={kb_id}", headers=headers)).json()
        second = (await client.get(f"/api/recommend/daily?kb_id={kb_id}", headers=headers)).json()

        assert [item["chunk_id"] for item in first["items"]] == [
            item["chunk_id"] for item in second["items"]
        ]
        assert first["items"], "一条推荐都没出，说明候选被过滤光了"

    async def test_variation_changes_pick(self, rec_env) -> None:
        client, _ = rec_env
        _, headers = await make_user(client)
        kb_id = await create_kb(client, headers)
        await seed(client, kb_id)

        base = (await client.get(f"/api/recommend/daily?kb_id={kb_id}", headers=headers)).json()
        other = (
            await client.get(f"/api/recommend/daily?kb_id={kb_id}&variation=1", headers=headers)
        ).json()

        assert [i["chunk_id"] for i in base["items"]] != [i["chunk_id"] for i in other["items"]]

    async def test_has_doc_title_and_page(self, rec_env) -> None:
        client, _ = rec_env
        _, headers = await make_user(client)
        kb_id = await create_kb(client, headers)
        await seed(client, kb_id)

        data = (await client.get(f"/api/recommend/daily?kb_id={kb_id}", headers=headers)).json()
        item = data["items"][0]

        # 不带书名，用户不知道这条是哪本书里的
        assert item["doc_title"] == "普通心理学（第3版）"
        assert item["page"] >= 1
        assert item["title"] and len(item["title"]) <= 40

    async def test_foreign_kb_404(self, rec_env) -> None:
        client, _ = rec_env
        _, headers_a = await make_user(client)
        _, headers_b = await make_user(client)
        kb_b = await create_kb(client, headers_b)

        response = await client.get(f"/api/recommend/daily?kb_id={kb_b}", headers=headers_a)
        assert response.status_code == 404


class TestDetail:
    async def test_returns_parent_content(self, rec_env) -> None:
        """详情必须是父块全文，不能用 300 字的子块糊弄。"""
        client, _ = rec_env
        _, headers = await make_user(client)
        kb_id = await create_kb(client, headers)
        await seed(client, kb_id)

        data = (await client.get("/api/recommend/chunks/c3", headers=headers)).json()

        assert data["is_full_parent"] is True
        assert data["content"].startswith("父块全文")
        # 比子块（约 95 字）长得多，说明拿到的确实是父块而不是原文
        assert data["char_count"] > len(LONG) * 2

    async def test_related_excludes_self_and_siblings(self, rec_env) -> None:
        client, runtime = rec_env
        _, headers = await make_user(client)
        kb_id = await create_kb(client, headers)
        await seed(client, kb_id)

        data = (await client.get("/api/recommend/chunks/c1", headers=headers)).json()
        ids = [item["chunk_id"] for item in data["related"]]

        assert "c1" not in ids, "把自己当成了关联内容"
        # FakeIndex 里 c0 与 c1 的 parent_id 不同，应当保留；同父块的会被排掉
        assert runtime.index.search_calls == 1

    async def test_missing_chunk_404(self, rec_env) -> None:
        client, _ = rec_env
        _, headers = await make_user(client)
        response = await client.get("/api/recommend/chunks/not-exist", headers=headers)
        assert response.status_code == 404


class TestFilters:
    def test_rejects_table_of_contents(self) -> None:
        toc = "第一章 心理学概述........3\n第二章 研究方法........15\n第三章 感觉........28"
        assert recommend_api._usable(toc) is False

    def test_rejects_heading_only(self) -> None:
        assert recommend_api._usable("一、自变量和因变量的确定") is False

    def test_accepts_real_paragraph(self) -> None:
        assert recommend_api._usable(LONG * 2) is True

    def test_title_strips_numbering(self) -> None:
        assert recommend_api._title("（一）自变量和因变量的确定。") == "自变量和因变量的确定。"

    def test_title_marks_continued_fragment(self) -> None:
        """块首落在句子中间时要标出省略号，否则用户以为那是完整段落。"""
        fragment = "信地址是作者的重要信息之一。一般在发表作品时，应尽可能注明作者的详细工作单位。"
        assert recommend_api._title(fragment).startswith("…")

    def test_title_keeps_heading_without_marker(self) -> None:
        assert not recommend_api._title("三、实验设计的基本类型").startswith("…")

    def test_title_truncates(self) -> None:
        title = recommend_api._title("很长" * 60)
        # 首尾各可能有一个省略号，正文不超过 36 字
        assert len(title) <= 38
        assert title.startswith("…") and title.endswith("…")
