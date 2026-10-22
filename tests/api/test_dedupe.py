"""上传去重测试：同一本书不该在同一个库里存在两份。

判重依据是文件内容的 sha256，范围限定在同一个知识库内——
同一本书进两个库是合理需求，不该被拦。

为什么这道闸门值得单独建测试：
    没有它时，重复入库会产生两份向量，检索返回两条一模一样的片段。
    这种问题在演示时被随手点出来的概率很高，而事后清理要按 doc_id 逐条删向量。
"""

from __future__ import annotations

from httpx import AsyncClient

from tests.api.test_kb import cleanup_uploads, create_kb, make_user, upload  # noqa: F401


class TestUploadDedupe:
    async def test_same_content_returns_existing(
        self, sqlite_client: AsyncClient, cleanup_uploads
    ) -> None:
        _, headers = await make_user(sqlite_client)
        kb_id = await create_kb(sqlite_client, headers)
        cleanup_uploads.append(kb_id)

        first = await upload(sqlite_client, headers, kb_id, payload=b"%PDF-1.4 same")
        second = await upload(sqlite_client, headers, kb_id, payload=b"%PDF-1.4 same")

        assert first.status_code == 200 and second.status_code == 200
        assert first.json()["duplicated"] is False
        assert second.json()["duplicated"] is True
        assert second.json()["id"] == first.json()["id"]

        kb = (await sqlite_client.get(f"/api/kb/{kb_id}", headers=headers)).json()
        assert kb["doc_count"] == 1, "重复上传把 doc_count 也算进去了"

    async def test_different_content_creates_new(
        self, sqlite_client: AsyncClient, cleanup_uploads
    ) -> None:
        _, headers = await make_user(sqlite_client)
        kb_id = await create_kb(sqlite_client, headers)
        cleanup_uploads.append(kb_id)

        await upload(sqlite_client, headers, kb_id, payload=b"%PDF-1.4 A")
        second = await upload(sqlite_client, headers, kb_id, payload=b"%PDF-1.4 B")

        assert second.json()["duplicated"] is False
        kb = (await sqlite_client.get(f"/api/kb/{kb_id}", headers=headers)).json()
        assert kb["doc_count"] == 2

    async def test_same_content_allowed_across_kbs(
        self, sqlite_client: AsyncClient, cleanup_uploads
    ) -> None:
        """同一本书进两个知识库是合理需求，去重范围只限同一个库。"""
        _, headers = await make_user(sqlite_client)
        kb_a = await create_kb(sqlite_client, headers, name="库A")
        kb_b = await create_kb(sqlite_client, headers, name="库B")
        cleanup_uploads.extend([kb_a, kb_b])

        a = await upload(sqlite_client, headers, kb_a, payload=b"%PDF-1.4 same")
        b = await upload(sqlite_client, headers, kb_b, payload=b"%PDF-1.4 same")

        assert a.json()["duplicated"] is False
        assert b.json()["duplicated"] is False

    async def test_failed_document_does_not_block_reupload(
        self, sqlite_client: AsyncClient, cleanup_uploads
    ) -> None:
        """上次解析失败的文档必须允许重传，否则用户永远修不好那本书。"""
        from app.db.models import Document

        _, headers = await make_user(sqlite_client)
        kb_id = await create_kb(sqlite_client, headers)
        cleanup_uploads.append(kb_id)

        first = (await upload(sqlite_client, headers, kb_id, payload=b"%PDF-1.4 bad")).json()

        async with sqlite_client.session_factory() as session:
            doc = await session.get(Document, first["id"])
            doc.status = "failed"
            await session.commit()

        again = await upload(sqlite_client, headers, kb_id, payload=b"%PDF-1.4 bad")
        assert again.json()["duplicated"] is False
        assert again.json()["id"] != first["id"]
