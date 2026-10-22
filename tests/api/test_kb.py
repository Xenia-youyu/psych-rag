"""知识库 / 文档 / 文本块 接口测试（内存 SQLite，不需要 MySQL）。"""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import pytest
from httpx import AsyncClient

from app.config import BASE_DIR, settings
from app.core.captcha import hash_code
from app.db.models import Captcha, Chunk

VALID_CODE = "AB34"


async def add_captcha(client: AsyncClient, scene: str) -> str:
    """直接往表里塞一条已知答案的验证码。图片无法回读，这是测试唯一可行的做法。"""
    from datetime import datetime, timedelta

    captcha_id = str(uuid.uuid4())
    async with client.session_factory() as session:
        session.add(
            Captcha(
                id=captcha_id,
                code_hash=hash_code(VALID_CODE),
                scene=scene,
                expires_at=datetime.now() + timedelta(minutes=5),
            )
        )
        await session.commit()
    return captcha_id


async def make_user(client: AsyncClient, password: str = "abc12345") -> tuple[str, dict]:
    """注册 + 登录，返回 (username, 认证头)。"""
    username = f"stu{uuid.uuid4().hex[:8]}"

    await client.post(
        "/api/auth/register",
        json={
            "username": username,
            "password": password,
            "password_confirm": password,
            "captcha_id": await add_captcha(client, "register"),
            "captcha_code": VALID_CODE,
        },
    )

    login = await client.post(
        "/api/auth/login",
        json={
            "username": username,
            "password": password,
            "captcha_id": await add_captcha(client, "login"),
            "captcha_code": VALID_CODE,
        },
    )
    token = login.json()["access_token"]
    return username, {"Authorization": f"Bearer {token}"}


async def create_kb(client: AsyncClient, headers: dict, name: str = "普心") -> str:
    response = await client.post("/api/kb", json={"name": name, "description": "测试"}, headers=headers)
    assert response.status_code == 200, response.text
    return response.json()["id"]


@pytest.fixture
async def cleanup_uploads():
    created: list[str] = []
    yield created
    root = Path(settings.app.upload_dir)
    if not root.is_absolute():
        root = BASE_DIR / root
    for kb_id in created:
        shutil.rmtree(root / kb_id, ignore_errors=True)


async def upload(client: AsyncClient, headers: dict, kb_id: str, *,
                 filename: str = "普通心理学.pdf", payload: bytes = b"%PDF-1.4 test") -> object:
    return await client.post(
        f"/api/kb/{kb_id}/documents",
        headers=headers,
        files={"file": (filename, payload, "application/octet-stream")},
        data={"chunk_size": "800", "overlap_ratio": "0.1", "hybrid": "true", "mode": "single"},
    )


class TestAuthRequired:
    @pytest.mark.parametrize(
        ("method", "path"),
        [
            ("GET", "/api/kb"),
            ("POST", "/api/kb"),
            ("GET", "/api/documents/xxx/chunks"),
            ("GET", "/api/chunks/xxx"),
        ],
    )
    async def test_requires_login(self, sqlite_client: AsyncClient, method: str, path: str) -> None:
        response = await sqlite_client.request(method, path)
        assert response.status_code == 401


class TestKnowledgeBase:
    async def test_create_and_list(self, sqlite_client: AsyncClient) -> None:
        _, headers = await make_user(sqlite_client)
        kb_id = await create_kb(sqlite_client, headers, "普心")

        response = await sqlite_client.get("/api/kb", headers=headers)
        body = response.json()

        assert response.status_code == 200
        assert len(body) == 1
        assert body[0]["id"] == kb_id
        assert body[0]["name"] == "普心"
        assert body[0]["doc_count"] == 0

    async def test_create_requires_name(self, sqlite_client: AsyncClient) -> None:
        _, headers = await make_user(sqlite_client)
        response = await sqlite_client.post("/api/kb", json={"name": "  "}, headers=headers)
        assert response.status_code == 400

    async def test_list_only_own(self, sqlite_client: AsyncClient) -> None:
        _, headers_a = await make_user(sqlite_client)
        _, headers_b = await make_user(sqlite_client)
        await create_kb(sqlite_client, headers_a, "A 的库")
        await create_kb(sqlite_client, headers_b, "B 的库")

        body = (await sqlite_client.get("/api/kb", headers=headers_a)).json()
        assert len(body) == 1
        assert body[0]["name"] == "A 的库"

    async def test_cross_user_access_returns_404(self, sqlite_client: AsyncClient) -> None:
        _, headers_a = await make_user(sqlite_client)
        _, headers_b = await make_user(sqlite_client)
        kb_id = await create_kb(sqlite_client, headers_a)

        # 越权必须是 404 而不是 403：403 会暴露"该 ID 存在"
        for method in ("GET", "PATCH", "DELETE"):
            response = await sqlite_client.request(
                method, f"/api/kb/{kb_id}", headers=headers_b,
                json={"name": "偷改"} if method == "PATCH" else None,
            )
            assert response.status_code == 404, f"{method} 越权应为 404"

    async def test_update_name(self, sqlite_client: AsyncClient) -> None:
        _, headers = await make_user(sqlite_client)
        kb_id = await create_kb(sqlite_client, headers)

        response = await sqlite_client.patch(
            f"/api/kb/{kb_id}", json={"name": "发展心理学"}, headers=headers
        )
        assert response.status_code == 200
        assert response.json()["name"] == "发展心理学"

    async def test_delete_kb(self, sqlite_client: AsyncClient) -> None:
        _, headers = await make_user(sqlite_client)
        kb_id = await create_kb(sqlite_client, headers)

        assert (await sqlite_client.delete(f"/api/kb/{kb_id}", headers=headers)).status_code == 200
        assert (await sqlite_client.get(f"/api/kb/{kb_id}", headers=headers)).status_code == 404

    async def test_stats(self, sqlite_client: AsyncClient, cleanup_uploads) -> None:
        _, headers = await make_user(sqlite_client)
        kb_id = await create_kb(sqlite_client, headers)
        cleanup_uploads.append(kb_id)
        await upload(sqlite_client, headers, kb_id)

        body = (await sqlite_client.get(f"/api/kb/{kb_id}/stats", headers=headers)).json()
        assert body["documents"] == 1
        assert body["chunks"] == 0
        assert body["failed_documents"] == 0


class TestDocuments:
    async def test_upload_pdf(self, sqlite_client: AsyncClient, cleanup_uploads) -> None:
        _, headers = await make_user(sqlite_client)
        kb_id = await create_kb(sqlite_client, headers)
        cleanup_uploads.append(kb_id)

        response = await upload(sqlite_client, headers, kb_id)
        body = response.json()

        assert response.status_code == 200, response.text
        assert body["status"] == "uploaded"
        assert body["last_stage"] == "uploaded"
        assert body["file_type"] == "pdf"
        assert body["size_bytes"] > 0
        assert body["parse_params"]["chunk_size"] == 800

        # doc_count 不能多算
        kb = (await sqlite_client.get(f"/api/kb/{kb_id}", headers=headers)).json()
        assert kb["doc_count"] == 1

    async def test_unsupported_type_rejected(
        self, sqlite_client: AsyncClient, cleanup_uploads
    ) -> None:
        _, headers = await make_user(sqlite_client)
        kb_id = await create_kb(sqlite_client, headers)
        cleanup_uploads.append(kb_id)

        response = await upload(sqlite_client, headers, kb_id, filename="病毒.exe")
        assert response.status_code == 415

    async def test_size_limit(self, sqlite_client: AsyncClient, cleanup_uploads, monkeypatch) -> None:
        _, headers = await make_user(sqlite_client)
        kb_id = await create_kb(sqlite_client, headers)
        cleanup_uploads.append(kb_id)

        monkeypatch.setattr(settings.app, "max_upload_mb", 0)
        response = await upload(sqlite_client, headers, kb_id, payload=b"x" * 2048)
        assert response.status_code == 413

    async def test_list_and_delete_document(
        self, sqlite_client: AsyncClient, cleanup_uploads
    ) -> None:
        _, headers = await make_user(sqlite_client)
        kb_id = await create_kb(sqlite_client, headers)
        cleanup_uploads.append(kb_id)

        doc_id = (await upload(sqlite_client, headers, kb_id)).json()["id"]

        listed = await sqlite_client.get(f"/api/kb/{kb_id}/documents", headers=headers)
        assert len(listed.json()) == 1

        assert (
            await sqlite_client.delete(f"/api/documents/{doc_id}", headers=headers)
        ).status_code == 200
        assert (await sqlite_client.get(f"/api/kb/{kb_id}/documents", headers=headers)).json() == []

    async def test_cross_user_document_access_404(
        self, sqlite_client: AsyncClient, cleanup_uploads
    ) -> None:
        _, headers_a = await make_user(sqlite_client)
        _, headers_b = await make_user(sqlite_client)
        kb_id = await create_kb(sqlite_client, headers_a)
        cleanup_uploads.append(kb_id)
        doc_id = (await upload(sqlite_client, headers_a, kb_id)).json()["id"]

        response = await sqlite_client.get(f"/api/documents/{doc_id}", headers=headers_b)
        assert response.status_code == 404

    async def test_supported_types(self, sqlite_client: AsyncClient) -> None:
        response = await sqlite_client.get("/api/supported-types")
        body = response.json()
        assert ".pdf" in body["extensions"]
        assert ".epub" in body["extensions"]
        assert body["max_upload_mb"] > 0


class TestChunks:
    async def _prepare_doc(self, client: AsyncClient, headers: dict) -> tuple[str, str]:
        kb_id = await create_kb(client, headers)
        doc_id = (await upload(client, headers, kb_id)).json()["id"]
        return kb_id, doc_id

    async def _seed_chunks(self, client: AsyncClient, kb_id: str, doc_id: str, count: int = 5) -> None:
        async with client.session_factory() as session:
            for index in range(count):
                session.add(
                    Chunk(
                        id=f"{doc_id}_c{index}",
                        doc_id=doc_id,
                        kb_id=kb_id,
                        parent_id=None,
                        seq=index,
                        content=f"第 {index} 块内容：操作性条件作用",
                        char_count=20,
                        page_start=index + 1,
                        page_end=index + 1,
                        meta={},
                        enabled=True,
                    )
                )
            await session.commit()

    async def test_pagination(self, sqlite_client: AsyncClient, cleanup_uploads) -> None:
        _, headers = await make_user(sqlite_client)
        kb_id, doc_id = await self._prepare_doc(sqlite_client, headers)
        cleanup_uploads.append(kb_id)
        await self._seed_chunks(sqlite_client, kb_id, doc_id, 5)

        page1 = (await sqlite_client.get(
            f"/api/documents/{doc_id}/chunks", params={"page": 1, "size": 2}, headers=headers
        )).json()

        assert page1["total"] == 5
        assert len(page1["items"]) == 2
        assert page1["items"][0]["seq"] == 0

        page3 = (await sqlite_client.get(
            f"/api/documents/{doc_id}/chunks", params={"page": 3, "size": 2}, headers=headers
        )).json()
        assert len(page3["items"]) == 1

    async def test_keyword_filter(self, sqlite_client: AsyncClient, cleanup_uploads) -> None:
        _, headers = await make_user(sqlite_client)
        kb_id, doc_id = await self._prepare_doc(sqlite_client, headers)
        cleanup_uploads.append(kb_id)
        await self._seed_chunks(sqlite_client, kb_id, doc_id, 3)

        body = (await sqlite_client.get(
            f"/api/documents/{doc_id}/chunks", params={"keyword": "第 1 块"}, headers=headers
        )).json()
        assert body["total"] == 1

    async def test_update_marks_needs_reindex(
        self, sqlite_client: AsyncClient, cleanup_uploads
    ) -> None:
        _, headers = await make_user(sqlite_client)
        kb_id, doc_id = await self._prepare_doc(sqlite_client, headers)
        cleanup_uploads.append(kb_id)
        await self._seed_chunks(sqlite_client, kb_id, doc_id, 1)

        chunk_id = f"{doc_id}_c0"
        response = await sqlite_client.patch(
            f"/api/chunks/{chunk_id}", json={"content": "改写后的内容"}, headers=headers
        )
        body = response.json()

        assert response.status_code == 200
        assert body["content"] == "改写后的内容"
        assert body["char_count"] == 6
        assert body["meta"]["needs_reindex"] is True

        pending = (await sqlite_client.get(
            f"/api/documents/{doc_id}/pending-reindex", headers=headers
        )).json()
        assert pending["pending"] == 1

    async def test_disable_chunk(self, sqlite_client: AsyncClient, cleanup_uploads) -> None:
        _, headers = await make_user(sqlite_client)
        kb_id, doc_id = await self._prepare_doc(sqlite_client, headers)
        cleanup_uploads.append(kb_id)
        await self._seed_chunks(sqlite_client, kb_id, doc_id, 2)

        chunk_id = f"{doc_id}_c0"
        await sqlite_client.patch(f"/api/chunks/{chunk_id}", json={"enabled": False}, headers=headers)

        enabled_only = (await sqlite_client.get(
            f"/api/documents/{doc_id}/chunks", params={"only_enabled": "true"}, headers=headers
        )).json()
        assert enabled_only["total"] == 1

    async def test_empty_content_rejected(self, sqlite_client: AsyncClient, cleanup_uploads) -> None:
        _, headers = await make_user(sqlite_client)
        kb_id, doc_id = await self._prepare_doc(sqlite_client, headers)
        cleanup_uploads.append(kb_id)
        await self._seed_chunks(sqlite_client, kb_id, doc_id, 1)

        response = await sqlite_client.patch(
            f"/api/chunks/{doc_id}_c0", json={"content": "   "}, headers=headers
        )
        assert response.status_code == 400

    async def test_delete_chunk(self, sqlite_client: AsyncClient, cleanup_uploads) -> None:
        _, headers = await make_user(sqlite_client)
        kb_id, doc_id = await self._prepare_doc(sqlite_client, headers)
        cleanup_uploads.append(kb_id)
        await self._seed_chunks(sqlite_client, kb_id, doc_id, 2)

        assert (
            await sqlite_client.delete(f"/api/chunks/{doc_id}_c0", headers=headers)
        ).status_code == 200
        body = (await sqlite_client.get(
            f"/api/documents/{doc_id}/chunks", headers=headers
        )).json()
        assert body["total"] == 1

    async def test_cross_user_chunk_404(self, sqlite_client: AsyncClient, cleanup_uploads) -> None:
        _, headers_a = await make_user(sqlite_client)
        _, headers_b = await make_user(sqlite_client)
        kb_id, doc_id = await self._prepare_doc(sqlite_client, headers_a)
        cleanup_uploads.append(kb_id)
        await self._seed_chunks(sqlite_client, kb_id, doc_id, 1)

        response = await sqlite_client.get(f"/api/chunks/{doc_id}_c0", headers=headers_b)
        assert response.status_code == 404
