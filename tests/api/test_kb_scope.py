"""检索范围的解析逻辑（resolve_kb_ids）。

这组用例守的是一个真实故障：教材语料全部挂在做批量入库的账号名下，
其他用户登录后 kb_ids 为空 → 检索范围为空 → 问什么都答「未覆盖」，
而前端看不出任何异常。共享语料库的配置就是为了修它。
"""

from __future__ import annotations

import uuid

import pytest
from httpx import AsyncClient

from app.api.chat import resolve_kb_ids
from app.config import settings
from app.db.models import KnowledgeBase


async def _seed_kb(client: AsyncClient, owner_id: int, name: str) -> str:
    kb_id = uuid.uuid4().hex[:32]
    async with client.session_factory() as session:
        session.add(KnowledgeBase(id=kb_id, owner_id=owner_id, name=name))
        await session.commit()
    return kb_id


@pytest.fixture
def shared(monkeypatch):
    """把「共享库」临时指到测试造的 kb 上，避免依赖真实配置。"""
    ids: list[str] = []
    monkeypatch.setattr(settings.retrieval, "shared_kb_ids", ids)
    return ids


class TestResolveKbIds:
    async def test_empty_request_falls_back_to_shared_plus_own(
        self, sqlite_client: AsyncClient, shared
    ) -> None:
        pub = await _seed_kb(sqlite_client, owner_id=1, name="公共教材")
        own = await _seed_kb(sqlite_client, owner_id=7, name="我的库")
        other = await _seed_kb(sqlite_client, owner_id=8, name="别人的库")
        shared.append(pub)

        async with sqlite_client.session_factory() as session:
            got = await resolve_kb_ids([], user_id=7, session=session)

        assert set(got) == {pub, own}, "应包含共享库和自己的库"
        assert other not in got, "不能把别人的私有库带进来"

    async def test_requested_shared_kb_is_allowed(
        self, sqlite_client: AsyncClient, shared
    ) -> None:
        pub = await _seed_kb(sqlite_client, owner_id=1, name="公共教材")
        shared.append(pub)

        async with sqlite_client.session_factory() as session:
            got = await resolve_kb_ids([pub], user_id=7, session=session)
        assert got == [pub]

    async def test_requested_foreign_kb_is_rejected(
        self, sqlite_client: AsyncClient, shared
    ) -> None:
        other = await _seed_kb(sqlite_client, owner_id=8, name="别人的库")

        async with sqlite_client.session_factory() as session:
            with pytest.raises(Exception) as exc:
                await resolve_kb_ids([other], user_id=7, session=session)
        assert "KB" in str(exc.value) or "知识库" in str(exc.value)

    async def test_stale_shared_id_is_dropped(
        self, sqlite_client: AsyncClient, shared
    ) -> None:
        """配置里写了已删除的库：应剔除而不是让检索静默返回空。"""
        pub = await _seed_kb(sqlite_client, owner_id=1, name="公共教材")
        shared.extend([pub, "deadbeef" * 4])

        async with sqlite_client.session_factory() as session:
            got = await resolve_kb_ids([], user_id=7, session=session)
        assert got == [pub]
