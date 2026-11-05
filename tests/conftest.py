"""pytest 全局配置。

必须在任何 app.* 导入之前注入环境变量：app.config 是 fail-fast 的，
缺密钥会直接抛错，而测试不应该依赖开发者本机的真实密钥。

两套数据库策略：
- sqlite_client：内存 SQLite，跑全部接口逻辑测试，快且不依赖网络
- 真实 MySQL：只在 tests/api/test_integration_mysql.py 里用，默认跳过
"""

from __future__ import annotations

import os

os.environ.setdefault("DEEPSEEK_API_KEY", "test-deepseek-key")
os.environ.setdefault("DASHSCOPE_API_KEY", "test-dashscope-key")
os.environ.setdefault("JWT_SECRET", "test-jwt-secret-for-pytest-only-32bytes")

from collections.abc import AsyncGenerator  # noqa: E402

import pytest  # noqa: E402
import pytest_asyncio  # noqa: E402
from httpx import ASGITransport, AsyncClient  # noqa: E402
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

# 测试里把 bcrypt 成本降到 4：正确性不变，速度提升一个数量级。
# 生产仍用 config.ini 的 12。
from app.config import settings  # noqa: E402

settings.security.bcrypt_rounds = 4


def db_available() -> bool:
    """探测共享 MySQL 是否可连，用于决定真实集成用例是否跳过。"""
    try:
        import pymysql
    except ImportError:
        return False

    from app.config import settings

    try:
        conn = pymysql.connect(
            host=settings.mysql.host,
            port=settings.mysql.port,
            user=settings.mysql.user,
            password=settings.mysql.password,
            database=settings.mysql.database,
            connect_timeout=3,
        )
        conn.close()
        return True
    except Exception:  # noqa: BLE001
        return False


requires_db = pytest.mark.skipif(
    not db_available(),
    reason="共享 MySQL 不可达，跳过真实集成测试（先 python -m app.check_env --create）",
)


def milvus_available() -> bool:
    """探测共享 Milvus 是否可连。"""
    try:
        import socket

        from app.config import settings
    except ImportError:
        return False

    host, port = settings.milvus.host, int(settings.milvus.port)
    sock = socket.socket()
    sock.settimeout(3)
    try:
        return sock.connect_ex((host, port)) == 0
    finally:
        sock.close()


requires_milvus = pytest.mark.skipif(
    not milvus_available(),
    reason="共享 Milvus 不可达，跳过向量库测试",
)


@pytest.fixture(scope="session")
def anyio_backend() -> str:
    return "asyncio"


@pytest_asyncio.fixture
async def sqlite_client(monkeypatch) -> AsyncGenerator[AsyncClient, None]:
    """用内存 SQLite 跑接口测试。

    注意 StaticPool：:memory: 库是按连接隔离的，
    不共享连接池的话每个会话看到的是空库，测试会莫名其妙全挂。

    还要把 `app.db.base.SessionLocal` 一起换掉：上传接口的后台解析任务
    在函数内 `from app.db.base import SessionLocal` 取会话，只覆盖 FastAPI 的
    `get_session` 依赖管不住它——那批用例会真去连生产 MySQL，库一不可达就整片红，
    看着像代码坏了（2026-11-05 就这么白过一轮）。
    """
    from app.db.base import get_session
    from app.db.models import Base
    from app.main import app

    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    import app.db.base as db_base

    monkeypatch.setattr(db_base, "SessionLocal", session_factory)

    # 上传接口会把解析丢给 BackgroundTasks。接口测试不关心解析结果，
    # 但真让它跑会去解析测试用的假 PDF（PDFium 报格式错），进而把文档状态改成
    # failed，影响去重与统计两类断言。这里直接把它换成空操作。
    import app.api.documents as documents_api

    async def _noop_ingest(doc_id: str) -> None:
        return None

    monkeypatch.setattr(documents_api, "_run_ingest", _noop_ingest)

    async def override() -> AsyncGenerator:
        async with session_factory() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise

    app.dependency_overrides[get_session] = override
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        client.session_factory = session_factory  # type: ignore[attr-defined]
        yield client

    app.dependency_overrides.clear()
    await engine.dispose()
