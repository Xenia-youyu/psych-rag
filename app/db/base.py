"""数据库连接与会话。

- 异步引擎，连接串由 config.ini 拼装。
- get_session 作为 FastAPI 依赖，请求结束自动关闭。
- init_models() 仅用于开发期建表；生产应走迁移脚本（本脚本不替代迁移）。
"""

from __future__ import annotations

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import settings
from app.core.logging import get_logger
from app.db.models import Base

logger = get_logger(__name__)

engine = create_async_engine(
    settings.mysql.dsn,
    pool_size=settings.mysql.pool_size,
    pool_recycle=settings.mysql.pool_recycle,
    pool_pre_ping=True,
    echo=False,
)

SessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI 依赖：一次请求一个会话。"""
    async with SessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise


async def init_models() -> None:
    """开发期建表。已有表不会被修改，改字段请写迁移 SQL。"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("db.init_models.done", extra={"extra_fields": {"tables": len(Base.metadata.tables)}})


async def ping() -> bool:
    """健康检查用：能否执行一次 SELECT 1。"""
    from sqlalchemy import text

    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("db.ping.failed", extra={"extra_fields": {"error": repr(exc)}})
        return False
