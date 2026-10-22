"""数据库表定义（SQLAlchemy 2.0 声明式）。

与 v3 文档 3.2 节一致。改动字段前先确认三件事：
1. 是否影响前端已有字段
2. 是否需要迁移脚本（项目未引入 alembic 前，一律手工写 SQL）
3. 是否有索引支撑查询路径
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# MySQL 用 BIGINT 自增；SQLite（测试用）只认 INTEGER PRIMARY KEY 才能自增，
# 所以这里用 with_variant 让两边都能跑，生产 schema 不变。
BIGINT_PK = BigInteger().with_variant(Integer, "sqlite")
BIGINT_FK = BigInteger().with_variant(Integer, "sqlite")


class Base(DeclarativeBase):
    pass


# 表名统一带 psy_ 前缀。
# 原因：共享 MySQL 上我们只有 subjects_kg 一个库的权限，里面已经有别人的
# conversations / jpkb 两张表。不带前缀迟早撞名，撞了就是别人的数据被覆盖。


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now(), nullable=False
    )


class User(Base, TimestampMixin):
    __tablename__ = "psy_users"

    id: Mapped[int] = mapped_column(BIGINT_PK, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(32), unique=True, nullable=False, index=True)
    password_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[str] = mapped_column(String(16), default="active", nullable=False)
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class Captcha(Base):
    """M1 用表存验证码；M4 换 Redis 时本表保留做兜底。"""

    __tablename__ = "psy_captcha"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    code_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    scene: Mapped[str] = mapped_column(String(16), default="login", nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    used: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )


class KnowledgeBase(Base, TimestampMixin):
    __tablename__ = "psy_knowledge_bases"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    owner_id: Mapped[int] = mapped_column(
        BIGINT_FK, ForeignKey("psy_users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(50), nullable=False)
    description: Mapped[str] = mapped_column(String(200), default="", nullable=False)
    doc_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)


class Document(Base, TimestampMixin):
    __tablename__ = "psy_documents"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    kb_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("psy_knowledge_bases.id", ondelete="CASCADE"), nullable=False
    )
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    file_type: Mapped[str] = mapped_column(String(16), nullable=False)
    size_bytes: Mapped[int] = mapped_column(BigInteger, default=0, nullable=False)
    page_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    status: Mapped[str] = mapped_column(
        Enum(
            "uploaded",
            "normalizing",
            "parsing",
            "chunking",
            "embedding",
            "done",
            "failed",
            name="doc_status",
        ),
        default="uploaded",
        nullable=False,
    )
    # 断点续跑：记录最后一次成功的阶段，失败重试从这里继续
    last_stage: Mapped[str] = mapped_column(String(16), default="uploaded", nullable=False)
    error_msg: Mapped[str] = mapped_column(String(500), default="", nullable=False)
    parse_params: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    storage_path: Mapped[str] = mapped_column(String(500), default="", nullable=False)
    # 文件内容 sha256，用于上传去重。同一个库内同哈希视为同一本书，
    # 否则重复上传会入库两份一样的向量，检索时用户看到两条重复结果。
    content_sha256: Mapped[str] = mapped_column(
        String(64), default="", nullable=False, index=True
    )

    __table_args__ = (
        Index("idx_doc_kb_status", "kb_id", "status"),
        Index("idx_doc_kb_hash", "kb_id", "content_sha256"),
    )


class Chunk(Base, TimestampMixin):
    __tablename__ = "psy_chunks"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    doc_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("psy_documents.id", ondelete="CASCADE"), nullable=False
    )
    kb_id: Mapped[str] = mapped_column(String(32), nullable=False)
    parent_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    seq: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    char_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    page_start: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    page_end: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    meta: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    __table_args__ = (
        Index("idx_chunk_kb_doc", "kb_id", "doc_id"),
        Index("idx_chunk_parent", "parent_id"),
    )


class Session(Base, TimestampMixin):
    __tablename__ = "psy_sessions"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    user_id: Mapped[int] = mapped_column(
        BIGINT_FK, ForeignKey("psy_users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    title: Mapped[str] = mapped_column(String(60), default="新对话", nullable=False)
    kb_ids: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    model: Mapped[str] = mapped_column(String(64), default="deepseek-flash", nullable=False)
    params: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)


class Message(Base):
    __tablename__ = "psy_messages"

    id: Mapped[int] = mapped_column(BIGINT_PK, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("psy_sessions.id", ondelete="CASCADE"), nullable=False
    )
    role: Mapped[str] = mapped_column(Enum("user", "assistant", name="msg_role"), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    citations: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list, nullable=False)
    tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )

    __table_args__ = (Index("idx_msg_session", "session_id", "id"),)


class QueryLog(Base):
    """全链路 trace：复盘与指标下钻的唯一来源。"""

    __tablename__ = "psy_query_logs"

    id: Mapped[int] = mapped_column(BIGINT_PK, primary_key=True, autoincrement=True)
    trace_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    user_id: Mapped[int] = mapped_column(BIGINT_FK, nullable=False, index=True)
    session_id: Mapped[str] = mapped_column(String(32), default="", nullable=False)
    query_raw: Mapped[str] = mapped_column(Text, nullable=False)
    intent: Mapped[str] = mapped_column(String(24), default="", nullable=False)
    strategy: Mapped[str] = mapped_column(String(16), default="", nullable=False)
    rewritten_queries: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    recall_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    rerank_json: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list, nullable=False)
    context_ids: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    answer: Mapped[str] = mapped_column(Text, default="", nullable=False)
    citations: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list, nullable=False)
    model: Mapped[str] = mapped_column(String(64), default="", nullable=False)
    tokens_in: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    tokens_out: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    verify_pass_rate: Mapped[float] = mapped_column(default=0.0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
