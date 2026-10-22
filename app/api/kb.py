"""知识库接口：增删改查。

权限规则（v3 9.4 节）：
- 所有操作都按 owner_id 过滤
- 越权访问返回 404 而不是 403——403 等于告诉对方"这个 ID 存在但你没权限"，
  可以被用来枚举他人的知识库 ID
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter
from sqlalchemy import delete, func, select

from app.core.deps import CurrentUser, SessionDep
from app.core.errors import AppError, ErrorCode
from app.core.logging import get_logger
from app.db.models import Chunk, Document, KnowledgeBase
from app.schemas.kb import KBCreate, KBOut, KBUpdate, MessageOut

router = APIRouter(prefix="/api/kb", tags=["知识库"])
logger = get_logger(__name__)


async def get_owned_kb(kb_id: str, user_id: int, session: SessionDep) -> KnowledgeBase:
    """取知识库并校验归属。不存在与无权限一律 404。"""
    kb = await session.scalar(select(KnowledgeBase).where(KnowledgeBase.id == kb_id))
    if kb is None or kb.owner_id != user_id:
        raise AppError(ErrorCode.KB_NOT_FOUND)
    return kb


@router.get("", response_model=list[KBOut], summary="我的知识库列表")
async def list_kbs(user: CurrentUser, session: SessionDep) -> list[KBOut]:
    rows = await session.scalars(
        select(KnowledgeBase)
        .where(KnowledgeBase.owner_id == user.id)
        .order_by(KnowledgeBase.updated_at.desc())
    )
    return [KBOut.model_validate(row) for row in rows]


@router.post("", response_model=KBOut, summary="创建知识库")
async def create_kb(payload: KBCreate, user: CurrentUser, session: SessionDep) -> KBOut:
    kb = KnowledgeBase(
        id=uuid.uuid4().hex[:16],
        owner_id=user.id,
        name=payload.name,
        description=payload.description,
    )
    session.add(kb)
    await session.commit()
    await session.refresh(kb)

    logger.info(
        "kb.create",
        extra={"extra_fields": {"kb_id": kb.id, "user_id": user.id}},
    )
    return KBOut.model_validate(kb)


@router.get("/{kb_id}", response_model=KBOut, summary="知识库详情")
async def get_kb(kb_id: str, user: CurrentUser, session: SessionDep) -> KBOut:
    kb = await get_owned_kb(kb_id, user.id, session)
    return KBOut.model_validate(kb)


@router.patch("/{kb_id}", response_model=KBOut, summary="修改知识库")
async def update_kb(
    kb_id: str, payload: KBUpdate, user: CurrentUser, session: SessionDep
) -> KBOut:
    kb = await get_owned_kb(kb_id, user.id, session)

    if payload.name is not None:
        kb.name = payload.name.strip()
    if payload.description is not None:
        kb.description = payload.description

    await session.commit()
    await session.refresh(kb)
    logger.info("kb.update", extra={"extra_fields": {"kb_id": kb.id}})
    return KBOut.model_validate(kb)


@router.delete("/{kb_id}", response_model=MessageOut, summary="删除知识库")
async def delete_kb(kb_id: str, user: CurrentUser, session: SessionDep) -> MessageOut:
    await get_owned_kb(kb_id, user.id, session)

    # 先清数据库侧的文档与文本块；向量库由上层调用 milvus.delete_by_kb
    await session.execute(delete(Chunk).where(Chunk.kb_id == kb_id))
    await session.execute(delete(Document).where(Document.kb_id == kb_id))
    await session.execute(delete(KnowledgeBase).where(KnowledgeBase.id == kb_id))
    await session.commit()

    logger.info(
        "kb.delete",
        extra={"extra_fields": {"kb_id": kb_id, "user_id": user.id}},
    )
    return MessageOut(message="知识库已删除")


@router.get("/{kb_id}/stats", summary="知识库统计")
async def kb_stats(kb_id: str, user: CurrentUser, session: SessionDep) -> dict:
    await get_owned_kb(kb_id, user.id, session)

    doc_total = await session.scalar(
        select(func.count(Document.id)).where(Document.kb_id == kb_id)
    )
    chunk_total = await session.scalar(
        select(func.count(Chunk.id)).where(Chunk.kb_id == kb_id)
    )
    enabled_total = await session.scalar(
        select(func.count(Chunk.id)).where(Chunk.kb_id == kb_id, Chunk.enabled.is_(True))
    )
    failed = await session.scalar(
        select(func.count(Document.id)).where(
            Document.kb_id == kb_id, Document.status == "failed"
        )
    )

    return {
        "kb_id": kb_id,
        "documents": int(doc_total or 0),
        "chunks": int(chunk_total or 0),
        "enabled_chunks": int(enabled_total or 0),
        "failed_documents": int(failed or 0),
    }
