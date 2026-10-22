"""文本块接口：分页查看 / 编辑 / 禁用 / 删除。

关于编辑后重建向量：
本接口只改 MySQL 里的正文，并在 meta 里打 `needs_reindex=true`，
真正的向量重建交给后台任务（M2 接 embedding 后实现）。

不做「编辑接口里同步重建向量」的理由：
1. 重建要加载 bge-m3 并调 Milvus，一次几百毫秒到几秒，接口会超时
2. 向量库写失败时，MySQL 已经改了，两边不一致，需要补偿逻辑
3. 后台重建可以合并同一文档的多次编辑，避免反复重算

所以这里的语义是：**改内容立即生效（检索仍用旧向量），重建后台异步跟进**。
前端在 chunk 卡片上显示"待重新索引"状态，别假装已经好了。
"""

from __future__ import annotations

from fastapi import APIRouter, Query
from sqlalchemy import func, select

from app.api.documents import _get_owned_doc
from app.core.deps import CurrentUser, SessionDep
from app.core.errors import AppError, ErrorCode
from app.core.logging import get_logger
from app.db.models import Chunk
from app.schemas.kb import ChunkOut, ChunkPage, ChunkUpdate, MessageOut

router = APIRouter(prefix="/api", tags=["文本块"])
logger = get_logger(__name__)


async def _get_owned_chunk(chunk_id: str, user_id: int, session: SessionDep) -> Chunk:
    chunk = await session.scalar(select(Chunk).where(Chunk.id == chunk_id))
    if chunk is None:
        raise AppError(ErrorCode.NOT_FOUND, "文本块不存在")
    # 沿 doc → kb → owner 校验
    await _get_owned_doc(chunk.doc_id, user_id, session)
    return chunk


@router.get("/documents/{doc_id}/chunks", response_model=ChunkPage, summary="文本块分页列表")
async def list_chunks(
    doc_id: str,
    user: CurrentUser,
    session: SessionDep,
    page: int = Query(default=1, ge=1),
    size: int = Query(default=20, ge=1, le=100),
    keyword: str | None = Query(default=None, max_length=50),
    only_enabled: bool = Query(default=False),
) -> ChunkPage:
    await _get_owned_doc(doc_id, user.id, session)

    conditions = [Chunk.doc_id == doc_id]
    if keyword:
        conditions.append(Chunk.content.like(f"%{keyword}%"))
    if only_enabled:
        conditions.append(Chunk.enabled.is_(True))

    total = int(await session.scalar(select(func.count(Chunk.id)).where(*conditions)) or 0)

    rows = await session.scalars(
        select(Chunk)
        .where(*conditions)
        .order_by(Chunk.seq.asc())
        .offset((page - 1) * size)
        .limit(size)
    )

    return ChunkPage(
        items=[ChunkOut.model_validate(row) for row in rows],
        total=total,
        page=page,
        size=size,
    )


@router.get("/chunks/{chunk_id}", response_model=ChunkOut, summary="文本块详情")
async def get_chunk(chunk_id: str, user: CurrentUser, session: SessionDep) -> ChunkOut:
    chunk = await _get_owned_chunk(chunk_id, user.id, session)
    return ChunkOut.model_validate(chunk)


@router.patch("/chunks/{chunk_id}", response_model=ChunkOut, summary="编辑文本块")
async def update_chunk(
    chunk_id: str, payload: ChunkUpdate, user: CurrentUser, session: SessionDep
) -> ChunkOut:
    chunk = await _get_owned_chunk(chunk_id, user.id, session)

    if payload.content is not None:
        chunk.content = payload.content
        chunk.char_count = len(payload.content)
        # 标记待重建向量，后台任务会扫这个标记
        meta = dict(chunk.meta or {})
        meta["needs_reindex"] = True
        chunk.meta = meta

    if payload.enabled is not None:
        chunk.enabled = payload.enabled

    await session.commit()
    await session.refresh(chunk)

    logger.info(
        "chunk.update",
        extra={
            "extra_fields": {
                "chunk_id": chunk_id,
                "content_changed": payload.content is not None,
                "enabled": payload.enabled,
            }
        },
    )
    return ChunkOut.model_validate(chunk)


@router.delete("/chunks/{chunk_id}", response_model=MessageOut, summary="删除文本块")
async def delete_chunk(chunk_id: str, user: CurrentUser, session: SessionDep) -> MessageOut:
    chunk = await _get_owned_chunk(chunk_id, user.id, session)
    await session.delete(chunk)
    await session.commit()

    logger.info("chunk.delete", extra={"extra_fields": {"chunk_id": chunk_id}})
    return MessageOut(message="文本块已删除")


@router.get("/documents/{doc_id}/pending-reindex", summary="待重建向量的文本块数")
async def pending_reindex(doc_id: str, user: CurrentUser, session: SessionDep) -> dict:
    await _get_owned_doc(doc_id, user.id, session)
    rows = await session.scalars(
        select(Chunk.id).where(Chunk.doc_id == doc_id, Chunk.enabled.is_(True))
    )
    # 简单起见：meta 里的标记在 Python 侧检查（JSON 查询在不同数据库方言下写法不同）
    pending = 0
    for chunk_id in rows:
        chunk = await session.get(Chunk, chunk_id)
        if chunk and (chunk.meta or {}).get("needs_reindex"):
            pending += 1
    return {"doc_id": doc_id, "pending": pending}
