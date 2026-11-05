"""文本块接口：分页查看 / 编辑 / 禁用 / 删除 / 重建向量。

关于编辑后重建向量：
编辑接口只改 MySQL 里的正文，并在 meta 里打 `needs_reindex=true`；
重建走显式的 `POST /documents/{doc_id}/reindex`，**不是**在编辑时同步做。

不在编辑接口里同步重建的理由：
1. 编一次要调一次 embedding API + 写 Milvus，几百毫秒级，连续改几处就变成串行等待
2. 向量库写失败时 MySQL 已经改了，两边不一致，需要补偿逻辑
3. 用户可能连续改几次，显式重建能把多次编辑合并成一次重算

重建接口的顺序是「先写向量、成功后才清标记」——反过来一旦向量写失败，
标记就没了，这个块永远检索不到新内容，而且没人会发现。
"""

from __future__ import annotations

from datetime import datetime, timezone

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
        select(Chunk).where(Chunk.doc_id == doc_id, Chunk.enabled.is_(True))
    )
    # meta 是 JSON 列，查询侧跨方言写法不统一，直接在 Python 侧数。
    # 用一次 select 拿到实体，不要再按 id 逐个 get——那是 N+1。
    pending = sum(1 for chunk in rows if (chunk.meta or {}).get("needs_reindex"))
    return {"doc_id": doc_id, "pending": pending}


@router.post("/documents/{doc_id}/reindex", summary="重建待重建文本块的向量")
async def reindex_document(doc_id: str, user: CurrentUser, session: SessionDep) -> dict:
    """把标了 needs_reindex 的块重新编码并写回向量库。

    只处理子块：MySQL 的 psy_chunks 只存子块，父块只在 Milvus 里做回填用，
    父块正文不从子块推导，所以子块改了不影响父块向量。

    已知行为：upsert 之后 Milvus 有约 1 秒的可见延迟（growing segment）。
    实测单块：接口 0.82s、新正文 1.35s 可见（2026-11-05）。不调 flush——
    整集合 flush 要 30 秒，为 1 秒延迟付这个代价不值。前端提示里写明了这点。

    先写向量、成功后才清标记。写失败就把标记原样留着，让用户能重试——
    反过来（先清标记）等于把"未生效"变成"永久未生效且无人知晓"。
    """
    await _get_owned_doc(doc_id, user.id, session)

    rows = list(
        await session.scalars(
            select(Chunk).where(Chunk.doc_id == doc_id, Chunk.enabled.is_(True))
        )
    )
    pending = [chunk for chunk in rows if (chunk.meta or {}).get("needs_reindex")]
    if not pending:
        return {"doc_id": doc_id, "reindexed": 0, "remaining": 0}

    from app.clients.embed_client import build_embedder
    from app.clients.milvus_client import MilvusIndex

    embedder = build_embedder()
    index = MilvusIndex()
    try:
        vectors = await embedder.encode([chunk.content for chunk in pending])
        records = []
        for chunk, vector in zip(pending, vectors, strict=True):
            record = {
                "chunk_id": chunk.id,
                "kb_id": chunk.kb_id,
                "doc_id": chunk.doc_id,
                "parent_id": chunk.parent_id or "",
                "content": chunk.content[:8000],
                "dense_vector": vector.dense,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "domain": "",
                "evidence_grade": "",
                "is_parent": False,
            }
            # 稀疏向量只在 provider=local 时有；dense-only 集合塞空 dict 会被拒
            if vector.sparse:
                record["sparse_vector"] = {int(k): v for k, v in vector.sparse.items()}
            records.append(record)
        await index.upsert(records)
    except Exception as exc:  # noqa: BLE001 - 任何失败都不许清标记
        logger.error(
            "chunk.reindex_failed",
            extra={"extra_fields": {"doc_id": doc_id, "count": len(pending), "error": repr(exc)}},
        )
        raise AppError(
            ErrorCode.INTERNAL, f"向量重建失败（{len(pending)} 个块仍待重建）：{exc}"
        ) from exc

    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for chunk in pending:
        meta = dict(chunk.meta or {})
        meta.pop("needs_reindex", None)
        meta["reindexed_at"] = stamp
        # JSON 列要整体重新赋值才会被 ORM 识别为变更
        chunk.meta = meta
    await session.commit()

    logger.info(
        "chunk.reindexed",
        extra={"extra_fields": {"doc_id": doc_id, "count": len(pending)}},
    )
    return {"doc_id": doc_id, "reindexed": len(pending), "remaining": 0}
