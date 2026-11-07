"""收藏接口：把问答里值得留下的答案存成自己的知识卡片。

三个设计决定：

1. **存全文快照，不存 message_id 外键。**
   会话被删掉后收藏不该跟着消失——它已经是用户自己的东西。
   代价是同一段文字在库里多存一份，但这个量（每条几百字）换来的
   是「收藏夹不会因为清理会话而变空」，值。

2. **收藏是幂等的。**
   同一个人对同一个问题同一条答案重复点收藏，返回已有记录而不是
   堆两条。前端拿返回的 id 直接标成「已收藏」，不需要额外查询。

3. **引用正文截断到 400 字再存。**
   引用块原文可能上千字，20 条引用存全量会让单行 JSON 到几十 KB。
   收藏夹要的是「能回看要点和出处」，回查原文有 chunk_id 就够了。
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Query
from sqlalchemy import func, or_, select

from app.core.deps import CurrentUser, SessionDep
from app.core.errors import AppError, ErrorCode
from app.core.logging import get_logger
from app.db.models import Favorite
from app.schemas.favorite import (
    CitationIn,
    FavoriteCreate,
    FavoriteOut,
    FavoritePage,
    FavoriteUpdate,
)

router = APIRouter(prefix="/api/favorites", tags=["收藏"])
logger = get_logger(__name__)

CITATION_PREVIEW_CHARS = 400
DEDUPE_PREFIX = 200


def _to_out(row: Favorite) -> FavoriteOut:
    return FavoriteOut(
        id=row.id,
        question=row.question,
        answer=row.answer,
        note=row.note,
        session_id=row.session_id,
        source=row.source,
        citations=[CitationIn(**item) for item in (row.citations or [])],
        created_at=row.created_at,
    )


def _trim_citations(items: list[CitationIn]) -> list[dict]:
    return [
        {
            "sid": item.sid,
            "doc_title": item.doc_title,
            "page": item.page,
            "page_end": item.page_end,
            "content": (item.content or "")[:CITATION_PREVIEW_CHARS],
        }
        for item in items
    ]


@router.post("", response_model=FavoriteOut, summary="收藏一条回答")
async def create_favorite(
    payload: FavoriteCreate, user: CurrentUser, session: SessionDep
) -> FavoriteOut:
    citations = _trim_citations(payload.citations)
    source = citations[0]["doc_title"] if citations else ""

    # 幂等：同一人 + 同一问题 + 同一答案前缀，视为同一条
    existing = await session.scalar(
        select(Favorite).where(
            Favorite.owner_id == user.id,
            Favorite.question == payload.question,
            Favorite.answer.like(payload.answer[:DEDUPE_PREFIX] + "%"),
        )
    )
    if existing is not None:
        return _to_out(existing)

    row = Favorite(
        id=uuid.uuid4().hex,
        owner_id=user.id,
        session_id=payload.session_id,
        question=payload.question,
        answer=payload.answer,
        citations=citations,
        note=payload.note,
        source=source[:120],
    )
    session.add(row)
    await session.commit()
    await session.refresh(row)
    logger.info(
        "favorite.created",
        extra={"extra_fields": {"user_id": user.id, "favorite_id": row.id}},
    )
    return _to_out(row)


@router.get("", response_model=FavoritePage, summary="我的收藏")
async def list_favorites(
    user: CurrentUser,
    session: SessionDep,
    q: str = Query(default="", max_length=100),
    page: int = Query(default=1, ge=1),
    size: int = Query(default=10, ge=1, le=50),
) -> FavoritePage:
    conditions = [Favorite.owner_id == user.id]
    if q.strip():
        like = f"%{q.strip()}%"
        conditions.append(
            or_(
                Favorite.question.like(like),
                Favorite.answer.like(like),
                Favorite.note.like(like),
                Favorite.source.like(like),
            )
        )

    total = int(await session.scalar(select(func.count()).select_from(Favorite).where(*conditions)) or 0)
    rows = (
        await session.scalars(
            select(Favorite)
            .where(*conditions)
            .order_by(Favorite.created_at.desc(), Favorite.id.desc())
            .offset((page - 1) * size)
            .limit(size)
        )
    ).all()

    return FavoritePage(
        total=total, page=page, size=size, items=[_to_out(row) for row in rows]
    )


async def _get_owned(favorite_id: str, user_id: int, session: SessionDep) -> Favorite:
    row = await session.get(Favorite, favorite_id)
    # 越权一律 404：403 等于告诉对方「这个 id 存在」
    if row is None or row.owner_id != user_id:
        raise AppError(ErrorCode.NOT_FOUND, "这条收藏不存在或已删除")
    return row


@router.patch("/{favorite_id}", response_model=FavoriteOut, summary="修改收藏备注")
async def update_favorite(
    favorite_id: str, payload: FavoriteUpdate, user: CurrentUser, session: SessionDep
) -> FavoriteOut:
    row = await _get_owned(favorite_id, user.id, session)
    row.note = payload.note
    await session.commit()
    await session.refresh(row)
    return _to_out(row)


@router.delete("/{favorite_id}", summary="取消收藏")
async def delete_favorite(favorite_id: str, user: CurrentUser, session: SessionDep) -> dict[str, str]:
    row = await _get_owned(favorite_id, user.id, session)
    await session.delete(row)
    await session.commit()
    return {"id": favorite_id}
