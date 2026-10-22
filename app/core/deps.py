"""FastAPI 依赖：当前用户、trace_id。"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Header
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.errors import AppError, ErrorCode
from app.core.security import user_id_from_token
from app.db.base import get_session
from app.db.models import User

SessionDep = Annotated[AsyncSession, Depends(get_session)]


async def get_current_user(
    session: SessionDep,
    authorization: Annotated[str | None, Header()] = None,
) -> User:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise AppError(ErrorCode.UNAUTHORIZED)

    token = authorization.split(" ", 1)[1].strip()
    user_id = user_id_from_token(token)

    user = await session.scalar(select(User).where(User.id == user_id))
    if user is None or user.status != "active":
        raise AppError(ErrorCode.UNAUTHORIZED)
    return user


CurrentUser = Annotated[User, Depends(get_current_user)]
