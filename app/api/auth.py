"""鉴权接口：验证码 / 注册 / 登录 / 当前用户。

约定：
- 验证码一次有效，用过即作废。
- 登录失败统一返回 CREDENTIALS_INVALID，不区分账号不存在与密码错误。
- 注册成功不自动登录，跳登录页（任务描述：暂不支持改密码，此处也不发 token）。
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta

from fastapi import APIRouter
from sqlalchemy import select

from app.config import settings
from app.core import captcha as captcha_util
from app.core.deps import CurrentUser, SessionDep
from app.core.errors import AppError, ErrorCode
from app.core.logging import get_logger
from app.core.security import create_access_token, hash_password, verify_password
from app.db.models import Captcha, User
from app.schemas.auth import (
    CaptchaOut,
    LoginRequest,
    MessageOut,
    RegisterRequest,
    TokenOut,
    UserOut,
)

router = APIRouter(prefix="/api/auth", tags=["鉴权"])
logger = get_logger(__name__)


async def _issue_captcha(session: SessionDep, scene: str) -> CaptchaOut:
    code = captcha_util.generate_code()
    captcha_id = str(uuid.uuid4())
    ttl = settings.security.captcha_expire_seconds

    session.add(
        Captcha(
            id=captcha_id,
            code_hash=captcha_util.hash_code(code),
            scene=scene,
            expires_at=datetime.now() + timedelta(seconds=ttl),
        )
    )
    await session.commit()

    return CaptchaOut(
        captcha_id=captcha_id,
        image_base64=captcha_util.render_base64(code),
        expires_in=ttl,
    )


async def _consume_captcha(
    session: SessionDep, captcha_id: str, code: str, scene: str
) -> None:
    """校验并作废验证码。顺序：存在 → 过期 → 已用 → 匹配。"""
    record = await session.scalar(select(Captcha).where(Captcha.id == captcha_id))
    if record is None:
        raise AppError(ErrorCode.CAPTCHA_INVALID, "验证码无效，请刷新后重试")

    if record.scene != scene:
        raise AppError(ErrorCode.CAPTCHA_INVALID, "验证码与场景不匹配")

    if record.expires_at < datetime.now():
        raise AppError(ErrorCode.CAPTCHA_EXPIRED)

    if record.used:
        raise AppError(ErrorCode.CAPTCHA_INVALID, "验证码已使用，请刷新后重试")

    if not captcha_util.verify_code(code, record.code_hash):
        raise AppError(ErrorCode.CAPTCHA_INVALID)

    record.used = True
    await session.commit()


@router.get("/captcha", response_model=CaptchaOut, summary="获取图形验证码")
async def get_captcha(session: SessionDep, scene: str = "login") -> CaptchaOut:
    return await _issue_captcha(session, scene)


@router.post("/register", response_model=MessageOut, summary="注册")
async def register(payload: RegisterRequest, session: SessionDep) -> MessageOut:
    await _consume_captcha(session, payload.captcha_id, payload.captcha_code, "register")

    if not payload.passwords_match():
        raise AppError(ErrorCode.VALIDATION, "两次输入的密码不一致")

    exists = await session.scalar(select(User.id).where(User.username == payload.username))
    if exists is not None:
        raise AppError(ErrorCode.USERNAME_EXISTS)

    user = User(username=payload.username, password_hash=hash_password(payload.password))
    session.add(user)
    await session.commit()

    logger.info("auth.register", extra={"extra_fields": {"username": payload.username}})
    return MessageOut(message="注册成功，请登录")


@router.post("/login", response_model=TokenOut, summary="登录")
async def login(payload: LoginRequest, session: SessionDep) -> TokenOut:
    await _consume_captcha(session, payload.captcha_id, payload.captcha_code, "login")

    user = await session.scalar(select(User).where(User.username == payload.username))
    if user is None or not verify_password(payload.password, user.password_hash):
        # 不区分「账号不存在」与「密码错误」，避免账号枚举
        logger.warning(
            "auth.login.failed",
            extra={"extra_fields": {"username": payload.username}},
        )
        raise AppError(ErrorCode.CREDENTIALS_INVALID)

    if user.status != "active":
        raise AppError(ErrorCode.UNAUTHORIZED, "账号已被禁用，请联系管理员")

    user.last_login_at = datetime.now()
    await session.commit()

    token, expires_in = create_access_token(user.id, user.username)
    logger.info(
        "auth.login.success",
        extra={"extra_fields": {"user_id": user.id, "username": user.username}},
    )
    return TokenOut(
        access_token=token,
        expires_in=expires_in,
        user=UserOut.model_validate(user),
    )


@router.get("/me", response_model=UserOut, summary="当前登录用户")
async def me(user: CurrentUser) -> UserOut:
    return UserOut.model_validate(user)
