"""密码哈希与 JWT。

约束（来自 v3 规范）：
- 密码用 bcrypt，cost 从 config.ini 读（默认 12），禁止明文 / MD5。
- JWT 用 HS256，密钥从环境变量读（config.ini 只存变量名）。
- 登录失败不区分「账号不存在」与「密码错误」，统一 CREDENTIALS_INVALID，避免账号枚举。
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import bcrypt
import jwt

from app.config import settings
from app.core.errors import AppError, ErrorCode

ALGORITHM = "HS256"


# ---------------------------------------------------------------- 密码
def hash_password(plain: str) -> str:
    """生成 bcrypt 哈希。bcrypt 上限 72 字节，先截断避免异常。"""
    if not plain:
        raise ValueError("密码不能为空")
    payload = plain.encode("utf-8")[:72]
    rounds = settings.security.bcrypt_rounds
    return bcrypt.hashpw(payload, bcrypt.gensalt(rounds=rounds)).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    """校验密码。任何异常都返回 False，不把内部错误透给调用方。"""
    if not plain or not hashed:
        return False
    try:
        return bcrypt.checkpw(plain.encode("utf-8")[:72], hashed.encode("utf-8"))
    except (ValueError, TypeError):
        return False


# ---------------------------------------------------------------- JWT
def create_access_token(user_id: int, username: str) -> tuple[str, int]:
    """返回 (token, expires_in_seconds)。"""
    expire_seconds = settings.security.jwt_expire_hours * 3600
    now = datetime.now(UTC)
    payload: dict[str, Any] = {
        "sub": str(user_id),
        "username": username,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(seconds=expire_seconds)).timestamp()),
        "type": "access",
    }
    token = jwt.encode(payload, settings.security.jwt_secret, algorithm=ALGORITHM)
    return token, expire_seconds


def decode_token(token: str) -> dict[str, Any]:
    """解析 token。过期与非法分开报错，前端才能提示「重新登录」而不是「凭证无效」。"""
    try:
        return jwt.decode(token, settings.security.jwt_secret, algorithms=[ALGORITHM])
    except jwt.ExpiredSignatureError as exc:
        raise AppError(ErrorCode.TOKEN_EXPIRED) from exc
    except jwt.InvalidTokenError as exc:
        raise AppError(ErrorCode.TOKEN_INVALID) from exc


def user_id_from_token(token: str) -> int:
    payload = decode_token(token)
    sub = payload.get("sub")
    if not payload.get("type") == "access" or not sub:
        raise AppError(ErrorCode.TOKEN_INVALID)
    try:
        return int(sub)
    except (TypeError, ValueError) as exc:
        raise AppError(ErrorCode.TOKEN_INVALID) from exc
