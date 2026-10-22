"""密码与 JWT 单元测试。

覆盖：bcrypt 哈希与校验、超长密码、token 往返、过期、篡改、错误密钥。
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import jwt
import pytest

from app.config import settings
from app.core.errors import AppError, ErrorCode
from app.core.security import (
    ALGORITHM,
    create_access_token,
    decode_token,
    hash_password,
    user_id_from_token,
    verify_password,
)


class TestPassword:
    def test_hash_is_not_plaintext(self) -> None:
        hashed = hash_password("abc12345")
        assert hashed != "abc12345"
        assert hashed.startswith("$2")

    def test_verify_correct_password(self) -> None:
        hashed = hash_password("abc12345")
        assert verify_password("abc12345", hashed) is True

    def test_verify_wrong_password(self) -> None:
        hashed = hash_password("abc12345")
        assert verify_password("abc123456", hashed) is False

    def test_same_password_yields_different_hash(self) -> None:
        """加盐后同一密码两次哈希应不同。"""
        assert hash_password("abc12345") != hash_password("abc12345")

    def test_long_password_is_truncated_not_crashing(self) -> None:
        """bcrypt 上限 72 字节，超长密码不能抛异常。"""
        long_password = "a1" * 100
        hashed = hash_password(long_password)
        assert verify_password(long_password, hashed) is True

    def test_empty_values_return_false(self) -> None:
        assert verify_password("", "x") is False
        assert verify_password("x", "") is False

    def test_empty_password_hash_raises(self) -> None:
        with pytest.raises(ValueError):
            hash_password("")

    def test_corrupted_hash_returns_false(self) -> None:
        assert verify_password("abc12345", "not-a-bcrypt-hash") is False


class TestJWT:
    def test_roundtrip(self) -> None:
        token, expires_in = create_access_token(42, "student01")
        payload = decode_token(token)

        assert payload["sub"] == "42"
        assert payload["username"] == "student01"
        assert payload["type"] == "access"
        assert expires_in == settings.security.jwt_expire_hours * 3600

    def test_user_id_extraction(self) -> None:
        token, _ = create_access_token(7, "u7")
        assert user_id_from_token(token) == 7

    def test_expired_token_raises_expired(self) -> None:
        expired = jwt.encode(
            {
                "sub": "1",
                "username": "u1",
                "type": "access",
                "exp": int((datetime.now(UTC) - timedelta(seconds=10)).timestamp()),
            },
            settings.security.jwt_secret,
            algorithm=ALGORITHM,
        )
        with pytest.raises(AppError) as exc:
            decode_token(expired)
        assert exc.value.code == ErrorCode.TOKEN_EXPIRED

    def test_tampered_token_raises_invalid(self) -> None:
        token, _ = create_access_token(1, "u1")
        tampered = token[:-3] + ("aaa" if not token.endswith("aaa") else "bbb")

        with pytest.raises(AppError) as exc:
            decode_token(tampered)
        assert exc.value.code == ErrorCode.TOKEN_INVALID

    def test_wrong_secret_raises_invalid(self) -> None:
        foreign = jwt.encode({"sub": "1", "type": "access"}, "another-secret", algorithm=ALGORITHM)
        with pytest.raises(AppError) as exc:
            decode_token(foreign)
        assert exc.value.code == ErrorCode.TOKEN_INVALID

    def test_non_access_token_rejected(self) -> None:
        refresh = jwt.encode(
            {"sub": "1", "username": "u1", "type": "refresh"},
            settings.security.jwt_secret,
            algorithm=ALGORITHM,
        )
        with pytest.raises(AppError) as exc:
            user_id_from_token(refresh)
        assert exc.value.code == ErrorCode.TOKEN_INVALID
