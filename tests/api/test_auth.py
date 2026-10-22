"""鉴权接口测试。

用内存 SQLite 跑（见 conftest.sqlite_client），不依赖 MySQL，因此每次都真跑。
真实 MySQL 的连通性验证放在 test_integration_mysql.py。

设计说明：
- 验证码图片无法回读，测试直接向 captcha 表插入已知答案的记录，
  这样既覆盖了校验逻辑，又不依赖 OCR。
- 每个用例用 uuid 后缀生成唯一账号，避免互相污染。
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta

import pytest
from httpx import AsyncClient
from sqlalchemy import select

from app.core.captcha import hash_code
from app.db.models import Captcha, User

VALID_CODE = "AB34"


@pytest.fixture
def client(sqlite_client: AsyncClient) -> AsyncClient:
    return sqlite_client


async def make_captcha(client: AsyncClient, scene: str = "register", *, code: str = VALID_CODE, expired: bool = False) -> str:
    captcha_id = str(uuid.uuid4())
    expires_at = datetime.now() + (timedelta(seconds=-10) if expired else timedelta(minutes=5))
    async with client.session_factory() as session:
        session.add(
            Captcha(
                id=captcha_id,
                code_hash=hash_code(code),
                scene=scene,
                expires_at=expires_at,
            )
        )
        await session.commit()
    return captcha_id


def unique_username(prefix: str = "stu") -> str:
    return f"{prefix}{uuid.uuid4().hex[:8]}"


async def register_user(client: AsyncClient, username: str, password: str = "abc12345"):
    captcha_id = await make_captcha(client, "register")
    return await client.post(
        "/api/auth/register",
        json={
            "username": username,
            "password": password,
            "password_confirm": password,
            "captcha_id": captcha_id,
            "captcha_code": VALID_CODE,
        },
    )


class TestCaptcha:
    async def test_get_captcha_returns_image(self, client: AsyncClient) -> None:
        response = await client.get("/api/auth/captcha", params={"scene": "login"})
        body = response.json()

        assert response.status_code == 200
        assert body["image_base64"].startswith("data:image/png;base64,")
        assert body["expires_in"] == 300
        assert body["captcha_id"]


class TestRegister:
    async def test_register_success(self, client: AsyncClient) -> None:
        username = unique_username()
        response = await register_user(client, username)

        assert response.status_code == 200
        assert response.json()["message"].startswith("注册成功")

        async with client.session_factory() as session:
            user = await session.scalar(select(User).where(User.username == username))
            assert user is not None
            assert user.password_hash.startswith("$2")
            assert user.password_hash != "abc12345"

    async def test_register_password_mismatch(self, client: AsyncClient) -> None:
        captcha_id = await make_captcha(client, "register")
        response = await client.post(
            "/api/auth/register",
            json={
                "username": unique_username(),
                "password": "abc12345",
                "password_confirm": "abc12346",
                "captcha_id": captcha_id,
                "captcha_code": VALID_CODE,
            },
        )

        assert response.status_code == 400
        assert response.json()["code"] == "VALIDATION_ERROR"

    async def test_register_duplicate_username(self, client: AsyncClient) -> None:
        username = unique_username()
        assert (await register_user(client, username)).status_code == 200

        second = await register_user(client, username)
        assert second.status_code == 409
        assert second.json()["code"] == "USERNAME_EXISTS"

    async def test_register_wrong_captcha(self, client: AsyncClient) -> None:
        captcha_id = await make_captcha(client, "register")
        response = await client.post(
            "/api/auth/register",
            json={
                "username": unique_username(),
                "password": "abc12345",
                "password_confirm": "abc12345",
                "captcha_id": captcha_id,
                "captcha_code": "ZZ99",
            },
        )

        assert response.status_code == 400
        assert response.json()["code"] == "CAPTCHA_INVALID"

    async def test_register_expired_captcha(self, client: AsyncClient) -> None:
        captcha_id = await make_captcha(client, "register", expired=True)
        response = await client.post(
            "/api/auth/register",
            json={
                "username": unique_username(),
                "password": "abc12345",
                "password_confirm": "abc12345",
                "captcha_id": captcha_id,
                "captcha_code": VALID_CODE,
            },
        )

        assert response.status_code == 400
        assert response.json()["code"] == "CAPTCHA_EXPIRED"

    async def test_register_captcha_is_single_use(self, client: AsyncClient) -> None:
        captcha_id = await make_captcha(client, "register")
        payload = {
            "username": unique_username(),
            "password": "abc12345",
            "password_confirm": "abc12345",
            "captcha_id": captcha_id,
            "captcha_code": VALID_CODE,
        }
        first = await client.post("/api/auth/register", json=payload)
        assert first.status_code == 200

        payload["username"] = unique_username()
        second = await client.post("/api/auth/register", json=payload)
        assert second.status_code == 400
        assert second.json()["code"] == "CAPTCHA_INVALID"

    async def test_register_weak_password_rejected(self, client: AsyncClient) -> None:
        captcha_id = await make_captcha(client, "register")
        response = await client.post(
            "/api/auth/register",
            json={
                "username": unique_username(),
                "password": "12345678",
                "password_confirm": "12345678",
                "captcha_id": captcha_id,
                "captcha_code": VALID_CODE,
            },
        )

        assert response.status_code == 400
        assert response.json()["code"] == "VALIDATION_ERROR"


class TestLogin:
    async def test_login_success(self, client: AsyncClient) -> None:
        username = unique_username()
        await register_user(client, username)

        captcha_id = await make_captcha(client, "login")
        response = await client.post(
            "/api/auth/login",
            json={
                "username": username,
                "password": "abc12345",
                "captcha_id": captcha_id,
                "captcha_code": VALID_CODE,
            },
        )
        body = response.json()

        assert response.status_code == 200
        assert body["token_type"] == "bearer"
        assert body["access_token"]
        assert body["user"]["username"] == username

    async def test_login_wrong_password(self, client: AsyncClient) -> None:
        username = unique_username()
        await register_user(client, username)

        captcha_id = await make_captcha(client, "login")
        response = await client.post(
            "/api/auth/login",
            json={
                "username": username,
                "password": "wrong1234",
                "captcha_id": captcha_id,
                "captcha_code": VALID_CODE,
            },
        )

        assert response.status_code == 401
        assert response.json()["code"] == "CREDENTIALS_INVALID"

    async def test_login_unknown_user_same_error(self, client: AsyncClient) -> None:
        """账号不存在与密码错误必须返回同一错误码，避免账号枚举。"""
        captcha_id = await make_captcha(client, "login")
        response = await client.post(
            "/api/auth/login",
            json={
                "username": "nobody" + uuid.uuid4().hex[:6],
                "password": "abc12345",
                "captcha_id": captcha_id,
                "captcha_code": VALID_CODE,
            },
        )

        assert response.status_code == 401
        assert response.json()["code"] == "CREDENTIALS_INVALID"

    async def test_login_updates_last_login(self, client: AsyncClient) -> None:
        username = unique_username()
        await register_user(client, username)

        captcha_id = await make_captcha(client, "login")
        await client.post(
            "/api/auth/login",
            json={
                "username": username,
                "password": "abc12345",
                "captcha_id": captcha_id,
                "captcha_code": VALID_CODE,
            },
        )

        async with client.session_factory() as session:
            user = await session.scalar(select(User).where(User.username == username))
            assert user is not None and user.last_login_at is not None


class TestMe:
    async def test_me_without_token(self, client: AsyncClient) -> None:
        response = await client.get("/api/auth/me")
        assert response.status_code == 401
        assert response.json()["code"] == "UNAUTHORIZED"

    async def test_me_with_token(self, client: AsyncClient) -> None:
        username = unique_username()
        await register_user(client, username)
        captcha_id = await make_captcha(client, "login")
        login = await client.post(
            "/api/auth/login",
            json={
                "username": username,
                "password": "abc12345",
                "captcha_id": captcha_id,
                "captcha_code": VALID_CODE,
            },
        )
        token = login.json()["access_token"]

        response = await client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert response.status_code == 200
        assert response.json()["username"] == username

    async def test_me_with_bad_token(self, client: AsyncClient) -> None:
        response = await client.get("/api/auth/me", headers={"Authorization": "Bearer nonsense"})
        assert response.status_code == 401
        assert response.json()["code"] == "TOKEN_INVALID"


class TestHealth:
    async def test_health_reports_db_state(self, client: AsyncClient) -> None:
        response = await client.get("/api/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
