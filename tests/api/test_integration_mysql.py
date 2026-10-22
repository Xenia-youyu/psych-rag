"""真实 MySQL 集成测试。

与 tests/api/test_kb.py 的分工：
- test_kb.py 用内存 SQLite 跑逻辑，快、不依赖网络，每次必跑
- 本文件验证「生产数据库上确实能跑通」，默认在服务器不可达时跳过

跑法：
    python -m pytest tests/api/test_integration_mysql.py -v
    python -m pytest tests/ -m integration -v

隔离策略：使用**独立数据库** `psych_rag`，不再与同学共用 `subjects_kg`。
服务器上还有别人的库（如 agentic_rag），所以有两条用例盯着"我们只动自己的库"：
- 我们的库里不允许出现非 psy_ 前缀的表
- 我们的库名不允许是别人的业务库

注意：本文件会写入少量测试数据，并在 finally 里清理。
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta

import pymysql
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.config import settings
from app.core.captcha import hash_code
from tests.conftest import requires_db

pytestmark = [pytest.mark.integration, requires_db]

VALID_CODE = "AB34"
OUR_TABLES = {
    "psy_users",
    "psy_captcha",
    "psy_knowledge_bases",
    "psy_documents",
    "psy_chunks",
    "psy_sessions",
    "psy_messages",
    "psy_query_logs",
}
# 共享库里的邻居表，不允许被我们的初始化动到
# 我们的库必须与别人的业务库分开。这些名字是别人在用的，绝不能配进来。
FOREIGN_DATABASES = {"agentic_rag", "subjects_kg", "itcast", "mysql", "sys"}


def mysql_conn(**overrides):
    conf = settings.mysql
    params = {
        "host": conf.host,
        "port": conf.port,
        "user": conf.user,
        "password": conf.password,
        "database": conf.database,
        "charset": conf.charset,
        "connect_timeout": 8,
    }
    params.update(overrides)
    return pymysql.connect(**params)


@pytest_asyncio.fixture
async def mysql_client() -> AsyncClient:
    """用真实 MySQL 的客户端：不覆盖 get_session 依赖。"""
    from app.main import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


class TestConnectivity:
    def test_mysql_reachable(self) -> None:
        conn = mysql_conn()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT VERSION()")
                version = cur.fetchone()[0]
            assert version.startswith("8.")
        finally:
            conn.close()

    def test_target_database_is_writable(self) -> None:
        conn = mysql_conn()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT DATABASE()")
                assert cur.fetchone()[0] == settings.mysql.database
                # 写一张临时表再删，确认真的可写
                cur.execute("CREATE TEMPORARY TABLE _psy_probe (id INT)")
                cur.execute("DROP TEMPORARY TABLE _psy_probe")
        finally:
            conn.close()


class TestSchema:
    def test_our_tables_exist(self) -> None:
        conn = mysql_conn()
        try:
            with conn.cursor() as cur:
                cur.execute("SHOW TABLES")
                existing = {row[0] for row in cur.fetchall()}
        finally:
            conn.close()

        missing = OUR_TABLES - existing
        assert not missing, f"缺少数据表：{missing}，先跑 python -m scripts.init_db"

    def test_uses_dedicated_database(self) -> None:
        """必须连在自己的库上。配到别人的库里，一张 CREATE TABLE 就可能覆盖别人的数据。"""
        assert settings.mysql.database not in FOREIGN_DATABASES, (
            f"database={settings.mysql.database} 是别人的库，"
            f"请改成独立库名（如 psych_rag）"
        )

    def test_our_database_holds_only_our_tables(self) -> None:
        """我们的库里不该出现任何非 psy_ 前缀的表。

        反过来说：一旦这条挂了，说明有人把别人的表迁进来了，
        或者建表脚本漏了前缀——两种都该立刻修。
        """
        conn = mysql_conn()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT DATABASE()")
                db_name = cur.fetchone()[0]
                cur.execute("SHOW TABLES")
                tables = {row[0] for row in cur.fetchall()}
        finally:
            conn.close()

        assert db_name == settings.mysql.database, (
            f"实际连的库是 {db_name}，与配置 {settings.mysql.database} 不一致"
        )
        foreign = {t for t in tables if not t.startswith("psy_")}
        assert not foreign, f"库 {db_name} 里出现了不属于我们的表：{foreign}"

    def test_chunks_table_has_key_columns(self) -> None:
        conn = mysql_conn()
        try:
            with conn.cursor() as cur:
                cur.execute("SHOW COLUMNS FROM psy_chunks")
                columns = {row[0] for row in cur.fetchall()}
        finally:
            conn.close()

        for expected in ("id", "doc_id", "kb_id", "parent_id", "content", "enabled"):
            assert expected in columns, f"psy_chunks 缺字段 {expected}"


class TestEndToEnd:
    async def test_register_and_login_on_mysql(self, mysql_client: AsyncClient) -> None:
        username = f"its{uuid.uuid4().hex[:8]}"
        password = "abc12345"
        created_user_id: int | None = None

        try:
            # 直接写验证码记录，图片无法回读
            async def make_captcha(scene: str) -> str:
                captcha_id = str(uuid.uuid4())
                conn = mysql_conn()
                try:
                    with conn.cursor() as cur:
                        cur.execute(
                            "INSERT INTO psy_captcha (id, code_hash, scene, expires_at, used) "
                            "VALUES (%s, %s, %s, %s, 0)",
                            (captcha_id, hash_code(VALID_CODE), scene,
                             datetime.now() + timedelta(minutes=5)),
                        )
                    conn.commit()
                finally:
                    conn.close()
                return captcha_id

            reg = await mysql_client.post(
                "/api/auth/register",
                json={
                    "username": username,
                    "password": password,
                    "password_confirm": password,
                    "captcha_id": await make_captcha("register"),
                    "captcha_code": VALID_CODE,
                },
            )
            assert reg.status_code == 200, reg.text

            login = await mysql_client.post(
                "/api/auth/login",
                json={
                    "username": username,
                    "password": password,
                    "captcha_id": await make_captcha("login"),
                    "captcha_code": VALID_CODE,
                },
            )
            assert login.status_code == 200, login.text
            token = login.json()["access_token"]

            me = await mysql_client.get(
                "/api/auth/me", headers={"Authorization": f"Bearer {token}"}
            )
            assert me.status_code == 200
            assert me.json()["username"] == username
            created_user_id = me.json()["id"]

            # 密码必须是 bcrypt 哈希，不能是明文
            conn = mysql_conn()
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT password_hash FROM psy_users WHERE username=%s", (username,)
                    )
                    stored = cur.fetchone()[0]
            finally:
                conn.close()
            assert stored != password
            assert stored.startswith("$2")
        finally:
            # 无论成败都清理，别在共享库留垃圾
            conn = mysql_conn()
            try:
                with conn.cursor() as cur:
                    if created_user_id is not None:
                        cur.execute(
                            "DELETE FROM psy_knowledge_bases WHERE owner_id=%s", (created_user_id,)
                        )
                        cur.execute("DELETE FROM psy_sessions WHERE user_id=%s", (created_user_id,))
                    cur.execute("DELETE FROM psy_users WHERE username=%s", (username,))
                conn.commit()
            finally:
                conn.close()

    async def test_health_reports_db_up(self, mysql_client: AsyncClient) -> None:
        response = await mysql_client.get("/api/health")
        body = response.json()

        assert response.status_code == 200
        assert body["db"] == "up", "健康检查应报告数据库可用"
