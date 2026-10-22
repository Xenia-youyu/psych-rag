"""初始化数据库：确认库可用 → 建表。

用法：
    python -m scripts.init_db

本项目用的是共享 MySQL，账号往往只有业务库权限、没有 CREATE DATABASE 权限。
所以流程是「先直连目标库，连得上就直接建表；连不上才尝试建库」，
不要一上来就 CREATE DATABASE 然后因为权限不足直接挂掉。

只建不改：已存在的表不会被修改。改字段请写迁移 SQL。
"""

from __future__ import annotations

import asyncio
import sys

import pymysql

from app.config import settings
from app.core.logging import get_logger, setup_logging
from app.db.base import init_models
from app.db.models import Base

logger = get_logger(__name__)

UNKNOWN_DATABASE = 1049


def ensure_database() -> None:
    conf = settings.mysql

    # 1. 先试直连目标库
    try:
        conn = pymysql.connect(
            host=conf.host,
            port=conf.port,
            user=conf.user,
            password=conf.password,
            database=conf.database,
            connect_timeout=8,
            charset=conf.charset,
        )
        conn.close()
        print(f"[OK]   数据库 {conf.database} 已存在且可访问")
        return
    except pymysql.err.OperationalError as exc:
        code = exc.args[0] if exc.args else None
        if code != UNKNOWN_DATABASE:
            raise SystemExit(
                f"[FAIL] 连接 MySQL 失败：{exc}\n"
                f"       检查 config.ini 的 [mysql] 与网络可达性"
            ) from exc
        print(f"[WARN] 数据库 {conf.database} 不存在，尝试创建…")

    # 2. 库不存在，才尝试建库（需要 CREATE 权限）
    try:
        conn = pymysql.connect(
            host=conf.host,
            port=conf.port,
            user=conf.user,
            password=conf.password,
            connect_timeout=8,
            charset=conf.charset,
        )
        with conn.cursor() as cur:
            cur.execute(
                f"CREATE DATABASE IF NOT EXISTS `{conf.database}` "
                f"CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            )
        conn.commit()
        conn.close()
        print(f"[OK]   已创建数据库 {conf.database}")
    except pymysql.err.OperationalError as exc:
        raise SystemExit(
            f"[FAIL] 无法创建数据库 {conf.database}：{exc}\n"
            f"       共享服务器通常只给业务库权限，请让管理员建库，或改用已有库名"
        ) from exc


async def main() -> None:
    setup_logging()
    ensure_database()

    tables = sorted(Base.metadata.tables)
    print(f"       即将创建 {len(tables)} 张表：{', '.join(tables)}")

    await init_models()
    print("[OK]   数据表创建完成")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] 建表失败：{exc!r}")
        sys.exit(1)
