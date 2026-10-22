"""数据库结构增量迁移（幂等）。

为什么不引 Alembic：
    这个项目只有一个人在推，表结构改动频率低、改动量小。
    引 Alembic 要维护 versions 目录和迁移链，代价比收益大。
    这个脚本做的是同一件事的极简版：检查字段在不在，不在就加。

设计约束：
    - 只用 `ADD COLUMN IF NOT EXISTS` 的等价写法（MySQL 8 不支持这个语法，
      所以先查 information_schema）
    - 绝不 DROP 任何东西：共享服务器上有别人的数据，删错了没得救
    - 可反复执行，第二次跑什么都不做

用法：
    .venv/Scripts/python.exe -m scripts.migrate
"""

from __future__ import annotations

import pymysql

from app.config import settings

# (表名, 字段名, 建列语句, 索引名, 建索引语句)
MIGRATIONS: list[tuple[str, str, str, str, str]] = [
    (
        "psy_documents",
        "content_sha256",
        "ALTER TABLE psy_documents ADD COLUMN content_sha256 VARCHAR(64) NOT NULL DEFAULT '' "
        "COMMENT '文件内容 sha256，用于同库去重' AFTER storage_path",
        "idx_doc_kb_hash",
        "CREATE INDEX idx_doc_kb_hash ON psy_documents (kb_id, content_sha256)",
    ),
]


def main() -> int:
    conf = settings.mysql
    conn = pymysql.connect(
        host=conf.host,
        port=conf.port,
        user=conf.user,
        password=conf.password,
        database=conf.database,
        charset="utf8mb4",
        connect_timeout=8,
    )
    applied = 0
    try:
        with conn.cursor() as cur:
            for table, column, add_sql, index_name, index_sql in MIGRATIONS:
                cur.execute(
                    "SELECT COUNT(*) FROM information_schema.columns "
                    "WHERE table_schema=%s AND table_name=%s AND column_name=%s",
                    (conf.database, table, column),
                )
                if cur.fetchone()[0]:
                    print(f"[SKIP] {table}.{column} 已存在")
                else:
                    cur.execute(add_sql)
                    print(f"[ADD ] {table}.{column}")
                    applied += 1

                cur.execute(
                    "SELECT COUNT(*) FROM information_schema.statistics "
                    "WHERE table_schema=%s AND table_name=%s AND index_name=%s",
                    (conf.database, table, index_name),
                )
                if cur.fetchone()[0]:
                    print(f"[SKIP] 索引 {index_name} 已存在")
                else:
                    cur.execute(index_sql)
                    print(f"[ADD ] 索引 {index_name}")
                    applied += 1
        conn.commit()
    finally:
        conn.close()

    print(f"\n完成，本次变更 {applied} 项。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
