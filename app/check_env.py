"""环境连通性自检：MySQL / Milvus / Redis 能不能连、库和 collection 对不对、权限够不够。

填完 config.ini 第一件事就跑这个，比启动服务后看报错快得多。

用法：
    python -m app.check_env
    python -m app.check_env --create        # 缺表/缺 collection 时自动创建

退出码：0 全部通过；1 有项目失败。

Milvus 部分用 MilvusClient（pymilvus 3.x 的推荐接口），
老的 connections / utility / db 那套在 3.x 已经被标记废弃。
"""

from __future__ import annotations

import sys

from app.config import settings

OK = "[OK]  "
BAD = "[FAIL]"
WARN = "[WARN]"

# sparse_vector 只有在本地 BGE-M3（稠密+稀疏）下才存在。
# 走百炼 text-embedding-v4 时集合是纯稠密的，没有这个字段——
# 写死期望会让自检把"正常的集合"报成 FAIL。
BASE_FIELDS = {
    "chunk_id",
    "kb_id",
    "doc_id",
    "parent_id",
    "content",
    "dense_vector",
    "page_start",
    "page_end",
    "domain",
    "evidence_grade",
    "is_parent",
}

REQUIRED_FIELDS = BASE_FIELDS | (
    {"sparse_vector"} if settings.embedding.provider != "dashscope" else set()
)


# ---------------------------------------------------------------- MySQL
def check_mysql(create: bool) -> str:
    try:
        import pymysql
    except ImportError:
        print(f"{WARN} MySQL 跳过：未安装 pymysql。安装：pip install pymysql")
        return "skip"

    conf = settings.mysql
    print(f"{OK} 目标 {conf.masked}")

    try:
        conn = pymysql.connect(
            host=conf.host, port=conf.port, user=conf.user,
            password=conf.password, connect_timeout=6, charset=conf.charset,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"{BAD} 连接失败：{exc}")
        return "fail"

    try:
        with conn.cursor() as cur:
            cur.execute("SELECT VERSION()")
            print(f"{OK} 连接成功，服务端版本 {cur.fetchone()[0]}")

            cur.execute("SHOW DATABASES LIKE %s", (conf.database,))
            if cur.fetchone() is None:
                print(f"{BAD} 数据库 {conf.database} 不存在且无建库说明，先跑 python -m scripts.init_db")
                return "fail"
            print(f"{OK} 数据库 {conf.database} 可访问")

            cur.execute(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema=%s",
                (conf.database,),
            )
            table_count = cur.fetchone()[0]

            cur.execute(
                "SELECT COUNT(*) FROM information_schema.tables "
                "WHERE table_schema=%s AND table_name LIKE 'psy\\_%%'",
                (conf.database,),
            )
            ours = cur.fetchone()[0]

            print(f"{OK} 库内共 {table_count} 张表，其中本项目 {ours} 张（psy_ 前缀）")
            if ours == 0:
                print("      还没建表，跑 python -m scripts.init_db")
                return "fail"
            if ours < 8:
                print(f"{WARN} 只有 {ours} 张表，预期 8 张，可能建表未完成")
                return "fail"
            return "ok"
    finally:
        conn.close()


# ---------------------------------------------------------------- Redis
def check_redis() -> str:
    try:
        import redis
    except ImportError:
        print(f"{WARN} Redis 跳过：未安装驱动。安装：pip install redis")
        return "skip"

    conf = settings.redis
    try:
        client = redis.Redis(
            host=conf.host, port=conf.port, password=conf.password or None,
            db=conf.db, socket_connect_timeout=5, socket_timeout=5,
        )
        client.ping()
        client.set("__psy_probe__", "1", ex=10)
        client.delete("__psy_probe__")
        print(f"{OK} 连接成功 {conf.host}:{conf.port} db={conf.db}（读写正常）")
        return "ok"
    except Exception as exc:  # noqa: BLE001
        print(f"{BAD} 连接失败：{exc}")
        return "fail"


# ---------------------------------------------------------------- Milvus
def _build_schema(client):
    from pymilvus import DataType

    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("chunk_id", DataType.VARCHAR, is_primary=True, max_length=64)
    schema.add_field("kb_id", DataType.VARCHAR, max_length=64, is_partition_key=True)
    schema.add_field("doc_id", DataType.VARCHAR, max_length=64)
    schema.add_field("parent_id", DataType.VARCHAR, max_length=64)
    schema.add_field("content", DataType.VARCHAR, max_length=8192)
    schema.add_field("dense_vector", DataType.FLOAT_VECTOR, dim=1024)
    schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)
    # 不设 default_value：Milvus 会把 Python 的 0 推断成 Int64，与 INT32 冲突。
    # 这两个字段由写入方保证提供（见 retrieve / ingest 的 upsert 记录）。
    schema.add_field("page_start", DataType.INT32)
    schema.add_field("page_end", DataType.INT32)
    schema.add_field("domain", DataType.VARCHAR, max_length=32)
    schema.add_field("evidence_grade", DataType.VARCHAR, max_length=16)
    # 父块与子块混在同一个 collection 里（父块用于命中后回填全文）。
    # 但父块不该参与召回：父块比分块长 4 倍，它的向量是整页混在一起的
    # 平均语义，容易被无关问题以"大而全"的相似度挤进 topK，
    # 把真正的子块命中位置占掉。检索时用 is_parent == false 排除。
    # 必须显式给值：Milvus 的 BOOL 字段没有可靠的默认值语义。
    schema.add_field("is_parent", DataType.BOOL)
    return schema


def _build_index(client):
    index = client.prepare_index_params()
    index.add_index(
        field_name="dense_vector", index_type="HNSW",
        metric_type="COSINE", params={"M": 16, "efConstruction": 200},
    )
    index.add_index(
        field_name="sparse_vector", index_type="SPARSE_INVERTED_INDEX",
        metric_type="IP", params={"drop_ratio_build": 0.2},
    )
    return index


def _use_db(client, name: str) -> None:
    """兼容 pymilvus 3.0 的 use_database 与旧版 using_database。"""
    if hasattr(client, "use_database"):
        try:
            client.use_database(name)
            return
        except (AttributeError, TypeError):
            pass
    client.using_database(name)


def _list_collection_names(client) -> list[str]:
    names = client.list_collections()
    # 不同版本可能返回名字列表或 [{name: ...}]
    return [n if isinstance(n, str) else n.get("name", "") for n in names]


def check_milvus(create: bool) -> str:
    try:
        from pymilvus import MilvusClient
    except ImportError:
        print(f"{WARN} Milvus 跳过：未安装驱动。安装：pip install --no-deps pymilvus")
        return "skip"

    conf = settings.milvus
    print(f"{OK} 目标 {conf.uri}")

    try:
        client = MilvusClient(uri=conf.uri, timeout=10)
        databases = client.list_databases()
        print(f"{OK} 连接成功，databases={databases}")
    except Exception as exc:  # noqa: BLE001
        print(f"{BAD} 连接失败：{exc}")
        return "fail"

    try:
        if conf.database not in databases:
            print(f"{BAD} database {conf.database} 不存在。共享 Milvus 上不建议自建 database，请改用已有库名")
            return "fail"

        _use_db(client, conf.database)
        names = _list_collection_names(client)
        print(f"{OK} [{conf.database}] 现有 collection：{names}")

        if conf.collection not in names:
            print(f"{WARN} collection {conf.collection} 不存在")
            if not create:
                print("      加 --create 可自动创建（dense 1024 HNSW/COSINE + sparse 倒排/IP + kb_id 分区键）")
                return "fail"
            client.create_collection(
                collection_name=conf.collection,
                schema=_build_schema(client),
                index_params=_build_index(client),
            )
            print(f"{OK} 已创建 collection {conf.collection}")
            return "ok"

        desc = client.describe_collection(conf.collection)
        fields = {f.get("name") for f in desc.get("fields", [])}
        missing = REQUIRED_FIELDS - fields
        if missing:
            print(f"{BAD} collection {conf.collection} 缺字段 {sorted(missing)}")
            print("      两种可能：")
            print("      a) 这个 collection 是别人的（例如 edurag_1006），把 collection_name 改成自己独占的名字")
            print("      b) 本项目 schema 升级过（如 is_parent），需要重建：")
            print("         python -m scripts.rebuild_collection          # 先看影响")
            print("         python -m scripts.rebuild_collection --yes --purge-mysql")
            return "fail"

        # pymilvus 3.x 的 describe_collection 不再填充顶层 partition_key_field，
        # 分区键信息只保留在 fields[].is_partition_key 上
        partition_key = next(
            (f.get("name") for f in desc.get("fields", []) if f.get("is_partition_key")),
            None,
        )
        rows = client.query(collection_name=conf.collection, filter="", output_fields=["count(*)"])
        total = rows[0].get("count(*)", 0) if rows else 0
        print(f"{OK} collection 结构正确，partition_key={partition_key}，实体数 {total}")
        if partition_key != "kb_id":
            print(f"{WARN} partition_key 不是 kb_id，多知识库隔离会退化为标量过滤（能用但更慢）")
        return "ok"
    except Exception as exc:  # noqa: BLE001
        print(f"{BAD} 检查失败：{exc}")
        return "fail"


# ---------------------------------------------------------------- main
def main() -> int:
    create = "--create" in sys.argv
    print("=" * 64)
    print("心理学 RAG · 环境连通性自检")
    print("=" * 64)

    print("\n--- MySQL ---")
    results = {"MySQL": check_mysql(create)}

    print("\n--- Milvus ---")
    results["Milvus"] = check_milvus(create)

    print("\n--- Redis（enabled 时必需）---")
    if settings.redis.enabled:
        results["Redis"] = check_redis()
    else:
        print(f"{WARN} config.ini 中 redis.enabled=false，跳过（M4 才需要）")
        results["Redis"] = "skip"

    fails = [k for k, v in results.items() if v not in ("ok", "skip")]
    skips = [k for k, v in results.items() if v == "skip"]
    oks = [k for k, v in results.items() if v == "ok"]

    print("\n" + "=" * 64)
    print(f"通过 {len(oks)} 项{('：' + '、'.join(oks)) if oks else ''}")
    if skips:
        print(f"跳过 {len(skips)} 项{('：' + '、'.join(skips))}（缺驱动或未启用）")
    if fails:
        print(f"失败 {len(fails)} 项{('：' + '、'.join(fails))}")
        print("结论：有项目未通过，先解决再往下走。")
    elif skips:
        print("结论：已测项目全部通过，有项目未验证。")
    else:
        print("结论：全部通过，可以开始开发。")
    print("=" * 64)
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
