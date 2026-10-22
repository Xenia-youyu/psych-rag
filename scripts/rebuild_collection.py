"""重建 Milvus collection（schema 变更后用）。

为什么需要它：Milvus 不支持给已有 collection 加字段。`is_parent` 这种
影响召回范围的字段只能靠重建 —— 而重建意味着向量全部消失，必须重新入库。

**这是破坏性操作。** 三道闸门，缺一不可：

1. collection 名必须以 `psych_` 开头。共享 Milvus 上还有别人的
   `edurag_0706` / `edurag_1006`，配置手滑指过去就是删别人的数据。
2. 必须带 `--yes`。裸跑只打印将要做什么。
3. 删 collection 前打印现有实体数，让人看清代价。

用法：

    # 只看会发生什么
    python -m scripts.rebuild_collection

    # 真删真建（向量会没，需要重新入库）
    python -m scripts.rebuild_collection --yes

    # 连 MySQL 里的文档记录一起清（否则去重会认为"已经入过了"而跳过）
    python -m scripts.rebuild_collection --yes --purge-mysql
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from app.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# 共享 Milvus 上别人的 collection 前缀。出现在这里说明配置指错了地方。
FOREIGN_PREFIXES = ("edurag_", "langchain_", "test_")


def _guard(name: str) -> None:
    if not name.startswith("psych_"):
        raise SystemExit(
            f"拒绝执行：collection 名 {name!r} 不以 psych_ 开头。\n"
            f"共享 Milvus 上有别人的数据，只允许操作本项目独占的 collection。"
        )
    if name.startswith(FOREIGN_PREFIXES):
        raise SystemExit(f"拒绝执行：{name!r} 看起来是别人的 collection。")


def _describe(client, name: str) -> tuple[int, list[str]]:
    from pymilvus import Collection, connections

    connections.connect(
        alias="rebuild_probe", uri=settings.milvus.uri, db_name=settings.milvus.database
    )
    collection = Collection(name, using="rebuild_probe")
    try:
        collection.load()
        count = collection.num_entities
    except Exception:  # noqa: BLE001 - 空 collection 加载失败不该中断流程
        count = 0
    fields = [f.name for f in collection.schema.fields]
    connections.disconnect("rebuild_probe")
    return count, fields


def rebuild_collection(confirm: bool) -> int:
    from pymilvus import MilvusClient

    conf = settings.milvus
    name = conf.collection
    _guard(name)

    client = MilvusClient(uri=conf.uri, timeout=10)
    if conf.database not in client.list_databases():
        raise SystemExit(f"database {conf.database} 不存在")
    client.use_database(conf.database)

    existing = client.list_collections()
    exists = name in existing

    print(f"目标   {conf.uri} / {conf.database} / {name}")
    if not exists:
        print("状态   collection 不存在，直接按新 schema 创建")
        count, fields = 0, []
    else:
        count, fields = _describe(client, name)
        print(f"状态   已存在：{count} 条实体")
        print(f"字段   {fields}")
        missing = "is_parent" not in fields
        print(f"缺 is_parent：{'是（需要重建）' if missing else '否（已是新 schema）'}")
        if not missing and not confirm:
            print("\n已经是新 schema，无需重建。")
            return 0

    if not confirm:
        print("\n以上是预览。确认要删掉并重建，请加 --yes。")
        print("注意：重建后向量全部消失，必须重新入库（可用 ingest_batch.py）。")
        return 1

    from app.check_env import _build_index, _build_schema

    if exists:
        client.drop_collection(name)
        print(f"已删除 collection {name}（丢弃 {count} 条实体）")

    client.create_collection(
        collection_name=name,
        schema=_build_schema(client),
        index_params=_build_index(client),
    )
    print(f"已按新 schema 创建 {name}")

    desc = client.describe_collection(name)
    names = [f.get("name") for f in desc.get("fields", [])]
    print(f"新字段列表：{names}")
    if "is_parent" not in names:
        print("警告：is_parent 没有建出来，schema 定义有问题")
        return 2

    print(
        "\n下一步必须重新入库，否则检索全空。\n"
        "  .venv/Scripts/python.exe scripts/ingest_batch.py --files <...>\n"
        "MySQL 里的文档记录若不清，去重会把重跑全部跳过 —— 加 --purge-mysql 一并清。"
    )
    return 0


async def purge_mysql() -> int:
    """清掉本项目的文档与块记录。

    向量没了而 MySQL 记录还在，会出现两种假象：
    1. 文档列表显示"已完成"，但检索一条都搜不到；
    2. 内容哈希去重认为已经入过，重跑全部跳过 —— 于是永远补不回来。
    """
    from sqlalchemy import delete, func, select

    from app.db.base import SessionLocal
    from app.db.models import Chunk, Document, KnowledgeBase

    async with SessionLocal() as session:
        docs = await session.scalar(select(func.count()).select_from(Document)) or 0
        chunks = await session.scalar(select(func.count()).select_from(Chunk)) or 0
        print(f"MySQL 现有 {docs} 个文档 / {chunks} 个块，即将删除")

        await session.execute(delete(Chunk))
        await session.execute(delete(Document))
        kbs = (await session.scalars(select(KnowledgeBase))).all()
        for kb in kbs:
            kb.doc_count = 0
            kb.chunk_count = 0
        await session.commit()
        print(f"已清空（知识库 {len(kbs)} 个，文档与块计数归零；知识库本身保留）")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="重建 Milvus collection")
    parser.add_argument("--yes", action="store_true", help="确认执行删除与重建")
    parser.add_argument("--purge-mysql", action="store_true",
                        help="同时清空 MySQL 的文档与块记录（重建后必须重入）")
    args = parser.parse_args()

    code = rebuild_collection(args.yes)
    if code != 0 or not args.yes:
        return code
    if args.purge_mysql:
        asyncio.run(purge_mysql())
    return 0


if __name__ == "__main__":
    sys.exit(main())
