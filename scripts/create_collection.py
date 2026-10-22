"""新建 collection，支持 dense-only 与 hybrid 两种结构。

为什么要单独建一个 v2 而不是改 v1：
    v1 里 285 条向量是 bge-m3 产出的（稠密+稀疏），API 版只有稠密。
    两种向量混在一个集合里，检索分数的语义不一致，RRF 融合会算出没法解释的结果。
    分集合存，才能拿同一批问题做 A/B 对比——评估系统正需要这个。

用法：
    # 纯稠密（API 向量用这个）
    python -m scripts.create_collection --name psych_chunks_v2 --dense-only

    # 混合检索（本地 bge-m3 用这个）
    python -m scripts.create_collection --name psych_chunks_v1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402


def build_schema(client, dim: int, dense_only: bool):
    from pymilvus import DataType

    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("chunk_id", DataType.VARCHAR, is_primary=True, max_length=64)
    schema.add_field("kb_id", DataType.VARCHAR, max_length=64, is_partition_key=True)
    schema.add_field("doc_id", DataType.VARCHAR, max_length=64)
    schema.add_field("parent_id", DataType.VARCHAR, max_length=64)
    schema.add_field("content", DataType.VARCHAR, max_length=8192)
    schema.add_field("dense_vector", DataType.FLOAT_VECTOR, dim=dim)
    if not dense_only:
        schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)
    schema.add_field("page_start", DataType.INT32)
    schema.add_field("page_end", DataType.INT32)
    schema.add_field("domain", DataType.VARCHAR, max_length=32)
    schema.add_field("evidence_grade", DataType.VARCHAR, max_length=16)
    schema.add_field("is_parent", DataType.BOOL)
    return schema


def build_index(client, dense_only: bool):
    index = client.prepare_index_params()
    index.add_index(
        field_name="dense_vector",
        index_type="HNSW",
        metric_type="COSINE",
        params={"M": 16, "efConstruction": 200},
    )
    if not dense_only:
        index.add_index(
            field_name="sparse_vector",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP",
            params={"drop_ratio_build": 0.2},
        )
    return index


def main() -> int:
    parser = argparse.ArgumentParser(description="新建 Milvus collection")
    parser.add_argument("--name", default="psych_chunks_v2")
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--dense-only", action="store_true", help="不建稀疏向量字段")
    parser.add_argument("--force", action="store_true", help="已存在则先删除重建")
    args = parser.parse_args()

    from pymilvus import MilvusClient

    conf = settings.milvus
    client = MilvusClient(
        uri=conf.uri,
        token=f"{conf.user}:{conf.password}" if conf.user else "",
    )
    if conf.database:
        client.use_database(conf.database)

    existing = client.list_collections()
    if args.name in existing:
        if not args.force:
            print(f"collection {args.name} 已存在（加 --force 可重建）")
            return 1
        client.drop_collection(args.name)
        print(f"已删除旧 collection {args.name}")

    client.create_collection(
        collection_name=args.name,
        schema=build_schema(client, args.dim, args.dense_only),
        index_params=build_index(client, args.dense_only),
        consistency_level=conf.consistency_level,
    )
    kind = "dense-only" if args.dense_only else "hybrid(dense+sparse)"
    print(f"已创建 {args.name} | {kind} | dim={args.dim} | db={conf.database}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
