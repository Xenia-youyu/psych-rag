"""清洗已入库数据：剔除目录页、去掉页脚广告。

为什么需要单独跑一次：
    清洗逻辑是这次才加进 pipeline 的，已经入库的 11599 个块没有经过它。
    重跑全量入库能修好，但要 23 分钟和一笔 API 费用；
    只处理命中的块，几秒钟、几分钱。

处理三类：
    1. 目录/版权块 → 连向量一起删（它们是纯噪音）
    2. 含水印的块 → 删掉水印行，重新向量化（内容变了向量必须重算）
    3. 父块 → 只清洗 content，不重算向量（父块不参与召回，只用于回填全文）

用法：
    python -m scripts.clean_existing --dry-run
    python -m scripts.clean_existing --apply
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pymilvus import MilvusClient  # noqa: E402
from sqlalchemy import select  # noqa: E402

from app.clients.embed_client import build_embedder  # noqa: E402
from app.config import settings  # noqa: E402
from app.core.logging import setup_logging  # noqa: E402
from app.db.base import SessionLocal  # noqa: E402
from app.db.models import Chunk, Document, KnowledgeBase  # noqa: E402
from app.ingest.cleaner import is_junk, repair_text  # noqa: E402
from app.ingest.frontmatter import classify_page  # noqa: E402


async def main() -> int:
    parser = argparse.ArgumentParser(description="清洗已入库数据")
    parser.add_argument("--apply", action="store_true", help="不加这个参数只统计不写入")
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()

    setup_logging()
    dry = not args.apply

    async with SessionLocal() as session:
        docs = {d.id: d.filename for d in (await session.execute(select(Document.id, Document.filename))).all()}
        rows = (
            await session.execute(
                select(Chunk.id, Chunk.doc_id, Chunk.kb_id, Chunk.content, Chunk.char_count)
            )
        ).all()

    to_delete: list[str] = []
    to_update: list[tuple] = []   # (chunk_id, doc_id, kb_id, new_text)
    to_drop: list[str] = []       # 清洗后过短的块

    for chunk_id, doc_id, kb_id, content, _chars in rows:
        kind = classify_page(content or "")
        if kind in ("toc", "copyright"):
            to_delete.append(chunk_id)
            continue
        cleaned = repair_text(content or "")
        if cleaned != (content or ""):
            if is_junk(cleaned):
                to_drop.append(chunk_id)
            else:
                to_update.append((chunk_id, doc_id, kb_id, cleaned))

    print(f"扫描 {len(rows)} 块 | 目录/版权 {len(to_delete)} | 含水印 {len(to_update) + len(to_drop)} "
          f"| 其中整块丢弃 {len(to_drop)}")
    if dry:
        print("\n[dry-run] 未写入。加 --apply 执行")
        return 0

    # ---- 1. 删目录块 ----
    if to_delete or to_drop:
        removing = set(to_delete) | set(to_drop)
        client = MilvusClient(
            uri=settings.milvus.uri,
            token=f"{settings.milvus.user}:{settings.milvus.password}",
        )
        client.use_database(settings.milvus.database)
        quoted = ", ".join(f'"{cid}"' for cid in removing)
        client.delete(collection_name=settings.milvus.collection, filter=f"chunk_id in [{quoted}]")
        async with SessionLocal() as session:
            for chunk_id in removing:
                row = await session.get(Chunk, chunk_id)
                if row is not None:
                    await session.delete(row)
            await session.commit()
        print(f"已删除 {len(removing)} 块（含向量）")

    # ---- 2. 清洗并重新向量化 ----
    if to_update:
        embedder = build_embedder()
        print(f"重新向量化 {len(to_update)} 块…")
        texts = [item[3] for item in to_update]
        vectors = []
        for start in range(0, len(texts), args.batch):
            vectors.extend(await embedder.encode(texts[start : start + args.batch]))
        print(f"  消耗 tokens: {getattr(embedder, 'total_tokens', 0)}")

        client = MilvusClient(
            uri=settings.milvus.uri,
            token=f"{settings.milvus.user}:{settings.milvus.password}",
        )
        client.use_database(settings.milvus.database)

        records = []
        async with SessionLocal() as session:
            for (chunk_id, doc_id, kb_id, new_text), vector in zip(to_update, vectors, strict=True):
                row = await session.get(Chunk, chunk_id)
                if row is None:
                    continue
                row.content = new_text
                row.char_count = len(new_text)
                records.append(
                    {
                        "chunk_id": chunk_id,
                        "kb_id": kb_id,
                        "doc_id": doc_id,
                        "parent_id": row.parent_id or "",
                        "content": new_text[:8000],
                        "dense_vector": vector.dense,
                        "page_start": row.page_start,
                        "page_end": row.page_end,
                        "domain": "",
                        "evidence_grade": "",
                        "is_parent": False,
                    }
                )
            await session.commit()

        for start in range(0, len(records), 200):
            client.upsert(collection_name=settings.milvus.collection, data=records[start : start + 200])
        print(f"  已更新 {len(records)} 块的正文与向量")

    # ---- 3. 父块只清洗正文，向量保持不动 ----
    client = MilvusClient(
        uri=settings.milvus.uri,
        token=f"{settings.milvus.user}:{settings.milvus.password}",
    )
    client.use_database(settings.milvus.database)
    parents = client.query(
        collection_name=settings.milvus.collection,
        filter="is_parent == true",
        output_fields=[
            "chunk_id", "kb_id", "doc_id", "parent_id", "content",
            "dense_vector", "page_start", "page_end", "domain", "evidence_grade", "is_parent",
        ],
        limit=16384,
    )
    fixed = []
    for item in parents:
        cleaned = repair_text(item.get("content", ""))
        if cleaned != item.get("content", "") and cleaned:
            item["content"] = cleaned[:8000]
            fixed.append(item)
    for start in range(0, len(fixed), 200):
        client.upsert(collection_name=settings.milvus.collection, data=fixed[start : start + 200])
    print(f"父块清洗 {len(fixed)} 条（向量未重算）")

    # ---- 4. 重算知识库计数 ----
    async with SessionLocal() as session:
        from sqlalchemy import func

        for kb in (await session.scalars(select(KnowledgeBase))).all():
            kb.doc_count = await session.scalar(
                select(func.count(Document.id)).where(Document.kb_id == kb.id)
            ) or 0
            kb.chunk_count = await session.scalar(
                select(func.count(Chunk.id)).where(Chunk.kb_id == kb.id)
            ) or 0
            print(f"{kb.name}: {kb.doc_count} 文档 / {kb.chunk_count} 块")
        await session.commit()

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
