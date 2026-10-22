"""按 corpus_plan.json 执行入库。

为什么要有这个脚本，而不是直接敲 ingest_batch 的参数：
选书标准（为什么是这 6 本、为什么排除那 4 类）必须留痕。否则三个月后
看到库里 14 本书，没人说得清当初为什么没入《心理学与生活》英文版。
plan 文件是选书决策的唯一来源，脚本只负责执行。

用法：
    # 先检查文件都在、页数对得上
    python -m scripts.ingest_plan --batch batch1 --dry-run

    # 清空旧样本后全量入库
    python -m scripts.ingest_plan --batch batch1 --reset
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import delete, func, select  # noqa: E402

from app.clients.embed_client import build_embedder  # noqa: E402
from app.clients.milvus_client import MilvusIndex  # noqa: E402
from app.core.logging import setup_logging  # noqa: E402
from app.db.base import SessionLocal  # noqa: E402
from app.db.models import Chunk, Document  # noqa: E402
from app.ingest.parsers import MinerUParser, SimpleParser  # noqa: E402
from app.ingest.pipeline import IngestPipeline  # noqa: E402
from scripts.ingest_batch import (  # noqa: E402
    create_document,
    ensure_kb,
    ensure_user,
    sha256_of,
)

PLAN_DEFAULT = Path(__file__).resolve().parent.parent / "data" / "corpus_plan.json"


def load_batch(plan_path: Path, batch_id: str) -> tuple[dict, list[Path]]:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    root = Path(plan["root"])
    batch = next((b for b in plan["batches"] if b["id"] == batch_id), None)
    if batch is None:
        ids = [b["id"] for b in plan["batches"]]
        raise SystemExit(f"plan 里没有 {batch_id}，可选：{ids}")

    paths: list[Path] = []
    for item in batch.get("items", []):
        paths.append(root / item["path"])
    for item in batch.get("texts", []):
        paths.append(root / item["path"])
    return batch, paths


async def reset_kb(kb_id: str, index: MilvusIndex) -> dict[str, int]:
    """清空一个知识库的文档、块与向量。

    向量必须先删：MySQL 的块记录一删，就再也查不到该删哪些向量了，
    那些向量会永远留在 Milvus 里，检索时冒出来但回填不到正文。
    """
    async with SessionLocal() as session:
        docs = (await session.scalars(select(Document.id).where(Document.kb_id == kb_id))).all()
        chunks = int(
            await session.scalar(
                select(func.count(Chunk.id)).where(Chunk.kb_id == kb_id)
            )
            or 0
        )

    await index.delete_by_kb(kb_id)

    async with SessionLocal() as session:
        await session.execute(delete(Chunk).where(Chunk.kb_id == kb_id))
        await session.execute(delete(Document).where(Document.kb_id == kb_id))
        await session.commit()

    return {"documents": len(docs), "chunks": chunks}


async def main() -> int:
    parser = argparse.ArgumentParser(description="按计划入库")
    parser.add_argument("--plan", default=str(PLAN_DEFAULT))
    parser.add_argument("--batch", default="batch1")
    parser.add_argument("--kb-name", default="批量入库库")
    parser.add_argument("--reset", action="store_true", help="先清空该知识库的文档与向量")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--pages", type=int, default=0, help="0 = 全部页")
    parser.add_argument("--match", default="", help="只处理文件名包含该关键字的文件")
    parser.add_argument("--limit", type=int, default=0, help="最多处理前 N 个（0 = 不限）")
    parser.add_argument("--report", default="")
    args = parser.parse_args()

    setup_logging()
    batch, paths = load_batch(Path(args.plan), args.batch)
    pages = None if args.pages == 0 else args.pages

    if args.match:
        paths = [p for p in paths if args.match in p.name]
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        print(f"没有匹配的文件（--match {args.match!r}）")
        return 1

    print("=" * 84)
    print(f"计划 {args.batch} · {batch['name']}")
    print(f"解析器 {batch['parser']} | 每本 {pages or '全部'} 页 | 文件 {len(paths)} 个")
    print("=" * 84)

    missing = [p for p in paths if not p.is_file()]
    for path in paths:
        flag = "OK " if path.is_file() else "缺失"
        size = f"{path.stat().st_size / 1024 / 1024:6.1f}MB" if path.is_file() else " " * 8
        print(f"  [{flag}] {size}  {path.relative_to(Path(json.loads(Path(args.plan).read_text(encoding='utf-8'))['root']))}")

    if missing:
        print(f"\n{len(missing)} 个文件不存在，终止")
        return 1
    if args.dry_run:
        print("\n[dry-run] 文件齐全，未执行")
        return 0

    owner = await ensure_user()
    kb_id = await ensure_kb(owner, args.kb_name)
    index = MilvusIndex()

    if args.reset:
        removed = await reset_kb(kb_id, index)
        print(f"\n已清空旧数据：{removed['documents']} 个文档 / {removed['chunks']} 个块（向量同步删除）\n")

    parser_impl = MinerUParser() if batch["parser"] == "mineru" else SimpleParser()
    pipeline = IngestPipeline(parser=parser_impl, embedder=build_embedder(), index=index)

    rows: list[dict] = []
    started = time.perf_counter()

    for order, path in enumerate(paths, 1):
        content_hash = sha256_of(path)
        doc_id = await create_document(kb_id, path, pages, content_hash)
        label = path.name if len(path.name) <= 34 else path.name[:31] + "…"
        print(f"[{order}/{len(paths)}] {label:36}", end="", flush=True)

        report = await pipeline.run(doc_id, max_pages=pages)
        print(
            f"\r[{order}/{len(paths)}] {report.status:6} {label:36} "
            f"页 {report.pages:4d} 块 {report.children:5d} 向量 {report.vectors:5d} "
            f"{report.elapsed_ms / 1000:7.1f}s"
        )
        rows.append(
            {
                "file": path.name,
                "status": report.status,
                "pages": report.pages,
                "children": report.children,
                "vectors": report.vectors,
                "seconds": round(report.elapsed_ms / 1000, 1),
                "error": report.error,
            }
        )

    total_s = time.perf_counter() - started
    ok = [r for r in rows if r["status"] == "done"]
    failed = [r for r in rows if r["status"] == "failed"]

    print("\n" + "=" * 84)
    print(f"成功 {len(ok)} | 失败 {len(failed)} | 总耗时 {total_s / 60:.1f} 分钟")
    if ok:
        print(
            f"累计 页 {sum(r['pages'] for r in ok)} | 子块 {sum(r['children'] for r in ok)} "
            f"| 向量 {sum(r['vectors'] for r in ok)}"
        )
    for r in failed:
        print(f"  失败 {r['file']}: {str(r['error'])[:140]}")
    print("=" * 84)

    if args.report:
        payload = {
            "batch": args.batch,
            "kb_id": kb_id,
            "kb_name": args.kb_name,
            "pages": pages,
            "total_seconds": round(total_s, 1),
            "items": rows,
        }
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"报告已写入 {args.report}")

    return 0 if not failed else 2


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
