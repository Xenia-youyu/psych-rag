"""批量入库：一个目录或一串文件，逐本跑入库流水线。

为什么要单独写一个，而不是循环调 ingest_book.py：
    模型加载一次约 40 秒（bge-m3 2.2 GB）。逐本调用会把 40 秒乘以文件数，
    164 个文件光加载就要一个多小时。这里全程复用同一份 embedder / index。

去重：按「内容哈希 + max_pages」判断，同库同文件同页数直接跳过。

用法：
    # 先试 5 本、每本 10 页
    .venv/Scripts/python.exe scripts/ingest_batch.py --dir "D:/心理学文档" --limit 5 --pages 10

    # 指定文件
    .venv/Scripts/python.exe scripts/ingest_batch.py --files "a.pdf" "b.pdf"
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select  # noqa: E402

from app.clients.embed_client import build_embedder  # noqa: E402
from app.clients.milvus_client import MilvusIndex  # noqa: E402
from app.core.logging import setup_logging  # noqa: E402
from app.db.base import SessionLocal  # noqa: E402
from app.db.models import Document, KnowledgeBase, User  # noqa: E402
from app.ingest.parsers import SimpleParser  # noqa: E402
from app.ingest.pipeline import IngestPipeline  # noqa: E402

DEV_USER = "dev_ingest"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


async def ensure_user() -> str:
    async with SessionLocal() as session:
        user = await session.scalar(select(User).where(User.username == DEV_USER))
        if user is None:
            user = User(username=DEV_USER, password_hash="!dev-no-login")
            session.add(user)
            await session.commit()
        return user.id


async def ensure_kb(owner_id: str, name: str) -> str:
    async with SessionLocal() as session:
        kb = await session.scalar(
            select(KnowledgeBase).where(
                KnowledgeBase.owner_id == owner_id, KnowledgeBase.name == name
            )
        )
        if kb is None:
            kb = KnowledgeBase(id=uuid.uuid4().hex, owner_id=owner_id, name=name, description="批量入库")
            session.add(kb)
            await session.commit()
        return kb.id


async def find_existing(kb_id: str, content_hash: str, pages: int | None) -> Document | None:
    async with SessionLocal() as session:
        candidates = (
            await session.scalars(
                select(Document).where(
                    Document.kb_id == kb_id,
                    Document.content_sha256 == content_hash,
                    Document.status != "failed",
                )
            )
        ).all()
    for item in candidates:
        if (item.parse_params or {}).get("max_pages") == pages:
            return item
    return None


async def create_document(
    kb_id: str,
    path: Path,
    pages: int | None,
    content_hash: str,
    *,
    start_page: int | None = None,
    end_page: int | None = None,
) -> str:
    doc_id = uuid.uuid4().hex
    # 分段落库时同一个文件名会出现多条记录，光看文件名分不清是哪一段。
    # 在展示名后标上页码区间，引用来源里也看得出这段覆盖到哪。
    display_name = path.name
    if start_page or end_page:
        span = f"{start_page or 1}-{end_page or '末'}页"
        display_name = f"{path.stem}（{span}）{path.suffix}"
    async with SessionLocal() as session:
        session.add(
            Document(
                id=doc_id,
                kb_id=kb_id,
                filename=display_name,
                file_type=path.suffix.lstrip(".").lower(),
                size_bytes=path.stat().st_size,
                status="uploaded",
                last_stage="uploaded",
                storage_path=str(path),
                content_sha256=content_hash,
                parse_params={
                    "max_pages": pages,
                    "start_page": start_page,
                    "end_page": end_page,
                    "source": "ingest_batch.py",
                },
            )
        )
        await session.commit()
    return doc_id


def collect_files(args) -> list[Path]:
    if args.files:
        return [Path(f) for f in args.files if Path(f).exists()]
    root = Path(args.dir)
    paths = sorted(
        p for p in root.rglob("*.pdf")
        if p.is_file() and p.stat().st_size > 0
    )
    return paths[: args.limit] if args.limit else paths


async def main() -> int:
    parser = argparse.ArgumentParser(description="批量入库")
    parser.add_argument("--dir", default=r"D:\心理学文档")
    parser.add_argument("--files", nargs="*")
    parser.add_argument("--kb-name", default="批量入库库")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--pages", type=int, default=10, help="每本只解析前 N 页；0 表示全部")
    parser.add_argument("--parser", choices=["simple", "mineru"], default="simple")
    args = parser.parse_args()

    setup_logging()
    pages = None if args.pages == 0 else args.pages
    files = collect_files(args)
    if not files:
        print("没有找到可入库的文件")
        return 1

    print("=" * 78)
    print(f"待入库 {len(files)} 个文件 | 每本 {pages or '全部'} 页 | 解析器 {args.parser}")
    print("=" * 78)

    owner = await ensure_user()
    kb_id = await ensure_kb(owner, args.kb_name)

    embedder = build_embedder()
    index = MilvusIndex()
    if args.parser == "mineru":
        from app.ingest.parsers import MinerUParser

        parser_impl = MinerUParser()
    else:
        parser_impl = SimpleParser()

    pipeline = IngestPipeline(
        parser=parser_impl, embedder=embedder, index=index
    )

    rows: list[tuple[str, str, int, int, int, float, str]] = []
    started = time.perf_counter()

    for order, path in enumerate(files, 1):
        name = path.name if len(path.name) <= 34 else path.name[:31] + "…"
        content_hash = sha256_of(path)
        existing = await find_existing(kb_id, content_hash, pages)
        if existing is not None:
            print(f"[{order}/{len(files)}] 跳过（已入库）{name}")
            rows.append((name, "skipped", 0, 0, 0, 0.0, existing.id))
            continue

        doc_id = await create_document(kb_id, path, pages, content_hash)
        print(f"[{order}/{len(files)}] 入库中 {name} …", end="", flush=True)
        report = await pipeline.run(doc_id, max_pages=pages)
        print(
            f"\r[{order}/{len(files)}] {report.status:6} {name:36} "
            f"页 {report.pages:4d} 块 {report.children:5d} 向量 {report.vectors:5d} "
            f"{report.elapsed_ms/1000:6.1f}s"
        )
        rows.append(
            (name, report.status, report.pages, report.children, report.vectors,
             report.elapsed_ms / 1000, report.error)
        )

    total_s = time.perf_counter() - started
    ok = [r for r in rows if r[1] == "done"]
    failed = [r for r in rows if r[1] == "failed"]
    skipped = [r for r in rows if r[1] == "skipped"]

    print("\n" + "=" * 78)
    print(f"成功 {len(ok)} | 跳过 {len(skipped)} | 失败 {len(failed)} | 总耗时 {total_s:.1f}s")
    if ok:
        print(f"累计页数 {sum(r[2] for r in ok)} | 子块 {sum(r[3] for r in ok)} | 向量 {sum(r[4] for r in ok)}")
    if failed:
        print("\n失败明细：")
        for name, _status, _p, _c, _v, _s, err in failed:
            print(f"  {name}: {err[:120]}")
    print("=" * 78)
    return 0 if not failed else 2


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
