"""直接从磁盘文件跑一遍入库：建库 → 建文档记录 → 解析 → 切块 → 向量化 → 可选检索验证。

为什么绕过 HTTP：
    上传接口要先注册登录拿 token，调试链路时全是噪音。
    这个脚本直接写 DB 记录再调 IngestPipeline，跑一次就能看到
    每阶段耗时、切了多少块、写进 Milvus 多少条。

用法：
    # 先试 30 页，确认没问题再放量
    .venv/Scripts/python.exe scripts/ingest_book.py --pdf "D:/心理学文档/心理学教材/实验心理学（杨治良）.pdf" --pages 30

    # 用 PyMuPDF 解析（电子版 PDF 更快，不跑 MinerU）
    .venv/Scripts/python.exe scripts/ingest_book.py --pdf "..." --parser simple

    # 入库后顺手跑两个问题验证检索
    .venv/Scripts/python.exe scripts/ingest_book.py --pdf "..." --pages 30 --ask
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select  # noqa: E402

from app.core.logging import get_logger, setup_logging  # noqa: E402
from app.db.base import SessionLocal  # noqa: E402
from app.db.models import Document, KnowledgeBase, User  # noqa: E402
from app.ingest.pipeline import IngestPipeline  # noqa: E402

logger = get_logger(__name__)

DEV_USER = "dev_ingest"


async def ensure_user() -> str:
    async with SessionLocal() as session:
        user = await session.scalar(select(User).where(User.username == DEV_USER))
        if user is None:
            user = User(
                username=DEV_USER,
                # 只用于本地调试，不参与任何鉴权流程；密码哈希给个占位值
                password_hash="!dev-no-login",
            )
            session.add(user)
            await session.commit()
        return user.id


async def ensure_kb(user_id: str, name: str) -> str:
    async with SessionLocal() as session:
        kb = await session.scalar(
            select(KnowledgeBase).where(
                KnowledgeBase.owner_id == user_id, KnowledgeBase.name == name
            )
        )
        if kb is None:
            kb = KnowledgeBase(id=uuid.uuid4().hex, owner_id=user_id, name=name, description="入库调试")
            session.add(kb)
            await session.commit()
        return kb.id


async def create_document(kb_id: str, path: Path, pages: int | None) -> tuple[str, bool]:
    """建文档记录。返回 (doc_id, 是否命中已存在文档)。

    按内容哈希去重：同一本书重复入库会产生两份一样的向量，
    检索时用户看到两条重复结果，清理还得按 doc_id 逐条删。
    """
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    content_hash = digest.hexdigest()

    async with SessionLocal() as session:
        # 去重键是「内容哈希 + max_pages」。只按哈希会把"前 10 页试跑"和
        # "全书入库"当成同一份，导致正式入库被拦下来。
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
                return item.id, True

        doc_id = uuid.uuid4().hex
        session.add(
            Document(
                id=doc_id,
                kb_id=kb_id,
                filename=path.name,
                file_type=path.suffix.lstrip(".").lower(),
                size_bytes=path.stat().st_size,
                status="uploaded",
                last_stage="uploaded",
                storage_path=str(path),
                content_sha256=content_hash,
                parse_params={"max_pages": pages, "source": "ingest_book.py"},
            )
        )
        await session.commit()
    return doc_id, False


async def ask(kb_id: str, questions: list[str]) -> None:
    from app.clients.embed_client import build_embedder
    from app.clients.milvus_client import MilvusIndex
    from app.clients.rerank_client import BGEReranker
    from app.generate.intent import Intent
    from app.retrieve.service import retrieve

    embedder, index, reranker = build_embedder(), MilvusIndex(), BGEReranker()
    for question in questions:
        outcome = await retrieve(
            question, kb_ids=[kb_id], embedder=embedder, index=index,
            reranker=reranker, intent=Intent.DEFINE,
        )
        print(f"\nQ: {question}")
        if outcome.is_empty:
            print("   → 未命中（低于阈值）")
            continue
        print(f"   耗时 {outcome.stats.get('latency_ms')}ms 命中 {len(outcome.hits)} 条")
        for rank, hit in enumerate(outcome.hits[:3], 1):
            page = hit.meta.get("page_start", "?")
            print(f"   {rank}. p{page} {hit.score:.4f}  {hit.content[:60].replace(chr(10), ' ')}…")


async def main() -> int:
    parser = argparse.ArgumentParser(description="直接对磁盘文件跑一遍入库")
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--kb-name", default="入库调试库")
    parser.add_argument("--pages", type=int, default=None, help="只解析前 N 页")
    parser.add_argument("--parser", choices=["auto", "mineru", "simple"], default="auto")
    parser.add_argument("--ask", action="store_true", help="入库后跑检索验证")
    args = parser.parse_args()

    setup_logging()
    path = Path(args.pdf)
    if not path.exists():
        print(f"文件不存在：{path}")
        return 1

    print("=" * 70)
    print(f"文件：{path.name}   {path.stat().st_size/1048576:.1f} MB")
    print(f"解析器：{args.parser}    页数限制：{args.pages or '全部'}")
    print("=" * 70)

    user_id = await ensure_user()
    kb_id = await ensure_kb(user_id, args.kb_name)
    doc_id, duplicated = await create_document(kb_id, path, args.pages)
    if duplicated:
        print(f"该文件已在库中（内容哈希相同），复用 doc_id={doc_id}")
    else:
        print(f"新建 doc_id={doc_id}")
    print(f"kb_id={kb_id}\ndoc_id={doc_id}")

    parser_impl = None
    if args.parser == "mineru":
        from app.ingest.parsers import MinerUParser

        parser_impl = MinerUParser()
    elif args.parser == "simple":
        from app.ingest.parsers import SimpleParser

        parser_impl = SimpleParser()

    report = await IngestPipeline(parser=parser_impl).run(doc_id, max_pages=args.pages)

    print("\n" + "=" * 70)
    print(f"状态      : {report.status}")
    print(f"页数      : {report.pages}")
    print(f"父块/子块 : {report.parents} / {report.children}")
    print(f"写入向量  : {report.vectors}")
    print(f"总耗时    : {report.elapsed_ms/1000:.1f}s")
    if report.error:
        print(f"错误      : {report.error}")
    print("=" * 70)

    if args.ask and report.ok:
        await ask(kb_id, ["自变量和因变量怎么区分", "被试内设计是什么"])

    return 0 if report.ok else 2


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
