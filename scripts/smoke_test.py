"""小样本端到端冒烟测试：真实 PDF → 真实向量 → 真实 Milvus → 真实重排。

为什么要有这个脚本：
    单元测试全用替身，能证明逻辑对，证明不了"真模型 + 真 Milvus 串起来能出正确结果"。
    而全量建库要几十小时，任何问题都要跑完才发现。所以先拿几十页真书跑一遍闭环。

这一版刻意**不走 MinerU**：
    冒烟测试用 PyMuPDF 直接抽文本层、按页切父块。
    MinerU 的价值在扫描件与复杂版面，这里验证的是检索链路本身。
    正式链路（MinerU → IR → 版面感知切块）在 M2 剩余部分接。

用法：
    .venv/Scripts/python.exe scripts/smoke_test.py                  # 建库 + 检索
    .venv/Scripts/python.exe scripts/smoke_test.py --pages 60
    .venv/Scripts/python.exe scripts/smoke_test.py --query-only     # 只复验检索（改阈值后）
    .venv/Scripts/python.exe scripts/smoke_test.py --cleanup        # 跑完删数据
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pymupdf  # noqa: E402

from app.clients.embed_client import BGEM3Embedder  # noqa: E402
from app.clients.milvus_client import MilvusIndex  # noqa: E402
from app.clients.rerank_client import BGEReranker  # noqa: E402
from app.config import settings  # noqa: E402
from app.generate.intent import Intent  # noqa: E402
from app.ingest.chunker import split_parent_child  # noqa: E402
from app.retrieve.service import retrieve  # noqa: E402

DEFAULT_PDF = r"D:\心理学文档\心理学教材\实验心理学（杨治良）.pdf"

QUERIES = [
    ("实验中的自变量和因变量怎么区分", Intent.CONTRAST),
    ("什么是被试内设计", Intent.DEFINE),
    ("实验心理学研究的基本程序有哪些", Intent.FRAMEWORK),
]


def extract_pages(pdf_path: str, limit: int) -> list[tuple[int, str]]:
    """逐页抽文本层，返回 [(页码, 正文)]。"""
    doc = pymupdf.open(pdf_path)
    pages: list[tuple[int, str]] = []
    for index in range(min(limit, doc.page_count)):
        text = doc[index].get_text().strip()
        if len(text) >= 100:  # 封面、目录页跳过
            pages.append((index + 1, text))
    doc.close()
    return pages


async def ingest(args, index: MilvusIndex) -> tuple[float, int]:
    """抽文本 → 切父子块 → 向量化 → 写 Milvus。返回 (向量化耗时, 页数)。"""
    t0 = time.perf_counter()
    pages = extract_pages(args.pdf, args.pages)
    total_chars = sum(len(text) for _, text in pages)

    parents = []
    for page_no, text in pages:
        for chunk in split_parent_child(
            text,
            parent_size=settings.chunk.parent_size,
            parent_min_size=settings.chunk.parent_min_size,
            child_size=settings.chunk.child_size,
            child_min_size=settings.chunk.child_min_size,
            child_overlap=int(settings.chunk.child_size * settings.chunk.overlap_ratio),
            separators=settings.chunk.separators or None,
            id_prefix=f"p{page_no:04d}_",
        ):
            parents.append((page_no, chunk))

    children = sum(len(chunk.children) for _, chunk in parents)
    print(f"[1] 抽取 {len(pages)} 页 / {total_chars} 字 → 父块 {len(parents)} / 子块 {children}"
          f"  ({time.perf_counter()-t0:.1f}s)")

    doc_id = "doc_" + hashlib.sha1(
        (args.pdf + str(args.pages)).encode("utf-8")
    ).hexdigest()[:12]

    texts: list[str] = []
    owners: list[tuple[str, str, int]] = []  # (chunk_id, parent_id, page)
    for page_no, parent in parents:
        pid = f"{doc_id}_{parent.parent_id}"
        texts.append(parent.content)
        owners.append((pid, "", page_no))
        for child_no, child in enumerate(parent.children):
            texts.append(child)
            owners.append((f"{pid}_c{child_no:02d}", pid, page_no))

    embedder = BGEM3Embedder()
    t0 = time.perf_counter()
    vectors = await embedder.encode(texts)
    embed_sec = time.perf_counter() - t0
    print(f"[2] 向量化 {len(texts)} 条 ({embed_sec:.1f}s, {len(texts)/embed_sec:.1f} 条/秒)")

    records = [
        {
            "chunk_id": chunk_id,
            "kb_id": args.kb,
            "doc_id": doc_id,
            "parent_id": parent_id,
            "content": text[:8000],
            "dense_vector": vector.dense,
            "sparse_vector": {int(k): v for k, v in vector.sparse.items()},
            "page_start": page_no,
            "page_end": page_no,
            "domain": "实验心理学",
            "evidence_grade": "教材共识",
            # parent_id 为空 = 父块。检索会按 is_parent 把父块排除在召回之外
            "is_parent": not parent_id,
        }
        for (chunk_id, parent_id, page_no), vector, text in zip(owners, vectors, texts, strict=True)
    ]

    t0 = time.perf_counter()
    inserted = await index.upsert(records)
    print(f"[3] 写入 Milvus {inserted} 条 ({time.perf_counter()-t0:.1f}s)")
    return embed_sec, len(pages)


async def run_queries(args, index: MilvusIndex) -> None:
    embedder = BGEM3Embedder()
    reranker = BGEReranker()

    print(f"\n[4] 检索验证（阈值 rerank_score_threshold="
          f"{settings.retrieval.rerank_score_threshold}）")
    print("-" * 70)

    for question, intent in QUERIES:
        t0 = time.perf_counter()
        outcome = await retrieve(
            question,
            kb_ids=[args.kb],
            embedder=embedder,
            index=index,
            reranker=reranker,
            intent=intent,
        )
        elapsed = time.perf_counter() - t0
        print(f"\nQ: {question}")
        print(f"   意图={outcome.intent.value} 策略={outcome.strategy} "
              f"耗时={elapsed*1000:.0f}ms "
              f"重排后候选={outcome.stats.get('after_threshold', 0)}")
        print(f"   recalled={outcome.stats.get('recalled')} "
              f"fused={outcome.stats.get('fused')} "
              f"truncated={outcome.stats.get('rerank_truncated')} "
              f"rerank_ok={outcome.stats.get('rerank_ok')}")
        if outcome.is_empty:
            print("   → 无命中（低于阈值），生成阶段应回答「知识库未覆盖」")
            continue
        for rank, hit in enumerate(outcome.hits[:3], 1):
            page = hit.meta.get("page_start", "?")
            snippet = hit.content.replace("\n", " ")[:70]
            print(f"   {rank}. p{page} score={hit.score:.4f}  {snippet}…")
        print(f"   引用数={len(outcome.citations)} 上下文字数={len(outcome.context)}")


async def main() -> int:
    parser = argparse.ArgumentParser(description="小样本端到端冒烟测试")
    parser.add_argument("--pdf", default=DEFAULT_PDF)
    parser.add_argument("--pages", type=int, default=40)
    parser.add_argument("--kb", default="smoke_kb_001")
    parser.add_argument("--cleanup", action="store_true", help="跑完删除写入的数据")
    parser.add_argument("--query-only", action="store_true",
                        help="跳过解析与入库，只对已有 kb 跑检索（改阈值后复验用）")
    args = parser.parse_args()

    if not args.query_only and not Path(args.pdf).exists():
        print(f"样本不存在：{args.pdf}")
        return 1

    print("=" * 70)
    if args.query_only:
        print("模式：只检索（复用已有数据）")
    else:
        print(f"样本：{Path(args.pdf).name}  前 {args.pages} 页")
    print(f"Milvus：{settings.milvus.uri} / db={settings.milvus.database} "
          f"/ collection={settings.milvus.collection}")
    print("=" * 70)

    index = MilvusIndex()
    embed_sec, page_count = 0.0, 0

    if args.query_only:
        print(f"[复用] kb={args.kb} 现有 {await index.count()} 条记录")
    else:
        embed_sec, page_count = await ingest(args, index)

    await run_queries(args, index)

    if args.cleanup:
        await index.delete_by_kb(args.kb)
        print(f"\n[5] 已删除 kb={args.kb} 的全部数据")

    print("\n" + "=" * 70)
    if not args.query_only:
        print(f"向量化 {embed_sec:.0f}s / {page_count} 页")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
