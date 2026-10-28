"""整本抢救入库：自动递归细分，定位并跳过坏页。

为什么需要这个脚本：
    MinerU 的 OCR 后处理会对检测框做旋转裁剪，遇到几何退化的框（宽或高为 0）
    直接抛 ValueError。**一个坏框会让它所在的那一段返回 0 页**，整段白跑。
    上一批（心理与教育测量 / 统计学）是用手工四级细分救回来的：
    64 → 16 → 8 → 4 → 1，一共跑了两轮、看了十几个日志才把 3 个坏页定位出来。
    手工做这件事的问题是：每级的窗口都要自己算，容易漏段、容易重复跑。

这个脚本把「分级细分」变成一次调用：
    1. 先按 --chunk-pages 全量跑一遍（默认 64 页一段）
    2. 失败的段自动二分，直到宽度降到 --min-window（默认 1 页）
    3. 宽度 = 1 还失败 → 记为坏页，跳过，不阻塞其余部分
    4. 结果写 JSON 报告，坏页清单单独列出来

用法：
    # 先看会跑多少段
    .venv/Scripts/python.exe scripts/recover_book.py --file "…/心理学导论.pdf" --dry-run

    # 整本抢救
    .venv/Scripts/python.exe scripts/recover_book.py --file "…/心理学导论.pdf" \
        --chunk-pages 64 --report data/recover_intro.json

注意：
    每个窗口都是一条独立 Document 记录（文件名带页码区间）。
    失败的窗口会留下一条 status=failed、0 块的记录——这是残留，不是数据。
    清理要用 scripts/clean_failed_docs.py 先出清单再删，别直接 DELETE。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.clients.embed_client import build_embedder  # noqa: E402
from app.clients.milvus_client import MilvusIndex  # noqa: E402
from app.core.logging import setup_logging  # noqa: E402
from app.ingest.parsers import MinerUParser, SimpleParser  # noqa: E402
from app.ingest.pipeline import IngestPipeline  # noqa: E402
from scripts.ingest_batch import (  # noqa: E402
    create_document,
    ensure_kb,
    ensure_user,
    sha256_of,
)

PLAN_DEFAULT = Path(__file__).resolve().parent.parent / "data" / "corpus_plan.json"


def resolve_target(plan_path: Path, file_arg: str) -> tuple[Path, int]:
    """把 --file 解析成绝对路径，并从 plan 里取这本书的页数（0 = 未知）。"""
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    root = Path(plan["root"])

    raw = Path(file_arg)
    if raw.is_absolute():
        path = raw
    elif raw.exists():
        path = raw.resolve()
    else:
        path = root / file_arg

    total = 0
    for item in plan.get("pending_recovery", []) + [
        i for b in plan.get("batches", []) for i in b.get("items", [])
    ]:
        if item.get("path") and Path(item["path"]).name == path.name:
            total = int(item.get("pages") or 0)
            break
    return path, total


async def main() -> int:
    ap = argparse.ArgumentParser(description="整本抢救入库（递归细分定位坏页）")
    ap.add_argument("--plan", default=str(PLAN_DEFAULT))
    ap.add_argument("--file", required=True, help="PDF 路径，或相对 plan.root 的路径")
    ap.add_argument("--kb-name", default="批量入库库")
    ap.add_argument("--parser", default="mineru", choices=["mineru", "simple"])
    ap.add_argument("--pages", type=int, default=0, help="限制解析页数（0 = 不限制）")
    ap.add_argument("--chunk-pages", type=int, default=64, help="首轮分段大小")
    ap.add_argument("--min-window", type=int, default=1, help="细分到多小就认定坏页")
    ap.add_argument("--total-pages", type=int, default=0, help="plan 里没写页数时手工给")
    ap.add_argument("--report", default="")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    setup_logging()
    plan_path = Path(args.plan)
    path, total = resolve_target(plan_path, args.file)
    total = args.total_pages or total

    if not path.is_file():
        print(f"文件不存在：{path}")
        return 1
    if total <= 0:
        print("不知道总页数：plan 里没有这本，用 --total-pages 指定")
        return 1

    windows = [
        (ws, min(ws + args.chunk_pages - 1, total))
        for ws in range(1, total + 1, args.chunk_pages)
    ]

    print("=" * 84)
    print(f"目标 {path.name}")
    print(f"总页 {total} | 首轮 {len(windows)} 段 × {args.chunk_pages} 页 | 最小窗口 {args.min_window} 页")
    for ws, we in windows:
        print(f"   {ws:>4}-{we:<4}", end="")
    print()
    print("=" * 84)
    if args.dry_run:
        print("[dry-run] 未执行")
        return 0

    owner = await ensure_user()
    kb_id = await ensure_kb(owner, args.kb_name)
    index = MilvusIndex()
    parser_impl = MinerUParser() if args.parser == "mineru" else SimpleParser()
    pipeline = IngestPipeline(parser=parser_impl, embedder=build_embedder(), index=index)

    content_hash = sha256_of(path)
    pages = None if args.pages == 0 else args.pages
    rows: list[dict] = []
    bad_pages: list[int] = []
    created: list[str] = []
    started = time.perf_counter()

    async def run_window(ws: int, we: int, depth: int) -> None:
        doc_id = await create_document(
            kb_id, path, pages, content_hash, start_page=ws, end_page=we
        )
        created.append(doc_id)
        label = f"[{ws}-{we}]" + ("" if depth == 0 else f"↳{depth}")
        print(f"  {label:14}", end="", flush=True)
        report = await pipeline.run(doc_id, max_pages=pages, start_page=ws, end_page=we)
        print(
            f"\r  {label:14} {report.status:6} 页 {report.pages:4d} 块 {report.children:5d} "
            f"{report.elapsed_ms / 1000:7.1f}s   ",
            flush=True,
        )
        rows.append(
            {
                "window": [ws, we],
                "depth": depth,
                "doc_id": doc_id,
                "status": report.status,
                "pages": report.pages,
                "children": report.children,
                "vectors": report.vectors,
                "seconds": round(report.elapsed_ms / 1000, 1),
                "error": report.error,
            }
        )

        if report.status == "done":
            return
        if we - ws + 1 <= args.min_window:
            bad_pages.append(ws)
            print(f"      坏页 {ws}：{str(report.error)[:90]}", flush=True)
            return
        mid = (ws + we) // 2
        await run_window(ws, mid, depth + 1)
        await run_window(mid + 1, we, depth + 1)

    for ws, we in windows:
        await run_window(ws, we, 0)

    elapsed = time.perf_counter() - started
    ok = [r for r in rows if r["status"] == "done"]
    failed = [r for r in rows if r["status"] != "done"]

    print("\n" + "=" * 84)
    print(f"成功 {len(ok)} 段 | 失败 {len(failed)} 段 | 坏页 {len(bad_pages)} 个 | 耗时 {elapsed / 60:.1f} 分钟")
    print(
        f"累计 页 {sum(r['pages'] for r in ok)} | 子块 {sum(r['children'] for r in ok)} "
        f"| 向量 {sum(r['vectors'] for r in ok)}"
    )
    print(f"坏页清单：{bad_pages or '无'}")
    covered = sorted({p for r in ok for p in range(r["window"][0], r["window"][1] + 1)})
    missing = [p for p in range(1, total + 1) if p not in set(covered)]
    print(f"未覆盖页（坏页 + 空页）：{missing or '无'}")
    print("=" * 84)

    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(
            json.dumps(
                {
                    "file": str(path),
                    "kb_id": kb_id,
                    "kb_name": args.kb_name,
                    "total_pages": total,
                    "chunk_pages": args.chunk_pages,
                    "total_seconds": round(elapsed, 1),
                    "bad_pages": bad_pages,
                    "missing_pages": missing,
                    "failed_doc_ids": [r["doc_id"] for r in failed],
                    "items": rows,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"报告已写入 {args.report}")

    return 0 if not bad_pages else 3


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
