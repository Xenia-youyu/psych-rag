"""逐页调 MinerU，找出抛「Unsupported warp settings」的那一页。

一次调用崩一页就整本失败，所以只能一页一次调用、逐页捕获。
用法：python -m scripts._probe_page 129 192
"""

from __future__ import annotations

import sys

from app.core.logging import setup_logging
from app.ingest.parsers import MinerUParser

START = int(sys.argv[1]) if len(sys.argv) > 1 else 129
END = int(sys.argv[2]) if len(sys.argv) > 2 else 160

PATH = r"D:\心理学文档\普通心理学\心理学导论 梁宁建 上海教育出版社 第2版.pdf"


def main() -> None:
    setup_logging()
    parser = MinerUParser()

    from mineru.parser import parse as mineru_parse

    bad = []
    for page_no in range(START, END + 1):
        try:
            result = mineru_parse(
                PATH, tier=parser.tier, ocr_mode=parser.ocr_mode,
                page_range=f"{page_no}-{page_no}",
            )
            pages = getattr(result, "pages", []) or []
            chars = sum(
                len(t) for p in pages for t in [getattr(p, "text", "") or ""]
            )
            print(f"  p{page_no} ok  {len(pages)} 页 / {chars} 字")
        except Exception as exc:  # noqa: BLE001
            print(f"  p{page_no} FAIL {type(exc).__name__}: {str(exc)[:80]}")
            bad.append(page_no)

    print(f"\n区间 {START}-{END} 失败 {len(bad)} 页：{bad}")


if __name__ == "__main__":
    main()
