"""OCR 探针：指定页码区间跑 MinerU，打印产物原文，只读不改库。

为什么要单独探：昨天「单页验证通过」是假象——单页时刚好有 text 字段，
整本走的是 markdown 兜底路径，3 页返回 86745 字，99% 是 base64。
验证解析器必须看产物内容，不能只看页数和字数。

另外：一本书的前几页往往是封面/扉页/版权页，抽前 3 页会得出「OCR 不出字」的
错误结论。必须抽正文页。用法：

    python -m scripts._probe_ocr "D:\\书.pdf" 8 11     # 第 8~11 页
"""

from __future__ import annotations

import sys

from app.ingest.parsers import MinerUParser, _block_text
from app.ingest.cleaner import repair_text
from app.ingest.frontmatter import classify_page


def main() -> None:
    path = sys.argv[1]
    start = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    end = int(sys.argv[3]) if len(sys.argv) > 3 else start + 2

    print(f"文件：{path}\n页码：{start}-{end}\n")

    from mineru.parser import parse as mineru_parse

    parser = MinerUParser()
    result = mineru_parse(path, tier=parser.tier, ocr_mode=parser.ocr_mode,
                          page_range=f"{start}-{end}")

    pages = getattr(result, "pages", []) or []
    print(f"返回 {len(pages)} 页")
    for index, page in enumerate(pages):
        blocks = getattr(page, "blocks", None) or []
        text = "\n".join(t for t in (_block_text(b) for b in blocks) if t)
        cleaned = repair_text(text)
        printable = sum(1 for ch in cleaned if ch.isprintable() or ch.isspace())
        ratio = printable / max(len(cleaned), 1)
        kind = classify_page(cleaned)
        print(f"\n--- p{int(getattr(page, 'page_idx', index)) + 1} [{kind}] "
              f"{len(cleaned)} 字 / 可打印 {ratio:.0%} ---")
        print("  ", cleaned[:300].replace("\n", " / "))

    if not pages:
        print("!! 解析结果为空，别入库")


if __name__ == "__main__":
    main()
