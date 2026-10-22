"""扫描语料目录，给出「解析成本 + 概括性」画像。

选书入库前必须先看清两件事：
1. **有没有文本层。** 文字版 PDF 用 PyMuPDF 抽，10 页 1 秒；扫描版得走 OCR，
   一本 500 页的书是几小时。混在一起算总账会严重低估。
2. **是不是知识体系。** 「XX学概论」讲的是学科骨架，「XX考研真题」是题目合集，
   两者都能检索，但只有前者适合当知识底座。

只读不写，不动任何原始文件。
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

SUPPORTED = {".pdf", ".epub", ".mobi", ".azw3", ".doc", ".docx", ".txt", ".md"}

# 概括性强的书名特征
GENERAL_MARKERS = ("概论", "导论", "基础", "原理", "教程", "通识", "纲要", "精要", "教科书")
# 不是知识体系的书名特征
NON_SYSTEM_MARKERS = ("习题", "真题", "考点", "笔记", "速记", "大纲", "押题", "模拟", "答案", "背诵")
# 解析成本高的特征
EXPENSIVE_MARKERS = ("扫描", "扫描版", "影印", "ocr", "OCR")


@dataclass
class Item:
    folder: str
    name: str
    ext: str
    size_mb: float
    pages: int = 0
    chars_per_page: float = 0.0
    has_text_layer: bool = False
    scanned_hint: bool = False
    empty_ratio: float = 0.0
    score: float = 0.0
    reasons: list[str] = field(default_factory=list)


def probe_pdf(path: str, sample_pages: int = 5) -> tuple[int, float, float]:
    """返回 (总页数, 抽样页平均字符数, 抽样页空页占比)。

    第三项必须有：有一类 PDF 只有封面/版权页有文本层，正文全是扫描图。
    只看平均字符数会把它判成"文字版"——实测《人格心理学》368 页里
    只有第 1 页有 2057 字，均值 411 字/页看起来非常健康，
    实际入库只得到 7 个块。
    """
    try:
        import pymupdf
    except ImportError:
        return 0, 0.0, 1.0

    try:
        doc = pymupdf.open(path)
    except Exception:
        return 0, 0.0, 1.0

    try:
        total = doc.page_count
        step = max(1, total // sample_pages)
        sampled, chars, empties = 0, 0, 0
        for index in range(0, min(total, step * sample_pages), step):
            count = len(doc[index].get_text().strip())
            chars += count
            if count < 20:
                empties += 1
            sampled += 1
        if not sampled:
            return total, 0.0, 1.0
        return total, chars / sampled, empties / sampled
    finally:
        doc.close()


def scan(root: str) -> list[Item]:
    items: list[Item] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for filename in filenames:
            ext = Path(filename).suffix.lower()
            if ext not in SUPPORTED:
                continue

            full = os.path.join(dirpath, filename)
            item = Item(
                folder=os.path.relpath(dirpath, root),
                name=filename,
                ext=ext,
                size_mb=round(os.path.getsize(full) / 1024 / 1024, 1),
            )

            lowered = filename.lower()
            item.scanned_hint = any(m in lowered for m in EXPENSIVE_MARKERS)

            if ext == ".pdf":
                item.pages, item.chars_per_page, item.empty_ratio = probe_pdf(full)
                # 平均值高但空页多 = 只有封面/版权页有字，正文仍是扫描件
                item.has_text_layer = item.chars_per_page >= 100 and item.empty_ratio < 0.6
            elif ext in (".epub", ".mobi", ".azw3"):
                # 需要先转格式，成本另算
                item.has_text_layer = True
                item.reasons.append("需先经 Calibre 转换")
            else:
                item.has_text_layer = True

            # 打分：概括性为主，成本为辅
            score = 0.0
            if any(m in filename for m in GENERAL_MARKERS):
                score += 3
                item.reasons.append("书名是学科概括类")
            if any(m in filename for m in NON_SYSTEM_MARKERS):
                score -= 3
                item.reasons.append("不是知识体系（题集/笔记类）")
            if item.has_text_layer:
                score += 2
                item.reasons.append("有文本层")
            else:
                score -= 2
                item.reasons.append("无文本层，需 OCR")
            if 200 <= item.pages <= 800:
                score += 1
            elif item.pages > 1200:
                score -= 1
                item.reasons.append(f"页数偏多（{item.pages}）")
            if item.scanned_hint:
                score -= 1
            item.score = score
            items.append(item)
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description="扫描语料，产出解析成本与概括性画像")
    parser.add_argument("--root", default=r"D:\心理学文档")
    parser.add_argument("--out", default="corpus_scan.json")
    parser.add_argument("--top", type=int, default=3, help="每个分类列出前几名")
    args = parser.parse_args()

    items = scan(args.root)
    with open(args.out, "w", encoding="utf-8") as sink:
        json.dump([asdict(item) for item in items], sink, ensure_ascii=False, indent=2)

    by_folder: dict[str, list[Item]] = {}
    for item in items:
        by_folder.setdefault(item.folder, []).append(item)

    total_pages = sum(item.pages for item in items)
    ocr_pages = sum(item.pages for item in items if item.ext == ".pdf" and not item.has_text_layer)
    print(f"共 {len(items)} 个文件，PDF 总页数 {total_pages}，其中需 OCR 约 {ocr_pages} 页")
    print(f"（{ocr_pages / max(total_pages, 1):.0%} 的页没有文本层）\n")

    for folder in sorted(by_folder):
        group = sorted(by_folder[folder], key=lambda x: -x.score)
        print(f"\n【{folder}】{len(group)} 个")
        for item in group[: args.top]:
            flag = "文字" if item.has_text_layer else "扫描"
            print(
                f"  {item.score:+.1f}  {flag}  {item.pages:>5}页  "
                f"{item.chars_per_page:>6.0f}字/页  {item.name[:52]}"
            )
    print(f"\n明细已写入 {args.out}")


if __name__ == "__main__":
    main()
