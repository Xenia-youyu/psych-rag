"""定位「Unsupported warp settings」：逐页试渲染，找出崩在哪一页、它长什么样。

MinerU 内部用 pymupdf 把 PDF 页面转位图，遇到异常的 /Rotate 或 MediaBox 会抛这个错。
不先定位就改代码，等于猜。
"""

from __future__ import annotations

import sys

import pymupdf

PATH = sys.argv[1] if len(sys.argv) > 1 else (
    r"D:\心理学文档\普通心理学\心理学导论 梁宁建 上海教育出版社 第2版.pdf"
)
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else 40

doc = pymupdf.open(PATH)
print(f"文件：{PATH}")
print(f"总页数 {doc.page_count} | 元数据 rotation={doc.metadata.get('rotation')}")
print(f"前 {LIMIT} 页逐页试渲染：\n")

bad = []
for i in range(min(doc.page_count, LIMIT)):
    page = doc[i]
    rotation = page.rotation
    rect = page.rect
    try:
        page.get_pixmap(matrix=pymupdf.Matrix(2, 2))
        status = "ok"
    except Exception as exc:  # noqa: BLE001
        status = f"FAIL {type(exc).__name__}: {exc}"
        bad.append((i + 1, rotation, rect))
    if i < 6 or "FAIL" in status:
        print(f"  p{i+1:>4} rot={rotation:>4} rect={rect.width:.0f}x{rect.height:.0f} {status}")

print(f"\n失败 {len(bad)} 页" + (f"：{[b[0] for b in bad[:10]]}" if bad else ""))
doc.close()
