"""验证 MinerU 能否真正解析一页 PDF。

为什么单独一个脚本：
    `mineru-kit models verify` 只能证明权重文件齐全，证明不了"能出正确 Markdown"。
    而整条入库链路的入口就是它——这里不通，4 万页数据一页都进不来。

用法：
    .venv/Scripts/python.exe scripts/verify_mineru.py
    .venv/Scripts/python.exe scripts/verify_mineru.py --pdf "某本书.pdf" --page 15
    .venv/Scripts/python.exe scripts/verify_mineru.py --text-layer   # 对比有文本层的页

注意（本机特有）：MinerU 每次运行会在 ~/.mineru/models/.locks 下建文件锁，
释放时要删掉它。如果进程被强制中断，锁文件会残留成坏状态，下一次报
`PermissionError: [WinError 5]`。清理方式：

    rm -rf ~/.mineru/models/.locks
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 有文本层的样书（PyMuPDF 抽取验证过）
DEFAULT_TEXT_PDF = r"D:\心理学文档\心理学教材\实验心理学（杨治良）.pdf"


def main() -> int:
    parser = argparse.ArgumentParser(description="验证 MinerU 解析")
    parser.add_argument("--pdf", default=DEFAULT_TEXT_PDF)
    parser.add_argument("--page", default="29", help="页码，支持 '1-5,8' 形式")
    parser.add_argument("--tier", default="basic", help="basic（ONNX 小模型）或 standard（含 VLM）")
    parser.add_argument("--ocr", default="auto", choices=["auto", "txt", "ocr"])
    parser.add_argument("--head", type=int, default=400, help="打印前 N 字")
    args = parser.parse_args()

    if not Path(args.pdf).exists():
        print(f"文件不存在：{args.pdf}")
        return 1

    from mineru.parser import parse

    print("=" * 68)
    print(f"文件：{Path(args.pdf).name}")
    print(f"页码：{args.page}   档位：{args.tier}   ocr_mode={args.ocr}")
    print("=" * 68)

    started = time.perf_counter()
    try:
        result = parse(args.pdf, tier=args.tier, page_range=args.page, ocr_mode=args.ocr)
    except Exception as exc:  # noqa: BLE001
        print(f"解析失败：{type(exc).__name__}: {exc}")
        print("\n若是 WinError 5 / PermissionError，先清残留锁：")
        print("  rm -rf ~/.mineru/models/.locks")
        return 2

    elapsed = time.perf_counter() - started
    markdown = result.markdown()

    print(f"耗时：{elapsed:.1f}s")
    print(f"页对象：{len(result.pages)} 个")
    print(f"Markdown：{len(markdown)} 字")
    print("-" * 68)
    print(markdown[: args.head])
    print("-" * 68)

    # 简单质量判据：能出正文、不是满屏乱码
    if len(markdown) < 50:
        print("⚠ Markdown 过短，可能这页是纯图或解析失败")
        return 3
    cjk = sum(1 for ch in markdown if "\u4e00" <= ch <= "\u9fff")
    ratio = cjk / max(len(markdown), 1)
    print(f"中文字符占比：{ratio:.1%}")
    if ratio < 0.15:
        print("⚠ 中文占比偏低，检查是否串栏或 OCR 语言设错")
        return 4

    print("✓ 解析可用")
    return 0


if __name__ == "__main__":
    sys.exit(main())
