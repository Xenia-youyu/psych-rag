"""文档解析器：MinerU 走复杂版面，PyMuPDF / python-docx 走简单格式。

分工原则（见 config.ini 的 [parser]）：
- PDF / 图片 → MinerU（版面分析 + OCR + 表格公式）。电子版也不亏：
  它有目录结构与阅读顺序，PyMuPDF 只有裸文本流，双栏教材会串行。
- DOCX / TXT / MD → 直接读。这些格式本身就是结构化的，上 MinerU 是浪费。

关于 MinerU 的 block 结构：
    4.x 的 PageBlock 是 13 种类型（文本/公式/表格/图片/列表…）的判别联合，
    每种字段都不同。这里不逐类硬编码，而是按"能拿到什么就要什么"的顺序提取，
    并对拿不到内容的 block 静默跳过。理由是：MinerU 小版本升级经常改字段名，
    硬编码 13 个分支的代码，三个月后没人敢动。
    验证手段不靠这里，靠 scripts/verify_mineru.py 打每页字符数——
    某页突然变成 0 字，说明这个版本的 block 结构变了，该去看文档了。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from app.core.logging import get_logger
from app.ingest.base import ParsedDoc, ParsedPage

logger = get_logger(__name__)

# 提取 block 正文时依次尝试的字段名
_TEXT_ATTRS = ("text", "content", "markdown", "latex", "html")

# MinerU 输出的 markdown 会把图片内嵌成 data URI
_DATA_URI_IMG = re.compile(r"!\[[^\]]*\]\(data:image/[^)]*\)", re.S)
_BARE_BASE64 = re.compile(r"[A-Za-z0-9+/]{200,}={0,2}")


def strip_data_uri(text: str) -> str:
    """去掉 markdown 内嵌的 base64 图片，只留文字。"""
    if not text:
        return text
    cleaned = _DATA_URI_IMG.sub("", text)
    cleaned = _BARE_BASE64.sub("", cleaned)
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


def _block_text(block: Any) -> str:
    """从一个 block 里尽力取出正文。取不到就返回空串，绝不抛错。

    MinerU 4.x 把文字放在 `block.content` 的 span 列表里，每个 span 才有
    `.content` 字符串；`block.text` 根本不存在。旧代码只认字符串字段，
    结果每个 block 都取到空，最后退化成把整篇 markdown 当一页正文——
    而那篇 markdown 里 99% 是 base64 图片。
    """
    spans = getattr(block, "content", None)
    if isinstance(spans, list) and spans:
        parts = [
            text.strip()
            for span in spans
            for text in [getattr(span, "content", None)]
            if isinstance(text, str) and text.strip()
        ]
        if parts:
            return "".join(parts)

    for attr in _TEXT_ATTRS:
        value = getattr(block, attr, None)
        if isinstance(value, str) and value.strip():
            return value.strip()

    # 表格类 block 常见结构：rows / cells
    rows = getattr(block, "rows", None)
    if isinstance(rows, list) and rows:
        lines: list[str] = []
        for row in rows:
            cells = getattr(row, "cells", None) or row
            if isinstance(cells, (list, tuple)):
                texts = [str(getattr(c, "text", c)).strip() for c in cells]
                lines.append(" | ".join(t for t in texts if t))
        if lines:
            return "\n".join(lines)

    return ""


class MinerUParser:
    """MinerU 4.x 解析器。

    调用一次会做三件事：版面分析、OCR（扫描页）、表格/公式识别，
    产出 Markdown + 每页文本。CPU 上单页约数秒到数十秒，取决于页面复杂度。
    """

    name = "mineru"

    def __init__(self, *, tier: str | None = None, ocr_mode: str | None = None) -> None:
        from app.config import settings

        conf = settings.parser
        self.tier = tier or conf.mineru_tier
        self.ocr_mode = ocr_mode or conf.mineru_ocr_mode

    def parse(
        self,
        path: str,
        *,
        max_pages: int | None = None,
        start_page: int | None = None,
        end_page: int | None = None,
    ) -> ParsedDoc:
        """start_page/end_page 是 1 基闭区间，用于分段入库。

        为什么需要分段：MinerU 的 OCR 后处理会把检测框旋转裁剪，
        遇到几何退化的框（宽或高为 0）就抛 ValueError，
        **一个坏框会让整本书返回 0 页**。分段落库能把损失限制在一段内。
        """
        try:
            from mineru.parser import parse as mineru_parse
        except ImportError as exc:  # pragma: no cover - 环境问题
            raise RuntimeError(
                "缺少 mineru，先执行：uv pip install mineru && "
                "mineru-models-download --tier basic --small-backend onnx"
            ) from exc

        if start_page or end_page:
            page_range = f"{start_page or 1}-{end_page or max_pages or 100000}"
        elif max_pages:
            page_range = f"1-{max_pages}"
        else:
            page_range = ""
        logger.info(
            "parser.mineru.start",
            extra={
                "extra_fields": {
                    "path": path,
                    "tier": self.tier,
                    "ocr_mode": self.ocr_mode,
                    "page_range": page_range or "all",
                }
            },
        )

        result = mineru_parse(
            path,
            tier=self.tier,
            ocr_mode=self.ocr_mode,
            page_range=page_range,
        )
        markdown = result.markdown() or ""
        # markdown 里内嵌的图片是 data URI，动辄几万字，
        # 进了块就是纯噪音（还会吃掉向量预算），先剥掉再用
        markdown = strip_data_uri(markdown)

        pages: list[ParsedPage] = []
        for index, page in enumerate(getattr(result, "pages", []) or []):
            # page_idx 是 0 基，对外统一成 1 基
            page_no = int(getattr(page, "page_idx", index)) + 1
            blocks = getattr(page, "blocks", None) or []
            text = "\n".join(t for t in (_block_text(b) for b in blocks) if t)
            pages.append(ParsedPage(page_no=page_no, text=text))

        # 逐页提取全空但 markdown 有内容 → block 结构和预期不一致。
        # 这时宁可用 markdown 当单页正文，也不能入库一堆空块。
        if markdown and all(not p.text.strip() for p in pages):
            logger.warning(
                "parser.mineru.blocks_empty",
                extra={"extra_fields": {"path": path, "pages": len(pages)}},
            )
            pages = [ParsedPage(page_no=1, text=markdown)]

        doc = ParsedDoc(
            source_path=path,
            parser=self.name,
            pages=pages,
            markdown=markdown,
            meta={"tier": self.tier, "ocr_mode": self.ocr_mode},
        )
        logger.info(
            "parser.mineru.done",
            extra={
                "extra_fields": {
                    "pages": doc.page_count,
                    "chars": doc.total_chars,
                    "markdown_chars": len(markdown),
                }
            },
        )
        return doc


class SimpleParser:
    """PyMuPDF / python-docx / 纯文本。用于电子版 PDF 与结构化文本格式。"""

    name = "simple"

    def parse(
        self,
        path: str,
        *,
        max_pages: int | None = None,
        start_page: int | None = None,
        end_page: int | None = None,
    ) -> ParsedDoc:
        suffix = Path(path).suffix.lower()
        if suffix == ".pdf":
            pages = self._read_pdf(path, max_pages, start_page, end_page)
        elif suffix in (".docx", ".doc"):
            pages = self._read_docx(path)
        else:
            pages = self._read_text(path)

        doc = ParsedDoc(
            source_path=path,
            parser=self.name,
            pages=pages,
            markdown="\n\n".join(p.text for p in pages),
            meta={"suffix": suffix},
        )
        logger.info(
            "parser.simple.done",
            extra={"extra_fields": {"path": path, "pages": doc.page_count, "chars": doc.total_chars}},
        )
        return doc

    def _read_pdf(
        self,
        path: str,
        max_pages: int | None,
        start_page: int | None = None,
        end_page: int | None = None,
    ) -> list[ParsedPage]:
        import pymupdf

        document = pymupdf.open(path)
        try:
            first = (start_page or 1) - 1
            last = end_page or document.page_count
            if max_pages:
                last = min(last, first + max_pages)
            last = min(last, document.page_count)
            pages: list[ParsedPage] = []
            for index in range(max(first, 0), max(last, 0)):
                text = document[index].get_text().strip()
                if text:
                    pages.append(ParsedPage(page_no=index + 1, text=text))
            return pages
        finally:
            document.close()

    def _read_docx(self, path: str) -> list[ParsedPage]:
        import docx

        document = docx.Document(path)
        text = "\n".join(p.text for p in document.paragraphs if p.text.strip())
        return [ParsedPage(page_no=1, text=text)] if text else []

    def _read_text(self, path: str) -> list[ParsedPage]:
        for encoding in ("utf-8", "gbk", "utf-16"):
            try:
                text = Path(path).read_text(encoding=encoding)
            except (UnicodeDecodeError, UnicodeError):
                continue
            return [ParsedPage(page_no=1, text=text)] if text.strip() else []
        raise RuntimeError(f"无法识别编码：{path}")


def pick_parser(filename: str) -> MinerUParser | SimpleParser:
    """按扩展名选解析器。图片一律走 MinerU（它本身就是 OCR）。"""
    suffix = Path(filename).suffix.lower()
    if suffix in {".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}:
        return MinerUParser()
    return SimpleParser()
