"""文档类型识别与解析后端决策（纯逻辑，不碰外部工具）。

流程（v3 文档 A.2 节）：
1. 看扩展名 → 决定怎么归一到 PDF/图片
2. 看文本层密度 → 决定 MinerU 用 pipeline 还是 vlm backend
3. 只有过不了质量校验的页面才交 qwen3-max 兜底

这些判断全部是纯函数，可以先于任何解析器落地、先于服务器连通就测通。
"""

from __future__ import annotations

from dataclasses import dataclass

from app.core.errors import AppError, ErrorCode

# 扩展名 → 大类
EXT_KIND: dict[str, str] = {
    ".pdf": "pdf",
    ".epub": "ebook",
    ".mobi": "ebook",
    ".azw3": "ebook",
    ".docx": "office",
    ".doc": "office",
    ".pptx": "office",
    ".ppt": "office",
    ".md": "text",
    ".txt": "text",
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".webp": "image",
    ".bmp": "image",
}

SUPPORTED_EXTENSIONS = sorted(EXT_KIND)

# 归一方式
NORMALIZER_BY_KIND: dict[str, str] = {
    "pdf": "none",  # 本来就是 PDF
    "ebook": "ebook_convert",  # Calibre
    "office": "libreoffice",
    "text": "none",  # 直读，不进 MinerU
    "image": "none",  # 本身就是图片，直接进 MinerU
}

# 文本层密度阈值（字符/页）
PIPELINE_THRESHOLD = 100
SCAN_THRESHOLD = 20


@dataclass
class FormatPlan:
    extension: str
    kind: str
    normalizer: str
    needs_mineru: bool
    is_text_direct: bool


def plan_for(filename: str) -> FormatPlan:
    """根据文件名给出归一与解析计划。"""
    lowered = (filename or "").lower()
    dot = lowered.rfind(".")
    extension = lowered[dot:] if dot != -1 else ""

    kind = EXT_KIND.get(extension)
    if kind is None:
        raise AppError(
            ErrorCode.FILE_TYPE_UNSUPPORTED,
            f"不支持的文件类型：{extension or '未知'}。支持：{'、'.join(SUPPORTED_EXTENSIONS)}",
        )

    normalizer = NORMALIZER_BY_KIND[kind]
    return FormatPlan(
        extension=extension,
        kind=kind,
        normalizer=normalizer,
        needs_mineru=kind in ("pdf", "ebook", "office", "image"),
        is_text_direct=kind == "text",
    )


def text_layer_stats(pages: list[str]) -> dict[str, float]:
    """统计文本层密度。pages 是每页抽到的纯文本。"""
    if not pages:
        return {"chars_per_page": 0.0, "empty_ratio": 1.0, "pages": 0}

    counts = [len((page or "").strip()) for page in pages]
    total = sum(counts)
    empty = sum(1 for count in counts if count == 0)
    return {
        "chars_per_page": round(total / len(pages), 2),
        "empty_ratio": round(empty / len(pages), 3),
        "pages": len(pages),
    }


def detect_backend(pages: list[str], *, threshold: int = PIPELINE_THRESHOLD) -> str:
    """决定 MinerU backend。

    chars_per_page > threshold            -> pipeline（有文本层，快速通道）
    SCAN_THRESHOLD < x <= threshold       -> vlm（混合版，部分页是图）
    x <= SCAN_THRESHOLD                   -> vlm（扫描版）
    """
    stats = text_layer_stats(pages)
    if stats["chars_per_page"] > threshold:
        return "pipeline"
    return "vlm"


def needs_vl_fallback(page_text: str, *, min_chars: int = 50) -> bool:
    """单页是否需要 qwen3-max 兜底：解析出来几乎没字，说明这一页没解出来。"""
    return len((page_text or "").strip()) < min_chars


def estimate_cost(pages: int, vl_ratio: float) -> dict[str, float]:
    """粗估解析成本结构，用于判断 qwen3-max 兜底会不会失控。

    vl_ratio 是走 vlm backend 的页面比例，成本相对 pipeline 记 1.0 基准。
    """
    if pages <= 0:
        return {"pages": 0, "pipeline_pages": 0, "vlm_pages": 0, "relative_cost": 0.0}

    vlm_pages = int(pages * max(0.0, min(1.0, vl_ratio)))
    pipeline_pages = pages - vlm_pages
    # vlm 单页成本约为 pipeline 的 8 倍（视觉编码 + 更长输入），经验系数
    relative = pipeline_pages * 1.0 + vlm_pages * 8.0
    return {
        "pages": pages,
        "pipeline_pages": pipeline_pages,
        "vlm_pages": vlm_pages,
        "relative_cost": round(relative, 1),
    }
