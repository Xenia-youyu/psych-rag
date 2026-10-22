"""文档类型识别与解析后端决策测试（纯逻辑，不碰 MinerU）。"""

from __future__ import annotations

import pytest

from app.core.errors import AppError, ErrorCode
from app.ingest.detect import (
    SUPPORTED_EXTENSIONS,
    detect_backend,
    estimate_cost,
    needs_vl_fallback,
    plan_for,
    text_layer_stats,
)

TEXT_PAGE = "操作性条件作用强调行为的结果对行为本身的反作用。" * 6  # 约 150 字
SCAN_PAGE = ""


class TestPlanFor:
    @pytest.mark.parametrize(
        ("filename", "kind", "normalizer", "needs_mineru"),
        [
            ("普通心理学.pdf", "pdf", "none", True),
            ("发展心理学.epub", "ebook", "ebook_convert", True),
            ("红书.MOBI", "ebook", "ebook_convert", True),
            ("讲义.docx", "office", "libreoffice", True),
            ("课件.pptx", "office", "libreoffice", True),
            ("笔记.md", "text", "none", False),
            ("清单.txt", "text", "none", False),
            ("扫描页.png", "image", "none", True),
        ],
    )
    def test_kind_and_normalizer(self, filename: str, kind: str, normalizer: str,
                                 needs_mineru: bool) -> None:
        plan = plan_for(filename)
        assert plan.kind == kind
        assert plan.normalizer == normalizer
        assert plan.needs_mineru == needs_mineru

    def test_case_insensitive(self) -> None:
        assert plan_for("BOOK.PDF").kind == "pdf"

    def test_text_direct_flag(self) -> None:
        assert plan_for("note.md").is_text_direct is True
        assert plan_for("book.pdf").is_text_direct is False

    @pytest.mark.parametrize("filename", ["病毒.exe", "压缩包.rar", "无扩展名", ""])
    def test_unsupported_raises(self, filename: str) -> None:
        with pytest.raises(AppError) as exc:
            plan_for(filename)
        assert exc.value.code == ErrorCode.FILE_TYPE_UNSUPPORTED

    def test_error_message_lists_supported(self) -> None:
        with pytest.raises(AppError) as exc:
            plan_for("a.exe")
        assert ".pdf" in str(exc.value)
        assert plan_for.__name__ in dir() or SUPPORTED_EXTENSIONS


class TestTextLayer:
    def test_stats_on_text_pages(self) -> None:
        stats = text_layer_stats([TEXT_PAGE, TEXT_PAGE])
        assert stats["chars_per_page"] > 100
        assert stats["empty_ratio"] == 0.0
        assert stats["pages"] == 2

    def test_stats_on_scan_pages(self) -> None:
        stats = text_layer_stats([SCAN_PAGE, SCAN_PAGE, "几个字"])
        assert stats["empty_ratio"] > 0.6
        assert stats["chars_per_page"] < 20

    def test_stats_on_empty_input(self) -> None:
        stats = text_layer_stats([])
        assert stats["pages"] == 0
        assert stats["empty_ratio"] == 1.0


class TestDetectBackend:
    def test_text_layer_uses_pipeline(self) -> None:
        assert detect_backend([TEXT_PAGE] * 5) == "pipeline"

    def test_scanned_uses_vlm(self) -> None:
        assert detect_backend([SCAN_PAGE] * 5) == "vlm"

    def test_mixed_uses_vlm(self) -> None:
        """混合版按最低档走 vlm，宁可慢也不丢页面。"""
        pages = [TEXT_PAGE, SCAN_PAGE, "短", SCAN_PAGE]
        assert detect_backend(pages) == "vlm"

    def test_custom_threshold(self) -> None:
        pages = ["甲" * 60] * 4
        assert detect_backend(pages, threshold=50) == "pipeline"
        assert detect_backend(pages, threshold=200) == "vlm"

    def test_empty_pages_default_to_vlm(self) -> None:
        assert detect_backend([]) == "vlm"


class TestVlFallback:
    def test_empty_page_needs_fallback(self) -> None:
        assert needs_vl_fallback("") is True
        assert needs_vl_fallback("   ") is True

    def test_short_page_needs_fallback(self) -> None:
        assert needs_vl_fallback("只有几个字") is True

    def test_normal_page_does_not(self) -> None:
        assert needs_vl_fallback(TEXT_PAGE) is False


class TestCostEstimate:
    def test_all_pipeline(self) -> None:
        est = estimate_cost(100, 0.0)
        assert est["vlm_pages"] == 0
        assert est["relative_cost"] == 100.0

    def test_all_vlm_is_much_dearer(self) -> None:
        est = estimate_cost(100, 1.0)
        assert est["vlm_pages"] == 100
        assert est["relative_cost"] == 800.0

    def test_mixed(self) -> None:
        est = estimate_cost(100, 0.25)
        assert est["vlm_pages"] == 25
        assert est["relative_cost"] == 275.0

    def test_zero_pages(self) -> None:
        assert estimate_cost(0, 0.5)["relative_cost"] == 0.0

    def test_ratio_clamped(self) -> None:
        assert estimate_cost(10, 1.5)["vlm_pages"] == 10
        assert estimate_cost(10, -1)["vlm_pages"] == 0
