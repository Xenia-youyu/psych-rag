"""页脚广告清洗。

实测数据里《实验心理学》33% 的块带盗版水印，所以这里的用例要覆盖
"该删的" 和 "绝对不能删的"（正文里的网址、文献引用格式）。
"""

from __future__ import annotations

from app.ingest.cleaner import is_watermark_only, strip_watermark
from app.ingest.pipeline import ChunkPlan, clean_watermarks

BODY = (
    "实验者效应是指实验者的期望对实验结果产生的影响。"
    "为控制这种效应，可以采用双盲设计，让主试和被试都不知道实验条件。"
    "此外还应注意标准化指导语、统一施测程序，并在正式实验前进行充分的预实验，"
    "以排除环境因素与被试状态的干扰。"
)


class TestStripWatermark:
    def test_removes_footer_ad(self) -> None:
        text = (
            f"{BODY}\n"
            "这里  提供  最专业的  心理学考研辅导\n"
            "资料下载：bbs.libidos.cn\n"
            "心理学考研答疑QQ： 2253446843"
        )
        cleaned, removed = strip_watermark(text)

        assert removed == 3
        assert "bbs.libidos.cn" not in cleaned
        assert "QQ" not in cleaned
        assert "实验者效应" in cleaned

    def test_keeps_long_body_text_with_url(self) -> None:
        """写作规范里讲电子文献引用格式，含 http 但是正文，必须留下。"""
        text = (
            "h.电子文献  标引项顺序号  作者.题名.出处或可获得地址，"
            "发表或更新日期/引用日期（任选）. 例如 http://www.example.com/paper.pdf"
        )
        cleaned, removed = strip_watermark(text)

        assert removed == 0
        assert cleaned == text

    def test_removes_short_wechat_line(self) -> None:
        text = f"{BODY}\n公众号：心理学考研"
        cleaned, removed = strip_watermark(text)

        assert removed == 1
        assert "公众号" not in cleaned

    def test_collapses_blank_lines(self) -> None:
        text = f"{BODY}\n\n\n资料下载：bbs.x.cn"
        cleaned, _ = strip_watermark(text)

        assert "\n\n\n" not in cleaned

    def test_no_watermark_returns_same_object(self) -> None:
        cleaned, removed = strip_watermark(BODY)
        assert removed == 0
        assert cleaned == BODY

    def test_watermark_only_block(self) -> None:
        text = "这里提供最专业的心理学考研辅导\n资料下载：bbs.libidos.cn"
        assert is_watermark_only(text) is True

    def test_real_content_is_not_watermark_only(self) -> None:
        assert is_watermark_only(BODY) is False


class TestCleanPlan:
    def test_cleans_children_and_reports(self) -> None:
        plan = ChunkPlan(
            parents=[{"chunk_id": "p1", "content": BODY * 3}],
            children=[
                {"chunk_id": "c1", "content": f"{BODY}\n资料下载：bbs.libidos.cn"},
                {"chunk_id": "c2", "content": BODY},
            ],
        )
        cleaned, touched, dropped = clean_watermarks(plan)

        assert touched == 1
        assert dropped == 0
        assert len(cleaned.children) == 2
        assert "bbs.libidos.cn" not in cleaned.children[0]["content"]

    def test_drops_block_that_was_only_watermark(self) -> None:
        plan = ChunkPlan(
            parents=[],
            children=[{"chunk_id": "c1", "content": "资料下载：bbs.libidos.cn\nQQ：12345"}],
        )
        cleaned, _touched, dropped = clean_watermarks(plan)

        assert dropped == 1
        assert cleaned.children == []
