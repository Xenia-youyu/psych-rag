"""切块算法单元测试。

覆盖：空输入、短文本、长文本、中文标点优先、重叠、最小块合并、参数校验、父子块、按页切分。
这些用例不需要数据库，必须秒级跑完。
"""

from __future__ import annotations

import pytest

from app.ingest.chunker import (
    DEFAULT_SEPARATORS,
    split_by_pages,
    split_parent_child,
    split_text,
)

SENTENCE = "操作性条件作用强调行为的结果对行为本身的反作用，这是斯金纳理论的核心主张。"


def build_text(sentences: int) -> str:
    return "".join(SENTENCE for _ in range(sentences))


class TestSplitText:
    def test_empty_input_returns_empty_list(self) -> None:
        assert split_text("") == []
        assert split_text("   \n\t ") == []

    def test_short_text_single_chunk(self) -> None:
        text = "信度是指测验结果的稳定性与一致性程度。"
        chunks = split_text(text, size=800, overlap=0)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_each_chunk_within_size(self) -> None:
        text = build_text(40)
        size, overlap = 300, 0
        chunks = split_text(text, size=size, overlap=overlap, min_size=1)

        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= size

    def test_chunks_with_overlap_do_not_exceed_size_plus_overlap(self) -> None:
        text = build_text(40)
        size, overlap = 300, 50
        chunks = split_text(text, size=size, overlap=overlap, min_size=1)

        for chunk in chunks:
            assert len(chunk) <= size + overlap

    def test_prefers_chinese_sentence_boundary(self) -> None:
        """切块应优先落在中文句末标点上，不把句子拦腰截断。"""
        text = build_text(20)
        chunks = split_text(text, size=200, overlap=0, min_size=1)

        for chunk in chunks[:-1]:
            assert chunk[-1] in "。！？；", f"块未在句末标点结束：...{chunk[-20:]}"

    def test_overlap_prefixes_previous_tail(self) -> None:
        text = build_text(30)
        size, overlap = 250, 40
        base = split_text(text, size=size, overlap=0, min_size=1)
        chunks = split_text(text, size=size, overlap=overlap, min_size=1)

        assert len(chunks) == len(base)
        assert chunks[0] == base[0]
        for index in range(1, len(chunks)):
            assert chunks[index].startswith(base[index - 1][-overlap:])

    def test_min_size_does_not_break_size_limit(self) -> None:
        """碎块合并不能以突破 size 为代价（回归测试：初版会把块撑到 363 字）。"""
        text = build_text(10) + "。"
        chunks = split_text(text, size=300, overlap=0, min_size=280)

        assert len(chunks) == 2
        for chunk in chunks:
            assert len(chunk) <= int(300 * 1.10)

    def test_min_size_merges_when_within_tolerance(self) -> None:
        """前块有余量时，尾部落进容差内应被合并。"""
        text = "甲" * 250 + "\n\n" + "乙" * 50
        chunks = split_text(text, size=300, overlap=0, min_size=200)

        assert len(chunks) == 1
        # 250 + 分隔符 2 + 50 = 302，落进 size×1.10 容差内被合并
        assert len(chunks[0]) == 302

    def test_custom_separators(self) -> None:
        text = "第一部分||第二部分||第三部分"
        chunks = split_text(text, size=7, overlap=0, min_size=1, separators=["||", ""])
        assert chunks == ["第一部分||", "第二部分||", "第三部分"]

    def test_no_content_loss(self) -> None:
        """所有块按顺序拼接（去重叠后）应覆盖原文全部字符。"""
        text = build_text(25)
        chunks = split_text(text, size=250, overlap=0, min_size=1)
        assert "".join(chunks) == text

    @pytest.mark.parametrize(
        ("kwargs", "reason"),
        [
            ({"size": 0}, "size 为 0"),
            ({"size": -1}, "size 为负"),
            ({"overlap": -5}, "overlap 为负"),
            ({"size": 100, "overlap": 100}, "overlap 等于 size"),
            ({"size": 100, "overlap": 200}, "overlap 大于 size"),
        ],
    )
    def test_invalid_params_raise(self, kwargs: dict, reason: str) -> None:
        with pytest.raises(ValueError):
            split_text("内容" * 100, **kwargs)
        assert reason  # 便于失败时定位

    def test_default_separators_include_chinese_punctuation(self) -> None:
        for mark in ("。", "！", "？", "；", "，", "、"):
            assert mark in DEFAULT_SEPARATORS


class TestSplitParentChild:
    def test_parent_respects_size_and_children_cover_parent(self) -> None:
        text = build_text(80)
        parents = split_parent_child(
            text,
            parent_size=800,
            parent_min_size=100,
            child_size=300,
            child_min_size=50,
            child_overlap=30,
        )

        assert len(parents) > 1
        for parent in parents:
            assert len(parent.content) <= 800
            assert parent.children, "父块必须有子块"
            for child in parent.children:
                assert len(child) <= 300 + 30
            # 子块拼接（去掉重叠前缀）应覆盖父块全部内容
            joined = "".join(
                child if index == 0 else child[30:]
                for index, child in enumerate(parent.children)
            )
            assert joined == parent.content

    def test_parent_ids_are_sequential(self) -> None:
        parents = split_parent_child(build_text(60), parent_size=600, id_prefix="doc1_")

        assert [p.parent_id for p in parents] == [
            f"doc1_{index:04d}" for index in range(len(parents))
        ]

    def test_short_text_yields_single_parent(self) -> None:
        parents = split_parent_child("短文本内容。", parent_size=1200, child_size=320)
        assert len(parents) == 1
        assert parents[0].content == "短文本内容。"


class TestSplitByPages:
    def test_page_numbers_are_recorded(self) -> None:
        pages = [(1, "第一页内容。" * 20), (2, "第二页内容。" * 20)]
        result = split_by_pages(pages, size=200, overlap=0, min_size=1)

        assert result
        for _text, page_start, page_end in result:
            assert page_start == page_end
            assert page_start in (1, 2)

    def test_pages_are_not_mixed(self) -> None:
        pages = [(1, "甲" * 50), (2, "乙" * 50)]
        result = split_by_pages(pages, size=1000, overlap=0, min_size=1)

        assert len(result) == 2
        assert result[0][0] == "甲" * 50
        assert result[1][0] == "乙" * 50
