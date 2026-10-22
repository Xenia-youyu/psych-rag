"""检索融合单元测试。

覆盖：RRF 融合顺序与分数、k 的影响、父块去重、阈值过滤、上下文预算裁剪、上下文拼装格式。
"""

from __future__ import annotations

import pytest

from app.retrieve.fusion import (
    Hit,
    build_context_block,
    dedupe_by_parent,
    filter_by_threshold,
    rrf_fuse,
    select_context,
)


def hit(chunk_id: str, score: float = 0.5, parent_id: str | None = None, content: str = "") -> Hit:
    return Hit(chunk_id=chunk_id, score=score, parent_id=parent_id, content=content)


class TestRRF:
    def test_single_list_keeps_order(self) -> None:
        dense = [hit("a"), hit("b"), hit("c")]
        fused = rrf_fuse({"dense": dense}, k=60)

        assert [h.chunk_id for h in fused] == ["a", "b", "c"]
        assert fused[0].score == pytest.approx(1 / 61)
        assert fused[1].score == pytest.approx(1 / 62)

    def test_item_in_both_lists_wins(self) -> None:
        dense = [hit("a"), hit("b")]
        sparse = [hit("c"), hit("a")]
        fused = rrf_fuse({"dense": dense, "sparse": sparse}, k=60)

        # a 在两路分别排第 1 与第 2，分数累加后应排第一
        assert fused[0].chunk_id == "a"
        assert fused[0].score == pytest.approx(1 / 61 + 1 / 62)
        assert {h.chunk_id for h in fused} == {"a", "b", "c"}

    def test_larger_k_flattens_score_gap(self) -> None:
        dense = [hit("a"), hit("b")]
        small = rrf_fuse({"dense": dense}, k=1)
        large = rrf_fuse({"dense": dense}, k=100)

        gap_small = small[0].score - small[1].score
        gap_large = large[0].score - large[1].score
        assert gap_large < gap_small

    def test_top_n_truncates(self) -> None:
        fused = rrf_fuse({"dense": [hit(str(i)) for i in range(10)]}, k=60, top_n=3)
        assert len(fused) == 3

    def test_empty_input(self) -> None:
        assert rrf_fuse({}, k=60) == []
        assert rrf_fuse({"dense": []}, k=60) == []

    def test_invalid_k_and_top_n(self) -> None:
        with pytest.raises(ValueError):
            rrf_fuse({"dense": [hit("a")]}, k=0)
        with pytest.raises(ValueError):
            rrf_fuse({"dense": [hit("a")]}, k=60, top_n=0)

    def test_metadata_preserved_from_first_seen_list(self) -> None:
        dense = [Hit(chunk_id="a", score=0.9, parent_id="p1", content="正文", meta={"page_start": 3})]
        fused = rrf_fuse({"dense": dense, "sparse": [hit("a")]}, k=60)

        assert fused[0].parent_id == "p1"
        assert fused[0].content == "正文"
        assert fused[0].meta["page_start"] == 3


class TestDedupe:
    def test_keeps_highest_score_per_parent(self) -> None:
        hits = [
            hit("c1", 0.4, "p1"),
            hit("c2", 0.8, "p1"),
            hit("c3", 0.3, "p2"),
        ]
        result = dedupe_by_parent(hits)

        assert [h.chunk_id for h in result] == ["c2", "c3"]

    def test_none_parent_falls_back_to_own_id(self) -> None:
        hits = [hit("c1", 0.4), hit("c2", 0.3)]
        result = dedupe_by_parent(hits)
        assert len(result) == 2

    def test_first_appearance_order_preserved(self) -> None:
        hits = [hit("c1", 0.9, "p1"), hit("c2", 0.1, "p2"), hit("c3", 0.5, "p1")]
        result = dedupe_by_parent(hits)
        assert [h.chunk_id for h in result] == ["c1", "c2"]


class TestFilterAndSelect:
    def test_threshold_filters_low_scores(self) -> None:
        hits = [hit("a", 0.9), hit("b", 0.1), hit("c", 0.15)]
        result = filter_by_threshold(hits, 0.15)
        assert [h.chunk_id for h in result] == ["a", "c"]

    def test_zero_threshold_disables_filter(self) -> None:
        hits = [hit("a", 0.0), hit("b", 0.0)]
        assert len(filter_by_threshold(hits, 0)) == 2

    def test_select_context_applies_all_rules(self) -> None:
        hits = [
            hit("c1", 0.9, "p1", "甲" * 100),
            hit("c2", 0.8, "p1", "乙" * 100),  # 同父块，应被去重
            hit("c3", 0.7, "p2", "丙" * 100),
            hit("c4", 0.05, "p3", "丁" * 100),  # 低于阈值
            hit("c5", 0.6, "p4", "戊" * 100),
        ]
        selected = select_context(hits, top_n=2, threshold=0.15, max_chars=6000)

        assert [h.chunk_id for h in selected] == ["c1", "c3"]

    def test_max_chars_budget(self) -> None:
        hits = [hit(f"c{i}", 0.9 - i * 0.1, f"p{i}", "字" * 100) for i in range(5)]
        selected = select_context(hits, top_n=5, threshold=0, max_chars=250)

        # 第一条必留（即使超预算），之后放不下就停
        assert len(selected) == 2

    def test_build_context_block_has_numbered_sources(self) -> None:
        hits = [
            Hit(chunk_id="c1", score=0.9, content="负强化增加行为。", meta={"source_title": "普通心理学", "page_start": 412}),
            Hit(chunk_id="c2", score=0.8, content="惩罚减少行为。", meta={"source_title": "教育心理学"}),
        ]
        block = build_context_block(hits)

        assert "[S1] 普通心理学 第412页" in block
        assert "[S2] 教育心理学" in block
        assert "负强化增加行为。" in block
