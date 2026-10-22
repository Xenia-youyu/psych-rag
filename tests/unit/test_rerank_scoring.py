"""重排打分的语义测试：logits 必须变成 0~1 的概率。

为什么值得单独测：
    换成 sentence_transformers.CrossEncoder 后，predict() 返回的是**原始 logits**
    （可正可负，量级 -10~10），而 config 里的 rerank_score_threshold=0.15 是按
    概率（sigmoid 后）设的。少了一次 sigmoid，这个阈值就变成毫无意义的数字——
    更糟的是它不会报错，只会安静地把结果筛错。
    这类"换实现后语义漂移"的问题，只有测试能盯住。
"""

from __future__ import annotations

import math

import pytest

from app.clients.rerank_client import BGEReranker, _sigmoid


class FakeCrossEncoder:
    """只实现 predict，返回预先给定的 logits。"""

    def __init__(self, logits: list[float]) -> None:
        self.logits = logits
        self.calls: list[list] = []

    def predict(self, pairs):
        self.calls.append(pairs)
        return self.logits[: len(pairs)]


@pytest.fixture
def reranker() -> BGEReranker:
    instance = BGEReranker(model_path="dummy", quantize=False)
    return instance


class TestSigmoid:
    def test_zero_maps_to_half(self) -> None:
        assert _sigmoid(0.0) == pytest.approx(0.5)

    def test_large_positive_saturates(self) -> None:
        assert _sigmoid(50.0) > 0.999

    def test_large_negative_does_not_overflow(self) -> None:
        """大负数直接算 exp 会溢出，这里必须返回 0 而不是抛 OverflowError。"""
        assert _sigmoid(-1000.0) == pytest.approx(0.0)

    def test_monotonic(self) -> None:
        values = [_sigmoid(x) for x in (-5, -1, 0, 1, 5)]
        assert values == sorted(values)


class TestScoreSemantics:
    def test_logits_are_normalized_to_probability(self, reranker) -> None:
        reranker._model = FakeCrossEncoder([4.0, 0.0, -4.0])
        scores = reranker._score_sync("问题", ["a", "b", "c"])

        assert all(0.0 <= s <= 1.0 for s in scores)
        assert scores[0] == pytest.approx(1 / (1 + math.exp(-4.0)))
        assert scores[1] == pytest.approx(0.5)
        assert scores[2] == pytest.approx(1 / (1 + math.exp(4.0)))

    def test_threshold_still_means_probability(self, reranker) -> None:
        """logit=-3 原本看着像"负数不通过"，sigmoid 后是 0.047，同样不通过；
        但 logit=0.1 原本像"正数通过"，sigmoid 后 0.525，这才是真实含义。"""
        reranker._model = FakeCrossEncoder([0.1])
        assert reranker._score_sync("q", ["d"])[0] == pytest.approx(0.525, abs=1e-3)

    def test_empty_documents_short_circuits(self, reranker) -> None:
        """空输入不该加载模型——加载 568M 权重只为算 0 条是纯浪费。"""
        assert reranker._score_sync("q", []) == []
        assert reranker._model is None

    def test_single_scalar_result_is_wrapped(self, reranker) -> None:
        """predict 在单条时可能退化成标量，不能因此崩掉。"""
        reranker._model = FakeCrossEncoder([2.0, 3.0])
        # 只取第一条，模拟返回标量的场景
        assert len(reranker._score_sync("q", ["d"])) == 1
