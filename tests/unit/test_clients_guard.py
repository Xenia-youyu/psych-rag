"""基础设施客户端的可测部分。

Milvus / 重排模型需要真实服务与模型权重，绝大部分逻辑只能等环境就绪后验证。
但有两处纯逻辑必须提前测死，因为它们错了不会报错，只会悄悄返回错结果：
1. 多知识库隔离的过滤表达式 —— 写错了就是跨库串数据
2. 空文档列表 —— 不提前返回会去加载模型，浪费几秒还要吃显存
"""

from __future__ import annotations

import pytest

from app.clients.milvus_client import MilvusIndex
from app.clients.rerank_client import BGEReranker


class TestKbIsolation:
    """kb_id 过滤表达式是整个多租户隔离的最后一环，必须精确。"""

    def test_single_kb(self) -> None:
        index = MilvusIndex()
        assert index._kb_expr(["kb1"]) == 'kb_id in ["kb1"]'

    def test_multiple_kb(self) -> None:
        index = MilvusIndex()
        assert index._kb_expr(["kb1", "kb2"]) == 'kb_id in ["kb1", "kb2"]'

    def test_empty_kb_never_matches(self) -> None:
        """没选知识库时必须返回永不匹配的表达式，而不是空字符串。

        空字符串在 Milvus 里等于"不过滤"——那会把全库数据都捞出来。
        """
        index = MilvusIndex()
        expr = index._kb_expr([])
        assert "__none__" in expr
        assert expr != ""

    def test_blank_ids_filtered(self) -> None:
        index = MilvusIndex()
        assert index._kb_expr(["", "kb1", ""]) == 'kb_id in ["kb1"]'


class TestSearchGuard:
    async def test_no_kb_skips_query(self) -> None:
        index = MilvusIndex()
        # 不应尝试连接 Milvus
        assert await index.search(kb_ids=[], dense=[0.1] * 1024) == []


class TestRerankerEdge:
    async def test_empty_documents_short_circuits(self) -> None:
        """空列表必须直接返回，不能去加载模型（加载要几秒 + 占显存）。"""
        reranker = BGEReranker(model_path="/nonexistent/path")
        assert await reranker.score("问题", []) == []

    @pytest.mark.parametrize("documents", [[], []])
    async def test_repeated_empty_calls_are_cheap(self, documents: list[str]) -> None:
        reranker = BGEReranker(model_path="/nonexistent/path")
        for _ in range(3):
            assert await reranker.score("问题", documents) == []
