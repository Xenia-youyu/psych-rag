"""检索编排测试：整条链路用替身跑，不碰 Milvus / 模型 / 数据库。

重点验证那些「只在串联处才会暴露」的问题：
- 重排分与 RRF 分混用会怎样
- 父块回填在重排之前还是之后，结果差别在哪
- 某一环挂掉时是整条链路崩，还是有降级
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from app.config import settings
from app.generate.intent import Intent
from app.retrieve.service import retrieve


@dataclass
class FakeEmbedding:
    dense: list[float]
    sparse: dict[str, float]


class FakeEmbedder:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def encode(self, texts: list[str]) -> list[FakeEmbedding]:
        self.calls.append(texts)
        return [FakeEmbedding(dense=[0.1, 0.2], sparse={"1": 0.5}) for _ in texts]


class BrokenEmbedder:
    async def encode(self, texts):
        raise RuntimeError("model not loaded")


class FakeIndex:
    def __init__(self, rows: list[dict], parents: dict | None = None,
                 fail_on_call: bool = False) -> None:
        self.rows = rows
        self.parents = parents or {}
        self.fail_on_call = fail_on_call
        self.search_calls: list[dict] = []
        self.fetch_calls: list[list[str]] = []

    async def search(self, *, kb_ids, dense=None, sparse=None, topk=None):
        self.search_calls.append({"kb_ids": kb_ids, "dense": dense, "sparse": sparse,
                                  "topk": topk})
        if self.fail_on_call:
            raise RuntimeError("milvus down")
        return self.rows[: (topk or 10)]

    async def fetch_chunks(self, chunk_ids):
        self.fetch_calls.append(list(chunk_ids))
        return {cid: self.parents[cid] for cid in chunk_ids if cid in self.parents}


class FakeReranker:
    def __init__(self, scores: list[float]) -> None:
        self.scores = scores
        self.calls: list[tuple[str, list[str]]] = []

    async def score(self, query: str, documents: list[str]) -> list[float]:
        self.calls.append((query, documents))
        return self.scores[: len(documents)]


class BrokenReranker:
    async def score(self, query, documents):
        raise RuntimeError("reranker down")


class FakeLLM:
    def __init__(self, json_replies: list[dict] | None = None) -> None:
        self.json_replies = list(json_replies or [])

    async def chat(self, messages, **kwargs):
        class R:
            text = "假设答案文本"

        return R()

    async def chat_json(self, messages, **kwargs):
        return self.json_replies.pop(0) if self.json_replies else {}


def row(chunk_id: str, *, parent: str = "", content: str = "内容", page: int = 10,
        score: float = 0.5) -> dict:
    return {
        "chunk_id": chunk_id,
        "score": score,
        "doc_id": "d1",
        "parent_id": parent,
        "content": content,
        "page_start": page,
        "page_end": page,
        "domain": "普通",
        "evidence_grade": "教材共识",
    }


class TestGuards:
    async def test_no_kb_returns_empty(self) -> None:
        outcome = await retrieve("问题", kb_ids=[], embedder=FakeEmbedder(), index=FakeIndex([]))

        assert outcome.is_empty
        assert outcome.stats["reason"] == "no_kb"

    async def test_embed_failure_returns_empty_with_reason(self) -> None:
        outcome = await retrieve("问题", kb_ids=["kb1"], embedder=BrokenEmbedder(),
                                 index=FakeIndex([row("c1")]))

        assert outcome.is_empty
        assert outcome.stats["error"] == "embed_failed"

    async def test_search_failure_does_not_crash(self) -> None:
        outcome = await retrieve("问题", kb_ids=["kb1"], embedder=FakeEmbedder(),
                                 index=FakeIndex([row("c1")], fail_on_call=True))

        assert outcome.is_empty
        assert outcome.stats["recalled"] == 0


class TestHappyPath:
    async def test_full_pipeline(self) -> None:
        index = FakeIndex([row("c1", content="负强化增加行为"), row("c2", content="惩罚减少行为")])
        reranker = FakeReranker([0.9, 0.6])
        embedder = FakeEmbedder()

        outcome = await retrieve(
            "负强化和惩罚的区别", kb_ids=["kb1"], embedder=embedder,
            index=index, reranker=reranker, intent=Intent.CONTRAST,
        )

        assert not outcome.is_empty
        assert outcome.intent == Intent.CONTRAST
        assert outcome.strategy == "direct"
        assert outcome.queries == ["负强化和惩罚的区别"]
        assert "[S1]" in outcome.context
        assert len(outcome.citations) == 2
        assert outcome.citations[0].chunk_id == "c1"

    async def test_kb_ids_passed_to_index(self) -> None:
        index = FakeIndex([row("c1")])
        await retrieve("问题", kb_ids=["kb1", "kb2"], embedder=FakeEmbedder(),
                       index=index, intent=Intent.DEFINE)

        assert index.search_calls[0]["kb_ids"] == ["kb1", "kb2"]
        assert index.search_calls[0]["sparse"] is not None, "默认开启混合检索"

    async def test_rerank_uses_original_question(self) -> None:
        """重排必须用原句，不能用改写后的问题——否则判断的是"改写像不像"，不是"能不能回答"。"""
        index = FakeIndex([row("c1")])
        reranker = FakeReranker([0.9])

        await retrieve("负强化和惩罚区别", kb_ids=["kb1"], embedder=FakeEmbedder(),
                       index=index, reranker=reranker, intent=Intent.CONTRAST)

        assert reranker.calls[0][0] == "负强化和惩罚区别"


class TestRerankAndThreshold:
    async def test_rerank_changes_order(self) -> None:
        """三个分数都在阈值以上，这样断言的是"重排真的改变了顺序"，
        而不是"低分的被阈值顺手筛掉了"——两件事混在一起会让测试失去意义。"""
        index = FakeIndex([row("c1"), row("c2"), row("c3")])
        # RRF 顺序是 c1>c2>c3，但重排后 c3 分最高
        reranker = FakeReranker([0.70, 0.80, 0.95])

        outcome = await retrieve("问题", kb_ids=["kb1"], embedder=FakeEmbedder(),
                                 index=index, reranker=reranker, intent=Intent.DEFINE)

        assert [hit.chunk_id for hit in outcome.hits] == ["c3", "c2", "c1"]

    async def test_irrelevant_score_near_half_is_rejected(self) -> None:
        """实测结论：bge-reranker 对不相关段落的 logit 接近 0，sigmoid 后就是 0.5。
        阈值若设在 0.15 以下，等于不过滤——真实书测时确实把无关内容当过答案。
        这条用例钉住"0.5 左右必须被拒"这个语义。"""
        index = FakeIndex([row("c1")])
        reranker = FakeReranker([0.51])

        outcome = await retrieve("问题", kb_ids=["kb1"], embedder=FakeEmbedder(),
                                 index=index, reranker=reranker, intent=Intent.DEFINE)

        assert outcome.is_empty
        assert settings.retrieval.rerank_score_threshold > 0.51

    async def test_threshold_filters_low_scores(self) -> None:
        index = FakeIndex([row("c1"), row("c2")])
        reranker = FakeReranker([0.9, 0.05])  # 0.05 < 0.15 阈值

        outcome = await retrieve("问题", kb_ids=["kb1"], embedder=FakeEmbedder(),
                                 index=index, reranker=reranker, intent=Intent.DEFINE)

        assert len(outcome.hits) == 1
        assert outcome.stats["after_threshold"] == 1

    async def test_all_below_threshold_yields_empty(self) -> None:
        index = FakeIndex([row("c1")])
        reranker = FakeReranker([0.01])

        outcome = await retrieve("问题", kb_ids=["kb1"], embedder=FakeEmbedder(),
                                 index=index, reranker=reranker, intent=Intent.DEFINE)

        assert outcome.is_empty
        assert outcome.citations == []

    async def test_reranker_failure_keeps_order(self) -> None:
        index = FakeIndex([row("c1"), row("c2")])
        outcome = await retrieve("问题", kb_ids=["kb1"], embedder=FakeEmbedder(),
                                 index=index, reranker=BrokenReranker(), intent=Intent.DEFINE)

        assert len(outcome.hits) == 2
        assert outcome.hits[0].chunk_id == "c1"

    async def test_no_reranker_skips_threshold(self) -> None:
        """没重排时分数是 RRF 分（量级 0.01），拿 0.15 阈值去筛会全灭，所以跳过。"""
        index = FakeIndex([row("c1"), row("c2")])
        outcome = await retrieve("问题", kb_ids=["kb1"], embedder=FakeEmbedder(),
                                 index=index, intent=Intent.DEFINE)

        assert len(outcome.hits) == 2


class TestParentBackfill:
    async def test_child_replaced_by_parent_content(self) -> None:
        index = FakeIndex(
            [row("c1", parent="p1", content="子块片段", page=12)],
            parents={"p1": {"content": "父块完整内容，上下文更完整", "page_start": 10, "page_end": 14}},
        )

        outcome = await retrieve("问题", kb_ids=["kb1"], embedder=FakeEmbedder(),
                                 index=index, intent=Intent.DEFINE)

        assert outcome.hits[0].content.startswith("父块完整内容")
        assert outcome.hits[0].meta["page_start"] == 10
        assert outcome.hits[0].meta["hit_page"] == 12, "命中位置要留着，引用时用得上"

    async def test_missing_parent_keeps_child(self) -> None:
        index = FakeIndex([row("c1", parent="p1", content="子块片段")], parents={})

        outcome = await retrieve("问题", kb_ids=["kb1"], embedder=FakeEmbedder(),
                                 index=index, intent=Intent.DEFINE)

        assert outcome.hits[0].content == "子块片段"

    async def test_no_parent_id_skips_fetch(self) -> None:
        index = FakeIndex([row("c1", parent="")])
        await retrieve("问题", kb_ids=["kb1"], embedder=FakeEmbedder(),
                       index=index, intent=Intent.DEFINE)

        assert index.fetch_calls == []


class TestMultiQuery:
    async def test_subquery_fans_out(self) -> None:
        llm = FakeLLM(json_replies=[{"sub_queries": ["子问题1", "子问题2"]}])
        index = FakeIndex([row("c1")])
        embedder = FakeEmbedder()

        outcome = await retrieve("发展心理学主线", kb_ids=["kb1"], embedder=embedder,
                                 index=index, llm=llm, intent=Intent.FRAMEWORK)

        assert outcome.strategy == "subquery"
        assert len(outcome.queries) == 2
        assert len(embedder.calls[0]) == 2, "多个 query 应批量编码"
        assert len(index.search_calls) == 2, "每个 query 各召回一次"

    async def test_hyde_document_not_leaked_into_citations(self) -> None:
        """HyDE 的假设答案只用于取向量，绝不能出现在引用里。"""
        index = FakeIndex([row("c1", content="教材原文")])
        outcome = await retrieve("为啥背了想不起来", kb_ids=["kb1"], embedder=FakeEmbedder(),
                                 index=index, llm=FakeLLM(), intent=Intent.APPLY, reranker=FakeReranker([0.9]))

        assert "假设答案文本" not in outcome.context


class TestDocMeta:
    async def test_title_filled_from_provider(self) -> None:
        index = FakeIndex([row("c1")])

        async def provider(doc_id: str) -> dict:
            return {"title": "普通心理学", "edition": "第5版"}

        outcome = await retrieve("问题", kb_ids=["kb1"], embedder=FakeEmbedder(),
                                 index=index, intent=Intent.DEFINE, doc_meta=provider)

        assert outcome.citations[0].doc == "普通心理学"
        assert "普通心理学" in outcome.context

    async def test_provider_failure_does_not_break(self) -> None:
        index = FakeIndex([row("c1")])

        async def provider(doc_id: str) -> dict:
            raise RuntimeError("db down")

        outcome = await retrieve("问题", kb_ids=["kb1"], embedder=FakeEmbedder(),
                                 index=index, intent=Intent.DEFINE, doc_meta=provider)

        assert len(outcome.hits) == 1


class TestBudget:
    async def test_context_truncated_by_budget(self) -> None:
        rows = [row(f"c{i}", parent=f"p{i}", content="字" * 2000) for i in range(5)]
        parents = {f"p{i}": {"content": "父" * 2000, "page_start": 1, "page_end": 2} for i in range(5)}
        index = FakeIndex(rows, parents=parents)

        outcome = await retrieve("问题", kb_ids=["kb1"], embedder=FakeEmbedder(),
                                 index=index, intent=Intent.DEFINE, reranker=FakeReranker([0.9] * 5))

        # 6000 字预算，父块各 2000 字 → 最多 3 条
        assert len(outcome.hits) <= 3
        assert len(outcome.context) <= 6100

    async def test_stats_recorded(self) -> None:
        index = FakeIndex([row("c1")])
        outcome = await retrieve("问题", kb_ids=["kb1"], embedder=FakeEmbedder(),
                                 index=index, intent=Intent.DEFINE)

        for key in ("recalled", "fused", "after_dedupe", "selected", "latency_ms"):
            assert key in outcome.stats


class TestRerankOnChildContent:
    """重排看子块，父块只在选中之后回填给生成用。

    这条顺序是拿数据换的，不是猜的（scripts/_rerank_input_ab.py，26 道概念题）：
      金标平均分  子块 0.7043 / 父块 0.6472
      逐题对比    子块更好 15 题，父块更好 11 题
      灾难案例    c006 #1→#15、c011 #1→#13、c016 #1→#13，全被阈值砍掉
    父块里只有一小段和问题相关，交叉编码看整段相关性，会被稀释。

    之前这里断言的是相反的行为（"回填必须发生在重排之前"），那版判断已被证伪。
    """

    async def test_rerank_sees_child_content(self) -> None:
        index = FakeIndex(
            [row("c1", parent="p1", content="子块片段：负强化")],
            parents={"p1": {"content": "父块全文：负强化通过撤销厌恶刺激来增加行为", "page_start": 3}},
        )
        reranker = FakeReranker([0.9])

        await retrieve("负强化", kb_ids=["kb1"], embedder=FakeEmbedder(),
                       index=index, reranker=reranker, intent=Intent.DEFINE)

        documents = reranker.calls[0][1]
        assert documents == ["子块片段：负强化"], "重排应当只看到子块"
        assert "父块全文" not in documents[0]

    async def test_parent_backfill_happens_after_rerank(self) -> None:
        """回填发生在重排之后：送进重排的是子块，最终命中里是父块全文。"""
        index = FakeIndex(
            [row("c1", parent="p1", content="子块")],
            parents={"p1": {"content": "父块", "page_start": 1}},
        )
        reranker = FakeReranker([0.9])

        outcome = await retrieve("问题", kb_ids=["kb1"], embedder=FakeEmbedder(),
                                 index=index, reranker=reranker, intent=Intent.DEFINE)

        assert reranker.calls[0][1] == ["子块"], "重排前不该已经回填"
        assert index.fetch_calls, "选出结果后必须回填父块"
        assert outcome.hits and outcome.hits[0].content == "父块", "生成阶段要拿到父块全文"

    async def test_rerank_candidates_capped(self) -> None:
        """去重后候选可能有几十个，全送去交叉编码在 CPU 上会慢到不可接受。"""
        rows = [row(f"c{i}", parent=f"p{i}") for i in range(30)]
        index = FakeIndex(rows)
        reranker = FakeReranker([0.5] * 30)

        await retrieve("问题", kb_ids=["kb1"], embedder=FakeEmbedder(),
                       index=index, reranker=reranker, intent=Intent.DEFINE)

        cap = settings.retrieval.rerank_max_candidates
        assert len(reranker.calls[0][1]) == cap

    async def test_truncation_counted_in_stats(self) -> None:
        rows = [row(f"c{i}", parent=f"p{i}") for i in range(30)]
        index = FakeIndex(rows)

        outcome = await retrieve("问题", kb_ids=["kb1"], embedder=FakeEmbedder(),
                                 index=index, reranker=FakeReranker([0.5] * 30),
                                 intent=Intent.DEFINE)

        # 召回上限是 dense_topk/sparse_topk（各 20，取 max），所以去重后是 20 而非 30。
        # 断言从实际去重数推导，不把 topk 常量抄进测试。
        cap = settings.retrieval.rerank_max_candidates
        assert outcome.stats["rerank_truncated"] == outcome.stats["after_dedupe"] - cap

    async def test_single_candidate_still_reranked(self) -> None:
        """只有一条候选也要重排：否则跳过阈值过滤，一条垃圾会被原样送进生成。"""
        index = FakeIndex([row("c1")])
        reranker = FakeReranker([0.05])

        outcome = await retrieve("问题", kb_ids=["kb1"], embedder=FakeEmbedder(),
                                 index=index, reranker=reranker, intent=Intent.DEFINE)

        assert reranker.calls, "单条候选被跳过了重排"
        assert outcome.is_empty, "低于阈值的结果没有被过滤掉"
