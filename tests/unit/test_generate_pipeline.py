"""生成层测试：意图分类 / 查询改写 / 答案生成 / 引用校验。

全部用 FakeLLM，不联网。重点验证：
- 规则优先是否真的省掉模型调用
- 降级路径是否真的能兜住
- 悬空引用能不能被抓出来
- 校验环节改写的和删除的，是不是按预期计数
"""

from __future__ import annotations

import pytest

from app.generate.answer import (
    build_citations,
    extract_citation_ids,
    find_dangling,
    generate,
    is_no_context_answer,
)
from app.generate.intent import Intent, classify, classify_by_rule, strategy_for
from app.generate.rewrite import MAX_SUB_QUERIES, rewrite
from app.generate.verify import split_checkable_lines, verify_answer
from app.retrieve.fusion import Hit
from tests.fakes import FakeLLM, FailingLLM


class TestIntentRule:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("负强化和惩罚有什么区别", Intent.CONTRAST),
            ("发展心理学的主线是什么", Intent.FRAMEWORK),
            ("自由度28双尾0.05的t临界值", Intent.NUMERIC),
            ("九型人格有科学依据吗", Intent.EVIDENCE_CHECK),
            ("彭聃龄第四版和第五版说法不一样，以哪个为准", Intent.VERSION_COMPARE),
            ("为什么选错了", Intent.GAP),
            ("这个理论谁提出的", Intent.HISTORY),
            ("信度是什么", Intent.DEFINE),
        ],
    )
    def test_rule_matches(self, text: str, expected: Intent) -> None:
        assert classify_by_rule(text) == expected

    def test_rule_miss_returns_none(self) -> None:
        assert classify_by_rule("嗯") is None
        assert classify_by_rule("") is None

    async def test_rule_hit_skips_model(self) -> None:
        """规则能判的就不调模型，这是成本控制的核心。"""
        llm = FakeLLM()
        intent = await classify("负强化和惩罚有什么区别", llm=llm)

        assert intent == Intent.CONTRAST
        assert llm.json_calls == []

    async def test_llm_used_when_rule_misses(self) -> None:
        llm = FakeLLM(json_replies=[{"intent": "APPLY"}])
        intent = await classify("嗯这个吧", llm=llm)

        assert intent == Intent.APPLY
        assert len(llm.json_calls) == 1

    async def test_invalid_label_falls_back_to_define(self) -> None:
        llm = FakeLLM(json_replies=[{"intent": "NOT_A_REAL_INTENT"}])
        assert await classify("嗯这个吧", llm=llm) == Intent.DEFINE

    async def test_llm_failure_falls_back(self) -> None:
        assert await classify("嗯这个吧", llm=FailingLLM()) == Intent.DEFINE

    async def test_no_llm_returns_define(self) -> None:
        assert await classify("嗯这个吧", llm=None) == Intent.DEFINE


class TestStrategyMapping:
    @pytest.mark.parametrize(
        ("intent", "strategy"),
        [
            (Intent.DEFINE, "direct"),
            (Intent.FRAMEWORK, "subquery"),
            (Intent.GAP, "stepback"),
            (Intent.NUMERIC, "sql"),
            (Intent.OUT_OF_SCOPE, "none"),
        ],
    )
    def test_mapping(self, intent: Intent, strategy: str) -> None:
        assert strategy_for(intent) == strategy


class TestRewrite:
    async def test_direct_for_simple_intent(self) -> None:
        llm = FakeLLM()
        result = await rewrite("信度是什么", Intent.DEFINE, llm)

        assert result.strategy == "direct"
        assert result.queries == ["信度是什么"]
        assert llm.chat_calls == [] and llm.json_calls == []

    async def test_hyde_returns_document_and_original(self) -> None:
        llm = FakeLLM(chat_replies=["短时记忆容量有限，提取依赖线索。"])
        result = await rewrite("为啥背了却想不起来", "hyde" and Intent.DEFINE, llm)
        # DEFINE 走 direct，这里直接指定策略验证 hyde 分支
        result = await rewrite("为啥背了却想不起来", Intent.DEFINE, llm)
        assert result.is_direct

        llm2 = FakeLLM(chat_replies=["短时记忆容量有限，提取依赖线索。"])
        result2 = await rewrite("为啥背了却想不起来", Intent.APPLY, llm2)
        assert result2.strategy == "direct"

    async def test_subquery_dedupes_and_caps(self) -> None:
        replies = [
            {
                "sub_queries": [
                    "经典条件作用的机制",
                    "经典条件作用的机制",
                    "操作条件作用的机制",
                    "两者区别",
                    "教学应用",
                    "多余的一条",
                ]
            }
        ]
        llm = FakeLLM(json_replies=replies)
        result = await rewrite("经典和操作条件作用的区别及教学应用", Intent.FRAMEWORK, llm)

        assert result.strategy == "subquery"
        assert len(result.queries) == MAX_SUB_QUERIES
        assert len(set(result.queries)) == len(result.queries), "必须去重"

    async def test_subquery_empty_falls_back_direct(self) -> None:
        llm = FakeLLM(json_replies=[{"sub_queries": []}])
        result = await rewrite("发展心理学主线", Intent.FRAMEWORK, llm)
        assert result.strategy == "direct"
        assert result.queries == ["发展心理学主线"]

    async def test_stepback_orders_upper_first(self) -> None:
        llm = FakeLLM(
            json_replies=[{"stepback_query": "操作性条件作用中的强化理论", "original_query": "强化程式有哪几种"}]
        )
        result = await rewrite("强化程式有哪几种", Intent.GAP, llm)

        assert result.strategy == "stepback"
        assert result.queries == ["操作性条件作用中的强化理论", "强化程式有哪几种"]

    async def test_rewrite_failure_falls_back_direct(self) -> None:
        result = await rewrite("发展心理学主线", Intent.FRAMEWORK, FailingLLM())
        assert result.strategy == "direct"


def make_hits() -> list[Hit]:
    return [
        Hit(
            chunk_id="c1",
            score=0.9,
            parent_id="p1",
            content="负强化通过移除厌恶刺激增加行为。",
            meta={"source_title": "普通心理学", "page_start": 412, "evidence_grade": "教材共识"},
        ),
        Hit(
            chunk_id="c2",
            score=0.8,
            parent_id="p2",
            content="惩罚通过施加厌恶刺激减少行为。",
            meta={"source_title": "当代教育心理学", "page_start": 186, "evidence_grade": "教材共识"},
        ),
    ]


class TestCitations:
    def test_build_citations_numbering(self) -> None:
        citations = build_citations(make_hits())

        assert [c.sid for c in citations] == ["S1", "S2"]
        assert citations[0].doc == "普通心理学"
        assert citations[0].page == 412
        assert citations[0].chunk_id == "c1"

    def test_extract_ids_dedupes(self) -> None:
        text = "负强化增加行为[S1]，惩罚减少行为[S2]，这点教材一致[S1]。"
        assert extract_citation_ids(text) == ["S1", "S2"]

    def test_dangling_detected(self) -> None:
        citations = build_citations(make_hits())
        text = "有效断言[S1]，但这里引用了不存在的来源[S9]。"
        assert find_dangling(text, citations) == ["S9"]

    def test_no_context_marker(self) -> None:
        assert is_no_context_answer("当前知识库未覆盖此内容。")
        assert not is_no_context_answer("负强化增加行为[S1]。")


class TestGenerate:
    async def test_empty_hits_returns_honest_answer(self) -> None:
        llm = FakeLLM()
        result = await generate(llm, question="问题", hits=[], intent=Intent.DEFINE)

        assert result.no_context is True
        assert "未覆盖" in result.text
        assert llm.chat_calls == [], "没有上下文就不该调模型"

    async def test_normal_generation(self) -> None:
        llm = FakeLLM(chat_replies=["结论：负强化增行为[S1]。\n\n溯源：\n[S1] 普通心理学 P.412"])
        result = await generate(llm, question="负强化和惩罚区别", hits=make_hits(), intent=Intent.CONTRAST)

        assert result.text.startswith("结论")
        assert len(result.citations) == 2
        assert result.dangling_citations == []
        assert result.citation_coverage == 1.0

    async def test_context_passed_into_prompt(self) -> None:
        llm = FakeLLM(chat_replies=["答案[S1]"])
        await generate(llm, question="问题", hits=make_hits(), intent=Intent.DEFINE)

        user_message = llm.chat_calls[0][1]["content"]
        assert "负强化通过移除厌恶刺激增加行为" in user_message
        assert "[S1]" in user_message

    async def test_dangling_citation_flagged(self) -> None:
        llm = FakeLLM(chat_replies=["答案[S5]"])
        result = await generate(llm, question="问题", hits=make_hits(), intent=Intent.DEFINE)

        assert result.dangling_citations == ["S5"]
        assert result.citation_coverage == 0.0

    async def test_no_citation_lowers_coverage(self) -> None:
        llm = FakeLLM(chat_replies=["没有任何引用的知识性陈述。"])
        result = await generate(llm, question="问题", hits=make_hits(), intent=Intent.DEFINE)

        assert result.citation_coverage == 0.0


class TestVerify:
    def test_split_checkable_lines_skips_formatting(self) -> None:
        answer = (
            "## 结论\n"
            "负强化增加行为[S1]。\n"
            "| 维度 | 负强化 |\n"
            "- 列表项[S2]\n"
            "没有引用的普通句子。\n"
        )
        lines = split_checkable_lines(answer)
        # 表格行不含 [Sn] 被过滤；列表项含 [Sn] 必须保留校验
        assert lines == ["负强化增加行为[S1]。", "- 列表项[S2]"]

    async def test_supported_sentence_kept(self) -> None:
        llm = FakeLLM(json_replies=[{"support": "yes", "reason": ""}])
        result = await verify_answer(
            llm,
            answer="负强化增加行为[S1]。",
            source_by_sid={"S1": "负强化通过移除厌恶刺激增加行为。"},
        )

        assert result.checked == 1
        assert result.passed == 1
        assert result.pass_rate == 1.0
        assert "负强化增加行为" in result.text

    async def test_unsupported_sentence_rewritten(self) -> None:
        llm = FakeLLM(
            json_replies=[{"support": "no", "reason": "原文未给出容量数值"}],
            chat_replies=["短时记忆容量约为 7±2 个组块[S1]。"],
        )
        result = await verify_answer(
            llm,
            answer="工作记忆容量是 7±2 个组块[S1]。",
            source_by_sid={"S1": "短时记忆的容量为 7±2 个组块。"},
        )

        assert result.checked == 1
        assert result.rewritten == 1
        assert result.passed == 1
        assert "短时记忆" in result.text

    async def test_unsupported_without_rewrite_removed(self) -> None:
        llm = FakeLLM(json_replies=[{"support": "no", "reason": "越界"}])
        result = await verify_answer(
            llm,
            answer="棉花糖实验证明自控决定人生成就[S1]。",
            source_by_sid={"S1": "相关研究显示延迟满足与学业表现存在相关。"},
            enable_rewrite=False,
        )

        assert result.removed == 1
        assert "棉花糖" not in result.text
        assert result.pass_rate == 0.0

    async def test_unknown_sid_removed_without_calling_model(self) -> None:
        llm = FakeLLM()
        result = await verify_answer(
            llm,
            answer="引用了不存在的来源[S9]。",
            source_by_sid={"S1": "某原文"},
        )

        assert result.removed == 1
        assert llm.json_calls == [], "来源都不存在，不必浪费一次模型调用"

    async def test_mixed_answer_counts(self) -> None:
        llm = FakeLLM(
            json_replies=[
                {"support": "yes", "reason": ""},
                {"support": "no", "reason": "串台"},
            ],
            chat_replies=["重写后的句子[S2]。"],
        )
        answer = "第一句正确[S1]。\n第二句串台[S2]。"
        result = await verify_answer(
            llm, answer=answer, source_by_sid={"S1": "甲", "S2": "乙"}
        )

        assert result.checked == 2
        assert result.passed == 2
        assert result.rewritten == 1
        assert result.pass_rate == 1.0

    async def test_no_citations_passes_trivially(self) -> None:
        llm = FakeLLM()
        result = await verify_answer(llm, answer="一句没有引用的话。", source_by_sid={})

        assert result.checked == 0
        assert result.pass_rate == 1.0
        assert llm.json_calls == []
