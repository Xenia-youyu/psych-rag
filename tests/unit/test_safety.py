"""安全护栏测试。

这是全项目最有必要测的一块：危机词漏拦的代价不是"答得不好"，是安全事故。
因此这里同时验证「该拦的拦住」和「不该拦的别误伤」。
"""

from __future__ import annotations

import pytest

from app.generate.safety import TriggerType, check, fallback_reply


class TestCrisis:
    @pytest.mark.parametrize(
        "text",
        [
            "我想自杀",
            "有时候真的不想活了",
            "活着没意思",
            "我想结束生命",
            "最近总想割腕",
        ],
    )
    def test_crisis_detected(self, text: str) -> None:
        verdict = check(text)
        assert verdict.triggered
        assert verdict.trigger_type == TriggerType.CRISIS

    def test_reply_contains_hotline(self) -> None:
        reply = fallback_reply(check("我想自杀"))
        assert "12356" in reply
        assert "120" in reply

    def test_reply_contains_no_knowledge(self) -> None:
        """危机话术里不允许出现任何知识性解释，避免用户拿来对照自己。"""
        reply = fallback_reply(check("我想自杀"))
        for word in ("症状", "诊断标准", "患病率", "治疗方法"):
            assert word not in reply


class TestDiagnosis:
    @pytest.mark.parametrize(
        "text",
        [
            "我是不是得了抑郁症",
            "我有没有焦虑症",
            "这算不算强迫症",
            "我最近总是失眠，是不是抑郁了",
        ],
    )
    def test_diagnosis_detected(self, text: str) -> None:
        verdict = check(text)
        assert verdict.triggered
        assert verdict.trigger_type == TriggerType.DIAGNOSIS

    def test_reply_refuses_to_judge(self) -> None:
        reply = fallback_reply(check("我是不是得了抑郁症"))
        assert "不能判断" in reply or "不应该判断" in reply
        assert "诊断" in reply


class TestNoFalsePositive:
    @pytest.mark.parametrize(
        "text",
        [
            "负强化和惩罚有什么区别",
            "抑郁症的诊断标准是什么",
            "自杀风险评估在心理咨询中怎么做",
            "变态心理学里怎么讲自伤行为",
            "发展心理学的主线是什么",
        ],
    )
    def test_knowledge_questions_not_blocked(self, text: str) -> None:
        """学术性提问不能被熔断误伤——这些是学员真会问的。"""
        assert not check(text)

    def test_empty_input(self) -> None:
        assert not check("")
        assert not check("   ")


class TestBoundary:
    """误拦与漏拦的分界线。这条线画在哪，决定系统是"稳妥"还是"吓人"。"""

    def test_academic_topic_with_keyword_not_blocked(self) -> None:
        verdict = check("自杀风险评估在心理咨询中怎么做")
        assert not verdict.triggered

    def test_first_person_overrides_academic_context(self) -> None:
        """第一人称求助信号优先于学术标记——这句如果放过就是事故。"""
        verdict = check("我想自杀，有什么量表可以评估吗")
        assert verdict.triggered
        assert verdict.trigger_type == TriggerType.CRISIS

    def test_ambiguous_help_seeking_defaults_to_block(self) -> None:
        """无第一人称、无学术语境的模糊表述，宁可拦下。"""
        verdict = check("如何应对自杀想法")
        assert verdict.triggered


class TestVerdictSemantics:
    def test_bool_conversion(self) -> None:
        assert bool(check("我想自杀")) is True
        assert bool(check("信度是什么")) is False

    def test_matched_keyword_recorded(self) -> None:
        verdict = check("我想自杀")
        assert verdict.matched
        assert verdict.matched in "我想自杀"

    def test_reply_is_never_empty(self) -> None:
        for text in ("我想自杀", "我是不是得了抑郁症"):
            assert len(fallback_reply(check(text))) > 50
