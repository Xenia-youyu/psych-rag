"""安全护栏：危机干预熔断 + 求诊断拦截。

这是全项目唯一「不走大模型」的分支。命中即返回固定话术，理由见 prompts/safe_guard.v1.md：
让模型"共情地回应"自杀倾向发言，是拿使用者安全赌模型稳定性。固定话术笨，但可预测。

两类触发：
1. CRISIS：命中危机词表 → 输出援助资源，终止一切知识性延展
2. DIAGNOSIS：问"我是不是/我有没有 + 精神障碍"或描述自身症状求判断 → 免责 + 转介
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from app.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class TriggerType(StrEnum):
    CRISIS = "CRISIS"
    DIAGNOSIS = "DIAGNOSIS"


@dataclass
class SafetyVerdict:
    triggered: bool
    trigger_type: TriggerType | None = None
    matched: str = ""

    def __bool__(self) -> bool:
        return self.triggered


# 求诊断的句式：第一人称 + 障碍名 + 判断性提问
DISORDER_NAMES = [
    "抑郁症",
    "抑郁",
    "焦虑症",
    "双相",
    "躁郁",
    "强迫症",
    "精神分裂",
    "人格障碍",
    "创伤后应激",
    "PTSD",
    "进食障碍",
    "厌食症",
    "暴食症",
    "社交恐惧",
    "惊恐障碍",
    "ADHD",
    "多动症",
    "自闭症",
    "孤独症",
    "成瘾",
]

FIRST_PERSON_RE = re.compile(r"(我|这|那|他|她)\s*(是不是|有没有|算不算|会不会是|算|得了)")

# 学术/专业语境标记：出现这些词时，说明用户在讨论"这个主题"而不是在求助
ACADEMIC_MARKERS = (
    "评估",
    "干预",
    "预防",
    "识别",
    "筛查",
    "研究",
    "教材",
    "章节",
    "理论",
    "督导",
    "教学",
    "心理咨询中",
    "心理学里",
    "考点",
    "量表",
)

# 第一人称求助信号：与危机词同现时无条件熔断，学术标记也不豁免
SELF_CRISIS_RE = re.compile(
    r"(我|自己)\s*.{0,6}(想|要|打算|准备|计划|总是|一直|最近|曾经).{0,6}"
    r"(自杀|自残|不想活|结束生命|割腕|跳楼|死)"
)

SELF_SYMPTOM_RE = re.compile(
    r"我(最近|总是|经常|一直|老是).{0,12}(失眠|睡不着|想哭|没兴趣|提不起|暴躁|心慌|不想动)"
)


def _crisis_keywords() -> list[str]:
    keywords = settings.security.crisis_keywords
    if keywords:
        return keywords
    # 配置缺失时的兜底，宁可多拦不可漏拦
    return ["自杀", "自残", "不想活", "活着没意思", "结束生命", "割腕", "跳楼"]


def _is_academic_context(text: str) -> bool:
    return any(marker in text for marker in ACADEMIC_MARKERS)


def check(text: str) -> SafetyVerdict:
    """检测是否需要熔断。返回首个命中的类型。

    危机词的判定不是简单的字符串包含，而是分三种情况：
    1. 第一人称求助信号 + 危机词 → 无条件熔断（学术标记也不豁免）
    2. 危机词 + 学术语境词 → 视为专业提问，正常走知识问答
    3. 只有危机词、两者都不满足 → 熔断（安全优先，宁可多拦）

    第 3 条是刻意保守的：漏拦一次的代价远大于误拦一次。
    """
    if not text:
        return SafetyVerdict(triggered=False)

    normalized = text.strip()

    if SELF_CRISIS_RE.search(normalized):
        for keyword in _crisis_keywords():
            if keyword and keyword in normalized:
                return SafetyVerdict(True, TriggerType.CRISIS, keyword)

    hit_keyword = ""
    for keyword in _crisis_keywords():
        if keyword and keyword in normalized:
            hit_keyword = keyword
            break

    if hit_keyword:
        if _is_academic_context(normalized):
            # 专业提问，不做熔断，但仍记录，便于事后核查是否有误判
            logger.info(
                "safety.academic_crisis_topic",
                extra={"extra_fields": {"keyword": hit_keyword, "text": normalized[:40]}},
            )
        else:
            return SafetyVerdict(True, TriggerType.CRISIS, hit_keyword)

    for disorder in DISORDER_NAMES:
        if disorder in normalized and FIRST_PERSON_RE.search(normalized):
            return SafetyVerdict(True, TriggerType.DIAGNOSIS, disorder)

    if SELF_SYMPTOM_RE.search(normalized) and ("是不是" in normalized or "算不算" in normalized):
        return SafetyVerdict(True, TriggerType.DIAGNOSIS, "症状自述")

    return SafetyVerdict(triggered=False)


_BLOCK_A = re.compile(r"###\s*话术\s*A[^\n]*\n(.*?)(?=\n###|\Z)", re.DOTALL)
_BLOCK_B = re.compile(r"###\s*话术\s*B[^\n]*\n(.*?)(?=\n##|\Z)", re.DOTALL)


def _strip_fence(block: str) -> str:
    text = block.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
    if text.rstrip().endswith("```"):
        text = text.rstrip()[:-3]
    return text.strip()


def fallback_reply(verdict: SafetyVerdict) -> str:
    """固定话术，直接从提示词模板里取，保证文案只有一个来源。"""
    from app.prompt_loader import registry

    body = registry.get("safe_guard").fixed_text()
    pattern = _BLOCK_B if verdict.trigger_type == TriggerType.DIAGNOSIS else _BLOCK_A

    match = pattern.search(body)
    if match:
        return _strip_fence(match.group(1))
    return "本系统仅提供心理学知识科普，不能替代专业帮助。如有困扰请及时联系专业人士。"
