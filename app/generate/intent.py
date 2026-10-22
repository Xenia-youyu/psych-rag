"""意图分类：把提问归入 10 类，供下游选检索策略。

实现顺序（与 v3 文档一致）：
1. 先走规则 —— 明确的信号词直接判定，省一次模型调用
2. 规则拿不准再走 deepseek-flash + prompt
3. 模型也失败就兜底 DEFINE（最保守，走向量检索不会出错）

规则优先不是偷懒：意图分类是全链路调用量最大的一环，
把 40%~60% 的高置信样本用规则挡掉，token 成本直接砍半。
"""

from __future__ import annotations

import re
from enum import StrEnum

from app.core.logging import get_logger

logger = get_logger(__name__)


class Intent(StrEnum):
    DEFINE = "DEFINE"
    CONTRAST = "CONTRAST"
    FRAMEWORK = "FRAMEWORK"
    GAP = "GAP"
    APPLY = "APPLY"
    HISTORY = "HISTORY"
    VERSION_COMPARE = "VERSION_COMPARE"
    NUMERIC = "NUMERIC"
    EVIDENCE_CHECK = "EVIDENCE_CHECK"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"


# 规则表：按优先级从高到低匹配，命中即返回
RULES: list[tuple[Intent, re.Pattern[str]]] = [
    (Intent.NUMERIC, re.compile(r"(临界值|查表|自由度|显著性水平|T分|标准分|常模|α系数|p值)")),
    (Intent.VERSION_COMPARE, re.compile(r"(第[一二三四五六七八九十\d]+版|版本|版次).{0,12}(不一样|不同|差异|哪个|以哪个)")),
    (Intent.EVIDENCE_CHECK, re.compile(r"(科学依据|可不可信|靠谱吗|伪科学|是不是真的|有没有用|有效吗)")),
    (Intent.CONTRAST, re.compile(r"(区别|不同|差别|差异|分辨|区分|搞混|总是选错|vs)")),
    (Intent.FRAMEWORK, re.compile(r"(主线|框架|体系|怎么串|脉络|整体|梳理|知识结构)")),
    (Intent.GAP, re.compile(r"(为什么选错|这题|不懂|看不懂|卡在|不会做|错在哪)")),
    (Intent.HISTORY, re.compile(r"(谁提出|哪位|发展过程|演变|源流|后来怎么|提出者)")),
    (Intent.APPLY, re.compile(r"(怎么用|应用|举个例子|举例|实际场景|案例)")),
    (Intent.DEFINE, re.compile(r"(是什么|定义|含义|指什么|概念)")),
]


def classify_by_rule(text: str) -> Intent | None:
    """规则命中返回意图，否则 None。"""
    if not text:
        return None
    normalized = text.strip()
    for intent, pattern in RULES:
        if pattern.search(normalized):
            return intent
    return None


def _build_messages(question: str) -> list[dict[str, str]]:
    from app.prompt_loader import registry

    template = registry.get("intent_classify")
    return [
        {"role": "system", "content": template.system()},
        {"role": "user", "content": template.render(question=question)},
    ]


async def classify(text: str, *, llm=None, use_rule_first: bool = True) -> Intent:
    """分类入口。llm 为 None 时只用规则（便于测试与降级）。"""
    if use_rule_first:
        ruled = classify_by_rule(text)
        if ruled is not None:
            logger.info(
                "intent.rule_hit",
                extra={"extra_fields": {"intent": ruled.value, "query": text[:40]}},
            )
            return ruled

    if llm is None:
        return Intent.DEFINE

    try:
        # max_tokens 不能压到 64：实测模型在 json mode 下会先吐一点思考内容，
        # 64 个 token 被吃干，content 直接是空串，chat_json 再重试一次——
        # 一次分类白跑三次请求，检索平白多两秒。256 是够用又不浪费的值。
        data = await llm.chat_json(_build_messages(text), temperature=0.0, max_tokens=256)
        raw = str(data.get("intent", "")).upper().strip()
        intent = Intent(raw)
        logger.info(
            "intent.llm_hit",
            extra={"extra_fields": {"intent": intent.value, "query": text[:40]}},
        )
        return intent
    except (ValueError, AttributeError):
        logger.warning(
            "intent.invalid_label",
            extra={"extra_fields": {"query": text[:40]}},
        )
        return Intent.DEFINE
    except Exception as exc:  # noqa: BLE001
        logger.warning("intent.llm_failed", extra={"extra_fields": {"error": repr(exc)}})
        return Intent.DEFINE


# 意图 → 检索策略，与 v3 文档 2.3 节对应
STRATEGY_BY_INTENT: dict[Intent, str] = {
    Intent.DEFINE: "direct",
    Intent.CONTRAST: "direct",
    Intent.FRAMEWORK: "subquery",
    Intent.GAP: "stepback",
    Intent.APPLY: "direct",
    Intent.HISTORY: "direct",
    Intent.VERSION_COMPARE: "subquery",
    Intent.EVIDENCE_CHECK: "direct",
    Intent.NUMERIC: "sql",
    Intent.OUT_OF_SCOPE: "none",
}


def strategy_for(intent: Intent) -> str:
    return STRATEGY_BY_INTENT.get(intent, "direct")
