"""查询改写：四类策略。

| 策略 | 适用 | 做什么 |
|---|---|---|
| direct   | 意图明确 | 原样检索 |
| hyde     | 口语化、与教材表述差距大 | 生成假设答案，用答案向量检索 |
| subquery | 多跳问题 | 拆成 2~4 个子问题并行检索 |
| stepback | 缺上位概念 | 抽象成更一般的问题，先补前置 |

硬约束（v3 文档 2.3 节）：
- HyDE 生成的假设答案**绝不进入最终上下文**，只取向量
- 子查询上限 4 条，超了合并
- 统一 temperature=0.1，输出 ≤200 token，异常降级为 direct
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.core.logging import get_logger
from app.generate.intent import Intent, strategy_for

logger = get_logger(__name__)

MAX_SUB_QUERIES = 4


@dataclass
class RewriteResult:
    strategy: str
    queries: list[str] = field(default_factory=list)
    # HyDE 的假设答案只用于取向量，不外传
    hyde_document: str = ""

    @property
    def is_direct(self) -> bool:
        return self.strategy == "direct"


def _messages(prompt_id: str, **slots: str) -> list[dict[str, str]]:
    from app.prompt_loader import registry

    template = registry.get(prompt_id)
    return [
        {"role": "system", "content": template.system()},
        {"role": "user", "content": template.render(**slots)},
    ]


async def _hyde(question: str, llm) -> RewriteResult:
    result = await llm.chat(_messages("query_rewrite_hyde", question=question), temperature=0.1,
                            max_tokens=200)
    document = (result.text or "").strip()
    if not document:
        return RewriteResult("direct", [question])
    # 用假设答案去检索，原问题也保留一路，防止假设跑偏
    return RewriteResult("hyde", [document, question], hyde_document=document)


async def _subquery(question: str, llm) -> RewriteResult:
    data = await llm.chat_json(
        _messages("query_rewrite_subquery", question=question), temperature=0.1, max_tokens=200
    )
    raw = data.get("sub_queries") or []
    queries = [str(item).strip() for item in raw if str(item).strip()]

    if not queries:
        return RewriteResult("direct", [question])

    # 去重保序
    deduped: list[str] = []
    for query in queries:
        if query not in deduped:
            deduped.append(query)
    if len(deduped) > MAX_SUB_QUERIES:
        deduped = deduped[:MAX_SUB_QUERIES]
    return RewriteResult("subquery", deduped)


async def _stepback(question: str, llm) -> RewriteResult:
    data = await llm.chat_json(
        _messages("query_rewrite_stepback", question=question), temperature=0.1, max_tokens=200
    )
    upper = str(data.get("stepback_query", "")).strip()
    original = str(data.get("original_query", "")).strip() or question

    if not upper or upper == original:
        return RewriteResult("direct", [question])
    # 上位问题在前：先补前置，再回到细节
    return RewriteResult("stepback", [upper, original])


async def rewrite(question: str, intent: Intent, llm) -> RewriteResult:
    """按意图选择策略并执行。任何异常降级为 direct，绝不阻断主链路。"""
    strategy = strategy_for(intent)
    if strategy in ("direct", "none", "sql") or llm is None:
        return RewriteResult("direct", [question])

    try:
        if strategy == "hyde":
            return await _hyde(question, llm)
        if strategy == "subquery":
            return await _subquery(question, llm)
        if strategy == "stepback":
            return await _stepback(question, llm)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "rewrite.failed_fallback_direct",
            extra={"extra_fields": {"strategy": strategy, "error": repr(exc)}},
        )
        return RewriteResult("direct", [question])

    return RewriteResult("direct", [question])
