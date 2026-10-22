"""引用断言校验：生成后逐句核查，拦掉"串台"。

为什么必须有这一层：
deepseek-flash 这类轻量模型写长答案时，容易把相邻概念的结论安错对象——
"工作记忆容量 7±2"（其实是短时记忆的）、"棉花糖实验证明自控决定成就"（原文只说相关）。
这类句子读起来非常顺，Ragas 的 faithfulness 不一定抓得住，人眼也容易放过。

策略：
1. 按行切分，只校验**带引用标记**的行（无引用的行不校验，交给"无引用断言"指标）
2. 每行连同其引用来源的原文一起送模型做 yes/no 判定
3. 判定否的行：重写（限定只用原文信息）→ 重写仍失败则删除该行
4. 输出校验通过率与改写条数，写入 query_logs
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.core.logging import get_logger

logger = get_logger(__name__)

CITATION_RE = re.compile(r"\[S(\d+)\]")

# 只跳过纯排版行。注意：表格行与列表项**要**校验——
# 对照表和要点列表恰恰是最容易塞进错误结论的地方，跳过它们等于放过高风险区。
# 不含 [Sn] 的行本来就会被 CITATION_RE 过滤掉，不需要在这里额外排除。
SKIP_PREFIXES = ("#", ">", "**溯源**")


@dataclass
class VerifyResult:
    text: str
    checked: int = 0
    passed: int = 0
    rewritten: int = 0
    removed: int = 0
    failures: list[tuple[str, str]] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        if self.checked == 0:
            return 1.0
        return self.passed / self.checked


def split_checkable_lines(text: str) -> list[str]:
    """挑出需要校验的行：含 [Sn] 且不是纯排版行。"""
    lines: list[str] = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line or not CITATION_RE.search(line):
            continue
        if line.startswith(SKIP_PREFIXES):
            continue
        lines.append(line)
    return lines


def _messages(sentence: str, source: str) -> list[dict[str, str]]:
    from app.prompt_loader import registry

    template = registry.get("citation_verify")
    return [
        {"role": "system", "content": template.system()},
        {"role": "user", "content": template.render(sentence=sentence, cited_source=source)},
    ]


async def _judge(llm, sentence: str, source: str) -> tuple[bool, str]:
    data = await llm.chat_json(_messages(sentence, source), temperature=0.0, max_tokens=96)
    support = str(data.get("support", "")).strip().lower()
    reason = str(data.get("reason", "")).strip()
    return support == "yes", reason


async def _rewrite(llm, sentence: str, source: str) -> str:
    """让模型只用原文信息重写这一句。失败返回空串，由调用方删除该行。"""
    prompt = (
        "下面这句话与给定原文不符。请只使用【原文】中的信息重写这句话，"
        "不得添加原文没有的内容。如果原文不足以支撑任何陈述，回复「无法支撑」。\n\n"
        f"【原句】{sentence}\n\n【原文】{source}\n\n"
        "只输出重写后的句子，不要解释。"
    )
    try:
        result = await llm.chat(
            [{"role": "user", "content": prompt}], temperature=0.0, max_tokens=200
        )
        text = (result.text or "").strip()
        if not text or "无法支撑" in text:
            return ""
        return text
    except Exception as exc:  # noqa: BLE001
        logger.warning("verify.rewrite_failed", extra={"extra_fields": {"error": repr(exc)}})
        return ""


async def verify_answer(
    llm,
    *,
    answer: str,
    source_by_sid: dict[str, str],
    enable_rewrite: bool = True,
) -> VerifyResult:
    """逐行校验并修复。source_by_sid: {"S1": 原文片段}"""
    lines = (answer or "").splitlines()
    checkable = set(split_checkable_lines(answer))

    result = VerifyResult(text="")
    output: list[str] = []

    for raw in lines:
        line = raw.strip()
        if line not in checkable:
            output.append(raw)
            continue

        sids = {f"S{number}" for number in CITATION_RE.findall(line)}
        sources = [source_by_sid[sid] for sid in sids if sid in source_by_sid]
        if not sources:
            # 引用了不存在的来源，直接删掉，宁可少一句
            result.checked += 1
            result.removed += 1
            result.failures.append((line, "引用的来源不存在"))
            continue

        result.checked += 1
        ok, reason = await _judge(llm, line, "\n".join(sources))
        if ok:
            result.passed += 1
            output.append(raw)
            continue

        result.failures.append((line, reason))
        if not enable_rewrite:
            result.removed += 1
            continue

        fixed = await _rewrite(llm, line, "\n".join(sources))
        if fixed:
            result.rewritten += 1
            # 重写成功的句子按"通过"计，因为它现在被原文支撑了
            result.passed += 1
            output.append(fixed)
        else:
            result.removed += 1

    result.text = "\n".join(output).strip()
    logger.info(
        "verify.done",
        extra={
            "extra_fields": {
                "checked": result.checked,
                "passed": result.passed,
                "rewritten": result.rewritten,
                "removed": result.removed,
                "pass_rate": round(result.pass_rate, 3),
            }
        },
    )
    return result
