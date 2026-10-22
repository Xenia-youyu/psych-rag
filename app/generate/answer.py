"""答案生成与引用体系。

职责边界：
1. 把检索结果拼成带 [S1] 编号的上下文（复用 retrieve/fusion.build_context_block）
2. 调 deepseek-flash 生成三段式答案
3. 抽取答案里的引用编号，做覆盖性检查：悬空引用（答案引用了不存在的 S9）必须被抓出来
4. 识别"知识库未覆盖"的诚实回答

不做的事：不在这里做事实校验（那是 verify.py），也不做安全熔断（那是 safety.py）。
"""

from __future__ import annotations

import re
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from app.core.logging import get_logger
from app.generate.intent import Intent
from app.retrieve.fusion import Hit, build_context_block

logger = get_logger(__name__)

CITATION_RE = re.compile(r"\[S(\d+)\]")

# 模型表示"没查到"时的合理表述，命中即视为诚实回答，不判为失败
NO_CONTEXT_MARKERS = ("未覆盖", "没有找到", "未检索到", "知识库中没有", "暂无相关内容")

# 检索为空时的固定回答。不调模型：模型在没有上下文时只会编。
NO_CONTEXT_TEXT = "当前知识库未覆盖此内容。建议换个问法，或先在知识库中上传相关教材。"


@dataclass
class Citation:
    sid: str
    doc: str
    page: int | None
    chunk_id: str
    excerpt: str
    # 前端点引用要跳到那一页核对，没有 doc_id 就只能显示书名给人看。
    # 而"能自己去翻原文核对"正是这套系统相对纯聊天框的唯一价值。
    doc_id: str = ""

    def to_dict(self) -> dict:
        return {
            "sid": self.sid,
            "doc": self.doc,
            "page": self.page,
            "chunk_id": self.chunk_id,
            "excerpt": self.excerpt,
            "doc_id": self.doc_id,
        }


@dataclass
class AnswerResult:
    text: str
    citations: list[Citation] = field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: int = 0
    no_context: bool = False
    dangling_citations: list[str] = field(default_factory=list)

    @property
    def citation_coverage(self) -> float:
        """引用覆盖率：引用了有效来源的 S 编号数 / 出现的全部编号数。"""
        used = {f"S{m}" for m in CITATION_RE.findall(self.text)}
        if not used:
            return 1.0 if self.no_context else 0.0
        valid = {c.sid for c in self.citations}
        return len(used & valid) / len(used)


def build_citations(hits: list[Hit]) -> list[Citation]:
    """把命中转成引用条目，编号与上下文里的 [S1] 一一对应。"""
    citations: list[Citation] = []
    for index, hit in enumerate(hits, start=1):
        meta = hit.meta or {}
        page = meta.get("page_start")
        citations.append(
            Citation(
                sid=f"S{index}",
                doc=str(meta.get("source_title", "")),
                page=int(page) if isinstance(page, int) or str(page).isdigit() else None,
                chunk_id=hit.chunk_id,
                excerpt=hit.content[:200],
                doc_id=str(meta.get("doc_id", "")),
            )
        )
    return citations


def extract_citation_ids(text: str) -> list[str]:
    """按出现顺序返回去重后的 S 编号。"""
    seen: list[str] = []
    for number in CITATION_RE.findall(text or ""):
        sid = f"S{number}"
        if sid not in seen:
            seen.append(sid)
    return seen


def find_dangling(text: str, citations: list[Citation]) -> list[str]:
    """答案引用了上下文里不存在的编号（模型幻觉的典型症状）。"""
    valid = {c.sid for c in citations}
    return [sid for sid in extract_citation_ids(text) if sid not in valid]


def is_no_context_answer(text: str) -> bool:
    return any(marker in (text or "") for marker in NO_CONTEXT_MARKERS)


def _messages(
    question: str,
    context: str,
    intent: Intent,
    user_level: str,
    evidence_flags: str,
) -> list[dict[str, str]]:
    from app.prompt_loader import registry

    template = registry.get("answer_generate")
    return [
        {"role": "system", "content": template.system()},
        {
            "role": "user",
            "content": template.render(
                question=question,
                context=context,
                intent=intent.value,
                user_level=user_level,
                evidence_flags=evidence_flags,
            ),
        },
    ]


def _evidence_flags(hits: list[Hit]) -> str:
    flags = []
    for hit in hits:
        grade = (hit.meta or {}).get("evidence_grade", "")
        if grade and grade not in flags:
            flags.append(str(grade))
    return "、".join(flags)


async def generate(
    llm,
    *,
    question: str,
    hits: list[Hit],
    intent: Intent,
    user_level: str = "备考学员",
    max_chars: int = 6000,
) -> AnswerResult:
    """生成答案。hits 为空时不调模型，直接给诚实的"未覆盖"回答。"""
    if not hits:
        return AnswerResult(text=NO_CONTEXT_TEXT, no_context=True)

    context = build_context_block(hits, max_chars=max_chars)
    citations = build_citations(hits)
    messages = _messages(question, context, intent, user_level, _evidence_flags(hits))

    started = time.perf_counter()
    # 不写死 max_tokens：跟 config.ini 走。写死 1200 时实测把答案截断在表格中间
    result = await llm.chat(messages)
    latency_ms = int((time.perf_counter() - started) * 1000)

    dangling = find_dangling(result.text, citations)
    if dangling:
        logger.warning(
            "answer.dangling_citations",
            extra={"extra_fields": {"citations": dangling, "query": question[:40]}},
        )

    return AnswerResult(
        text=result.text,
        citations=citations,
        tokens_in=result.tokens_in,
        tokens_out=result.tokens_out,
        latency_ms=latency_ms,
        no_context=is_no_context_answer(result.text),
        dangling_citations=dangling,
    )


@dataclass
class AnswerEvent:
    """流式生成的单个事件。event 名与 SSE 的 event: 字段一一对应。"""

    event: str
    data: dict[str, Any]


async def generate_stream(
    llm,
    *,
    question: str,
    hits: list[Hit],
    intent: Intent,
    user_level: str = "备考学员",
    max_chars: int = 6000,
) -> AsyncIterator[AnswerEvent]:
    """流式生成答案。

    与 generate() 的差别只有一处，但这一处是本质的：**引用校验被推迟到流之后**。

    verify_answer 需要完整答案才能逐行判断，而且它会重写某一行——
    流出去的字收不回来，所以在线链路上只做零成本的悬空引用检测
    （纯正则），把结果放进 done 事件让前端标记，而不是假装校验过了。
    逐行事实校验留给离线评估（M5 黄金集）。
    """
    if not hits:
        yield AnswerEvent("token", {"text": NO_CONTEXT_TEXT})
        yield AnswerEvent(
            "done",
            {
                "no_context": True,
                "text_len": len(NO_CONTEXT_TEXT),
                "dangling": [],
                "coverage": 1.0,
                "tokens_in": 0,
                "tokens_out": 0,
            },
        )
        return

    context = build_context_block(hits, max_chars=max_chars)
    citations = build_citations(hits)
    messages = _messages(question, context, intent, user_level, _evidence_flags(hits))

    started = time.perf_counter()
    pieces: list[str] = []
    tokens_in = tokens_out = 0

    async for chunk in llm.chat_stream(messages):
        # usage 走的是末尾那个空文本 chunk，别和正文混在一起
        if chunk.tokens_in or chunk.tokens_out:
            tokens_in, tokens_out = chunk.tokens_in, chunk.tokens_out
        if not chunk.text:
            continue
        pieces.append(chunk.text)
        yield AnswerEvent("token", {"text": chunk.text})

    text = "".join(pieces)
    dangling = find_dangling(text, citations)
    if dangling:
        logger.warning(
            "answer.dangling_citations",
            extra={"extra_fields": {"citations": dangling, "query": question[:40]}},
        )

    used = {f"S{number}" for number in CITATION_RE.findall(text)}
    valid = {c.sid for c in citations}
    coverage = len(used & valid) / len(used) if used else 0.0

    yield AnswerEvent(
        "done",
        {
            "no_context": is_no_context_answer(text) or not text,
            "text_len": len(text),
            "dangling": dangling,
            "coverage": round(coverage, 4),
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "latency_ms": int((time.perf_counter() - started) * 1000),
        },
    )
