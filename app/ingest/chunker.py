"""文本切块：中文分隔符递归切分 + 父子块双层。

为什么不用固定字符数硬切：
心理学教材的定义句常常"概念 + 例证"连在一起，按 800 字硬切会把例证切到下一块，
检索到定义时看不到例子，检索到例子时不知道在讲什么。必须优先在中文标点上断开。

两层结构：
- 父块（parent）：1000~1500 字，按语义边界切，作为**送给 LLM 的上下文单位**
- 子块（child）：250~400 字，作为**向量检索单位**，粒度细召回才准
子块命中后回填父块，兼顾召回精度与上下文完整度。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# 中文优先分隔符：段落 > 换行 > 句末标点 > 句中停顿 > 空格 > 字符
# min_size 合并时允许的最大溢出比例（仅此一处允许突破 size）
MIN_SIZE_TOLERANCE = 1.10

DEFAULT_SEPARATORS: list[str] = [
    "\n\n\n",
    "\n\n",
    "\n",
    "。",
    "！",
    "？",
    "；",
    "，",
    "、",
    " ",
    "",
]


@dataclass
class ParentChunk:
    """父块及其子块。"""

    parent_id: str
    content: str
    children: list[str] = field(default_factory=list)

    @property
    def char_count(self) -> int:
        return len(self.content)


def _split_by_separator(text: str, separator: str) -> list[str]:
    """按分隔符切分，并把分隔符保留在前一段末尾（不丢标点）。"""
    if separator == "":
        return list(text)

    parts = text.split(separator)
    pieces: list[str] = []
    for index, part in enumerate(parts):
        if index < len(parts) - 1:
            pieces.append(part + separator)
        else:
            pieces.append(part)
    return [p for p in pieces if p]


def _recursive_split(text: str, separators: list[str], size: int) -> list[str]:
    """递归下探分隔符，直到每片不超过 size。"""
    if len(text) <= size:
        return [text]

    if not separators:
        return [text[i : i + size] for i in range(0, len(text), size)]

    separator, rest = separators[0], separators[1:]

    if separator != "" and separator not in text:
        return _recursive_split(text, rest, size)

    pieces = _split_by_separator(text, separator)
    result: list[str] = []
    for piece in pieces:
        if len(piece) <= size:
            result.append(piece)
        else:
            result.extend(_recursive_split(piece, rest, size))
    return result


def _merge_pieces(pieces: list[str], size: int) -> list[str]:
    """把小片贪心合并到不超过 size 的块。"""
    merged: list[str] = []
    buffer = ""
    for piece in pieces:
        if not buffer:
            buffer = piece
            continue
        if len(buffer) + len(piece) <= size:
            buffer += piece
        else:
            merged.append(buffer)
            buffer = piece
    if buffer:
        merged.append(buffer)
    return merged


def _apply_min_size(chunks: list[str], min_size: int, size: int) -> list[str]:
    """过短的块合并进前一块，避免碎块污染索引。

    size 是硬约束，但碎块留在索引里同样是污染。因此这里允许唯一的例外：
    合并后长度上限为 size * MIN_SIZE_TOLERANCE（10%），超过就保留独立碎块。

    初版实现是无条件合并，会把块撑到 363 字（size=300），测试抓出来后改成带容差。
    """
    if not chunks:
        return []
    limit = int(size * MIN_SIZE_TOLERANCE)
    result: list[str] = []
    for chunk in chunks:
        if len(chunk) < min_size and result and len(result[-1]) + len(chunk) <= limit:
            result[-1] += chunk
        else:
            result.append(chunk)
    return result


def _apply_overlap(chunks: list[str], overlap: int) -> list[str]:
    """给除第一块外的每块前置上一块尾部，形成重叠。"""
    if overlap <= 0 or len(chunks) <= 1:
        return list(chunks)

    result: list[str] = [chunks[0]]
    for index in range(1, len(chunks)):
        tail = chunks[index - 1][-overlap:]
        result.append(tail + chunks[index])
    return result


def split_text(
    text: str,
    *,
    size: int = 800,
    overlap: int = 80,
    min_size: int = 80,
    separators: list[str] | None = None,
) -> list[str]:
    """把长文本切成块。

    单块正文长度 ≤ size（唯一例外：min_size 合并时允许 ≤ size × 1.10），
    重叠部分额外附加，因此单块实际长度 ≤ size × 1.10 + overlap。
    """
    if size <= 0:
        raise ValueError("size 必须为正整数")
    if overlap < 0:
        raise ValueError("overlap 不能为负")
    if overlap >= size:
        raise ValueError("overlap 必须小于 size，否则每块几乎完全重复")

    cleaned = (text or "").strip()
    if not cleaned:
        return []

    seps = separators if separators is not None else DEFAULT_SEPARATORS
    pieces = _recursive_split(cleaned, seps, size)
    merged = _merge_pieces(pieces, size)
    merged = _apply_min_size(merged, min_size, size)
    return _apply_overlap(merged, overlap)


def split_parent_child(
    text: str,
    *,
    parent_size: int = 1200,
    parent_min_size: int = 800,
    child_size: int = 320,
    child_min_size: int = 200,
    child_overlap: int = 40,
    separators: list[str] | None = None,
    id_prefix: str = "p",
) -> list[ParentChunk]:
    """切父子块。

    父块之间不设重叠（重叠会让同一段内容在上下文里出现两次，浪费 token 且干扰模型判断），
    子块之间保留小重叠。
    """
    parents = split_text(
        text,
        size=parent_size,
        overlap=0,
        min_size=parent_min_size,
        separators=separators,
    )

    result: list[ParentChunk] = []
    for index, parent_text in enumerate(parents):
        children = split_text(
            parent_text,
            size=child_size,
            overlap=child_overlap,
            min_size=child_min_size,
            separators=separators,
        )
        result.append(
            ParentChunk(
                parent_id=f"{id_prefix}{index:04d}",
                content=parent_text,
                children=children,
            )
        )
    return result


def split_by_pages(pages: list[tuple[int, str]], **kwargs) -> list[tuple[str, int, int]]:
    """按页切分，返回 [(chunk_text, page_start, page_end)]。

    逐页处理，避免跨页切断；页码用于溯源引用。
    """
    size = kwargs.get("size", 800)
    overlap = kwargs.get("overlap", 80)
    min_size = kwargs.get("min_size", 80)
    separators = kwargs.get("separators")

    output: list[tuple[str, int, int]] = []
    for page_no, page_text in pages:
        for chunk in split_text(
            page_text,
            size=size,
            overlap=overlap,
            min_size=min_size,
            separators=separators,
        ):
            output.append((chunk, page_no, page_no))
    return output


CN_SENTENCE_END = re.compile(r"[。！？；]")
