"""检索结果融合与过滤。

设计要点：
1. RRF（Reciprocal Rank Fusion）：只用排名不用分数，避免稠密与稀疏两路分数量纲不同的问题。
   bge-m3 的 sparse 分数与 cosine 分数不可直接加权求和，硬加权要么牺牲一路，要么得先做分数标定。
2. 父块去重：子块命中多个时，同一父块只保留最高分那一个，防止上下文被同段内容占满。
3. 阈值过滤：低分命中宁可不要，也不要拿去污染上下文。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


@dataclass
class Hit:
    """一条检索命中。"""

    chunk_id: str
    score: float
    parent_id: str | None = None
    source: str = "dense"
    content: str = ""
    meta: dict = field(default_factory=dict)

    def key(self) -> str:
        return self.chunk_id


def rrf_fuse(
    ranked_lists: dict[str, list[Hit]],
    *,
    k: int = 60,
    top_n: int | None = None,
) -> list[Hit]:
    """RRF 融合多路召回结果。

    公式：score(d) = Σ 1 / (k + rank_i(d))，rank 从 1 开始。
    k 越大，排名差异被抹得越平；k=60 是经验值，适合 20 条量级的候选列表。

    返回按融合分降序排列的 Hit 列表；同一 chunk 在多路出现时分数累加。
    """
    if k <= 0:
        raise ValueError("k 必须为正整数")

    scores: dict[str, float] = {}
    registry: dict[str, Hit] = {}

    for source, hits in ranked_lists.items():
        for rank, hit in enumerate(hits, start=1):
            key = hit.key()
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            if key not in registry:
                # 记录首次出现的来源与原始分；后续路命中只加分不改来源
                registry[key] = Hit(
                    chunk_id=hit.chunk_id,
                    score=hit.score,
                    parent_id=hit.parent_id,
                    source=source,
                    content=hit.content,
                    meta=dict(hit.meta),
                )

    fused = [
        Hit(
            chunk_id=key,
            score=score,
            parent_id=registry[key].parent_id,
            source=registry[key].source,
            content=registry[key].content,
            meta=registry[key].meta,
        )
        for key, score in scores.items()
    ]
    fused.sort(key=lambda h: h.score, reverse=True)

    if top_n is not None:
        if top_n <= 0:
            raise ValueError("top_n 必须为正整数")
        fused = fused[:top_n]
    return fused


def dedupe_by_parent(hits: Iterable[Hit]) -> list[Hit]:
    """同一父块只保留分数最高的子块。无 parent_id 的按自身去重。"""
    best: dict[str, Hit] = {}
    order: list[str] = []

    for hit in hits:
        key = hit.parent_id or hit.chunk_id
        if key not in best:
            best[key] = hit
            order.append(key)
        elif hit.score > best[key].score:
            best[key] = hit

    return [best[key] for key in order]


def filter_by_threshold(hits: Iterable[Hit], threshold: float) -> list[Hit]:
    """过滤低于阈值的结果。threshold <= 0 表示不过滤。"""
    if threshold <= 0:
        return list(hits)
    return [hit for hit in hits if hit.score >= threshold]


def select_context(
    hits: Iterable[Hit],
    *,
    top_n: int = 6,
    threshold: float = 0.0,
    max_chars: int = 6000,
) -> list[Hit]:
    """最终上下文选择：阈值过滤 → 父块去重 → 截断到 top_n → 按字符预算裁剪。"""
    candidates = filter_by_threshold(hits, threshold)
    candidates = dedupe_by_parent(candidates)

    selected: list[Hit] = []
    used_chars = 0
    for hit in candidates:
        length = len(hit.content)
        if length and used_chars + length > max_chars and selected:
            break
        selected.append(hit)
        used_chars += length
        if len(selected) >= top_n:
            break
    return selected


def build_context_block(hits: list[Hit], *, max_chars: int = 6000) -> str:
    """把命中拼成给模型看的上下文。每条带 [S1] 编号，与 citations 的 sid 对应。"""
    lines: list[str] = []
    for index, hit in enumerate(hits, start=1):
        title = hit.meta.get("source_title", "")
        page = hit.meta.get("page_start", "")
        header = f"[S{index}] {title}" + (f" 第{page}页" if page != "" else "")
        lines.append(header)
        lines.append(hit.content.strip())

    block = "\n".join(lines)
    return block[:max_chars]
