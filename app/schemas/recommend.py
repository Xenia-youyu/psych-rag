"""每日知识推荐的请求响应模型。"""

from __future__ import annotations

from pydantic import BaseModel, Field


class RecommendItem(BaseModel):
    chunk_id: str
    title: str
    preview: str
    doc_id: str
    doc_title: str
    page: int = 0
    char_count: int = 0


class RecommendList(BaseModel):
    date: str
    variation: int
    total_candidates: int
    items: list[RecommendItem] = Field(default_factory=list)


class RelatedItem(BaseModel):
    chunk_id: str
    doc_id: str
    doc_title: str
    page: int = 0
    preview: str


class RecommendDetail(RecommendItem):
    # 详情给父块全文（约 1200 字）。子块只有 300 字，常是半句假设半句结论，
    # 拿它当"详细介绍"会让用户看到一段没头没尾的话。
    content: str
    is_full_parent: bool = False
    related: list[RelatedItem] = Field(default_factory=list)
