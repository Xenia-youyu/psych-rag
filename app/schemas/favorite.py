"""收藏相关的请求响应模型。"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class CitationIn(BaseModel):
    """一条引用。字段与问答流下发的 citations 保持一致，便于前端直接透传。"""

    sid: str = ""
    doc_title: str = ""
    page: int = 0
    page_end: int = 0
    content: str = ""


class FavoriteCreate(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    answer: str = Field(min_length=1)
    session_id: str = Field(default="", max_length=32)
    note: str = Field(default="", max_length=500)
    citations: list[CitationIn] = Field(default_factory=list, max_length=20)


class FavoriteUpdate(BaseModel):
    note: str = Field(default="", max_length=500)


class FavoriteOut(BaseModel):
    id: str
    question: str
    answer: str
    note: str
    session_id: str
    source: str
    citations: list[CitationIn] = Field(default_factory=list)
    created_at: datetime | None = None


class FavoritePage(BaseModel):
    total: int
    page: int
    size: int
    items: list[FavoriteOut] = Field(default_factory=list)
