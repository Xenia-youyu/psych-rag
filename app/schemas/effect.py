"""心理学效应相关的响应模型。"""

from __future__ import annotations

from pydantic import BaseModel, Field


class EffectBrief(BaseModel):
    slug: str
    name: str
    name_en: str = ""
    category: str = ""
    founder: str = ""
    year: str = ""
    summary: str


class EffectDetail(EffectBrief):
    detail: str = ""
    experiment: str = ""
    example: str = ""
    source: str = ""
    related: list[EffectBrief] = Field(default_factory=list)


class EffectDaily(BaseModel):
    date: str
    variation: int
    total: int
    items: list[EffectDetail] = Field(default_factory=list)


class EffectList(BaseModel):
    total: int
    categories: list[str] = Field(default_factory=list)
    items: list[EffectBrief] = Field(default_factory=list)
