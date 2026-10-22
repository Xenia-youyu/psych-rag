"""知识库 / 文档 / 文本块 的请求响应模型。"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class KBCreate(BaseModel):
    name: str = Field(min_length=1, max_length=50)
    description: str = Field(default="", max_length=200)

    @field_validator("name")
    @classmethod
    def _strip(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("知识库名称不能为空")
        return value


class KBUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=50)
    description: str | None = Field(default=None, max_length=200)


class KBOut(BaseModel):
    id: str
    name: str
    description: str
    doc_count: int
    chunk_count: int
    created_at: datetime | None = None
    updated_at: datetime | None = None

    model_config = {"from_attributes": True}


class ParseParams(BaseModel):
    """解析参数。前端「知识库解析页」的表单直接映射到这里。"""

    chunk_size: int = Field(default=800, ge=200, le=1500)
    overlap_ratio: float = Field(default=0.10, ge=0.0, le=0.30)
    hybrid: bool = True
    mode: Literal["single", "parent_child"] = "single"
    parent_size: int = Field(default=1200, ge=800, le=2000)
    child_size: int = Field(default=320, ge=150, le=600)


class DocumentOut(BaseModel):
    id: str
    kb_id: str
    filename: str
    file_type: str
    size_bytes: int
    page_count: int
    status: str
    last_stage: str
    error_msg: str
    parse_params: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime | None = None
    # 命中内容哈希去重时置 true：返回的是库里已存在的文档，本次没有新建。
    # 前端据此提示"该文件已在库中"，而不是让用户以为上传失败。
    duplicated: bool = False

    model_config = {"from_attributes": True}


class ChunkOut(BaseModel):
    id: str
    doc_id: str
    kb_id: str
    parent_id: str | None = None
    seq: int
    content: str
    char_count: int
    page_start: int
    page_end: int
    enabled: bool
    meta: dict[str, Any] = Field(default_factory=dict)

    model_config = {"from_attributes": True}


class ChunkUpdate(BaseModel):
    content: str | None = None
    enabled: bool | None = None

    @field_validator("content")
    @classmethod
    def _check_content(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("内容不能为空")
        if len(value) > 8000:
            raise ValueError("单块内容不能超过 8000 字")
        return value


class PageMeta(BaseModel):
    total: int
    page: int
    size: int


class ChunkPage(BaseModel):
    items: list[ChunkOut]
    total: int
    page: int
    size: int


class MessageOut(BaseModel):
    message: str = "操作成功"
