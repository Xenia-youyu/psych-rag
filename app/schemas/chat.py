"""问答与会话的请求响应模型。

命名上刻意避开 `MessageOut`——auth 和 kb 里各有一个同名的，
再加一个会让 import 变成猜谜游戏。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    """一次问答请求。

    session_id 为空表示"新对话"：后端懒创建会话并在 meta 事件里回传 id。
    所以不需要单独的「创建会话」接口，也不会留下空会话垃圾数据。
    """

    question: str = Field(min_length=1, max_length=1000)
    kb_ids: list[str] = Field(default_factory=list, max_length=10)
    session_id: str | None = Field(default=None, max_length=32)
    model: str | None = Field(default=None, max_length=64)
    user_level: str = Field(default="备考学员", max_length=32)

    @field_validator("question")
    @classmethod
    def _strip(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("问题不能为空")
        return value


class CitationOut(BaseModel):
    sid: str
    doc: str
    page: int | None = None
    chunk_id: str
    excerpt: str = ""


class SessionUpdate(BaseModel):
    title: str = Field(min_length=1, max_length=60)

    @field_validator("title")
    @classmethod
    def _strip(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("标题不能为空")
        return value


class SessionOut(BaseModel):
    id: str
    title: str
    kb_ids: list[str] = Field(default_factory=list)
    model: str = "deepseek-flash"
    message_count: int = 0
    created_at: datetime | None = None
    updated_at: datetime | None = None

    model_config = {"from_attributes": True}


class ChatMessageOut(BaseModel):
    id: int
    session_id: str
    role: Literal["user", "assistant"]
    content: str
    citations: list[dict[str, Any]] = Field(default_factory=list)
    tokens: int = 0
    latency_ms: int = 0
    created_at: datetime | None = None

    model_config = {"from_attributes": True}


class MessagePage(BaseModel):
    items: list[ChatMessageOut]
    total: int
    page: int
    size: int
