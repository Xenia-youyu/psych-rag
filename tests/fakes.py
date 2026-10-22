"""测试替身。

FakeLLM 让业务层测试完全不碰网络：
- chat / chat_json 按预设队列返回，用完后返回空
- 记录每次调用的 messages，便于断言"提示词里确实带了上下文"
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from app.clients.llm_client import ChatResult, StreamChunk


class FakeLLM:
    def __init__(
        self,
        chat_replies: list[str] | None = None,
        json_replies: list[dict] | None = None,
    ) -> None:
        self._chat = list(chat_replies or [])
        self._json = list(json_replies or [])
        self.chat_calls: list[list[dict[str, str]]] = []
        self.json_calls: list[list[dict[str, str]]] = []

    async def chat(self, messages, *, temperature=None, top_p=None, max_tokens=None,
                   response_format=None) -> ChatResult:
        self.chat_calls.append(messages)
        text = self._chat.pop(0) if self._chat else ""
        return ChatResult(text=text, tokens_in=100, tokens_out=50, latency_ms=10)

    async def chat_json(self, messages, *, temperature=0.0, max_tokens=None) -> dict:
        self.json_calls.append(messages)
        return self._json.pop(0) if self._json else {}

    async def chat_stream(self, messages, **_kwargs):  # pragma: no cover
        self.chat_calls.append(messages)
        for piece in self._chat:
            yield StreamChunk(text=piece)


class FailingLLM:
    """所有调用都抛异常，用于验证降级路径。"""

    async def chat(self, *args, **kwargs):
        raise RuntimeError("model down")

    async def chat_json(self, *args, **kwargs):
        raise RuntimeError("model down")
