"""LLM 客户端测试：全部走 httpx.MockTransport，不联网、不花钱。"""

from __future__ import annotations

import json

import httpx
import pytest

from app.clients.llm_client import LLMClient, extract_json
from app.config import LLMProviderConf
from app.core.errors import AppError, ErrorCode


def make_conf(**overrides) -> LLMProviderConf:
    base = {
        "model": "deepseek-flash",
        "base_url": "https://api.example.com",
        "api_key": "sk-test",
        "temperature": 0.2,
        "top_p": 0.9,
        "max_tokens": 100,
        "timeout": 5,
        "max_retries": 2,
    }
    base.update(overrides)
    return LLMProviderConf(**base)


def json_response(content: str, status: int = 200, tokens=(10, 20)) -> httpx.Response:
    return httpx.Response(
        status,
        json={
            "choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": tokens[0], "completion_tokens": tokens[1]},
        },
    )


class TestChat:
    async def test_normal_chat(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            assert body["model"] == "deepseek-flash"
            assert body["stream"] is False
            assert request.headers["Authorization"] == "Bearer sk-test"
            return json_response("  负强化增加行为  ")

        client = LLMClient(make_conf(), transport=httpx.MockTransport(handler))
        result = await client.chat([{"role": "user", "content": "问题"}])

        assert result.text == "负强化增加行为"
        assert result.tokens_in == 10
        assert result.tokens_out == 20

    async def test_endpoint_appends_v1_once(self) -> None:
        seen: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(str(request.url))
            return json_response("ok")

        for base in ("https://api.example.com", "https://api.example.com/v1"):
            client = LLMClient(make_conf(base_url=base), transport=httpx.MockTransport(handler))
            await client.chat([{"role": "user", "content": "q"}])

        assert seen[0] == "https://api.example.com/v1/chat/completions"
        assert seen[1] == "https://api.example.com/v1/chat/completions"

    async def test_client_error_not_retried(self) -> None:
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(400, json={"error": "bad request"})

        client = LLMClient(make_conf(), transport=httpx.MockTransport(handler))
        with pytest.raises(AppError) as exc:
            await client.chat([{"role": "user", "content": "q"}])

        assert exc.value.code == ErrorCode.LLM_FAILED
        assert calls["n"] == 1, "4xx 不该重试"

    async def test_server_error_retried_then_failed(self) -> None:
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(500, json={"error": "boom"})

        client = LLMClient(make_conf(max_retries=2), transport=httpx.MockTransport(handler))
        with pytest.raises(AppError):
            await client.chat([{"role": "user", "content": "q"}])

        assert calls["n"] == 3, "1 次初始 + 2 次重试"

    async def test_bad_response_shape(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"unexpected": True})

        client = LLMClient(make_conf(), transport=httpx.MockTransport(handler))
        with pytest.raises(AppError):
            await client.chat([{"role": "user", "content": "q"}])


class TestChatJson:
    async def test_plain_json(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return json_response('{"intent": "CONTRAST"}')

        client = LLMClient(make_conf(), transport=httpx.MockTransport(handler))
        assert await client.chat_json([{"role": "user", "content": "q"}]) == {"intent": "CONTRAST"}

    async def test_fenced_json(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return json_response('```json\n{"intent": "DEFINE"}\n```')

        client = LLMClient(make_conf(), transport=httpx.MockTransport(handler))
        assert (await client.chat_json([{"role": "user", "content": "q"}]))["intent"] == "DEFINE"

    async def test_json_with_explanation_retries_once(self) -> None:
        replies = iter(["我觉得应该是 CONTRAST 吧", '{"intent": "CONTRAST"}'])

        def handler(request: httpx.Request) -> httpx.Response:
            return json_response(next(replies))

        client = LLMClient(make_conf(), transport=httpx.MockTransport(handler))
        assert (await client.chat_json([{"role": "user", "content": "q"}]))["intent"] == "CONTRAST"

    async def test_unparseable_raises(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return json_response("完全不是 JSON")

        client = LLMClient(make_conf(), transport=httpx.MockTransport(handler))
        with pytest.raises(AppError):
            await client.chat_json([{"role": "user", "content": "q"}])


class TestStream:
    async def test_parses_sse_chunks(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            assert body["stream"] is True
            lines = [
                'data: {"choices":[{"delta":{"content":"负"}}]}',
                'data: {"choices":[{"delta":{"content":"强化"}}]}',
                "data: [DONE]",
            ]
            return httpx.Response(200, text="\n\n".join(lines))

        client = LLMClient(make_conf(), transport=httpx.MockTransport(handler))
        chunks = [c async for c in client.chat_stream([{"role": "user", "content": "q"}])]

        assert "".join(c.text for c in chunks) == "负强化"

    async def test_requests_usage_in_stream(self) -> None:
        """流式默认不返回 token 计数，必须显式要 usage，否则落库全是 0。"""
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured.update(json.loads(request.content))
            return httpx.Response(
                200, text='data: {"choices":[{"delta":{"content":"x"}}]}\n\ndata: [DONE]'
            )

        client = LLMClient(make_conf(), transport=httpx.MockTransport(handler))
        _ = [c async for c in client.chat_stream([{"role": "user", "content": "q"}])]

        assert captured["stream_options"] == {"include_usage": True}

    async def test_usage_tail_is_exposed(self) -> None:
        """usage 在末尾一个 choices 为空的 chunk 里，不能被当空白文本丢掉。"""
        def handler(request: httpx.Request) -> httpx.Response:
            lines = [
                'data: {"choices":[{"delta":{"content":"答"}}]}',
                'data: {"choices":[],"usage":{"prompt_tokens":120,"completion_tokens":45}}',
                "data: [DONE]",
            ]
            return httpx.Response(200, text="\n\n".join(lines))

        client = LLMClient(make_conf(), transport=httpx.MockTransport(handler))
        chunks = [c async for c in client.chat_stream([{"role": "user", "content": "q"}])]

        assert "".join(c.text for c in chunks) == "答"
        assert chunks[-1].tokens_in == 120
        assert chunks[-1].tokens_out == 45

    async def test_http_error_raises(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="boom")

        client = LLMClient(make_conf(), transport=httpx.MockTransport(handler))
        with pytest.raises(AppError):
            async for _ in client.chat_stream([{"role": "user", "content": "q"}]):
                pass


class TestExtractJson:
    @pytest.mark.parametrize(
        "raw",
        [
            '{"a": 1}',
            '```json\n{"a": 1}\n```',
            '好的，结果是：{"a": 1}',
            '```\n{"a": 1}\n```',
        ],
    )
    def test_extracts(self, raw: str) -> None:
        assert extract_json(raw) == {"a": 1}

    @pytest.mark.parametrize("raw", ["", "没有 JSON", "[1, 2, 3]", "{坏掉"])
    def test_returns_none(self, raw: str) -> None:
        assert extract_json(raw) is None
