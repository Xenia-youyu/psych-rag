"""大模型客户端（OpenAI 兼容协议）。

deepseek 与通义都提供 OpenAI 兼容接口，所以一个 client 覆盖两家，
只需换 base_url / model / api_key。

设计要点：
1. transport 可注入 —— 单元测试用 httpx.MockTransport，不花一分钱、不依赖网络。
2. 超时与重试在本层收口，业务层不关心 httpx 细节。
3. JSON 解析容错：模型爱把 JSON 包在 ```json 里，或前后加解释文字，这里统一剥掉。
4. 任何失败都转成 AppError(LLM_FAILED)，不把 httpx 异常漏到 API 层。
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import httpx

from app.config import LLMProviderConf
from app.core.errors import AppError, ErrorCode
from app.core.logging import get_logger

logger = get_logger(__name__)

_JSON_BLOCK = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


@dataclass
class ChatResult:
    text: str
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: int = 0


@dataclass
class StreamChunk:
    """流式增量。

    text 为空但 tokens 非零，表示这是"用量尾部"——OpenAI 兼容协议在
    stream_options.include_usage 打开时，会在 [DONE] 之前单独发一个
    choices 为空的 chunk 只带 usage。调用方要跳过空文本。
    """

    text: str = ""
    tokens_in: int = 0
    tokens_out: int = 0


class LLMClient:
    """单个 provider 的客户端。业务层按需构造，或注入假 client 做测试。"""

    def __init__(
        self,
        conf: LLMProviderConf,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.conf = conf
        self._transport = transport

    # ------------------------------------------------------------ 内部
    def _build_payload(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None,
        top_p: float | None,
        max_tokens: int | None,
        response_format: dict[str, str] | None,
        stream: bool,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.conf.model,
            "messages": messages,
            "temperature": self.conf.temperature if temperature is None else temperature,
            "top_p": self.conf.top_p if top_p is None else top_p,
            "max_tokens": self.conf.max_tokens if max_tokens is None else max_tokens,
            "stream": stream,
        }
        if response_format:
            payload["response_format"] = response_format
        return payload

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.conf.api_key}",
            "Content-Type": "application/json",
        }

    def _endpoint(self) -> str:
        base = self.conf.base_url.rstrip("/")
        # 允许 base_url 直接写到 /v1 或者更外层
        return f"{base}/chat/completions" if base.endswith("/v1") else f"{base}/v1/chat/completions"

    async def _post(
        self, payload: dict[str, Any], *, timeout: float | None = None
    ) -> httpx.Response:
        limit = timeout or self.conf.timeout
        last_error: Exception | None = None

        for attempt in range(self.conf.max_retries + 1):
            try:
                async with httpx.AsyncClient(transport=self._transport, timeout=limit) as client:
                    response = await client.post(
                        self._endpoint(), headers=self._headers(), json=payload
                    )
                if response.status_code < 400:
                    return response

                # 4xx 不重试（参数错误重试也没用），5xx 与 429 重试
                if response.status_code < 500 and response.status_code != 429:
                    logger.error(
                        "llm.http_error",
                        extra={
                            "extra_fields": {
                                "status": response.status_code,
                                "body": response.text[:300],
                                "model": self.conf.model,
                            }
                        },
                    )
                    raise AppError(ErrorCode.LLM_FAILED)

                last_error = RuntimeError(f"HTTP {response.status_code}")
            except AppError:
                raise
            except Exception as exc:  # noqa: BLE001
                last_error = exc

            if attempt < self.conf.max_retries:
                await asyncio.sleep(0.6 * (2**attempt))

        logger.error(
            "llm.request_failed",
            extra={"extra_fields": {"model": self.conf.model, "error": repr(last_error)}},
        )
        raise AppError(ErrorCode.LLM_FAILED)

    # ------------------------------------------------------------ 非流式
    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        response_format: dict[str, str] | None = None,
    ) -> ChatResult:
        payload = self._build_payload(
            messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            response_format=response_format,
            stream=False,
        )
        started = time.perf_counter()
        response = await self._post(payload)

        try:
            data = response.json()
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, ValueError) as exc:
            logger.error("llm.bad_response", extra={"extra_fields": {"body": response.text[:300]}})
            raise AppError(ErrorCode.LLM_FAILED) from exc

        usage = data.get("usage") or {}
        return ChatResult(
            text=text.strip(),
            tokens_in=int(usage.get("prompt_tokens", 0)),
            tokens_out=int(usage.get("completion_tokens", 0)),
            latency_ms=int((time.perf_counter() - started) * 1000),
        )

    async def chat_json(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """要求模型返回 JSON。解析失败只重试一次，仍失败就抛错而不是猜。"""
        result = await self.chat(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        parsed = extract_json(result.text)
        if parsed is not None:
            return parsed

        # 有些模型对 response_format 支持不完整，追加一句再试一次
        retry_messages = [
            *messages,
            {"role": "user", "content": "只输出 JSON，不要任何解释、不要代码块。"},
        ]
        retry = await self.chat(
            retry_messages, temperature=0.0, max_tokens=max_tokens
        )
        parsed = extract_json(retry.text)
        if parsed is None:
            logger.error("llm.json_parse_failed", extra={"extra_fields": {"text": retry.text[:300]}})
            raise AppError(ErrorCode.LLM_FAILED, "模型返回格式异常")
        return parsed

    # ------------------------------------------------------------ 流式
    async def chat_stream(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[StreamChunk]:
        payload = self._build_payload(
            messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            response_format=None,
            stream=True,
        )
        # 不显式要 usage 的话，OpenAI 兼容协议在流式模式下不返回 token 计数，
        # 于是 psy_messages.tokens 会全是 0——成本统计变成一堆假数据。
        payload["stream_options"] = {"include_usage": True}

        try:
            async with httpx.AsyncClient(
                transport=self._transport, timeout=self.conf.timeout
            ) as client:
                async with client.stream(
                    "POST", self._endpoint(), headers=self._headers(), json=payload
                ) as response:
                    if response.status_code >= 400:
                        body = await response.aread()
                        logger.error(
                            "llm.stream_error",
                            extra={
                                "extra_fields": {
                                    "status": response.status_code,
                                    "body": body[:300].decode("utf-8", "ignore"),
                                }
                            },
                        )
                        raise AppError(ErrorCode.LLM_FAILED)

                    async for line in response.aiter_lines():
                        if not line or not line.startswith("data:"):
                            continue
                        chunk = line[5:].strip()
                        if chunk == "[DONE]":
                            break
                        try:
                            data = json.loads(chunk)
                        except json.JSONDecodeError:
                            continue

                        usage = data.get("usage") or {}
                        if usage:
                            yield StreamChunk(
                                text="",
                                tokens_in=int(usage.get("prompt_tokens", 0) or 0),
                                tokens_out=int(usage.get("completion_tokens", 0) or 0),
                            )

                        try:
                            delta = data["choices"][0]["delta"].get("content")
                        except (KeyError, IndexError, TypeError):
                            continue
                        if delta:
                            yield StreamChunk(text=delta)
        except AppError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.error("llm.stream_failed", extra={"extra_fields": {"error": repr(exc)}})
            raise AppError(ErrorCode.LLM_FAILED) from exc


def extract_json(text: str) -> dict[str, Any] | None:
    """从模型输出里挖出 JSON 对象。

    处理三种常见脏输出：裸 JSON、```json 包裹、前后带解释文字。
    """
    if not text:
        return None

    candidates: list[str] = []
    block = _JSON_BLOCK.search(text)
    if block:
        candidates.append(block.group(1).strip())
    candidates.append(text.strip())

    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])

    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None
