"""复现意图分类失败：直接打 LLM，把原始返回打出来看。

线上日志：`intent.llm_failed {"error": "AppError('模型返回格式异常')"}`。
classify() 把它吞成 DEFINE 了，但看不到模型到底回了什么。
"""

from __future__ import annotations

import asyncio
import sys

from app.clients.llm_client import LLMClient
from app.config import settings
from app.core.logging import setup_logging
from app.generate.intent import _build_messages


async def main() -> None:
    question = sys.argv[1] if len(sys.argv) > 1 else "什么样的来访者不适合继续心理治疗？"
    setup_logging()
    client = LLMClient(settings.llm.active())
    messages = _build_messages(question)
    print(f"问题：{question}")
    print(f"模型：{settings.llm.active().model}\n")

    for tokens in (2048, 8192):
        try:
            raw = await client.chat(messages, temperature=0.0, max_tokens=tokens)
        except Exception as exc:  # noqa: BLE001
            print(f"max_tokens={tokens} → {type(exc).__name__}: {str(exc)[:200]}")
            continue
        text = getattr(raw, "content", raw)
        print(f"--- max_tokens={tokens} 原始返回（{len(str(text))} 字符）---")
        print(str(text)[:600] or "(空)")
        print()


if __name__ == "__main__":
    asyncio.run(main())
