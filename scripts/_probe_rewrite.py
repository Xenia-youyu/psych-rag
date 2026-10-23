"""验证 rewrite 的 max_tokens=200 是否同样被推理吃光。

intent 已经确认：256 时 content 为空、tokens_out 恰好等于上限；放到 2048 才出 JSON。
rewrite 三处写的是 200，如果同模型同现象，每次查询改写也在白跑两趟。
"""

from __future__ import annotations

import asyncio

from app.clients.llm_client import LLMClient
from app.config import settings
from app.core.logging import setup_logging
from app.generate.rewrite import _messages

QUESTION = "自变量和因变量怎么区分"


async def main() -> None:
    setup_logging()
    client = LLMClient(settings.llm.active())
    print(f"模型：{settings.llm.active().model}\n")

    for label, template in (("subquery", "query_rewrite_subquery"), ("stepback", "query_rewrite_stepback")):
        messages = _messages(template, question=QUESTION)
        for tokens in (200, 2048):
            try:
                result = await client.chat_json(
                    messages, temperature=0.1, max_tokens=tokens
                )
                print(f"{label:9} max_tokens={tokens:<5} → {str(result)[:110]}")
            except Exception as exc:  # noqa: BLE001
                print(f"{label:9} max_tokens={tokens:<5} → {type(exc).__name__}: {str(exc)[:90]}")


if __name__ == "__main__":
    asyncio.run(main())
