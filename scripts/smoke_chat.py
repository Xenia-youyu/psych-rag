"""用真实的模型与真实检索跑一次流式问答。

与 test_chat.py 的分工：测试用替身保证协议正确，这个脚本证明**整条链路真的能跑**。
替身能验事件顺序，验不了 bge-m3 的向量能不能检索到、DeepSeek 会不会真的流式吐字。

用法：

    export JWT_SECRET=... DEEPSEEK_API_KEY=...
    .venv/Scripts/python.exe scripts/smoke_chat.py --question "实验中的自变量怎么控制"

它会：
1. 在库里造一个临时用户（或复用），签一个真 token
2. 用 ASGITransport 直接调 /api/chat（不启 uvicorn，省一个进程）
3. 边收事件边打时间戳，让"首字延迟"和"总耗时"都看得见
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time

import httpx
from sqlalchemy import select

from app.core.security import create_access_token
from app.db.base import SessionLocal
from app.db.models import Document, KnowledgeBase, User


async def pick_target(kb_id: str | None) -> tuple[str, str, int, str]:
    """挑一个有数据的知识库，并借它所有者的身份签 token。

    不能随手造个新用户：/api/chat 会校验 kb 归属，新用户名下没有库，
    一定会 404。这里签的是真实所有者的 token，与前端登录后拿到的完全一样。
    """
    async with SessionLocal() as session:
        if kb_id:
            kb = await session.get(KnowledgeBase, kb_id)
        else:
            # 不能按 doc_count 挑：脚本直接建文档记录时这个计数曾经没同步过。
            # 从 documents 表取最近一条，才是真的"有数据的库"。
            latest = await session.scalar(
                select(Document.kb_id).order_by(Document.created_at.desc()).limit(1)
            )
            kb = await session.get(KnowledgeBase, latest) if latest else None

        if kb is None:
            raise SystemExit("找不到有数据的知识库。先跑 scripts/ingest_batch.py 入一批书。")

        owner = await session.get(User, kb.owner_id)
        if owner is None:
            raise SystemExit(f"知识库 {kb.id} 的 owner {kb.owner_id} 不存在")

        token, _ = create_access_token(owner.id, owner.username)
        return kb.id, kb.name, owner.id, token


def parse_stream(raw: str) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    for block in raw.strip().split("\n\n"):
        name, payload = "", None
        for line in block.splitlines():
            if line.startswith("event: "):
                name = line[7:]
            elif line.startswith("data: "):
                payload = json.loads(line[6:])
        if name:
            events.append((name, payload or {}))
    return events


async def warmup() -> float:
    """先把模型加载起来。

    正式部署里这一步由 app/main.py 的 lifespan 在启动时后台做；
    这里用 ASGITransport 直连，不跑 lifespan，所以手动预热一次，
    否则第一问的 retrieve_ms 会把 39 秒的模型加载算进检索耗时。
    """
    from app.api.chat import get_runtime

    started = time.perf_counter()
    runtime = get_runtime()
    await runtime.embedder.encode(["预热"])
    await runtime.reranker.score("预热", ["预热文本"])
    return time.perf_counter() - started


async def run(question: str, kb_id: str | None, token: str) -> int:
    from app.main import app

    transport = httpx.ASGITransport(app=app)
    headers = {"Authorization": f"Bearer {token}"}

    if kb_id is None:
        raise SystemExit("缺少 kb_id")

    payload = {"question": question, "kb_ids": [kb_id]}
    started = time.perf_counter()

    async with httpx.AsyncClient(transport=transport, base_url="http://smoke", timeout=300) as client:
        async with client.stream("POST", "/api/chat", json=payload, headers=headers) as resp:
            if resp.status_code != 200:
                body = await resp.aread()
                print(f"HTTP {resp.status_code}: {body[:400]!r}")
                return 1

            raw = await resp.aread()

    elapsed = time.perf_counter() - started
    events = parse_stream(raw.decode("utf-8"))

    print(f"\n问题：{question}")
    print(f"事件流：{' → '.join(name for name, _ in events)}")
    print(f"总耗时：{elapsed:.1f}s\n")

    for name, data in events:
        if name == "meta":
            print(f"  meta    session={data['session_id']} is_new={data['is_new']}")
        elif name == "sources":
            print(f"  sources 命中 {data['hit_count']} 条 | 意图 {data['intent']} | 策略 {data['strategy']} | 检索 {data['retrieve_ms']}ms")
            for citation in data["citations"]:
                print(f"          [{citation['sid']}] {citation['doc']} p{citation['page']} {citation['excerpt'][:40]}")
        elif name == "token":
            pass
        elif name == "done":
            print(f"  done    {data['text_len']} 字 | {data['latency_ms']}ms | tokens {data['tokens_in']}+{data['tokens_out']} | 未覆盖={data['no_context']} | 悬空引用={data['dangling']}")
        elif name == "error":
            print(f"  error   {data}")

    text = "".join(data["text"] for name, data in events if name == "token")
    print("\n--- 回答 ---")
    print(text[:800])
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="真实链路流式问答冒烟")
    parser.add_argument("--question", default="实验中的自变量和因变量怎么区分")
    parser.add_argument("--kb-id", default=None)
    parser.add_argument("--no-warmup", action="store_true", help="跳过预热（想观察冷启动代价时用）")
    args = parser.parse_args()

    async def _main() -> int:
        kb_id, kb_name, owner_id, token = await pick_target(args.kb_id)
        print(f"知识库 {kb_name} ({kb_id}) | owner {owner_id}")
        if not args.no_warmup:
            cost = await warmup()
            print(f"模型预热 {cost:.1f}s（部署时由 lifespan 后台完成）")
        return await run(args.question, kb_id, token)

    return asyncio.run(_main())


if __name__ == "__main__":
    sys.exit(main())
