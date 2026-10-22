"""问答接口测试：SSE 协议、安全熔断、会话生命周期、断流保全。

这一层最值得测的不是"能返回 200"，而是四件很容易写错又不容易发现的事：

1. 事件顺序必须是 meta → sources → token* → done。
   citations 晚于 token 发，前端会先渲染裸的 [S1] 再突然变链接。
2. 空命中不许调模型。没有上下文时模型只会编，而且还得为编出来的东西付钱。
3. 安全熔断必须发生在检索之前。说"想自杀"的人不该先等三秒检索。
4. 断流时已生成的部分必须落库。否则用户重进只看到自己的问题，没有回答也没有解释。
"""

from __future__ import annotations

import json
from typing import Any

import pytest
import pytest_asyncio
from httpx import AsyncClient

from app.api import chat as chat_api
from app.generate.answer import NO_CONTEXT_TEXT, AnswerEvent, build_citations
from app.generate.intent import Intent
from app.generate.safety import SafetyVerdict
from app.retrieve.fusion import Hit
from app.retrieve.service import RetrievalOutcome
from app.schemas.chat import ChatRequest
from tests.api.test_kb import create_kb, make_user  # noqa: F401


# ---------------------------------------------------------------- 替身
class FakeRuntime:
    """模型一律不加载。测试跑一次要 40 秒的话，没人会去跑它。"""

    embedder = object()
    index = object()
    reranker = object()
    llm = object()


def make_outcome(*, with_hits: bool) -> RetrievalOutcome:
    if not with_hits:
        return RetrievalOutcome(intent=Intent.DEFINE, strategy="direct", stats={"latency_ms": 12})
    hits = [
        Hit(
            chunk_id="c1",
            score=0.92,
            parent_id="p1",
            content="负强化通过撤销厌恶刺激来增加行为出现的频率。",
            meta={"doc_id": "d1", "page_start": 8, "source_title": "实验心理学"},
        )
    ]
    return RetrievalOutcome(
        hits=hits,
        citations=build_citations(hits),
        intent=Intent.DEFINE,
        strategy="direct",
        stats={"latency_ms": 15},
    )


def parse_sse(raw: str) -> list[tuple[str, dict[str, Any]]]:
    events: list[tuple[str, dict[str, Any]]] = []
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


@pytest_asyncio.fixture
async def chat_client(sqlite_client: AsyncClient, monkeypatch) -> AsyncClient:
    """接口测试夹具。

    两处必须替换：
    - SessionLocal：流内落库不能用请求作用域的 session（那时已关闭），
      而默认的 SessionLocal 指向真实 MySQL。测试必须切到内存库。
    - get_runtime：否则第一次请求会去加载 bge-m3，40 秒起步。
    """
    import app.db.base as db_base

    monkeypatch.setattr(db_base, "SessionLocal", sqlite_client.session_factory)
    monkeypatch.setattr(chat_api, "get_runtime", lambda: FakeRuntime())
    return sqlite_client


async def login(client: AsyncClient) -> tuple[str, dict[str, str]]:
    _, headers = await make_user(client)
    kb_id = await create_kb(client, headers)
    return kb_id, headers


class TestStreamProtocol:
    async def test_event_order_and_content(
        self, chat_client: AsyncClient, monkeypatch
    ) -> None:
        kb_id, headers = await login(chat_client)

        async def fake_retrieve(question: str, **_: Any) -> RetrievalOutcome:
            return make_outcome(with_hits=True)

        async def fake_stream(*_: Any, **__: Any):
            yield AnswerEvent("token", {"text": "负强化是"})
            yield AnswerEvent("token", {"text": "撤销厌恶刺激。"})
            yield AnswerEvent(
                "done",
                {"no_context": False, "text_len": 13, "dangling": [],
                 "coverage": 1.0, "tokens_in": 100, "tokens_out": 20, "latency_ms": 800},
            )

        monkeypatch.setattr(chat_api, "retrieve", fake_retrieve)
        monkeypatch.setattr(chat_api, "generate_stream", fake_stream)

        resp = await chat_client.post(
            "/api/chat", headers=headers, json={"question": "什么是负强化", "kb_ids": [kb_id]}
        )
        assert resp.status_code == 200
        events = parse_sse(resp.text)

        assert [name for name, _ in events] == ["meta", "sources", "token", "token", "done"]
        assert "".join(d["text"] for name, d in events if name == "token") == "负强化是撤销厌恶刺激。"
        assert events[1][1]["citations"][0]["sid"] == "S1"
        assert events[1][1]["citations"][0]["doc"] == "实验心理学"
        assert events[-1][1]["message_id"], "done 里没有 message_id，前端无法定位这条消息"

    async def test_empty_hits_never_calls_model(
        self, chat_client: AsyncClient, monkeypatch
    ) -> None:
        """空命中走真 generate_stream——它内部 hits 为空就不碰 llm。"""
        kb_id, headers = await login(chat_client)

        async def fake_retrieve(question: str, **_: Any) -> RetrievalOutcome:
            return make_outcome(with_hits=False)

        monkeypatch.setattr(chat_api, "retrieve", fake_retrieve)

        resp = await chat_client.post(
            "/api/chat", headers=headers, json={"question": "什么是九型人格", "kb_ids": [kb_id]}
        )
        events = parse_sse(resp.text)

        text = "".join(d["text"] for name, d in events if name == "token")
        assert text == NO_CONTEXT_TEXT
        assert events[-1][1]["no_context"] is True
        assert events[-1][1]["tokens_out"] == 0

    async def test_safety_bypasses_retrieval(
        self, chat_client: AsyncClient, monkeypatch
    ) -> None:
        """熔断发生在检索之前。调用记录是空的，说明真的没下去检索。"""
        kb_id, headers = await login(chat_client)
        called: list[str] = []

        async def fake_retrieve(question: str, **_: Any) -> RetrievalOutcome:
            called.append(question)
            return make_outcome(with_hits=True)

        monkeypatch.setattr(chat_api, "retrieve", fake_retrieve)

        resp = await chat_client.post(
            "/api/chat", headers=headers, json={"question": "我最近总是想自杀", "kb_ids": [kb_id]}
        )
        events = parse_sse(resp.text)

        assert called == [], "安全熔断之后仍然执行了检索"
        assert events[-1][1]["safety"] == "CRISIS"
        assert events[-1][1]["no_context"] is True

    async def test_dangling_citation_surfaced(
        self, chat_client: AsyncClient, monkeypatch
    ) -> None:
        """引用校验只能事后做，所以 done 必须如实报出悬空编号。"""
        kb_id, headers = await login(chat_client)

        async def fake_retrieve(question: str, **_: Any) -> RetrievalOutcome:
            return make_outcome(with_hits=True)

        async def fake_stream(*_: Any, **__: Any):
            yield AnswerEvent("token", {"text": "见 [S7]。"})
            yield AnswerEvent(
                "done",
                {"no_context": False, "text_len": 8, "dangling": ["S7"],
                 "coverage": 0.0, "tokens_in": 10, "tokens_out": 5, "latency_ms": 300},
            )

        monkeypatch.setattr(chat_api, "retrieve", fake_retrieve)
        monkeypatch.setattr(chat_api, "generate_stream", fake_stream)

        resp = await chat_client.post(
            "/api/chat", headers=headers, json={"question": "随便问问", "kb_ids": [kb_id]}
        )
        done = parse_sse(resp.text)[-1][1]

        assert done["dangling"] == ["S7"]
        assert done["coverage"] == 0.0


class TestSessionLifecycle:
    async def test_lazy_creation_and_reuse(self, chat_client: AsyncClient, monkeypatch) -> None:
        kb_id, headers = await login(chat_client)

        async def fake_retrieve(question: str, **_: Any) -> RetrievalOutcome:
            return make_outcome(with_hits=False)

        monkeypatch.setattr(chat_api, "retrieve", fake_retrieve)

        first = await chat_client.post(
            "/api/chat", headers=headers, json={"question": "什么是负强化", "kb_ids": [kb_id]}
        )
        meta = parse_sse(first.text)[0][1]
        assert meta["is_new"] is True
        session_id = meta["session_id"]

        second = await chat_client.post(
            "/api/chat",
            headers=headers,
            json={"question": "那惩罚呢", "kb_ids": [kb_id], "session_id": session_id},
        )
        meta2 = parse_sse(second.text)[0][1]
        assert meta2["is_new"] is False
        assert meta2["session_id"] == session_id

        # 标题取首问前 30 字
        detail = (await chat_client.get(f"/api/sessions/{session_id}", headers=headers)).json()
        assert detail["title"] == "什么是负强化"
        # 两问两答
        assert detail["message_count"] == 4

    async def test_rename_and_messages(self, chat_client: AsyncClient, monkeypatch) -> None:
        kb_id, headers = await login(chat_client)

        async def fake_retrieve(question: str, **_: Any) -> RetrievalOutcome:
            return make_outcome(with_hits=False)

        monkeypatch.setattr(chat_api, "retrieve", fake_retrieve)
        resp = await chat_client.post(
            "/api/chat", headers=headers, json={"question": "什么是负强化", "kb_ids": [kb_id]}
        )
        session_id = parse_sse(resp.text)[0][1]["session_id"]

        renamed = await chat_client.patch(
            f"/api/sessions/{session_id}", headers=headers, json={"title": "负强化专题"}
        )
        assert renamed.json()["title"] == "负强化专题"

        page = (
            await chat_client.get(f"/api/sessions/{session_id}/messages", headers=headers)
        ).json()
        assert [item["role"] for item in page["items"]] == ["user", "assistant"]
        assert page["items"][1]["content"] == NO_CONTEXT_TEXT

        listing = (await chat_client.get("/api/sessions", headers=headers)).json()
        assert listing[0]["id"] == session_id
        assert listing[0]["message_count"] == 2

    async def test_delete_cascades_messages(self, chat_client: AsyncClient, monkeypatch) -> None:
        kb_id, headers = await login(chat_client)

        async def fake_retrieve(question: str, **_: Any) -> RetrievalOutcome:
            return make_outcome(with_hits=False)

        monkeypatch.setattr(chat_api, "retrieve", fake_retrieve)
        resp = await chat_client.post(
            "/api/chat", headers=headers, json={"question": "问题", "kb_ids": [kb_id]}
        )
        session_id = parse_sse(resp.text)[0][1]["session_id"]

        assert (
            await chat_client.delete(f"/api/sessions/{session_id}", headers=headers)
        ).status_code == 200
        assert (
            await chat_client.get(f"/api/sessions/{session_id}", headers=headers)
        ).status_code == 404
        # 会话没了，消息也不该留下来变成孤儿
        page = (
            await chat_client.get(f"/api/sessions/{session_id}/messages", headers=headers)
        )
        assert page.status_code == 404


class TestAuthorization:
    async def test_foreign_kb_is_404(self, chat_client: AsyncClient) -> None:
        """越权返回 404 而不是 403：403 等于确认"这个 ID 存在"。"""
        _, headers_a = await make_user(chat_client)
        _, headers_b = await make_user(chat_client)
        kb_b = await create_kb(chat_client, headers_b)

        resp = await chat_client.post(
            "/api/chat", headers=headers_a, json={"question": "问题", "kb_ids": [kb_b]}
        )
        assert resp.status_code == 404

    async def test_foreign_session_is_404(self, chat_client: AsyncClient, monkeypatch) -> None:
        kb_id, headers_a = await login(chat_client)
        _, headers_b = await make_user(chat_client)

        async def fake_retrieve(question: str, **_: Any) -> RetrievalOutcome:
            return make_outcome(with_hits=False)

        monkeypatch.setattr(chat_api, "retrieve", fake_retrieve)
        resp = await chat_client.post(
            "/api/chat", headers=headers_a, json={"question": "问题", "kb_ids": [kb_id]}
        )
        session_id = parse_sse(resp.text)[0][1]["session_id"]

        assert (
            await chat_client.get(f"/api/sessions/{session_id}", headers=headers_b)
        ).status_code == 404


class TestInterruption:
    async def test_partial_answer_saved_on_disconnect(
        self, chat_client: AsyncClient, monkeypatch
    ) -> None:
        """客户端断开时，已经流出去的字必须落库。

        直接把生成器驱动到第三个事件然后 aclose，模拟"用户关了页面"。
        """
        kb_id, headers = await login(chat_client)
        from sqlalchemy import select

        from app.db.models import Message

        async def fake_retrieve(question: str, **_: Any) -> RetrievalOutcome:
            return make_outcome(with_hits=True)

        async def fake_stream(*_: Any, **__: Any):
            yield AnswerEvent("token", {"text": "负强化"})
            yield AnswerEvent("token", {"text": "是撤销"})
            yield AnswerEvent("token", {"text": "厌恶刺激。"})
            yield AnswerEvent("done", {"no_context": False, "text_len": 0, "dangling": [],
                                       "coverage": 1.0, "tokens_in": 0, "tokens_out": 0,
                                       "latency_ms": 1})

        monkeypatch.setattr(chat_api, "retrieve", fake_retrieve)
        monkeypatch.setattr(chat_api, "generate_stream", fake_stream)

        session_id = "interrupt_session_00000000000001"
        async with chat_client.session_factory() as session:
            from app.db.models import Session as ChatSession

            session.add(
                ChatSession(
                    id=session_id,
                    user_id=(await _user_id(chat_client, headers)),
                    title="断流测试",
                    kb_ids=[kb_id],
                    model="deepseek-flash",
                )
            )
            await session.commit()

        request = ChatRequest(question="什么是负强化", kb_ids=[kb_id])
        agen = chat_api._stream(
            request, session_id=session_id, is_new=False, verdict=SafetyVerdict(triggered=False)
        )
        await agen.__anext__()  # meta
        await agen.__anext__()  # sources
        await agen.__anext__()  # token 1
        await agen.__anext__()  # token 2
        await agen.aclose()  # 用户关掉页面

        async with chat_client.session_factory() as session:
            rows = (
                await session.scalars(
                    select(Message).where(
                        Message.session_id == session_id, Message.role == "assistant"
                    )
                )
            ).all()

        assert rows, "断流后半截答案没有被保存"
        assert rows[-1].content == "负强化是撤销"


async def _user_id(client: AsyncClient, headers: dict[str, str]) -> int:
    me = (await client.get("/api/auth/me", headers=headers)).json()
    return int(me["id"])
