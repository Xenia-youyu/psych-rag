"""问答与会话接口。

三个设计决定，每个都是踩过的坑：

1. **流里不能用 SessionDep。**
   FastAPI 在开始返回响应时就退出依赖上下文、关掉 AsyncSession。
   在 StreamingResponse 的生成器里再用它必然 `Session is closed`，
   所以流内所有落库都自己开 SessionLocal。

2. **引用在 sources 事件里先发，不等 done。**
   等生成完才发 citations，用户会先看到一堆裸的 [S1] 文本，
   再突然变成可点链接——这种跳变比不支持引用还难看。

3. **断流也要落库。**
   用户关掉页面时生成器收到 CancelledError，已生成的部分必须保存，
   否则他回来只看到自己孤零零的问题，没有回答也没有解释。

刻意不做的事：
- 不做断线续传（Last-Event-ID）。重连要重跑检索和生成，代价远大于重问一次。
- 不在线跑 verify_answer。它每行调一次模型，30 行答案等于 30 次请求，
  延迟三十秒起步。在线只做零成本的悬空引用检测。
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse
from sqlalchemy import delete, func, select, update

from app.clients.embed_client import build_embedder
from app.clients.llm_client import LLMClient
from app.clients.milvus_client import MilvusIndex
from app.clients.rerank_client import BGEReranker
from app.config import settings
from app.core.deps import CurrentUser, SessionDep
from app.core.errors import AppError, ErrorCode
from app.core.logging import get_logger
from app.db.models import Document, KnowledgeBase, Message
from app.db.models import Session as ChatSession
from app.generate import safety
from app.generate.answer import generate_stream
from app.retrieve.service import retrieve
from app.schemas.chat import (
    ChatMessageOut,
    ChatRequest,
    MessagePage,
    SessionOut,
    SessionUpdate,
)

router = APIRouter(prefix="/api", tags=["问答"])
logger = get_logger(__name__)

# 新会话标题取首问前 30 字。再长侧边栏就折成三行了。
TITLE_LIMIT = 30


# ---------------------------------------------------------------- 运行时依赖
class ChatRuntime:
    """问答链路用到的重对象。

    bge-m3 冷加载 39 秒、reranker 10 秒，逐请求新建等于服务不可用。
    所以做成进程级单例，第一次请求付这个代价，之后复用。
    """

    def __init__(self) -> None:
        self.embedder = build_embedder()
        self.index = MilvusIndex()
        self.reranker = BGEReranker()
        self.llm = LLMClient(settings.llm.active())


_runtime: ChatRuntime | None = None


def get_runtime() -> ChatRuntime:
    global _runtime
    if _runtime is None:
        _runtime = ChatRuntime()
    return _runtime


def set_runtime(runtime: ChatRuntime | None) -> None:
    """测试注入替身，避免加载真模型。"""
    global _runtime
    _runtime = runtime


def _make_doc_meta_provider():
    """给命中的文档补书名。

    没有它，引用里 `doc` 是空字符串——用户看到「[S1] 第 8 页」却不知道是哪本书。
    retrieve 内部按 doc_id 做了缓存，一本命多次只查一次库。
    """
    from app.db.base import SessionLocal

    async def provider(doc_id: str) -> dict[str, str]:
        async with SessionLocal() as session:
            document = await session.get(Document, doc_id)
            if document is None:
                return {}
            return {"title": Path(document.filename).stem}

    return provider


# ---------------------------------------------------------------- SSE
def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _persist_assistant(
    *,
    session_id: str,
    text: str,
    citations: list[dict[str, Any]],
    done: dict[str, Any],
) -> int | None:
    """落 assistant 消息。返回消息 id，空回答不落库。

    空文本只出现在"生成失败"的路径上，落一条空消息会让前端显示一个空气泡，
    用户看不懂发生了什么——错误已经由 error 事件说明了。
    """
    if not text:
        return None

    from app.db.base import SessionLocal

    async with SessionLocal() as session:
        message = Message(
            session_id=session_id,
            role="assistant",
            content=text,
            citations=citations,
            tokens=int(done.get("tokens_in", 0)) + int(done.get("tokens_out", 0)),
            latency_ms=int(done.get("latency_ms", 0)),
        )
        session.add(message)
        # 会话列表按 updated_at 倒序，写完一条要把它顶上去
        await session.execute(
            update(ChatSession)
            .where(ChatSession.id == session_id)
            .values(updated_at=func.now())
        )
        await session.commit()
        return message.id


async def _persist_guarded(**kwargs: Any) -> None:
    """断开连接时落库不能被打断。

    客户端断开会取消当前 task，裸 await 会立刻抛 CancelledError，
    写一半的数据就丢了。shield 让写入在后台跑完。
    """
    try:
        task = asyncio.create_task(_persist_assistant(**kwargs))
        await asyncio.shield(task)
    except asyncio.CancelledError:
        pass
    except Exception:  # noqa: BLE001 - 落库失败不该再影响已经结束的请求
        logger.exception("chat.persist_failed", extra={"extra_fields": {"session": kwargs.get("session_id")}})


async def _stream(
    request: ChatRequest,
    *,
    session_id: str,
    is_new: bool,
    verdict: safety.SafetyVerdict,
) -> AsyncIterator[str]:
    runtime = get_runtime()
    started = time.perf_counter()

    pieces: list[str] = []
    citations: list[dict[str, Any]] = []
    done: dict[str, Any] = {
        "no_context": False,
        "text_len": 0,
        "dangling": [],
        "coverage": 1.0,
        "tokens_in": 0,
        "tokens_out": 0,
        "latency_ms": 0,
    }
    error: dict[str, Any] | None = None

    yield _sse("meta", {"session_id": session_id, "is_new": is_new})

    try:
        if verdict.triggered:
            # 安全熔断：不检索、不调模型。一个说"想自杀"的人不该先等检索三秒。
            text = safety.fallback_reply(verdict)
            pieces.append(text)
            yield _sse("token", {"text": text})
            done.update(
                {
                    "text_len": len(text),
                    "no_context": True,
                    "safety": verdict.trigger_type.value if verdict.trigger_type else "SAFETY",
                }
            )
        else:
            outcome = await retrieve(
                request.question,
                kb_ids=request.kb_ids,
                embedder=runtime.embedder,
                index=runtime.index,
                reranker=runtime.reranker,
                llm=runtime.llm,
                doc_meta=_make_doc_meta_provider(),
            )
            citations = [citation.to_dict() for citation in outcome.citations]
            yield _sse(
                "sources",
                {
                    "citations": citations,
                    "hit_count": len(outcome.hits),
                    "intent": outcome.intent.value,
                    "strategy": outcome.strategy,
                    "retrieve_ms": int(outcome.stats.get("latency_ms", 0)),
                },
            )

            async for event in generate_stream(
                runtime.llm,
                question=request.question,
                hits=outcome.hits,
                intent=outcome.intent,
                user_level=request.user_level,
            ):
                if event.event == "token":
                    pieces.append(str(event.data.get("text", "")))
                    yield _sse("token", event.data)
                elif event.event == "done":
                    done.update(event.data)

    except (asyncio.CancelledError, GeneratorExit):
        # 客户端断开。已生成的部分要保住，否则用户重进只看到自己的问题。
        # CancelledError 是 Starlette 取消 task 的路径，GeneratorExit 是
        # 生成器被 aclose 的路径——两条都会出现在真实断连里，都得接。
        logger.info(
            "chat.interrupted",
            extra={
                "extra_fields": {
                    "session": session_id,
                    "chars": sum(len(p) for p in pieces),
                }
            },
        )
        await _persist_guarded(
            session_id=session_id,
            text="".join(pieces),
            citations=citations,
            done={**done, "latency_ms": int((time.perf_counter() - started) * 1000)},
        )
        raise
    except AppError as exc:
        logger.warning("chat.app_error", extra={"extra_fields": {"code": exc.code.value}})
        error = {"code": exc.code.value, "message": exc.message}
    except Exception:  # noqa: BLE001 - 任何异常都必须转成事件，不能静默断流
        logger.exception("chat.failed", extra={"extra_fields": {"session": session_id}})
        error = {"code": ErrorCode.INTERNAL.value, "message": "服务内部错误，请稍后重试"}

    if not done.get("latency_ms"):
        done["latency_ms"] = int((time.perf_counter() - started) * 1000)

    message_id = await _persist_assistant(
        session_id=session_id,
        text="".join(pieces),
        citations=citations,
        done=done,
    )
    done["message_id"] = message_id

    if error:
        yield _sse("error", error)
        return
    yield _sse("done", done)


@router.post("/chat", summary="问答（SSE 流式）")
async def chat(request: ChatRequest, user: CurrentUser, session: SessionDep) -> StreamingResponse:
    # 1. 知识库归属。越权一律 404，不泄露"存在但没权限"
    for kb_id in request.kb_ids:
        kb = await session.scalar(select(KnowledgeBase).where(KnowledgeBase.id == kb_id))
        if kb is None or kb.owner_id != user.id:
            raise AppError(ErrorCode.KB_NOT_FOUND)

    # 2. 安全熔断，在检索之前
    verdict = safety.check(request.question)

    # 3. 会话：有 id 校验归属，没有就懒创建
    is_new = False
    if request.session_id:
        chat_session = await session.get(ChatSession, request.session_id)
        if chat_session is None or chat_session.user_id != user.id:
            raise AppError(ErrorCode.SESSION_NOT_FOUND)
        session_id = chat_session.id
    else:
        is_new = True
        session_id = uuid.uuid4().hex[:32]
        session.add(
            ChatSession(
                id=session_id,
                user_id=user.id,
                title=request.question[:TITLE_LIMIT] or "新对话",
                kb_ids=list(request.kb_ids),
                model=settings.llm.active().model,
                params={"user_level": request.user_level},
            )
        )
        # 没有 relationship 时 SQLAlchemy 排不出插入顺序，
        # 先 flush 保证 messages 的外键能落上
        await session.flush()

    # 4. 用户消息先落库：即使生成中断，问题不会丢
    session.add(Message(session_id=session_id, role="user", content=request.question))
    await session.commit()

    logger.info(
        "chat.start",
        extra={
            "extra_fields": {
                "session_id": session_id,
                "is_new": is_new,
                "kb_count": len(request.kb_ids),
                "safety": verdict.triggered,
            }
        },
    )

    return StreamingResponse(
        _stream(request, session_id=session_id, is_new=is_new, verdict=verdict),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # nginx 默认会缓冲整个响应，不关掉的话流式等于没有（用户等全文一次性蹦出来）
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------- 会话
async def _get_owned_session(session_id: str, user_id: int, session: SessionDep) -> ChatSession:
    row = await session.get(ChatSession, session_id)
    if row is None or row.user_id != user_id:
        raise AppError(ErrorCode.SESSION_NOT_FOUND)
    return row


@router.get("/sessions", response_model=list[SessionOut], summary="会话列表")
async def list_sessions(
    user: CurrentUser,
    session: SessionDep,
    page: int = Query(default=1, ge=1),
    size: int = Query(default=20, ge=1, le=100),
) -> list[SessionOut]:
    rows = (
        await session.scalars(
            select(ChatSession)
            .where(ChatSession.user_id == user.id)
            .order_by(ChatSession.updated_at.desc())
            .offset((page - 1) * size)
            .limit(size)
        )
    ).all()
    if not rows:
        return []

    # 一条 group by 查完，不逐会话 count（那是 N+1）
    counts = dict(
        (
            await session.execute(
                select(Message.session_id, func.count(Message.id))
                .where(Message.session_id.in_([row.id for row in rows]))
                .group_by(Message.session_id)
            )
        ).all()
    )
    return [
        SessionOut.model_validate(row).model_copy(
            update={"message_count": int(counts.get(row.id, 0))}
        )
        for row in rows
    ]


@router.get("/sessions/{session_id}", response_model=SessionOut, summary="会话详情")
async def get_session(
    session_id: str, user: CurrentUser, session: SessionDep
) -> SessionOut:
    row = await _get_owned_session(session_id, user.id, session)
    total = int(
        await session.scalar(
            select(func.count(Message.id)).where(Message.session_id == session_id)
        )
        or 0
    )
    return SessionOut.model_validate(row).model_copy(update={"message_count": total})


@router.patch("/sessions/{session_id}", response_model=SessionOut, summary="修改会话标题")
async def rename_session(
    session_id: str, payload: SessionUpdate, user: CurrentUser, session: SessionDep
) -> SessionOut:
    row = await _get_owned_session(session_id, user.id, session)
    row.title = payload.title
    await session.commit()
    await session.refresh(row)
    return SessionOut.model_validate(row)


@router.delete("/sessions/{session_id}", summary="删除会话")
async def delete_session(
    session_id: str, user: CurrentUser, session: SessionDep
) -> dict[str, str]:
    await _get_owned_session(session_id, user.id, session)
    # 显式先删消息：不依赖 MySQL 的 ON DELETE CASCADE，
    # 因为接口测试跑在 SQLite 上，那边默认不启用外键约束
    await session.execute(delete(Message).where(Message.session_id == session_id))
    await session.execute(delete(ChatSession).where(ChatSession.id == session_id))
    await session.commit()
    return {"message": "会话已删除"}


@router.get(
    "/sessions/{session_id}/messages", response_model=MessagePage, summary="会话消息历史"
)
async def list_messages(
    session_id: str,
    user: CurrentUser,
    session: SessionDep,
    page: int = Query(default=1, ge=1),
    size: int = Query(default=50, ge=1, le=200),
) -> MessagePage:
    await _get_owned_session(session_id, user.id, session)

    total = int(
        await session.scalar(
            select(func.count(Message.id)).where(Message.session_id == session_id)
        )
        or 0
    )
    rows = (
        await session.scalars(
            select(Message)
            .where(Message.session_id == session_id)
            .order_by(Message.id.asc())
            .offset((page - 1) * size)
            .limit(size)
        )
    ).all()
    return MessagePage(
        items=[ChatMessageOut.model_validate(row) for row in rows],
        total=total,
        page=page,
        size=size,
    )
