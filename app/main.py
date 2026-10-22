"""FastAPI 入口。

中间件顺序（从外到内）：trace_id -> CORS -> 路由
trace_id 必须在最外层，否则异常处理器里拿不到 trace_id。
"""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.auth import router as auth_router
from app.api.chat import router as chat_router
from app.api.chunks import router as chunks_router
from app.api.documents import router as documents_router
from app.api.kb import router as kb_router
from app.api.recommend import router as recommend_router
from app.config import settings
from app.core.errors import register_exception_handlers
from app.core.logging import get_logger, set_trace_id, setup_logging

setup_logging()
logger = get_logger(__name__)


async def _warmup_models() -> None:
    """后台预热 embedding 与 reranker。

    bge-m3 冷加载 39 秒、reranker 10 秒。不预热的话第一个提问的人要等四十多秒
    才看到第一个字——他大概率会以为服务挂了，然后就没有第二个人了。

    放后台任务里跑：启动不阻塞，加载期间的请求由模型内部的锁排队等它。
    实测预热后首次提问的 retrieve 从 36 秒降到 0.5 秒。
    """
    try:
        from app.api.chat import get_runtime

        runtime = get_runtime()
        await runtime.embedder.encode(["预热"])
        await runtime.reranker.score("预热", ["预热文本"])
        logger.info("warmup.done")
    except Exception as exc:  # noqa: BLE001 - 预热失败不该让服务起不来
        logger.warning("warmup.failed", extra={"extra_fields": {"error": repr(exc)}})


def _warmup_enabled() -> bool:
    # 跑单元测试或临时调试时可以 PSY_WARMUP=0 关掉，省 40 秒
    return os.getenv("PSY_WARMUP", "1").lower() not in ("0", "false", "no")


@asynccontextmanager
async def lifespan(_: FastAPI):
    task = asyncio.create_task(_warmup_models()) if _warmup_enabled() else None
    yield
    if task is not None and not task.done():
        task.cancel()


app = FastAPI(
    title=settings.app.name,
    version="0.1.0",
    description="心理学知识科普 RAG 系统 · 全栈实现",
    lifespan=lifespan,
)

register_exception_handlers(app)


@app.middleware("http")
async def trace_middleware(request: Request, call_next):
    trace_id = request.headers.get("X-Trace-Id") or uuid.uuid4().hex
    set_trace_id(trace_id)
    started = time.perf_counter()

    try:
        response = await call_next(request)
    except Exception:
        logger.exception(
            "http.error",
            extra={"extra_fields": {"path": request.url.path, "method": request.method}},
        )
        raise

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    response.headers["X-Trace-Id"] = trace_id
    logger.info(
        "http.access",
        extra={
            "extra_fields": {
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "latency_ms": elapsed_ms,
            }
        },
    )
    return response


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Trace-Id"],
)

app.include_router(auth_router)
app.include_router(kb_router)
app.include_router(documents_router)
app.include_router(chunks_router)
app.include_router(chat_router)
app.include_router(recommend_router)


@app.get("/api/health", tags=["系统"], summary="健康检查")
async def health() -> JSONResponse:
    """DB 不通也返回 200，但把 db 状态标出来，方便前端显示「服务可用但数据库未连接」。"""
    from app.db.base import ping

    db_ok = await ping()
    return JSONResponse(
        content={
            "status": "ok",
            "db": "up" if db_ok else "down",
            "env": settings.app.env,
            "milvus": f"{settings.milvus.host}:{settings.milvus.port}",
        }
    )
