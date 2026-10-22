"""日志：JSON 结构化 + 双输出（控制台 / 文件）+ trace_id 全链路 + 敏感信息脱敏。

设计要点：
1. trace_id 用 ContextVar 存，请求入口设置，全链路自动带上，不用手传。
2. 控制台输出人类可读格式（开发时看得清），文件输出 JSON（便于检索与告警）。
3. 文件按天轮转，保留 config.ini 的 logger.retain_days 天。
4. 敏感字段统一脱敏，不靠调用方自觉。
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import re
import sys
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from typing import Any

from app.config import BASE_DIR, settings

# ---------------------------------------------------------------- trace_id
_trace_id: ContextVar[str] = ContextVar("trace_id", default="-")


def set_trace_id(value: str) -> None:
    _trace_id.set(value)


def get_trace_id() -> str:
    return _trace_id.get()


# ---------------------------------------------------------------- 脱敏
SENSITIVE_KEYS = {
    "password",
    "password_confirm",
    "old_password",
    "new_password",
    "api_key",
    "apikey",
    "token",
    "access_token",
    "authorization",
    "jwt_secret",
    "secret",
    "captcha_code",
}

# 手机号、邮箱、sk- 开头的 key，在文本里出现的也打码
_PATTERNS = [
    (re.compile(r"(1[3-9]\d)\d{4}(\d{4})"), r"\1****\2"),
    (re.compile(r"([\w.+-]{2})[\w.+-]*(@[\w.-]+)"), r"\1***\2"),
    (re.compile(r"(sk-)[A-Za-z0-9_\-]{6,}"), r"\1***"),
]


def mask_value(value: Any) -> Any:
    """值级脱敏：字符串走正则，其他原样。"""
    if isinstance(value, str):
        out = value
        for pattern, repl in _PATTERNS:
            out = pattern.sub(repl, out)
        return out
    return value


def mask_record(data: dict[str, Any]) -> dict[str, Any]:
    """字段级脱敏：命中敏感 key 的一律替换为 ***。"""
    result: dict[str, Any] = {}
    for key, value in data.items():
        if key.lower() in SENSITIVE_KEYS:
            result[key] = "***"
        elif isinstance(value, dict):
            result[key] = mask_record(value)
        else:
            result[key] = mask_value(value)
    return result


# ---------------------------------------------------------------- Formatter
class JsonFormatter(logging.Formatter):
    """文件用：一行一个 JSON 对象。"""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "trace_id": getattr(record, "trace_id", None) or get_trace_id(),
            "module": record.name,
            "event": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        extra = getattr(record, "extra_fields", None)
        if isinstance(extra, dict):
            payload.update(mask_record(extra))
        return json.dumps(mask_record(payload), ensure_ascii=False)


class ConsoleFormatter(logging.Formatter):
    """控制台用：时间 级别 [trace_id] 模块 - 消息。"""

    def format(self, record: logging.LogRecord) -> str:
        trace = getattr(record, "trace_id", None) or get_trace_id()
        base = (
            f"{datetime.fromtimestamp(record.created):%H:%M:%S} "
            f"{record.levelname:<5} [{trace[:8]:<8}] {record.name} - {record.getMessage()}"
        )
        extra = getattr(record, "extra_fields", None)
        if isinstance(extra, dict) and extra:
            base += " | " + json.dumps(mask_record(extra), ensure_ascii=False)
        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)
        return base


# ---------------------------------------------------------------- 初始化
_configured = False


def setup_logging() -> None:
    global _configured
    if _configured:
        return

    conf = settings.logger
    log_dir = Path(conf.log_dir)
    if not log_dir.is_absolute():
        log_dir = BASE_DIR / log_dir

    root = logging.getLogger()
    root.setLevel(getattr(logging, conf.level.upper(), logging.INFO))
    root.handlers.clear()

    if conf.console:
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(ConsoleFormatter())
        root.addHandler(console)

    # 文件日志是附加能力，不是启动前提。
    # 目录只读、磁盘满、文件被别的进程占住、容器没挂卷——都会让 mkdir/open 抛 OSError。
    # 原来的写法直接让进程起不来：为了"多存一份日志"把整个服务打死，不划算。
    # 现在降级成仅控制台输出，并把降级原因记一条 warning。
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.TimedRotatingFileHandler(
            filename=log_dir / conf.file_name,
            when="midnight",
            backupCount=conf.retain_days,
            encoding="utf-8",
        )
        file_handler.setFormatter(JsonFormatter())
        root.addHandler(file_handler)
    except OSError as exc:
        root.warning("日志文件不可写，已降级为仅控制台输出：%s（%s）", log_dir, exc)

    # 第三方库降噪
    for noisy in ("uvicorn.access", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)


class log_event:
    """结构化打日志的便捷入口：

        with log_event(logger, "kb.create", kb_id=..., user_id=...) as ctx:
            ...
    """

    def __init__(self, logger: logging.Logger, event: str, **fields: Any) -> None:
        self.logger = logger
        self.event = event
        self.fields = fields

    def __enter__(self) -> dict[str, Any]:
        return self.fields

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc is not None:
            self.logger.error(
                self.event + ".failed",
                exc_info=True,
                extra={"extra_fields": {**self.fields, "error": repr(exc)}},
            )
            return False
        self.logger.info(self.event, extra={"extra_fields": self.fields})
        return False
