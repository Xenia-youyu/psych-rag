"""统一异常与错误码。

原则：
1. 对外只暴露稳定的错误码 + 中文提示，不暴露堆栈与内部细节。
2. 鉴权失败一律 401；越权访问资源一律 404（不返回 403，避免被用来探测资源是否存在）。
3. 每个错误都带 trace_id，方便用户报障时定位。
"""

from __future__ import annotations

from enum import StrEnum

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.core.logging import get_logger, get_trace_id

logger = get_logger(__name__)


class ErrorCode(StrEnum):
    # 通用
    INTERNAL = "INTERNAL"
    VALIDATION = "VALIDATION_ERROR"
    NOT_FOUND = "NOT_FOUND"
    FORBIDDEN = "FORBIDDEN"

    # 鉴权
    UNAUTHORIZED = "UNAUTHORIZED"
    TOKEN_EXPIRED = "TOKEN_EXPIRED"
    TOKEN_INVALID = "TOKEN_INVALID"
    CAPTCHA_INVALID = "CAPTCHA_INVALID"
    CAPTCHA_EXPIRED = "CAPTCHA_EXPIRED"
    USERNAME_EXISTS = "USERNAME_EXISTS"
    CREDENTIALS_INVALID = "CREDENTIALS_INVALID"

    # 知识库
    KB_EMPTY = "KB_EMPTY"
    KB_NOT_FOUND = "KB_NOT_FOUND"
    DOC_NOT_FOUND = "DOC_NOT_FOUND"
    FILE_TYPE_UNSUPPORTED = "FILE_TYPE_UNSUPPORTED"
    FILE_TOO_LARGE = "FILE_TOO_LARGE"
    PARSE_FAILED = "PARSE_FAILED"

    # 检索与生成
    NO_RETRIEVAL = "NO_RETRIEVAL"
    LLM_FAILED = "LLM_FAILED"

    # 会话
    SESSION_NOT_FOUND = "SESSION_NOT_FOUND"


MESSAGES: dict[ErrorCode, str] = {
    ErrorCode.INTERNAL: "服务内部错误，请稍后重试",
    ErrorCode.VALIDATION: "请求参数不合法",
    ErrorCode.NOT_FOUND: "资源不存在",
    ErrorCode.FORBIDDEN: "没有权限执行该操作",
    ErrorCode.UNAUTHORIZED: "请先登录",
    ErrorCode.TOKEN_EXPIRED: "登录已过期，请重新登录",
    ErrorCode.TOKEN_INVALID: "登录凭证无效，请重新登录",
    ErrorCode.CAPTCHA_INVALID: "验证码错误",
    ErrorCode.CAPTCHA_EXPIRED: "验证码已过期，请刷新",
    ErrorCode.USERNAME_EXISTS: "该账号已被注册",
    ErrorCode.CREDENTIALS_INVALID: "账号或密码错误",
    ErrorCode.KB_EMPTY: "所选知识库没有可用内容",
    ErrorCode.KB_NOT_FOUND: "知识库不存在",
    ErrorCode.DOC_NOT_FOUND: "文档不存在",
    ErrorCode.FILE_TYPE_UNSUPPORTED: "不支持该文件类型",
    ErrorCode.FILE_TOO_LARGE: "文件超过大小上限",
    ErrorCode.PARSE_FAILED: "文档解析失败",
    ErrorCode.NO_RETRIEVAL: "未检索到相关内容",
    ErrorCode.LLM_FAILED: "模型服务暂不可用，请稍后重试",
    ErrorCode.SESSION_NOT_FOUND: "会话不存在",
}

HTTP_STATUS: dict[ErrorCode, int] = {
    ErrorCode.INTERNAL: status.HTTP_500_INTERNAL_SERVER_ERROR,
    ErrorCode.VALIDATION: status.HTTP_400_BAD_REQUEST,
    ErrorCode.NOT_FOUND: status.HTTP_404_NOT_FOUND,
    ErrorCode.KB_NOT_FOUND: status.HTTP_404_NOT_FOUND,
    ErrorCode.DOC_NOT_FOUND: status.HTTP_404_NOT_FOUND,
    ErrorCode.FORBIDDEN: status.HTTP_403_FORBIDDEN,
    ErrorCode.UNAUTHORIZED: status.HTTP_401_UNAUTHORIZED,
    ErrorCode.TOKEN_EXPIRED: status.HTTP_401_UNAUTHORIZED,
    ErrorCode.TOKEN_INVALID: status.HTTP_401_UNAUTHORIZED,
    ErrorCode.CREDENTIALS_INVALID: status.HTTP_401_UNAUTHORIZED,
    ErrorCode.USERNAME_EXISTS: status.HTTP_409_CONFLICT,
    ErrorCode.CAPTCHA_INVALID: status.HTTP_400_BAD_REQUEST,
    ErrorCode.CAPTCHA_EXPIRED: status.HTTP_400_BAD_REQUEST,
    ErrorCode.FILE_TYPE_UNSUPPORTED: status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
    ErrorCode.FILE_TOO_LARGE: status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
    ErrorCode.KB_EMPTY: status.HTTP_400_BAD_REQUEST,
    ErrorCode.LLM_FAILED: status.HTTP_503_SERVICE_UNAVAILABLE,
    ErrorCode.SESSION_NOT_FOUND: status.HTTP_404_NOT_FOUND,
}


class AppError(Exception):
    """业务异常。抛出后由全局处理器转成统一响应。"""

    def __init__(
        self,
        code: ErrorCode,
        message: str | None = None,
        *,
        http_status: int | None = None,
    ) -> None:
        self.code = code
        self.message = message or MESSAGES.get(code, "请求处理失败")
        self.http_status = http_status or HTTP_STATUS.get(code, status.HTTP_400_BAD_REQUEST)
        super().__init__(self.message)

    def to_response(self) -> JSONResponse:
        return JSONResponse(
            status_code=self.http_status,
            content={
                "code": self.code.value,
                "message": self.message,
                "trace_id": get_trace_id(),
            },
        )


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def _app_error(_: Request, exc: AppError) -> JSONResponse:
        logger.warning(
            "app_error",
            extra={"extra_fields": {"code": exc.code.value, "message": exc.message}},
        )
        return exc.to_response()

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        details = [
            {"field": ".".join(str(p) for p in err["loc"][1:]), "reason": err["msg"]}
            for err in exc.errors()
        ]
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "code": ErrorCode.VALIDATION.value,
                "message": "请求参数不合法",
                "details": details,
                "trace_id": get_trace_id(),
            },
        )

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled_error")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "code": ErrorCode.INTERNAL.value,
                "message": MESSAGES[ErrorCode.INTERNAL],
                "trace_id": get_trace_id(),
            },
        )
