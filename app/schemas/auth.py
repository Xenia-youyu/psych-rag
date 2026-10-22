"""鉴权相关请求/响应模型。

校验规则与前端保持一致：
- 账号 4~32 字符，字母/数字/下划线，首位必须字母
- 密码 8~64 字符，必须同时含字母与数字
"""

from __future__ import annotations

import re
from datetime import datetime

from pydantic import BaseModel, Field, field_validator

USERNAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{3,31}$")


def _check_username(value: str) -> str:
    value = value.strip()
    if not USERNAME_RE.match(value):
        raise ValueError("账号需 4~32 位，以字母开头，仅含字母、数字、下划线")
    return value


def _check_password(value: str) -> str:
    if not 8 <= len(value) <= 64:
        raise ValueError("密码长度需为 8~64 位")
    if not re.search(r"[A-Za-z]", value) or not re.search(r"\d", value):
        raise ValueError("密码需同时包含字母与数字")
    return value


class CaptchaOut(BaseModel):
    captcha_id: str
    image_base64: str
    expires_in: int


class RegisterRequest(BaseModel):
    username: str
    password: str
    password_confirm: str
    captcha_id: str
    captcha_code: str

    @field_validator("username")
    @classmethod
    def _v_username(cls, v: str) -> str:
        return _check_username(v)

    @field_validator("password")
    @classmethod
    def _v_password(cls, v: str) -> str:
        return _check_password(v)

    @field_validator("captcha_code")
    @classmethod
    def _v_captcha(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("请填写验证码")
        return v

    def passwords_match(self) -> bool:
        return self.password == self.password_confirm


class LoginRequest(BaseModel):
    username: str
    password: str
    captcha_id: str
    captcha_code: str

    @field_validator("username")
    @classmethod
    def _v_username(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("请填写账号")
        return v

    @field_validator("password")
    @classmethod
    def _v_password(cls, v: str) -> str:
        if not v:
            raise ValueError("请填写密码")
        return v


class UserOut(BaseModel):
    id: int
    username: str
    created_at: datetime | None = None
    last_login_at: datetime | None = None

    model_config = {"from_attributes": True}


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserOut


class MessageOut(BaseModel):
    message: str = Field(default="操作成功")
