"""图形验证码。

设计：
- 4 位字符（去掉 0/O/1/I/L 等易混字符），不区分大小写。
- PNG 输出 Base64，前端直接 <img src="data:image/png;base64,...">。
- 存储由调用方决定（M1 落 MySQL captcha 表，M4 可换 Redis），本模块只管生成与比对。
- 只在服务端保存「答案哈希」不存明文，避免库被读走后验证码可复用。
"""

from __future__ import annotations

import base64
import hashlib
import io
import random
import secrets

from PIL import Image, ImageDraw, ImageFont

# 去掉 0 O 1 I L 等易混字符
CHARSET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"

WIDTH, HEIGHT = 120, 40
BG_COLOR = (250, 250, 252)
NOISE_LINES = 3
NOISE_POINTS = 60


def generate_code(length: int | None = None) -> str:
    from app.config import settings

    size = length or settings.security.captcha_length
    return "".join(secrets.choice(CHARSET) for _ in range(size))


def hash_code(code: str) -> str:
    """验证码只存哈希，比对时哈希后比较。"""
    return hashlib.sha256(code.strip().upper().encode("utf-8")).hexdigest()


def verify_code(plain: str, code_hash: str) -> bool:
    if not plain or not code_hash:
        return False
    return hash_code(plain) == code_hash


def render_png(code: str) -> bytes:
    """把验证码画成 PNG 字节。"""
    image = Image.new("RGB", (WIDTH, HEIGHT), BG_COLOR)
    draw = ImageDraw.Draw(image)
    font = _load_font(26)

    # 干扰线
    for _ in range(NOISE_LINES):
        start = (random.randint(0, WIDTH), random.randint(0, HEIGHT))
        end = (random.randint(0, WIDTH), random.randint(0, HEIGHT))
        draw.line([start, end], fill=_random_color(150, 210), width=1)

    # 干扰点
    for _ in range(NOISE_POINTS):
        draw.point(
            (random.randint(0, WIDTH), random.randint(0, HEIGHT)),
            fill=_random_color(140, 200),
        )

    # 字符：逐个画，轻微错位增加识别难度
    step = WIDTH // (len(code) + 1)
    for index, char in enumerate(code):
        x = step * (index + 1) - 10 + random.randint(-3, 3)
        y = random.randint(4, 10)
        draw.text((x, y), char, font=font, fill=_random_color(20, 110))

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def render_base64(code: str) -> str:
    return "data:image/png;base64," + base64.b64encode(render_png(code)).decode("ascii")


def _load_font(size: int) -> ImageFont.ImageFont:
    """优先用系统等宽字体，找不到就退回 PIL 默认字体（仍可读，只是小一点）。"""
    for name in ("arial.ttf", "DejaVuSans.ttf", "consola.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _random_color(low: int, high: int) -> tuple[int, int, int]:
    return (random.randint(low, high), random.randint(low, high), random.randint(low, high))
