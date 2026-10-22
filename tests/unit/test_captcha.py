"""图形验证码单元测试。

覆盖：字符集不含易混字符、长度、哈希比对大小写不敏感、PNG 输出合法。
"""

from __future__ import annotations

import base64

from app.core.captcha import (
    CHARSET,
    generate_code,
    hash_code,
    render_base64,
    render_png,
    verify_code,
)

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class TestGenerateCode:
    def test_default_length_from_config(self) -> None:
        assert len(generate_code()) == 4

    def test_custom_length(self) -> None:
        assert len(generate_code(6)) == 6

    def test_charset_excludes_confusing_characters(self) -> None:
        for char in "0O1IL":
            assert char not in CHARSET

    def test_generated_chars_are_in_charset(self) -> None:
        for _ in range(50):
            assert set(generate_code()) <= set(CHARSET)

    def test_codes_are_random(self) -> None:
        codes = {generate_code() for _ in range(50)}
        assert len(codes) > 40


class TestHashAndVerify:
    def test_verify_correct_code(self) -> None:
        code = "AB34"
        assert verify_code(code, hash_code(code)) is True

    def test_verify_is_case_insensitive(self) -> None:
        assert verify_code("ab34", hash_code("AB34")) is True

    def test_verify_ignores_surrounding_spaces(self) -> None:
        assert verify_code(" AB34 ", hash_code("AB34")) is True

    def test_verify_wrong_code(self) -> None:
        assert verify_code("AB35", hash_code("AB34")) is False

    def test_empty_inputs(self) -> None:
        assert verify_code("", hash_code("AB34")) is False
        assert verify_code("AB34", "") is False

    def test_hash_is_not_plaintext(self) -> None:
        assert hash_code("AB34") != "AB34"
        assert len(hash_code("AB34")) == 64


class TestRender:
    def test_png_signature(self) -> None:
        data = render_png("AB34")
        assert data.startswith(PNG_SIGNATURE)
        assert len(data) > 200

    def test_base64_data_uri(self) -> None:
        uri = render_base64("AB34")
        assert uri.startswith("data:image/png;base64,")
        raw = base64.b64decode(uri.split(",", 1)[1])
        assert raw.startswith(PNG_SIGNATURE)
