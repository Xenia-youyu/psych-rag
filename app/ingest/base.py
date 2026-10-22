"""解析结果的数据结构与解析器协议。

为什么要抽一层：
    MinerU 负责扫描件与复杂版面，PyMuPDF 负责电子版与 txt/md。
    两者输出必须收敛成同一种结构，否则下游切块要写两套分支——
    那种代码跑三个月就会变成没人敢改的东西。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class ParsedPage:
    """单页解析结果。page_no 从 1 开始，用于引用溯源。"""

    page_no: int
    text: str

    @property
    def char_count(self) -> int:
        return len(self.text)


@dataclass
class ParsedDoc:
    """一份文档的解析产物，是所有下游环节的唯一输入。"""

    source_path: str
    parser: str
    pages: list[ParsedPage] = field(default_factory=list)
    markdown: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def total_chars(self) -> int:
        return sum(page.char_count for page in self.pages)

    def joined_text(self) -> str:
        """把各页拼成连续正文。

        页间用两个换行分隔而不是空串：教材的段落经常跨页，
        直接拼接会把"……因此" 和 "上述结论" 粘成一个词。
        """
        return "\n\n".join(page.text for page in self.pages if page.text.strip())


class DocumentParser(Protocol):
    """解析器协议。实现方只需要回答"给我这份文件，还我一页页正文"。"""

    name: str

    def parse(self, path: str, *, max_pages: int | None = None) -> ParsedDoc: ...
