"""识别并剔除正文之前的非内容页面（目录、版权页、CIP 数据）。

为什么值得单独做一个模块：
    目录页里有整本书的章节名，语义上"像"任何问题——问"记忆的分类"时，
    目录那条"第三章 记忆……85"的向量距离可能比正文还近，挤掉真正的答案。
    实测库内 11599 块里有 93 块目录、9 块版权页，占比 1%，
    但这 1% 出现在召回里的概率远高于 1%。

三个设计取舍：

1. **只判断前 15% 的页面。** 目录和版权页只会在开头，全书扫描一遍既慢
   又容易误伤——正文里出现"参考文献"四个字再正常不过。

2. **序言默认保留。** 用户说的"序文"往往包含作者的研究缘起和全书脉络，
   不是纯噪音；真正该删的是目录和版权页。要删序言把 drop_preface 打开。

3. **宁可漏，不可错。** 目录的判断要求"至少 2 处引导点+页码"或
   "至少 6 行行尾页码"，版权页要求 ISBN 或 2 个以上版权关键词。
   漏掉几页目录只是少删一点，误删正文是真丢内容。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.core.logging import get_logger
from app.ingest.base import ParsedDoc, ParsedPage

logger = get_logger(__name__)

# 「第三章 记忆.........85」这类引导点
LEADER_TOC = re.compile(r"\.{3,}\s*\d{1,4}")
# 行尾直接跟页码：「第三章 记忆 85」
TAIL_TOC = re.compile(r".{4,60}?\s+\d{1,4}\s*$", re.M)
# 版权页特征
COPY_KEYS = re.compile(
    r"ISBN|图书在版编目|CIP\s*数据|版权所有|出版发行|印\s*张|字\s*数|"
    r"开\s*本|定\s*价|印\s*刷|版\s*次|书\s*号"
)
# 前置部分的小标题（只用于识别边界，不一定删）
PREFACE = re.compile(r"^\s*(前言|序言|自序|代序|译者序|序|导言|引言|致谢)")

# 只看前 15%，且最多看 40 页
FRONT_RATIO = 0.15
FRONT_MAX_PAGES = 40


@dataclass
class FilterResult:
    parsed: ParsedDoc
    dropped: list[int] = field(default_factory=list)
    toc_pages: list[int] = field(default_factory=list)
    copyright_pages: list[int] = field(default_factory=list)
    preface_pages: list[int] = field(default_factory=list)

    @property
    def dropped_count(self) -> int:
        return len(self.dropped)


def _looks_like_toc(text: str) -> bool:
    """判断一段文本是不是目录。

    主判据是引导点（「第三章 记忆......85」），这是中文学术书目录的标志。
    行尾页码只作为辅助，而且必须严格递增——

    第一版实现只要求「6 行以上带行尾数字」，结果把表格判成了目录：
        「回答平均数 / 年级 / 组别 / na / 3 / 4 / 5」
    这种表格数据也是短行+数字，但页码不会递增。
    目录被漏掉只是少删一点，正文被删是真丢内容，所以这里宁可严格。
    """
    if len(LEADER_TOC.findall(text)) >= 2:
        return True

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 6 or len(text) > 2000:
        return False

    numbers: list[int] = []
    for line in lines:
        match = re.search(r"(\d{1,4})\s*$", line)
        # 任何一行不是「短文本 + 行尾数字」，就不是目录（表格会在这里被排除）
        if match is None or len(line) > 40:
            return False
        numbers.append(int(match.group(1)))

    increasing = sum(1 for left, right in zip(numbers, numbers[1:]) if right >= left)
    distinct = len(set(numbers))
    return increasing >= len(numbers) - 2 and distinct >= len(numbers) * 0.7


def _looks_like_copyright(text: str) -> bool:
    if "ISBN" in text.upper():
        return True
    return len(set(COPY_KEYS.findall(text))) >= 2


def strip_front_matter(
    parsed: ParsedDoc,
    *,
    drop_toc: bool = True,
    drop_copyright: bool = True,
    drop_preface: bool = False,
) -> FilterResult:
    """剔除正文前的目录 / 版权页。返回过滤后的文档与被删页码。

    如果过滤后一页不剩，说明判断全部误伤（比如整本都是目录体的工具书），
    这时退回原文档——宁可留着噪音，也不能把整本书删空。
    """
    pages = parsed.pages
    result = FilterResult(parsed=parsed)
    if not pages:
        return result

    window = min(len(pages), max(FRONT_MAX_PAGES, int(len(pages) * FRONT_RATIO)))
    drop: set[int] = set()

    for index in range(window):
        page = pages[index]
        text = (page.text or "").strip()
        if not text:
            continue

        if drop_toc and _looks_like_toc(text):
            drop.add(index)
            result.toc_pages.append(page.page_no)
            continue

        if drop_copyright and _looks_like_copyright(text):
            drop.add(index)
            result.copyright_pages.append(page.page_no)
            continue

        if PREFACE.match(text):
            result.preface_pages.append(page.page_no)
            if drop_preface:
                drop.add(index)

    if len(drop) >= len(pages):
        logger.warning(
            "ingest.frontmatter_would_empty",
            extra={"extra_fields": {"pages": len(pages), "dropped": len(drop)}},
        )
        return FilterResult(parsed=parsed, preface_pages=result.preface_pages)

    kept = [page for index, page in enumerate(pages) if index not in drop]
    result.dropped = sorted(pages[index].page_no for index in drop)
    result.parsed = ParsedDoc(
        source_path=parsed.source_path,
        parser=parsed.parser,
        pages=kept,
        markdown="\n\n".join(page.text for page in kept),
        meta={**parsed.meta, "front_matter_dropped": len(drop)},
    )

    if drop:
        logger.info(
            "ingest.frontmatter",
            extra={
                "extra_fields": {
                    "dropped": len(drop),
                    "toc": len(result.toc_pages),
                    "copyright": len(result.copyright_pages),
                    "preface_kept": len(result.preface_pages),
                }
            },
        )
    return result


def classify_page(text: str) -> str:
    """给单页分类，供清洗脚本复用。返回值：toc / copyright / preface / body。"""
    cleaned = (text or "").strip()
    if not cleaned:
        return "empty"
    if _looks_like_toc(cleaned):
        return "toc"
    if _looks_like_copyright(cleaned):
        return "copyright"
    if PREFACE.match(cleaned):
        return "preface"
    return "body"
