"""正文前非内容页的识别与剔除。

重点不是"能不能删掉目录"，而是"会不会把正文删掉"。
所以这里的用例大多是反向的：什么样的内容**必须留下**。
"""

from __future__ import annotations

from app.ingest.base import ParsedDoc, ParsedPage
from app.ingest.frontmatter import classify_page, strip_front_matter

BODY = (
    "第三节 记忆的三级加工模型\n"
    "记忆按照信息保持时间的长短，可以分为感觉记忆、短时记忆和长时记忆三个系统。"
    "感觉记忆保持时间极短，约0.25至2秒，信息以感觉后像的形式存在；"
    "短时记忆的容量为7±2个组块，保持时间约1分钟。"
)


def make_doc(texts: list[str]) -> ParsedDoc:
    pages = [ParsedPage(page_no=i + 1, text=text) for i, text in enumerate(texts)]
    return ParsedDoc(
        source_path="x.pdf",
        parser="simple",
        pages=pages,
        markdown="\n\n".join(texts),
        meta={},
    )


TOC_PAGE = (
    "目 录\n"
    "第一章 心理学概述 ........ 1\n"
    "第二章 心理的神经生理机制 ........ 25\n"
    "第三章 感觉 ........ 48\n"
    "第四章 知觉 ........ 76\n"
    "第五章 记忆 ........ 105\n"
)

COPYRIGHT_PAGE = (
    "图书在版编目（CIP）数据\n"
    "普通心理学 / 彭聃龄主编. — 4版. — 北京：北京师范大学出版社，2012.5\n"
    "ISBN 978-7-303-12345-6\n"
    "开本 787×1092 1/16  印张 32  字数 780千\n"
    "定价 68.00元\n"
)


class TestClassify:
    def test_toc_with_leaders(self) -> None:
        assert classify_page(TOC_PAGE) == "toc"

    def test_toc_with_tail_page_numbers(self) -> None:
        page = "\n".join(f"第{i}章 某某主题 {i * 13}" for i in range(1, 9))
        assert classify_page(page) == "toc"

    def test_copyright_by_isbn(self) -> None:
        assert classify_page(COPYRIGHT_PAGE) == "copyright"

    def test_body_is_not_toc(self) -> None:
        # 正文里出现章节号和数字，绝不能当目录删掉
        assert classify_page(BODY) == "body"

    def test_body_mentioning_references(self) -> None:
        text = "本章参考文献主要来自国外研究，其中 Smith（2018）提出了……。" * 3
        assert classify_page(text) == "body"

    def test_single_page_number_is_not_toc(self) -> None:
        # 只有一行带页码，可能是正文里的"第 3 版 2019"，不是目录
        assert classify_page("实验心理学 第3版 2019") == "body"


class TestStrip:
    def test_drops_toc_and_copyright(self) -> None:
        doc = make_doc([TOC_PAGE, COPYRIGHT_PAGE, BODY, BODY])
        result = strip_front_matter(doc)

        assert result.dropped_count == 2
        assert result.toc_pages == [1]
        assert result.copyright_pages == [2]
        assert [page.page_no for page in result.parsed.pages] == [3, 4]

    def test_keeps_preface_by_default(self) -> None:
        preface = "前言\n本书是我三十年教学工作的总结，写作过程中得到许多同行帮助。"
        doc = make_doc([preface, BODY])

        result = strip_front_matter(doc)
        assert result.dropped_count == 0
        assert result.preface_pages == [1]

    def test_can_drop_preface_when_asked(self) -> None:
        preface = "前言\n本书是我三十年教学工作的总结。"
        doc = make_doc([preface, BODY])

        result = strip_front_matter(doc, drop_preface=True)
        assert result.dropped_count == 1

    def test_never_empties_the_document(self) -> None:
        """整本都是目录体（比如工具书）时，宁可全留也不能删空。"""
        doc = make_doc([TOC_PAGE, TOC_PAGE, TOC_PAGE])
        result = strip_front_matter(doc)

        assert result.dropped_count == 0
        assert len(result.parsed.pages) == 3

    def test_only_scans_the_head(self) -> None:
        """正文中段即使长得像目录也不动——那多半是索引，但更可能是误判。"""
        body_pages = [BODY] * 40
        doc = make_doc(body_pages + [TOC_PAGE])

        result = strip_front_matter(doc)
        assert result.dropped_count == 0
        assert len(result.parsed.pages) == 41

    def test_markdown_is_rebuilt(self) -> None:
        doc = make_doc([TOC_PAGE, BODY])
        result = strip_front_matter(doc)

        assert "目 录" not in result.parsed.markdown
        assert "记忆的三级加工模型" in result.parsed.markdown

    def test_empty_text_is_skipped(self) -> None:
        doc = make_doc(["", BODY])
        result = strip_front_matter(doc)

        assert result.dropped_count == 0
        assert len(result.parsed.pages) == 2
