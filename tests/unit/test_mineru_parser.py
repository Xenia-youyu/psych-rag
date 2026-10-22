"""MinerU 解析器包装层测试：不加载真模型，只验证我们这层的映射逻辑。

为什么值得单独测：
    真模型跑一次要几十秒，而且受环境影响（锁、显存、模型路径）。
    但"把 MinerU 的输出翻译成我们的 ParsedDoc"这件事是纯逻辑，
    它错了（比如页码差 1、block 提取为空）会让整库数据的页码全是错的——
    引用溯源一旦错了，学生按页码翻书找不到，整个"可溯源"就成了摆设。

这里用 sys.modules 注入假 mineru 模块，覆盖三个关键行为：
- page_idx 是 0 基 → 转成 1 基
- 13 种 block 类型里只取得到正文的那几种，取不到就跳过
- 逐页提取全空时退回用 markdown，避免入库一堆空块
"""

from __future__ import annotations

import sys
import types

import pytest

from app.ingest.parsers import MinerUParser, pick_parser


class FakeBlock:
    def __init__(self, **kwargs) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)


class FakePage:
    def __init__(self, page_idx: int, blocks: list) -> None:
        self.page_idx = page_idx
        self.blocks = blocks


class FakeResult:
    def __init__(self, pages: list, markdown: str) -> None:
        self.pages = pages
        self._md = markdown

    def markdown(self) -> str:
        return self._md


def install_fake_mineru(result: FakeResult | Exception) -> None:
    module = types.ModuleType("mineru.parser")

    def fake_parse(path, **kwargs):
        if isinstance(result, Exception):
            raise result
        return result

    module.parse = fake_parse
    sys.modules["mineru.parser"] = module
    # 也注册父包，否则 from mineru.parser import 会先去找 mineru
    if "mineru" not in sys.modules:
        sys.modules["mineru"] = types.ModuleType("mineru")
    sys.modules["mineru"].parser = module


@pytest.fixture(autouse=True)
def _cleanup_modules():
    yield
    sys.modules.pop("mineru.parser", None)


class TestPageMapping:
    def test_page_idx_is_converted_to_one_based(self) -> None:
        """MinerU 的 page_idx 从 0 开始，对外必须从 1 开始——差一页就是引用错页。"""
        install_fake_mineru(
            FakeResult(
                pages=[
                    FakePage(0, [FakeBlock(text="第一页")]),
                    FakePage(1, [FakeBlock(text="第二页")]),
                ],
                markdown="第一页\n第二页",
            )
        )
        doc = MinerUParser().parse("x.pdf")

        assert [p.page_no for p in doc.pages] == [1, 2]
        assert doc.pages[0].text == "第一页"

    def test_multiple_blocks_joined(self) -> None:
        install_fake_mineru(
            FakeResult(
                pages=[
                    FakePage(
                        4,
                        [
                            FakeBlock(text="段落一"),
                            FakeBlock(text="段落二"),
                            FakeBlock(html="<p>段落三</p>"),
                        ],
                    )
                ],
                markdown="段落一段落二",
            )
        )
        doc = MinerUParser().parse("x.pdf")

        assert doc.pages[0].page_no == 5
        assert "段落一" in doc.pages[0].text
        assert "段落三" in doc.pages[0].text

    def test_table_rows_rendered(self) -> None:
        """表格类 block 没有 text，只有 rows/cells，要能拼成可读文本。"""
        cell_a = FakeBlock(text="项目")
        cell_b = FakeBlock(text="数值")
        row = types.SimpleNamespace(cells=[cell_a, cell_b])
        install_fake_mineru(
            FakeResult(pages=[FakePage(0, [FakeBlock(rows=[row])])], markdown="表格")
        )
        doc = MinerUParser().parse("x.pdf")

        assert "项目" in doc.pages[0].text and "数值" in doc.pages[0].text

    def test_block_without_text_is_skipped(self) -> None:
        """图片 block 之类的没文字，跳过即可，不能让整页崩掉。"""
        install_fake_mineru(
            FakeResult(
                pages=[FakePage(0, [FakeBlock(type="image"), FakeBlock(text="正文")])],
                markdown="正文",
            )
        )
        doc = MinerUParser().parse("x.pdf")

        assert doc.pages[0].text.strip() == "正文"


class TestFallback:
    def test_all_blocks_empty_falls_back_to_markdown(self) -> None:
        """所有 block 都取不到正文但 markdown 有内容 → 说明 MinerU 换了 block 结构，
        这时用 markdown 当单页正文，总比入库一堆空块强。"""
        install_fake_mineru(
            FakeResult(
                pages=[FakePage(0, [FakeBlock(type="unknown")]), FakePage(1, [])],
                markdown="这是一整篇正文。",
            )
        )
        doc = MinerUParser().parse("x.pdf")

        assert doc.page_count == 1
        assert doc.pages[0].text == "这是一整篇正文。"

    def test_empty_result_stays_empty(self) -> None:
        """markdown 也空就保持空，交给上层判失败——不能编内容出来。"""
        install_fake_mineru(FakeResult(pages=[], markdown=""))
        doc = MinerUParser().parse("x.pdf")

        assert doc.page_count == 0
        assert doc.total_chars == 0


class TestParserSelection:
    @pytest.mark.parametrize(
        ("filename", "expect_mineru"),
        [
            ("book.pdf", True),
            ("scan.PNG", True),
            ("note.docx", False),
            ("a.txt", False),
            ("b.md", False),
        ],
    )
    def test_pick_parser(self, filename: str, expect_mineru: bool) -> None:
        parser = pick_parser(filename)
        assert isinstance(parser, MinerUParser) is expect_mineru

    def test_pick_parser_pdf_creates_mineru(self) -> None:
        """PDF 一律走 MinerU：电子版也有版面与阅读顺序，PyMuPDF 双栏会串行。"""
        assert isinstance(pick_parser("x.pdf"), MinerUParser)
