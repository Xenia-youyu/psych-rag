"""剔除块内混入的页脚广告、水印与联系方式。

来源是一批扫描/数字化教材：每页页脚都印着
    「这里提供最专业的心理学考研辅导 / 资料下载：bbs.libidos.cn / 答疑QQ：xxxx」
切块是整页切，水印就跟着正文进了块。实测《实验心理学》2290 块里有 763 块带水印，
占 33%——检索「考研」相关问题时这些块会排到前面，引用展开还会把广告一起展示给用户。

只删行，不删块：块的主体是正文，水印通常只占末尾两三行。
删完如果整块不剩什么（< 80 字），由调用方丢弃该块。

判定要限定行长，因为正文里也会有网址：
    「h. 电子文献  标引项顺序号  作者.题名.出处或可获得地址」是写作规范的正文，
    含 http 但它是长句，不能删。
"""

from __future__ import annotations

import re

# 短行里出现这些就该删
WATERMARK = re.compile(
    r"bbs\.[\w-]+\.(?:cn|com|net)|www\.|[a-z0-9-]+\.(?:com|cn|net)/|https?://|"
    r"QQ\s*[:：]|微信\s*[:：]|微信号|公众号|"
    r"资料下载|考研辅导|考研真题|扫描全能王|百度网盘|"
    r"仅供学习|禁止商业|内部资料",
    re.IGNORECASE,
)

# 超过这个长度就不当水印处理：正文长句里出现网址是正常的
MAX_WATERMARK_LINE = 60

# 页眉式水印：不是独占一行，而是被 OCR 拼进了正文开头
#     「仅供交流学习，禁止倒卖！博 / “十二五”普通高等教育本科国家级规划教材…」
# 按行删不掉（这一行长着正文的样子），只能按短语删。
# 只收高置信短语：这些词在学术正文里不会自然出现，
# 不像「内部资料」那样可能是论文写作规范里的术语。
INLINE_WATERMARK = re.compile(
    r"仅供\s*(?:交流|学习|个人)[\s，,、]*"
    r"(?:学习|交流|参考|使用){0,2}"
    r"[\s，,、！!。]*"
    r"(?:禁止|严禁|请勿)\s*(?:倒卖|传播|翻印|复制|商用|商业使用)"
    r"[\s！!。，,、]*"
    r"|(?:禁止|严禁)\s*倒卖",
)


def strip_inline_watermark(text: str) -> tuple[str, int]:
    """删除混在正文里的页眉水印，返回 (清洗后文本, 删除处数)。

    与 `strip_watermark` 的区别：那个按行判定、限行长，用于页脚广告；
    这个按短语判定、不限行长，用于被 OCR 拼进正文的页眉。
    实测《发展心理学》1971 块里 533 块、《社会心理学》1419 块里 315 块带这个水印。
    """
    if not text:
        return text, 0
    hits = len(INLINE_WATERMARK.findall(text))
    if not hits:
        return text, 0
    cleaned = INLINE_WATERMARK.sub("", text)
    # 删掉短语后可能留下行首空白或连续空行
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.lstrip(), hits


def strip_watermark(text: str, *, max_line: int = MAX_WATERMARK_LINE) -> tuple[str, int]:
    """返回 (清洗后文本, 删除的行数)。"""
    if not text:
        return text, 0

    kept: list[str] = []
    removed = 0
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and len(stripped) <= max_line and WATERMARK.search(stripped):
            removed += 1
            continue
        kept.append(line)

    if removed == 0:
        return text, 0

    # 去掉因删除产生的连续空行，避免块里出现大片空白
    cleaned = re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()
    return cleaned, removed


def is_watermark_only(text: str) -> bool:
    """整块只剩水印（清洗后几乎没有内容）。"""
    return len(repair_text(text)) < 80


# 中文字符之间夹着空格：「学 习 的 结 果」「产 生 式」
# 排除英文与数字：只处理中文（含中文标点）紧邻的位置
_CJK_GAP = re.compile(
    r"(?<=[\u4e00-\u9fa5，。；：、！？）》」』])\s+(?=[\u4e00-\u9fa5，。；：、（《「『])"
)


def strip_cjk_gaps(text: str) -> tuple[str, int]:
    """去掉中文字符之间的空格，返回 (清洗后文本, 合并处数)。

    实测《当代教育心理学》2743 块里 2258 块（82%）是这种形态——
    PyMuPDF 按字符位置抽取时，字间距被当成了空格：

        「，自动 识 别 相关 的线 索，把一些小的步骤 形成大 的产 生式」

    中文本来不用空格分词，这些空格纯属噪音。它们会把「产生式」这种词
    切成「产 生式」，向量和倒排索引都对不上。
    """
    if not text:
        return text, 0
    merged = _CJK_GAP.sub("", text)
    if merged == text:
        return text, 0
    return merged, len(_CJK_GAP.findall(text))


def repair_text(text: str) -> str:
    """入库前统一清洗：短语级水印 → 行级水印 → 中文间距。

    顺序有讲究：先按短语删（它不看行长，能处理拼进正文的页眉），
    再按行删（处理独占一行的页脚广告），最后收空格。
    """
    cleaned, _ = strip_inline_watermark(text)
    cleaned, _ = strip_watermark(cleaned)
    cleaned, _ = strip_cjk_gaps(cleaned)
    return cleaned
