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
    cleaned, _ = strip_watermark(text)
    return len(cleaned) < 80
