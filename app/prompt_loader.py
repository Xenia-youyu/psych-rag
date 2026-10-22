"""提示词模板加载器：解析 front-matter、校验必备段落、渲染槽位。

规范（见 v3 文档第六节）：
- 文件名：<用途>_<变体>.v<主版本>.md
- front-matter 必填：id / version / purpose / inputs / outputs / model / temperature / max_tokens / owner / last_reviewed
- 正文必备四段：## 系统指令 / ## 用户模板 / ## 正例 / ## 反例
- 缺「反例」的模板视为不合格，加载时直接报错

用法：
    from app.prompt_loader import registry
    text = registry.render("answer_generate", context=..., question=...)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"

REQUIRED_META = [
    "id",
    "version",
    "purpose",
    "inputs",
    "outputs",
    "model",
    "temperature",
    "max_tokens",
    "owner",
    "last_reviewed",
]
REQUIRED_SECTIONS = ["系统指令", "用户模板", "正例", "反例"]

# kind=fixed 的模板（固定话术，不经大模型）不要求四段结构，
# 但必须提供 ## 输出正文 段，且禁止在运行时被模型改写。
FIXED_SECTIONS = ["输出正文"]

SLOT_RE = re.compile(r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}")
SECTION_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)


class PromptError(Exception):
    """提示词模板不合规。"""


@dataclass
class PromptTemplate:
    prompt_id: str
    version: str
    meta: dict
    sections: dict[str, str] = field(default_factory=dict)
    file: Path | None = None

    @property
    def kind(self) -> str:
        return str(self.meta.get("kind", "template"))

    @property
    def slots(self) -> list[str]:
        """模板中出现的全部槽位名。"""
        found: list[str] = []
        for body in self.sections.values():
            for name in SLOT_RE.findall(body):
                if name not in found:
                    found.append(name)
        return found

    def fixed_text(self) -> str:
        """kind=fixed 的模板正文，原样返回，不经任何模型。"""
        if self.kind != "fixed":
            raise PromptError(f"[{self.prompt_id}] 不是 fixed 类型")
        return self.sections.get("输出正文", "")

    def render(self, **kwargs: str) -> str:
        """渲染用户模板段。缺槽位直接报错，禁止静默传空。"""
        if self.kind == "fixed":
            raise PromptError(
                f"[{self.prompt_id}] 是 fixed 类型，不经模型渲染，请改用 fixed_text()"
            )
        missing = [s for s in self.slots if s not in kwargs]
        if missing:
            raise PromptError(
                f"[{self.prompt_id}] 缺少槽位 {missing}；模板声明 inputs={self.meta.get('inputs')}"
            )
        body = self.sections.get("用户模板", "")
        return SLOT_RE.sub(lambda m: str(kwargs[m.group(1)]), body)

    def system(self) -> str:
        return self.sections.get("系统指令", "")


def _parse_sections(text: str) -> dict[str, str]:
    matches = list(SECTION_RE.finditer(text))
    result: dict[str, str] = {}
    for i, m in enumerate(matches):
        title = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        result[title] = text[start:end].strip()
    return result


def load_one(path: Path) -> PromptTemplate:
    raw = path.read_text(encoding="utf-8")
    if not raw.startswith("---"):
        raise PromptError(f"{path.name} 缺少 front-matter")

    _, meta_raw, body = raw.split("---", 2)
    try:
        meta = yaml.safe_load(meta_raw) or {}
    except yaml.YAMLError as exc:
        raise PromptError(f"{path.name} front-matter 不是合法 YAML：{exc}") from exc

    absent = [k for k in REQUIRED_META if k not in meta]
    if absent:
        raise PromptError(f"{path.name} front-matter 缺字段：{absent}")

    sections = _parse_sections(body)
    kind = meta.get("kind", "template")
    needed = FIXED_SECTIONS if kind == "fixed" else REQUIRED_SECTIONS
    lack = [s for s in needed if s not in sections]
    if lack:
        raise PromptError(f"{path.name}（kind={kind}）缺必备段落：{lack}")

    if not str(meta["version"]).startswith("v"):
        raise PromptError(f"{path.name} version 必须以 v 开头，当前 {meta['version']!r}")

    return PromptTemplate(
        prompt_id=meta["id"],
        version=str(meta["version"]),
        meta=meta,
        sections=sections,
        file=path,
    )


class PromptRegistry:
    def __init__(self, directory: Path = PROMPT_DIR) -> None:
        self.dir = directory
        self._cache: dict[str, PromptTemplate] = {}
        self._load_all()

    def _load_all(self) -> None:
        if not self.dir.exists():
            raise PromptError(f"提示词目录不存在：{self.dir}")
        for path in sorted(self.dir.glob("*.md")):
            if path.name.upper() == "README.MD":
                continue
            tpl = load_one(path)
            key = tpl.prompt_id
            if key in self._cache:
                old = self._cache[key].file
                raise PromptError(f"提示词 id 重复：{key}（{old} 与 {path}）")
            self._cache[key] = tpl

    def get(self, prompt_id: str) -> PromptTemplate:
        if prompt_id not in self._cache:
            raise PromptError(f"未找到提示词：{prompt_id}")
        return self._cache[prompt_id]

    def render(self, prompt_id: str, **kwargs: str) -> str:
        return self.get(prompt_id).render(**kwargs)

    def list(self) -> list[dict]:
        return [
            {
                "id": t.prompt_id,
                "version": t.version,
                "model": t.meta["model"],
                "purpose": t.meta["purpose"],
                "file": t.file.name if t.file else "",
            }
            for t in self._cache.values()
        ]


registry = PromptRegistry()


if __name__ == "__main__":
    # 自检：python -m app.prompt_loader
    print(f"已加载 {len(registry.list())} 个提示词模板：")
    for item in registry.list():
        print(f"  {item['id']:<24} {item['version']:<5} {item['model']:<16} {item['file']}")
    print("\n全部模板通过校验。")
