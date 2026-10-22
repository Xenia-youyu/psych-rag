"""提示词模板规范校验测试。

这套校验是「规范靠代码强制」的落点：缺 front-matter 字段、缺「反例」段落、
槽位不齐、id 重复，都必须抛错而不是放行。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.prompt_loader import (
    REQUIRED_META,
    PromptError,
    PromptRegistry,
    load_one,
    registry,
)

VALID_TEMPLATE = """---
id: demo_prompt
version: v1
purpose: 测试用模板
inputs: [question]
outputs: plain_text
model: deepseek-flash
temperature: 0.2
max_tokens: 100
owner: 测试
last_reviewed: 2026-10-20
---

## 系统指令
你是测试助手。

## 用户模板
问题：{{question}}

## 正例
输入 A → 输出 A

## 反例
输入 B → 错误输出，错在哪：忘了引用。
"""


def write_template(tmp_path: Path, name: str, content: str) -> Path:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


class TestRealTemplates:
    def test_all_prompts_loaded(self) -> None:
        items = registry.list()
        assert len(items) == 11
        assert {item["id"] for item in items} >= {
            "intent_classify",
            "answer_generate",
            "citation_verify",
            "safe_guard",
            "vl_page_to_markdown",
        }

    def test_every_template_has_required_meta(self) -> None:
        for item in registry.list():
            tpl = registry.get(item["id"])
            for key in REQUIRED_META:
                assert key in tpl.meta, f"{item['id']} 缺字段 {key}"

    def test_normal_templates_have_four_sections(self) -> None:
        for item in registry.list():
            tpl = registry.get(item["id"])
            if tpl.kind == "fixed":
                continue
            for section in ("系统指令", "用户模板", "正例", "反例"):
                assert section in tpl.sections, f"{item['id']} 缺段落 {section}"

    def test_fixed_template_has_output_section(self) -> None:
        tpl = registry.get("safe_guard")
        assert tpl.kind == "fixed"
        assert "输出正文" in tpl.sections
        assert "12356" in tpl.fixed_text()

    def test_fixed_template_refuses_render(self) -> None:
        with pytest.raises(PromptError):
            registry.get("safe_guard").render(question="x")

    def test_answer_generate_renders_slots(self) -> None:
        text = registry.render(
            "answer_generate",
            context="[S1] 上下文内容",
            question="负强化和惩罚的区别",
            intent="CONTRAST",
            user_level="备考",
            evidence_flags="教材共识",
        )
        assert "负强化和惩罚的区别" in text
        assert "[S1] 上下文内容" in text

    def test_missing_slot_raises(self) -> None:
        with pytest.raises(PromptError) as exc:
            registry.render("answer_generate", question="只有问题")
        assert "缺少槽位" in str(exc.value)

    def test_all_prompts_are_chinese(self) -> None:
        """提示词必须是中文（任务描述硬要求）。抽查系统指令段含中文。"""
        for item in registry.list():
            tpl = registry.get(item["id"])
            body = tpl.sections.get("系统指令") or tpl.sections.get("输出正文", "")
            assert any("\u4e00" <= ch <= "\u9fff" for ch in body), f"{item['id']} 系统指令不是中文"


class TestValidation:
    def test_valid_template_passes(self, tmp_path: Path) -> None:
        path = write_template(tmp_path, "demo.v1.md", VALID_TEMPLATE)
        tpl = load_one(path)
        assert tpl.prompt_id == "demo_prompt"
        assert tpl.slots == ["question"]

    def test_missing_front_matter_raises(self, tmp_path: Path) -> None:
        path = write_template(tmp_path, "bad.v1.md", "## 系统指令\n内容\n")
        with pytest.raises(PromptError) as exc:
            load_one(path)
        assert "front-matter" in str(exc.value)

    def test_missing_meta_field_raises(self, tmp_path: Path) -> None:
        content = VALID_TEMPLATE.replace("owner: 测试\n", "")
        path = write_template(tmp_path, "bad.v1.md", content)
        with pytest.raises(PromptError) as exc:
            load_one(path)
        assert "缺字段" in str(exc.value)

    def test_missing_counter_example_raises(self, tmp_path: Path) -> None:
        """缺「反例」必须判不合格——这是规范里最容易被偷工减料的一段。"""
        content = VALID_TEMPLATE.split("## 反例")[0]
        path = write_template(tmp_path, "bad.v1.md", content)
        with pytest.raises(PromptError) as exc:
            load_one(path)
        assert "反例" in str(exc.value)

    def test_bad_version_raises(self, tmp_path: Path) -> None:
        content = VALID_TEMPLATE.replace("version: v1", "version: 1.0")
        path = write_template(tmp_path, "bad.v1.md", content)
        with pytest.raises(PromptError) as exc:
            load_one(path)
        assert "version" in str(exc.value)

    def test_duplicate_id_raises(self, tmp_path: Path) -> None:
        write_template(tmp_path, "a.v1.md", VALID_TEMPLATE)
        write_template(tmp_path, "b.v1.md", VALID_TEMPLATE)
        with pytest.raises(PromptError) as exc:
            PromptRegistry(tmp_path)
        assert "id 重复" in str(exc.value)

    def test_missing_directory_raises(self, tmp_path: Path) -> None:
        with pytest.raises(PromptError):
            PromptRegistry(tmp_path / "not-exists")
