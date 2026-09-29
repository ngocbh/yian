"""The load_skill tool schema, dispatch, and system-message listing."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_composer.skills import Skill, SkillError
from agent_composer.skills.tool import (
    LOAD_SKILL,
    invoke_load_skill,
    load_skill_tool,
    system_prompt_with_skills,
)


def test_listing_is_sorted_one_line_per_skill_and_empty_is_identity() -> None:
    skills = [Skill("zeta", "Last\n  one.", ""), Skill("alpha", "First.", "")]
    text = system_prompt_with_skills("BASE", skills)
    assert text.startswith("BASE\n\nSkills: ")
    assert f"call `{LOAD_SKILL}`" in text
    assert text.endswith("\n- alpha: First.\n- zeta: Last one.")
    assert system_prompt_with_skills("BASE", []) == "BASE"


def test_listing_truncates_long_descriptions() -> None:
    line = system_prompt_with_skills("B", [Skill("long", "x" * 2000, "")]).splitlines()[-1]
    assert line == "- long: " + "x" * 1023 + "…"


def test_tool_schema_is_name_and_optional_path() -> None:
    tool = load_skill_tool()
    assert tool.name == LOAD_SKILL
    assert set(tool.args) == {"name", "path"}
    assert tool.args["name"]["type"] == "string"
    assert tool.args["path"]["default"] is None


def test_invoke_returns_instructions_or_file(tmp_path: Path) -> None:
    (tmp_path / "notes.md").write_text("note\n", encoding="utf-8")
    granted = {"notes": Skill("notes", "Take notes.", "Use notes.md.", tmp_path.resolve())}
    assert invoke_load_skill(granted, {"name": "notes"}) == "Use notes.md."
    assert invoke_load_skill(granted, {"name": "notes", "path": None}) == "Use notes.md."
    assert invoke_load_skill(granted, {"name": "notes", "path": "notes.md"}) == "note\n"


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"name": "other"}, "not granted"),
        ({}, "string `name`"),
        ({"name": 3}, "string `name`"),
        ({"name": "notes", "path": 3}, "`path` must be"),
        ({"name": "notes", "extra": 1}, "unexpected arguments"),
    ],
)
def test_invoke_rejects_bad_arguments(arguments: dict, message: str) -> None:
    with pytest.raises(SkillError, match=message):
        invoke_load_skill({"notes": Skill("notes", "d", "")}, arguments)
