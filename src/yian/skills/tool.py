"""The built-in ``load_skill`` tool and the system-message skill listing."""

from __future__ import annotations

from typing import Mapping, Sequence

from langchain_core.tools import BaseTool, StructuredTool

from yian.skills.catalog import Skill, SkillError, read_skill_file

LOAD_SKILL = "load_skill"
# Longest description shown in the listing, in characters.
MAX_LISTED_DESCRIPTION = 1024

_LISTING_HEADER = (
    "Skills: each skill below is a package of instructions. When a task matches a "
    f"skill's description, call `{LOAD_SKILL}` with its name before acting and follow "
    "the instructions it returns. When those instructions mention a file, read it "
    f"with `{LOAD_SKILL}` using the same name and the file's relative path."
)
_TOOL_DESCRIPTION = (
    "Load a granted skill. Without `path`, returns the skill's SKILL.md "
    "instructions. With `path`, returns the UTF-8 text of a file inside the "
    "skill's directory, given relative to it (for example `references/api.md`)."
)


def system_prompt_with_skills(base: str, skills: Sequence[Skill]) -> str:
    """Append the granted skills' names and descriptions to a system prompt.

    Args:
        base (`str`):
            System prompt before skills.
        skills (`Sequence[Skill]`):
            Granted skills. An empty sequence returns `base` unchanged.

    Returns:
        `str`: `base`, a blank line, a short usage paragraph, and one
        ``- name: description`` line per skill sorted by name.
    """
    if not skills:
        return base
    lines = [base, "", _LISTING_HEADER]
    for skill in sorted(skills, key=lambda item: item.name):
        description = " ".join(skill.description.split())
        if len(description) > MAX_LISTED_DESCRIPTION:
            description = description[: MAX_LISTED_DESCRIPTION - 1] + "…"
        lines.append(f"- {skill.name}: {description}")
    return "\n".join(lines)


def _load_skill_schema(name: str, path: str | None = None) -> str:
    """Schema carrier for `load_skill`; the agent loop dispatches the call."""
    raise NotImplementedError


def load_skill_tool() -> BaseTool:
    """Return the `load_skill` tool definition bound to the model.

    Returns:
        `BaseTool`: A tool named ``load_skill`` with arguments ``name: str`` and
        ``path: str | None = None``. The agent loop runs it with
        `invoke_load_skill`, never through the tool registry.
    """
    return StructuredTool.from_function(
        _load_skill_schema, name=LOAD_SKILL, description=_TOOL_DESCRIPTION
    )


def invoke_load_skill(skills: Mapping[str, Skill], arguments: Mapping[str, object]) -> str:
    """Run one model call of `load_skill` against the granted skills.

    Args:
        skills (`Mapping[str, Skill]`):
            Skills granted to this node, keyed by name.
        arguments (`Mapping[str, object]`):
            Model-supplied arguments: ``name`` and optional ``path``.

    Returns:
        `str`: The SKILL.md instructions, or the requested bundled file text.

    Raises:
        `SkillError`: Raised for unexpected or mistyped arguments, a skill that is
            not granted, or an unsafe or unreadable path.
    """
    extra = set(arguments) - {"name", "path"}
    if extra:
        raise SkillError(f"{LOAD_SKILL} got unexpected arguments: {sorted(extra)}")
    name = arguments.get("name")
    path = arguments.get("path")
    if type(name) is not str:
        raise SkillError(f"{LOAD_SKILL} needs a string `name`")
    if path is not None and type(path) is not str:
        raise SkillError(f"{LOAD_SKILL} `path` must be a string or null")
    skill = skills.get(name)
    if skill is None:
        raise SkillError(f"{LOAD_SKILL}: skill {name!r} is not granted to this node")
    if path is None:
        return skill.instructions
    return read_skill_file(skill, path)
