"""Read Codex/Claude-compatible ``SKILL.md`` directories into `Skill` values.

Hosts call these helpers to build the ``skills=`` mapping for a run. Frontmatter
is parsed with the Compose safe YAML literal parser; only ``name`` and
``description`` are read and every other field is ignored.

Knows about: ``agent_composer.skills`` and ``compose.source``.
Never imports: other Compose phases, compile, runtime, the run boundary, or CLI.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable
import warnings

from agent_composer.compose.source import SourceSyntaxError, parse_yaml_literal
from agent_composer.skills.catalog import (
    SKILL_FILE,
    Skill,
    SkillError,
    check_skill_name,
    read_skill_text,
)


def load_skill_dir(directory: str | os.PathLike[str]) -> Skill:
    """Read one skill directory.

    The frontmatter ``name`` wins; without it the directory name is used, as
    Claude Code does. Fields other than ``name`` and ``description`` are ignored.

    Args:
        directory (`str | os.PathLike[str]`):
            Directory containing ``SKILL.md``. A symlinked directory is allowed;
            its resolved location bounds bundled-file reads.

    Returns:
        `Skill`: The parsed skill with its resolved directory.

    Raises:
        `SkillError`: Raised when ``SKILL.md`` is missing, too large, not UTF-8,
            or has missing or malformed frontmatter, name, or description.
    """
    given = Path(directory)
    root = given.resolve()
    path = root / SKILL_FILE
    if not path.is_file():
        raise SkillError(f"{given}: missing {SKILL_FILE}")
    meta, body = _split_frontmatter(read_skill_text(path), path)
    name = meta.get("name", given.name)
    try:
        check_skill_name(name)
    except SkillError as exc:
        raise SkillError(f"{path}: {exc}") from None
    description = meta.get("description")
    if type(description) is not str or not description.strip():
        raise SkillError(f"{path}: frontmatter needs a non-empty `description`")
    return Skill(name, description.strip(), body, root)


def load_skills(
    roots: Iterable[str | os.PathLike[str]],
    *,
    skip_invalid: bool = False,
) -> dict[str, Skill]:
    """Index every skill under ordered roots; an earlier root wins a name.

    Each root holds skill directories (``<root>/<name>/SKILL.md``). Missing roots,
    hidden entries, and subdirectories without ``SKILL.md`` are ignored. A name
    found again in a later root is shadowed. Two skills with one name in the same
    root are an error.

    Args:
        roots (`Iterable[str | os.PathLike[str]]`):
            Skill roots, highest precedence first.
        skip_invalid (`bool`, *optional*, defaults to `False`):
            Warn about invalid or unreadable skills with `warnings.warn` and skip them instead
            of raising.

    Returns:
        `dict[str, Skill]`: Skills keyed by name, sorted by name.

    Raises:
        `SkillError`: Raised when `skip_invalid` is `False` and any skill is
            invalid, unreadable, or duplicated within one root. The message lists every problem.

    Example:
        ```python
        skills = load_skills(["team/skills", "shared/skills"])
        run_flow(flow, inputs, skills=skills)
        ```
    """
    found: dict[str, Skill] = {}
    problems: list[str] = []
    for raw_root in roots:
        root = Path(raw_root)
        try:
            entries = sorted(root.iterdir()) if root.is_dir() else []
        except OSError as exc:
            problems.append(f"{root}: cannot read skill root: {exc}")
            continue
        in_root: dict[str, Path] = {}
        for entry in entries:
            if entry.name.startswith("."):
                continue
            try:
                if not (entry / SKILL_FILE).is_file():
                    continue
                skill = load_skill_dir(entry)
            except SkillError as exc:
                problems.append(str(exc))
                continue
            except OSError as exc:
                problems.append(f"{entry}: cannot read skill: {exc}")
                continue
            if skill.name in in_root:
                problems.append(
                    f"{root}: skill name {skill.name!r} is used by both "
                    f"{in_root[skill.name].name}/ and {entry.name}/"
                )
                continue
            in_root[skill.name] = entry
            found.setdefault(skill.name, skill)
    if problems and not skip_invalid:
        raise SkillError("invalid skills:\n- " + "\n- ".join(problems))
    for problem in problems:
        warnings.warn(f"skipped skill: {problem}", stacklevel=2)
    return dict(sorted(found.items()))


def default_skill_roots(
    cwd: str | os.PathLike[str] | None = None,
    home: str | os.PathLike[str] | None = None,
) -> list[Path]:
    """Return the Claude Code and Codex skill roots, highest precedence first.

    User roots come first (Claude Code ranks personal over project), then project
    roots from `cwd` up to the enclosing git root, nearest first. At each level the
    Claude root precedes the Codex root.

    Args:
        cwd (`str | os.PathLike[str] | None`, *optional*, defaults to `None`):
            Project directory. `None` uses the process working directory.
        home (`str | os.PathLike[str] | None`, *optional*, defaults to `None`):
            User home. `None` uses `Path.home()`.

    Returns:
        `list[Path]`: ``~/.claude/skills``, ``~/.agents/skills``, then
        ``<dir>/.claude/skills`` and ``<dir>/.agents/skills`` for each project level.
    """
    home_dir = Path.home() if home is None else Path(home)
    start = (Path.cwd() if cwd is None else Path(cwd)).resolve()
    chain = [start, *start.parents]
    levels = [start]
    for depth, level in enumerate(chain):
        if (level / ".git").exists():
            levels = chain[: depth + 1]
            break
    roots = [home_dir / ".claude" / "skills", home_dir / ".agents" / "skills"]
    for level in levels:
        roots += [level / ".claude" / "skills", level / ".agents" / "skills"]
    return roots


def _split_frontmatter(text: str, path: Path) -> tuple[dict, str]:
    """Split ``---`` YAML frontmatter from the Markdown body."""
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        raise SkillError(f"{path}: must start with `---` YAML frontmatter")
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            break
    else:
        raise SkillError(f"{path}: frontmatter has no closing `---`")
    try:
        meta = parse_yaml_literal("".join(lines[1:index])) or {}
    except SourceSyntaxError as exc:
        raise SkillError(f"{path}: frontmatter is not valid YAML: {exc}") from None
    if not isinstance(meta, dict):
        raise SkillError(f"{path}: frontmatter must be a mapping")
    return meta, "".join(lines[index + 1 :]).lstrip("\n")
