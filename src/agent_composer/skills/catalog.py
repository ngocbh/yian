"""Skill values and confined reads of the files a skill bundles.

A skill is a directory holding ``SKILL.md`` (YAML frontmatter with ``name`` and
``description``, then Markdown instructions) plus optional bundled files. Reading
skill directories from disk lives in `agent_composer.compose.skills`, because only
Compose parses YAML.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import re

SKILL_FILE = "SKILL.md"
# Largest SKILL.md or bundled file returned to a model, in bytes.
MAX_SKILL_FILE_BYTES = 256 * 1024
_NAME = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_MAX_NAME = 64


class SkillError(ValueError):
    """A skill package, grant, or resource request is invalid."""


@dataclass(frozen=True)
class Skill:
    """One skill a host makes available to a run.

    Attributes:
        name (`str`):
            Grant name: 1-64 lowercase letters, digits, and single hyphens.
        description (`str`):
            Non-empty text telling the model when to use the skill.
        instructions (`str`):
            The ``SKILL.md`` body after the frontmatter, read once when loaded.
        directory (`Path | None`, *optional*, defaults to `None`):
            Resolved skill directory for bundled files. `None` means the skill
            has no readable files.

    Example:
        ```python
        Skill("release-notes", "Write release notes.", "Group changes by area.")
        ```
    """

    name: str
    description: str
    instructions: str
    directory: Path | None = None

    def __post_init__(self) -> None:
        """Reject values the grant syntax and system message cannot carry."""
        check_skill_name(self.name)
        if type(self.description) is not str or not self.description.strip():
            raise SkillError(f"skill {self.name!r}: description must be a non-empty string")
        if type(self.instructions) is not str:
            raise SkillError(f"skill {self.name!r}: instructions must be a string")
        if self.directory is not None and not isinstance(self.directory, Path):
            raise SkillError(f"skill {self.name!r}: directory must be a Path or None")


def check_skill_name(name: object) -> str:
    """Return `name` when it is a valid skill name.

    Args:
        name (`object`):
            Candidate name from frontmatter, a directory, or a YAML grant.

    Returns:
        `str`: The unchanged name.

    Raises:
        `SkillError`: Raised when `name` is not 1-64 lowercase letters, digits,
            and single hyphens, starting and ending with a letter or digit.
    """
    if type(name) is not str or len(name) > _MAX_NAME or not _NAME.fullmatch(name):
        raise SkillError(
            f"invalid skill name {name!r}: use 1-{_MAX_NAME} lowercase letters, "
            "digits, and single hyphens"
        )
    return name


def read_skill_file(skill: Skill, path: str) -> str:
    """Return one UTF-8 file bundled with `skill`, confined to its directory.

    Args:
        skill (`Skill`):
            The granted skill.
        path (`str`):
            POSIX path relative to the skill directory, such as
            ``references/api.md``. Absolute paths, ``..``, backslashes, hidden
            (dot) files and folders, and symlinks that leave the directory are
            rejected.

    Returns:
        `str`: The file text.

    Raises:
        `SkillError`: Raised when the path is unsafe, hidden, missing, not a regular
            file, larger than `MAX_SKILL_FILE_BYTES`, or not UTF-8 text.
    """
    where = f"skill {skill.name!r}: file {path!r}"
    if skill.directory is None:
        raise SkillError(f"skill {skill.name!r} has no bundled files")
    relative = PurePosixPath(path)
    if (
        not path
        or "\\" in path
        or "\x00" in path
        or relative.is_absolute()
        or ".." in relative.parts
    ):
        raise SkillError(f"{where} must be a relative path inside the skill directory")
    if any(part.startswith(".") for part in relative.parts):
        raise SkillError(f"{where}: hidden files and folders are not readable")
    root = skill.directory.resolve()
    target = (root / relative).resolve()
    if not target.is_relative_to(root):
        raise SkillError(f"{where} resolves outside the skill directory")
    if not target.is_file():
        raise SkillError(f"{where} does not exist")
    return read_skill_text(target, where)


def read_skill_text(path: Path, where: str | None = None) -> str:
    """Read one bounded UTF-8 skill file.

    Args:
        path (`Path`):
            File to read.
        where (`str | None`, *optional*, defaults to `None`):
            Label for error messages; defaults to the path.

    Returns:
        `str`: The file text.

    Raises:
        `SkillError`: Raised when the file is larger than `MAX_SKILL_FILE_BYTES`
            or not UTF-8 text.
    """
    label = where or str(path)
    if path.stat().st_size > MAX_SKILL_FILE_BYTES:
        raise SkillError(f"{label} is larger than {MAX_SKILL_FILE_BYTES} bytes")
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raise SkillError(f"{label} is not UTF-8 text") from None
