"""Codex entry points mirror the canonical Claude guidance and project skills."""

import re
from pathlib import Path

import pytest
import yaml


_ROOT = Path(__file__).resolve().parents[1]
_CLAUDE_SKILLS = _ROOT / ".claude/skills"
_CODEX_SKILLS = _ROOT / ".agents/skills"
_ALLOWED_SKILL_KEYS = {"name", "description", "license", "allowed-tools", "metadata"}
_SKILL_NAME_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


def _skill_dirs(root: Path) -> dict[str, Path]:
    """Return skill directories keyed by the name of each folder."""

    return {
        path.name: path
        for path in root.iterdir()
        if path.is_dir() and (path / "SKILL.md").is_file()
    }


def test_codex_guidance_links_to_claude_guidance():
    """Codex and Claude must read one shared repository instruction file."""

    codex_guidance = _ROOT / "AGENTS.md"
    claude_guidance = _ROOT / "CLAUDE.md"

    assert claude_guidance.is_file()
    assert codex_guidance.is_symlink()
    assert codex_guidance.is_file()
    assert codex_guidance.readlink() == Path("CLAUDE.md")
    assert codex_guidance.resolve() == claude_guidance.resolve()


def test_codex_exposes_every_claude_skill():
    """Every canonical Claude skill must be discoverable through Codex's path."""

    assert _CODEX_SKILLS.is_dir()
    claude_skills = _skill_dirs(_CLAUDE_SKILLS)
    codex_skills = {path.name: path for path in _CODEX_SKILLS.iterdir()}

    assert codex_skills.keys() == claude_skills.keys()
    for name, codex_skill in codex_skills.items():
        assert codex_skill.is_symlink()
        assert codex_skill.is_dir()
        assert codex_skill.readlink() == Path(f"../../.claude/skills/{name}")
        assert codex_skill.resolve() == claude_skills[name].resolve()
        assert (codex_skill / "SKILL.md").is_file()


@pytest.mark.parametrize(
    "skill_dir",
    sorted(_skill_dirs(_CLAUDE_SKILLS).values()),
    ids=lambda path: path.name,
)
def test_project_skill_frontmatter_is_valid(skill_dir: Path):
    """Each shared skill must expose valid metadata with its directory name."""

    prefix, frontmatter, _ = (skill_dir / "SKILL.md").read_text().split("---", 2)
    assert prefix == ""
    metadata = yaml.safe_load(frontmatter)

    assert isinstance(metadata, dict)
    assert metadata.keys() <= _ALLOWED_SKILL_KEYS

    name = metadata["name"]
    assert name == skill_dir.name
    assert _SKILL_NAME_PATTERN.fullmatch(name)
    assert len(name) <= 64

    description = metadata["description"]
    assert isinstance(description, str)
    assert description.strip()
    assert "<" not in description and ">" not in description
    assert len(description) <= 1024
