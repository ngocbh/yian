"""Bundled-file confinement and default Claude/Codex skill roots."""

from __future__ import annotations

from pathlib import Path

import pytest

from yian.compose.skills import default_skill_roots
from yian.defaults import current
from yian.skills import Skill, SkillError
from yian.skills.catalog import read_skill_file


@pytest.fixture
def skill(tmp_path: Path) -> Skill:
    directory = tmp_path / "skills" / "api"
    (directory / "references").mkdir(parents=True)
    (directory / "references" / "api.md").write_text("GET /items\n", encoding="utf-8")
    (tmp_path / "secret.txt").write_text("top secret\n", encoding="utf-8")
    return Skill("api", "Call the API.", "Read references/api.md.", directory.resolve())


def test_skill_value_validates_name_and_description() -> None:
    with pytest.raises(SkillError, match="invalid skill name"):
        Skill("Has Space", "d", "")
    with pytest.raises(SkillError, match="description"):
        Skill("ok", "  ", "")


def test_read_skill_file_returns_bundled_text(skill: Skill) -> None:
    assert read_skill_file(skill, "references/api.md") == "GET /items\n"
    assert read_skill_file(skill, "./references/api.md") == "GET /items\n"


@pytest.mark.parametrize(
    "path", ["", "../../secret.txt", "references/../../../secret.txt", "/etc/passwd", "references\\api.md", "a\x00b"]
)
def test_read_skill_file_rejects_paths_outside_the_skill(skill: Skill, path: str) -> None:
    with pytest.raises(SkillError, match="relative path inside the skill directory"):
        read_skill_file(skill, path)


def test_read_skill_file_rejects_symlink_escape(skill: Skill, tmp_path: Path) -> None:
    assert skill.directory is not None
    (skill.directory / "leak.txt").symlink_to(tmp_path / "secret.txt")
    with pytest.raises(SkillError, match="outside the skill directory"):
        read_skill_file(skill, "leak.txt")


@pytest.mark.parametrize("path", [".env", ".git/config", "references/.secret"])
def test_read_skill_file_rejects_hidden_files(skill: Skill, path: str) -> None:
    assert skill.directory is not None
    (skill.directory / ".git").mkdir()
    for hidden in (".env", ".git/config", "references/.secret"):
        (skill.directory / hidden).write_text("TOKEN=1\n", encoding="utf-8")
    with pytest.raises(SkillError, match="hidden files and folders are not readable"):
        read_skill_file(skill, path)


def test_read_skill_file_rejects_missing_directories_binary_and_large(skill: Skill) -> None:
    assert skill.directory is not None
    (skill.directory / "blob.bin").write_bytes(b"\xff\xfe\x00")
    (skill.directory / "big.txt").write_text("x" * (current().max_skill_file_bytes + 1), encoding="utf-8")
    with pytest.raises(SkillError, match="does not exist"):
        read_skill_file(skill, "references")
    with pytest.raises(SkillError, match="does not exist"):
        read_skill_file(skill, "nope.md")
    with pytest.raises(SkillError, match="not UTF-8"):
        read_skill_file(skill, "blob.bin")
    with pytest.raises(SkillError, match="larger than"):
        read_skill_file(skill, "big.txt")
    with pytest.raises(SkillError, match="no bundled files"):
        read_skill_file(Skill("inline", "d", "body"), "x.md")


def test_default_roots_put_user_before_project_and_stop_at_git_root(tmp_path: Path) -> None:
    home = tmp_path / "home"
    repo = tmp_path / "outer" / "repo"
    work = repo / "pkg" / "sub"
    work.mkdir(parents=True)
    (repo / ".git").mkdir()
    roots = default_skill_roots(cwd=work, home=home)
    base = [Path(".claude/skills"), Path(".agents/skills")]
    assert roots == [
        *(home / part for part in base),
        *(work.resolve() / part for part in base),
        *(work.resolve().parent / part for part in base),
        *(repo.resolve() / part for part in base),
    ]


def test_default_roots_outside_git_use_only_cwd(tmp_path: Path) -> None:
    roots = default_skill_roots(cwd=tmp_path, home=tmp_path / "home")
    assert roots[2:] == [tmp_path.resolve() / ".claude/skills", tmp_path.resolve() / ".agents/skills"]


def test_compose_facade_exports_the_skill_loaders() -> None:
    from yian import compose
    from yian.compose import skills as compose_skills

    assert {"load_skills", "default_skill_roots"} <= set(compose.__all__)
    assert compose.default_skill_roots is compose_skills.default_skill_roots
    assert compose.load_skills is compose_skills.load_skills
