"""SKILL.md parsing, root precedence, and diagnostics."""

from __future__ import annotations

from pathlib import Path

import pytest

from yian.compose.skills import load_skill_dir, load_skills
from yian.skills import Skill, SkillError
from yian.skills.catalog import MAX_SKILL_FILE_BYTES


def write_skill(root: Path, dirname: str, frontmatter: str, body: str = "Do the thing.\n") -> Path:
    """Create ``root/dirname/SKILL.md`` and return the skill directory."""
    directory = root / dirname
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "SKILL.md").write_text(f"---\n{frontmatter}---\n\n{body}", encoding="utf-8")
    return directory


def test_load_skill_dir_reads_frontmatter_and_body(tmp_path: Path) -> None:
    directory = write_skill(
        tmp_path,
        "pdf-tools",
        "name: pdf-tools\ndescription: Extract text from PDFs.\nlicense: MIT\nallowed-tools: Bash\n",
        "# PDF\n\nRun `scripts/extract.py`.\n",
    )
    skill = load_skill_dir(directory)
    assert skill == Skill(
        "pdf-tools", "Extract text from PDFs.", "# PDF\n\nRun `scripts/extract.py`.\n", directory.resolve()
    )


def test_name_defaults_to_directory_and_frontmatter_name_wins(tmp_path: Path) -> None:
    assert load_skill_dir(write_skill(tmp_path, "review", "description: Review code.\n")).name == "review"
    assert load_skill_dir(write_skill(tmp_path, "x", "name: triage\ndescription: Triage.\n")).name == "triage"


def test_frontmatter_reads_yes_and_no_as_text(tmp_path: Path) -> None:
    # Frontmatter follows the flow's YAML 1.2 rules, as Claude Code and Codex read it.
    skill = load_skill_dir(write_skill(tmp_path, "no", "name: no\ndescription: yes\n"))
    assert (skill.name, skill.description) == ("no", "yes")


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("no frontmatter\n", "must start with `---`"),
        ("---\nname: a\n", "no closing `---`"),
        ("---\n: [\n---\nbody\n", "not valid YAML"),
        ("---\n- a\n---\nbody\n", "must be a mapping"),
        ("---\nname: a\n---\nbody\n", "non-empty `description`"),
        ("---\nname: Bad_Name\ndescription: d\n---\n", "invalid skill name"),
        ("---\nname: a--b\ndescription: d\n---\n", "invalid skill name"),
    ],
)
def test_bad_skill_files_name_the_file(tmp_path: Path, content: str, message: str) -> None:
    directory = tmp_path / "a"
    directory.mkdir()
    (directory / "SKILL.md").write_text(content, encoding="utf-8")
    with pytest.raises(SkillError, match="SKILL.md") as error:
        load_skill_dir(directory)
    assert message in str(error.value)


def test_missing_and_oversized_skill_files_are_rejected(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    with pytest.raises(SkillError, match="missing SKILL.md"):
        load_skill_dir(tmp_path / "empty")
    big = write_skill(tmp_path, "big", "description: d\n", "x" * MAX_SKILL_FILE_BYTES)
    with pytest.raises(SkillError, match="larger than"):
        load_skill_dir(big)


def test_load_skills_earlier_root_wins_and_skips_non_skills(tmp_path: Path) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    write_skill(first, "review", "description: First review.\n")
    write_skill(second, "review", "description: Second review.\n")
    write_skill(second, "deploy", "description: Deploy.\n")
    (second / "notes").mkdir()
    write_skill(second, ".hidden", "description: Hidden.\n")
    skills = load_skills([first, second, tmp_path / "missing"])
    assert list(skills) == ["deploy", "review"]
    assert skills["review"].description == "First review."


def test_load_skills_follows_symlinked_skill_directories(tmp_path: Path) -> None:
    real = write_skill(tmp_path / "store", "shared", "description: Shared.\n")
    root = tmp_path / "root"
    root.mkdir()
    (root / "shared").symlink_to(real, target_is_directory=True)
    assert load_skills([root])["shared"].directory == real.resolve()


def test_load_skills_reports_every_problem_at_once(tmp_path: Path) -> None:
    write_skill(tmp_path, "one", "name: same\ndescription: One.\n")
    write_skill(tmp_path, "two", "name: same\ndescription: Two.\n")
    write_skill(tmp_path, "bad", "name: bad\n")
    with pytest.raises(SkillError) as error:
        load_skills([tmp_path])
    text = str(error.value)
    assert "'same' is used by both one/ and two/" in text
    assert "bad" in text and "description" in text


def test_load_skills_skip_invalid_warns_and_keeps_the_rest(tmp_path: Path) -> None:
    write_skill(tmp_path, "good", "description: Good.\n")
    write_skill(tmp_path, "bad", "name: bad\n")
    with pytest.warns(UserWarning, match="skipped skill"):
        skills = load_skills([tmp_path], skip_invalid=True)
    assert list(skills) == ["good"]


def test_load_skills_reports_unreadable_skill_files(tmp_path: Path) -> None:
    write_skill(tmp_path, "good", "description: Good.\n")
    locked = write_skill(tmp_path, "locked", "description: Locked.\n") / "SKILL.md"
    locked.chmod(0)
    try:
        with pytest.raises(SkillError, match="locked: cannot read skill"):
            load_skills([tmp_path])
        with pytest.warns(UserWarning, match="locked: cannot read skill"):
            assert list(load_skills([tmp_path], skip_invalid=True)) == ["good"]
    finally:
        locked.chmod(0o644)
