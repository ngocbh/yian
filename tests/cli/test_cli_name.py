"""The console script and the Typer app are both named yian."""

from __future__ import annotations

import re
from pathlib import Path

from typer.testing import CliRunner

from yian.cli import app

_ANSI_SGR = re.compile(r"\x1b\[[0-9;]*m")
_PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"


def _console_scripts() -> dict[str, str]:
    """Return the `[project.scripts]` table of `pyproject.toml` as name -> target."""
    text = _PYPROJECT.read_text()
    section = text.split("[project.scripts]", 1)[1].split("\n[", 1)[0]
    return dict(re.findall(r'^(\S+)\s*=\s*"([^"]+)"', section, flags=re.MULTILINE))


def test_console_script_is_composer():
    assert _console_scripts() == {"yian": "yian.cli:main"}


def test_help_names_composer():
    r = CliRunner().invoke(app, ["--help"], env={"COLUMNS": "200"})
    assert r.exit_code == 0
    assert "Usage: yian " in _ANSI_SGR.sub("", r.output)


_ROOT = _PYPROJECT.parent
_STALE = re.compile(r"\bac (?:(?:run|chat|auth)\b|--help\b|\.\.\.)|`ac`|\bThe ac CLI\b")
_SCANNED = [
    _ROOT / "README.md",
    _ROOT / "CLAUDE.md",
    _ROOT / "CONTRIBUTING.md",
    _ROOT / "mkdocs.yml",
    _ROOT / ".env.example",
    _ROOT / ".github" / "PULL_REQUEST_TEMPLATE.md",
    *(_ROOT / "docs").glob("*.md"),
    *(_ROOT / ".github" / "ISSUE_TEMPLATE").glob("*.yml"),
    *(_ROOT / ".claude" / "skills").rglob("*.md"),
    *(_ROOT / ".claude" / "skills").rglob("*.yaml"),
    *(_ROOT / "examples").glob("*.*"),
    *(_ROOT / "src").rglob("*.py"),
    *(_ROOT / "src").rglob("*.md"),
    *(_ROOT / "src").rglob("*.yaml"),
    *(_ROOT / "tests").rglob("*.py"),
    *(_ROOT / "tests" / "seeds").rglob("*.yaml"),
]


def test_docs_and_sources_name_the_composer_command():
    """User-facing text must not tell anyone to run the old command name."""
    stale = [
        f"{path.relative_to(_ROOT)}:{lineno}: {line.strip()}"
        for path in _SCANNED
        if path.resolve() != Path(__file__).resolve()
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if _STALE.search(line)
    ]
    assert stale == []
