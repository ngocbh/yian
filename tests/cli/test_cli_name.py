"""The console script and the Typer app are both named composer."""

from __future__ import annotations

import re
from pathlib import Path

from typer.testing import CliRunner

from agent_composer.cli import app

_ANSI_SGR = re.compile(r"\x1b\[[0-9;]*m")
_PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"


def _console_scripts() -> dict[str, str]:
    """Return the `[project.scripts]` table of `pyproject.toml` as name -> target."""
    text = _PYPROJECT.read_text()
    section = text.split("[project.scripts]", 1)[1].split("\n[", 1)[0]
    return dict(re.findall(r'^(\S+)\s*=\s*"([^"]+)"', section, flags=re.MULTILINE))


def test_console_script_is_composer():
    assert _console_scripts() == {"composer": "agent_composer.cli:main"}


def test_help_names_composer():
    r = CliRunner().invoke(app, ["--help"], env={"COLUMNS": "200"})
    assert r.exit_code == 0
    assert "Usage: composer " in _ANSI_SGR.sub("", r.output)
