"""Shared isolation for CLI tests that exercise default user-state paths."""

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolated_cli_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point ``Path.home()`` at a fresh directory for every CLI test."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    return home
