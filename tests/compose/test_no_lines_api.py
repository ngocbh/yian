"""Compose diagnostics have one range-native location path.

The source document records `SourceRange` values, Compose resolves them into canonical
diagnostic frames, and the CLI renders those frames directly. These ratchets keep the
retired line-map vocabulary out of production code and the public package surface.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest


_ROOT = Path(__file__).resolve().parents[2]
_PRODUCTION_PATHS = (
    _ROOT / "src" / "agent_composer" / "compose",
    _ROOT / "src" / "agent_composer" / "cli" / "run.py",
)
_RETIRED_LOCATION_NAMES = (
    "node_lines",
    "outputs_line",
    "asserts_line",
    "n_lines",
    "s_lines",
    "_section_lines",
    "LoadError.line",
    "LoadError.lines",
)
_RETIRED_LOCATION_PATTERNS = {
    name: re.compile(rf"(?<!\w){re.escape(name)}(?!\w)")
    for name in _RETIRED_LOCATION_NAMES
}


def test_retired_line_map_names_are_absent_from_production() -> None:
    offenders: list[str] = []
    for production_path in _PRODUCTION_PATHS:
        paths = (
            production_path.rglob("*.py")
            if production_path.is_dir()
            else (production_path,)
        )
        for path in paths:
            text = path.read_text()
            for name, pattern in _RETIRED_LOCATION_PATTERNS.items():
                if pattern.search(text):
                    offenders.append(f"{path.relative_to(_ROOT)}: {name}")

    assert offenders == []


def test_lines_api_not_reexported_from_compose():
    import agent_composer.compose as compose

    for name in ("node_lines", "section_lines"):
        assert not hasattr(compose, name), f"{name} should be removed from compose"
    with pytest.raises(ImportError):
        from agent_composer.compose import node_lines  # noqa: F401
    with pytest.raises(ImportError):
        from agent_composer.compose import section_lines  # noqa: F401


def test_lines_producers_gone_from_parser():
    import agent_composer.compose.parse.file as parser

    for name in (
        "node_lines",
        "section_lines",
        "node_field_lines",
        "node_input_lines",
        "assert_lines",
        "input_decl_lines",
        "code_content_lines",
    ):
        assert not hasattr(parser, name), f"parser.{name} should be removed"
