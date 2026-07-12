"""The Step-8 cutover removed the `*_lines` line-map API — locations ride `SourceRange` now.

The single-pass span parser records a `SourceRange` on every node/field, and the CLI renders
via `render_diagnostic` fed by those ranges. The legacy secondary-parse `*_lines` producers
(and their public re-exports) are gone; this pins that they cannot be imported again.
"""

from __future__ import annotations

import pytest


def test_lines_api_not_reexported_from_compose():
    import agent_composer.compose as compose

    for name in ("node_lines", "section_lines"):
        assert not hasattr(compose, name), f"{name} should be removed from compose"
    with pytest.raises(ImportError):
        from agent_composer.compose import node_lines  # noqa: F401
    with pytest.raises(ImportError):
        from agent_composer.compose import section_lines  # noqa: F401


def test_lines_producers_gone_from_parser():
    import agent_composer.compose.parser as parser

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
