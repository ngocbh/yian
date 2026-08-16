"""The deleted SourceFrame graph bridge cannot return unnoticed.

Runtime diagnostics now cross the Compose boundary only as opaque origin traces resolved by
the installed authority. These ratchets keep the retired graph carriers and CLI reconstruction
helpers out of production code and constructor surfaces.
"""

from __future__ import annotations

import importlib
import inspect
import re
from pathlib import Path

from agent_composer.nodes.call import CallNode
from agent_composer.nodes.loop import LoopNode
from agent_composer.nodes.map import MapNode


_ROOT = Path(__file__).resolve().parents[2]
_PRODUCTION_ROOT = _ROOT / "src" / "agent_composer"
_RETIRED_GRAPH_NAMES = ("SourceFrame", "child_source", "top_frame_ranges")
_RETIRED_GRAPH_PATTERNS = {
    name: re.compile(rf"(?<!\w){re.escape(name)}(?!\w)")
    for name in _RETIRED_GRAPH_NAMES
}
_RETIRED_CLI_HELPERS = (
    "_KIND_FALLBACK_FIELDS",
    "_FIELD_KEY_ALIAS",
    "_field_range",
    "_top_ranges",
    "_node_range",
    "_resolve_range",
    "_strip_clone_suffix",
    "_node_owns_resolved_range",
    "_walk_call_frames",
    "_render_frame_stack",
)


def test_retired_source_graph_names_are_absent_from_production() -> None:
    """No production module may recreate the deleted cross-layer graph vocabulary."""
    offenders: list[str] = []
    for path in _PRODUCTION_ROOT.rglob("*.py"):
        text = path.read_text()
        for name, pattern in _RETIRED_GRAPH_PATTERNS.items():
            if pattern.search(text):
                offenders.append(f"{path.relative_to(_ROOT)}: {name}")

    assert offenders == []


def test_composite_nodes_have_no_child_source_parameter() -> None:
    """CALL, MAP, and LOOP carry executable children, never presentation graphs."""
    for node_type in (CallNode, MapNode, LoopNode):
        assert "child_source" not in inspect.signature(node_type).parameters


def test_retired_source_and_cli_apis_are_absent() -> None:
    """The old frame carrier and CLI graph walker are not importable private APIs."""
    source_helpers = importlib.import_module("agent_composer.compose.helpers.source")
    cli_run = importlib.import_module("agent_composer.cli.run")

    assert not hasattr(source_helpers, "SourceFrame")
    assert not hasattr(source_helpers, "top_frame_ranges")
    for name in _RETIRED_CLI_HELPERS:
        assert not hasattr(cli_run, name), f"cli.run.{name} should remain deleted"
