"""Tests for the edges phase: deriving ordering edges (`edges/ordering.py`).

`infer_ordering_edges` lowers each `depends_on:`/`runs_after:` declaration into a
zero-data ordering edge, and reports an unknown target as a `LoadError` located at
the declaring node descriptor's range.
"""

import pytest

from agent_composer.compose.helpers.errors import LoadDiagnostics, LoadError
from agent_composer.compose.edges.ordering import infer_ordering_edges
from agent_composer.compose.parse.nodes import parse_nodes
from agent_composer.compose.source import SourceDocument


def _nodes_reader(text):
    """Return the physical cursor over a flow document's ``nodes:`` mapping."""

    return SourceDocument.parse(text).root.child("nodes")


def test_infer_ordering_edges_unknown_target_is_located():
    # A `depends_on:`/`runs_after:` that names an unknown node is a LoadError located at the
    # DECLARING node descriptor's range.
    text = (
        "id: f\nname: f\nnodes:\n"
        "  a:\n    kind: code\n    code: m:f\n    depends_on: [ghost]\n    output: str\n"
        "output: ${a.output}\n"
    )
    nodes_cursor = _nodes_reader(text)
    descriptors = parse_nodes(nodes_cursor)
    with pytest.raises(LoadError) as exc:
        infer_ordering_edges(
            descriptors,
            set(descriptors),
            diagnostics=LoadDiagnostics.from_cursor(nodes_cursor),
        )
    assert "ghost" in str(exc.value)
    assert len(exc.value.diagnostic.frames) == 1
    assert exc.value.diagnostic.frames[0].range == descriptors["a"].range
    assert exc.value.diagnostic.frames[0].text == text


def test_infer_ordering_edges_builds_ordering_edges():
    # A valid `depends_on:` lowers to a zero-data ordering edge (optional=False).
    text = (
        "id: f\nname: f\nnodes:\n"
        "  warm:\n    kind: code\n    code: m:f\n    output: str\n"
        "  a:\n    kind: code\n    code: m:f\n    depends_on: [warm]\n    output: str\n"
        "output: ${a.output}\n"
    )
    descriptors = parse_nodes(_nodes_reader(text))
    edges = infer_ordering_edges(descriptors, set(descriptors))
    assert len(edges) == 1
    assert edges[0].from_ == "warm"
    assert edges[0].to == "a"
    assert edges[0].ordering is True
    assert edges[0].optional is False
