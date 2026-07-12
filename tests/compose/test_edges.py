"""Tests for the edges phase: deriving ordering edges (`edges/ordering.py`).

`infer_ordering_edges` lowers each `depends_on:`/`runs_after:` declaration into a
zero-data ordering edge, and reports an unknown target as a `LoadError` located at
the declaring node descriptor's range.
"""

import pytest

from agent_composer.compose.helpers.errors import LoadError
from agent_composer.compose.helpers.reader import NodeReader
from agent_composer.compose.helpers.spanloader import load_with_spans
from agent_composer.compose.edges.ordering import infer_ordering_edges
from agent_composer.compose.parse.nodes import parse_nodes


def _nodes_reader(text):
    """A span-aware NodeReader over the `nodes:` mapping of a flow document."""
    value, spans, root = load_with_spans(text, return_root=True)
    return NodeReader(value, spans, root).child("nodes")


def test_infer_ordering_edges_unknown_target_is_located():
    # A `depends_on:`/`runs_after:` that names an unknown node is a LoadError located at the
    # DECLARING node descriptor's range (no `node_lines` map).
    text = (
        "id: f\nname: f\nnodes:\n"
        "  a:\n    kind: code\n    code: m:f\n    depends_on: [ghost]\n    output: str\n"
        "output: ${a.output}\n"
    )
    descriptors = parse_nodes(_nodes_reader(text))
    with pytest.raises(LoadError) as exc:
        infer_ordering_edges(descriptors, set(descriptors))
    assert "ghost" in str(exc.value)
    assert exc.value.range == descriptors["a"].range


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
