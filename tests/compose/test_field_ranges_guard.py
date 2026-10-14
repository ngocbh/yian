"""Tests for transient Compose-owned range metadata."""

from yian.compose.nodes.base import BuildContext
from yian.compose.parse.nodes import parse_nodes
from yian.compose.source import SourceDocument


_FLOW = """
id: ranges
name: ranges
input:
  topic: str
nodes:
  n:
    kind: code
    input:
      topic: ${input.topic}
    output: int
    code: "return {'output': 1}"
output: ${n.output}
"""


def test_descriptor_ranges_move_to_build_result_only() -> None:
    """Building retains YAML ranges in Compose without stamping the semantic node."""
    document = SourceDocument.parse(_FLOW)
    descriptor = parse_nodes(document.root.child("nodes"))["n"]

    result = descriptor.build(BuildContext({}))

    assert type(descriptor.field_ranges) is dict
    assert result.node_range is descriptor.range
    assert result.field_ranges == descriptor.field_ranges
    assert result.field_ranges is not descriptor.field_ranges
    assert result.field_ranges[("input", "topic")].start_line == 10
    assert not hasattr(result.node, "range")
    assert not hasattr(result.node, "field_ranges")


def test_case_generated_input_range_stays_in_build_result() -> None:
    """CASE adds synthetic parameter ranges to the same transient carrier."""
    text = """
nodes:
  gate:
    kind: case
    cases:
      - when: missing.output > 0
        then: done
    else: done
"""
    document = SourceDocument.parse(text)
    descriptor = parse_nodes(document.root.child("nodes"))["gate"]

    result = descriptor.build(BuildContext({}))

    assert result.field_ranges[("input", "__r0")].start_line == 6
    assert not hasattr(result.node, "field_ranges")
