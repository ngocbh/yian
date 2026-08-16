"""Tests for the parse phase: the `nodes:` mapping pass (`parse/nodes.py`).

`parse_nodes` consumes a `SourceCursor` over the source document and dispatches each node
body to its kind's `parse` via `KINDS`, returning located descriptors that carry
`.range`/`field_ranges` (or `None`/`{}` for a virtual cursor).
"""

import pytest

from agent_composer.compose.helpers.errors import LoadError
from agent_composer.compose.nodes.agent import AgentDescriptor
from agent_composer.compose.parse.nodes import parse_nodes
from agent_composer.compose.source import SourceCursor, SourceDocument


def _nodes_reader(text):
    """Return the physical cursor over a flow document's ``nodes:`` mapping."""

    return SourceDocument.parse(text).root.child("nodes")


_FLOW = """\
id: f
name: f
nodes:
  writer:
    kind: agent
    prompt: "Write about ${topic}"
    input:
      topic: ${input.topic}
    output: string
output: ${writer.output}
"""


def test_parse_nodes_uses_span_tree():
    descriptors = parse_nodes(_nodes_reader(_FLOW))

    desc = descriptors["writer"]
    # The descriptor is exactly today's AgentDescriptor (field_ranges excluded from ==);
    # the singular `input:`/`output:` source keys are back-mapped to the plural fields.
    assert desc == AgentDescriptor(
        id="writer",
        prompt="Write about ${topic}",
        inputs={"topic": "${input.topic}"},
        outputs="string",
    )
    # Located: the physical cursor stamps ranges onto the descriptor + its fields.
    assert desc.range is not None
    assert desc.field_ranges[("field", "prompt")].start_line == 6  # 1-based, the prompt line
    # The back-mapped plural fields resolve their source ranges (via the rewritten key).
    assert desc.field_ranges[("field", "inputs")] is not None
    assert desc.field_ranges[("field", "outputs")] is not None


def test_parse_nodes_unknown_kind_is_located():
    text = "id: f\nname: f\nnodes:\n  n:\n    kind: wizard\n    output: str\noutput: 1\n"
    with pytest.raises(LoadError) as exc:
        parse_nodes(_nodes_reader(text))
    assert "wizard" in str(exc.value)
    assert exc.value.diagnostic.frames[-1].range is not None


def test_parse_nodes_reserved_node_id_is_loud():
    text = "id: f\nname: f\nnodes:\n  input:\n    kind: agent\n    prompt: hi\noutput: 1\n"
    with pytest.raises(LoadError, match="reserved"):
        parse_nodes(_nodes_reader(text))


def test_parse_nodes_virtual_cursor():
    # A virtual cursor (defs / compact / in-memory) parses the already-plural body with
    # no source ranges — descriptors still build, ranges are None.
    reader = SourceCursor.virtual(
        {"w": {"kind": "agent", "prompt": "hi", "inputs": {"x": "${input.x}"}, "outputs": "str"}}
    )
    desc = parse_nodes(reader)["w"]
    assert desc == AgentDescriptor(id="w", prompt="hi", inputs={"x": "${input.x}"}, outputs="str")
    assert desc.range is None
    assert desc.field_ranges == {}


def test_parse_nodes_normalizes_without_mutating_cursor():
    nodes = SourceDocument.parse(_FLOW).root.child("nodes")
    before = nodes.value

    desc = parse_nodes(nodes)["writer"]

    assert desc.inputs == {"topic": "${input.topic}"}
    assert desc.outputs == "string"
    assert nodes.value == before
    assert nodes.child("writer").has("input")
    assert not nodes.child("writer").has("inputs")


def test_case_on_normalization_does_not_mutate_cursor():
    source = """\
nodes:
  route:
    kind: case
    on: ${score.output}
    cases:
      - when: high
        then: hot
"""
    nodes = SourceDocument.parse(source).root.child("nodes")
    route = nodes.child("route")

    desc = parse_nodes(nodes)["route"]

    assert desc.on == "${score.output}"
    assert route.has(True)
    assert not route.has("on")


def test_case_normalization_uses_last_effective_boolean_key():
    source = """\
nodes:
  route:
    kind: case
    1: ignored
    on: ${score.output}
    cases: []
"""
    nodes = SourceDocument.parse(source).root.child("nodes")

    desc = parse_nodes(nodes)["route"]

    assert desc.on == "${score.output}"


def test_case_normalization_rejects_last_effective_numeric_key():
    source = """\
nodes:
  route:
    kind: case
    on: ${score.output}
    1: ignored
    cases: []
"""
    nodes = SourceDocument.parse(source).root.child("nodes")
    route = nodes.child("route")

    with pytest.raises(LoadError, match="field 1 is not allowed") as caught:
        parse_nodes(nodes)

    assert caught.value.diagnostic.frames[-1].range == route.range_of(True)
