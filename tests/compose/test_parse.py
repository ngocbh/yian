"""Tests for the parse phase: the `nodes:` mapping pass (`parse/nodes.py`).

`parse_nodes` consumes a `NodeReader` over the span tree and dispatches each node
body to its kind's `parse` via `KINDS`, returning located descriptors that carry
`.range`/`field_ranges` (or `None`/`{}` for a spanless reader).
"""

import pytest

from agent_composer.compose.helpers.errors import LoadError
from agent_composer.compose.helpers.reader import NodeReader
from agent_composer.compose.helpers.spanloader import load_with_spans
from agent_composer.compose.nodes.agent import AgentDescriptor
from agent_composer.compose.parse.nodes import parse_nodes


def _nodes_reader(text):
    """A span-aware NodeReader over the `nodes:` mapping of a flow document."""
    value, spans, root = load_with_spans(text, return_root=True)
    return NodeReader(value, spans, root).child("nodes")


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
    # Located: the span tree stamps ranges onto the descriptor + its fields.
    assert desc.range is not None
    assert desc.field_ranges["prompt"].start_line == 6  # 1-based, the prompt line
    # The back-mapped plural fields resolve their source ranges (via the rewritten key).
    assert desc.field_ranges["inputs"] is not None
    assert desc.field_ranges["outputs"] is not None


def test_parse_nodes_unknown_kind_is_located():
    text = "id: f\nname: f\nnodes:\n  n:\n    kind: wizard\n    output: str\noutput: 1\n"
    with pytest.raises(LoadError) as exc:
        parse_nodes(_nodes_reader(text))
    assert "wizard" in str(exc.value)
    assert exc.value.range is not None  # located at the `kind:` value


def test_parse_nodes_reserved_node_id_is_loud():
    text = "id: f\nname: f\nnodes:\n  input:\n    kind: agent\n    prompt: hi\noutput: 1\n"
    with pytest.raises(LoadError, match="reserved"):
        parse_nodes(_nodes_reader(text))


def test_parse_nodes_spanless_reader():
    # A spanless reader (defs / compact / in-memory) parses the already-plural body with
    # no source ranges — descriptors still build, ranges are None.
    reader = NodeReader.spanless(
        {"w": {"kind": "agent", "prompt": "hi", "inputs": {"x": "${input.x}"}, "outputs": "str"}}
    )
    desc = parse_nodes(reader)["w"]
    assert desc == AgentDescriptor(id="w", prompt="hi", inputs={"x": "${input.x}"}, outputs="str")
    assert desc.range is None
    assert desc.field_ranges == {}
