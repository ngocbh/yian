"""Step 7 — the `compose/stages/` graph-stage axis: cross-node assembly passes.

7.0 scaffold: the package exists with its charter. Later sub-tasks (7.1+) populate it with
one relocated assembly pass each, driven off the span tree's `.range`/`field_ranges`.

7.1 — `stages/parse.parse_nodes`: the `nodes:` mapping pass. Consumes a `NodeReader` over
the span tree (not `yaml.safe_load` + `node_lines`) and dispatches each body to its kind's
`parse` via `KINDS`, returning located descriptors that carry `field_ranges`."""

import pytest

import agent_composer.compose.stages as stages
from agent_composer.compose.errors import LoadError
from agent_composer.compose.kinds.agent import AgentDescriptor
from agent_composer.compose.reader import NodeReader
from agent_composer.compose.spanloader import load_with_spans
from agent_composer.compose.stages.inline_calls import (
    _synth_id_gen,
    desugar_inline_calls,
)
from agent_composer.compose.stages.parse import parse_nodes


def test_stages_package_imports():
    assert stages.__doc__ is not None


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


# ---------- 7.2 desugar_inline_calls: synth call nodes inherit host ranges ----------


def test_desugar_inline_calls_preserves_ranges():
    # A whole-value `call(...)` in a node's `input:` binding lifts to a synth `call` node.
    # The synth CallDescriptor must inherit the HOST node's range/field_ranges (so its
    # downstream errors locate at the host binding site) — no `node_lines` map needed.
    text = (
        "id: f\nname: f\nnodes:\n"
        "  writer:\n    kind: code\n    code: m:f\n"
        "    input:\n      x: call(helper, n=3)\n    output: str\n"
        "output: ${writer.output}\n"
    )
    descriptors = parse_nodes(_nodes_reader(text))
    host = descriptors["writer"]
    new_descriptors, _outputs, _asserts = desugar_inline_calls(
        descriptors, None, next_id=_synth_id_gen()
    )
    synth = [d for nid, d in new_descriptors.items() if nid.startswith("__call_")]
    assert len(synth) == 1
    call_desc = synth[0]
    assert call_desc.call == "helper"
    assert call_desc.range == host.range
    assert call_desc.field_ranges == host.field_ranges
