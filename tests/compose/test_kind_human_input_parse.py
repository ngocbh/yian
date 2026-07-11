"""6.6 — the `human_input` kind handle: parse enforces the one-of-three question-surface
legality and records field ranges; build returns a runtime HumanInputNode carrying its
questions config + derived answer_schema, with wiring from `inputs` and stamped ranges."""

import pytest

from agent_composer.compose.errors import LoadError
from agent_composer.compose.kinds.human_input import HumanInputDescriptor, build, parse
from agent_composer.compose.reader import NodeReader
from agent_composer.compose.spanloader import load_with_spans
from agent_composer.events import SourceRange


_FLOW = """\
nodes:
  ask:
    kind: human_input
    prompt: Approve this?
    inputs:
      ctx: ${input.ctx}
"""

_QUESTIONS_FLOW = """\
nodes:
  ask:
    kind: human_input
    questions:
      - question: Pick one
        header: Choice
        options: [a, b]
"""


def _reader_for(text, node_id):
    value, spans, root = load_with_spans(text, return_root=True)
    root_reader = NodeReader(value, spans, root)
    return root_reader.child("nodes").child(node_id)


def test_human_input_parse_matches_descriptor_and_records_ranges():
    reader = _reader_for(_FLOW, "ask")
    desc = parse("ask", reader)

    assert desc == HumanInputDescriptor(
        id="ask",
        prompt="Approve this?",
        inputs={"ctx": "${input.ctx}"},
    )
    assert isinstance(desc.range, SourceRange)
    assert isinstance(desc.field_ranges["prompt"], SourceRange)
    assert isinstance(desc.field_ranges["inputs"], SourceRange)


def test_human_input_build_returns_gate_and_wires_inputs():
    reader = _reader_for(_FLOW, "ask")
    desc = parse("ask", reader)
    node, wiring = build(desc, registry={}, resolver=None)

    from agent_composer.nodes.human_input import HumanInputNode

    assert isinstance(node, HumanInputNode)
    assert node.prompt == "Approve this?"
    assert wiring == {"ctx": "${input.ctx}"}
    assert node.range == desc.range


def test_human_input_build_questions_gate_defaults_to_open_object():
    reader = _reader_for(_QUESTIONS_FLOW, "ask")
    desc = parse("ask", reader)
    node, wiring = build(desc, registry={}, resolver=None)

    from agent_composer.nodes.human_input import HumanInputNode
    from agent_composer.typesys.values import ValueKind

    assert isinstance(node, HumanInputNode)
    # A questions gate with no explicit `outputs:` defaults to a bare open OBJECT.
    assert node.output_type is not None
    assert node.output_type.kind == ValueKind.OBJECT
    assert node.answer_schema is not None


def test_human_input_parse_requires_a_question_surface():
    flow = "nodes:\n  ask:\n    kind: human_input\n    inputs:\n      x: ${input.x}\n"
    reader = _reader_for(flow, "ask")
    with pytest.raises(LoadError, match="needs `prompt:`"):
        parse("ask", reader)


def test_human_input_parse_questions_and_adaptive_are_mutually_exclusive():
    flow = (
        "nodes:\n"
        "  ask:\n"
        "    kind: human_input\n"
        "    questions:\n"
        "      - question: Pick\n"
        "        header: H\n"
        "        options: [a, b]\n"
        "    adaptive_questions:\n"
        "      prompt: author questions\n"
    )
    reader = _reader_for(flow, "ask")
    with pytest.raises(LoadError, match="mutually exclusive"):
        parse("ask", reader)
