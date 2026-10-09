"""The `human_input` kind handle: parsing enforces the one-of-three question-surface
legality and records field ranges; build returns a runtime HumanInputNode carrying its
questions config + derived answer_schema, with wiring from `inputs` and stamped ranges."""

import pytest

from yian.compose.helpers.errors import LoadError
from yian.compose.nodes.base import BuildContext
from yian.compose.nodes.human_input import HumanInputDescriptor
from yian.compose.source import SourceDocument
from yian.diagnostics import SourceRange


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
    return SourceDocument.parse(text).root.child("nodes").child(node_id)


def test_human_input_parse_matches_descriptor_and_records_ranges():
    reader = _reader_for(_FLOW, "ask")
    desc = HumanInputDescriptor.parse("ask", reader)

    assert desc == HumanInputDescriptor(
        id="ask",
        prompt="Approve this?",
        inputs={"ctx": "${input.ctx}"},
    )
    assert isinstance(desc.range, SourceRange)
    assert isinstance(desc.field_ranges[("field", "prompt")], SourceRange)
    assert isinstance(desc.field_ranges[("field", "inputs")], SourceRange)


def test_human_input_build_returns_gate_and_wires_inputs():
    reader = _reader_for(_FLOW, "ask")
    desc = HumanInputDescriptor.parse("ask", reader)
    result = desc.build(BuildContext({}))
    node, wiring = result.node, result.wiring

    from yian.nodes.human_input import HumanInputNode

    assert isinstance(node, HumanInputNode)
    assert node.prompt == "Approve this?"
    assert wiring == {"ctx": "${input.ctx}"}
    assert result.node_range == desc.range


def test_human_input_build_questions_gate_defaults_to_open_object():
    reader = _reader_for(_QUESTIONS_FLOW, "ask")
    desc = HumanInputDescriptor.parse("ask", reader)
    result = desc.build(BuildContext({}))
    node = result.node

    from yian.nodes.human_input import HumanInputNode
    from yian.typesys.values import ValueKind

    assert isinstance(node, HumanInputNode)
    # A questions gate with no explicit `output:` defaults to a bare open OBJECT.
    assert node.output_contract is not None
    assert node.output_contract.type.kind == ValueKind.OBJECT
    assert node.answer_schema is not None


def test_human_input_parse_requires_a_question_surface():
    flow = "nodes:\n  ask:\n    kind: human_input\n    inputs:\n      x: ${input.x}\n"
    reader = _reader_for(flow, "ask")
    with pytest.raises(LoadError, match="needs `prompt:`") as exc:
        HumanInputDescriptor.parse("ask", reader)
    # The absent-question-surface error anchors on the node body range (no question field
    # exists to carry one) — a regression guard: it must not silently lose its location.
    assert isinstance(exc.value.diagnostic.frames[-1].range, SourceRange)


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
        HumanInputDescriptor.parse("ask", reader)
