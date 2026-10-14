"""The `human_input` kind handle: parsing enforces the one-of-three question-surface
legality and records field ranges; build returns a runtime HumanInputNode carrying its
questions config + derived answer_schema, with wiring from `inputs` and stamped ranges."""

import pytest

from yian.compose.helpers.errors import LoadError
from yian.compose.nodes.base import BuildContext
from yian.compose.nodes.human_input import HumanInputDescriptor
from yian.compose.source import SourceDocument
from yian.diagnostics import SourceRange
from yian.nodes.human_input.node import site_label, template_sites


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


def _questions_ref_flow(questions: str) -> str:
    return f"""\
id: ask
name: ask
input:
  order: object
nodes:
  ask:
    kind: human_input
    input:
      order: ${{input.order}}
    questions: '{questions}'
output: ${{ask.output}}
"""


@pytest.mark.parametrize("questions", ["${isinstance(order, Buy)}", "${bogus(order)}"])
def test_human_input_questions_prompt_function_errors_point_at_questions(questions):
    from yian.compose import load_flow

    with pytest.raises(LoadError, match=r"exactly one `\$\{name\}`") as exc:
        load_flow(_questions_ref_flow(questions))
    assert exc.value.diagnostic.frames[-1].range.start_line == 10


def _load_questions(questions_yaml):
    from yian.compose import load_flow

    return load_flow(
        "id: f\n"
        "name: f\n"
        "input: {qs: \"list[object]\"}\n"
        "nodes:\n"
        "  ask:\n"
        "    kind: human_input\n"
        "    input: {qs: \"${input.qs}\"}\n"
        f"    questions: {questions_yaml}\n"
        "output: ${ask.output}\n"
    )


def _questions_line(exc):
    # 1-based line of the `questions:` field in the `_load_questions` flow.
    return exc.value.diagnostic.frames[-1].range.start_line


@pytest.mark.parametrize(
    "questions_yaml",
    ["5", "true", '{question: "Pick one", header: P}'],
)
def test_human_input_build_rejects_questions_that_are_not_a_list_or_ref(questions_yaml):
    # A non-list, non-string `questions:` used to build a plain gate with an empty prompt.
    with pytest.raises(LoadError, match=r"`questions:` must be a list") as exc:
        _load_questions(questions_yaml)
    assert _questions_line(exc) == 8


@pytest.mark.parametrize(
    "questions_yaml",
    [
        '"Answer these carefully: ${qs}"',
        '"${qs} "',
        '"${upper(qs)}"',
        '"${qs or qs}"',
        '"${qs.items}"',
        '"${qs}${qs}"',
        '"qs"',
        '"${qs[0]}"',
    ],
)
def test_human_input_build_rejects_questions_string_that_is_not_one_whole_ref(questions_yaml):
    # Text or an expression around the ref used to be dropped, keeping only the name.
    with pytest.raises(LoadError, match=r"exactly one `\$\{name\}`") as exc:
        _load_questions(questions_yaml)
    assert _questions_line(exc) == 8


@pytest.mark.parametrize("questions_yaml", ['"${qs}"', '"${ qs }"', "${qs}"])
def test_human_input_build_accepts_a_whole_questions_ref(questions_yaml):
    node = _load_questions(questions_yaml).compiled.nodes["ask"]
    assert node.questions_input == "qs" and node.questions is None


_TWO_QUESTIONS_FLOW = """\
nodes:
  ask:
    kind: human_input
    questions:
      - question: First
        header: One
      - question: Second
        header: Two
        options:
          - label: Alpha
            description: First letter
          - label: Beta
            description: Second letter
          - plain
"""


def test_human_input_parse_records_a_range_per_question_string():
    desc = HumanInputDescriptor.parse("ask", _reader_for(_TWO_QUESTIONS_FLOW, "ask"))

    label = desc.field_ranges[("text", "questions[1].options[0].label")]
    assert isinstance(label, SourceRange)
    assert label.start_line == 10
    sites = template_sites(None, desc.questions)
    assert len(sites) == 6
    for path, _ in sites:
        assert isinstance(desc.field_ranges[("text", site_label(path))], SourceRange)
    assert not any(
        key[0] == "text" and key[1].startswith("questions[1].options[2]")
        for key in desc.field_ranges
    )


def test_human_input_parse_records_ranges_in_a_flow_style_list():
    flow = """\
nodes:
  ask:
    kind: human_input
    questions:
      - question: Pick
        options: [{label: A, description: B}]
"""
    desc = HumanInputDescriptor.parse("ask", _reader_for(flow, "ask"))

    for name in ("label", "description"):
        rng = desc.field_ranges[("text", f"questions[0].options[0].{name}")]
        assert rng.start_line == 6


def test_human_input_parse_ranges_an_aliased_or_merged_question_at_its_authored_string():
    flow = """\
x-q: &q
  question: Anchored?
  header: A
nodes:
  ask:
    kind: human_input
    questions:
      - *q
      - &r {question: Earlier, header: B}
      - *r
      - <<: *q
        header: C
"""
    desc = HumanInputDescriptor.parse("ask", _reader_for(flow, "ask"))

    lines = {
        i: desc.field_ranges[("text", f"questions[{i}].question")].start_line
        for i in range(4)
    }
    assert lines == {0: 2, 1: 9, 2: 9, 3: 2}


def _brief_line(flow):
    desc = HumanInputDescriptor.parse("ask", _reader_for(flow, "ask"))
    return desc.field_ranges[("text", "adaptive_questions.prompt")].start_line


def test_human_input_parse_records_the_adaptive_brief_range():
    flow = """\
nodes:
  ask:
    kind: human_input
    adaptive_questions:
      retries: 1
      prompt: Ask about ${x}
"""
    assert _brief_line(flow) == 6


def test_human_input_parse_records_a_flow_style_adaptive_brief_range():
    flow = """\
nodes:
  ask:
    kind: human_input
    adaptive_questions: {prompt: "Q ${x}"}
"""
    assert _brief_line(flow) == 4


def test_human_input_parse_records_a_block_scalar_adaptive_brief_range():
    flow = """\
nodes:
  ask:
    kind: human_input
    adaptive_questions:
      prompt: |
        Ask about ${x}
        and more.
"""
    assert _brief_line(flow) == 5


def test_human_input_parse_ranges_a_merged_adaptive_brief_at_the_anchor():
    flow = """\
x-brief: &b
  prompt: Ask about ${x}
nodes:
  ask:
    kind: human_input
    adaptive_questions:
      <<: *b
      retries: 1
"""
    assert _brief_line(flow) == 2


def test_human_input_parse_without_adaptive_questions_has_no_brief_range():
    desc = HumanInputDescriptor.parse("ask", _reader_for(_FLOW, "ask"))
    assert ("text", "adaptive_questions.prompt") not in desc.field_ranges
