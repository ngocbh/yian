"""Tests for the desugar phase: flow-level rewrites that run before the build loop.

Each desugar pass rewrites a flow's parsed descriptors into ordinary descriptors +
bindings. A synth node a pass mints locates at the SITE it desugared from — the
`call(...)` directive's own binding line, not the host node header — so its downstream
errors caret the author's actual source. The exception is `adaptive_questions`, whose
synth composer genuinely SHARES the gate's inputs verbatim and so correctly inherits the
gate's `.range`/`field_ranges`.
"""

from dataclasses import replace

import pytest

from yian.compose.desugar.adaptive_questions import desugar_adaptive_questions
from yian.compose.desugar.case_outputs import expand_case_outputs
from yian.compose.desugar.case_targets import desugar_case_call_targets
from yian.compose.desugar.case_values import desugar_case_values
from yian.compose.desugar.inline_calls import (
    _synth_id_gen,
    desugar_inline_calls,
)
from yian.compose.helpers.errors import LoadDiagnostics, LoadError
from yian.compose.parse.file import (
    CaseDescriptor,
    CodeDescriptor,
    MapDescriptor,
)
from yian.compose.parse.nodes import parse_nodes
from yian.compose.source import SourceDocument
from yian.diagnostics import OriginRef
from yian.diagnostics import SourceRange


def _nodes_reader(text):
    """Return the physical cursor over a flow document's ``nodes:`` mapping."""

    return SourceDocument.parse(text).root.child("nodes")


def _diagnostics() -> LoadDiagnostics:
    """Return source ownership for direct range-only desugar tests."""

    return LoadDiagnostics(label="flow.yaml", text=(" " * 100 + "\n") * 200)


def _only_frame(error: LoadError):
    assert len(error.diagnostic.frames) == 1
    return error.diagnostic.frames[0]


def test_desugar_inline_calls_locate_at_their_binding_site():
    # A whole-value `call(...)` in a node's `input:` binding lifts to a synth `call` node.
    # The synth CallDescriptor locates at its OWN binding line (the `x:` entry the directive
    # was written on), NOT the host node header — and carries an empty `field_ranges` (it owns
    # no authored fields), so a bad ref inside the directive carets the directive.
    text = (
        "id: f\nname: f\nnodes:\n"
        "  writer:\n    kind: code\n    code: m:f\n"
        "    input:\n      x: call(helper, n=3)\n    output: str\n"
        "output: ${writer.output}\n"
    )
    nodes_cursor = _nodes_reader(text)
    descriptors = parse_nodes(nodes_cursor)
    host = descriptors["writer"]
    new_descriptors, _outputs, _asserts = desugar_inline_calls(
        descriptors, None, next_id=_synth_id_gen()
    )
    synth = [d for nid, d in new_descriptors.items() if nid.startswith("__call_")]
    assert len(synth) == 1
    call_desc = synth[0]
    assert call_desc.call == "helper"
    assert (
        call_desc.range == host.field_ranges[("input", "x")]
    )  # the `x:` binding, not the header
    assert call_desc.range != host.range  # ... which is strictly the header
    assert call_desc.field_ranges == {}  # owns no authored fields


def test_inline_call_error_uses_binding_range():
    text = (
        "id: f\nname: f\nnodes:\n"
        "  writer:\n    kind: code\n    code: m:f\n"
        "    input:\n      x: ${helper(topic=input.topic)}\n    output: str\n"
    )
    nodes_cursor = _nodes_reader(text)
    descriptors = parse_nodes(nodes_cursor)
    binding_range = descriptors["writer"].field_ranges[("input", "x")]

    with pytest.raises(LoadError) as caught:
        desugar_inline_calls(
            descriptors,
            None,
            next_id=_synth_id_gen(),
            diagnostics=LoadDiagnostics.from_cursor(nodes_cursor),
        )

    frame = _only_frame(caught.value)
    assert frame.range == binding_range
    assert frame.text == text


def test_inline_output_error_uses_section_range():
    output_range = SourceRange(17, 9, 17, 28)

    with pytest.raises(LoadError) as caught:
        desugar_inline_calls(
            {},
            "call(helper, topic=${item})",
            outputs_range=output_range,
            next_id=_synth_id_gen(),
            diagnostics=_diagnostics(),
        )

    assert _only_frame(caught.value).range == output_range


def test_inline_assert_error_uses_positional_range():
    assertion_ranges = [
        SourceRange(18, 7, 18, 11),
        SourceRange(19, 7, 19, 25),
    ]

    with pytest.raises(LoadError) as caught:
        desugar_inline_calls(
            {},
            None,
            asserts_section=["true", "${helper(topic=1)}"],
            assertion_range_at=assertion_ranges.__getitem__,
            next_id=_synth_id_gen(),
            diagnostics=_diagnostics(),
        )

    assert _only_frame(caught.value).range == assertion_ranges[1]


def test_duplicate_inline_assertions_keep_positional_ranges():
    assertion_ranges = [
        SourceRange(21, 7, 21, 19),
        SourceRange(22, 7, 22, 19),
    ]

    descriptors, _outputs, _asserts = desugar_inline_calls(
        {},
        None,
        asserts_section=["call(helper)", "call(helper)"],
        assertion_range_at=assertion_ranges.__getitem__,
        next_id=_synth_id_gen(),
    )

    assert descriptors["__call_0"].range == assertion_ranges[0]
    assert descriptors["__call_1"].range == assertion_ranges[1]


def test_inline_map_scalar_over_does_not_use_input_named_over_range():
    input_range = SourceRange(71, 9, 71, 25)
    scalar_range = SourceRange(72, 9, 72, 36)
    descriptor = MapDescriptor(
        id="mapped",
        call="child",
        inputs={"over": "${input.seed}"},
        over="${helper(topic=item)}",
        field_ranges={
            ("input", "over"): input_range,
            ("field", "over"): scalar_range,
        },
    )

    with pytest.raises(LoadError) as caught:
        desugar_inline_calls(
            {"mapped": descriptor},
            None,
            diagnostics=_diagnostics(),
        )

    assert _only_frame(caught.value).range == scalar_range


def test_desugar_adaptive_questions_preserves_ranges():
    # An adaptive human_input gate lifts a synth composer agent (`<gate>__compose`).
    # The synth agent must inherit the gate node's range/field_ranges.
    text = (
        "id: f\nname: f\nnodes:\n"
        "  ask:\n    kind: human_input\n"
        "    adaptive_questions:\n      prompt: Ask about ${input.topic}\n"
        "    input:\n      topic: ${input.topic}\n"
        "output: ${ask.output}\n"
    )
    descriptors = parse_nodes(_nodes_reader(text))
    gate = descriptors["ask"]
    new_descriptors = desugar_adaptive_questions(descriptors)
    composer = new_descriptors["ask__compose"]
    assert composer.range == gate.range
    assert composer.field_ranges == gate.field_ranges


def test_desugar_case_call_targets_locate_at_their_target_line():
    # A `case` then:/else: that is a whole-value `call(...)` directive lifts a synth `call`
    # node. That synth node locates at its OWN `then:` line (captured positionally in the case
    # descriptor's `then_ranges`), NOT the host case header, and carries an empty `field_ranges`.
    text = (
        "id: f\nname: f\nnodes:\n"
        "  route:\n    kind: case\n    on: ${score.output}\n"
        "    cases:\n      - when: high\n        then: call(helper, n=3)\n"  # `then:` on line 9
        "    else: cold\n"
        "output: ${route.output}\n"
    )
    descriptors = parse_nodes(_nodes_reader(text))
    predicate_origin = OriginRef("case/0/when")
    host = replace(descriptors["route"], when_origins=(predicate_origin,))
    descriptors["route"] = host
    new_descriptors = desugar_case_call_targets(descriptors, _synth_id_gen())
    synth = [d for nid, d in new_descriptors.items() if nid.startswith("__call_")]
    assert len(synth) == 1
    call_desc = synth[0]
    assert call_desc.call == "helper"
    assert (
        call_desc.range == host.then_ranges[0]
    )  # the `then:` line, not the case header
    assert call_desc.range != host.range
    assert call_desc.field_ranges == {}  # owns no authored fields
    assert new_descriptors["route"].when_origins == (predicate_origin,)


def _case_gate() -> CaseDescriptor:
    return CaseDescriptor(
        id="gate",
        cases=[{"when": "yes", "then": "left"}],
        else_="right",
    )


def test_case_binding_error_uses_binding_range():
    host_range = SourceRange(30, 3, 35, 1)
    binding_range = SourceRange(33, 10, 33, 44)
    consumer = CodeDescriptor(
        id="consumer",
        code="m:f",
        inputs={"value": "${gate.output | fallback.output}"},
        range=host_range,
        field_ranges={("input", "value"): binding_range},
    )

    with pytest.raises(LoadError) as caught:
        expand_case_outputs(
            {"gate": _case_gate(), "consumer": consumer},
            None,
            [],
            diagnostics=_diagnostics(),
        )

    assert _only_frame(caught.value).range == binding_range


def test_case_output_error_uses_section_range():
    output_range = SourceRange(40, 9, 40, 23)
    empty_gate = CaseDescriptor(id="gate", cases=[{"when": "yes"}])

    with pytest.raises(LoadError) as caught:
        expand_case_outputs(
            {"gate": empty_gate},
            "${gate.output}",
            [],
            outputs_range=output_range,
            diagnostics=_diagnostics(),
        )

    assert _only_frame(caught.value).range == output_range


@pytest.mark.parametrize("condition_kind", ["on", "when"])
def test_case_condition_error_uses_exact_condition_range(condition_kind):
    condition_range = SourceRange(51, 11, 51, 25)
    host_range = SourceRange(48, 3, 55, 1)
    if condition_kind == "on":
        other = CaseDescriptor(
            id="other",
            on="${gate.output}",
            cases=[{"when": "yes", "then": "done"}],
            range=host_range,
            field_ranges={("field", "on"): condition_range},
        )
    else:
        other = CaseDescriptor(
            id="other",
            cases=[{"when": "${gate.output}", "then": "done"}],
            range=host_range,
            when_ranges=[condition_range],
        )

    with pytest.raises(LoadError) as caught:
        expand_case_outputs(
            {"gate": _case_gate(), "other": other},
            None,
            [],
            diagnostics=_diagnostics(),
        )

    assert _only_frame(caught.value).range == condition_range


def test_case_assert_error_uses_positional_range():
    assertion_ranges = [
        SourceRange(61, 7, 61, 20),
        SourceRange(62, 7, 62, 30),
    ]

    with pytest.raises(LoadError) as caught:
        expand_case_outputs(
            {"gate": _case_gate()},
            None,
            ["true", "gate.output == 1"],
            assertion_range_at=assertion_ranges.__getitem__,
            diagnostics=_diagnostics(),
        )

    assert _only_frame(caught.value).range == assertion_ranges[1]


def test_case_map_scalar_over_does_not_use_input_named_over_range():
    input_range = SourceRange(81, 9, 81, 25)
    scalar_range = SourceRange(82, 9, 82, 45)
    descriptor = MapDescriptor(
        id="mapped",
        call="child",
        inputs={"over": "${input.seed}"},
        over="${gate.output | fallback.output}",
        field_ranges={
            ("input", "over"): input_range,
            ("field", "over"): scalar_range,
        },
    )

    with pytest.raises(LoadError) as caught:
        expand_case_outputs(
            {"gate": _case_gate(), "mapped": descriptor},
            None,
            [],
            diagnostics=_diagnostics(),
        )

    assert _only_frame(caught.value).range == scalar_range


def test_desugar_case_values_lifts_a_record_value_to_a_code_node():
    then_range = SourceRange(12, 15, 12, 40)
    gate = CaseDescriptor(
        id="gate",
        cases=[{"when": "s.output > 1", "then": {"a": 1, "b": "${x.output}"}}],
        outputs="object",
        then_ranges=[then_range],
    )
    descriptors = {
        "gate": gate,
        "s": CodeDescriptor(id="s", code="m:f"),
        "x": CodeDescriptor(id="x", code="m:f"),
    }

    new = desugar_case_values(descriptors, diagnostics=_diagnostics())

    synth = new["__value_0"]
    assert isinstance(synth, CodeDescriptor)
    assert synth.inputs == {"v0": "${x.output}"}
    assert synth.code == "return {'a': 1, 'b': inputs['v0']}"
    assert synth.node_name == "gate then[0]"
    assert synth.range == then_range
    assert synth.field_ranges == {("input", "v0"): then_range}
    assert new["gate"].cases == [{"when": "s.output > 1", "then": "__value_0"}]
    assert new["gate"].value_branches == {"__value_0": {"a": 1, "b": "${x.output}"}}
    assert new["s"] is descriptors["s"] and new["x"] is descriptors["x"]


def test_desugar_case_values_writes_scalars_as_python_literals():
    import datetime

    gate = CaseDescriptor(
        id="gate",
        cases=[
            {"when": "s.output > 1", "then": -1.5},
            {"when": "s.output > 2", "then": datetime.date(2026, 1, 2)},
            {"when": "s.output > 3", "then": ["costs $$5", None, True]},
        ],
        else_=0,
        outputs="object",
    )
    new = desugar_case_values(
        {"gate": gate, "s": CodeDescriptor(id="s", code="m:f")},
        diagnostics=_diagnostics(),
    )

    assert new["__value_0"].code == "return -1.5"
    assert new["__value_1"].code == "return '2026-01-02'"
    assert new["__value_2"].code == "return ['costs $5', None, True]"
    assert new["__value_3"].code == "return 0"
    assert new["__value_3"].node_name == "gate else"
    assert all(new[f"__value_{n}"].inputs == {} for n in range(4))
    assert new["gate"].else_ == "__value_3"


@pytest.mark.parametrize("number", [float("inf"), float("-inf"), float("nan")])
def test_desugar_case_values_rejects_non_finite_numbers(number):
    gate = CaseDescriptor(
        id="gate",
        cases=[{"when": "s.output > 1", "then": {"a": [number]}}],
        outputs="object",
    )
    with pytest.raises(LoadError, match="case node 'gate' then: numbers must be finite"):
        desugar_case_values(
            {"gate": gate, "s": CodeDescriptor(id="s", code="m:f")},
            diagnostics=_diagnostics(),
        )
