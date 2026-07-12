"""Unit tests for the `case` desugar -> strict `CaseNode`.

A `case` node carries no inputs and no built leaf Node; `desugar_case` lowers it to
a strict `CaseNode` plus control + data edges, with NO new NodeKind:

- **searched** (`cases: [{when: "<bool>", then: <t>}]`, `else: <e>`): each distinct
  ref across the `when:`s -> one `CaseNode` input `__rN`
  (`source=<original ref>`); the `when:` is rewritten with each ref replaced by its
  local `__rN` (preserving the original spelling — bare stays bare); control
  edges `gate->then(source_handle=then)` + `gate->else(source_handle="default")`;
  the data edges carry the `__rN` `input_group` (reconciling the data-edge pass's provisional).
- **`on:`** (`on: ${ref}`, `cases: [{when: <value>, then: <t>}]`): one input
  `__on = ${ref}`; each `when: <value>` -> `${__on} == "<value>"`.
- **exhaustiveness**: when `on:` names an ENUM producer (the dotted ref resolves to a
  `Type` with `.tags`), every tag must be covered by a case OR a present `else:`.

The desugared `CaseNode.when`s evaluate via the EXISTING `evaluate_when_record`.
"""

from pathlib import Path

import pytest

from agent_composer.expr.expressions import evaluate_when_record
from agent_composer.nodes.case import DEFAULT_HANDLE, CaseNode
from agent_composer.typesys.values import ValueKind, Type
from agent_composer.compose.build import build_leaf_node
from agent_composer.compose.edges.data import infer_data_edges
from agent_composer.compose.cases import desugar_case, reconcile_case_edges
from agent_composer.compose.helpers.errors import LoadError
from agent_composer.compose.parse.file import (
    AgentDescriptor,
    CaseDescriptor,
    CodeDescriptor,
    parse_file,
)
from agent_composer.compose.helpers.reader import NodeReader
from agent_composer.compose.parse.nodes import parse_nodes

_SEEDS = Path(__file__).resolve().parents[2] / "tests" / "seeds"


def _case_desc(seed: str, node_id: str) -> CaseDescriptor:
    f = parse_file((_SEEDS / seed).read_text())
    desc = parse_nodes(NodeReader.spanless(f.nodes))[node_id]
    assert isinstance(desc, CaseDescriptor)
    return desc


# The desugared gate's input sources now live on the flow-owned wiring (CaseDesugar.wiring),
# keyed by the __rN/__on param name (the node carries only the param names).


# --------------------------------------------------------------------------- #
# 9a — searched form
# --------------------------------------------------------------------------- #


def test_searched_desugar_builds_case():
    # seed 02: when "score.output >= 0.5" then positive, else cautious.
    desc = _case_desc("02-case.yaml", "gate")
    result = desugar_case(desc, {})
    node = result.node
    assert isinstance(node, CaseNode)
    assert node.id == "gate"
    # one distinct ref -> one param __r0, wired to ${score.output}.
    assert [p.name for p in node.params] == ["__r0"]
    assert result.wiring == {"__r0": "${score.output}"}
    # the case's when is rewritten to the local __r0 (bare, matching the seed's spelling).
    assert len(node.cases) == 1
    assert node.cases[0].when == "__r0 >= 0.5"
    assert node.cases[0].handle == "positive"


def test_searched_control_edges():
    desc = _case_desc("02-case.yaml", "gate")
    result = desugar_case(desc, {})
    ctl = {(e.from_, e.to, e.source_handle) for e in result.control_edges}
    assert ("gate", "positive", "positive") in ctl
    assert ("gate", "cautious", DEFAULT_HANDLE) in ctl


def test_searched_data_edge_reconciled_to_input_group():
    # the data edge score -> gate carries the __r0 input_group (not a <case>:<n>).
    desc = _case_desc("02-case.yaml", "gate")
    result = desugar_case(desc, {})
    data = {(e.from_, e.to, e.input_group) for e in result.data_edges}
    assert ("score", "gate", "__r0") in data
    # no provisional <case>:<n> group survives.
    assert not any(":" in (e.input_group or "") for e in result.data_edges)


def test_searched_multiref_allocates_r0_r1():
    # seed 10: "score.output > 0 and input.weight * 100 <= 5".
    desc = _case_desc("10-asserts-arithmetic.yaml", "size")
    result = desugar_case(desc, {})
    assert result.wiring == {"__r0": "${score.output}", "__r1": "${input.weight}"}
    assert result.node.cases[0].when == "__r0 > 0 and __r1 * 100 <= 5"
    # only the ${X.output} ref is a data edge; ${input.X} is not.
    data = {(e.from_, e.to, e.input_group) for e in result.data_edges}
    assert ("score", "size", "__r0") in data
    assert not any(e.from_ == "weight" for e in result.data_edges)


# A searched `when:` may be written in any of the three condition spellings — the
# desugar extracts refs (_refs_in) and rewrites them (_rewrite_when) through the SAME
# parse, so all three allocate the same __rN, wire the same source, and route the same.
@pytest.mark.parametrize(
    "when, expect_when",
    [
        ("${score.output} >= 0.5", "${__r0} >= 0.5"),   # mixed
        ("score.output >= 0.5", "__r0 >= 0.5"),          # bare
        ("${score.output >= 0.5}", "${__r0 >= 0.5}"),    # whole-span
    ],
)
def test_searched_when_all_spellings_desugar_equivalently(when, expect_when):
    desc = CaseDescriptor(
        id="gate",
        cases=[{"when": when, "then": "positive"}],
        else_="cautious",
    )
    result = desugar_case(desc, {})
    node = result.node
    assert [p.name for p in node.params] == ["__r0"]
    assert result.wiring == {"__r0": "${score.output}"}
    assert node.cases[0].when == expect_when
    # the rewritten when: routes identically against a bound record (all spellings evaluate the same).
    assert evaluate_when_record(node.cases[0].when, {"__r0": 0.7}) is True
    assert evaluate_when_record(node.cases[0].when, {"__r0": 0.3}) is False
    # the data edge reconciles to __r0 regardless of spelling.
    data = {(e.from_, e.to, e.input_group) for e in result.data_edges}
    assert ("score", "gate", "__r0") in data


def test_searched_when_malformed_is_loud():
    desc = CaseDescriptor(
        id="gate",
        cases=[{"when": "score >= >= 5", "then": "positive"}],
        else_="cautious",
    )
    with pytest.raises(LoadError, match="valid boolean expression"):
        desugar_case(desc, {})


# --------------------------------------------------------------------------- #
# 9b — on: form
# --------------------------------------------------------------------------- #


def test_on_desugar_builds_case():
    # seed 06: on ${classify.output}; when pro/con/mixed.
    desc = _case_desc("06-case-on.yaml", "route")
    result = desugar_case(desc, {})
    node = result.node
    assert result.wiring == {"__on": "${classify.output}"}
    whens = [c.when for c in node.cases]
    assert whens == ['${__on} == "pro"', '${__on} == "con"', '${__on} == "mixed"']
    handles = [c.handle for c in node.cases]
    assert handles == ["pro_note", "con_note", "choppy_note"]


def test_on_control_edges():
    desc = _case_desc("06-case-on.yaml", "route")
    result = desugar_case(desc, {})
    ctl = {(e.from_, e.to, e.source_handle) for e in result.control_edges}
    assert ("route", "pro_note", "pro_note") in ctl
    assert ("route", "con_note", "con_note") in ctl
    assert ("route", "choppy_note", "choppy_note") in ctl
    # else: choppy_note -> the default handle (a distinct edge to the same target).
    assert ("route", "choppy_note", DEFAULT_HANDLE) in ctl


def test_on_data_edge_reconciled():
    desc = _case_desc("06-case-on.yaml", "route")
    result = desugar_case(desc, {})
    data = {(e.from_, e.to, e.input_group) for e in result.data_edges}
    assert ("classify", "route", "__on") in data


# --------------------------------------------------------------------------- #
# the desugared when: routes via the EXISTING evaluate_when_record
# --------------------------------------------------------------------------- #


def test_desugared_searched_routes_via_evaluate_when_record():
    desc = _case_desc("02-case.yaml", "gate")
    node = desugar_case(desc, {}).node
    # bound input record uses the bare local names.
    assert evaluate_when_record(node.cases[0].when, {"__r0": 0.7}) is True
    assert evaluate_when_record(node.cases[0].when, {"__r0": 0.3}) is False


def test_desugared_on_routes_via_evaluate_when_record():
    desc = _case_desc("06-case-on.yaml", "route")
    node = desugar_case(desc, {}).node
    assert evaluate_when_record(node.cases[0].when, {"__on": "pro"}) is True
    assert evaluate_when_record(node.cases[0].when, {"__on": "con"}) is False
    assert evaluate_when_record(node.cases[1].when, {"__on": "con"}) is True


# --------------------------------------------------------------------------- #
# 9c — exhaustiveness over an enum producer
# --------------------------------------------------------------------------- #


def _enum_case(tags: list[str], cases: list[str], else_: str | None):
    """A constructed `case ... on ${cls.output}` over a Literal-enum producer.

    `producers` maps `cls` -> a record? no: cls produces the enum value directly
    (its output_type carries `.tags`). The on: ref is `${cls.output}`.
    """
    desc = CaseDescriptor(
        id="route",
        on="${cls.output}",
        cases=[{"when": t, "then": f"{t}_note"} for t in cases],
        else_=else_,
    )
    producers = {"cls": Type(kind=ValueKind.STRING, tags=frozenset(tags))}
    return desc, producers


def test_exhaustiveness_missing_tag_no_else_raises():
    desc, producers = _enum_case(
        tags=["pro", "con", "mixed"], cases=["pro", "con"], else_=None
    )
    with pytest.raises(LoadError) as exc:
        desugar_case(desc, producers)
    assert "mixed" in str(exc.value)


def test_exhaustiveness_else_satisfies_coverage():
    # 2 of 3 tags + an else: -> NOT flagged (seed-18 style).
    desc, producers = _enum_case(
        tags=["pro", "con", "mixed"], cases=["pro", "con"], else_="neutral_note"
    )
    result = desugar_case(desc, producers)  # no raise
    assert isinstance(result.node, CaseNode)


def test_exhaustiveness_all_tags_covered_no_else_ok():
    desc, producers = _enum_case(
        tags=["pro", "con"], cases=["pro", "con"], else_=None
    )
    result = desugar_case(desc, producers)  # no raise
    assert isinstance(result.node, CaseNode)


def test_exhaustiveness_walks_dotted_field_into_record():
    # seed 18 shape: on ${synth.output.stance}; synth.output_type is a View record
    # whose `stance` field carries the Stance enum tags.
    view = Type(
        kind=ValueKind.OBJECT,
        fields={
            "stance": Type(
                kind=ValueKind.STRING,
                tags=frozenset({"positive", "negative", "neutral"}),
            ),
            "claim": Type.scalar(ValueKind.STRING),
        },
        required=frozenset({"stance", "claim"}),
    )
    desc = CaseDescriptor(
        id="route",
        on="${synth.output.stance}",
        cases=[{"when": "positive", "then": "pro_note"}, {"when": "negative", "then": "con_note"}],
        else_="neutral_note",
    )
    result = desugar_case(desc, {"synth": view})  # else: covers `neutral` -> ok
    assert isinstance(result.node, CaseNode)


def test_exhaustiveness_dotted_missing_tag_raises():
    view = Type(
        kind=ValueKind.OBJECT,
        fields={
            "stance": Type(
                kind=ValueKind.STRING,
                tags=frozenset({"positive", "negative", "neutral"}),
            ),
        },
        required=frozenset({"stance"}),
    )
    desc = CaseDescriptor(
        id="route",
        on="${synth.output.stance}",
        cases=[{"when": "positive", "then": "pro_note"}, {"when": "negative", "then": "con_note"}],
        else_=None,
    )
    with pytest.raises(LoadError) as exc:
        desugar_case(desc, {"synth": view})
    assert "neutral" in str(exc.value)


def test_no_exhaustiveness_for_searched_form():
    # a searched case (no on:) over an enum-ish producer is NOT exhaustiveness-checked.
    desc = _case_desc("02-case.yaml", "gate")
    desugar_case(desc, {"score": Type(kind=ValueKind.STRING, tags=frozenset({"a"}))})


def test_on_non_enum_producer_skips_exhaustiveness():
    # on: a plain scalar producer (no .tags) -> lenient, no exhaustiveness, no raise.
    desc = _case_desc("06-case-on.yaml", "route")
    result = desugar_case(desc, {"classify": Type.scalar(ValueKind.STRING)})
    assert isinstance(result.node, CaseNode)


# --------------------------------------------------------------------------- #
# the desugared CASE passes handle alignment (every case has an edge)
# --------------------------------------------------------------------------- #


def test_handles_align_with_control_edges():
    desc = _case_desc("06-case-on.yaml", "route")
    result = desugar_case(desc, {})
    case_handles = {c.handle for c in result.node.cases}
    edge_handles = {e.source_handle for e in result.control_edges}
    # every case handle has an outgoing edge.
    assert case_handles <= edge_handles
    # every edge handle is a case handle or the default.
    assert edge_handles <= case_handles | {DEFAULT_HANDLE}


# --------------------------------------------------------------------------- #
# reconciliation: drop the provisional data-edge-pass case edges, splice in the desugar's
# --------------------------------------------------------------------------- #


def test_reconcile_replaces_provisional_case_edges():
    # seed 02: the data-edge pass emits a provisional score->gate (group "gate:0"); reconciliation
    # drops it and splices the desugar's __r0 data edge + control edges.
    f = parse_file((_SEEDS / "02-case.yaml").read_text())
    descriptors = parse_nodes(NodeReader.spanless(f.nodes))
    flow_wiring = {
        nid: build_leaf_node(d, {})[1]
        for nid, d in descriptors.items()
        if isinstance(d, (AgentDescriptor, CodeDescriptor))
    }
    data_edges = infer_data_edges(descriptors, flow_wiring)
    # the data-edge pass keys the case edge by a provisional <case>:<n> group.
    assert any(e.to == "gate" and ":" in (e.input_group or "") for e in data_edges)

    desugars = {"gate": desugar_case(descriptors["gate"], {})}
    merged = reconcile_case_edges(data_edges, desugars)
    # the provisional case edge is gone; the reconciled __r0 data edge is present.
    assert not any(e.to == "gate" and ":" in (e.input_group or "") for e in merged)
    assert ("score", "gate", "__r0") in {(e.from_, e.to, e.input_group) for e in merged}
    # the control edges are spliced in.
    ctl = {(e.from_, e.to, e.source_handle) for e in merged}
    assert ("gate", "positive", "positive") in ctl
    assert ("gate", "cautious", DEFAULT_HANDLE) in ctl


def test_on_value_with_double_quote_uses_single_quote_and_routes():
    # an on: match value containing " must compile to a parseable when: (single-quoted,
    # since the grammar has no escape support) and route correctly at eval time.
    from agent_composer.compose.cases import _quote

    expr = '${__on} == ' + _quote('a"b')
    assert evaluate_when_record(expr, {"__on": 'a"b'}) is True
    assert evaluate_when_record(expr, {"__on": "other"}) is False


def test_on_value_with_both_quotes_is_loud_load_error():
    # a value containing BOTH ' and " is unrepresentable in the escape-less grammar
    # -> a loud load error, not a route-time crash.
    from agent_composer.compose.cases import _quote

    with pytest.raises(LoadError):
        _quote("a'b\"c")
