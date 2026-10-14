"""Unit tests for the `case` build -> strict `CaseNode`.

A `case` node carries no inputs and no built leaf Node; its own `.build(ctx)` lowers it to
a strict `CaseNode` plus control + data edges (its `BuildResult.edges`), with NO new NodeKind:

- **searched** (`cases: [{when: "<bool>", then: <t>}]`, `else: <e>`): each distinct
  ref across the `when:`s -> one `CaseNode` input `__rN`
  (`source=<original ref>`); the `when:` is rewritten with each ref replaced by its
  local `__rN` (preserving the original spelling — bare stays bare); control
  edges `gate->then(source_handle=then)` + `gate->else(source_handle=ELSE_HANDLE)`;
  the data edges carry the `__rN` `input_group`.
- **`on:`** (`on: ${ref}`, `cases: [{when: <value>, then: <t>}]`): one input
  `__on = ${ref}`; each `when: <value>` -> a typed `__on` match (`== "<text>"`, or a
  number/bool literal with an `isinstance(__on, bool)` guard).
- **exhaustiveness**: a SEPARATE validate-phase check (`check_case_exhaustive`), NOT part of
  `.build()` — when `on:` names an ENUM producer (the dotted ref resolves to a `Type` with
  `.tags`), every tag must be covered by a case OR a present `else:`.

`BuildResult.edges` is a FLAT tuple — data edges (an `input_group`, no `source_handle`)
FOLLOWED BY control edges (a `source_handle`, no `input_group`); `_split` recovers the two
by that attribute invariant. The built `CaseNode.when`s evaluate via the EXISTING
`evaluate_when_record`.
"""

from pathlib import Path

import pytest

from yian.expr.expressions import evaluate_when_record
from yian.nodes.case import ELSE_HANDLE, CaseNode
from yian.nodes.contracts import PartialRecordContract, ValueContract
from yian.typesys.values import ValueKind, Type
from yian.compose.nodes.base import BuildContext
from yian.compose.helpers.errors import LoadDiagnostics, LoadError
from yian.compose.parse.file import CaseDescriptor, parse_file
from yian.compose.parse.nodes import parse_nodes
from yian.compose.source import SourceDocument
from yian.compose.validate.case_exhaustive import check_case_exhaustive
from yian.diagnostics import SourceRange

_SEEDS = Path(__file__).resolve().parents[2] / "tests" / "seeds"


def _case_desc(seed: str, node_id: str) -> CaseDescriptor:
    path = _SEEDS / seed
    document = SourceDocument.parse(path.read_text(), label=str(path))
    parse_file(document)
    desc = parse_nodes(document.root.child("nodes"))[node_id]
    assert isinstance(desc, CaseDescriptor)
    return desc


def _split(result):
    """Recover the (data_edges, control_edges) split from a case `BuildResult`.

    The flat `BuildResult.edges` tuple is data-then-control; the edge-field invariant is the
    discriminator — a DATA edge carries an `input_group` (and no `source_handle`), a CONTROL
    edge carries a `source_handle` (and no `input_group`). Returns `(data, control)` lists."""
    data = [e for e in result.edges if e.source_handle is None]
    control = [e for e in result.edges if e.source_handle is not None]
    return data, control


def _build_case(desc: CaseDescriptor, producers=None):
    """Build a case descriptor -> its `BuildResult` (node + wiring + flat edges)."""
    return desc.build(BuildContext(producers if producers is not None else {}))


# The built gate's input sources live on the flow-owned wiring (BuildResult.wiring),
# keyed by the __rN/__on param name (the node carries only the param names).


# --------------------------------------------------------------------------- #
# Searched form
# --------------------------------------------------------------------------- #


def test_searched_desugar_builds_case():
    # seed 02: when "score.output >= 0.5" then positive, else cautious.
    desc = _case_desc("02-case.yaml", "gate")
    result = _build_case(desc)
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
    _, control_edges = _split(_build_case(desc))
    ctl = {(e.from_, e.to, e.source_handle) for e in control_edges}
    assert ("gate", "positive", "positive") in ctl
    assert ("gate", "cautious", ELSE_HANDLE) in ctl


def test_searched_data_edge_reconciled_to_input_group():
    # the data edge score -> gate carries the __r0 input_group (not a <case>:<n>).
    desc = _case_desc("02-case.yaml", "gate")
    data_edges, _ = _split(_build_case(desc))
    data = {(e.from_, e.to, e.input_group) for e in data_edges}
    assert ("score", "gate", "__r0") in data
    # no provisional <case>:<n> group survives.
    assert not any(":" in (e.input_group or "") for e in data_edges)


def test_searched_multiref_allocates_r0_r1():
    # seed 10: "score.output > 0 and input.weight * 100 <= 5".
    desc = _case_desc("10-asserts-arithmetic.yaml", "size")
    result = _build_case(desc)
    data_edges, _ = _split(result)
    assert result.wiring == {"__r0": "${score.output}", "__r1": "${input.weight}"}
    assert result.node.cases[0].when == "__r0 > 0 and __r1 * 100 <= 5"
    # only the ${X.output} ref is a data edge; ${input.X} is not.
    data = {(e.from_, e.to, e.input_group) for e in data_edges}
    assert ("score", "size", "__r0") in data
    assert not any(e.from_ == "weight" for e in data_edges)


# A searched `when:` may be written bare or as one whole `${...}` — the build extracts
# refs (_searched_params) and rewrites them (_rewrite_when) through the SAME parse, so
# both allocate the same __rN, wire the same source, and route the same.
@pytest.mark.parametrize(
    "when, expect_when",
    [
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
    result = _build_case(desc)
    node = result.node
    assert [p.name for p in node.params] == ["__r0"]
    assert result.wiring == {"__r0": "${score.output}"}
    assert node.cases[0].when == expect_when
    # the rewritten when: routes identically against a bound record (all spellings evaluate the same).
    assert evaluate_when_record(node.cases[0].when, {"__r0": 0.7}) is True
    assert evaluate_when_record(node.cases[0].when, {"__r0": 0.3}) is False
    # the data edge reconciles to __r0 regardless of spelling.
    data_edges, _ = _split(result)
    data = {(e.from_, e.to, e.input_group) for e in data_edges}
    assert ("score", "gate", "__r0") in data


def _searched(*whens: str):
    """Build a searched case `gate` with one case per `when:` -> its `BuildResult`."""
    desc = CaseDescriptor(
        id="gate",
        cases=[{"when": when, "then": f"t{i}"} for i, when in enumerate(whens)],
        else_="cautious",
    )
    return _build_case(desc)


def _edge_stances(result) -> list[tuple[str, str, bool]]:
    """Each data edge of a case build as `(from, input_group, optional)`, in order."""
    data_edges, _ = _split(result)
    return [(e.from_, e.input_group, e.optional) for e in data_edges]


@pytest.mark.parametrize(
    "when, expect_when",
    [
        ("(score.output :- 0) >= 0.5", "(__r0 :- 0) >= 0.5"),
        ("${(score.output :- 0) >= 0.5}", "${(__r0 :- 0) >= 0.5}"),
    ],
)
def test_searched_when_default_reads_through_an_optional_edge(when, expect_when):
    result = _searched(when)
    assert result.wiring == {"__r0": "${score.output :- null}"}
    assert result.node.cases[0].when == expect_when
    assert _edge_stances(result) == [("score", "__r0", True)]
    # the rewrite evaluates as the authored condition: a missing score takes the default.
    assert evaluate_when_record(result.node.cases[0].when, {"__r0": None}) is False
    assert evaluate_when_record(result.node.cases[0].when, {"__r0": 0.7}) is True


def test_searched_when_coalesce_adds_a_required_join_input():
    result = _searched("(score.output | other.output) >= 0.5")
    assert result.wiring == {
        "__r0": "${score.output :- null}",
        "__r1": "${other.output :- null}",
        "__r2": "${score.output | other.output}",
    }
    assert [p.name for p in result.node.params] == ["__r0", "__r1", "__r2"]
    assert result.node.cases[0].when == "(__r0 | __r1) >= 0.5"
    assert _edge_stances(result) == [
        ("score", "__r0", True),
        ("other", "__r1", True),
        ("score", "__r2", False),
        ("other", "__r2", False),
    ]


def test_searched_when_join_reuses_a_plain_read():
    result = _searched("score.output > 5 or (0 :- score.output) < 1")
    assert result.wiring == {"__r0": "${score.output}", "__r1": "${score.output :- null}"}
    assert result.node.cases[0].when == "__r0 > 5 or (0 :- __r1) < 1"
    assert _edge_stances(result) == [("score", "__r0", False), ("score", "__r1", True)]


def test_searched_when_same_escape_in_two_cases_shares_locals():
    result = _searched("(score.output | other.output) > 1", "(score.output | other.output) < 0")
    assert list(result.wiring) == ["__r0", "__r1", "__r2"]
    assert [c.when for c in result.node.cases] == ["(__r0 | __r1) > 1", "(__r0 | __r1) < 0"]


def test_searched_when_malformed_is_loud():
    desc = CaseDescriptor(
        id="gate",
        cases=[{"when": "score >= >= 5", "then": "positive"}],
        else_="cautious",
    )
    with pytest.raises(LoadError, match="valid boolean expression"):
        _build_case(desc)


# --------------------------------------------------------------------------- #
# Discriminant (`on:`) form
# --------------------------------------------------------------------------- #


def test_on_desugar_builds_case():
    # seed 06: on ${classify.output}; when pro/con/mixed.
    desc = _case_desc("06-case-on.yaml", "route")
    result = _build_case(desc)
    node = result.node
    assert result.wiring == {"__on": "${classify.output}"}
    whens = [c.when for c in node.cases]
    assert whens == ['__on == "pro"', '__on == "con"', '__on == "mixed"']
    handles = [c.handle for c in node.cases]
    assert handles == ["pro_note", "con_note", "choppy_note"]


def test_on_control_edges():
    desc = _case_desc("06-case-on.yaml", "route")
    _, control_edges = _split(_build_case(desc))
    ctl = {(e.from_, e.to, e.source_handle) for e in control_edges}
    assert ("route", "pro_note", "pro_note") in ctl
    assert ("route", "con_note", "con_note") in ctl
    assert ("route", "choppy_note", "choppy_note") in ctl
    # else: choppy_note -> the else handle (a distinct edge to the same target).
    assert ("route", "choppy_note", ELSE_HANDLE) in ctl


def test_on_data_edge_reconciled():
    desc = _case_desc("06-case-on.yaml", "route")
    data_edges, _ = _split(_build_case(desc))
    data = {(e.from_, e.to, e.input_group) for e in data_edges}
    assert ("classify", "route", "__on") in data


# --------------------------------------------------------------------------- #
# the built when: routes via the EXISTING evaluate_when_record
# --------------------------------------------------------------------------- #


def test_desugared_searched_routes_via_evaluate_when_record():
    desc = _case_desc("02-case.yaml", "gate")
    node = _build_case(desc).node
    # bound input record uses the bare local names.
    assert evaluate_when_record(node.cases[0].when, {"__r0": 0.7}) is True
    assert evaluate_when_record(node.cases[0].when, {"__r0": 0.3}) is False


def test_desugared_on_routes_via_evaluate_when_record():
    desc = _case_desc("06-case-on.yaml", "route")
    node = _build_case(desc).node
    assert evaluate_when_record(node.cases[0].when, {"__on": "pro"}) is True
    assert evaluate_when_record(node.cases[0].when, {"__on": "con"}) is False
    assert evaluate_when_record(node.cases[1].when, {"__on": "con"}) is True


# --------------------------------------------------------------------------- #
# Exhaustiveness over an enum producer (the separate validate-phase check)
# --------------------------------------------------------------------------- #


def _enum_case(tags: list[str], cases: list[str], else_: str | None):
    """A constructed `case ... on ${cls.output}` over a Literal-enum producer.

    `producers` maps `cls` -> a record? no: cls produces the enum value directly
    (its output contract's type carries `.tags`). The on: ref is `${cls.output}`.
    """
    desc = CaseDescriptor(
        id="route",
        on="${cls.output}",
        cases=[{"when": t, "then": f"{t}_note"} for t in cases],
        else_=else_,
    )
    producers = {
        "cls": ValueContract(Type(kind=ValueKind.STRING, tags=frozenset(tags)))
    }
    return desc, producers


def test_exhaustiveness_missing_tag_no_else_raises():
    desc, producers = _enum_case(
        tags=["pro", "con", "mixed"], cases=["pro", "con"], else_=None
    )
    _build_case(desc, producers)  # build lowers unconditionally — no raise
    with pytest.raises(LoadError) as exc:
        check_case_exhaustive({desc.id: desc}, producers)
    assert "mixed" in str(exc.value)


def test_exhaustiveness_else_satisfies_coverage():
    # 2 of 3 tags + an else: -> NOT flagged (seed-18 style).
    desc, producers = _enum_case(
        tags=["pro", "con", "mixed"], cases=["pro", "con"], else_="neutral_note"
    )
    result = _build_case(desc, producers)  # no raise
    assert isinstance(result.node, CaseNode)
    check_case_exhaustive({desc.id: desc}, producers)  # no raise


def test_exhaustiveness_all_tags_covered_no_else_ok():
    desc, producers = _enum_case(
        tags=["pro", "con"], cases=["pro", "con"], else_=None
    )
    result = _build_case(desc, producers)  # no raise
    assert isinstance(result.node, CaseNode)
    check_case_exhaustive({desc.id: desc}, producers)  # no raise


def test_exhaustiveness_walks_dotted_field_into_record():
    # seed 18 shape: on ${synth.output.stance}; synth's contract is a View record
    # whose `stance` field type carries the Stance enum tags.
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
    check_case_exhaustive(
        {desc.id: desc}, {"synth": ValueContract(view)}
    )  # else: covers `neutral` -> ok


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
    descriptor_range = SourceRange(17, 3, 17, 8)
    desc = CaseDescriptor(
        id="route",
        on="${synth.output.stance}",
        cases=[{"when": "positive", "then": "pro_note"}, {"when": "negative", "then": "con_note"}],
        else_=None,
        range=descriptor_range,
    )
    with pytest.raises(LoadError) as exc:
        check_case_exhaustive(
            {desc.id: desc},
            {"synth": ValueContract(view)},
            diagnostics=LoadDiagnostics(
                label="flow.yaml",
                text=("x" * 80 + "\n") * 30,
            ),
        )
    assert "neutral" in str(exc.value)
    assert exc.value.diagnostic.frames[0].range == descriptor_range


def test_exhaustiveness_projects_known_field_from_partial_record():
    """An opaque sibling does not hide a proven enum field from coverage checks."""
    stance = ValueContract(
        Type(
            kind=ValueKind.STRING,
            tags=frozenset({"positive", "negative", "neutral"}),
        )
    )
    desc = CaseDescriptor(
        id="route",
        on="${synth.output.stance}",
        cases=[
            {"when": "positive", "then": "pro_note"},
            {"when": "negative", "then": "con_note"},
        ],
        else_=None,
    )

    with pytest.raises(LoadError, match="neutral"):
        check_case_exhaustive(
            {desc.id: desc},
            {
                "synth": PartialRecordContract(
                    fields=(("stance", stance), ("note", None))
                )
            },
        )


def test_exhaustiveness_rejects_noncanonical_producer_contract():
    desc, _ = _enum_case(tags=["pro", "con"], cases=["pro"], else_=None)
    with pytest.raises(TypeError, match="ValueContract or PartialRecordContract"):
        check_case_exhaustive(
            {desc.id: desc},
            {"cls": Type(kind=ValueKind.STRING, tags=frozenset({"pro", "con"}))},
        )


def test_exhaustiveness_rejects_noncanonical_unused_producer_contract():
    """The CASE validator enforces its carrier map before selecting a case form."""
    desc = _case_desc("02-case.yaml", "gate")
    with pytest.raises(TypeError, match="ValueContract or PartialRecordContract"):
        check_case_exhaustive(
            {desc.id: desc},
            {"unused": Type.scalar(ValueKind.STRING)},  # type: ignore[dict-item]
        )


def test_no_exhaustiveness_for_searched_form():
    # a searched case (no on:) over an enum-ish producer is NOT exhaustiveness-checked.
    desc = _case_desc("02-case.yaml", "gate")
    check_case_exhaustive(
        {desc.id: desc},
        {
            "score": ValueContract(
                Type(kind=ValueKind.STRING, tags=frozenset({"a"}))
            )
        },
    )  # no raise


def test_on_non_enum_producer_skips_exhaustiveness():
    # on: a plain scalar producer (no .tags) -> lenient, no exhaustiveness, no raise.
    desc = _case_desc("06-case-on.yaml", "route")
    producers = {
        "classify": ValueContract(Type.scalar(ValueKind.STRING)),
    }
    result = _build_case(desc, producers)
    assert isinstance(result.node, CaseNode)
    check_case_exhaustive({desc.id: desc}, producers)


# --------------------------------------------------------------------------- #
# the built CASE passes handle alignment (every case has an edge)
# --------------------------------------------------------------------------- #


def test_handles_align_with_control_edges():
    desc = _case_desc("06-case-on.yaml", "route")
    result = _build_case(desc)
    _, control_edges = _split(result)
    case_handles = {c.handle for c in result.node.cases}
    edge_handles = {e.source_handle for e in control_edges}
    # every case handle has an outgoing edge.
    assert case_handles <= edge_handles
    # every edge handle is a case handle or the else handle.
    assert edge_handles <= case_handles | {ELSE_HANDLE}


# --------------------------------------------------------------------------- #
# the case's BuildResult.edges carry the reconciled __rN data group + control edges
# (no provisional <case>:<n> group is ever emitted for a case)
# --------------------------------------------------------------------------- #


def test_build_emits_reconciled_edges_no_provisional_group():
    # seed 02: the case's own build emits the reconciled __r0 data edge + control edges
    # directly — there is no provisional <case>:<n> group to reconcile away (infer_data_edges
    # emits nothing for a case; the case owns its edges).
    from yian.compose.edges.data import infer_data_edges

    path = _SEEDS / "02-case.yaml"
    document = SourceDocument.parse(path.read_text(), label=str(path))
    parse_file(document)
    descriptors = parse_nodes(document.root.child("nodes"))
    # infer_data_edges emits NO edge for the case node (it `continue`s on a case).
    flow_wiring = {
        nid: d.build(BuildContext({})).wiring
        for nid, d in descriptors.items()
        if not isinstance(d, CaseDescriptor)
    }
    data_edges = infer_data_edges(descriptors, flow_wiring)
    assert not any(e.to == "gate" for e in data_edges)

    result = _build_case(descriptors["gate"])
    # the reconciled __r0 data edge is present (no provisional <case>:<n> group).
    assert not any(":" in (e.input_group or "") for e in result.edges)
    assert ("score", "gate", "__r0") in {
        (e.from_, e.to, e.input_group) for e in result.edges
    }
    # the control edges are present.
    ctl = {(e.from_, e.to, e.source_handle) for e in result.edges if e.source_handle}
    assert ("gate", "positive", "positive") in ctl
    assert ("gate", "cautious", ELSE_HANDLE) in ctl


def test_on_value_with_double_quote_uses_single_quote_and_routes():
    # an on: match value containing " must compile to a parseable when: (single-quoted,
    # since the grammar has no escape support) and route correctly at eval time.
    from yian.compose.nodes.case import _quote

    expr = '__on == ' + _quote('a"b')
    assert evaluate_when_record(expr, {"__on": 'a"b'}) is True
    assert evaluate_when_record(expr, {"__on": "other"}) is False


def test_on_value_with_both_quotes_is_loud_load_error():
    # a value containing BOTH ' and " is unrepresentable in the escape-less grammar
    # -> a loud load error, not a route-time crash.
    from yian.compose.nodes.case import _quote

    with pytest.raises(LoadError):
        _quote("a'b\"c")
