"""Unit tests for `lower_flow_asserts` — the pure flow-assert lowering pass.

This is a PURE function (no engine): it takes a flow's `asserts:` list plus the
name environment (flow inputs, node ids, producer contracts) and lowers each assert
into a node-local assert for the synthesized START/END, plus the synth END
params + wiring needed to make cross-node post-asserts structural. These tests
pin the exact rewritten-string forms and the synth naming scheme.
"""

from __future__ import annotations

import pytest

from agent_composer.compose.validate.asserts import lower_flow_asserts
from agent_composer.compose.helpers.errors import LoadDiagnostics, LoadError
from agent_composer.diagnostics import OriginRef, SourceRange
from agent_composer.expr import condition_refs
from agent_composer.nodes.contracts import (
    Assertion,
    ContractOrigins,
    Parameter,
    PartialRecordContract,
    ValueContract,
)
from agent_composer.typesys.values import Type, ValueKind


# A small shared environment: two opaque producers + two flow inputs. Opaque
# producers are absent from the contract map and stay lenient in the Type walker.
PRODUCERS = {}
VALID_TARGETS = {"emit", "classify"}
FLOW_INPUTS = {"age", "threshold"}


def _lower(asserts, **kw):
    return lower_flow_asserts(
        [a if isinstance(a, Assertion) else Assertion(a, a) for a in asserts],
        kw.get("flow_inputs", FLOW_INPUTS),
        kw.get("valid_targets", VALID_TARGETS),
        kw.get("producers", PRODUCERS),
        kw.get("assert_ranges"),
        diagnostics=kw.get("diagnostics"),
    )


def _expressions(assertions: list[Assertion]) -> list[str]:
    """Project executable text while keeping carrier assertions explicit."""
    return [assertion.expression for assertion in assertions]


# ---- PRE (boundary) asserts: rewrite `input.` -> `output.` (START's coerced output) ------

def test_pre_mixed_spelling_rewrites_input_to_output():
    r = _lower(["${input.age} >= 18"])
    assert _expressions(r.start_pre) == ["${output.age} >= 18"]
    assert r.end_post == []
    assert r.synth_params == []
    assert r.synth_wiring == {}


def test_pre_bare_spelling_rewrites_input_to_output():
    r = _lower(["input.age >= 18"])
    assert _expressions(r.start_pre) == ["output.age >= 18"]
    assert r.synth_params == []


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("age >= 18", "output.age >= 18"),
        ("${age} >= 18", "${output.age} >= 18"),
        ("${age >= 18}", "${output.age >= 18}"),
    ],
)
def test_pre_unqualified_flow_input_rewrites_in_every_condition_spelling(
    source, expected
):
    r = _lower([source])
    assert _expressions(r.start_pre) == [expected]
    assert r.end_post == []
    assert r.synth_params == []
    assert r.synth_wiring == {}


def test_unqualified_flow_input_precedes_same_named_node():
    r = _lower(
        ["age >= 18", "age.output >= 18"],
        valid_targets={"age"},
    )

    assert _expressions(r.start_pre) == ["output.age >= 18"]
    assert _expressions(r.end_post) == ["__assert_age >= 18"]
    assert r.synth_params == [Parameter(name="__assert_age")]
    assert r.synth_wiring == {"__assert_age": "${age.output:-null}"}


# ---- POST asserts: synth END params ---------------------------------------------

def test_post_bare_producer_ref_lowers_to_synth_param():
    r = _lower(["emit.output.n >= 1"])
    # bare form rewrites without braces.
    assert _expressions(r.end_post) == ["__assert_emit.n >= 1"]
    assert r.synth_params == [Parameter(name="__assert_emit")]
    assert r.synth_wiring == {"__assert_emit": "${emit.output:-null}"}
    assert r.start_pre == []


def test_post_dotted_field_ref_lowers_to_synth_field():
    r = _lower(['${classify.output.label} == "x"'])
    assert _expressions(r.end_post) == ['${__assert_classify.label} == "x"']
    assert r.synth_params == [Parameter(name="__assert_classify")]
    assert r.synth_wiring == {"__assert_classify": "${classify.output:-null}"}


def test_post_same_producer_two_fields_share_one_synth():
    r = _lower(["emit.output.n >= 1", "emit.output.m >= 2"])
    assert len(r.synth_params) == 1
    assert r.synth_params[0].name == "__assert_emit"
    assert _expressions(r.end_post) == ["__assert_emit.n >= 1", "__assert_emit.m >= 2"]
    assert r.synth_wiring == {"__assert_emit": "${emit.output:-null}"}


def test_post_mixed_producer_and_input_two_synths():
    r = _lower(["emit.output.n >= input.threshold"])
    names = [p.name for p in r.synth_params]
    assert set(names) == {"__assert_emit", "__assert_input_threshold"}
    assert _expressions(r.end_post) == ["__assert_emit.n >= __assert_input_threshold"]
    # producer edge defaults to null; the input edge does NOT (input is always present).
    assert r.synth_wiring["__assert_emit"] == "${emit.output:-null}"
    assert r.synth_wiring["__assert_input_threshold"] == "${input.threshold}"


def test_post_mixed_producer_and_unqualified_input_two_synths():
    r = _lower(["emit.output.n >= age"])

    assert _expressions(r.end_post) == ["__assert_emit.n >= __assert_input_age"]
    assert r.synth_params == [
        Parameter(name="__assert_emit"),
        Parameter(name="__assert_input_age"),
    ]
    assert r.synth_wiring == {
        "__assert_emit": "${emit.output:-null}",
        "__assert_input_age": "${input.age}",
    }


# ---- Errors ---------------------------------------------------------------------

def test_system_namespace_is_forbidden():
    with pytest.raises(LoadError):
        _lower(["${system.run_id} != null"])


def test_dangling_producer_ref_raises():
    with pytest.raises(LoadError):
        _lower(["nonexist.output.x > 0"])


def test_dangling_input_ref_raises():
    # A `${input.X}` naming an undeclared input is a loud LoadError (mirrors the old
    # classify_asserts dangling-input check).
    with pytest.raises(LoadError, match="nope"):
        _lower(["${input.nope} != []"])


def test_unknown_unqualified_name_remains_a_located_error():
    rng = SourceRange(start_line=3, start_col=5, end_line=3, end_col=18)
    diagnostics = LoadDiagnostics(
        label="flow.yaml",
        text="first\nsecond\n    missing >= 18\n",
    )

    with pytest.raises(LoadError, match="unknown namespace 'missing'") as caught:
        _lower(
            ["missing >= 18"],
            assert_ranges=[rng],
            diagnostics=diagnostics,
        )

    assert caught.value.diagnostic.frames[0].range == rng


def test_dotted_input_name_is_not_unqualified_shorthand():
    with pytest.raises(LoadError, match="unknown namespace 'profile'"):
        _lower(["profile.age >= 18"], flow_inputs={"profile"})


def test_unknown_field_on_record_producer_raises():
    # A dotted ref against a RECORD producer must name a real field; an unknown field is
    # a loud LoadError (mirrors the old classify_asserts bad-field walk). A typed producer
    # is needed so the dotted walk actually runs (an absent producer contract stays lenient).
    view = Type(
        kind=ValueKind.OBJECT,
        fields={"score": Type.scalar(ValueKind.NUMBER)},
        required=frozenset({"score"}),
    )
    with pytest.raises(LoadError, match="badfield"):
        _lower(
            ["${synth.output.badfield} >= 0"],
            valid_targets={"synth"},
            producers={"synth": ValueContract(view)},
        )


def test_unknown_field_on_partial_record_producer_raises():
    """A partial producer is opaque by type, not by its closed root shape."""
    partial = PartialRecordContract(
        fields=(
            ("score", ValueContract(Type.scalar(ValueKind.NUMBER))),
            ("note", None),
        )
    )

    with pytest.raises(LoadError, match="badfield"):
        _lower(
            ["${synth.output.badfield} >= 0"],
            valid_targets={"synth"},
            producers={"synth": partial},
        )


def test_type_walk_preserves_the_producer_contract_and_origins():
    """Assert validation projects `.type` without replacing the semantic carrier."""
    view = Type(
        kind=ValueKind.OBJECT,
        fields={"score": Type.scalar(ValueKind.NUMBER)},
        required=frozenset({"score"}),
    )
    contract = ValueContract(
        view,
        ContractOrigins(default=OriginRef("producer-output")),
    )

    lowered = _lower(
        ["${synth.output.score} >= 0"],
        valid_targets={"synth"},
        producers={"synth": contract},
    )

    assert _expressions(lowered.end_post) == ["${__assert_synth.score} >= 0"]
    assert contract.type is view
    assert contract.origins.default == OriginRef("producer-output")


def test_assert_lowering_rejects_non_contract_producer_map_values():
    """The flow-assert boundary accepts exact ValueContract values only."""
    with pytest.raises(TypeError, match="producer contracts.*ValueContract"):
        _lower(  # type: ignore[dict-item]
            ["${synth.output} != null"],
            valid_targets={"synth"},
            producers={"synth": Type.scalar(ValueKind.STRING)},
        )


def test_malformed_expression_raises():
    with pytest.raises(LoadError):
        _lower(["emit.output.n >= "])


def test_lowering_error_reports_display_not_rewritten_expression():
    """Compile diagnostics use stable authored presentation from the carrier."""
    assertion = Assertion("missing.output > 0", "the authored business rule")

    with pytest.raises(LoadError, match="authored business rule"):
        _lower([assertion])


# ---- Every lowered string must remain a parseable condition ---------------------

def test_all_lowered_strings_parse():
    r = _lower(
        [
            "${input.age} >= 18",
            "emit.output.n >= 1",
            '${classify.output.label} == "x"',
            "emit.output.n >= input.threshold",
        ]
    )
    for assertion in r.start_pre + r.end_post:
        # condition_refs parses the whole value as a boolean expression; a broken
        # ref rewrite would raise here.
        condition_refs(assertion.expression)
    # Wiring sources are also ${...} refs — confirm they parse too.
    for src in r.synth_wiring.values():
        condition_refs(src)


# ---- Ranges stay in Compose validation ------------------------------------------

def test_ranges_are_not_carried_by_lowered_product():
    """Positional validation ranges do not cross into the lowered semantic product."""
    from agent_composer.diagnostics import SourceRange

    rng_pre = SourceRange(start_line=3, start_col=1, end_line=3, end_col=20)
    rng_post = SourceRange(start_line=4, start_col=1, end_line=4, end_col=20)
    r = _lower(
        ["${input.age} >= 18", "emit.output.n >= 1"],  # [0] PRE, [1] POST
        assert_ranges=[rng_pre, rng_post],
    )
    assert len(r.start_pre) == 1
    assert len(r.end_post) == 1
    assert not hasattr(r, "pre_ranges")
    assert not hasattr(r, "post_ranges")


def test_lowering_without_ranges_needs_no_padding():
    """A source-free assert lowers without synthetic location state."""
    r = _lower(["${input.age} >= 18"])
    assert len(r.start_pre) == 1
    assert r.end_post == []


def test_lowering_changes_only_expression_and_preserves_occurrence_metadata():
    """Display and origin belong to each occurrence even when executable text is rewritten."""
    first = Assertion("${input.age} >= 18", "age must be adult", OriginRef("asserts/0"))
    second = Assertion("${input.age} >= 18", "age must be adult", OriginRef("asserts/1"))

    lowered = _lower([first, second])

    assert _expressions(lowered.start_pre) == ["${output.age} >= 18"] * 2
    assert [a.display for a in lowered.start_pre] == ["age must be adult"] * 2
    assert [a.origin for a in lowered.start_pre] == [first.origin, second.origin]
    assert lowered.start_pre[0] is not lowered.start_pre[1]
