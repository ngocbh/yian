"""Unit tests for `lower_flow_asserts` — the pure flow-assert lowering pass.

This is a PURE function (no engine): it takes a flow's `asserts:` list plus the
name environment (flow inputs, node ids, producer types) and lowers each assert
into a node-local assert for the synthesized START/END, plus the synth END
params + wiring needed to make cross-node post-asserts structural. These tests
pin the exact rewritten-string forms and the synth naming scheme.
"""

from __future__ import annotations

import pytest

from agent_composer.compose.validate.asserts import lower_flow_asserts
from agent_composer.compose.helpers.errors import LoadError
from agent_composer.expr import condition_refs
from agent_composer.nodes.base import ParamDecl


# A small shared environment: two producers (opaque output types) + two flow inputs.
PRODUCERS = {"emit": None, "classify": None}
VALID_TARGETS = {"emit", "classify"}
FLOW_INPUTS = {"age", "threshold"}


def _lower(asserts, **kw):
    return lower_flow_asserts(
        asserts,
        kw.get("flow_inputs", FLOW_INPUTS),
        kw.get("valid_targets", VALID_TARGETS),
        kw.get("producers", PRODUCERS),
        kw.get("assert_ranges"),
    )


# ---- PRE (boundary) asserts: rewrite `input.` -> `output.` (START's coerced output) ------

def test_pre_mixed_spelling_rewrites_input_to_output():
    r = _lower(["${input.age} >= 18"])
    assert r.start_pre == ["${output.age} >= 18"]
    assert r.end_post == []
    assert r.synth_params == []
    assert r.synth_wiring == {}


def test_pre_bare_spelling_rewrites_input_to_output():
    r = _lower(["input.age >= 18"])
    assert r.start_pre == ["output.age >= 18"]
    assert r.synth_params == []


# ---- POST asserts: synth END params ---------------------------------------------

def test_post_bare_producer_ref_lowers_to_synth_param():
    r = _lower(["emit.output.n >= 1"])
    # bare form rewrites without braces.
    assert r.end_post == ["__assert_emit.n >= 1"]
    assert r.synth_params == [ParamDecl(name="__assert_emit")]
    assert r.synth_wiring == {"__assert_emit": "${emit.output:-null}"}
    assert r.start_pre == []


def test_post_dotted_field_ref_lowers_to_synth_field():
    r = _lower(['${classify.output.label} == "x"'])
    assert r.end_post == ['${__assert_classify.label} == "x"']
    assert r.synth_params == [ParamDecl(name="__assert_classify")]
    assert r.synth_wiring == {"__assert_classify": "${classify.output:-null}"}


def test_post_same_producer_two_fields_share_one_synth():
    r = _lower(["emit.output.n >= 1", "emit.output.m >= 2"])
    assert len(r.synth_params) == 1
    assert r.synth_params[0].name == "__assert_emit"
    assert r.end_post == ["__assert_emit.n >= 1", "__assert_emit.m >= 2"]
    assert r.synth_wiring == {"__assert_emit": "${emit.output:-null}"}


def test_post_mixed_producer_and_input_two_synths():
    r = _lower(["emit.output.n >= input.threshold"])
    names = [p.name for p in r.synth_params]
    assert set(names) == {"__assert_emit", "__assert_input_threshold"}
    assert r.end_post == ["__assert_emit.n >= __assert_input_threshold"]
    # producer edge defaults to null; the input edge does NOT (input is always present).
    assert r.synth_wiring["__assert_emit"] == "${emit.output:-null}"
    assert r.synth_wiring["__assert_input_threshold"] == "${input.threshold}"


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


def test_unknown_field_on_record_producer_raises():
    # A dotted ref against a RECORD producer must name a real field; an unknown field is
    # a loud LoadError (mirrors the old classify_asserts bad-field walk). A typed producer
    # is needed so the dotted walk actually runs (opaque `None` producers stay lenient).
    from agent_composer.typesys.values import Type, ValueKind

    view = Type(
        kind=ValueKind.OBJECT,
        fields={"score": Type.scalar(ValueKind.NUMBER)},
        required=frozenset({"score"}),
    )
    with pytest.raises(LoadError, match="badfield"):
        _lower(
            ["${synth.output.badfield} >= 0"],
            valid_targets={"synth"},
            producers={"synth": view},
        )


def test_malformed_expression_raises():
    with pytest.raises(LoadError):
        _lower(["emit.output.n >= "])


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
    for s in r.start_pre + r.end_post:
        # condition_refs parses the whole value as a boolean expression; a broken
        # ref rewrite would raise here.
        condition_refs(s)
    # Wiring sources are also ${...} refs — confirm they parse too.
    for src in r.synth_wiring.values():
        condition_refs(src)


# ---- Ranges (render-only) -------------------------------------------------------

def test_ranges_are_positional():
    # `assert_ranges` is a POSITIONAL list aligned 1:1 with the asserts; `pre_ranges`/`post_ranges`
    # come out aligned with `start_pre`/`end_post`. Distinct ranges prove the alignment (no text key).
    from agent_composer.events import SourceRange

    rng_pre = SourceRange(start_line=3, start_col=1, end_line=3, end_col=20)
    rng_post = SourceRange(start_line=4, start_col=1, end_line=4, end_col=20)
    r = _lower(
        ["${input.age} >= 18", "emit.output.n >= 1"],  # [0] PRE, [1] POST
        assert_ranges=[rng_pre, rng_post],
    )
    assert r.pre_ranges == [rng_pre]     # aligned with start_pre
    assert r.post_ranges == [rng_post]   # aligned with end_post


def test_no_ranges_gives_none_padded_lists():
    # No `assert_ranges` supplied -> one None PER assert, so the lists stay aligned with start_pre /
    # end_post (never shorter). A single PRE assert -> pre_ranges == [None], post_ranges == [].
    r = _lower(["${input.age} >= 18"])
    assert r.pre_ranges == [None]
    assert r.post_ranges == []
