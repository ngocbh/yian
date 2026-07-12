"""Unit tests for `asserts:` parse + compile-validate + boundary/post classify.

`classify_asserts(assert_list, flow_inputs, valid_targets, producers)` parses each assert
string in the `when:`/`asserts:` boolean grammar (arithmetic + boolean + `in`), validates
every `${...}` ref it reads (reusing `compile.validation._classify_path` — a dangling/bad-field
ref -> a loud `LoadError`), and CLASSIFIES each assert by whether ANY of its refs has head
`outputs`:

- `${input.X}` / `${system.X}`-only asserts (seeds 05/10) -> **boundary** (fire pre-run).
- a `${X.output}`-referencing assert (seed 18) -> **post-terminal** (fire after the run).

Classification does NOT run the asserts — it only parses, validates, and splits.
"""

from pathlib import Path

import pytest

from agent_composer.typesys.values import ValueKind, Type
from agent_composer.compose.asserts import AssertSet, classify_asserts
from agent_composer.compose.helpers.errors import LoadError

_SEEDS = Path(__file__).resolve().parents[2] / "tests" / "seeds"


# --------------------------------------------------------------------------- #
# asserts parse — the arithmetic / `in` / boolean forms validate (not run)
# --------------------------------------------------------------------------- #


def test_seed05_assert_parses_and_classifies_boundary():
    # ${input.topics} != []  — an inputs-only assert.
    result = classify_asserts(
        ["${input.topics} != []"],
        flow_inputs={"topics", "as_of"},
        valid_targets=set(),
        producers={},
    )
    assert isinstance(result, AssertSet)
    assert result.boundary == ["${input.topics} != []"]
    assert result.post == []


def test_seed10_arithmetic_and_in_forms_validate_to_boundary():
    # seed 10's four asserts: arithmetic ranges, `*`, `in` membership, `not`+parens.
    asserts = [
        "${input.window} >= 5 and ${input.window} <= 252",
        "${input.weight} * 100 <= 10",
        '${input.style} in ["relevance", "value", "quality"]',
        "not (${input.weight} < 0)",
    ]
    result = classify_asserts(
        asserts,
        flow_inputs={"topic", "window", "weight", "style"},
        valid_targets=set(),
        producers={},
    )
    assert result.boundary == asserts  # all inputs-only -> boundary, order preserved
    assert result.post == []


def test_seed18_output_assert_classifies_post_terminal():
    # ${synth.output.confidence} >= 0 and ${synth.output.confidence} <= 1
    # synth produces a View record {stance, claim, confidence} -> the dotted ref validates.
    view = Type(
        kind=ValueKind.OBJECT,
        fields={
            "stance": Type.scalar(ValueKind.STRING),
            "claim": Type.scalar(ValueKind.STRING),
            "confidence": Type.scalar(ValueKind.NUMBER),
        },
        required=frozenset({"stance", "claim", "confidence"}),
    )
    assert_str = "${synth.output.confidence} >= 0 and ${synth.output.confidence} <= 1"
    result = classify_asserts(
        [assert_str],
        flow_inputs={"topic", "as_of"},
        valid_targets={"synth", "route", "pro_note", "con_note", "neutral_note"},
        producers={"synth": view},
    )
    assert result.boundary == []
    assert result.post == [assert_str]  # has an ${X.output} ref -> post-terminal


# --------------------------------------------------------------------------- #
# the three condition spellings load + classify identically (Class A unification)
#
# A condition value can be written three ways that evaluate identically at runtime:
#   bare        `a > 5`      — no braces
#   mixed       `${a} > 5`   — braces on the ref leaf, operator outside
#   whole-span  `${a > 5}`   — braces around the whole expression
# `classify_asserts` extracts refs by PARSING the whole value (via `condition_refs`),
# so all three must validate refs the same way and land in the same split. (The old
# flat-regex extractor read a whole-span interior as one bogus path -> spurious
# LoadError, and saw no refs in a bare form -> missed validation.)
# --------------------------------------------------------------------------- #


def test_three_spellings_of_boundary_assert_all_classify_boundary():
    # Same inputs-only invariant written bare / mixed / whole-span -> all boundary,
    # all validated (a dangling ref in any spelling would raise below).
    for spelling in (
        "input.topics != []",        # bare
        "${input.topics} != []",     # mixed
        "${input.topics != []}",     # whole-span
    ):
        result = classify_asserts(
            [spelling],
            flow_inputs={"topics", "as_of"},
            valid_targets=set(),
            producers={},
        )
        assert result.boundary == [spelling], spelling
        assert result.post == [], spelling


def test_three_spellings_of_output_assert_all_classify_post():
    view = Type(
        kind=ValueKind.OBJECT,
        fields={"confidence": Type.scalar(ValueKind.NUMBER)},
        required=frozenset({"confidence"}),
    )
    for spelling in (
        "synth.output.confidence >= 0",        # bare
        "${synth.output.confidence} >= 0",     # mixed
        "${synth.output.confidence >= 0}",     # whole-span
    ):
        result = classify_asserts(
            [spelling],
            flow_inputs={"topic"},
            valid_targets={"synth"},
            producers={"synth": view},
        )
        assert result.boundary == [], spelling
        assert result.post == [spelling], spelling


def test_dangling_ref_is_loud_in_every_spelling():
    # The whole-span spelling in particular used to slip past the flat-regex validator
    # (its interior read as one bogus path); it must now raise like the others.
    for spelling in (
        "input.nope != []",
        "${input.nope} != []",
        "${input.nope != []}",
    ):
        with pytest.raises(LoadError, match="nope"):
            classify_asserts(
                [spelling],
                flow_inputs={"topics"},
                valid_targets=set(),
                producers={},
            )


# --------------------------------------------------------------------------- #
# dangling / bad refs are loud
# --------------------------------------------------------------------------- #


def test_dangling_input_ref_is_loud():
    with pytest.raises(LoadError) as exc:
        classify_asserts(
            ["${input.nope} != []"],
            flow_inputs={"topics"},
            valid_targets=set(),
            producers={},
        )
    assert "nope" in str(exc.value)


def test_dangling_output_node_ref_is_loud():
    with pytest.raises(LoadError) as exc:
        classify_asserts(
            ["${typo.output} >= 0"],
            flow_inputs=set(),
            valid_targets={"synth"},
            producers={},
        )
    assert "typo" in str(exc.value)


def test_unknown_field_on_record_producer_is_loud():
    view = Type(
        kind=ValueKind.OBJECT,
        fields={"score": Type.scalar(ValueKind.NUMBER)},
        required=frozenset({"score"}),
    )
    with pytest.raises(LoadError) as exc:
        classify_asserts(
            ["${synth.output.badfield} >= 0"],
            flow_inputs=set(),
            valid_targets={"synth"},
            producers={"synth": view},
        )
    assert "badfield" in str(exc.value)


# --------------------------------------------------------------------------- #
# malformed boolean expression is loud
# --------------------------------------------------------------------------- #


def test_malformed_expression_is_loud():
    # a syntactically malformed expression (a doubled comparison operator) is rejected by
    # the unified grammar. (Under the unified engine a bare reference is a truthiness test,
    # not a parse error, so it is no longer the malformed case.)
    with pytest.raises(LoadError):
        classify_asserts(
            ["${input.x} == == 5"],
            flow_inputs={"x"},
            valid_targets=set(),
            producers={},
        )


# --------------------------------------------------------------------------- #
# empty asserts -> empty set
# --------------------------------------------------------------------------- #


def test_no_asserts_yields_empty_set():
    result = classify_asserts([], flow_inputs=set(), valid_targets=set(), producers={})
    assert result.boundary == []
    assert result.post == []


# --------------------------------------------------------------------------- #
# a mix of boundary + post asserts splits correctly, order preserved within each
# --------------------------------------------------------------------------- #


def test_mixed_asserts_split_correctly():
    score = Type.scalar(ValueKind.NUMBER)
    asserts = [
        "${input.weight} >= 0",            # boundary
        "${score.output} <= 1",          # post (outputs ref)
        "${input.weight} <= 1",            # boundary
    ]
    result = classify_asserts(
        asserts,
        flow_inputs={"weight"},
        valid_targets={"score"},
        producers={"score": score},
    )
    assert result.boundary == ["${input.weight} >= 0", "${input.weight} <= 1"]
    assert result.post == ["${score.output} <= 1"]
