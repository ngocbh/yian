"""Compose flow inputs lower directly into canonical semantic parameters."""

import re

import pytest

from yian.diagnostics import SourceRange
from yian.nodes.contracts import Parameter, ValueContract
from yian.typesys.values import ValueKind
from yian.nodes.binding import apply_defaults, coerce_inputs
from yian.compose import LoadError
from yian.compose.helpers.errors import LoadDiagnostics
from yian.compose.helpers.types import read_flow_inputs
from tests.compose._declarations import parameters_from_yaml


def _by_name(decls):
    return {d.name: d for d in decls}


def _inputs(body: str) -> list[Parameter]:
    """Read flow `input:` declarations from their YAML body, one per line."""
    lines = "".join(f"  {line}\n" for line in body.splitlines())
    return parameters_from_yaml("input:\n" + lines, {})


def _type(parameter: Parameter):
    """Return one required flow parameter's semantic type."""
    assert type(parameter.contract) is ValueContract
    return parameter.contract.type


def test_required_scalar():
    decls = _by_name(read_flow_inputs({"topic": "str"}, {}))
    d = decls["topic"]
    assert type(d) is Parameter
    assert d.required is True
    assert d.has_default is False
    assert d.default is None
    assert _type(d).kind == ValueKind.STRING
    assert not hasattr(d, "type_str")


def test_int_with_default_and_coercion():
    decls = _inputs("window: !int 30")
    d = _by_name(decls)["window"]
    assert _type(d).kind == ValueKind.INTEGER
    assert d.default == 30 and isinstance(d.default, int)
    assert d.has_default is True
    assert d.required is False
    assert d.coerce_strings is True
    # coerce_inputs must coerce a passed numeric STRING through the canonical name
    coerced = coerce_inputs(decls, {"window": "5"})
    assert coerced["window"] == 5 and isinstance(coerced["window"], int)


def test_optional_nullable():
    d = _by_name(read_flow_inputs({"as_of": "Optional(date)"}, {}))["as_of"]
    assert _type(d).nullable is True
    assert d.required is False
    assert d.has_default is False
    assert d.default is None


def test_nullable_null_default_is_distinct_from_no_default():
    absent = _by_name(read_flow_inputs({"value": "Optional(int)"}, {}))["value"]
    explicit_null = _by_name(_inputs("value: !Optional(int) null"))["value"]

    assert absent.has_default is False
    assert explicit_null.has_default is True
    assert explicit_null.default is None
    assert explicit_null.required is False
    assert explicit_null != absent


def test_nonnullable_null_default_is_rejected_during_load():
    with pytest.raises(LoadError, match=re.escape("'value': `null` is not a valid int")):
        _inputs("value: !int null")


def test_source_free_default_assignment_is_rejected_during_load():
    with pytest.raises(
        LoadError,
        match=re.escape(
            "input 'value': source-free declarations take a type only; "
            "write defaults in YAML with a tag"
        ),
    ):
        read_flow_inputs({"value": "int ="}, {})


def test_quoted_integer_default_is_not_coerced():
    with pytest.raises(LoadError, match=re.escape("""'value': `"30"` is not a valid int""")):
        _inputs('value: !int "30"')


def test_former_boolean_word_default_for_bool_names_the_boolean_to_write():
    with pytest.raises(LoadError) as caught:
        _inputs("flag: !bool no")

    assert str(caught.value) == "'flag': `no` is not a valid bool; `no` is text; write `false`"


def test_former_boolean_word_default_for_int_gets_no_fix():
    with pytest.raises(LoadError) as caught:
        _inputs("flag: !int no")

    assert "is text" not in str(caught.value)


def test_former_boolean_word_in_a_list_default_names_the_boolean_to_write():
    with pytest.raises(LoadError) as caught:
        _inputs("flags: !list(bool) [yes, no]")

    assert str(caught.value).endswith("; `yes` is text; write `true`")


def test_number_default_retains_compatible_integer_payload():
    parameter = _by_name(_inputs("value: !float 30"))["value"]

    assert _type(parameter).kind == ValueKind.NUMBER
    assert parameter.has_default is True
    assert type(parameter.default) is int
    assert parameter.default == 30


@pytest.mark.parametrize(
    ("declaration", "message"),
    [
        ("list(", "bad type expression"),
        ({"field": 3}, "a source-free declaration is type text, got dict"),
        ([{"field": "int"}], "a source-free declaration is type text, got list"),
        (3, "a source-free declaration is type text, got int"),
    ],
)
def test_input_declaration_errors_retain_their_source_range(declaration, message):
    source_range = SourceRange(4, 10, 4, 18)
    diagnostics = LoadDiagnostics(label="flow.yaml", text=(" " * 40 + "\n") * 5)

    with pytest.raises(LoadError, match=message) as caught:
        read_flow_inputs(
            {"value": declaration},
            {},
            {"value": source_range},
            diagnostics=diagnostics,
        )

    assert len(caught.value.diagnostic.frames) == 1
    assert caught.value.diagnostic.frames[0].range is source_range
    assert caught.value.diagnostic.frames[0].label == "flow.yaml"


def test_float_default():
    d = _by_name(_inputs("budget: !float 1000.0"))["budget"]
    assert _type(d).kind == ValueKind.NUMBER
    assert d.default == 1000.0 and isinstance(d.default, float)


def test_list_default():
    d = _by_name(_inputs('bundle: !list(str) ["ACME"]'))["bundle"]
    assert _type(d).kind == ValueKind.LIST_STRING
    assert d.default == ["ACME"]
    assert d.has_default is True
    assert d.required is False


def test_bare_word_string_default():
    d = _by_name(_inputs("style: !str relevance"))["style"]
    assert _type(d).kind == ValueKind.STRING
    assert d.default == "relevance"
    assert d.has_default is True
    assert d.required is False


def test_apply_defaults_fills_omitted():
    decls = _inputs("topic: !str\nwindow: !int 30")
    out = apply_defaults(decls, coerce_inputs(decls, {"topic": "ACME"}))
    assert out == {"topic": "ACME", "window": 30}


def test_reader_returns_only_exact_parameter_carriers():
    decls = read_flow_inputs({"topic": "str"}, {})
    assert all(type(d) is Parameter for d in decls)
    assert all(type(d.contract) is ValueContract for d in decls)
    assert all(d.origin is None and d.default_origin is None for d in decls)
    assert all(not d.contract.origins.entries for d in decls)
    assert all(d.contract.origins.default is None for d in decls)


@pytest.mark.parametrize("raw", ["1", "true", "TRUE", "yes", "y", "on", " True "])
def test_bool_coercion_true_spellings(raw):
    decls = read_flow_inputs({"flag": "bool"}, {})
    assert coerce_inputs(decls, {"flag": raw})["flag"] is True


@pytest.mark.parametrize("raw", ["0", "false", "FALSE", "no", "n", "off", " False "])
def test_bool_coercion_false_spellings(raw):
    decls = read_flow_inputs({"flag": "bool"}, {})
    assert coerce_inputs(decls, {"flag": raw})["flag"] is False


@pytest.mark.parametrize("raw", ["maybe", "ture", "approve", "No thanks", ""])
def test_bool_coercion_passes_unrecognized_text_through(raw):
    # Like a bad number, an unrecognized spelling stays a string so the contract rejects it.
    decls = read_flow_inputs({"flag": "bool"}, {})
    assert coerce_inputs(decls, {"flag": raw})["flag"] == raw


def test_bad_number_passes_through_unchanged():
    # Scalar coercion returns a non-numeric string as entered; the generic binder's
    # subsequent contract validation rejects it.
    decls = read_flow_inputs({"n": "int"}, {})
    assert coerce_inputs(decls, {"n": "not-a-number"})["n"] == "not-a-number"
