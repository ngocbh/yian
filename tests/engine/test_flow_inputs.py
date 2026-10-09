"""Compose flow inputs lower directly into canonical semantic parameters."""

import pytest

from yian.diagnostics import SourceRange
from yian.nodes.contracts import Parameter, ValueContract
from yian.typesys.values import ValueKind
from yian.nodes.binding import apply_defaults, coerce_inputs
from yian.compose import LoadError
from yian.compose.helpers.errors import LoadDiagnostics
from yian.compose.helpers.types import read_flow_inputs


def _by_name(decls):
    return {d.name: d for d in decls}


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
    decls = read_flow_inputs({"window": "int = 30"}, {})
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
    d = _by_name(read_flow_inputs({"as_of": "Optional[date]"}, {}))["as_of"]
    assert _type(d).nullable is True
    assert d.required is False
    assert d.has_default is False
    assert d.default is None


def test_nullable_null_default_is_distinct_from_no_default():
    absent = _by_name(read_flow_inputs({"value": "Optional[int]"}, {}))["value"]
    explicit_null = _by_name(read_flow_inputs({"value": "Optional[int] = null"}, {}))[
        "value"
    ]

    assert absent.has_default is False
    assert explicit_null.has_default is True
    assert explicit_null.default is None
    assert explicit_null.required is False
    assert explicit_null != absent


def test_nonnullable_null_default_is_rejected_during_load():
    with pytest.raises(LoadError, match="default"):
        read_flow_inputs({"value": "int = null"}, {})


def test_empty_default_assignment_is_rejected_during_load():
    with pytest.raises(LoadError, match="default"):
        read_flow_inputs({"value": "int ="}, {})


def test_quoted_integer_default_is_not_coerced():
    with pytest.raises(LoadError, match="default"):
        read_flow_inputs({"value": 'int = "30"'}, {})


def test_number_default_retains_compatible_integer_payload():
    parameter = _by_name(read_flow_inputs({"value": "float = 30"}, {}))["value"]

    assert _type(parameter).kind == ValueKind.NUMBER
    assert parameter.has_default is True
    assert type(parameter.default) is int
    assert parameter.default == 30


@pytest.mark.parametrize(
    ("declaration", "message"),
    [
        ("list[", "bad type expression"),
        ({"field": 3}, "cannot read type"),
        ([{"field": "int"}], "declaration must be a type string or a record map"),
        (3, "declaration must be a type string or a record map"),
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
    d = _by_name(read_flow_inputs({"budget": "float = 1000.0"}, {}))["budget"]
    assert _type(d).kind == ValueKind.NUMBER
    assert d.default == 1000.0 and isinstance(d.default, float)


def test_list_default():
    d = _by_name(read_flow_inputs({"bundle": 'list[str] = ["ACME"]'}, {}))["bundle"]
    assert _type(d).kind == ValueKind.LIST_STRING
    assert d.default == ["ACME"]
    assert d.has_default is True
    assert d.required is False


def test_bare_word_string_default():
    d = _by_name(read_flow_inputs({"style": "str = relevance"}, {}))["style"]
    assert _type(d).kind == ValueKind.STRING
    assert d.default == "relevance"
    assert d.has_default is True
    assert d.required is False


def test_record_typed_dict_value():
    d = _by_name(
        read_flow_inputs(
            {
                "config": {
                    "regroup": "bool",
                    "bands": {"lower": "float", "upper": "float"},
                }
            },
            {},
        )
    )["config"]
    typ = _type(d)
    assert typ.kind == ValueKind.OBJECT
    assert typ.fields["regroup"].kind == ValueKind.BOOLEAN
    bands = typ.fields["bands"]
    assert bands.kind == ValueKind.OBJECT
    assert bands.fields["lower"].kind == ValueKind.NUMBER
    assert bands.fields["upper"].kind == ValueKind.NUMBER
    assert d.default is None
    assert d.required is True


def test_apply_defaults_fills_omitted():
    decls = read_flow_inputs({"topic": "str", "window": "int = 30"}, {})
    out = apply_defaults(decls, coerce_inputs(decls, {"topic": "ACME"}))
    assert out == {"topic": "ACME", "window": 30}


def test_reader_returns_only_exact_parameter_carriers():
    decls = read_flow_inputs({"topic": "str"}, {})
    assert all(type(d) is Parameter for d in decls)
    assert all(type(d.contract) is ValueContract for d in decls)
    assert all(d.origin is None and d.default_origin is None for d in decls)
    assert all(not d.contract.origins.entries for d in decls)
    assert all(d.contract.origins.default is None for d in decls)


def test_bool_coercion_truthy_and_falsy():
    decls = read_flow_inputs({"flag": "bool"}, {})
    assert coerce_inputs(decls, {"flag": "yes"})["flag"] is True  # truthy set
    assert coerce_inputs(decls, {"flag": "off"})["flag"] is False  # outside the set


def test_bad_number_passes_through_unchanged():
    # Scalar coercion returns a non-numeric string as entered; the generic binder's
    # subsequent contract validation rejects it.
    decls = read_flow_inputs({"n": "int"}, {})
    assert coerce_inputs(decls, {"n": "not-a-number"})["n"] == "not-a-number"
