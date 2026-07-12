"""Unit tests for read_flow_inputs — the Compose `inputs:` declaration reader."""

import pytest

from agent_composer.typesys.values import ValueKind
from agent_composer.typesys.seeding import apply_defaults, coerce_inputs
from agent_composer.compose import LoadError
from agent_composer.compose.helpers.types import InputDecl, read_flow_inputs


def _by_name(decls):
    return {d.name: d for d in decls}


def test_required_scalar():
    decls = _by_name(read_flow_inputs({"topic": "str"}, {}))
    d = decls["topic"]
    assert d.required is True
    assert d.default is None
    assert d.type_str == "str"
    assert d.type.kind == ValueKind.STRING


def test_int_with_default_and_coercion():
    decls = read_flow_inputs({"window": "int = 30"}, {})
    d = _by_name(decls)["window"]
    assert d.type_str == "int"
    assert d.default == 30 and isinstance(d.default, int)
    assert d.required is False
    # coerce_inputs must coerce a passed numeric STRING through the canonical name
    coerced = coerce_inputs(decls, {"window": "5"})
    assert coerced["window"] == 5 and isinstance(coerced["window"], int)


def test_optional_nullable():
    d = _by_name(read_flow_inputs({"as_of": "Optional[date]"}, {}))["as_of"]
    assert d.type.nullable is True
    assert d.required is False
    assert d.default is None


def test_float_default():
    d = _by_name(read_flow_inputs({"budget": "float = 1000.0"}, {}))["budget"]
    assert d.type_str == "float"
    assert d.default == 1000.0 and isinstance(d.default, float)


def test_list_default():
    d = _by_name(read_flow_inputs({"bundle": 'list[str] = ["ACME"]'}, {}))["bundle"]
    assert d.type.kind == ValueKind.LIST_STRING
    assert d.default == ["ACME"]
    assert d.required is False


def test_bare_word_string_default():
    d = _by_name(read_flow_inputs({"style": "str = relevance"}, {}))["style"]
    assert d.type_str == "str"
    assert d.default == "relevance"
    assert d.required is False


def test_record_typed_dict_value():
    d = _by_name(
        read_flow_inputs(
            {"config": {"regroup": "bool", "bands": {"lower": "float", "upper": "float"}}},
            {},
        )
    )["config"]
    assert d.type.kind == ValueKind.OBJECT
    assert d.type.fields["regroup"].kind == ValueKind.BOOLEAN
    bands = d.type.fields["bands"]
    assert bands.kind == ValueKind.OBJECT
    assert bands.fields["lower"].kind == ValueKind.NUMBER
    assert bands.fields["upper"].kind == ValueKind.NUMBER
    assert d.default is None
    assert d.required is True


def test_apply_defaults_fills_omitted():
    decls = read_flow_inputs({"topic": "str", "window": "int = 30"}, {})
    out = apply_defaults(decls, coerce_inputs(decls, {"topic": "ACME"}))
    assert out == {"topic": "ACME", "window": 30}


def test_isinstance_inputdecl():
    decls = read_flow_inputs({"topic": "str"}, {})
    assert all(isinstance(d, InputDecl) for d in decls)


def test_bool_coercion_truthy_and_falsy():
    decls = read_flow_inputs({"flag": "bool"}, {})
    assert coerce_inputs(decls, {"flag": "yes"})["flag"] is True   # truthy set
    assert coerce_inputs(decls, {"flag": "off"})["flag"] is False  # outside the set


def test_bad_number_passes_through_unchanged():
    # A non-numeric string for an int input is returned as-entered (the boundary
    # does NOT raise; a downstream type-enforcing write/assert surfaces it).
    decls = read_flow_inputs({"n": "int"}, {})
    assert coerce_inputs(decls, {"n": "not-a-number"})["n"] == "not-a-number"
