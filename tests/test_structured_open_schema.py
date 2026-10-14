"""Open AGENT output schemas (`object`, `Any`, elementless lists) under strict structured output."""

import pytest

from yian.llm_clients.capabilities import has_open_schema
from yian.nodes.agent.structured import type_to_schema
from yian.typesys.types import read_typedefs, type_for
from yian.typesys.values import Type, ValueKind

REGISTRY = read_typedefs({
    "R": {"a": "int", "m": "object"},
    "Buy": {"kind": "Literal(buy)", "x": "Any"},
    "Sell": {"kind": "Literal(sell)", "n": "int"},
    "D": "Union(Buy, Sell)",
    "Outer": {"inner": "R", "n": "int"},
    "C": {"a": "int", "b": "list(str)"},
    "C2": {"kind": "Literal(c2)", "a": "int"},
    "C3": {"kind": "Literal(c3)", "b": "str"},
    "CU": "Union(C2, C3)",
})


def _schema(text):
    return type_to_schema(type_for(text, REGISTRY))


@pytest.mark.parametrize("text", [
    "object", "Any", "Optional(Any)", "list(object)", "list(Any)", "list(Optional(Any))",
    "Optional(object)", "list(list(object))", "Optional(R)", "list(R)", "R", "D", "Outer",
])
def test_open_types_have_an_open_schema(text):
    assert has_open_schema(_schema(text))


def test_an_elementless_list_has_an_open_schema():
    assert has_open_schema(type_to_schema(Type(kind=ValueKind.LIST_ANY)))


@pytest.mark.parametrize("text", [
    "int", "float(0.0..1.0)", "bool", "date", "Literal(a, b)", "list(str)", "list(int)",
    "C", "CU", "Optional(int)",
])
def test_closed_types_have_no_open_schema(text):
    assert not has_open_schema(_schema(text))


@pytest.mark.parametrize(("schema", "expected"), [
    ({"type": "object", "additionalProperties": True}, True),
    ({"type": "object", "properties": {}, "additionalProperties": False}, False),
    ({"type": "array", "items": {}}, True),
])
def test_dict_schemas_are_read_as_given(schema, expected):
    assert has_open_schema(schema) is expected
