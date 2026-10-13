"""A tool's JSON Schema translates into a yian `Type` that is never narrower than it.

The cases are raw JSON Schema dicts, so the same rules cover pydantic-built
tools and MCP tools that only publish a schema.
"""

from __future__ import annotations

from typing import Any

import pytest

from yian.compile.validation import type_relation
from yian.tools.types import type_from_json_schema
from yian.typesys.values import Type, ValueKind, canonical_type_label


def _label(schema: dict[str, Any], *, closed: bool = True) -> str | None:
    typ = type_from_json_schema(schema, closed=closed)
    return None if typ is None else canonical_type_label(typ)


_BUY = {
    "type": "object",
    "properties": {
        "action": {"const": "buy", "title": "Action", "type": "string"},
        "qty": {"title": "Qty", "type": "integer"},
    },
    "required": ["action", "qty"],
    "title": "Buy",
}
_SELL = {
    "type": "object",
    "properties": {
        "action": {"const": "sell", "title": "Action", "type": "string"},
        "price": {"title": "Price", "type": "number"},
    },
    "required": ["action", "price"],
    "title": "Sell",
}
_TAGGED_UNION = {
    "$defs": {"Buy": _BUY, "Sell": _SELL},
    "anyOf": [{"$ref": "#/$defs/Sell"}, {"$ref": "#/$defs/Buy"}],
}


@pytest.mark.parametrize(
    ("schema", "label"),
    [
        ({"type": "string"}, "str"),
        ({"type": "string", "format": "date"}, "str"),
        ({"type": "string", "format": "date-time"}, "str"),
        ({"type": "string", "format": "email", "minLength": 1, "pattern": "@"}, "str"),
        ({"enum": ["b", "a"]}, 'Literal["a","b"]'),
        ({"enum": ["a", "b"], "type": "string"}, 'Literal["a","b"]'),
        ({"const": "buy"}, 'Literal["buy"]'),
        ({"enum": [1, 2], "type": "integer"}, "int"),
        ({"const": 3, "type": "number"}, "float"),
        ({"enum": [1, 2]}, None),
        ({"type": "integer"}, "int"),
        ({"type": "number"}, "float"),
        ({"type": "integer", "minimum": 1, "maximum": 10}, "int[1,10]"),
        ({"type": "integer", "minimum": 1, "exclusiveMaximum": 10}, "int[1,9]"),
        ({"type": "integer", "exclusiveMinimum": 0, "maximum": 5}, "int[1,5]"),
        ({"type": "integer", "minimum": 0.5, "maximum": 3.7}, "int[1,3]"),
        ({"type": "integer", "minimum": 1}, "int"),
        ({"type": "integer", "maximum": 1}, "int"),
        ({"type": "number", "minimum": 0, "maximum": 1.5}, "float[0,1.5]"),
        ({"type": "number", "exclusiveMinimum": 0, "maximum": 1}, "float"),
        ({"type": "number", "minimum": 0}, "float"),
        ({"type": "boolean"}, "bool"),
        ({"type": "null"}, "None"),
        ({"type": "array", "items": {"type": "integer"}}, "list[int]"),
        ({"type": "array", "items": {"enum": ["x", "y"]}}, 'list[Literal["x","y"]]'),
        ({"type": "array", "items": {"type": "array", "items": {"type": "string"}}},
         "list[list[str]]"),
        ({"type": "array", "items": {"anyOf": [{"type": "integer"}, {"type": "null"}]}},
         "list[Optional[int]]"),
        ({"type": "array"}, "list[Any]"),
        ({"type": "array", "items": {}}, "list[Any]"),
        ({"type": "object"}, "object"),
        ({"type": "object", "additionalProperties": {"type": "integer"}}, "object"),
        ({"anyOf": [{"type": "integer"}, {"type": "null"}]}, "Optional[int]"),
        ({"oneOf": [{"type": "null"}, {"type": "string"}]}, "Optional[str]"),
        ({"type": ["integer", "null"]}, "Optional[int]"),
        ({"type": ["integer", "null"], "minimum": 0, "maximum": 3}, "Optional[int[0,3]]"),
        ({"anyOf": [{"type": "integer"}, {"type": "string"}]}, None),
        ({"type": ["integer", "string"]}, None),
        ({"anyOf": [{"type": "object"}, _BUY]}, "object"),
        ({"anyOf": [{"type": "integer"}, {}]}, None),
        ({"allOf": [{"type": "integer"}]}, "int"),
        ({"allOf": [{"type": "integer"}, {"minimum": 0}]}, None),
        ({}, None),
        ({"not": {"type": "string"}}, None),
        ({"type": "array", "prefixItems": [{"type": "integer"}]}, None),
        ({"minimum": 0}, None),
        ({"type": "integer", "title": "N", "description": "d", "default": 1,
          "examples": [1], "multipleOf": 2}, "int"),
        ({"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 3,
          "uniqueItems": True}, "list[str]"),
    ],
)
def test_schema_translates_to_label(schema: dict[str, Any], label: str | None) -> None:
    assert _label(schema) == label


def test_integer_bounds_count_exclusive_endpoints() -> None:
    typ = type_from_json_schema(
        {"type": "integer", "minimum": 1, "exclusiveMaximum": 10}, closed=True
    )
    assert typ == Type(kind=ValueKind.INTEGER, bounds=(1, 9))


def test_list_kinds_follow_the_type_resolver() -> None:
    def kind(items: dict[str, Any]) -> ValueKind:
        return type_from_json_schema({"type": "array", "items": items}, closed=True).kind

    assert kind({"type": "string"}) is ValueKind.LIST_STRING
    assert kind({"enum": ["a"]}) is ValueKind.LIST_STRING
    assert kind({"type": "boolean"}) is ValueKind.LIST_BOOLEAN
    assert kind({"type": "number"}) is ValueKind.LIST_NUMBER
    assert kind({"type": "object"}) is ValueKind.LIST_OBJECT
    assert kind({"type": "object", "properties": {"a": {"type": "integer"}}}) is (
        ValueKind.LIST_OBJECT
    )
    assert kind({"type": "array"}) is ValueKind.LIST_ANY
    assert kind({"type": "null"}) is ValueKind.LIST_ANY
    untyped = type_from_json_schema({"type": "array"}, closed=True)
    assert untyped == Type.scalar(ValueKind.LIST_ANY)


_OBJECT = {
    "type": "object",
    "properties": {
        "a": {"anyOf": [{"type": "integer"}, {"type": "null"}], "default": None},
        "b": {"type": "array", "items": {"type": "string"}},
        "c": {"enum": ["x", "y"], "type": "string"},
        "d": {"type": "integer"},
    },
    "required": ["a", "b", "c"],
}


def test_open_object_with_properties_is_plain_object() -> None:
    assert type_from_json_schema(_OBJECT, closed=False) == Type.scalar(ValueKind.OBJECT)


@pytest.mark.parametrize(
    ("schema", "closed"),
    [(_OBJECT, True), ({**_OBJECT, "additionalProperties": False}, False)],
)
def test_closed_object_is_a_record(schema: dict[str, Any], closed: bool) -> None:
    typ = type_from_json_schema(schema, closed=closed)
    assert typ is not None
    assert typ.kind is ValueKind.OBJECT
    assert {name: canonical_type_label(field) for name, field in typ.fields.items()} == {
        "a": "Optional[int]",
        "b": "list[str]",
        "c": 'Literal["x","y"]',
        "d": "int",
    }
    # `a` is nullable and `d` is not listed in JSON `required`.
    assert typ.required == frozenset({"b", "c"})


def test_closed_object_with_an_untranslatable_property_has_no_type() -> None:
    schema = {"type": "object", "properties": {"a": {}}, "required": ["a"]}
    assert type_from_json_schema(schema, closed=True) is None


def test_open_object_with_an_untranslatable_property_is_plain_object() -> None:
    schema = {"type": "object", "properties": {"a": {}}}
    assert type_from_json_schema(schema, closed=False) == Type.scalar(ValueKind.OBJECT)


def test_nested_objects_follow_the_direction() -> None:
    schema = {
        "type": "object",
        "properties": {"inner": {"type": "object", "properties": {"x": {"type": "integer"}}}},
        "required": ["inner"],
        "additionalProperties": False,
    }
    assert _label(schema, closed=True) == '{"inner":{"x":int}}'
    assert _label(schema, closed=False) == '{"inner":object}'


def test_tagged_union_of_defs_is_a_union_of_records_when_closed() -> None:
    typ = type_from_json_schema(_TAGGED_UNION, closed=True)
    assert typ is not None
    assert typ.variants is not None
    assert [canonical_type_label(member) for member in typ.variants] == [
        '{"action":Literal["buy"],"qty":int}',
        '{"action":Literal["sell"],"price":float}',
    ]
    assert all(member.required == frozenset(member.fields) for member in typ.variants)


def test_tagged_union_of_defs_is_plain_object_when_open() -> None:
    assert type_from_json_schema(_TAGGED_UNION, closed=False) == Type.scalar(
        ValueKind.OBJECT
    )


def test_nullable_union_of_records() -> None:
    schema = {**_TAGGED_UNION, "anyOf": [*_TAGGED_UNION["anyOf"], {"type": "null"}]}
    typ = type_from_json_schema(schema, closed=True)
    assert typ is not None
    assert typ.nullable
    assert typ.variants is not None
    assert not any(member.nullable for member in typ.variants)


def test_definitions_refs_resolve_against_root() -> None:
    root = {"definitions": {"N": {"type": "integer"}}, "type": "array"}
    assert type_from_json_schema({"$ref": "#/definitions/N"}, closed=True, root=root) == (
        Type.scalar(ValueKind.INTEGER)
    )


def test_self_referential_ref_has_no_type() -> None:
    schema = {
        "$defs": {
            "Node": {
                "type": "object",
                "properties": {"child": {"$ref": "#/$defs/Node"}},
                "additionalProperties": False,
            }
        },
        "$ref": "#/$defs/Node",
    }
    assert type_from_json_schema(schema, closed=True) is None


@pytest.mark.parametrize("ref", ["#/$defs/Missing", "https://example.com/s.json", "#"])
def test_unresolvable_ref_has_no_type(ref: str) -> None:
    assert type_from_json_schema({"$defs": {}, "$ref": ref}, closed=True) is None


def test_an_invalid_shape_has_no_type() -> None:
    # An empty property name is not a valid record field.
    schema = {"type": "object", "properties": {"": {"type": "integer"}}}
    assert type_from_json_schema(schema, closed=True) is None


def test_open_input_never_reports_extra_fields_as_disjoint() -> None:
    record = Type(
        kind=ValueKind.OBJECT,
        fields={"a": Type.scalar(ValueKind.INTEGER), "b": Type.scalar(ValueKind.INTEGER)},
        required=frozenset({"a", "b"}),
    )
    declared = type_from_json_schema(
        {"type": "object", "properties": {"a": {"type": "integer"}}}, closed=False
    )
    assert declared is not None
    assert type_relation(record, declared) != "disjoint"


@pytest.mark.parametrize(
    "extra",
    [
        {"additionalProperties": True},
        {"additionalProperties": {"type": "integer"}},
        {"patternProperties": {"^x_": {"type": "integer"}}},
        {"required": ["a", "z"]},
    ],
)
def test_closed_object_that_admits_other_keys_is_plain_object(extra: dict[str, Any]) -> None:
    schema = {"type": "object", "properties": {"a": {"type": "integer"}}, **extra}
    assert type_from_json_schema(schema, closed=True) == Type.scalar(ValueKind.OBJECT)
