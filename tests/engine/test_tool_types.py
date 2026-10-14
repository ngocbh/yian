"""A tool's JSON Schema translates into a yian `Type` that is never narrower than it.

The cases are raw JSON Schema dicts, so the same rules cover pydantic-built
tools and MCP tools that only publish a schema.
"""

from __future__ import annotations

import dataclasses
import datetime
from typing import Annotated, Any, Literal, Optional, Union

import pytest
from langchain_core.tools import BaseTool, InjectedToolArg, StructuredTool, tool
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    SkipValidation,
    field_validator,
)
from pydantic.v1 import BaseModel as BaseModelV1
from typing_extensions import TypedDict

from yian.compile.validation import type_relation
from yian.tools import TOOL_REGISTRY, ToolSet, register_tool, registered_tools
from yian.tools.types import ToolTypes, tool_types, type_from_json_schema
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
        ({"enum": ["b", "a"]}, 'Literal(a, b)'),
        ({"enum": ["a", "b"], "type": "string"}, 'Literal(a, b)'),
        ({"const": "buy"}, 'Literal(buy)'),
        ({"enum": [1, 2], "type": "integer"}, "int"),
        ({"const": 3, "type": "number"}, "float"),
        ({"enum": [1, 2]}, None),
        ({"type": "integer"}, "int"),
        ({"type": "number"}, "float"),
        ({"type": "integer", "minimum": 1, "maximum": 10}, "int(1..10)"),
        ({"type": "integer", "minimum": 1, "exclusiveMaximum": 10}, "int(1..9)"),
        ({"type": "integer", "exclusiveMinimum": 0, "maximum": 5}, "int(1..5)"),
        ({"type": "integer", "minimum": 0.5, "maximum": 3.7}, "int(1..3)"),
        ({"type": "integer", "minimum": 1}, "int"),
        ({"type": "integer", "maximum": 1}, "int"),
        ({"type": "number", "minimum": 0, "maximum": 1.5}, "float(0..1.5)"),
        ({"type": "number", "exclusiveMinimum": 0, "maximum": 1}, "float"),
        ({"type": "number", "minimum": 0}, "float"),
        ({"type": "boolean"}, "bool"),
        ({"type": "null"}, "None"),
        ({"type": "array", "items": {"type": "integer"}}, "list(int)"),
        ({"type": "array", "items": {"enum": ["x", "y"]}}, 'list(Literal(x, y))'),
        ({"type": "array", "items": {"type": "array", "items": {"type": "string"}}},
         "list(list(str))"),
        ({"type": "array", "items": {"anyOf": [{"type": "integer"}, {"type": "null"}]}},
         "list(Optional(int))"),
        ({"type": "array"}, "list(Any)"),
        ({"type": "array", "items": {}}, "list(Any)"),
        ({"type": "object"}, "object"),
        ({"type": "object", "additionalProperties": {"type": "integer"}}, "object"),
        ({"anyOf": [{"type": "integer"}, {"type": "null"}]}, "Optional(int)"),
        ({"oneOf": [{"type": "null"}, {"type": "string"}]}, "Optional(str)"),
        ({"type": ["integer", "null"]}, "Optional(int)"),
        ({"type": ["integer", "null"], "minimum": 0, "maximum": 3}, "Optional(int(0..3))"),
        ({"anyOf": [{"type": "integer"}, {"type": "string"}]}, None),
        ({"type": ["integer", "string"]}, None),
        ({"anyOf": [{"type": "object"}, _BUY]}, "object"),
        ({"anyOf": [{"type": "integer"}, {}]}, None),
        ({"anyOf": [{"type": "object"}, {"type": "integer"}]}, None),
        ({"type": ["object", "string"]}, None),
        ({"anyOf": [_BUY, {"type": "string"}]}, None),
        ({"allOf": [{"type": "integer"}]}, "int"),
        ({"allOf": [{"type": "integer"}, {"minimum": 0}]}, None),
        ({}, None),
        ({"not": {"type": "string"}}, None),
        ({"type": "string", "not": {"const": "x"}}, "str"),
        ({"type": "array", "prefixItems": [{"type": "integer"}]}, "list(Any)"),
        ({"type": "array", "prefixItems": [{"type": "integer"}], "items": {"type": "integer"}},
         "list(Any)"),
        ({"minimum": 0}, None),
        ({"type": "integer", "title": "N", "description": "d", "default": 1,
          "examples": [1], "multipleOf": 2}, "int"),
        ({"type": "string", "minLength": 1, "maxLength": 9}, "str"),
        ({"anyOf": [_BUY, _SELL], "discriminator": {"propertyName": "action"}},
         'Union({"action":Literal(buy),"qty":int},{"action":Literal(sell),"price":float})'),
        ({"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 3,
          "uniqueItems": True}, "list(str)"),
    ],
)
def test_schema_translates_to_label(schema: dict[str, Any], label: str | None) -> None:
    assert _label(schema) == label


def test_list_kinds_follow_the_type_resolver() -> None:
    def kind(items: dict[str, Any]) -> ValueKind:
        return type_from_json_schema({"type": "array", "items": items}, closed=True).kind

    assert kind({"enum": ["a"]}) is ValueKind.LIST_STRING
    assert kind({"type": "object", "properties": {"a": {"type": "integer"}}}) is (
        ValueKind.LIST_OBJECT
    )


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
        "a": "Optional(int)",
        "b": "list(str)",
        "c": 'Literal(x, y)',
        "d": "int",
    }
    # `a` is nullable and `d` is not listed in JSON `required`.
    assert typ.required == frozenset({"b", "c"})


def test_closed_object_with_an_untranslatable_property_is_plain_object() -> None:
    schema = {"type": "object", "properties": {"a": {}}, "required": ["a"]}
    assert type_from_json_schema(schema, closed=True) == Type.scalar(ValueKind.OBJECT)


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
        '{"action":Literal(buy),"qty":int}',
        '{"action":Literal(sell),"price":float}',
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


def test_record_with_a_self_referential_field_is_plain_object() -> None:
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
    assert type_from_json_schema(schema, closed=True) == Type.scalar(ValueKind.OBJECT)


def test_ref_to_itself_has_no_type() -> None:
    schema = {"$defs": {"A": {"$ref": "#/$defs/A"}}, "$ref": "#/$defs/A"}
    assert type_from_json_schema(schema, closed=True) is None


@pytest.mark.parametrize("ref", ["#/$defs/Missing", "https://example.com/s.json", "#"])
def test_unresolvable_ref_has_no_type(ref: str) -> None:
    assert type_from_json_schema({"$defs": {}, "$ref": ref}, closed=True) is None


def test_an_invalid_shape_has_no_type() -> None:
    # An empty property name is not a valid record field.
    schema = {"type": "object", "properties": {"": {"type": "integer"}}}
    assert type_from_json_schema(schema, closed=True) is None


class _Inner(BaseModel):
    x: int


class _WithUnion(BaseModel):
    v: Union[_Inner, str]


def _field_schema(model: type[BaseModel], name: str) -> dict[str, Any]:
    """Return one field's pydantic-generated schema, keeping its `$defs`."""
    schema = model.model_json_schema()
    return {"$defs": schema.get("$defs", {}), **schema["properties"][name]}


_INT = Type.scalar(ValueKind.INTEGER)


@pytest.mark.parametrize(
    ("schema", "closed", "inferred"),
    [
        (
            {"type": "object", "properties": {"a": {"type": "integer"}}},
            False,
            Type(
                kind=ValueKind.OBJECT,
                fields={"a": _INT, "b": _INT},
                required=frozenset({"a", "b"}),
            ),
        ),
        (
            {"type": "object", "properties": {"a": {}}, "additionalProperties": False},
            True,
            Type(kind=ValueKind.OBJECT, fields={"a": _INT}, required=frozenset({"a"})),
        ),
        (
            {"type": "object", "properties": {"a": {"type": "integer"}, "d": {"type": "integer"}},
             "required": ["a"]},
            True,
            Type(kind=ValueKind.OBJECT, fields={"a": _INT}, required=frozenset({"a"})),
        ),
        (
            _TAGGED_UNION,
            False,
            Type(
                kind=ValueKind.OBJECT,
                fields={"action": Type(kind=ValueKind.STRING, tags=frozenset({"buy"})),
                        "qty": _INT, "note": Type.scalar(ValueKind.STRING)},
                required=frozenset({"action", "qty", "note"}),
            ),
        ),
        ({"anyOf": [{"type": "integer"}, {"type": "null"}]}, True, _INT),
        ({"type": "string", "format": "date"}, True, Type.scalar(ValueKind.DATE)),
        (
            {"type": "array", "items": {"type": "number"}},
            True,
            Type.scalar(ValueKind.LIST_INTEGER),
        ),
        ({"type": "array", "prefixItems": [{"type": "integer"}]}, True,
         Type.scalar(ValueKind.LIST_INTEGER)),
        ({"type": "integer"}, True, Type.scalar(ValueKind.NUMBER)),
        ({"type": "integer", "minimum": 0, "maximum": 9}, True, Type.scalar(ValueKind.NUMBER)),
    ],
)
def test_translation_never_reports_an_accepted_value_as_disjoint(
    schema: dict[str, Any], closed: bool, inferred: Type
) -> None:
    declared = type_from_json_schema(schema, closed=closed)
    assert declared is not None
    assert type_relation(inferred, declared) != "disjoint"


@pytest.mark.parametrize("closed", [False, True])
def test_pydantic_union_of_a_model_and_a_scalar_has_no_type(closed: bool) -> None:
    assert type_from_json_schema(_field_schema(_WithUnion, "v"), closed=closed) is None


@pytest.mark.parametrize(
    ("schema", "expected"),
    [
        ({"type": "integer", "minimum": 10**400, "maximum": 10**401},
         f"int({10**400}..{10**401})"),
        ({"type": "number", "minimum": 0, "maximum": 10**400}, "float"),
        ({"type": "integer", "minimum": 0, "exclusiveMaximum": 10**400},
         f"int(0..{10**400 - 1})"),
    ],
    ids=["integer", "number", "exclusive-integer"],
)
def test_huge_bounds_never_raise(schema: dict[str, Any], expected: str) -> None:
    assert _label(schema) == expected


def test_malformed_required_is_plain_object() -> None:
    schema = {
        "type": "object",
        "properties": {"a": {"type": "integer"}},
        "required": [{"x": 1}],
    }
    assert type_from_json_schema(schema, closed=True) == Type.scalar(ValueKind.OBJECT)


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


# --------------------------------------------------------------------------- #
# Declared tool types
# --------------------------------------------------------------------------- #


@pytest.fixture
def restore_tool_registry():
    """Keep process-global registrations isolated between tests."""
    original = dict(TOOL_REGISTRY)
    yield
    TOOL_REGISTRY.clear()
    TOOL_REGISTRY.update(original)


class Filled(TypedDict):
    type: Literal["filled"]
    qty: int


def place(sym: str, qty: Annotated[int, Field(ge=1)], mode: Literal["a", "b"] = "a") -> Filled:
    """Place an order."""
    return {"type": "filled", "qty": qty}


_FILLED = Type(
    kind=ValueKind.OBJECT,
    fields={
        "type": Type(kind=ValueKind.STRING, tags=frozenset({"filled"})),
        "qty": Type.scalar(ValueKind.INTEGER),
    },
    required=frozenset({"type", "qty"}),
)


def _input_labels(types: ToolTypes) -> dict[str, str | None]:
    assert types.inputs is not None
    return {
        name: None if typ is None else canonical_type_label(typ)
        for name, typ in types.inputs.items()
    }


def _registered_place() -> BaseTool:
    register_tool("place")(place)
    return TOOL_REGISTRY["place"]


@pytest.mark.parametrize(
    "build",
    [
        _registered_place,
        lambda: ToolSet({"place": place}).resolve(["place"])[0],
        lambda: ToolSet({"g": [place]}).resolve(["place"])[0],
    ],
    ids=["register_tool", "mapping-entry", "group-item"],
)
def test_function_tool_types(build, restore_tool_registry) -> None:
    types = tool_types(build())

    assert _input_labels(types) == {"sym": "str", "qty": "int", "mode": 'Literal(a, b)'}
    assert types.required == frozenset({"sym", "qty"})
    assert types.output == _FILLED


def test_registered_tools_copy_keeps_the_output_type(restore_tool_registry) -> None:
    register_tool("place")(place)
    tool = ToolSet(registered_tools()).resolve(["place"])[0]

    assert tool_types(tool).output == _FILLED


def _returns_none(x: int) -> None:
    """Return nothing."""


def _unannotated(x: int):
    """Return without an annotation."""
    return x


def _returns_any(x: int) -> Any:
    """Return anything."""
    return x


def _returns_unresolvable(x: int) -> "NoSuchName":  # noqa: F821
    """Return a type that does not resolve."""
    return x


class Model(BaseModel):
    a: int


class WithModel(TypedDict):
    m: Model


@dataclasses.dataclass
class Point:
    x: int


def _returns_model(x: int) -> Model:
    """Return a model."""
    return Model(a=x)


def _returns_model_list(x: int) -> list[Model]:
    """Return models."""
    return [Model(a=x)]


def _returns_typed_dict_with_model(x: int) -> WithModel:
    """Return a record holding a model."""
    return {"m": Model(a=x)}


def _returns_dataclass(x: int) -> Point:
    """Return a dataclass."""
    return Point(x)


def _returns_set(x: int) -> set[int]:
    """Return a set."""
    return {x}


def _returns_date(x: int) -> datetime.date:
    """Return a date."""
    return datetime.date.today()


class _XArgs(BaseModel):
    x: int


def _output(fn) -> Type | None:
    # An explicit schema keeps langchain from resolving the return annotation.
    return tool_types(StructuredTool.from_function(fn, args_schema=_XArgs)).output


def test_explicit_none_return_is_none_kind() -> None:
    assert _output(_returns_none) == Type.scalar(ValueKind.NONE)


@pytest.mark.parametrize(
    "fn",
    [
        _unannotated,
        _returns_any,
        _returns_unresolvable,
        _returns_model,
        _returns_model_list,
        _returns_typed_dict_with_model,
        _returns_dataclass,
        _returns_set,
        _returns_date,
    ],
)
def test_return_without_a_json_like_type_has_no_output(fn) -> None:
    assert _output(fn) is None


def _check(x: str) -> str:
    return x


class Nested(BaseModel):
    model_config = ConfigDict(extra="forbid")
    a: int

    @field_validator("a", mode="before")
    @classmethod
    def _coerce(cls, value: Any) -> Any:
        return int(value)


def _validated(
    bounded: Annotated[int, Field(ge=1, le=9)],
    before: list[Annotated[str, BeforeValidator(_check)]],
    nested: Nested,
    skipped: Annotated[int, SkipValidation],
    maybe: Optional[int],
    when: datetime.datetime,
) -> int:
    """Take inputs with validators."""
    return bounded


def test_inputs_with_custom_validation_are_unchecked() -> None:
    types = tool_types(StructuredTool.from_function(_validated))

    assert _input_labels(types) == {
        "bounded": "int(1..9)",
        "before": None,
        "nested": None,
        "skipped": None,
        "maybe": "Optional(int)",
        "when": "str",
    }
    # `Optional[int]` with no default is still required.
    assert types.required == frozenset(types.inputs)


class _StrOrInt:
    """A custom type whose JSON Schema says `str` but whose Python validation takes an int too."""

    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: Any) -> Any:
        from pydantic_core import core_schema

        return core_schema.json_or_python_schema(
            json_schema=core_schema.str_schema(),
            python_schema=core_schema.union_schema(
                [core_schema.str_schema(), core_schema.int_schema()]
            ),
        )


def _json_or_python(q: _StrOrInt) -> str:
    """Take a value whose JSON and Python validation differ."""
    return str(q)


def test_input_with_separate_python_validation_is_unchecked() -> None:
    types = tool_types(StructuredTool.from_function(_json_or_python))

    assert _input_labels(types) == {"q": None}


def test_too_deeply_nested_input_schema_is_unchecked() -> None:
    deep: dict[str, Any] = {"type": "integer"}
    for _ in range(5000):
        deep = {"type": "array", "items": deep}

    class _DeepTool(BaseTool):
        name: str = "deep"
        description: str = "A tool with a very deep JSON Schema."
        args_schema: dict = {"type": "object", "properties": {}}

        def _run(self, **kwargs: Any) -> None:
            return None

    deep_tool = _DeepTool()
    # Set after construction: the BaseTool constructor deep-copies its fields.
    deep_tool.args_schema = {
        "type": "object",
        "properties": {"x": deep, "y": {"type": "string"}},
    }
    types = tool_types(deep_tool)

    assert _input_labels(types) == {"x": None, "y": "str"}


class _NoFuncTool(BaseTool):
    name: str = "no_func"
    description: str = "A tool with no wrapped function."

    def _run(self, value: int) -> int:
        return value


def test_base_tool_without_a_function_has_no_output() -> None:
    types = tool_types(_NoFuncTool())

    assert types.output is None


def _double(x: int) -> list[int]:
    """Double a number."""
    return [x, x]


def test_host_built_structured_tool_reads_the_function_return() -> None:
    assert tool_types(StructuredTool.from_function(_double)).output == Type(
        kind=ValueKind.LIST_INTEGER, element=Type.scalar(ValueKind.INTEGER)
    )


def _ask_handler(**request: Any) -> Any:
    return "yes"


def test_injected_arguments_are_not_inputs() -> None:
    tool = ToolSet({"ask_user": _ask_handler}).resolve(["ask_user"])[0]
    types = tool_types(tool)

    assert _input_labels(types) == {"question": "str", "options": None}
    assert types.required == frozenset({"question"})


class _LegacyArgs(BaseModelV1):
    value: int
    note: Optional[str] = None


def _legacy(value: int, note: Optional[str] = None) -> int:
    """Use a pydantic v1 schema."""
    return value


def test_pydantic_v1_schema_reads_names_only() -> None:
    tool = StructuredTool.from_function(_legacy, name="legacy", args_schema=_LegacyArgs)
    types = tool_types(tool)

    assert types.inputs == {"value": None, "note": None}
    assert types.required == frozenset({"value"})


class _JsonSchemaTool(BaseTool):
    name: str = "json_schema_tool"
    description: str = "Increment one integer."
    args_schema: dict[str, Any] = {
        "type": "object",
        "$defs": {"Mode": {"enum": ["x", "y"], "type": "string"}},
        "properties": {
            "value": {"type": "integer", "minimum": 0, "maximum": 5},
            "mode": {"$ref": "#/$defs/Mode"},
        },
        "required": ["value"],
        "additionalProperties": False,
    }

    def _run(self, value: int, mode: str = "x") -> int:
        return value + 1


def test_json_schema_dict_tool_inputs_are_translated() -> None:
    types = tool_types(_JsonSchemaTool())

    assert _input_labels(types) == {"value": "int(0..5)", "mode": 'Literal(x, y)'}
    assert types.required == frozenset({"value"})
    assert types.output is None


class Plain:
    """A class with no schema."""


class _PlainArgs(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    sym: str
    thing: Plain


def _with_plain(sym: str, thing: Plain) -> str:
    """Take an arbitrary object."""
    return sym


def test_input_with_an_arbitrary_type_is_unchecked() -> None:
    tool = StructuredTool.from_function(_with_plain, args_schema=_PlainArgs)
    types = tool_types(tool)

    assert _input_labels(types) == {"sym": "str", "thing": None}
    assert types.required == frozenset({"sym", "thing"})
    assert types.output == Type.scalar(ValueKind.STRING)


async def _fetch(x: int) -> list[str]:
    """Fetch names."""
    return [str(x)]


def test_async_tool_output_is_read_from_the_coroutine() -> None:
    built = StructuredTool.from_function(coroutine=_fetch)

    assert built.func is None
    assert tool_types(built).output == Type(
        kind=ValueKind.LIST_STRING, element=Type.scalar(ValueKind.STRING)
    )


@tool(response_format="content_and_artifact")
def _with_artifact(q: str) -> tuple[str, list[int]]:
    """Answer and attach the raw rows."""
    return q, [1]


def test_artifact_tool_has_no_output() -> None:
    # The function returns (content, artifact) but invoking the tool gives only
    # the content.
    types = tool_types(_with_artifact)

    assert types.output is None
    assert _input_labels(types) == {"q": "str"}


class _UnresolvedArgs(BaseModel):
    x: "NoSuchArgType"  # noqa: F821


def _takes_unresolved(x: Any) -> int:
    """Take an argument whose type does not resolve."""
    return 1


def test_unreadable_argument_schema_has_no_inputs() -> None:
    built = StructuredTool.from_function(
        _takes_unresolved, name="unresolved", args_schema=_UnresolvedArgs
    )
    types = tool_types(built)

    assert types.inputs is None
    assert types.required == frozenset()
    assert types.output == Type.scalar(ValueKind.INTEGER)
