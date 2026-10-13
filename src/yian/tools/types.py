"""Read a tool's declared input and return types as yian `Type`s.

The translation is never narrower than what the schema accepts; it may widen.
That keeps a run-start check against the result from reporting a binding as
disjoint when the tool would accept it. A schema with no yian equivalent
translates to `None`, which means "do not check".

Knows about: typesys, pydantic, langchain-core, and the Python standard library.
Never imports: nodes, compile, compose, runtime, the run boundary, or CLI.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from math import ceil, floor, isfinite
from typing import Annotated, Any, get_type_hints

from langchain_core.tools import BaseTool
from pydantic import SkipValidation, TypeAdapter

from yian.typesys.values import (
    Type,
    ValueKind,
    canonical_union_order,
    list_kind_for_element,
)

_SCALAR_KINDS = {
    "string": ValueKind.STRING,
    "integer": ValueKind.INTEGER,
    "number": ValueKind.NUMBER,
    "boolean": ValueKind.BOOLEAN,
    "null": ValueKind.NONE,
}

_REF_PREFIXES = ("#/$defs/", "#/definitions/")


def type_from_json_schema(
    schema: Mapping[str, Any],
    *,
    closed: bool,
    root: Mapping[str, Any] | None = None,
) -> Type | None:
    """Translate a JSON Schema into a yian `Type` that accepts at least what it accepts.

    Args:
        schema (`Mapping[str, Any]`):
            The JSON Schema to translate.
        closed (`bool`):
            Whether an object schema with `properties` describes only those keys.
            Use `False` for tool inputs, which accept extra keys, and `True` for
            tool outputs. An object with `additionalProperties: false` is closed
            either way.
        root (`Mapping[str, Any] | None`, *optional*, defaults to `None`):
            The document `$ref` pointers resolve against. `None` uses `schema`.

    Returns:
        `Type | None`:
            The translated type, or `None` when the schema has no yian equivalent.

    Example:
        `{"anyOf": [{"type": "integer", "minimum": 1, "exclusiveMaximum": 10},
        {"type": "null"}]}` gives `Optional[int[1,9]]` either way. An object with
        `properties` gives a record when `closed=True` and plain `object` when
        `closed=False`.
    """
    typ = _translate(schema, closed=closed, root=schema if root is None else root, refs=())
    if typ is None:
        return None
    try:
        typ.validate_shape()
    except (TypeError, ValueError):
        return None
    return typ


def _translate(
    schema: Any,
    *,
    closed: bool,
    root: Mapping[str, Any],
    refs: tuple[str, ...],
) -> Type | None:
    """Translate one schema node; `refs` holds the `$ref`s being resolved above it."""
    if not isinstance(schema, Mapping):
        return None

    def sub(node: Any, more_refs: tuple[str, ...] = ()) -> Type | None:
        return _translate(node, closed=closed, root=root, refs=refs + more_refs)

    if "$ref" in schema:
        ref = schema["$ref"]
        if ref in refs:
            return None
        target = _resolve_ref(ref, root)
        return None if target is None else sub(target, (ref,))

    if "allOf" in schema:
        members = schema["allOf"]
        if isinstance(members, list) and len(members) == 1:
            return sub(members[0])
        return None

    for keyword in ("anyOf", "oneOf"):
        if keyword in schema:
            members = schema[keyword]
            if not isinstance(members, list) or not members:
                return None
            return _union([sub(member) for member in members])

    tags = _string_tags(schema)
    if tags is not None:
        return Type(kind=ValueKind.STRING, tags=tags)

    json_type = schema.get("type")
    if isinstance(json_type, list):
        if not json_type:
            return None
        return _union([sub({**schema, "type": member}) for member in json_type])
    if json_type == "array":
        items = schema.get("items")
        # Tuple positions have no yian spelling; `list[Any]` accepts every tuple.
        element = None if "prefixItems" in schema or not isinstance(items, Mapping) else sub(items)
        if element is None:
            return Type.scalar(ValueKind.LIST_ANY)
        return Type(kind=list_kind_for_element(element), element=element)
    if json_type == "object":
        return _object(schema, closed=closed, sub=sub)
    kind = _SCALAR_KINDS.get(json_type) if isinstance(json_type, str) else None
    if kind is None:
        return None
    if kind is ValueKind.INTEGER or kind is ValueKind.NUMBER:
        return Type(kind=kind, bounds=_bounds(schema, kind))
    return Type.scalar(kind)


def _resolve_ref(ref: Any, root: Mapping[str, Any]) -> Any:
    """Return the `$defs` / `definitions` entry `ref` points at, else `None`."""
    if not isinstance(ref, str):
        return None
    for prefix in _REF_PREFIXES:
        if ref.startswith(prefix):
            defs = root.get(prefix[2:-1])
            name = ref[len(prefix):].replace("~1", "/").replace("~0", "~")
            return defs.get(name) if isinstance(defs, Mapping) else None
    return None


def _string_tags(schema: Mapping[str, Any]) -> frozenset[str] | None:
    """Return the tags of a string `const` or `enum`, else `None`."""
    if "const" in schema:
        values = [schema["const"]]
    elif isinstance(schema.get("enum"), list):
        values = schema["enum"]
    else:
        return None
    # A non-string or empty value has no `Literal` spelling; falling back to
    # `type` widens instead.
    if not values or any(type(value) is not str or not value for value in values):
        return None
    return frozenset(values)


def _bounds(
    schema: Mapping[str, Any], kind: ValueKind
) -> tuple[int | float, int | float] | None:
    """Return inclusive bounds when both ends are known, else `None`."""

    def endpoint(keyword: str) -> int | float | None:
        value = schema.get(keyword)
        if type(value) is int or (type(value) is float and isfinite(value)):
            return value
        return None

    low, high = endpoint("minimum"), endpoint("maximum")
    if kind is ValueKind.INTEGER:
        lows = [] if low is None else [ceil(low)]
        highs = [] if high is None else [floor(high)]
        exclusive_low, exclusive_high = endpoint("exclusiveMinimum"), endpoint("exclusiveMaximum")
        if exclusive_low is not None:
            lows.append(floor(exclusive_low) + 1)
        if exclusive_high is not None:
            highs.append(ceil(exclusive_high) - 1)
        if not lows or not highs:
            return None
        bounds = (max(lows), min(highs))
    else:
        if low is None or high is None:
            return None
        # Float bounds must survive a float round trip; dropping them widens.
        try:
            if float(low) != low or float(high) != high:
                return None
        except OverflowError:
            return None
        bounds = (low, high)
    return bounds if bounds[0] <= bounds[1] else None


def _object(
    schema: Mapping[str, Any],
    *,
    closed: bool,
    sub: Callable[[Any], Type | None],
) -> Type:
    """Translate an object schema to a record, or to plain `object` when open.

    A property with no yian type also gives plain `object`, which still rejects
    non-objects.
    """
    plain = Type.scalar(ValueKind.OBJECT)
    properties = schema.get("properties")
    if not isinstance(properties, Mapping):
        return plain
    extra = schema.get("additionalProperties")
    if extra is not False and not (closed and extra is None):
        return plain
    if schema.get("patternProperties"):
        return plain
    required = schema.get("required", [])
    if (
        not isinstance(required, list)
        or any(type(name) is not str for name in required)
        or not set(required) <= properties.keys()
    ):
        # A required key outside `properties` would be rejected by the record.
        return plain
    fields = {}
    for name, prop in properties.items():
        field = sub(prop)
        if field is None:
            return plain
        fields[name] = field
    return Type(
        kind=ValueKind.OBJECT,
        fields=fields,
        required=frozenset(name for name in required if not fields[name].nullable),
    )


def _union(members: list[Type | None]) -> Type | None:
    """Combine `anyOf` members into one type, or `None` when yian has no spelling."""
    if any(member is None for member in members):
        return None
    nullable = any(member.nullable for member in members)
    others: list[Type] = []
    for member in members:
        if member == Type.scalar(ValueKind.NONE):
            nullable = True
            continue
        for part in member.variants or (member,):
            part = replace(part, nullable=False)
            if part not in others:
                others.append(part)
    if not others:
        return Type.scalar(ValueKind.NONE)
    if len(others) == 1:
        return replace(others[0], nullable=nullable)
    if any(other.kind is not ValueKind.OBJECT for other in others):
        return None
    if any(other.fields is None for other in others):
        return Type(kind=ValueKind.OBJECT, nullable=nullable)
    return Type(
        kind=ValueKind.OBJECT,
        variants=canonical_union_order(others),
        nullable=nullable,
    )


# Validators that run on the raw input can accept values the JSON Schema
# does not describe. A `json-or-python` schema validates Python input with a
# branch the JSON Schema doesn't show.
_RAW_INPUT_VALIDATORS = frozenset(
    {"function-before", "function-wrap", "function-plain", "json-or-python"}
)

# Core-schema node types of values that wrap as yian values. Anything else
# (models, dataclasses, sets, dates, enums, custom validators) is not a JSON-like
# value, so a node cannot output it.
_JSON_LIKE_CORE_TYPES = frozenset(
    {
        "any",
        "str",
        "int",
        "float",
        "bool",
        "none",
        "literal",
        "list",
        "tuple",
        "dict",
        "typed-dict",
        "typed-dict-field",
        "union",
        "tagged-union",
        "nullable",
        "default",
        "definitions",
        "definition-ref",
    }
)


@dataclass(frozen=True)
class ToolTypes:
    """The types a tool declares for its inputs and its return value.

    Args:
        inputs (`Mapping[str, Type | None] | None`):
            The model-visible inputs by name. `None` when the argument schema
            cannot be read, so no input is checked. A `None` value means that
            input is not checked.
        required (`frozenset[str]`):
            The names of the inputs a call must pass.
        output (`Type | None`):
            The return type, or `None` when it is not checked.
    """

    inputs: Mapping[str, Type | None] | None
    required: frozenset[str]
    output: Type | None


def tool_types(tool: BaseTool) -> ToolTypes:
    """Read the declared input and return types of `tool`.

    Inputs come from the model-visible argument schema, so injected arguments are
    left out. The return type comes from the annotation of the wrapped function.

    Args:
        tool (`BaseTool`):
            The tool to read.

    Returns:
        `ToolTypes`:
            The input types, required input names, and return type. Each part is
            `None` where it cannot be read without risking a narrower type than
            the tool accepts or returns.
    """
    try:
        schema = tool.tool_call_schema
    except Exception:
        # Building the schema can fail, for example on an unresolved forward
        # reference; the tool's inputs are then unchecked.
        return ToolTypes(inputs=None, required=frozenset(), output=_output_type(tool))
    inputs, required = _input_types(schema)
    return ToolTypes(inputs=inputs, required=required, output=_output_type(tool))


def _input_types(schema: Any) -> tuple[dict[str, Type | None] | None, frozenset[str]]:
    """Return the input types and required names of a tool argument schema."""
    if isinstance(schema, Mapping):
        properties = schema.get("properties")
        if not isinstance(properties, Mapping):
            return None, frozenset()
        required = schema.get("required", [])
        names = required if isinstance(required, list) else []
        return (
            {
                name: _input_type_from_json_schema(prop, root=schema)
                for name, prop in properties.items()
            },
            frozenset(name for name in names if isinstance(name, str)),
        )
    fields = getattr(schema, "model_fields", None)
    if isinstance(fields, Mapping):
        return (
            {name: _field_type(field) for name, field in fields.items()},
            frozenset(name for name, field in fields.items() if field.is_required()),
        )
    fields = getattr(schema, "__fields__", None)
    if isinstance(fields, Mapping):
        # Pydantic v1 schemas drop nullability, so their types are not read.
        return (
            dict.fromkeys(fields),
            frozenset(name for name, field in fields.items() if field.required),
        )
    return None, frozenset()


def _field_type(field: Any) -> Type | None:
    """Return the input type of one pydantic v2 field, or `None` to leave it unchecked."""
    metadata = list(field.metadata)
    if any(item is SkipValidation or isinstance(item, SkipValidation) for item in metadata):
        return None
    try:
        # The metadata keeps constraints such as `ge`/`le` that the bare
        # annotation loses.
        hint = Annotated[(field.annotation, *metadata)] if metadata else field.annotation
        adapter = TypeAdapter(hint)
        core_schema = adapter.core_schema
        json_schema = adapter.json_schema()
    except Exception:
        return None
    if _core_types(core_schema) & _RAW_INPUT_VALIDATORS:
        return None
    return _input_type_from_json_schema(json_schema)


def _input_type_from_json_schema(
    schema: Any, *, root: Mapping[str, Any] | None = None
) -> Type | None:
    """Translate one input's JSON Schema, or return `None` when it nests too deeply."""
    try:
        return type_from_json_schema(schema, closed=False, root=root)
    except RecursionError:
        # A host or MCP schema can nest deeper than the translator recurses;
        # such an input is left unchecked rather than failing the run.
        return None


def _output_type(tool: BaseTool) -> Type | None:
    """Return the JSON-like return type of the function `tool` wraps, else `None`."""
    # Any other format splits the function's return value, and invoking the tool
    # gives only part of it.
    if tool.response_format != "content":
        return None
    fn = getattr(tool, "func", None) or getattr(tool, "coroutine", None)
    if not callable(fn):
        return None
    try:
        hints = get_type_hints(fn)
    except Exception:
        return None
    if "return" not in hints:
        return None
    try:
        adapter = TypeAdapter(hints["return"])
        core_schema = adapter.core_schema
        json_schema = adapter.json_schema()
    except Exception:
        return None
    if not _core_types(core_schema) <= _JSON_LIKE_CORE_TYPES:
        return None
    return type_from_json_schema(json_schema, closed=True)


def _core_types(schema: Any) -> set[str]:
    """Return every node `type` in a pydantic core schema, serializers included."""
    found: set[str] = set()
    stack = [schema]
    while stack:
        node = stack.pop()
        if isinstance(node, Mapping):
            # A field named `type` maps to a schema, not a string.
            if isinstance(node.get("type"), str):
                found.add(node["type"])
            stack.extend(node.values())
        elif isinstance(node, (list, tuple)):
            stack.extend(node)
    return found
