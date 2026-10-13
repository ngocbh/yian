"""Translate a tool's JSON Schema into a yian `Type`.

The translation is never narrower than what the schema accepts; it may widen.
That keeps a run-start check against the result from reporting a binding as
disjoint when the tool would accept it. A schema with no yian equivalent
translates to `None`, which means "do not check".

Knows about: typesys and the Python standard library.
Never imports: nodes, compile, compose, runtime, the run boundary, or CLI.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace
from math import ceil, floor, isfinite
from typing import Any

from yian.typesys.values import Type, ValueKind, canonical_union_order

_SCALAR_KINDS = {
    "string": ValueKind.STRING,
    "integer": ValueKind.INTEGER,
    "number": ValueKind.NUMBER,
    "boolean": ValueKind.BOOLEAN,
    "null": ValueKind.NONE,
}

_LIST_KIND_FOR_ELEMENT = {
    kind.element_kind: kind for kind in ValueKind if kind.is_list() and kind.element_kind
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
    if not isinstance(schema, Mapping) or "not" in schema or "prefixItems" in schema:
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
        element = sub(items) if isinstance(items, Mapping) else None
        if element is None:
            return Type.scalar(ValueKind.LIST_ANY)
        return Type(kind=_list_kind(element), element=element)
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


def _list_kind(element: Type) -> ValueKind:
    """Return the list storage kind the type resolver picks for `element`."""
    if element.tags is not None:
        return ValueKind.LIST_STRING
    if element.fields is not None or element.variants is not None:
        return ValueKind.LIST_OBJECT
    return _LIST_KIND_FOR_ELEMENT.get(element.kind, ValueKind.LIST_ANY)


def _object(
    schema: Mapping[str, Any],
    *,
    closed: bool,
    sub: Callable[[Any], Type | None],
) -> Type | None:
    """Translate an object schema to a record, or to plain `object` when open."""
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
            return None
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
