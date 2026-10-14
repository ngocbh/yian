"""Typed runtime value system.

Every value flowing through a flow run is wrapped in a `TypedValue`: a frozen,
self-describing container carrying its `ValueKind`. The tag is what makes the
variable pool serialize losslessly — on a JSON round-trip the discriminated
`AnyValue` union decodes each value back into the right subclass, so an int
stays an int and a float stays a float regardless of JSON's number ambiguity.
That losslessness lets the append-only run log persist typed values and rebuild
the same pool during replay.

Trimmed from graphon's `variables/` to the load-bearing core: scalars, lists,
object — plus a reserved `FILE` placeholder that is never auto-inferred and not
yet exposed to flow authors (so adding real file handling later is additive,
not a schema migration).

This module's only package-internal dependencies are the stdlib-only date/datetime
grammar in `yian.typesys.temporal` and the stdlib-only `yian.json_codec`; otherwise it
is the leaf.
"""

import json
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, is_dataclass, replace
from dataclasses import fields as _dataclass_fields
from enum import Enum
from math import inf, isfinite
from typing import Annotated, Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator

import yian.json_codec as json_codec
from yian.typesys.temporal import parse_date, parse_datetime


class TypeCheckError(ValueError):
    """A value violates a declared type at one source-neutral structural path.

    ``path`` is relative to the checked type: strings name record fields and
    integers are actual list indices. Type-language errors and root mismatches
    use the empty path. Positional ``ValueError`` arguments remain unrestricted
    so adding structural context does not change existing messages or ``args``.

    Args:
        *args (`object`): Positional arguments retained by ``ValueError``.
        path (`tuple[str | int, ...]`, *optional*, defaults to ``()``):
            Immutable record-field/list-index path to the rejected value.
    """

    def __init__(
        self,
        *args: object,
        path: tuple[str | int, ...] = (),
    ) -> None:
        """Retain one validated contract-relative path without changing args."""
        if type(path) is not tuple:
            raise TypeError("type-check error path must be a tuple")
        for segment in path:
            if type(segment) not in (str, int):
                raise TypeError(
                    "type-check error path segment must be an exact str or int, "
                    f"got {type(segment).__name__}"
                )
        super().__init__(*args)
        self.path = path


def value_repr(value: Any) -> str:
    """Return `repr(value)`, or a placeholder naming its type when it can't be rendered.

    Args:
        value (`Any`):
            The value to show in a message.

    Returns:
        `str`: ``repr(value)``, or ``<{type} too large to display>``.
    """
    try:
        return repr(value)
    except ValueError:
        # repr of an int past Python's str-digit limit raises, at any nesting depth.
        return f"<{type(value).__name__} too large to display>"


def reject_nonfinite(value: Any, where: str) -> None:
    """Raise if a value contains a number that is not a yian value: NaN, infinity, or complex.

    Walks `dict` values, `list`/`tuple` items, `TypedValue` payloads, pydantic
    models, and dataclass instances, depth-first and in order. It walks declared
    fields and extras only; a non-str key renders as its repr, or a
    ``<type too large to display>`` placeholder. A container
    already on the current path is skipped; the canonical codecs reject cycles.

    Args:
        value (`Any`):
            The value to check.
        where (`str`):
            Names the boundary for the message, e.g. ``"input 'x'"``.

    Raises:
        `TypeCheckError`:
            Such a number was found; ``path`` locates it and the message reads
            ``<where>[ at <path>]: non-finite number <repr> is not a yian value``, or
            ``... complex number <repr> ...`` for a complex.
    """
    found = _find_nonfinite(value, (), set())
    if found is None:
        return
    path, number = found
    location = f"{where} at {_render_value_path(path)}" if path else where
    label = "complex number" if isinstance(number, complex) else "non-finite number"
    raise TypeCheckError(f"{location}: {label} {number!r} is not a yian value", path=path)


def _find_nonfinite(
    value: Any,
    path: tuple[str | int, ...],
    active: set[int],
) -> tuple[tuple[str | int, ...], float | complex] | None:
    """Return the path to and value of the first NaN, infinity, or complex, or `None`."""
    if isinstance(value, float):
        return None if isfinite(value) else (path, value)
    if isinstance(value, complex):
        return (path, value)
    if id(value) in active:
        return None
    # `TypedValue` subclasses `BaseModel`; unwrap it first so paths gain no `value` segment.
    if isinstance(value, TypedValue):
        children = [(value.to_object(), path)]
    elif isinstance(value, BaseModel):
        children = [(getattr(value, name), path + (name,)) for name in type(value).model_fields]
        children.append((value.model_extra or {}, path))
    elif isinstance(value, dict):
        # `TypeCheckError` paths hold exact `str` segments, so a `str` subclass key is unwrapped.
        children = [
            (item, path + (str.__str__(key) if isinstance(key, str) else value_repr(key),))
            for key, item in value.items()
        ]
    elif isinstance(value, (list, tuple)):
        children = [(item, path + (index,)) for index, item in enumerate(value)]
    elif is_dataclass(value) and not isinstance(value, type):
        children = [
            (getattr(value, field.name), path + (field.name,))
            for field in _dataclass_fields(value)
        ]
    else:
        return None
    active.add(id(value))
    try:
        for child, child_path in children:
            found = _find_nonfinite(child, child_path, active)
            if found is not None:
                return found
        return None
    finally:
        active.discard(id(value))


def _render_value_path(path: tuple[str | int, ...]) -> str:
    """Render a path as ``a[1].b``, quoting a non-identifier key as ``['my key']``."""
    parts = []
    for segment in path:
        if isinstance(segment, int):
            parts.append(f"[{segment}]")
        elif segment.isidentifier():
            parts.append(f".{segment}" if parts else segment)
        else:
            parts.append(f"[{segment!r}]")
    return "".join(parts)


MAX_VALUE_NESTING = 100
"""Deepest nesting a yian value may have.

Each dict, list, tuple, model, or dataclass is one level.
"""


def reject_deep_nesting(value: Any, where: str, *, enclosing_levels: int = 0) -> None:
    """Raise if a value nests deeper than `MAX_VALUE_NESTING` levels.

    Each `dict`, `list`, `tuple`, pydantic model, and dataclass instance is one
    level; a `TypedValue` is unwrapped and adds none. A model's declared fields
    and extras are its direct children. The walk is depth-first and in order. A
    container already on the current path is skipped; the canonical codecs
    reject cycles.

    Args:
        value (`Any`):
            The value to check.
        where (`str`):
            Names the boundary for the message, e.g. ``"output"``.
        enclosing_levels (`int`, *optional*, defaults to `0`):
            Levels counted around the value; -1 lets the value itself go one
            level over, as for a START's input record.

    Raises:
        `TypeCheckError`:
            A container sits past the limit; ``path`` locates the first one and the
            message reads ``<where>[ at <path>]: value nests deeper than <limit>
            levels, the limit for a yian value (each record and list is a level)``.
    """
    limit = MAX_VALUE_NESTING
    while isinstance(value, TypedValue):
        value = value.to_object()
    children = _nesting_children(value)
    if children is None:
        return
    if 1 + enclosing_levels > limit:
        raise _too_deep(where, (), limit)
    # An explicit stack, not recursion, so the walk does not depend on the caller's stack
    # depth or the configured recursion limit.
    # Each frame holds its children iterator, the container, and its key in the parent;
    # the path is built only when rejecting.
    stack: list[tuple[Iterator[tuple[Any, Any]], Any, Any]] = [(iter(children), value, None)]
    active = {id(value)}
    while stack:
        iterator, container, _ = stack[-1]
        entry = next(iterator, None)
        if entry is None:
            stack.pop()
            active.discard(id(container))
            continue
        key, child = entry
        if child is None or type(child) in _NESTING_SCALARS:
            continue
        while isinstance(child, TypedValue):
            child = child.to_object()
        if id(child) in active:
            continue
        grandchildren = _nesting_children(child)
        if grandchildren is None:
            continue
        if len(stack) + 1 + enclosing_levels > limit:
            path = tuple(
                _nesting_segment(parent, frame_key)
                for (_, parent, _), (_, _, frame_key) in zip(stack, stack[1:])
            ) + (_nesting_segment(container, key),)
            raise _too_deep(where, path, limit)
        stack.append((iter(grandchildren), child, key))
        active.add(id(child))


_NESTING_SCALARS = frozenset({str, int, float, bool})


def _nesting_children(value: Any) -> Iterable[tuple[Any, Any]] | None:
    """Return a container's ``(key, child)`` pairs lazily, or `None` for a leaf."""
    if isinstance(value, BaseModel):
        children = [(name, getattr(value, name)) for name in type(value).model_fields]
        children.extend((value.model_extra or {}).items())
        return children
    if isinstance(value, dict):
        return value.items()
    if isinstance(value, (list, tuple)):
        return enumerate(value)
    if is_dataclass(value) and not isinstance(value, type):
        return [(field.name, getattr(value, field.name)) for field in _dataclass_fields(value)]
    return None


def _nesting_segment(container: Any, key: Any) -> str | int:
    """Return the path segment for `key` within `container`."""
    if isinstance(container, dict):
        # `TypeCheckError` paths hold exact `str` segments, so a `str` subclass key is unwrapped.
        return str.__str__(key) if isinstance(key, str) else repr(key)
    return key


def _too_deep(where: str, path: tuple[str | int, ...], limit: int) -> TypeCheckError:
    """Build the error for a container past the nesting limit at `path`."""
    location = f"{where} at {_render_value_path(path)}" if path else where
    return TypeCheckError(
        f"{location}: value nests deeper than {limit} levels, "
        "the limit for a yian value (each record and list is a level)",
        path=path,
    )


# --------------------------------------------------------------------------- #
# Type vocabulary
# --------------------------------------------------------------------------- #


class ValueKind(str, Enum):
    """Closed type vocabulary for runtime values.

    `.value` IS the one Python-surface vocabulary: it is simultaneously the serialized
    discriminator tag, the author-written `type:` token, and the displayed name (so there
    is no second "engine name" to translate — `str`, never `string`).

    Every kind but `ANY` is a storage tag and wire discriminator. `ANY` is type-only: no
    carrier stores it, and an `Any` value persists under its natural kind (`str`,
    `object`, `list[Any]`, ...). The list kinds keep the bracket storage spelling
    (`list[str]`); type labels spell them `list(str)`.
    """

    NONE = "None"
    STRING = "str"
    INTEGER = "int"
    NUMBER = "float"
    DATE = "date"  # grammar in yian.typesys.temporal; stored as str; never auto-inferred
    DATETIME = "datetime"  # grammar in yian.typesys.temporal; stored as str; never auto-inferred
    BOOLEAN = "bool"
    OBJECT = "object"
    FILE = "file"  # reserved: never auto-inferred, not authorable yet
    ANY = "Any"  # type-only: any non-null JSON value, stored in its natural carrier

    LIST_ANY = "list[Any]"
    LIST_STRING = "list[str]"
    LIST_INTEGER = "list[int]"
    LIST_NUMBER = "list[float]"
    LIST_BOOLEAN = "list[bool]"
    LIST_OBJECT = "list[object]"

    def is_list(self) -> bool:
        """Whether this kind is one of the six `list[...]` kinds."""
        return self in _LIST_ELEMENT_KIND

    @property
    def element_kind(self) -> Optional["ValueKind"]:
        """Scalar element type for a list type (`None` for `LIST_ANY`/scalars)."""
        return _LIST_ELEMENT_KIND.get(self)


# --------------------------------------------------------------------------- #
# Reserved FILE placeholder
# --------------------------------------------------------------------------- #


class FileRef(BaseModel):
    """A minimal, opaque file reference — reserved placeholder.

    Carries only what is needed to round-trip through the typed-value wire
    representation used by run records. Storage, transfer, size and markdown
    rendering are deliberately absent until file handling is actually built.
    """

    model_config = ConfigDict(frozen=True)

    uri: str
    mime: Optional[str] = None
    name: Optional[str] = None


# --------------------------------------------------------------------------- #
# TypedValue subclasses (one per type, each pinning a Literal discriminator)
# --------------------------------------------------------------------------- #


class TypedValue(BaseModel):
    """Abstract base. Use a concrete subclass; construct via `build_value`."""

    model_config = ConfigDict(frozen=True)

    kind: ValueKind
    value: Any

    def to_object(self) -> Any:
        """Return the plain Python value (what expressions and node code see)."""
        return self.value

    @property
    def text(self) -> str:
        """Render the value as a string (empty string for `None`)."""
        return "" if self.value is None else str(self.value)


class NoneValue(TypedValue):
    kind: Literal[ValueKind.NONE] = ValueKind.NONE
    value: None = None

    @property
    def text(self) -> str:
        return ""


class StringValue(TypedValue):
    kind: Literal[ValueKind.STRING] = ValueKind.STRING
    value: str


class IntegerValue(TypedValue):
    kind: Literal[ValueKind.INTEGER] = ValueKind.INTEGER
    value: int


class NumberValue(TypedValue):
    kind: Literal[ValueKind.NUMBER] = ValueKind.NUMBER
    value: float


class DateValue(TypedValue):
    kind: Literal[ValueKind.DATE] = ValueKind.DATE
    value: str  # "YYYY-MM-DD"; see yian.typesys.temporal

    @field_validator("value")
    @classmethod
    def _iso(cls, v: str) -> str:
        parse_date(v)
        return v


class DateTimeValue(TypedValue):
    kind: Literal[ValueKind.DATETIME] = ValueKind.DATETIME
    value: str  # "YYYY-MM-DD[T| ]HH:MM[:SS[.f]][Z|±HH:MM]"; see yian.typesys.temporal

    @field_validator("value")
    @classmethod
    def _iso(cls, v: str) -> str:
        parse_datetime(v)
        return v


class BooleanValue(TypedValue):
    kind: Literal[ValueKind.BOOLEAN] = ValueKind.BOOLEAN
    value: bool


class ObjectValue(TypedValue):
    kind: Literal[ValueKind.OBJECT] = ValueKind.OBJECT
    value: dict[str, Any]


class FileValue(TypedValue):
    kind: Literal[ValueKind.FILE] = ValueKind.FILE
    value: FileRef


class ListAnyValue(TypedValue):
    kind: Literal[ValueKind.LIST_ANY] = ValueKind.LIST_ANY
    value: list[Any]


# Refined `Type.element` contracts decide whether None is legal before these
# storage carriers are built; their annotations must preserve that accepted value.
class ListStringValue(TypedValue):
    kind: Literal[ValueKind.LIST_STRING] = ValueKind.LIST_STRING
    value: list[Optional[str]]


class ListIntegerValue(TypedValue):
    kind: Literal[ValueKind.LIST_INTEGER] = ValueKind.LIST_INTEGER
    value: list[Optional[int]]


class ListNumberValue(TypedValue):
    kind: Literal[ValueKind.LIST_NUMBER] = ValueKind.LIST_NUMBER
    value: list[Optional[float]]


class ListBooleanValue(TypedValue):
    kind: Literal[ValueKind.LIST_BOOLEAN] = ValueKind.LIST_BOOLEAN
    value: list[Optional[bool]]


class ListObjectValue(TypedValue):
    kind: Literal[ValueKind.LIST_OBJECT] = ValueKind.LIST_OBJECT
    value: list[Optional[dict[str, Any]]]


# Discriminated union — `kind` selects the subclass on validate, which is
# what gives lossless decode of an arbitrary persisted typed value.
AnyValue = Annotated[
    Union[
        NoneValue,
        StringValue,
        IntegerValue,
        NumberValue,
        DateValue,
        DateTimeValue,
        BooleanValue,
        ObjectValue,
        FileValue,
        ListAnyValue,
        ListStringValue,
        ListIntegerValue,
        ListNumberValue,
        ListBooleanValue,
        ListObjectValue,
    ],
    Field(discriminator="kind"),
]

# Module-level adapter so callers can round-trip a bare typed value losslessly.
ANY_VALUE_ADAPTER: TypeAdapter = TypeAdapter(AnyValue)


# --------------------------------------------------------------------------- #
# Lookup tables
# --------------------------------------------------------------------------- #

_LIST_ELEMENT_KIND: dict[ValueKind, Optional[ValueKind]] = {
    ValueKind.LIST_ANY: None,
    ValueKind.LIST_STRING: ValueKind.STRING,
    ValueKind.LIST_INTEGER: ValueKind.INTEGER,
    ValueKind.LIST_NUMBER: ValueKind.NUMBER,
    ValueKind.LIST_BOOLEAN: ValueKind.BOOLEAN,
    ValueKind.LIST_OBJECT: ValueKind.OBJECT,
}

_SCALAR_VALUE_CLASS: dict[ValueKind, type[TypedValue]] = {
    ValueKind.NONE: NoneValue,
    ValueKind.STRING: StringValue,
    ValueKind.INTEGER: IntegerValue,
    ValueKind.NUMBER: NumberValue,
    ValueKind.DATE: DateValue,
    ValueKind.DATETIME: DateTimeValue,
    ValueKind.BOOLEAN: BooleanValue,
    ValueKind.OBJECT: ObjectValue,
    ValueKind.FILE: FileValue,
}

_LIST_VALUE_CLASS: dict[ValueKind, type[TypedValue]] = {
    ValueKind.LIST_ANY: ListAnyValue,
    ValueKind.LIST_STRING: ListStringValue,
    ValueKind.LIST_INTEGER: ListIntegerValue,
    ValueKind.LIST_NUMBER: ListNumberValue,
    ValueKind.LIST_BOOLEAN: ListBooleanValue,
    ValueKind.LIST_OBJECT: ListObjectValue,
}

# --------------------------------------------------------------------------- #
# Structural type (records / variants / typed lists)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Type:
    """Resolved runtime structure of a declared Type — drives the write-boundary check.

    `kind` is the storage tag (what the value persists as). The optional fields
    refine validation beyond the bare tag:
      - record:  kind=OBJECT,   fields={name: Type}, required={names}
      - variant: kind=STRING,   tags={labels}   (a `Literal[...]` of strings)
      - list:    kind=LIST_*,   element=<element Type>
      - number:  kind=INTEGER/NUMBER, bounds=(inclusive minimum, maximum);
                 `None` for an open end
      - union:   kind=OBJECT,   variants=(record member, ...)   (a `Union[...]` of records)
    A refined specialized list uses the element's matching storage tag;
    `LIST_ANY` is the generic carrier and may refine any element type.
    A plain scalar / flat list uses only `kind` (see `Type.scalar`).
    Despite the name, `variants` holds union members, not the string tags of a
    "variant" above. Members are non-nullable records; a nullable union sets
    `nullable` on the union itself.
    """

    kind: ValueKind
    fields: Optional[dict[str, "Type"]] = None
    required: Optional[frozenset[str]] = None
    tags: Optional[frozenset[str]] = None
    element: Optional["Type"] = None
    nullable: bool = False  # an Optional[X] slot — accepts None (present-None or absent)
    bounds: tuple[int | float | None, int | float | None] | None = None
    variants: Optional[tuple["Type", ...]] = None

    @classmethod
    def scalar(cls, seg: ValueKind) -> "Type":
        """Build a plain `Type` carrying only a storage tag, without refinements."""
        return cls(kind=seg)

    def validate_shape(self) -> None:
        """Require recursively well-formed structural type metadata.

        This is an execution-boundary invariant check, not value validation. It
        catches a corrupted or manually assembled type before value rejection is
        classified against that contract.

        Raises:
            `TypeError`:
                A type field or nested carrier has the wrong runtime type.
            `ValueError`:
                Refinements conflict, cannot be canonicalized, name an absent
                record field, use incompatible storage, or form a cycle.
        """
        _validate_type_shape(self, active=set(), validated=set())


def _validate_type_shape(
    typ: Type,
    *,
    active: set[int],
    validated: set[int],
) -> None:
    """Validate one recursive type graph while permitting shared child objects."""
    if not isinstance(typ, Type):
        raise TypeError("nested type must be a Type")
    identity = id(typ)
    if identity in active:
        raise ValueError("type refinements must form an acyclic graph")
    if identity in validated:
        return
    if not isinstance(typ.kind, ValueKind):
        raise TypeError("type kind must be a ValueKind")
    if type(typ.nullable) is not bool:
        raise TypeError("type nullable must be a bool")
    if typ.bounds is not None:
        _validate_numeric_bounds(typ.kind, typ.bounds)

    active.add(identity)
    try:
        fields = typ.fields
        if fields is not None:
            if type(fields) is not dict:
                raise TypeError("type fields must be a dict or None")
            if typ.kind is not ValueKind.OBJECT:
                raise ValueError("only object types may declare record fields")
            for name, field_type in fields.items():
                if type(name) is not str or not name:
                    raise TypeError("type field names must be non-empty strings")
                _validate_type_metadata_string(name)
                _validate_type_shape(
                    field_type,
                    active=active,
                    validated=validated,
                )

        required = typ.required
        if required is not None:
            if type(required) is not frozenset or any(
                type(name) is not str or not name for name in required
            ):
                raise TypeError(
                    "type required fields must be a frozenset of non-empty strings or None"
                )
            for name in required:
                _validate_type_metadata_string(name)
            if fields is None:
                raise ValueError("type required fields need a record field mapping")
            unknown = required - fields.keys()
            if unknown:
                raise ValueError(
                    f"type requires unknown record fields: {sorted(unknown)!r}"
                )

        tags = typ.tags
        if tags is not None:
            if type(tags) is not frozenset or any(
                type(tag) is not str or not tag for tag in tags
            ):
                raise TypeError(
                    "type tags must be a frozenset of non-empty strings or None"
                )
            for tag in tags:
                _validate_type_metadata_string(tag)
            if typ.kind is not ValueKind.STRING:
                raise ValueError("only string types may declare variant tags")

        element = typ.element
        if element is not None:
            if not typ.kind.is_list():
                raise ValueError("only list types may declare an element type")
            _validate_type_shape(
                element,
                active=active,
                validated=validated,
            )
            expected_kind = list_kind_for_element(element)
            if typ.kind is not ValueKind.LIST_ANY and typ.kind is not expected_kind:
                raise ValueError(
                    f"list storage kind {typ.kind.value!r} does not match "
                    f"element type; expected {expected_kind.value!r}"
                )

        variants = typ.variants
        if variants is not None:
            if type(variants) is not tuple:
                raise TypeError("type variants must be a tuple or None")
            if typ.kind is not ValueKind.OBJECT:
                raise ValueError("only object types may declare union variants")
            if len(variants) < 2:
                raise ValueError("a union needs at least two members")
            for member in variants:
                _validate_type_shape(member, active=active, validated=validated)
                if member.fields is None:
                    raise ValueError("every union member must be a record type")
                if member.nullable:
                    raise ValueError("union members must not be nullable")

        refinements = sum(
            value is not None
            for value in (fields, tags, element, typ.bounds, variants)
        )
        if refinements > 1:
            raise ValueError("a type may declare only one structural refinement")
    finally:
        active.remove(identity)
    validated.add(identity)


def _validate_numeric_bounds(
    kind: ValueKind, bounds: tuple[int | float | None, int | float | None]
) -> None:
    """Require inclusive bounds with at least one endpoint; `None` is an open end.

    Present endpoints must be exact numbers, lossless and finite for float storage,
    and ordered when both are present.
    """
    if kind is not ValueKind.INTEGER and kind is not ValueKind.NUMBER:
        raise ValueError("only int and float types may declare bounds")
    if type(bounds) is not tuple or len(bounds) != 2:
        raise TypeError("type bounds must be a two-item tuple")
    endpoints = [endpoint for endpoint in bounds if endpoint is not None]
    if not endpoints:
        raise ValueError("type bounds need at least one endpoint")
    allowed = (int,) if kind is ValueKind.INTEGER else (int, float)
    if any(type(endpoint) not in allowed for endpoint in endpoints):
        raise TypeError(f"{kind.value} bounds require exact numeric endpoints, excluding bool")
    if kind is ValueKind.NUMBER:
        for endpoint in endpoints:
            try:
                represented = float(endpoint)
            except OverflowError as exc:
                raise ValueError("float bounds require finite endpoints") from exc
            if not isfinite(represented):
                raise ValueError("float bounds require finite endpoints")
            if represented != endpoint:
                raise ValueError("float bounds require lossless float endpoints")
    if len(endpoints) == 2 and bounds[0] > bounds[1]:
        raise ValueError("type bounds must be ordered from minimum to maximum")


def numeric_interval(typ: Type) -> tuple[int | float, int | float]:
    """Return a numeric type's inclusive interval, with open ends as infinities.

    Args:
        typ (`Type`):
            The type whose `bounds` to read.

    Returns:
        `tuple[int | float, int | float]`: ``(low, high)``; an open end, or absent
        bounds, is ``-math.inf`` / ``math.inf``.

    Example:
        ```python
        numeric_interval(Type(ValueKind.INTEGER, bounds=(0, None)))  # (0, inf)
        ```
    """
    low, high = typ.bounds if typ.bounds is not None else (None, None)
    return (-inf if low is None else low, inf if high is None else high)


def _validate_type_metadata_string(value: str) -> None:
    """Require a type-owned name to be representable in canonical JSON."""
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError("type metadata contains a Unicode surrogate") from exc


def list_kind_for_element(element: Type) -> ValueKind:
    """Return the list storage kind the type resolver picks for an element type.

    A `Literal` element stores as `list[str]`, a record or union element as
    `list[object]`, a scalar element as its own list kind, and anything else as
    `list[Any]`.

    Args:
        element (`Type`): The list's element type.

    Returns:
        `ValueKind`: The `list[...]` storage kind.
    """
    semantic_kind = element.kind
    if element.tags is not None:
        semantic_kind = ValueKind.STRING
    elif element.fields is not None or element.variants is not None:
        semantic_kind = ValueKind.OBJECT
    return next(
        (
            list_kind
            for list_kind, element_kind in _LIST_ELEMENT_KIND.items()
            if element_kind is semantic_kind
        ),
        ValueKind.LIST_ANY,
    )


# --------------------------------------------------------------------------- #
# Type labels
# --------------------------------------------------------------------------- #


def canonical_type_label(typ: Type) -> str:
    """Render one deterministic, unambiguous label for a structural type.

    Record keys use compact JSON quoting; variant tags print bare when they read back
    as names, else JSON-quoted (`Literal(go, "no go")`). Structurally refined
    lists render from their element; a valid flat-list type renders from its
    storage kind (`list(str)`). A union lists its members in canonical order.

    Args:
        typ (`Type`): The resolved structural type to label.

    Returns:
        `str`: A compact canonical type expression, e.g. `Optional(int(1..12))`.

    """
    return _labels(typ)[0]


def _labels(typ: Type) -> tuple[str, str]:
    """Return `(label, order key)` for `typ`.

    The label is the parenthesized spelling; its variant tags print bare when they read
    back as names, else JSON-quoted. The order key is the same type in the older bracket
    spelling (`Optional[int[1,12]]`, `Literal["go","no_go"]`); union members sort by it, so
    member order, JSON Schema `anyOf` order and manifest fingerprints did not move
    when the spelling changed. Both come from one walk so nested unions stay linear.
    """
    label, key = _canonical_nonnullable_label(typ)
    if typ.nullable:
        return f"Optional({label})", f"Optional[{key}]"
    return label, key


def _canonical_nonnullable_label(typ: Type) -> tuple[str, str]:
    """Return `(label, order key)` for ``typ`` without its outer nullable wrapper."""
    if typ.variants is not None:
        ordered = _ordered_member_labels(typ.variants)
        return (
            "Union(" + ",".join(label for _, label, _ in ordered) + ")",
            "Union[" + ",".join(key for _, _, key in ordered) + "]",
        )

    if typ.fields is not None:
        pairs = [(_json_string(name), _labels(typ.fields[name])) for name in sorted(typ.fields)]
        return (
            "{" + ",".join(f"{name}:{label}" for name, (label, _) in pairs) + "}",
            "{" + ",".join(f"{name}:{key}" for name, (_, key) in pairs) + "}",
        )

    if typ.tags is not None:
        tags = sorted(typ.tags)
        return literal_label(tags), "Literal[" + ",".join(_json_string(tag) for tag in tags) + "]"

    if typ.kind.is_list():
        if typ.element is None:
            return _kind_label(typ.kind), typ.kind.value
        label, key = _labels(typ.element)
        return f"list({label})", f"list[{key}]"

    if typ.bounds is not None:
        _validate_numeric_bounds(typ.kind, typ.bounds)
        # An open end renders empty in the label (`int(0..)`) and as null in the key.
        key_ends = [json_codec.dumps(value, allow_nan=False) for value in typ.bounds]
        label_ends = ["" if value is None else end for value, end in zip(typ.bounds, key_ends)]
        return (
            f"{typ.kind.value}({'..'.join(label_ends)})",
            f"{typ.kind.value}[{','.join(key_ends)}]",
        )

    return typ.kind.value, typ.kind.value


def _kind_label(kind: ValueKind) -> str:
    """Spell a list storage kind as a type for messages, e.g. `list(str)` for `LIST_STRING`.

    `ValueKind` values are stored and sent as they are (`"list[str]"`).
    """
    element = kind.element_kind
    return f"list({'Any' if element is None else element.value})"


def _json_string(value: str) -> str:
    """Encode one record key or variant tag as compact canonical JSON."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


# The type tokenizer's NAME token; a `Literal` member printed bare must match it.
_NAME_PATTERN = r"[^\W\d]\w*"
_NAME_TOKEN = re.compile(_NAME_PATTERN)

# Names the type parser reads as constants, never as `Literal` members.
_BUILTIN_CONSTANTS = ("True", "False", "None")


def literal_label(tags: Iterable[str]) -> str:
    """Spell a `Literal` type from its members, in the given order.

    A member prints bare when it reads back as a name, else JSON-quoted; members are
    separated by `", "`.

    Args:
        tags (`Iterable[str]`):
            The `Literal` members, in the order to print them.

    Returns:
        `str`:
            The `Literal` type text, which `parse_type` reads back.

    Example:
        `literal_label(["go", "no go", "True"])` gives `'Literal(go, "no go", "True")'`.
    """
    return "Literal(" + ", ".join(_literal_member(tag) for tag in tags) + ")"


def _literal_member(tag: str) -> str:
    """Spell one `Literal` member bare when it reads back as a name, else quoted."""
    # The tokenizer's NAME and `str.isidentifier` disagree on a few Unicode characters;
    # a bare member must pass both to read back as the same single name.
    if _NAME_TOKEN.fullmatch(tag) and tag.isidentifier() and tag not in _BUILTIN_CONSTANTS:
        return tag
    return json.dumps(tag, ensure_ascii=False)


# --------------------------------------------------------------------------- #
# Unions of records
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class FieldRead:
    """Every union member has the field; `type` is what a read gives.

    Args:
        type (`Type | None`):
            The field's type across the members, or `None` when the members'
            types differ and the read is untyped.
    """

    type: Optional[Type]


@dataclass(frozen=True)
class FieldPartial:
    """Only some union members have the field.

    Args:
        members (`tuple[Type, ...]`):
            The members that declare the field, in the union's member order.
    """

    members: tuple[Type, ...]


@dataclass(frozen=True)
class FieldAbsent:
    """No union member has the field."""


def _single_tag(typ: Type) -> Optional[str]:
    """Return the one value of a single-value `Literal`, else `None`."""
    if typ.tags is None or len(typ.tags) != 1:
        return None
    return next(iter(typ.tags))


def _members_tag_fields(members: tuple[Type, ...]) -> tuple[str, ...]:
    """Return the tag fields shared by `members`, sorted by name."""
    if not members or any(member.fields is None for member in members):
        return ()
    qualifying = []
    for name in sorted(members[0].fields):
        values = []
        for member in members:
            field_type = member.fields.get(name)
            if (
                field_type is None
                or name not in (member.required or frozenset())
                or field_type.nullable
                or _single_tag(field_type) is None
            ):
                break
            values.append(_single_tag(field_type))
        else:
            if len(set(values)) == len(members):
                qualifying.append(name)
    return tuple(qualifying)


def union_tag_fields(typ: Type) -> tuple[str, ...]:
    """Return the fields that tell a union's members apart, sorted by name.

    A field is a tag when every member declares it as required, not nullable, and
    a single-value `Literal`, and the values are pairwise distinct. Any tag field
    selects the same member.

    Args:
        typ (`Type`):
            The type to inspect. A type that is not a union has no tag fields.

    Returns:
        `tuple[str, ...]`:
            The tag field names, or `()` for an untagged union or a non-union.

    Example:
        A union of `{action: Literal(buy), qty: int}` and
        `{action: Literal(sell), price: float}` gives `("action",)`.
    """
    if typ.variants is None:
        return ()
    return _members_tag_fields(typ.variants)


def canonical_union_order(members: Iterable[Type]) -> tuple[Type, ...]:
    """Return union members in canonical order.

    Members with a tag field sort by the value of the first tag field by name;
    otherwise they sort by their canonical label in the bracket spelling
    (`Optional[int]`), which predates the parenthesized one.

    Args:
        members (`Iterable[Type]`):
            The record members of one union.

    Returns:
        `tuple[Type, ...]`:
            The same members, reordered.
    """
    members = tuple(members)
    tag_fields = _members_tag_fields(members)
    if tag_fields:
        # Tag order reads best in labels and manifests, and needs no member labels.
        field = tag_fields[0]
        return tuple(sorted(members, key=lambda member: _single_tag(member.fields[field])))
    return tuple(member for member, _, _ in _ordered_member_labels(members))


def _ordered_member_labels(members: tuple[Type, ...]) -> list[tuple[Type, str, str]]:
    """Return `(member, label, order key)` triples in canonical order, labelling each member once.

    Labelling once keeps nested untagged unions linear: their order depends on the
    labels, so ordering and rendering share them. Untagged members sort by their
    bracket-spelled order key (see `_labels`).
    """
    labelled = [(member, *_labels(member)) for member in members]
    tag_fields = _members_tag_fields(members)
    if tag_fields:
        field = tag_fields[0]
        return sorted(labelled, key=lambda triple: _single_tag(triple[0].fields[field]))
    return sorted(labelled, key=lambda triple: triple[2])


def _tag_field(typ: Type, field: Optional[str]) -> str:
    """Pick a union's tag field: `field` when given, else the first by name."""
    tag_fields = union_tag_fields(typ)
    if not tag_fields:
        raise ValueError("the type is not a tagged union")
    if field is None:
        return tag_fields[0]
    if field not in tag_fields:
        raise ValueError(f"{field!r} is not a tag field of the union")
    return field


def union_member_for_tag(
    typ: Type,
    tag: Any,
    *,
    field: Optional[str] = None,
) -> Optional[Type]:
    """Return the member of a tagged union whose tag field equals `tag`.

    Args:
        typ (`Type`):
            A tagged union.
        tag (`Any`):
            The tag value to look up. A value that is not one of the tags,
            including a non-string, finds no member.
        field (`str`, *optional*, defaults to `None`):
            The tag field to match on. `None` uses the first tag field by name.

    Returns:
        `Type | None`:
            The matching member, or `None` when no member has that tag.

    Raises:
        `ValueError`:
            `typ` is not a tagged union, or `field` is not one of its tag fields.
    """
    return _member_with_tag(typ.variants, _tag_field(typ, field), tag)


def _member_with_tag(members: tuple[Type, ...], field: str, tag: Any) -> Optional[Type]:
    """Return the member whose single-value tag field `field` equals `tag`."""
    index = _tag_index(members, field, tag)
    return None if index is None else members[index]


def _tag_index(members: tuple[Type, ...], field: str, tag: Any) -> Optional[int]:
    """Return the index of the member whose tag field `field` equals `tag`."""
    if type(tag) is not str:
        return None
    return next(
        (index for index, member in enumerate(members) if tag in member.fields[field].tags),
        None,
    )


def restrict_union(
    typ: Type,
    keep: Iterable[str | Type],
    *,
    field: Optional[str] = None,
) -> Type:
    """Return the part of a union that `keep` selects, keeping its nullability.

    One selected member gives that member record; several give a smaller union
    with its members in canonical order.

    Args:
        typ (`Type`):
            The union to restrict.
        keep (`Iterable[str | Type]`):
            The members to keep, each named by its tag value (tagged unions only)
            or given as a member `Type`. Must not be empty.
        field (`str`, *optional*, defaults to `None`):
            The tag field that tag values refer to. `None` uses the first tag field
            by name.

    Returns:
        `Type`:
            The selected member, or a union of the selected members. Either is
            nullable when `typ` is.

    Raises:
        `TypeError`:
            `keep` is a bare string rather than a collection of tags.
        `ValueError`:
            `typ` is not a union, `keep` is empty, or an item is neither a tag
            nor a member of `typ`.

    Example:
        Restricting `Union[Buy, Hold, Sell]` to `{"buy", "sell"}` gives
        `Union[Buy, Sell]`.
    """
    if isinstance(keep, str):
        raise TypeError("restrict_union keeps a collection of tags, not a bare string")
    if typ.variants is None:
        raise ValueError("only a union can be restricted")
    members = typ.variants
    tag_field: Optional[str] = None
    selected: set[int] = set()
    for item in keep:
        if isinstance(item, Type):
            index = next((i for i, member in enumerate(members) if member == item), None)
            if index is None:
                raise ValueError("restrict_union got a type that is not a union member")
        else:
            if tag_field is None:
                tag_field = _tag_field(typ, field)
            index = _tag_index(members, tag_field, item)
            if index is None:
                raise ValueError(f"{item!r} is not a tag of the union")
        selected.add(index)
    if not selected:
        raise ValueError("restrict_union needs at least one member")
    if len(selected) == 1:
        return replace(members[selected.pop()], nullable=typ.nullable)
    kept = canonical_union_order(members[index] for index in sorted(selected))
    return Type(kind=ValueKind.OBJECT, variants=kept, nullable=typ.nullable)


def union_field(typ: Type, name: str) -> FieldRead | FieldPartial | FieldAbsent:
    """Describe reading field `name` on a value of union type `typ`.

    When every member has the field, the read's type is the members' type if they
    are equal ignoring `nullable`, the merged `Literal` if they are all string
    `Literal`s, and untyped otherwise. It is nullable when any member's field is
    `Optional` or not required.

    Args:
        typ (`Type`):
            A union.
        name (`str`):
            The field name to read.

    Returns:
        `FieldRead | FieldPartial | FieldAbsent`:
            `FieldRead` when every member has the field, `FieldPartial` with the
            members that have it when only some do, and `FieldAbsent` when none do.

    Raises:
        `ValueError`:
            `typ` is not a union.
    """
    if typ.variants is None:
        raise ValueError("union_field needs a union type")
    having = tuple(member for member in typ.variants if name in member.fields)
    if not having:
        return FieldAbsent()
    if len(having) < len(typ.variants):
        return FieldPartial(having)
    field_types = [member.fields[name] for member in having]
    nullable = any(
        field_type.nullable or name not in (member.required or frozenset())
        for member, field_type in zip(having, field_types)
    )
    strict = [replace(field_type, nullable=False) for field_type in field_types]
    if all(field_type == strict[0] for field_type in strict):
        return FieldRead(replace(strict[0], nullable=nullable))
    if all(field_type.tags is not None for field_type in strict):
        merged = frozenset().union(*(field_type.tags for field_type in strict))
        return FieldRead(Type(kind=ValueKind.STRING, tags=merged, nullable=nullable))
    return FieldRead(None)


def _admits_none(typ: Type) -> bool:
    """Whether `None` satisfies `typ`."""
    return typ.nullable or typ.kind is ValueKind.NONE


def _is_plain_object(typ: Type) -> bool:
    """Whether `typ` is `object` with no record or union refinement."""
    return typ.kind is ValueKind.OBJECT and typ.fields is None and typ.variants is None


def _records_overlap(a: Type, b: Type) -> bool:
    """Whether two closed records can match the same value."""
    a_required = a.required or frozenset()
    b_required = b.required or frozenset()
    if not a_required <= b.fields.keys() or not b_required <= a.fields.keys():
        return False
    return all(types_overlap(a.fields[name], b.fields[name]) for name in a_required | b_required)


def _bounds_intersect(a: Type, b: Type) -> bool:
    """Whether two numeric types' inclusive intervals share a point."""
    a_low, a_high = numeric_interval(a)
    b_low, b_high = numeric_interval(b)
    return max(a_low, b_low) <= min(a_high, b_high)


_TEXT_KINDS = frozenset({ValueKind.STRING, ValueKind.DATE, ValueKind.DATETIME})
_NUMERIC_KINDS = frozenset({ValueKind.INTEGER, ValueKind.NUMBER})


def types_overlap(a: Type, b: Type) -> bool:
    """Whether some value could satisfy both `a` and `b`.

    The check is conservative: it may say two types overlap when no value
    satisfies both, but never the reverse. It decides whether an untagged union's
    members can be told apart from the data alone.

    Args:
        a (`Type`):
            One type.
        b (`Type`):
            The other type.

    Returns:
        `bool`:
            `True` when the two types may share a value.

    Example:
        `{action: Literal(buy), qty: int}` and `{action: Literal(sell), price: float}`
        do not overlap; `{conviction: float}` overlaps itself.
    """
    if _admits_none(a) and _admits_none(b):
        return True
    if a.variants is not None:
        return any(types_overlap(member, b) for member in a.variants)
    if b.variants is not None:
        return any(types_overlap(a, member) for member in b.variants)
    if a.fields is not None and b.fields is not None:
        return _records_overlap(a, b)
    if a.kind is ValueKind.OBJECT and b.kind is ValueKind.OBJECT:
        # Only plain `object` remains on at least one side, and it accepts any record.
        return _is_plain_object(a) or _is_plain_object(b)
    if a.kind.is_list() and b.kind.is_list():
        return True
    if a.tags is not None and b.tags is not None:
        return bool(a.tags & b.tags)
    if a.tags is not None or b.tags is not None:
        literal, other = (a, b) if a.tags is not None else (b, a)
        if other.kind is ValueKind.STRING:
            return True
        if other.kind in (ValueKind.DATE, ValueKind.DATETIME):
            return any(_scalar_matches(other.kind, tag) for tag in literal.tags)
        return False
    if a.kind in _NUMERIC_KINDS and b.kind in _NUMERIC_KINDS:
        return _bounds_intersect(a, b)
    if a.kind in _TEXT_KINDS and b.kind in _TEXT_KINDS:
        return a.kind is b.kind or ValueKind.STRING in (a.kind, b.kind)
    return a.kind is b.kind


# --------------------------------------------------------------------------- #
# Scalar type checks / coercion
# --------------------------------------------------------------------------- #


def _infer_scalar_type(value: Any) -> Optional[ValueKind]:
    # bool before int — bool is a subclass of int in Python.
    if value is None:
        return ValueKind.NONE
    if isinstance(value, bool):
        return ValueKind.BOOLEAN
    if isinstance(value, int):
        return ValueKind.INTEGER
    if isinstance(value, float):
        return ValueKind.NUMBER
    if isinstance(value, str):
        return ValueKind.STRING
    if isinstance(value, FileRef):
        return ValueKind.FILE
    if isinstance(value, dict):
        return ValueKind.OBJECT
    return None


_TEMPORAL_PARSERS = {ValueKind.DATE: parse_date, ValueKind.DATETIME: parse_datetime}


def _temporal_error(declared: ValueKind, value: str) -> Optional[str]:
    """Return why `value` is not a valid `declared` string, or `None` when it is.

    Args:
        declared (`ValueKind`):
            `ValueKind.DATE` or `ValueKind.DATETIME`.
        value (`str`):
            The string to check against that kind's grammar.

    Returns:
        `Optional[str]`:
            The grammar's error message, or `None` when `value` parses.
    """
    try:
        _TEMPORAL_PARSERS[declared](value)
    except ValueError as exc:
        return str(exc)
    return None


def _scalar_matches(declared: ValueKind, value: Any) -> bool:
    if declared == ValueKind.NONE:
        return value is None
    if declared == ValueKind.STRING:
        return isinstance(value, str)
    if declared == ValueKind.BOOLEAN:
        return isinstance(value, bool)
    if declared == ValueKind.INTEGER:
        return isinstance(value, int) and not isinstance(value, bool)
    if declared == ValueKind.NUMBER:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if declared in _TEMPORAL_PARSERS:
        return isinstance(value, str) and _temporal_error(declared, value) is None
    if declared == ValueKind.OBJECT:
        return isinstance(value, dict)
    if declared == ValueKind.FILE:
        return isinstance(value, FileRef)
    return False


def _coerce_scalar(declared: ValueKind, value: Any, path: tuple[str | int, ...]) -> Any:
    """Widen an int filling a `float` slot; return any other value unchanged."""
    if declared == ValueKind.NUMBER and isinstance(value, int) and not isinstance(value, bool):
        return _widen_int(value, path)
    return value


def _widen_int(value: int, path: tuple[str | int, ...]) -> float:
    """Widen an int to a float, raising `TypeCheckError` at `path` on overflow."""
    try:
        return float(value)
    except OverflowError:
        # No repr of the int: `str()` of a huge int hits Python's digit limit.
        raise TypeCheckError("int is too large for float", path=path) from None


def _json_value_error(
    value: Any, path: tuple[str | int, ...]
) -> Optional[TypeCheckError]:
    """Return why `value` is not a JSON value, or `None` when it is.

    A JSON value is `None`, a `bool`, `int`, `float` or `str`, a `dict` with only
    `str` keys and JSON values, or a `list`/`tuple` of JSON values. The walk checks
    structure only: nesting depth and non-finite numbers are left to
    `reject_deep_nesting` and `reject_nonfinite` at the boundaries, so a recorded
    value replays. A container inside itself is rejected.

    Args:
        value (`Any`):
            The value to check.
        path (`tuple[str | int, ...]`):
            Field/index path from the contract root to ``value``.

    Returns:
        `Optional[TypeCheckError]`:
            The error for the first bad part, depth-first and in order, with its
            path, or `None` when ``value`` is a JSON value.
    """
    if _is_json_scalar(value):
        return None
    entries = _json_entries(value)
    if entries is None:
        return TypeCheckError(f"{value_repr(value)} is not a JSON value", path=path)
    # An explicit stack, not recursion, so the walk does not depend on the caller's stack
    # depth or the configured recursion limit.
    # Each frame holds its entries iterator, the container, and its key in the parent;
    # the path is built only when rejecting.
    stack: list[tuple[Iterator[tuple[Any, Any]], Any, Any]] = [(entries, value, None)]
    active = {id(value)}

    def path_to(*keys: Any) -> tuple[str | int, ...]:
        frame_keys = [frame_key for _, _, frame_key in stack[1:]]
        # `TypeCheckError` paths hold exact `str` segments, so a `str` subclass key is
        # unwrapped.
        return path + tuple(
            str.__str__(key) if isinstance(key, str) else key for key in (*frame_keys, *keys)
        )

    while stack:
        iterator, container, _ = stack[-1]
        entry = next(iterator, None)
        if entry is None:
            stack.pop()
            active.discard(id(container))
            continue
        key, child = entry
        if isinstance(container, dict) and not isinstance(key, str):
            return TypeCheckError(
                f"object keys must be text, got {value_repr(key)}", path=path_to()
            )
        if _is_json_scalar(child):
            continue
        if id(child) in active:
            return TypeCheckError("value contains itself", path=path_to(key))
        child_entries = _json_entries(child)
        if child_entries is None:
            return TypeCheckError(f"{value_repr(child)} is not a JSON value", path=path_to(key))
        stack.append((child_entries, child, key))
        active.add(id(child))
    return None


def _is_json_scalar(value: Any) -> bool:
    """Whether `value` is `None`, a `bool`, `int`, `float` or `str`."""
    # The exact-type lookup is the fast path; `isinstance` admits subclasses.
    return (
        value is None
        or type(value) in _NESTING_SCALARS
        or isinstance(value, (str, int, float))
    )


def _json_entries(value: Any) -> Optional[Iterator[tuple[Any, Any]]]:
    """Return a `dict`'s items or a `list`/`tuple`'s indexed items, else `None`."""
    if isinstance(value, dict):
        return iter(value.items())
    if isinstance(value, (list, tuple)):
        return enumerate(value)
    return None


def _require_json_value(value: Any, path: tuple[str | int, ...]) -> None:
    """Raise the `_json_value_error` for `value` at `path`, if any."""
    error = _json_value_error(value, path)
    if error is not None:
        raise error


# --------------------------------------------------------------------------- #
# Constructors
# --------------------------------------------------------------------------- #


def build_value(value: Any) -> TypedValue:
    """Wrap a raw Python value in the natural TypedValue, inferring its type.

    `FILE` is never inferred from a dict/str — only an explicit `FileRef`
    produces a `FileValue`. An empty list infers as `LIST_ANY`.

    Args:
        value (`Any`):
            A plain Python value (scalar, `dict`, `list`/`tuple`, or `FileRef`), or an
            already-wrapped `TypedValue` (returned as-is, so the call is idempotent).

    Returns:
        `TypedValue`:
            The value wrapped in the subclass matching its inferred `ValueKind`.

    Raises:
        `TypeCheckError`:
            The value is of a kind that has no `TypedValue` mapping (e.g. a custom object).
    """
    if isinstance(value, TypedValue):
        return value  # idempotent

    scalar = _infer_scalar_type(value)
    if scalar is not None:
        return _SCALAR_VALUE_CLASS[scalar](value=value)

    if isinstance(value, (list, tuple)):
        return _infer_list(list(value))

    raise TypeCheckError(f"cannot wrap value of type {type(value).__name__!r}")


def _infer_list(items: list[Any]) -> TypedValue:
    if not items:
        return ListAnyValue(value=[])
    element_types = {_infer_scalar_type(x) for x in items}
    if None in element_types:
        return ListAnyValue(value=items)
    if element_types == {ValueKind.STRING}:
        return ListStringValue(value=items)
    if element_types == {ValueKind.BOOLEAN}:
        return ListBooleanValue(value=items)
    if element_types == {ValueKind.INTEGER}:
        return ListIntegerValue(value=items)
    if element_types <= {ValueKind.INTEGER, ValueKind.NUMBER}:
        return ListNumberValue(value=[
            _widen_int(x, (index,)) if isinstance(x, int) else float(x)
            for index, x in enumerate(items)
        ])
    if element_types == {ValueKind.OBJECT}:
        return ListObjectValue(value=items)
    return ListAnyValue(value=items)


def build_value_as(declared: "ValueKind | Type", value: Any) -> TypedValue:
    """Wrap `value` as a TypedValue matching `declared`, raising on a type mismatch.

    This is the write-boundary check the variable pool uses against each declared
    output type, so a node returning the wrong type fails loudly at the write rather
    than silently downstream. The only coercion is widening: an `int` fills a `float`
    slot at any depth (a record field, a list item, a union member's field).

    Args:
        declared (`ValueKind | Type`):
            The target type. A bare `ValueKind` is a scalar or flat list; a structural
            `Type` additionally checks records, unions, variants, typed lists, and
            numeric bounds.
        value (`Any`):
            The raw Python value to validate and wrap, or an already-wrapped
            `TypedValue` (unwrapped to its plain value first).

    Returns:
        `TypedValue`:
            The value wrapped in the subclass for `declared`'s storage kind.

    Raises:
        `TypeCheckError`:
            The value does not satisfy `declared` — wrong scalar type, a non-member
            variant tag, an out-of-bounds number, a record with missing/unknown fields,
            a union value with a missing/unknown tag or matching no member, or a bad
            list element.
    """
    typ = Type.scalar(declared) if isinstance(declared, ValueKind) else declared
    if isinstance(value, TypedValue):
        value = value.to_object()
    return _build_for_type(typ, value)


def _build_for_type(
    typ: Type,
    value: Any,
    *,
    path: tuple[str | int, ...] = (),
) -> TypedValue:
    """Validate one value while retaining its absolute contract-relative path.

    Args:
        typ (`Type`): Structural type enforced at this recursive position.
        value (`Any`): Plain value at ``path``.
        path (`tuple[str | int, ...]`, *optional*, defaults to ``()``):
            Field/index path from the checked contract root to ``value``.

    Returns:
        `TypedValue`: Value normalized for ``typ``'s storage kind.

    Raises:
        `TypeCheckError`: Value violates ``typ``; the error retains ``path`` or
            the failing descendant path.
    """
    if typ.bounds is not None:
        _validate_numeric_bounds(typ.kind, typ.bounds)
        low, high = numeric_interval(typ)
        # A one-sided range names the end a value fails; NaN compares false against
        # both ends, so its message claims no direction.
        if _scalar_matches(typ.kind, value) and not low <= value <= high:
            kind = typ.kind.value
            if None not in typ.bounds:
                low_text, high_text = (value_repr(end) for end in typ.bounds)
                message = f"is outside inclusive {kind} bounds ({low_text}, {high_text})"
            elif value < low:
                message = f"is below the inclusive {kind} minimum {value_repr(low)}"
            elif value > high:
                message = f"is above the inclusive {kind} maximum {value_repr(high)}"
            else:
                label = _canonical_nonnullable_label(typ)[0]
                message = f"is outside the inclusive range {label}"
            raise TypeCheckError(f"{value_repr(value)} {message}", path=path)

    # nullable (Optional[X]) — a None value is accepted as NoneValue
    if value is None and typ.nullable:
        return NoneValue()

    if typ.variants is not None:
        return _build_for_union(typ, value, path=path)

    # variant — a tag-constrained string
    if typ.tags is not None:
        if not isinstance(value, str) or value not in typ.tags:
            raise TypeCheckError(
                f"{value_repr(value)} is not a member of variant {sorted(typ.tags)}",
                path=path,
            )
        return StringValue(value=value)

    # record — an object with declared field types (closed: all required, no unknowns)
    if typ.fields is not None:
        if not isinstance(value, dict):
            raise TypeCheckError(
                f"{value_repr(value)} is not an object for a record type",
                path=path,
            )
        required = typ.required or frozenset()
        missing = required - value.keys()
        if missing:
            ordered_missing = sorted(missing)
            raise TypeCheckError(
                f"record missing required fields: {ordered_missing}",
                path=path + (ordered_missing[0],),
            )
        unknown = value.keys() - typ.fields.keys()
        if unknown:
            all_string_keys = all(type(name) is str for name in unknown)
            try:
                ordered_unknown = sorted(unknown)
            except TypeError:
                ordered_unknown = sorted(
                    unknown,
                    key=lambda name: (type(name).__qualname__, value_repr(name)),
                )
            unknown_path = (
                path + (ordered_unknown[0],)
                if all_string_keys
                else path
            )
            shown = ", ".join(value_repr(name) for name in ordered_unknown)
            raise TypeCheckError(
                f"record has unknown fields: [{shown}]",
                path=unknown_path,
            )
        # Key by the declared name: an equal `str` subclass key must not leak into
        # error paths or the stored record.
        declared_names = {name: name for name in typ.fields}
        # Store each field's normalized value, so a nested `float` slot widens an int
        # just as a top-level one does.
        fields = {}
        for fname, fval in value.items():
            name = declared_names[fname]
            fields[name] = _build_for_type(
                typ.fields[name],
                fval,
                path=path + (name,),
            ).to_object()
        return ObjectValue(value=fields)

    # list with a known element typ (incl. List[record])
    if typ.kind.is_list() and typ.element is not None:
        if not isinstance(value, (list, tuple)):
            raise TypeCheckError(
                f"{value_repr(value)} is not a list for declared {_kind_label(typ.kind)}",
                path=path,
            )
        items = [
            _build_for_type(
                typ.element,
                item,
                path=path + (index,),
            ).to_object()
            for index, item in enumerate(value)
        ]
        return _LIST_VALUE_CLASS[typ.kind](value=items)

    if typ.kind is ValueKind.ANY:
        if value is None:
            raise TypeCheckError(
                f"None does not match declared type {typ.kind.value}", path=path
            )
        _require_json_value(value, path)
        if isinstance(value, (list, tuple)):
            return ListAnyValue(value=list(value))
        return _SCALAR_VALUE_CLASS[_infer_scalar_type(value)](value=value)

    # plain scalar / flat list — the original behavior
    return _build_scalar_or_list(typ.kind, value, path=path)


def _build_for_union(
    typ: Type,
    value: Any,
    *,
    path: tuple[str | int, ...],
) -> TypedValue:
    """Validate one value against the union member it belongs to.

    A tagged union selects the member by its first tag field; an untagged union
    takes the first member the value satisfies.

    Args:
        typ (`Type`): Union type enforced at ``path``.
        value (`Any`): Plain, non-`None` value at ``path``.
        path (`tuple[str | int, ...]`): Field/index path from the contract root.

    Returns:
        `TypedValue`: The value stored as an object.

    Raises:
        `TypeCheckError`: The value is not an object, its tag is missing or
            unknown, or it fails the selected member (or every member).
    """
    if not isinstance(value, dict):
        raise TypeCheckError(
            f"{value_repr(value)} is not an object for a union type", path=path
        )
    members = typ.variants
    tag_fields = _members_tag_fields(members)
    if tag_fields:
        field = tag_fields[0]
        if field not in value:
            raise TypeCheckError(
                f"missing tag field {field!r} (expected one of {_tag_values(members, field)})",
                path=path + (field,),
            )
        member = _member_with_tag(members, field, value[field])
        if member is None:
            raise TypeCheckError(
                f"{value_repr(value[field])} is not a tag of the union "
                f"(expected one of {_tag_values(members, field)})",
                path=path + (field,),
            )
        return _build_for_type(member, value, path=path)
    reasons = []
    for member in members:
        try:
            return _build_for_type(member, value, path=path)
        except _NoUnionMember as exc:
            # A nested union contributes its summary, so the message stays bounded.
            reasons.append(f"{canonical_type_label(member)}: {exc.summary}")
        except TypeCheckError as exc:
            reasons.append(f"{canonical_type_label(member)}: {exc}")
    summary = f"value matches no member of {canonical_type_label(typ)}"
    raise _NoUnionMember(summary + ": " + "; ".join(reasons), summary=summary, path=path)


def _tag_values(members: tuple[Type, ...], field: str) -> list[str]:
    """Return the sorted values of one tag field across union members."""
    return sorted(_single_tag(member.fields[field]) for member in members)


class _NoUnionMember(TypeCheckError):
    """A value matches no member of an untagged union.

    Args:
        *args (`object`): Positional arguments retained by ``ValueError``.
        summary (`str`): The message without the per-member reasons.
        path (`tuple[str | int, ...]`, *optional*, defaults to ``()``):
            Path to the rejected value.
    """

    def __init__(
        self,
        *args: object,
        summary: str,
        path: tuple[str | int, ...] = (),
    ) -> None:
        """Keep the summary beside the full message."""
        super().__init__(*args, path=path)
        self.summary = summary


def _build_scalar_or_list(
    declared: ValueKind,
    value: Any,
    *,
    path: tuple[str | int, ...],
) -> TypedValue:
    """Validate one unrefined scalar/list at its contract-relative path.

    Args:
        declared (`ValueKind`): Scalar or flat-list storage kind to enforce.
        value (`Any`): Plain value at ``path``.
        path (`tuple[str | int, ...]`): Field/index path from the contract root.

    Returns:
        `TypedValue`: Value normalized for ``declared``.

    Raises:
        `TypeCheckError`: Value or one flat-list element violates ``declared``.
    """
    if declared.is_list():
        if not isinstance(value, (list, tuple)):
            raise TypeCheckError(
                f"{value_repr(value)} is not a list for declared {_kind_label(declared)}",
                path=path,
            )
        items = list(value)
        element = declared.element_kind
        if element is None:
            _require_json_value(items, path)
        else:
            for index, item in enumerate(items):
                if not _scalar_matches(element, item):
                    raise TypeCheckError(
                        f"list element {value_repr(item)} is not {element.value} "
                        f"(declared {_kind_label(declared)})",
                        path=path + (index,),
                    )
                if element is ValueKind.OBJECT:
                    _require_json_value(item, path + (index,))
            items = [
                _coerce_scalar(element, item, path + (index,))
                for index, item in enumerate(items)
            ]
        return _LIST_VALUE_CLASS[declared](value=items)

    if not _scalar_matches(declared, value):
        message = f"{value_repr(value)} does not match declared type {declared.value}"
        if declared in _TEMPORAL_PARSERS and isinstance(value, str):
            message += f": {_temporal_error(declared, value)}"
        raise TypeCheckError(message, path=path)
    if declared is ValueKind.OBJECT:
        _require_json_value(value, path)
    return _SCALAR_VALUE_CLASS[declared](value=_coerce_scalar(declared, value, path))
