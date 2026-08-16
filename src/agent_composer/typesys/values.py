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

This module has no package-internal dependencies on purpose; it is the leaf.
"""

from dataclasses import dataclass
from datetime import date as _date
from datetime import datetime as _datetime
from enum import Enum
from typing import Annotated, Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator


class TypeCheckError(ValueError):
    """A value cannot be wrapped in a TypedValue, or violates a declared type."""


# --------------------------------------------------------------------------- #
# Type vocabulary
# --------------------------------------------------------------------------- #


class ValueKind(str, Enum):
    """Closed type vocabulary for runtime values.

    `.value` IS the one Python-surface vocabulary: it is simultaneously the serialized
    discriminator tag, the author-written `type:` token, and the displayed name (so there
    is no second "engine name" to translate — `str`, never `string`).
    """

    NONE = "None"
    STRING = "str"
    INTEGER = "int"
    NUMBER = "float"
    DATE = "date"  # ISO-8601 calendar date, stored as str; never auto-inferred
    DATETIME = "datetime"  # ISO-8601 datetime, stored as str; never auto-inferred
    BOOLEAN = "bool"
    OBJECT = "object"
    FILE = "file"  # reserved: never auto-inferred, not authorable yet

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
    value: str  # ISO-8601 "YYYY-MM-DD"

    @field_validator("value")
    @classmethod
    def _iso(cls, v: str) -> str:
        _date.fromisoformat(v)  # raises ValueError if not a valid ISO date
        return v


class DateTimeValue(TypedValue):
    kind: Literal[ValueKind.DATETIME] = ValueKind.DATETIME
    value: str  # ISO-8601 "YYYY-MM-DDTHH:MM:SS[+HH:MM]"

    @field_validator("value")
    @classmethod
    def _iso(cls, v: str) -> str:
        # A bare date ("2026-06-12") parses as midnight; require a time component so
        # datetime stays distinct from date.
        if "T" not in v and " " not in v:
            raise ValueError("datetime must include a time component")
        _datetime.fromisoformat(v)  # raises ValueError if not a valid ISO datetime
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


class ListStringValue(TypedValue):
    kind: Literal[ValueKind.LIST_STRING] = ValueKind.LIST_STRING
    value: list[str]


class ListIntegerValue(TypedValue):
    kind: Literal[ValueKind.LIST_INTEGER] = ValueKind.LIST_INTEGER
    value: list[int]


class ListNumberValue(TypedValue):
    kind: Literal[ValueKind.LIST_NUMBER] = ValueKind.LIST_NUMBER
    value: list[float]


class ListBooleanValue(TypedValue):
    kind: Literal[ValueKind.LIST_BOOLEAN] = ValueKind.LIST_BOOLEAN
    value: list[bool]


class ListObjectValue(TypedValue):
    kind: Literal[ValueKind.LIST_OBJECT] = ValueKind.LIST_OBJECT
    value: list[dict[str, Any]]


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
      - variant: kind=STRING,   tags={labels}
      - list:    kind=LIST_*,   element=<element Type>
    A refined specialized list uses the element's matching storage tag;
    `LIST_ANY` is the generic carrier and may refine any element type.
    A plain scalar / flat list uses only `kind` (see `Type.scalar`).
    """

    kind: ValueKind
    fields: Optional[dict[str, "Type"]] = None
    required: Optional[frozenset[str]] = None
    tags: Optional[frozenset[str]] = None
    element: Optional["Type"] = None
    nullable: bool = False  # an Optional[X] slot — accepts None (present-None or absent)

    @classmethod
    def scalar(cls, seg: ValueKind) -> "Type":
        """Build a plain `Type` carrying only a storage tag (no field/tag/element refinement)."""
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
            expected_kind = _list_kind_for_element(element)
            if typ.kind is not ValueKind.LIST_ANY and typ.kind is not expected_kind:
                raise ValueError(
                    f"list storage kind {typ.kind.value!r} does not match "
                    f"element type; expected {expected_kind.value!r}"
                )

        refinements = sum(
            value is not None for value in (fields, tags, element)
        )
        if refinements > 1:
            raise ValueError("a type may declare only one structural refinement")
    finally:
        active.remove(identity)
    validated.add(identity)


def _validate_type_metadata_string(value: str) -> None:
    """Require a type-owned name to be representable in canonical JSON."""
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError("type metadata contains a Unicode surrogate") from exc


def _list_kind_for_element(element: Type) -> ValueKind:
    """Return the storage tag selected by the public type resolver for an element."""
    semantic_kind = element.kind
    if element.tags is not None:
        semantic_kind = ValueKind.STRING
    elif element.fields is not None:
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
    if declared == ValueKind.DATE:
        if not isinstance(value, str):
            return False
        try:
            _date.fromisoformat(value)
            return True
        except ValueError:
            return False
    if declared == ValueKind.DATETIME:
        if not isinstance(value, str):
            return False
        # a bare date is not a datetime — require an explicit time component
        if "T" not in value and " " not in value:
            return False
        try:
            _datetime.fromisoformat(value)
            return True
        except ValueError:
            return False
    if declared == ValueKind.OBJECT:
        return isinstance(value, dict)
    if declared == ValueKind.FILE:
        return isinstance(value, FileRef)
    return False


def _coerce_scalar(declared: ValueKind, value: Any) -> Any:
    # Only widening: an int may fill a NUMBER slot as a float.
    if declared == ValueKind.NUMBER and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    return value


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
        return ListNumberValue(value=[float(x) for x in items])
    if element_types == {ValueKind.OBJECT}:
        return ListObjectValue(value=items)
    return ListAnyValue(value=items)


def build_value_as(declared: "ValueKind | Type", value: Any) -> TypedValue:
    """Wrap `value` as a TypedValue matching `declared`, raising on a type mismatch.

    This is the write-boundary check the variable pool uses against each declared
    output type, so a node returning the wrong type fails loudly at the write rather
    than silently downstream. The only coercion is widening (an `int` fills a `float`
    slot).

    Args:
        declared (`ValueKind | Type`):
            The target type. A bare `ValueKind` is a scalar or flat list; a structural
            `Type` additionally checks records, variants, and typed/element lists.
        value (`Any`):
            The raw Python value to validate and wrap, or an already-wrapped
            `TypedValue` (unwrapped to its plain value first).

    Returns:
        `TypedValue`:
            The value wrapped in the subclass for `declared`'s storage kind.

    Raises:
        `TypeCheckError`:
            The value does not satisfy `declared` — wrong scalar type, a non-member
            variant tag, a record with missing/unknown fields, or a bad list element.
    """
    typ = Type.scalar(declared) if isinstance(declared, ValueKind) else declared
    if isinstance(value, TypedValue):
        value = value.to_object()
    return _build_for_type(typ, value)


def _build_for_type(typ: Type, value: Any) -> TypedValue:
    # nullable (Optional[X]) — a None value is accepted as NoneValue
    if value is None and typ.nullable:
        return NoneValue()

    # variant — a tag-constrained string
    if typ.tags is not None:
        if not isinstance(value, str) or value not in typ.tags:
            raise TypeCheckError(f"{value!r} is not a member of variant {sorted(typ.tags)}")
        return StringValue(value=value)

    # record — an object with declared field types (closed: all required, no unknowns)
    if typ.fields is not None:
        if not isinstance(value, dict):
            raise TypeCheckError(f"{value!r} is not an object for a record type")
        required = typ.required or frozenset()
        missing = required - value.keys()
        if missing:
            raise TypeCheckError(f"record missing required fields: {sorted(missing)}")
        unknown = value.keys() - typ.fields.keys()
        if unknown:
            raise TypeCheckError(f"record has unknown fields: {sorted(unknown)}")
        for fname, fval in value.items():
            _build_for_type(typ.fields[fname], fval)  # validates each field, raises on mismatch
        return ObjectValue(value=value)

    # list with a known element typ (incl. List[record])
    if typ.kind.is_list() and typ.element is not None:
        if not isinstance(value, (list, tuple)):
            raise TypeCheckError(
                f"{value!r} is not a list for declared {typ.kind.value}"
            )
        items = [_build_for_type(typ.element, item).to_object() for item in value]
        return _LIST_VALUE_CLASS[typ.kind](value=items)

    # plain scalar / flat list — the original behavior
    return _build_scalar_or_list(typ.kind, value)


def _build_scalar_or_list(declared: ValueKind, value: Any) -> TypedValue:
    if declared.is_list():
        if not isinstance(value, (list, tuple)):
            raise TypeCheckError(f"{value!r} is not a list for declared {declared.value}")
        items = list(value)
        element = declared.element_kind
        if element is not None:
            for item in items:
                if not _scalar_matches(element, item):
                    raise TypeCheckError(
                        f"list element {item!r} is not {element.value} "
                        f"(declared {declared.value})"
                    )
            items = [_coerce_scalar(element, item) for item in items]
        return _LIST_VALUE_CLASS[declared](value=items)

    if not _scalar_matches(declared, value):
        raise TypeCheckError(f"{value!r} does not match declared type {declared.value}")
    return _SCALAR_VALUE_CLASS[declared](value=_coerce_scalar(declared, value))
