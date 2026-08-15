"""Source-neutral semantic IR shared by node and flow construction.

These frozen carriers attach opaque origins to executable meanings without
depending on Compose/YAML or runtime state. They are definitions only: consumer
cutovers normalize wiring, assertions, parameters, and value contracts in later
steps of the origin-IR migration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, TypeAlias

from agent_composer.diagnostics import OriginRef
from agent_composer.typesys import Type


ValuePath: TypeAlias = tuple[str | int, ...]
"""A source-neutral path through record fields and list indices."""

_OriginMapper: TypeAlias = Callable[[OriginRef], OriginRef]


def _validate_optional_origin(origin: OriginRef | None, field_name: str) -> None:
    """Require an optional origin field to contain only the opaque leaf value."""
    if origin is not None and not isinstance(origin, OriginRef):
        raise TypeError(f"{field_name} must be an OriginRef or None")


def _map_origin(origin: OriginRef | None, mapper: _OriginMapper) -> OriginRef | None:
    """Transform one present origin and reject erasure or a non-origin result."""
    if origin is None:
        return None
    mapped = mapper(origin)
    if not isinstance(mapped, OriginRef):
        raise TypeError("origin mapper must return an OriginRef")
    return mapped


def _validate_path(path: ValuePath, *, field_name: str) -> None:
    """Validate one immutable value path using exact string/integer segments."""
    if type(path) is not tuple:
        raise TypeError(f"{field_name} path must be a tuple")
    for segment in path:
        if type(segment) not in (str, int):
            raise TypeError(
                f"{field_name} path segment must be an exact str or int, "
                f"got {type(segment).__name__}"
            )


def _path_sort_key(path: ValuePath) -> tuple[tuple[int, str | int], ...]:
    """Return a Python-3.10-safe total key with strings before integers."""
    return tuple(
        (0, segment) if type(segment) is str else (1, segment) for segment in path
    )


@dataclass(frozen=True)
class Binding:
    """One flow-owned input expression or literal plus its authored origin.

    Attributes:
        source (`Any`): Semantic binding payload; retained by identity.
        origin (`OriginRef | None`): Authored binding site, when available.
    """

    source: Any
    origin: OriginRef | None = None

    def __post_init__(self) -> None:
        """Reject non-origin provenance values."""
        _validate_optional_origin(self.origin, "binding origin")

    def map_origins(self, mapper: _OriginMapper) -> Binding:
        """Return this binding with its present origin transformed by ``mapper``."""
        origin = _map_origin(self.origin, mapper)
        return Binding(source=self.source, origin=origin)


@dataclass(frozen=True)
class Assertion:
    """An executable assertion and its stable author-facing presentation.

    Attributes:
        expression (`str`): Expression evaluated after any lowering rewrite.
        display (`str`): Original text reported to the author.
        origin (`OriginRef | None`): Authored assertion occurrence, when known.
    """

    expression: str
    display: str
    origin: OriginRef | None = None

    def __post_init__(self) -> None:
        """Reject non-origin provenance values."""
        _validate_optional_origin(self.origin, "assertion origin")

    def map_origins(self, mapper: _OriginMapper) -> Assertion:
        """Return this assertion with its present origin transformed."""
        origin = _map_origin(self.origin, mapper)
        return Assertion(
            expression=self.expression,
            display=self.display,
            origin=origin,
        )


@dataclass(frozen=True)
class ContractOrigins:
    """Immutable origin index for one value contract and its nested paths.

    ``default`` exclusively owns the root/fallback location. ``entries`` is a
    strictly sorted tuple of unique, non-empty ``(ValuePath, OriginRef)`` pairs.
    String path segments sort before integer segments; a prefix sorts before its
    descendants.

    Attributes:
        default (`OriginRef | None`): Root/fallback contract site.
        entries (`tuple[tuple[ValuePath, OriginRef], ...]`): Nested sites in
            canonical path order.
    """

    default: OriginRef | None = None
    entries: tuple[tuple[ValuePath, OriginRef], ...] = ()

    def __post_init__(self) -> None:
        """Reject mutable, ambiguous, duplicate, or noncanonical entries."""
        _validate_optional_origin(self.default, "contract default origin")
        if type(self.entries) is not tuple:
            raise TypeError("contract origin entries must be a tuple")

        paths: list[ValuePath] = []
        seen: set[ValuePath] = set()
        for entry in self.entries:
            if type(entry) is not tuple or len(entry) != 2:
                raise TypeError("each contract origin entry must be a (path, origin) tuple")
            path, origin = entry
            _validate_path(path, field_name="contract origin entry")
            if not path:
                raise ValueError(
                    "contract origin entry path cannot be empty; use default for the root"
                )
            if not isinstance(origin, OriginRef):
                raise TypeError("contract origin entry value must be an OriginRef")
            if path in seen:
                raise ValueError(f"duplicate contract origin path: {path!r}")
            seen.add(path)
            paths.append(path)

        if paths != sorted(paths, key=_path_sort_key):
            raise ValueError("contract origin entries must be sorted canonically")

    def at(self, path: ValuePath) -> OriginRef | None:
        """Select the exact, nearest ancestor, or default origin for ``path``."""
        _validate_path(path, field_name="contract lookup")
        indexed = dict(self.entries)
        for length in range(len(path), 0, -1):
            origin = indexed.get(path[:length])
            if origin is not None:
                return origin
        return self.default

    def map_origins(self, mapper: _OriginMapper) -> ContractOrigins:
        """Return this index with every present origin transformed in order."""
        default = _map_origin(self.default, mapper)
        entries = tuple((path, _map_origin(origin, mapper)) for path, origin in self.entries)
        return ContractOrigins(default=default, entries=entries)


@dataclass(frozen=True)
class ValueContract:
    """A semantic runtime type paired with source-neutral nested provenance.

    Attributes:
        type (`Type`): Provenance-free recursive runtime type.
        origins (`ContractOrigins`): Root and nested authored contract sites.
    """

    type: Type
    origins: ContractOrigins = field(default_factory=ContractOrigins)

    def __post_init__(self) -> None:
        """Require the dedicated semantic and provenance value types."""
        if not isinstance(self.type, Type):
            raise TypeError("value contract type must be a Type")
        if not isinstance(self.origins, ContractOrigins):
            raise TypeError("value contract origins must be ContractOrigins")

    def map_origins(self, mapper: _OriginMapper) -> ValueContract:
        """Return this contract with nested origins transformed, retaining its type."""
        return ValueContract(type=self.type, origins=self.origins.map_origins(mapper))


@dataclass(frozen=True)
class Parameter:
    """One formal input with one optional value contract and default policy.

    ``has_default`` distinguishes an explicit ``None`` default from no default.
    ``coerce_strings`` is boundary policy derived from ``contract.type`` by a
    later consumer; there is deliberately no parallel textual type field.

    Attributes:
        name (`str`): Bound record key.
        contract (`ValueContract | None`): Optional semantic input contract.
        required (`bool`): Whether omission is invalid.
        has_default (`bool`): Whether ``default`` is semantically present.
        default (`Any`): Retained default payload; meaningful only when present.
        coerce_strings (`bool`): Whether a boundary may coerce textual input.
        origin (`OriginRef | None`): Parameter declaration site.
        default_origin (`OriginRef | None`): Explicit default site, when present.
    """

    name: str
    contract: ValueContract | None = None
    required: bool = False
    has_default: bool = False
    default: Any = None
    coerce_strings: bool = False
    origin: OriginRef | None = None
    default_origin: OriginRef | None = None

    def __post_init__(self) -> None:
        """Enforce one unambiguous representation of required/default state."""
        if self.contract is not None and not isinstance(self.contract, ValueContract):
            raise TypeError("parameter contract must be a ValueContract or None")
        _validate_optional_origin(self.origin, "parameter origin")
        _validate_optional_origin(self.default_origin, "parameter default origin")
        if self.required and self.has_default:
            raise ValueError("a required parameter cannot have a default")
        if not self.has_default and self.default is not None:
            raise ValueError("parameter default requires has_default=True")
        if not self.has_default and self.default_origin is not None:
            raise ValueError("parameter default_origin requires has_default=True")

    def map_origins(self, mapper: _OriginMapper) -> Parameter:
        """Transform parameter, default, and nested contract origins without copying data."""
        contract = (
            self.contract.map_origins(mapper) if self.contract is not None else None
        )
        origin = _map_origin(self.origin, mapper)
        default_origin = _map_origin(self.default_origin, mapper)
        return Parameter(
            name=self.name,
            contract=contract,
            required=self.required,
            has_default=self.has_default,
            default=self.default,
            coerce_strings=self.coerce_strings,
            origin=origin,
            default_origin=default_origin,
        )


def as_binding(value: Any) -> Binding:
    """Return ``value`` unchanged when bound, otherwise wrap it origin-free."""
    return value if isinstance(value, Binding) else Binding(source=value)


def binding_source(value: Any) -> Any:
    """Unwrap exactly one ``Binding`` layer, or return a raw value unchanged."""
    return value.source if isinstance(value, Binding) else value


__all__ = [
    "Assertion",
    "Binding",
    "ContractOrigins",
    "Parameter",
    "ValueContract",
    "ValuePath",
    "as_binding",
    "binding_source",
]
