"""Shared contracts for source documents and their occurrence cursors."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence, Set
from dataclasses import dataclass, field
from datetime import date, datetime
import copy
from typing import Any, TypeAlias

from agent_composer.events import SourceRange


def _occurrence_key_token(value: Any) -> Any:
    """Encode a physical key without Python's cross-type equality collapse."""

    value_type = type(value)
    if value_type is tuple:
        return (tuple, tuple(_occurrence_key_token(item) for item in value))
    if value_type is frozenset:
        return (
            frozenset,
            frozenset(_occurrence_key_token(item) for item in value),
        )
    if value_type is float:
        return (float, value.hex())
    return (value_type, value)


@dataclass(frozen=True, slots=True, eq=False)
class _MapStep:
    """One mapping-pair step in an occurrence path.

    ``constructed_key`` is the SafeLoader-produced key and ``pair_ordinal`` is
    its zero-based position in the YAML mapping, including duplicate pairs.
    """

    constructed_key: Any
    pair_ordinal: int

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _MapStep):
            return NotImplemented
        return (
            self.pair_ordinal == other.pair_ordinal
            and _occurrence_key_token(self.constructed_key)
            == _occurrence_key_token(other.constructed_key)
        )

    def __hash__(self) -> int:
        return hash(
            (self.pair_ordinal, _occurrence_key_token(self.constructed_key))
        )


@dataclass(frozen=True, slots=True)
class _SeqStep:
    """One zero-based sequence-item step in an occurrence path."""

    index: int


OccurrencePath: TypeAlias = tuple[_MapStep | _SeqStep, ...]


@dataclass(frozen=True, slots=True)
class _PathLink:
    """One lazily materialized occurrence-path link."""

    parent: _PathLink | None
    step: _MapStep | _SeqStep | None
    definition_id: int


@dataclass(frozen=True, slots=True)
class _MapArc:
    """One flattened mapping pair retained without duplicate collapse."""

    constructed_key: Any = field(repr=False, compare=False)
    raw_key: str | None
    key_definition: int
    value_definition: int
    pair_ordinal: int


@dataclass(frozen=True, slots=True)
class _ScalarDef:
    """One scalar-shaped YAML definition."""

    value: Any = field(repr=False, compare=False)
    range: SourceRange | None
    raw: str | None


@dataclass(frozen=True, slots=True)
class _MappingDef:
    """One mapping-shaped YAML definition and its flattened pair arcs."""

    value: Any = field(repr=False, compare=False)
    range: SourceRange | None
    arcs: tuple[_MapArc, ...]
    effective_ordinals: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _SequenceDef:
    """One sequence-shaped YAML definition and its item definitions."""

    value: Any = field(repr=False, compare=False)
    range: SourceRange | None
    item_definitions: tuple[int, ...]


_Definition: TypeAlias = _ScalarDef | _MappingDef | _SequenceDef


_SAFE_SCALAR_TYPES = (
    type(None),
    bool,
    int,
    float,
    str,
    bytes,
    date,
    datetime,
)
_SAFE_CONTAINER_TYPES = (dict, list, tuple, set, frozenset)


def _validate_safe_value(value: Any, seen: set[int] | None = None) -> None:
    """Validate the exact recursive value algebra produced by ``SafeLoader``."""

    value_type = type(value)
    if value_type in _SAFE_SCALAR_TYPES:
        return
    if value_type not in _SAFE_CONTAINER_TYPES:
        raise TypeError(
            "source cursors accept only exact SafeLoader value types; "
            f"got {value_type.__name__}"
        )

    if seen is None:
        seen = set()
    identity = id(value)
    if identity in seen:
        return
    seen.add(identity)

    if value_type is dict:
        for key, item in value.items():
            _validate_safe_value(key, seen)
            _validate_safe_value(item, seen)
        return
    for item in value:
        _validate_safe_value(item, seen)


def _snapshot_safe_value(value: Any) -> Any:
    """Validate and defensively snapshot one virtual source value."""

    _validate_safe_value(value)
    return copy.deepcopy(value)


class _ReadOnlyMapping(Mapping[Any, Any]):
    """A recursive mapping view with no mutation surface."""

    __slots__ = ("_source", "_memo")

    def __init__(self, source: dict[Any, Any], memo: dict[int, Any]) -> None:
        self._source = source
        self._memo = memo

    def __getitem__(self, key: Any) -> Any:
        return _read_only(self._source[key], self._memo)

    def __iter__(self) -> Iterator[Any]:
        return iter(self._source)

    def __len__(self) -> int:
        return len(self._source)


class _ReadOnlySequence(Sequence[Any]):
    """A recursive list/tuple view with no mutation surface."""

    __slots__ = ("_source", "_memo")

    def __init__(
        self,
        source: list[Any] | tuple[Any, ...],
        memo: dict[int, Any],
    ) -> None:
        self._source = source
        self._memo = memo

    def __getitem__(self, index: int | slice) -> Any:
        if isinstance(index, slice):
            return tuple(_read_only(item, self._memo) for item in self._source[index])
        return _read_only(self._source[index], self._memo)

    def __len__(self) -> int:
        return len(self._source)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Sequence) or isinstance(other, (str, bytes)):
            return NotImplemented
        return len(self) == len(other) and all(
            left is right or left == right for left, right in zip(self, other)
        )

    def __hash__(self) -> int:
        if type(self._source) is list:
            raise TypeError("unhashable type: '_ReadOnlySequence'")
        return hash(tuple(self))


class _ReadOnlySet(Set[Any]):
    """A recursive set/frozenset view with no mutation surface."""

    __slots__ = ("_source", "_memo")

    def __init__(self, source: set[Any] | frozenset[Any], memo: dict[int, Any]) -> None:
        self._source = source
        self._memo = memo

    def __contains__(self, candidate: object) -> bool:
        return candidate in self._source

    def __iter__(self) -> Iterator[Any]:
        return (_read_only(item, self._memo) for item in self._source)

    def __len__(self) -> int:
        return len(self._source)

    @classmethod
    def _from_iterable(cls, iterable: Iterator[Any]) -> set[Any]:
        """Return a normal mutable set for inherited ABC operations."""

        return set(iterable)

    def _result(self, iterable: Iterator[Any]) -> set[Any] | frozenset[Any]:
        """Build an operation result matching the viewed set's mutability."""

        factory = frozenset if type(self._source) is frozenset else set
        return factory(iterable)

    def __and__(self, other: object) -> set[Any] | frozenset[Any]:
        if not isinstance(other, Set):
            return NotImplemented
        return self._result(item for item in self if item in other)

    def __or__(self, other: object) -> set[Any] | frozenset[Any]:
        if not isinstance(other, Set):
            return NotImplemented
        return self._result(iter((*self, *other)))

    def __sub__(self, other: object) -> set[Any] | frozenset[Any]:
        if not isinstance(other, Set):
            return NotImplemented
        return self._result(item for item in self if item not in other)

    def __xor__(self, other: object) -> set[Any] | frozenset[Any]:
        if not isinstance(other, Set):
            return NotImplemented
        return self._result(
            item
            for item in (*self, *other)
            if (item in self) != (item in other)
        )

    def __hash__(self) -> int:
        if type(self._source) is set:
            raise TypeError("unhashable type: '_ReadOnlySet'")
        return hash(frozenset(self))


def _read_only(value: Any, memo: dict[int, Any] | None = None) -> Any:
    """Return an identity-memoized recursive read-only view of ``value``."""

    value_type = type(value)
    if value_type in _SAFE_SCALAR_TYPES:
        return value
    if value_type not in _SAFE_CONTAINER_TYPES:
        raise TypeError(
            "source cursors accept only exact SafeLoader value types; "
            f"got {value_type.__name__}"
        )

    if memo is None:
        memo = {}
    identity = id(value)
    cached = memo.get(identity)
    if cached is not None:
        return cached

    if value_type is dict:
        view: Any = _ReadOnlyMapping(value, memo)
    elif value_type in (list, tuple):
        view = _ReadOnlySequence(value, memo)
    else:
        view = _ReadOnlySet(value, memo)
    memo[identity] = view
    return view


class SourceAccessError(ValueError):
    """A source cursor cannot satisfy a requested navigation or typed access.

    ``range`` is the closest concrete source location, or ``None`` for a
    virtual or otherwise unlocated value.
    """

    def __init__(self, message: str, range: SourceRange | None = None):
        super().__init__(message)
        self.range = range


class SourceSyntaxError(ValueError):
    """A YAML source document cannot be composed or constructed safely.

    ``range`` identifies PyYAML's problem mark when one is available; otherwise
    it is ``None``.
    """

    def __init__(self, message: str, range: SourceRange | None = None):
        super().__init__(message)
        self.range = range
