"""Shared contracts for source documents and their occurrence cursors."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence, Set
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, TypeAlias

from yian.diagnostics import SourceRange


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
class ScalarSource:
    """Source provenance for one unchanged constructed string scalar.

    Attributes:
        range (`SourceRange`):
            Whole authored scalar range used when exact line selection is
            unavailable.
        constructed_line_ranges (`tuple[SourceRange, ...] | None`):
            Zero-based ranges of constructed line contents for a literal block
            scalar. ``None`` means the scalar has physical source but its
            constructed lines cannot be selected exactly; ``()`` means an
            exactly mapped scalar with no constructed lines.
    """

    range: SourceRange
    constructed_line_ranges: tuple[SourceRange, ...] | None


@dataclass(frozen=True, slots=True)
class _PathLink:
    """One lazily materialized occurrence-path link."""

    parent: _PathLink | None
    step: _MapStep | _SeqStep | None
    definition_id: int


@dataclass(frozen=True, slots=True)
class _MapArc:
    """One authored non-merge mapping pair retained without duplicate collapse."""

    constructed_key: Any = field(repr=False, compare=False)
    raw_key: str | None
    key_definition: int
    value_definition: int


@dataclass(frozen=True, slots=True)
class _MapOccurrence:
    """One local arc at its logical position in a flattened merge view."""

    arc: _MapArc
    pair_ordinal: int


@dataclass(frozen=True, slots=True, eq=False)
class _ArcSequence:
    """Compact persistent sequence of mapping arcs.

    A leaf has ``arcs`` and no ``parts``. A concatenation has ``parts`` and no
    ``arcs``. ``part_offsets`` stores each child's logical start ordinal.
    """

    arcs: tuple[_MapArc, ...]
    parts: tuple[_ArcSequence, ...]
    part_offsets: tuple[int, ...]
    flat_pair_count: int


class _KeyIndex:
    """Private dict-semantic key index with no mutation surface."""

    __slots__ = ("_positions",)

    def __init__(self, entries: Iterable[tuple[Any, Any]]) -> None:
        positions: dict[Any, Any] = {}
        for key, position in entries:
            positions[key] = position
        self._positions = positions

    @classmethod
    def last_positions(cls, keys: Iterable[Any]) -> _KeyIndex:
        """Index each key by its last zero-based position."""

        return cls((key, position) for position, key in enumerate(keys))

    def get(self, key: Any) -> Any | None:
        """Return an indexed value, or ``None`` for absent/unhashable keys."""

        try:
            return self._positions.get(key)
        except (TypeError, ValueError):
            return None

    def __len__(self) -> int:
        """Return the number of dict-semantic keys retained by the index."""

        return len(self._positions)


@dataclass(frozen=True, slots=True)
class _ScalarDef:
    """One scalar-shaped YAML definition."""

    value: Any = field(repr=False, compare=False)
    range: SourceRange | None
    raw: str | None
    constructed_line_ranges: tuple[SourceRange, ...] | None


@dataclass(frozen=True, slots=True)
class _MappingDef:
    """One mapping definition backed by a compact persistent pair sequence."""

    value: Any = field(repr=False, compare=False)
    range: SourceRange | None
    pair_sequence: _ArcSequence
    effective_index: _KeyIndex = field(repr=False, compare=False)


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


def _snapshot_safe_value(value: Any) -> Any:
    """Validate and iteratively copy one exact SafeLoader value graph.

    Mutable containers are allocated before their contents, which preserves
    recursive aliases. Immutable containers are then finalized bottom-up and
    may safely point at those mutable placeholders.
    """

    originals: dict[int, Any] = {}
    containers: list[Any] = []
    copies: dict[int, Any] = {}
    pending = [value]

    while pending:
        current = pending.pop()
        current_type = type(current)
        if current_type in _SAFE_SCALAR_TYPES:
            if (
                current_type is datetime
                and current.tzinfo is not None
                and type(current.tzinfo) is not timezone
            ):
                raise TypeError(
                    "source cursors accept only exact SafeLoader value types; "
                    "got datetime with a non-SafeLoader timezone"
                )
            continue
        if current_type not in _SAFE_CONTAINER_TYPES:
            raise TypeError(
                "source cursors accept only exact SafeLoader value types; "
                f"got {current_type.__name__}"
            )

        identity = id(current)
        if identity in originals:
            continue
        originals[identity] = current
        containers.append(current)

        if current_type is dict:
            copies[identity] = {}
            for key, item in current.items():
                pending.append(item)
                pending.append(key)
        else:
            if current_type is list:
                copies[identity] = []
            elif current_type is set:
                copies[identity] = set()
            pending.extend(current)

    temporal_copies: dict[int, date | datetime] = {}

    def copied_item(item: Any) -> Any:
        """Return an atomic value or its already-allocated container copy."""

        item_type = type(item)
        if item_type in (date, datetime):
            identity = id(item)
            copied = temporal_copies.get(identity)
            if copied is None:
                copied = item.replace()
                temporal_copies[identity] = copied
            return copied
        if item_type in _SAFE_SCALAR_TYPES:
            return item
        return copies[id(item)]

    immutable_states: dict[int, bool] = {}
    for container in containers:
        if type(container) not in (tuple, frozenset):
            continue
        identity = id(container)
        if identity in copies:
            continue

        stack: list[tuple[Any, bool]] = [(container, False)]
        while stack:
            current, ready = stack.pop()
            current_identity = id(current)
            if current_identity in copies:
                continue
            if ready:
                if type(current) is tuple:
                    copied = tuple(copied_item(item) for item in current)
                else:
                    copied = frozenset(copied_item(item) for item in current)
                copies[current_identity] = copied
                immutable_states[current_identity] = False
                continue

            if immutable_states.get(current_identity, False):
                raise TypeError("source value contains an immutable container cycle")
            immutable_states[current_identity] = True
            stack.append((current, True))
            for item in reversed(tuple(current)):
                if type(item) in (tuple, frozenset) and id(item) not in copies:
                    stack.append((item, False))

    for container in containers:
        container_type = type(container)
        target = copies[id(container)]
        if container_type is dict:
            for key, item in container.items():
                target[copied_item(key)] = copied_item(item)
        elif container_type is list:
            target.extend(copied_item(item) for item in container)
        elif container_type is set:
            target.update(copied_item(item) for item in container)

    return copied_item(value)


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
