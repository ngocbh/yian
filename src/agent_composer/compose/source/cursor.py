"""Immutable cursors over physical and virtual YAML source values."""

from __future__ import annotations

import copy
from bisect import bisect_right
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from agent_composer.compose.source.common import (
    OccurrencePath,
    SourceAccessError,
    _ArcSequence,
    _KeyIndex,
    _MapArc,
    _MapOccurrence,
    _MapStep,
    _MappingDef,
    _PathLink,
    _ScalarDef,
    _SeqStep,
    _SequenceDef,
    _read_only,
    _snapshot_safe_value,
)
from agent_composer.events import SourceRange

if TYPE_CHECKING:
    from agent_composer.compose.source.document import SourceDocument


_NO_FILTER = object()


@dataclass(frozen=True, slots=True, eq=False)
class _ProjectedEntry:
    """One projected mapping entry and its optional physical source cursor."""

    key: Any = field(repr=False, compare=False)
    source: SourceCursor | None = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True, eq=False)
class SourceCursor:
    """Navigate one immutable, occurrence-aware YAML source value.

    Physical cursors retain a document definition, occurrence path, and ranges.
    Virtual cursors retain only a defensive value snapshot. Projected cursors
    expose replacement values while preserving explicitly supplied physical
    child sources. Cursor equality and hashing use object identity.

    Examples:
        >>> cursor = SourceCursor.virtual({"items": ["first"]})
        >>> cursor.child("items").child(0).value
        'first'
    """

    _document: SourceDocument | None = field(repr=False, compare=False)
    _definition: int | None = field(repr=False, compare=False)
    _value: Any = field(repr=False, compare=False)
    _path_link: _PathLink | None = field(repr=False, compare=False)
    _key_definition: int | None = field(repr=False, compare=False)
    _cycle_cut: bool = field(repr=False, compare=False)
    _overlay: bool = field(default=False, repr=False, compare=False)
    _projected_entries: tuple[_ProjectedEntry, ...] | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    _projected_index: _KeyIndex | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    _sequence_sources: tuple[SourceCursor | None, ...] | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    @classmethod
    def _physical(
        cls,
        document: SourceDocument,
        definition: int,
        *,
        path_link: _PathLink | None = None,
        key_definition: int | None = None,
        cycle_cut: bool = False,
    ) -> SourceCursor:
        """Build a cursor owned by ``document``."""

        if path_link is None:
            path_link = _PathLink(None, None, definition)
        return cls(
            _document=document,
            _definition=definition,
            _value=document._definitions[definition].value,
            _path_link=path_link,
            _key_definition=key_definition,
            _cycle_cut=cycle_cut,
            _overlay=False,
            _projected_entries=None,
            _projected_index=None,
            _sequence_sources=None,
        )

    @classmethod
    def virtual(cls, value: Any) -> SourceCursor:
        """Build a location-free cursor over a defensive value snapshot.

        Args:
            value (`Any`):
                Value in the exact recursive SafeLoader algebra: built-in
                containers and YAML-safe scalar types, without subclasses.

        Returns:
            `SourceCursor`:
                A virtual cursor with no document, path, or source ranges.

        Raises:
            `TypeError`:
                Raised when `value` contains an unsupported type or subclass.
        """

        return cls(
            _document=None,
            _definition=None,
            _value=_snapshot_safe_value(value),
            _path_link=None,
            _key_definition=None,
            _cycle_cut=False,
            _overlay=False,
            _projected_entries=None,
            _projected_index=None,
            _sequence_sources=None,
        )

    @property
    def document(self) -> SourceDocument | None:
        """Return the physical document retained by this cursor.

        Returns:
            `SourceDocument | None`:
                The owning document, or `None` for a source-free cursor.
        """

        return self._document

    @property
    def path(self) -> OccurrencePath:
        """Materialize the finite physical occurrence path on demand.

        Returns:
            `OccurrencePath`:
                Mapping-pair and sequence-index steps from the document root.
                A source-free cursor returns an empty path.
        """

        steps: list[_MapStep | _SeqStep] = []
        link = self._path_link
        while link is not None:
            if link.step is not None:
                steps.append(link.step)
            link = link.parent
        steps.reverse()
        return tuple(steps)

    @property
    def range(self) -> SourceRange | None:
        """Return the concrete range of the current value definition.

        Returns:
            `SourceRange | None`:
                The anchor-definition range, or `None` when source-free.
        """

        if self._document is None or self._definition is None:
            return None
        return self._document._definitions[self._definition].range

    @property
    def key_range(self) -> SourceRange | None:
        """Return the mapping-key range selecting this occurrence.

        Returns:
            `SourceRange | None`:
                The selecting key range, or `None` for roots, sequence items,
                and source-free cursors.
        """

        if self._document is None or self._key_definition is None:
            return None
        return self._document._definitions[self._key_definition].range

    @property
    def value(self) -> Any:
        """Return a defensive, alias-preserving value copy.

        Returns:
            `Any`:
                A fresh deep copy whose shared and recursive aliases are kept.
        """

        return _snapshot_safe_value(self._value)

    def _value_view(self) -> Any:
        """Return an identity-preserving recursive read-only value view."""

        return _read_only(self._value)

    def _ensure_traversable(self) -> None:
        """Reject traversal beyond a cycle-cut recursive alias."""

        if self._cycle_cut:
            raise SourceAccessError(
                "cannot descend through a recursive alias occurrence",
                self.range,
            )

    def _mapping_definition(self) -> _MappingDef | None:
        """Return the navigable physical mapping definition, if present."""

        if self._overlay or self._document is None or self._definition is None:
            return None
        definition = self._document._definitions[self._definition]
        if isinstance(definition, _MappingDef) and type(self._value) is dict:
            return definition
        return None

    def _sequence_definition(self) -> _SequenceDef | None:
        """Return the navigable physical sequence definition, if present."""

        if self._overlay or self._document is None or self._definition is None:
            return None
        definition = self._document._definitions[self._definition]
        if isinstance(definition, _SequenceDef) and type(self._value) in (list, tuple):
            return definition
        return None

    @staticmethod
    def _keys_equal(left: Any, right: Any) -> bool:
        """Compare keys with constructed-mapping equality and identity fallback."""

        if left is right:
            return True
        try:
            return hash(left) == hash(right) and left == right
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _mapping_contains(value: Any, key: Any) -> bool:
        """Test exact dict membership while treating unhashable keys as absent."""

        if type(value) is not dict:
            return False
        try:
            return key in value
        except (TypeError, ValueError):
            return False

    def _effective_occurrence_in_definition(
        self,
        definition_id: int,
        key: Any,
    ) -> _MapOccurrence | None:
        """Resolve one key from a compact pair rope's effective index."""

        assert self._document is not None
        definition = self._document._definitions[definition_id]
        if not isinstance(definition, _MappingDef):
            return None
        return definition.effective_index.get(key)

    def _effective_occurrence(self, key: Any) -> _MapOccurrence | None:
        """Find the last effective physical mapping occurrence for ``key``."""

        definition = self._mapping_definition()
        if definition is None or not self._mapping_contains(self._value, key):
            return None
        assert self._definition is not None
        return self._effective_occurrence_in_definition(self._definition, key)

    def _iter_flat_occurrences(
        self,
        sequence: _ArcSequence,
    ) -> Iterator[_MapOccurrence]:
        """Yield logical flattened occurrences using an explicit merge-DAG stack."""

        stack: list[tuple[str, Any]] = [("sequence", sequence)]
        ordinal = 0
        while stack:
            kind, item = stack.pop()
            if kind == "arc":
                yield _MapOccurrence(item, ordinal)
                ordinal += 1
                continue
            for arc in reversed(item.arcs):
                stack.append(("arc", arc))
            for part in reversed(item.parts):
                stack.append(("sequence", part))

    def _iter_unique_occurrences(
        self,
        sequence: _ArcSequence,
        *,
        reverse: bool = False,
    ) -> Iterator[_MapOccurrence]:
        """Yield shared leaves once at their first or last logical embedding."""

        seen: set[_ArcSequence] = set()
        stack: list[tuple[_ArcSequence, int]] = [(sequence, 0)]
        while stack:
            current, base = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            if current.parts:
                indices = range(len(current.parts))
                if not reverse:
                    indices = reversed(tuple(indices))
                for index in indices:
                    stack.append(
                        (
                            current.parts[index],
                            base + current.part_offsets[index],
                        )
                    )
                continue
            indices = (
                range(len(current.arcs) - 1, -1, -1)
                if reverse
                else range(len(current.arcs))
            )
            for index in indices:
                yield _MapOccurrence(current.arcs[index], base + index)

    def _flat_arc_at(self, sequence: _ArcSequence, ordinal: int) -> _MapArc:
        """Return one logical flattened arc without enumerating preceding pairs."""

        current = sequence
        current_ordinal = ordinal
        while True:
            if not current.parts:
                if not 0 <= current_ordinal < len(current.arcs):
                    raise SourceAccessError(
                        "invalid mapping occurrence path",
                        self.range,
                    )
                return current.arcs[current_ordinal]
            part_index = bisect_right(
                current.part_offsets,
                current_ordinal,
            ) - 1
            if part_index < 0:
                raise SourceAccessError("invalid mapping occurrence path", self.range)
            current_ordinal -= current.part_offsets[part_index]
            current = current.parts[part_index]

    def _descend_mapping_occurrence(
        self,
        occurrence: _MapOccurrence,
    ) -> SourceCursor:
        """Create a child cursor for one logical physical mapping occurrence."""

        assert self._document is not None
        assert self._path_link is not None
        arc = occurrence.arc
        link = self._path_link
        recursive = False
        while link is not None:
            if link.definition_id == arc.value_definition:
                recursive = True
                break
            link = link.parent
        child_link = _PathLink(
            self._path_link,
            _MapStep(arc.constructed_key, occurrence.pair_ordinal),
            arc.value_definition,
        )
        return SourceCursor._physical(
            self._document,
            arc.value_definition,
            path_link=child_link,
            key_definition=arc.key_definition,
            cycle_cut=recursive,
        )

    def _descend_sequence_item(self, index: int, definition_id: int) -> SourceCursor:
        """Create a child cursor for one physical sequence item."""

        assert self._document is not None
        assert self._path_link is not None
        link = self._path_link
        recursive = False
        while link is not None:
            if link.definition_id == definition_id:
                recursive = True
                break
            link = link.parent
        child_link = _PathLink(
            self._path_link,
            _SeqStep(index),
            definition_id,
        )
        return SourceCursor._physical(
            self._document,
            definition_id,
            path_link=child_link,
            cycle_cut=recursive,
        )

    def _projected_entry(self, key: Any) -> _ProjectedEntry | None:
        """Find one effective projected mapping entry."""

        if self._projected_entries is None or self._projected_index is None:
            return None
        index = self._projected_index.get(key)
        return self._projected_entries[index] if index is not None else None

    def _projected_child(self, entry: _ProjectedEntry) -> SourceCursor:
        """Expose a projected value with its optional retained child source."""

        value = self._value[entry.key]
        if entry.source is None:
            return SourceCursor.virtual(value)
        return entry.source.with_value(value)

    def _aligned_sequence_child(self, index: int) -> SourceCursor:
        """Expose one replacement item aligned with an optional source item."""

        assert self._sequence_sources is not None
        if type(index) is not int or index < 0:
            raise SourceAccessError(
                "sequence index must be a non-negative integer",
                self.range,
            )
        if index >= len(self._sequence_sources):
            raise SourceAccessError(
                f"sequence index {index} is out of range",
                self.range,
            )
        value = self._value[index]
        source = self._sequence_sources[index]
        if source is None:
            return SourceCursor.virtual(value)
        return source.with_value(value)

    def _value_only_child(self, key: Any) -> SourceCursor:
        """Navigate an unlocated mapping or sequence value."""

        if type(self._value) is dict:
            try:
                present = key in self._value
            except TypeError:
                present = False
            if not present:
                raise SourceAccessError(f"missing required field {key!r}", self.range)
            return SourceCursor.virtual(self._value[key])
        if type(self._value) in (list, tuple):
            if type(key) is not int or key < 0:
                raise SourceAccessError(
                    "sequence index must be a non-negative integer",
                    self.range,
                )
            if key >= len(self._value):
                raise SourceAccessError(
                    f"sequence index {key} is out of range",
                    self.range,
                )
            return SourceCursor.virtual(self._value[key])
        raise SourceAccessError("cannot select a child from a scalar value", self.range)

    def child(self, key: Any) -> SourceCursor:
        """Return one effective mapping child or indexed sequence child.

        Args:
            key (`Any`):
                A constructed mapping key, or an exact nonnegative `int` index
                for a sequence. A `bool` is never a sequence index.

        Returns:
            `SourceCursor`:
                The last effective mapping value or requested sequence item.

        Raises:
            `SourceAccessError`:
                Raised for a missing key, invalid or out-of-range index, wrong
                container shape, or traversal beyond a recursive alias cut.
        """

        self._ensure_traversable()
        if self._projected_entries is not None:
            entry = self._projected_entry(key)
            if entry is None:
                raise SourceAccessError(f"missing required field {key!r}", self.range)
            return self._projected_child(entry)
        if self._sequence_sources is not None:
            return self._aligned_sequence_child(key)
        if self._overlay or self._document is None:
            return self._value_only_child(key)

        mapping = self._mapping_definition()
        if mapping is not None:
            occurrence = self._effective_occurrence(key)
            if occurrence is None:
                raise SourceAccessError(f"missing required field {key!r}", self.range)
            return self._descend_mapping_occurrence(occurrence)

        sequence = self._sequence_definition()
        if sequence is not None:
            if type(key) is not int or key < 0:
                raise SourceAccessError(
                    "sequence index must be a non-negative integer",
                    self.range,
                )
            if key >= len(sequence.item_definitions):
                raise SourceAccessError(
                    f"sequence index {key} is out of range",
                    self.range,
                )
            return self._descend_sequence_item(key, sequence.item_definitions[key])

        raise SourceAccessError("cannot select a child from a scalar value", self.range)

    def children(self) -> tuple[SourceCursor, ...]:
        """Return all effective children in stable container order.

        Mapping children follow constructed first-key order with last values;
        sequence children follow index order. Scalars return an empty tuple.

        Returns:
            `tuple[SourceCursor, ...]`:
                The mapping values or sequence items as child cursors.

        Raises:
            `SourceAccessError`:
                Raised when traversal continues beyond a recursive alias cut.
        """

        self._ensure_traversable()
        if self._projected_entries is not None:
            return tuple(
                self._projected_child(entry)
                for entry in self._projected_entries
            )
        if self._sequence_sources is not None:
            return tuple(
                self._aligned_sequence_child(index)
                for index in range(len(self._sequence_sources))
            )
        if self._overlay or self._document is None:
            if type(self._value) is dict:
                return tuple(
                    SourceCursor.virtual(value)
                    for value in self._value.values()
                )
            if type(self._value) in (list, tuple):
                return tuple(SourceCursor.virtual(value) for value in self._value)
            return ()

        mapping = self._mapping_definition()
        if mapping is not None:
            children = []
            for key in self._value:
                occurrence = self._effective_occurrence(key)
                if occurrence is not None:
                    children.append(self._descend_mapping_occurrence(occurrence))
            return tuple(children)
        sequence = self._sequence_definition()
        if sequence is not None:
            return tuple(
                self._descend_sequence_item(index, definition_id)
                for index, definition_id in enumerate(sequence.item_definitions)
            )
        return ()

    def mapping_occurrences(
        self,
        key: Any = _NO_FILTER,
    ) -> tuple[SourceCursor, ...]:
        """Return raw flattened mapping-pair occurrences in pair order.

        Unlike `children`, this method retains ignored duplicates and inherited
        merge pairs. Passing `None` filters an authored YAML null key; omitting
        `key` disables filtering. Sequences and scalars return an empty tuple.

        Args:
            key (`Any`, *optional*):
                Constructed key to filter by. Omission returns every pair.

        Returns:
            `tuple[SourceCursor, ...]`:
                Occurrence-local value cursors in flattened pair order.

        Raises:
            `SourceAccessError`:
                Raised when traversal continues beyond a recursive alias cut.
        """

        self._ensure_traversable()
        if self._projected_entries is not None:
            return tuple(
                self._projected_child(entry)
                for entry in self._projected_entries
                if key is _NO_FILTER or self._keys_equal(entry.key, key)
            )
        if self._overlay or self._document is None:
            if type(self._value) is not dict:
                return ()
            return tuple(
                SourceCursor.virtual(value)
                for constructed_key, value in self._value.items()
                if key is _NO_FILTER or self._keys_equal(constructed_key, key)
            )

        definition = self._mapping_definition()
        if definition is None:
            return ()
        if key is not _NO_FILTER and not self._mapping_contains(self._value, key):
            return ()
        return tuple(
            self._descend_mapping_occurrence(occurrence)
            for occurrence in self._iter_flat_occurrences(
                definition.pair_sequence
            )
            if key is _NO_FILTER
            or self._keys_equal(occurrence.arc.constructed_key, key)
        )

    @staticmethod
    def _path_keys_equal(left: Any, right: Any) -> bool:
        """Compare replayed physical keys without cross-type coercion."""

        if left is right:
            return True
        if type(left) is not type(right):
            return False
        try:
            return bool(left == right)
        except (TypeError, ValueError):
            return False

    def _descend_occurrence_step(self, step: _MapStep | _SeqStep) -> SourceCursor:
        """Replay one already-validated physical occurrence step."""

        self._ensure_traversable()
        if isinstance(step, _MapStep):
            definition = self._mapping_definition()
            if (
                definition is None
                or not 0
                <= step.pair_ordinal
                < definition.pair_sequence.flat_pair_count
            ):
                raise SourceAccessError("invalid mapping occurrence path", self.range)
            arc = self._flat_arc_at(
                definition.pair_sequence,
                step.pair_ordinal,
            )
            if not self._path_keys_equal(arc.constructed_key, step.constructed_key):
                raise SourceAccessError(
                    "mapping occurrence path key does not match",
                    self.range,
                )
            return self._descend_mapping_occurrence(
                _MapOccurrence(arc, step.pair_ordinal)
            )

        definition = self._sequence_definition()
        if definition is None or not 0 <= step.index < len(definition.item_definitions):
            raise SourceAccessError("invalid sequence occurrence path", self.range)
        return self._descend_sequence_item(
            step.index,
            definition.item_definitions[step.index],
        )

    def has(self, key: Any) -> bool:
        """Return whether this cursor exposes an effective mapping key.

        Args:
            key (`Any`):
                Constructed mapping key to test.

        Returns:
            `bool`:
                `True` when the mapping contains `key`; `False` for absent keys
                and non-mapping values.

        Raises:
            `SourceAccessError`:
                Raised when traversal continues beyond a recursive alias cut.
        """

        self._ensure_traversable()
        if self._projected_entries is not None:
            return self._projected_entry(key) is not None
        if self._overlay or self._document is None:
            if type(self._value) is not dict:
                return False
            try:
                return key in self._value
            except TypeError:
                return False
        return self._effective_occurrence(key) is not None

    def get(self, key: Any, default: Any = None) -> Any:
        """Return a defensive mapping value or fallback copy.

        Args:
            key (`Any`):
                Constructed mapping key to retrieve.
            default (`Any`, *optional*, defaults to `None`):
                Value copied and returned when `key` is absent.

        Returns:
            `Any`:
                A fresh deep copy of the mapping value or `default`.

        Raises:
            `SourceAccessError`:
                Raised when traversal continues beyond a recursive alias cut.
        """

        if not self.has(key):
            return copy.deepcopy(default)
        return self.child(key).value

    def str(self, key: Any) -> str:
        """Return a required string mapping field.

        Args:
            key (`Any`):
                Constructed key of the required field.

        Returns:
            `str`:
                The immutable string value.

        Raises:
            `SourceAccessError`:
                Raised when the field is absent, is not an exact `str`, or
                cannot be traversed.
        """

        child = self.child(key)
        value = child._value
        if type(value) is not str:
            raise SourceAccessError(
                f"field {key!r} must be a string, got {type(value).__name__}",
                child.range,
            )
        return value

    def int(self, key: Any) -> int:
        """Return a required integer mapping field, rejecting booleans.

        Args:
            key (`Any`):
                Constructed key of the required field.

        Returns:
            `int`:
                The exact integer value.

        Raises:
            `SourceAccessError`:
                Raised when the field is absent, is not an exact `int`, or
                cannot be traversed.
        """

        child = self.child(key)
        value = child._value
        if type(value) is not int:
            raise SourceAccessError(
                f"field {key!r} must be an integer, got {type(value).__name__}",
                child.range,
            )
        return value

    def mapping(self, key: Any) -> dict[Any, Any]:
        """Return a defensive copy of a required mapping field.

        Args:
            key (`Any`):
                Constructed key of the required field.

        Returns:
            `dict[Any, Any]`:
                A fresh deep copy of the exact built-in mapping.

        Raises:
            `SourceAccessError`:
                Raised when the field is absent, is not an exact `dict`, or
                cannot be traversed.
        """

        child = self.child(key)
        value = child._value
        if type(value) is not dict:
            raise SourceAccessError(
                f"field {key!r} must be a mapping, got {type(value).__name__}",
                child.range,
            )
        return _snapshot_safe_value(value)

    def sequence(self, key: Any) -> list[Any]:
        """Return a defensive copy of a required sequence field.

        Args:
            key (`Any`):
                Constructed key of the required field.

        Returns:
            `list[Any]`:
                A fresh deep copy of the exact built-in list.

        Raises:
            `SourceAccessError`:
                Raised when the field is absent, is not an exact `list`, or
                cannot be traversed.
        """

        child = self.child(key)
        value = child._value
        if type(value) is not list:
            raise SourceAccessError(
                f"field {key!r} must be a sequence, got {type(value).__name__}",
                child.range,
            )
        return _snapshot_safe_value(value)

    def range_of(self, key: Any) -> SourceRange | None:
        """Return the effective child value range, requiring the field.

        Args:
            key (`Any`):
                Mapping key or sequence index selecting the child.

        Returns:
            `SourceRange | None`:
                The child's value range, or `None` when source-free.

        Raises:
            `SourceAccessError`:
                Raised when `key` cannot be selected or traversed.
        """

        return self.child(key).range

    def key_range_of(self, key: Any) -> SourceRange | None:
        """Return the effective mapping child's key range.

        Args:
            key (`Any`):
                Constructed mapping key to locate.

        Returns:
            `SourceRange | None`:
                The selecting key range, or `None` when absent or source-free.

        Raises:
            `SourceAccessError`:
                Raised when traversal continues beyond a recursive alias cut.
        """

        self._ensure_traversable()
        if self._projected_entries is not None:
            entry = self._projected_entry(key)
            return self._projected_child(entry).key_range if entry is not None else None
        if self._overlay or self._document is None:
            return None
        occurrence = self._effective_occurrence(key)
        if occurrence is None:
            return None
        return self._document._definitions[occurrence.arc.key_definition].range

    def _optional_child(self, key: Any) -> SourceCursor | None:
        """Return an effective child, suppressing best-effort range failures."""

        try:
            if self.has(key):
                return self.child(key)
        except SourceAccessError:
            pass
        return None

    def _source_scalar_definition(self) -> _ScalarDef | None:
        """Return the retained scalar definition behind this cursor, if any."""

        if self._document is None or self._definition is None:
            return None
        definition = self._document._definitions[self._definition]
        return definition if isinstance(definition, _ScalarDef) else None

    def _source_key_text(self) -> str | None:
        """Return the authored scalar text of this cursor's selecting key."""

        if self._document is None or self._key_definition is None:
            return None
        definition = self._document._definitions[self._key_definition]
        if isinstance(definition, _ScalarDef):
            return definition.raw
        return None

    @property
    def source_key_value(self) -> Any:
        """Return the constructed key selecting this physical occurrence.

        Returns:
            `Any`:
                Defensive key copy, or ``None`` for roots, sequence items, and
                source-free values. Callers that distinguish a YAML null key
                should pair this with ``key_range``.
        """

        if self._document is None or self._key_definition is None:
            return None
        definition = self._document._definitions[self._key_definition]
        if not isinstance(definition, _ScalarDef):
            return None
        return copy.deepcopy(definition.value)

    def _own_item_ranges(self) -> dict[str, SourceRange]:
        """Collect scalar item ranges from this sequence cursor."""

        self._ensure_traversable()
        ranges: dict[str, SourceRange] = {}
        if self._sequence_sources is not None:
            for source in self._sequence_sources:
                if source is None:
                    continue
                definition = source._source_scalar_definition()
                if (
                    definition is not None
                    and definition.raw is not None
                    and source.range is not None
                ):
                    ranges[definition.raw] = source.range
            return ranges

        sequence = self._sequence_definition()
        if sequence is None or self._document is None:
            return ranges
        for definition_id in sequence.item_definitions:
            definition = self._document._definitions[definition_id]
            if (
                isinstance(definition, _ScalarDef)
                and definition.raw is not None
                and definition.range is not None
            ):
                ranges[definition.raw] = definition.range
        return ranges

    def item_ranges(self, key: Any) -> dict[str, SourceRange]:
        """Return scalar sequence-item ranges keyed by authored text.

        Args:
            key (`Any`):
                Constructed key of the sequence field.

        Returns:
            `dict[str, SourceRange]`:
                Located scalar text to range. Duplicate texts keep the last
                range; absent, unlocated, or non-sequence fields return `{}`.

        Raises:
            `SourceAccessError`:
                Raised when traversal continues beyond a recursive alias cut.
        """

        self._ensure_traversable()
        child = self._optional_child(key)
        return child._own_item_ranges() if child is not None else {}

    def _own_item_range_list(self) -> list[SourceRange | None]:
        """Collect positional scalar item ranges from this sequence cursor."""

        self._ensure_traversable()
        if self._sequence_sources is not None:
            return [
                (
                    source.range
                    if source is not None
                    and source._source_scalar_definition() is not None
                    else None
                )
                for source in self._sequence_sources
            ]

        sequence = self._sequence_definition()
        if sequence is None or self._document is None:
            return []
        return [
            definition.range if isinstance(definition, _ScalarDef) else None
            for definition in (
                self._document._definitions[definition_id]
                for definition_id in sequence.item_definitions
            )
        ]

    def item_range_list(self, key: Any) -> list[SourceRange | None]:
        """Return positional scalar sequence-item ranges.

        Args:
            key (`Any`):
                Constructed key of the sequence field.

        Returns:
            `list[SourceRange | None]`:
                One range per authored item, preserving duplicates and using
                `None` for non-scalars. Missing or unlocated fields return `[]`.

        Raises:
            `SourceAccessError`:
                Raised when traversal continues beyond a recursive alias cut.
        """

        self._ensure_traversable()
        child = self._optional_child(key)
        return child._own_item_range_list() if child is not None else []

    def _own_mapping_value_ranges(self) -> dict[str, SourceRange]:
        """Collect child value ranges from this mapping cursor."""

        self._ensure_traversable()
        if self._projected_entries is not None:
            ranges: dict[str, SourceRange] = {}
            for entry in self._projected_entries:
                source = entry.source
                if source is None or source.range is None:
                    continue
                name = (
                    entry.key
                    if type(entry.key) is str
                    else source._source_key_text()
                )
                if name is not None:
                    ranges[name] = source.range
            return ranges

        definition = self._mapping_definition()
        if definition is None or self._document is None:
            return {}
        last_ranges: dict[str, SourceRange] = {}
        for occurrence in self._iter_unique_occurrences(
            definition.pair_sequence,
            reverse=True,
        ):
            arc = occurrence.arc
            child_range = self._document._definitions[arc.value_definition].range
            if (
                arc.raw_key is not None
                and child_range is not None
                and arc.raw_key not in last_ranges
            ):
                last_ranges[arc.raw_key] = child_range

        ranges: dict[str, SourceRange] = {}
        for occurrence in self._iter_unique_occurrences(definition.pair_sequence):
            raw_key = occurrence.arc.raw_key
            if raw_key is not None and raw_key not in ranges:
                located = last_ranges.get(raw_key)
                if located is not None:
                    ranges[raw_key] = located
        return ranges

    def mapping_value_ranges(self, key: Any) -> dict[str, SourceRange]:
        """Return child value ranges for one mapping field.

        Args:
            key (`Any`):
                Constructed key of the mapping field.

        Returns:
            `dict[str, SourceRange]`:
                Authored child key text to value range. Missing, unlocated, or
                non-mapping fields return `{}`.

        Raises:
            `SourceAccessError`:
                Raised when traversal continues beyond a recursive alias cut.
        """

        self._ensure_traversable()
        child = self._optional_child(key)
        return child._own_mapping_value_ranges() if child is not None else {}

    def _own_sequence_field_ranges(self, field: Any) -> list[SourceRange | None]:
        """Collect one nested field range per item in this sequence cursor."""

        self._ensure_traversable()
        if self._sequence_sources is not None:
            ranges: list[SourceRange | None] = []
            for index in range(len(self._sequence_sources)):
                item = self._aligned_sequence_child(index)
                child = item._optional_child(field)
                ranges.append(child.range if child is not None else None)
            return ranges

        sequence = self._sequence_definition()
        if sequence is None or self._document is None:
            return []
        ranges = []
        for definition_id in sequence.item_definitions:
            definition = self._document._definitions[definition_id]
            selected: SourceRange | None = None
            if (
                isinstance(definition, _MappingDef)
                and self._mapping_contains(definition.value, field)
            ):
                occurrence = self._effective_occurrence_in_definition(
                    definition_id,
                    field,
                )
                if occurrence is not None:
                    selected = self._document._definitions[
                        occurrence.arc.value_definition
                    ].range
            ranges.append(selected)
        return ranges

    def sequence_field_ranges(
        self,
        sequence_key: Any,
        field: Any,
    ) -> list[SourceRange | None]:
        """Return one nested mapping-field range per sequence item.

        Args:
            sequence_key (`Any`):
                Constructed key of the sequence field.
            field (`Any`):
                Constructed key to locate within each mapping item.

        Returns:
            `list[SourceRange | None]`:
                One value range per sequence item in order, using `None` when
                an item is not a mapping or lacks `field`. Missing, unlocated,
                or non-sequence fields return `[]`.

        Raises:
            `SourceAccessError`:
                Raised when traversal continues beyond a recursive alias cut.
        """

        self._ensure_traversable()
        child = self._optional_child(sequence_key)
        return child._own_sequence_field_ranges(field) if child is not None else []

    def with_value(self, value: Any) -> SourceCursor:
        """Expose a value snapshot while retaining compatible source metadata.

        Mapping keys and sequence indices that still exist are aligned with
        their physical child cursors. New nested values remain source-free.

        Args:
            value (`Any`):
                Replacement in the exact recursive SafeLoader value algebra.

        Returns:
            `SourceCursor`:
                A cursor with this cursor's document, path, and ranges, plus
                aligned nested sources where shapes still match.

        Raises:
            `TypeError`:
                Raised when `value` contains an unsupported type or subclass.
        """

        snapshot = _snapshot_safe_value(value)
        projected_entries: tuple[_ProjectedEntry, ...] | None = None
        sequence_sources: tuple[SourceCursor | None, ...] | None = None
        if type(snapshot) is dict:
            entries: list[_ProjectedEntry] = []
            for key in snapshot:
                source = None
                if (
                    self._document is not None
                    and not self._cycle_cut
                    and type(self._value) is dict
                ):
                    try:
                        if self.has(key):
                            source = self.child(key)
                    except SourceAccessError:
                        source = None
                entries.append(_ProjectedEntry(key, source))
            projected_entries = tuple(entries)
        elif type(snapshot) in (list, tuple):
            sources: list[SourceCursor | None] = []
            for index in range(len(snapshot)):
                source = None
                if (
                    self._document is not None
                    and not self._cycle_cut
                    and type(self._value) in (list, tuple)
                    and index < len(self._value)
                ):
                    try:
                        source = self.child(index)
                    except SourceAccessError:
                        source = None
                sources.append(source)
            sequence_sources = tuple(sources)

        return SourceCursor(
            _document=self._document,
            _definition=self._definition,
            _value=snapshot,
            _path_link=self._path_link,
            _key_definition=self._key_definition,
            _cycle_cut=self._cycle_cut,
            _overlay=self._document is not None,
            _projected_entries=projected_entries,
            _projected_index=(
                _KeyIndex.last_positions(entry.key for entry in projected_entries)
                if projected_entries is not None
                else None
            ),
            _sequence_sources=sequence_sources,
        )

    def project_mapping(
        self,
        entries: Mapping[Any, tuple[Any, SourceCursor | None]],
    ) -> SourceCursor:
        """Return a pure mapping projection with explicit child sources.

        Args:
            entries (`Mapping[Any, tuple[Any, SourceCursor | None]]`):
                Displayed keys mapped to `(value, source)` pairs. `source`
                retains a child's document, path, and ranges; `None` makes that
                child virtual. Insertion order becomes projection order.

        Returns:
            `SourceCursor`:
                A snapshotted mapping cursor retaining this parent's source.

        Raises:
            `TypeError`:
                Raised for a malformed pair, invalid source, or unsupported
                displayed key or value type.
        """

        displayed: dict[Any, Any] = {}
        sources: dict[Any, SourceCursor | None] = {}
        for key, pair in entries.items():
            if type(pair) is not tuple or len(pair) != 2:
                raise TypeError("projected entries must be (value, source) tuples")
            value, source = pair
            if source is not None and not isinstance(source, SourceCursor):
                raise TypeError("projected entry source must be a SourceCursor or None")
            displayed[key] = value
            sources[key] = source

        snapshot = _snapshot_safe_value(displayed)
        projected_entries = tuple(
            _ProjectedEntry(key, sources[key])
            for key in snapshot
        )
        return SourceCursor(
            _document=self._document,
            _definition=self._definition,
            _value=snapshot,
            _path_link=self._path_link,
            _key_definition=self._key_definition,
            _cycle_cut=self._cycle_cut,
            _overlay=self._document is not None,
            _projected_entries=projected_entries,
            _projected_index=_KeyIndex.last_positions(
                entry.key for entry in projected_entries
            ),
            _sequence_sources=None,
        )

    def renamed(self, names: Mapping[Any, Any]) -> SourceCursor:
        """Return a pure key-renamed mapping projection.

        Args:
            names (`Mapping[Any, Any]`):
                Old constructed keys mapped to replacement displayed keys.
                Unlisted keys keep their original names and order.

        Returns:
            `SourceCursor`:
                A projection whose children retain their original sources.

        Raises:
            `SourceAccessError`:
                Raised when this cursor is not a traversable mapping.
            `TypeError`:
                Raised when a replacement key is outside the SafeLoader value
                algebra.
        """

        self._ensure_traversable()
        if type(self._value) is not dict:
            raise SourceAccessError(
                "cannot rename keys on a non-mapping value",
                self.range,
            )
        entries: dict[Any, tuple[Any, SourceCursor | None]] = {}
        for key, value in self._value.items():
            renamed_key = names.get(key, key)
            entries[renamed_key] = (value, self.child(key))
        return self.project_mapping(entries)
