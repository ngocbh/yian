"""One-pass parsing and private retention of a Compose YAML document."""

from __future__ import annotations

from collections.abc import Hashable, Iterator
from dataclasses import dataclass, field
from typing import Any

import yaml
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode, SequenceNode

from agent_composer.compose.source.common import (
    OccurrencePath,
    SourceSyntaxError,
    _ArcSequence,
    _Definition,
    _KeyIndex,
    _MapArc,
    _MapOccurrence,
    _MappingDef,
    _ScalarDef,
    _SequenceDef,
)
from agent_composer.compose.source.cursor import SourceCursor
from agent_composer.events import SourceRange


_MISSING = object()


@dataclass(frozen=True, slots=True, eq=False)
class _PairSequence:
    """Persistent raw-pair leaf or concatenation used during YAML construction."""

    pairs: tuple[tuple[yaml.Node, yaml.Node], ...]
    parts: tuple[_PairSequence, ...]
    part_offsets: tuple[int, ...]
    flat_pair_count: int


@dataclass(slots=True, eq=False)
class _WorkingPairs:
    """One non-mutating simulation of a MappingNode's current pair list."""

    pairs: list[tuple[yaml.Node, yaml.Node]]


@dataclass(slots=True)
class _FlattenFrame:
    """One suspended merge-flatten invocation in the iterative planner."""

    node: MappingNode
    state: _WorkingPairs
    prefix: list[_PairSequence]
    index: int = 0
    pending_sources: tuple[yaml.Node, ...] | None = None
    pending_results: list[_PairSequence] | None = None


_EMPTY_PAIRS = _PairSequence((), (), (), 0)


def _pair_leaf(
    pairs: tuple[tuple[yaml.Node, yaml.Node], ...],
) -> _PairSequence:
    """Return a compact leaf for non-merge raw pairs."""

    return _PairSequence(pairs, (), (), len(pairs)) if pairs else _EMPTY_PAIRS


def _pair_concat(parts: list[_PairSequence]) -> _PairSequence:
    """Return a compact concatenation without expanding shared child plans."""

    retained = tuple(part for part in parts if part.flat_pair_count)
    if not retained:
        return _EMPTY_PAIRS
    if len(retained) == 1:
        return retained[0]
    offsets: list[int] = []
    count = 0
    for part in retained:
        offsets.append(count)
        count += part.flat_pair_count
    return _PairSequence((), retained, tuple(offsets), count)


def _pair_summary_frontier(
    sequence: _PairSequence,
    summaries: dict[_PairSequence, dict[Any, Any]],
    *,
    reverse: bool = False,
) -> Iterator[_PairSequence | tuple[yaml.Node, yaml.Node]]:
    """Yield cached summaries or unique raw pairs in logical traversal order."""

    seen: set[_PairSequence] = set()
    stack = [sequence]
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        if current in summaries:
            yield current
            continue
        if current.parts:
            indices = range(len(current.parts))
            if not reverse:
                indices = reversed(tuple(indices))
            for index in indices:
                stack.append(current.parts[index])
            continue
        indices = range(len(current.pairs) - 1, -1, -1) if reverse else range(
            len(current.pairs)
        )
        for index in indices:
            yield current.pairs[index]


class _DocumentLoader(yaml.SafeLoader):
    """SafeLoader that retains raw merge topology and exact constructed values."""

    def __init__(self, stream: str):
        self._source_values: dict[int, Any] = {}
        self._effective_mappings: dict[MappingNode, dict[Any, Any]] = {}
        self._mapping_states: dict[MappingNode, _WorkingPairs | _PairSequence] = {}
        self._sequence_values: dict[_PairSequence, dict[Any, Any]] = {}
        self._forced_string_nodes: set[yaml.Node] = set()
        super().__init__(stream)

    def construct_object(self, node: yaml.Node, deep: bool = False) -> Any:
        """Construct normally while retaining a temporary node/value identity map."""

        if node in self._forced_string_nodes and node not in self.constructed_objects:
            value = self.construct_scalar(node)
            self.constructed_objects[node] = value
        else:
            value = super().construct_object(node, deep=deep)
        self._source_values[id(node)] = value
        return value

    def _construct_key(self, node: yaml.Node, *, deep: bool) -> Any:
        """Construct a mapping key without mutating YAML's special value tag."""

        return self.construct_object(node, deep=deep)

    def _flatten_pairs(self, node: MappingNode) -> _PairSequence:
        """Simulate PyYAML merge flattening as a persistent compact pair rope."""

        state = self._mapping_states.get(node)
        if isinstance(state, _PairSequence):
            return state
        if state is None:
            state = _WorkingPairs(list(node.value))
            self._mapping_states[node] = state

        stack = [_FlattenFrame(node=node, state=state, prefix=[])]
        returned: _PairSequence | None = None
        while stack:
            frame = stack[-1]

            if returned is not None:
                if frame.pending_results is None:
                    raise AssertionError("flatten child returned without a caller")
                frame.pending_results.append(returned)
                returned = None

            if frame.pending_sources is not None:
                results = frame.pending_results
                if results is None:
                    raise AssertionError("flatten sources have no result buffer")
                if len(results) < len(frame.pending_sources):
                    source_node = frame.pending_sources[len(results)]
                    if not isinstance(source_node, MappingNode):
                        raise ConstructorError(
                            "while constructing a mapping",
                            frame.node.start_mark,
                            "expected a mapping for merging, but found "
                            f"{source_node.id}",
                            source_node.start_mark,
                        )
                    source_state = self._mapping_states.get(source_node)
                    if isinstance(source_state, _PairSequence):
                        results.append(source_state)
                        continue
                    if source_state is None:
                        source_state = _WorkingPairs(list(source_node.value))
                        self._mapping_states[source_node] = source_state
                    stack.append(
                        _FlattenFrame(
                            node=source_node,
                            state=source_state,
                            prefix=[],
                        )
                    )
                    continue

                frame.prefix.extend(reversed(results))
                frame.pending_sources = None
                frame.pending_results = None

                current = self._mapping_states[frame.node]
                if isinstance(current, _PairSequence):
                    returned = _pair_concat([*frame.prefix, current])
                    self._mapping_states[frame.node] = returned
                    stack.pop()
                continue

            if frame.index < len(frame.state.pairs):
                key_node, value_node = frame.state.pairs[frame.index]
                if key_node.tag == "tag:yaml.org,2002:merge":
                    del frame.state.pairs[frame.index]
                    if isinstance(value_node, MappingNode):
                        frame.pending_sources = (value_node,)
                    elif isinstance(value_node, SequenceNode):
                        frame.pending_sources = tuple(value_node.value)
                    else:
                        raise ConstructorError(
                            "while constructing a mapping",
                            frame.node.start_mark,
                            "expected a mapping or list of mappings for merging, but "
                            f"found {value_node.id}",
                            value_node.start_mark,
                        )
                    frame.pending_results = []
                    continue
                if key_node.tag == "tag:yaml.org,2002:value":
                    self._forced_string_nodes.add(key_node)
                frame.index += 1
                continue

            returned = _pair_concat(
                [*frame.prefix, _pair_leaf(tuple(frame.state.pairs))]
            )
            self._mapping_states[frame.node] = returned
            stack.pop()

        if returned is None:
            raise AssertionError("flatten planner completed without a result")
        return returned

    def _construct_pair_sequence(
        self,
        sequence: _PairSequence,
        *,
        deep: bool,
        context: MappingNode,
    ) -> dict[Any, Any]:
        """Fold one compact pair rope into SafeLoader's ordered dict semantics."""

        cached = self._sequence_values.get(sequence)
        if cached is not None:
            return cached
        first_keys: dict[Any, None] = {}
        for item in _pair_summary_frontier(sequence, self._sequence_values):
            if isinstance(item, _PairSequence):
                for key in self._sequence_values[item]:
                    first_keys.setdefault(key, None)
                continue
            key_node, value_node = item
            key = self._construct_key(key_node, deep=deep)
            if not isinstance(key, Hashable):
                raise ConstructorError(
                    "while constructing a mapping",
                    context.start_mark,
                    "found unhashable key",
                    key_node.start_mark,
                )
            self.construct_object(value_node, deep=deep)
            first_keys.setdefault(key, None)

        last_values: dict[Any, Any] = {}
        for item in _pair_summary_frontier(
            sequence,
            self._sequence_values,
            reverse=True,
        ):
            if isinstance(item, _PairSequence):
                for key, value in self._sequence_values[item].items():
                    last_values.setdefault(key, value)
                continue
            key_node, value_node = item
            key = self._construct_key(key_node, deep=deep)
            if key not in last_values:
                last_values[key] = self.construct_object(value_node, deep=deep)

        result = {key: last_values[key] for key in first_keys}
        self._sequence_values[sequence] = result
        return result

    def construct_mapping(self, node: yaml.Node, deep: bool = False) -> dict[Any, Any]:
        """Construct a mapping without expanding merge pairs into ``node.value``."""

        if not isinstance(node, MappingNode):
            raise ConstructorError(
                None,
                None,
                f"expected a mapping node, but found {node.id}",
                node.start_mark,
            )
        sequence = self._flatten_pairs(node)
        result = self._construct_pair_sequence(sequence, deep=deep, context=node)
        self._effective_mappings[node] = result
        return result


def _mark_to_range(node: yaml.Node | None) -> SourceRange | None:
    """Convert one composed node's half-open marks to 1-based coordinates."""

    if node is None:
        return None
    return SourceRange(
        node.start_mark.line + 1,
        node.start_mark.column + 1,
        node.end_mark.line + 1,
        node.end_mark.column + 1,
    )


def _build_definitions(
    root_node: yaml.Node | None,
    root_value: Any,
    node_values: dict[int, Any],
    mapping_values: dict[int, dict[Any, Any]] | None = None,
    pair_sequences: dict[int, _PairSequence] | None = None,
) -> tuple[tuple[_Definition, ...], int]:
    """Intern every reachable YAML node once into a frozen definition DAG.

    Args:
        root_node (`yaml.Node | None`):
            Root of the already-composed YAML graph, or ``None`` for an empty
            document.
        root_value (`Any`):
            Constructed value aligned with ``root_node``.
        node_values (`dict[int, Any]`):
            Exact constructed value for each YAML-node identity.
        mapping_values (`dict[int, dict[Any, Any]] | None`, *optional*):
            Effective mapping values retained by the document loader, keyed by
            YAML-node identity. Absent for non-mapping nodes.
        pair_sequences (`dict[int, _PairSequence] | None`, *optional*):
            Compact flattened merge plans retained by the document loader,
            keyed by mapping-node identity.

    Returns:
        `tuple[tuple[_Definition, ...], int]`:
            Frozen definitions and the root definition id.
    """

    if root_node is None:
        return (_ScalarDef(root_value, None, None),), 0

    definitions: list[_Definition | None] = []
    ids: dict[int, int] = {}
    converted_sequences: dict[_PairSequence, _ArcSequence] = {}
    effective_sequences: dict[
        _ArcSequence,
        tuple[dict[Any, _MapOccurrence], _KeyIndex],
    ] = {}

    def convert_sequence(sequence: _PairSequence) -> _ArcSequence:
        """Convert one raw pair rope to an identity-shared definition rope."""

        existing = converted_sequences.get(sequence)
        if existing is not None:
            return existing
        stack: list[tuple[_PairSequence, bool]] = [(sequence, False)]
        while stack:
            current, ready = stack.pop()
            if current in converted_sequences:
                continue
            if current.parts and not ready:
                stack.append((current, True))
                for part in reversed(current.parts):
                    if part not in converted_sequences:
                        stack.append((part, False))
                continue
            if current.parts:
                converted = _ArcSequence(
                    arcs=(),
                    parts=tuple(
                        converted_sequences[part] for part in current.parts
                    ),
                    part_offsets=current.part_offsets,
                    flat_pair_count=current.flat_pair_count,
                )
            else:
                arcs = tuple(
                    _MapArc(
                        constructed_key=node_values[id(key_node)],
                        raw_key=(
                            key_node.value
                            if isinstance(key_node, yaml.ScalarNode)
                            else None
                        ),
                        key_definition=intern(key_node),
                        value_definition=intern(value_node),
                    )
                    for key_node, value_node in current.pairs
                )
                converted = _ArcSequence(
                    arcs=arcs,
                    parts=(),
                    part_offsets=(),
                    flat_pair_count=len(arcs),
                )
            converted_sequences[current] = converted
        return converted_sequences[sequence]

    def effective_sequence(
        sequence: _ArcSequence,
    ) -> _KeyIndex:
        """Return last occurrences while reusing completed mapping summaries."""

        existing = effective_sequences.get(sequence)
        if existing is not None:
            return existing[1]
        winners: dict[Any, _MapOccurrence] = {}
        seen: set[_ArcSequence] = set()
        stack: list[tuple[_ArcSequence, int]] = [(sequence, 0)]
        while stack:
            current, base = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            cached = effective_sequences.get(current)
            if cached is not None:
                for key, occurrence in cached[0].items():
                    if key not in winners:
                        winners[key] = _MapOccurrence(
                            occurrence.arc,
                            base + occurrence.pair_ordinal,
                        )
                continue
            if current.parts:
                for index in range(len(current.parts)):
                    stack.append(
                        (
                            current.parts[index],
                            base + current.part_offsets[index],
                        )
                    )
                continue
            for index in range(len(current.arcs) - 1, -1, -1):
                arc = current.arcs[index]
                if arc.constructed_key not in winners:
                    winners[arc.constructed_key] = _MapOccurrence(
                        arc,
                        base + index,
                    )
        result = _KeyIndex(
            (occurrence.arc.constructed_key, occurrence)
            for occurrence in winners.values()
        )
        effective_sequences[sequence] = (winners, result)
        return result

    def intern(node: yaml.Node, derived_value: Any = _MISSING) -> int:
        """Return the stable definition id for one reachable YAML node."""

        identity = id(node)
        existing = ids.get(identity)
        if existing is not None:
            return existing

        definition_id = len(definitions)
        ids[identity] = definition_id
        definitions.append(None)

        value = node_values.get(identity, derived_value)
        if (
            value is _MISSING
            and mapping_values is not None
            and isinstance(node, MappingNode)
        ):
            value = mapping_values.get(identity, _MISSING)
        if value is _MISSING:
            raise TypeError("constructed value missing for retained YAML node")
        source_range = _mark_to_range(node)

        if isinstance(node, MappingNode):
            sequence = (
                pair_sequences.get(identity)
                if pair_sequences is not None
                else None
            )
            if sequence is None:
                sequence = _pair_leaf(tuple(node.value))
            pair_sequence = convert_sequence(sequence)
            definitions[definition_id] = _MappingDef(
                value=value,
                range=source_range,
                pair_sequence=pair_sequence,
                effective_index=effective_sequence(pair_sequence),
            )
        elif isinstance(node, yaml.SequenceNode):
            item_definitions: list[int] = []
            for index, item_node in enumerate(node.value):
                item_value = _MISSING
                if type(value) in (list, tuple) and index < len(value):
                    item_value = value[index]
                item_definitions.append(intern(item_node, item_value))
            definitions[definition_id] = _SequenceDef(
                value=value,
                range=source_range,
                item_definitions=tuple(item_definitions),
            )
        else:
            definitions[definition_id] = _ScalarDef(
                value=value,
                range=source_range,
                raw=(node.value if isinstance(node, yaml.ScalarNode) else None),
            )
        return definition_id

    root_definition = intern(root_node, root_value)
    return (
        tuple(definition for definition in definitions if definition is not None),
        root_definition,
    )


def _offset_to_line_column(text: str, offset: int) -> tuple[int, int]:
    """Convert a string offset to 1-based coordinates using YAML line breaks."""

    line = 1
    column = 1
    index = 0
    while index < offset:
        character = text[index]
        if character == "\r":
            if index + 1 < offset and text[index + 1] == "\n":
                index += 2
            else:
                index += 1
            line += 1
            column = 1
        elif character in ("\n", "\x85", "\u2028", "\u2029"):
            index += 1
            line += 1
            column = 1
        else:
            index += 1
            column += 1
    return line, column


def _problem_range(
    error: yaml.YAMLError | yaml.reader.ReaderError,
    text: str,
) -> SourceRange | None:
    """Return a one-character range at PyYAML's best available problem mark."""

    if isinstance(error, yaml.reader.ReaderError):
        line, column = _offset_to_line_column(text, error.position)
        return SourceRange(line, column, line, column + 1)

    mark = getattr(error, "problem_mark", None)
    if mark is None:
        mark = getattr(error, "context_mark", None)
    if mark is None:
        return None
    line = mark.line + 1
    column = mark.column + 1
    return SourceRange(line, column, line, column + 1)


@dataclass(frozen=True, slots=True)
class SourceDocument:
    """One safely constructed YAML document with its source graph kept private.

    ``text`` is the original YAML and ``label`` identifies it to later
    diagnostics.  The constructed value, composed node tree, and definition DAG
    remain private; navigation is exposed only through source cursors.
    """

    text: str
    label: str
    _value: Any = field(repr=False, compare=False)
    _root_node: yaml.Node | None = field(repr=False, compare=False)
    _definitions: tuple[_Definition, ...] = field(repr=False, compare=False)
    _root_definition: int = field(repr=False, compare=False)

    @property
    def root(self) -> SourceCursor:
        """Return a cursor at the document root."""

        return SourceCursor._physical(self, self._root_definition)

    def _resolve_occurrence(self, path: OccurrencePath) -> SourceCursor:
        """Replay one finite physical occurrence path through the definition DAG."""

        cursor = self.root
        for step in path:
            cursor = cursor._descend_occurrence_step(step)
        return cursor

    @classmethod
    def parse(cls, text: str, label: str = "<flow>") -> "SourceDocument":
        """Compose and construct exactly one YAML document with one loader."""

        loader = None
        try:
            loader = _DocumentLoader(text)
            root_node = loader.get_single_node()
            value = (
                loader.construct_document(root_node)
                if root_node is not None
                else None
            )
            definitions, root_definition = _build_definitions(
                root_node,
                value,
                loader._source_values,
                mapping_values={
                    id(node): mapping
                    for node, mapping in loader._effective_mappings.items()
                },
                pair_sequences={
                    id(node): state
                    for node, state in loader._mapping_states.items()
                    if isinstance(state, _PairSequence)
                },
            )
        except (yaml.YAMLError, yaml.reader.ReaderError) as exc:
            raise SourceSyntaxError(str(exc), _problem_range(exc, text)) from exc
        finally:
            if loader is not None:
                loader._source_values = {}
                loader._effective_mappings = {}
                loader._mapping_states = {}
                loader._sequence_values = {}
                loader._forced_string_nodes = set()
                loader.dispose()

        return cls(
            text=text,
            label=label,
            _value=value,
            _root_node=root_node,
            _definitions=definitions,
            _root_definition=root_definition,
        )
