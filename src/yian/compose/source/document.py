"""One-pass parsing and private retention of a Compose YAML document."""

from __future__ import annotations

import re
from collections.abc import Callable, Hashable, Iterator
from dataclasses import dataclass, field
from typing import Any, TypeAlias

import yaml
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode, ScalarNode, SequenceNode
from yaml.tokens import ScalarToken

from yian.compose.source.common import (
    OccurrencePath,
    SourceSyntaxError,
    TaggedNode,
    _ArcSequence,
    _Definition,
    _KeyIndex,
    _MapArc,
    _MapOccurrence,
    _MappingDef,
    _ScalarDef,
    _SequenceDef,
)
from yian.compose.source.cursor import SourceCursor
from yian.compose.source.scalars import _FlowLoader
from yian.diagnostics import SourceRange


_MISSING = object()

_CORE_TAG_PREFIX = "tag:yaml.org,2002:"
_STR_TAG = _CORE_TAG_PREFIX + "str"
_SEQ_TAG = _CORE_TAG_PREFIX + "seq"
_MAP_TAG = _CORE_TAG_PREFIX + "map"
# Keys that YAML resolves to these tags only mean something as untagged keys.
_KEY_ONLY_TAGS = frozenset((_CORE_TAG_PREFIX + "merge", _CORE_TAG_PREFIX + "value"))

_LINE_SPLIT = re.compile("\r\n|[\r\n\x85\u2028\u2029]")
# A tag directly followed by `,`, `}` or `]` inside a flow collection.
_FLOW_TAG = re.compile(r"(?:^|[\s\[{,])![^\s,{}\[\]]*[,}\]]")
_FLOW_TAG_HINT = (
    "hint: after a tag inside `{...}` or `[...]`, put a space before `,`, `}` or "
    "`]`, or use block style"
)


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


class _DocumentLoader(_FlowLoader):
    """Flow loader that retains raw merge topology and exact constructed values."""

    def __init__(self, stream: str):
        self._source_values: dict[int, Any] = {}
        self._effective_mappings: dict[MappingNode, dict[Any, Any]] = {}
        self._mapping_states: dict[MappingNode, _WorkingPairs | _PairSequence] = {}
        self._sequence_values: dict[_PairSequence, dict[Any, Any]] = {}
        self._forced_string_nodes: set[yaml.Node] = set()
        self._block_scalar_tokens: dict[int, ScalarToken] = {}
        self._local_tags: dict[int, str] = {}
        self._alias_ranges: list[SourceRange] = []
        super().__init__(stream)

    def scan_block_scalar(self, style: str) -> ScalarToken:
        """Scan once while retaining the real block-indicator token.

        A composed scalar node can start at an anchor or tag before the block
        indicator, so its start mark cannot identify the physical body.
        """

        token = super().scan_block_scalar(style)
        self._block_scalar_tokens[token.end_mark.index] = token
        return token

    def parse_node(
        self,
        block: bool = False,
        indentless_sequence: bool = False,
    ) -> yaml.Event:
        """Parse one node event, rejecting a bare ``!`` before YAML resolves it away.

        The parser is a state machine, so this check adds no frame per nesting
        level, unlike the recursive compose step.
        """

        event = super().parse_node(block, indentless_sequence)
        if isinstance(event, yaml.AliasEvent):
            # A composed alias is its anchor's node, so where it is written is kept here.
            self._alias_ranges.append(_marks_to_range(event.start_mark, event.end_mark))
        if getattr(event, "tag", None) == "!":
            raise SourceSyntaxError(
                "a tag names a type, e.g. `!int`",
                _marks_to_range(event.start_mark, event.end_mark),
            )
        return event

    def construct_object(self, node: yaml.Node, deep: bool = False) -> Any:
        """Construct normally while retaining a temporary node/value identity map.

        A local tag (``!...``) is recorded and the node is constructed as if it
        were untagged; YAML core tags (``!!...``) keep their YAML meaning.
        """

        tag = node.tag
        if node in self._forced_string_nodes and node not in self.constructed_objects:
            value = self.construct_scalar(node)
            self.constructed_objects[node] = value
        elif tag.startswith("!"):
            self._local_tags[id(node)] = tag[1:]
            node.tag = self._untagged_tag(node)
            try:
                value = super().construct_object(node, deep=deep)
            finally:
                node.tag = tag
        elif not tag.startswith(_CORE_TAG_PREFIX):
            raise _unknown_tag_error(node)
        else:
            value = super().construct_object(node, deep=deep)
        self._source_values[id(node)] = value
        return value

    def _untagged_tag(self, node: yaml.Node) -> str:
        """Return the core tag YAML would give ``node`` if it had no tag."""

        if isinstance(node, SequenceNode):
            return _SEQ_TAG
        if isinstance(node, MappingNode):
            return _MAP_TAG
        if node.style is None:
            resolved = self.resolve(ScalarNode, node.value, (True, False))
            if resolved not in _KEY_ONLY_TAGS:
                return resolved
        return _STR_TAG

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
                    _reject_merge_source_tag(value_node, key_node)
                    if isinstance(value_node, SequenceNode):
                        for source_node in value_node.value:
                            _reject_merge_source_tag(source_node, key_node)
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


def _unknown_tag_error(node: yaml.Node) -> SourceSyntaxError:
    """Return the error for a tag that is neither local nor a YAML core tag."""

    return SourceSyntaxError(
        f"unknown YAML tag {node.tag!r}; a type tag is local, e.g. `!int`",
        _mark_to_range(node),
    )


def _reject_merge_source_tag(node: yaml.Node, merge_key: yaml.Node) -> None:
    """Reject a non-core tag written at a ``<<:`` merge site.

    YAML never constructs a merge source, so a type tag written on the merge
    value, its list, or a list item would check nothing. A node reached
    through an alias is skipped: its tag sits at its anchor and checks the
    value there.
    """

    # An anchor always precedes its aliases, so an aliased node starts before
    # the merge key ends; a node written at the merge site starts after it.
    if node.start_mark.index < merge_key.end_mark.index:
        return
    if node.tag.startswith("!"):
        raise SourceSyntaxError(
            "a tag on a `<<:` merge source checks nothing; tag the mapping that "
            "merges it",
            _mark_to_range(node),
        )
    if not node.tag.startswith(_CORE_TAG_PREFIX):
        raise _unknown_tag_error(node)


def _marks_to_range(start: yaml.Mark, end: yaml.Mark) -> SourceRange:
    """Convert one pair of half-open PyYAML marks to 1-based coordinates."""

    return SourceRange(
        start.line + 1,
        start.column + 1,
        end.line + 1,
        end.column + 1,
    )


def _mark_to_range(node: yaml.Node | None) -> SourceRange | None:
    """Convert one composed node's half-open marks to 1-based coordinates."""

    if node is None:
        return None
    return _marks_to_range(node.start_mark, node.end_mark)


def _alias_key_ranges(
    root_node: yaml.Node | None,
    alias_ranges: list[SourceRange],
) -> dict[tuple[int, int], SourceRange]:
    """Return where each alias mapping key is written, by mapping id and pair position.

    ``alias_ranges`` holds every alias in document order. A walk in document
    order meets a node first where it is defined (an anchor precedes its
    aliases), so each later meeting is the next alias.
    """

    found: dict[tuple[int, int], SourceRange] = {}
    if not alias_ranges:
        return found
    aliases = iter(alias_ranges)
    stack: list[tuple[yaml.Node, tuple[int, int] | None]] = (
        [(root_node, None)] if root_node is not None else []
    )
    seen: set[int] = set()
    while stack:
        node, key_at = stack.pop()
        if id(node) in seen:
            alias_range = next(aliases)
            if key_at is not None:
                found[key_at] = alias_range
            continue
        seen.add(id(node))
        if isinstance(node, SequenceNode):
            stack.extend((item, None) for item in reversed(node.value))
        elif isinstance(node, MappingNode):
            for position in reversed(range(len(node.value))):
                key_node, value_node = node.value[position]
                stack.append((value_node, None))
                stack.append((key_node, (id(node), position)))
    return found


def _first_repeated_key(
    root_node: yaml.Node | None,
    source_values: dict[int, Any],
    alias_ranges: list[SourceRange],
) -> tuple[Any, SourceRange, SourceRange] | None:
    """Find the first mapping that repeats one of its own keys, in document order.

    Keys a ``<<:`` merge brings in are YAML's override mechanism and never
    count; only a mapping's own pairs are compared, by constructed key, so an
    alias of an earlier key is a repeat too. Each key is located where it is
    written, an alias key by ``alias_ranges`` (every alias, in document
    order), not at its anchor.
    """

    alias_key_ranges = _alias_key_ranges(root_node, alias_ranges)
    stack = [root_node] if root_node is not None else []
    seen: set[int] = set()
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        if isinstance(node, SequenceNode):
            stack.extend(reversed(node.value))
            continue
        if not isinstance(node, MappingNode):
            continue
        first_keys: dict[Any, SourceRange] = {}
        for position, (key_node, _value_node) in enumerate(node.value):
            if key_node.tag == "tag:yaml.org,2002:merge":
                continue
            key = source_values.get(id(key_node), _MISSING)
            if key is _MISSING:
                continue
            written = alias_key_ranges.get((id(node), position)) or _mark_to_range(key_node)
            if key in first_keys:
                return key, first_keys[key], written
            first_keys[key] = written
        for key_node, value_node in reversed(node.value):
            stack.append(value_node)
            stack.append(key_node)
    return None


def _first_non_text_key(
    root_node: yaml.Node | None,
    source_values: dict[int, Any],
) -> tuple[Any, str, SourceRange] | None:
    """Find the first mapping key, in document order, that YAML read as non-text.

    Every own pair of every reachable mapping is checked; ``<<:`` merge keys are
    skipped, and merged-in keys are checked where they are written. A key with
    no constructed value is skipped too; only merge keys lack one. An alias key
    is its anchored node, so it is reported at the anchor.

    Args:
        root_node (`yaml.Node | None`): The composed document root, or ``None`` for
            an empty document.
        source_values (`dict[int, Any]`): Constructed value per ``id(node)``, as
            the loader retained it.

    Returns:
        `tuple[Any, str, SourceRange] | None`:
            The constructed key, its scalar text, and its range, or ``None`` when
            every key is text.
    """

    first: yaml.Node | None = None
    stack = [root_node] if root_node is not None else []
    seen: set[int] = set()
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        if isinstance(node, SequenceNode):
            stack.extend(node.value)
            continue
        if not isinstance(node, MappingNode):
            continue
        for key_node, value_node in node.value:
            stack.append(value_node)
            if key_node.tag == "tag:yaml.org,2002:merge":
                continue
            key = source_values.get(id(key_node), _MISSING)
            if key is _MISSING or isinstance(key, str):
                continue
            if first is None or key_node.start_mark.index < first.start_mark.index:
                first = key_node
    if first is None:
        return None
    return source_values[id(first)], first.value, _mark_to_range(first)


_YAML_LINE_BREAKS = frozenset(("\r", "\n", "\x85", "\u2028", "\u2029"))
_CONSTRUCTED_LINE_BREAKS = frozenset(("\n", "\u2028", "\u2029"))


def _line_break_end(text: str, index: int, limit: int) -> int | None:
    """Return the offset after one YAML line break, if present."""

    character = text[index]
    if character not in _YAML_LINE_BREAKS:
        return None
    if character == "\r" and index + 1 < limit and text[index + 1] == "\n":
        return index + 2
    return index + 1


def _constructed_lines(value: str) -> tuple[str, ...]:
    """Split a block scalar into content lines without a phantom final line."""

    lines: list[str] = []
    start = 0
    for index, character in enumerate(value):
        if character in _CONSTRUCTED_LINE_BREAKS:
            lines.append(value[start:index])
            start = index + 1
    if start < len(value):
        lines.append(value[start:])
    return tuple(lines)


def _literal_line_ranges(
    text: str,
    token: ScalarToken | None,
    value: Any,
) -> tuple[SourceRange, ...] | None:
    """Conservatively align one constructed literal to authored content lines."""

    if (
        token is None
        or token.style != "|"
        or type(value) is not str
        or token.value != value
    ):
        return None

    token_start = token.start_mark.index
    token_end = token.end_mark.index
    if not 0 <= token_start < token_end <= len(text):
        return None

    body_start = None
    index = token_start
    while index < token_end:
        break_end = _line_break_end(text, index, token_end)
        if break_end is not None:
            body_start = break_end
            break
        index += 1
    if body_start is None:
        return () if value == "" else None

    physical_lines: list[tuple[str, int]] = []
    line_number = token.start_mark.line + 2
    index = body_start
    while index < token_end:
        line_start = index
        while index < token_end:
            break_end = _line_break_end(text, index, token_end)
            if break_end is not None:
                break
            index += 1
        physical_lines.append((text[line_start:index], line_number))
        if index < token_end:
            index = break_end
        line_number += 1

    constructed_lines = _constructed_lines(value)
    if len(constructed_lines) > len(physical_lines):
        return None

    ranges: list[SourceRange] = []
    structural_indent = None
    for constructed, (physical, source_line) in zip(
        constructed_lines,
        physical_lines,
    ):
        if not physical.endswith(constructed):
            return None
        prefix_length = len(physical) - len(constructed)
        if any(character != " " for character in physical[:prefix_length]):
            return None
        if constructed:
            if structural_indent is None:
                structural_indent = prefix_length
            elif prefix_length != structural_indent:
                return None
        ranges.append(
            SourceRange(
                source_line,
                prefix_length + 1,
                source_line,
                len(physical) + 1,
            )
        )

    if any(
        any(character != " " for character in physical)
        for physical, _ in physical_lines[len(constructed_lines) :]
    ):
        return None
    return tuple(ranges)


def _build_definitions(
    root_node: yaml.Node | None,
    root_value: Any,
    node_values: dict[int, Any],
    *,
    text: str,
    block_scalar_tokens: dict[int, ScalarToken],
    mapping_values: dict[int, dict[Any, Any]] | None = None,
    pair_sequences: dict[int, _PairSequence] | None = None,
    local_tags: dict[int, str] | None = None,
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
        text (`str`):
            Original document text retained by the source document.
        block_scalar_tokens (`dict[int, ScalarToken]`):
            Block tokens from the existing scanner pass, keyed by end offset.
        mapping_values (`dict[int, dict[Any, Any]] | None`, *optional*):
            Effective mapping values retained by the document loader, keyed by
            YAML-node identity. Absent for non-mapping nodes.
        pair_sequences (`dict[int, _PairSequence] | None`, *optional*):
            Compact flattened merge plans retained by the document loader,
            keyed by mapping-node identity.
        local_tags (`dict[int, str] | None`, *optional*):
            Local tag text without the leading ``!``, keyed by YAML-node
            identity. Untagged nodes are absent.

    Returns:
        `tuple[tuple[_Definition, ...], int]`:
            Frozen definitions and the root definition id.
    """

    if root_node is None:
        return (_ScalarDef(root_value, None, None, None),), 0
    if local_tags is None:
        local_tags = {}

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
        tag = local_tags.get(identity)

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
                tag=tag,
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
                tag=tag,
            )
        else:
            definitions[definition_id] = _ScalarDef(
                value=value,
                range=source_range,
                raw=(node.value if isinstance(node, yaml.ScalarNode) else None),
                constructed_line_ranges=_literal_line_ranges(
                    text,
                    block_scalar_tokens.get(node.end_mark.index),
                    value,
                ),
                style=(node.style if isinstance(node, yaml.ScalarNode) else None),
                tag=tag,
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


def _flow_tag_hint(error: yaml.YAMLError, text: str) -> str | None:
    """Return a hint when a YAML error's lines hold a tag glued to `,`/`}`/`]`."""

    marks = [
        mark
        for mark in (
            getattr(error, "context_mark", None),
            getattr(error, "problem_mark", None),
        )
        if mark is not None
    ]
    if not marks:
        return None
    lines = _LINE_SPLIT.split(text)
    first = min(mark.line for mark in marks)
    last = max(mark.line for mark in marks)
    if any(_FLOW_TAG.search(line) for line in lines[first : last + 1]):
        return _FLOW_TAG_HINT
    return None


# One authored key path step linked to its parent: a key's text, or an item index.
_KeyLink: TypeAlias = "tuple[_KeyLink, str | int | None] | None"


def _spell_key_link(link: _KeyLink) -> tuple[str | int | None, ...]:
    """Spell a linked authored key path out from the root."""

    steps: list[str | int | None] = []
    while link is not None:
        link, step = link
        steps.append(step)
    return tuple(reversed(steps))


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
    _repeated_key: tuple[Any, SourceRange, SourceRange] | None = field(
        default=None, repr=False, compare=False
    )
    _non_text_key: tuple[Any, str, SourceRange] | None = field(
        default=None, repr=False, compare=False
    )

    @property
    def root(self) -> SourceCursor:
        """Return a cursor at the document root."""

        return SourceCursor._physical(self, self._root_definition)

    @property
    def repeated_key(self) -> tuple[Any, SourceRange, SourceRange] | None:
        """Return the first key a mapping repeats among its own pairs.

        The tuple is the constructed key, the first pair's key range, and the
        repeating pair's key range. YAML lookup keeps the last pair; Compose
        rejects the document instead.
        """

        return self._repeated_key

    @property
    def non_text_key(self) -> tuple[Any, str, SourceRange] | None:
        """Return the first mapping key, in document order, YAML read as non-text.

        The tuple is the constructed key (a bool, number, null, date, or bytes),
        its scalar text as written, and its range. Flow names must be text, so
        Compose rejects the document.
        """

        return self._non_text_key

    def tagged_nodes(
        self, *, skip_top_level: Callable[[Any], bool] | None = None
    ) -> tuple[TaggedNode, ...]:
        """Return every node that carries a local tag, in document order.

        An aliased node is listed once, with the first occurrence path that
        reaches it.

        Examples:
            >>> doc = SourceDocument.parse("a: !int 1\\nb: [!str x]\\n")
            >>> [node.tag for node in doc.tagged_nodes()]
            ['int', 'str']

        Args:
            skip_top_level (`Callable[[Any], bool] | None`, *optional*, defaults to `None`):
                Picks top-level keys whose pairs are not walked. A node reached
                only through them is not listed; a node also reached elsewhere is
                listed with its first path outside them.

        Returns:
            `tuple[TaggedNode, ...]`:
                Tagged nodes ordered by source position; empty when the
                document has no local tags.
        """

        if not any(definition.tag is not None for definition in self._definitions):
            return ()
        first_paths: dict[int, OccurrencePath] = {}
        # Authored key paths are linked `(parent, step)` pairs, spelled out only
        # for tagged nodes, so the walk stays linear in the document size.
        key_links: dict[int, _KeyLink] = {}
        key_ids: set[int] = set()
        walked_parts: set[_ArcSequence] = set()
        stack: list[tuple[SourceCursor, int | None, _KeyLink]] = [
            (self.root, None, None)
        ]
        while stack:
            cursor, key_definition, link = stack.pop()
            path: OccurrencePath | None = None
            if key_definition is not None and key_definition not in first_paths:
                path = cursor.path
                first_paths[key_definition] = path
                key_links[key_definition] = link
                key_ids.add(key_definition)
            definition_id = cursor._definition
            assert definition_id is not None
            if definition_id in first_paths:
                continue
            first_paths[definition_id] = cursor.path if path is None else path
            key_links[definition_id] = link
            definition = self._definitions[definition_id]
            children: list[tuple[SourceCursor, int | None, _KeyLink]] = []
            skip = (
                skip_top_level
                if definition_id == self._root_definition
                else None
            )
            if isinstance(definition, _MappingDef):
                # A rope part shared by several mappings (a `<<:` source) is
                # walked once per document: the first walk already reached
                # every node in it, so merge-heavy documents stay linear.
                parts = [(definition.pair_sequence, 0)]
                while parts:
                    part, base = parts.pop()
                    # The root is walked once; with skipped keys, its parts stay
                    # unmarked so another mapping sharing them walks every pair.
                    if skip is None:
                        if part in walked_parts:
                            continue
                        walked_parts.add(part)
                    for index in range(len(part.parts) - 1, -1, -1):
                        parts.append(
                            (part.parts[index], base + part.part_offsets[index])
                        )
                    for index, arc in enumerate(part.arcs):
                        if skip is not None and skip(arc.constructed_key):
                            continue
                        occurrence = _MapOccurrence(arc, base + index)
                        child = cursor._descend_mapping_occurrence(occurrence)
                        children.append(
                            (child, arc.key_definition, (link, child.key_text))
                        )
            elif isinstance(definition, _SequenceDef):
                for index, item in enumerate(definition.item_definitions):
                    children.append(
                        (
                            cursor._descend_sequence_item(index, item),
                            None,
                            (link, index),
                        )
                    )
            stack.extend(reversed(children))

        tagged = [
            TaggedNode(
                definition_id=definition_id,
                tag=definition.tag,
                range=definition.range,
                path=first_paths.get(definition_id, ()),
                is_key=definition_id in key_ids,
                key_path=_spell_key_link(key_links.get(definition_id)),
            )
            for definition_id, definition in enumerate(self._definitions)
            if definition.tag is not None
            and (skip_top_level is None or definition_id in first_paths)
        ]
        tagged.sort(
            key=lambda node: (
                (0, node.range.start_line, node.range.start_col)
                if node.range is not None
                else (1, 0, 0),
                node.definition_id,
            )
        )
        return tuple(tagged)

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
                text=text,
                block_scalar_tokens=loader._block_scalar_tokens,
                mapping_values={
                    id(node): mapping
                    for node, mapping in loader._effective_mappings.items()
                },
                pair_sequences={
                    id(node): state
                    for node, state in loader._mapping_states.items()
                    if isinstance(state, _PairSequence)
                },
                local_tags=loader._local_tags,
            )
            repeated_key = _first_repeated_key(
                root_node, loader._source_values, loader._alias_ranges
            )
            non_text_key = _first_non_text_key(root_node, loader._source_values)
        except (yaml.YAMLError, yaml.reader.ReaderError) as exc:
            message = str(exc)
            hint = _flow_tag_hint(exc, text)
            if hint is not None:
                message = f"{message}\n{hint}"
            raise SourceSyntaxError(message, _problem_range(exc, text)) from exc
        finally:
            if loader is not None:
                loader._source_values = {}
                loader._effective_mappings = {}
                loader._mapping_states = {}
                loader._sequence_values = {}
                loader._forced_string_nodes = set()
                loader._block_scalar_tokens = {}
                loader._local_tags = {}
                loader._alias_ranges = []
                loader.dispose()

        return cls(
            text=text,
            label=label,
            _value=value,
            _root_node=root_node,
            _definitions=definitions,
            _root_definition=root_definition,
            _repeated_key=repeated_key,
            _non_text_key=non_text_key,
        )
