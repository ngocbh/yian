"""One-pass parsing and private retention of a Compose YAML document."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import yaml

from agent_composer.compose.source.common import (
    OccurrencePath,
    SourceSyntaxError,
    _Definition,
    _MapArc,
    _MappingDef,
    _ScalarDef,
    _SequenceDef,
)
from agent_composer.compose.source.cursor import SourceCursor
from agent_composer.events import SourceRange


_MISSING = object()


class _DocumentLoader(yaml.SafeLoader):
    """The private SafeLoader instance that composes and constructs one document."""

    def __init__(self, stream: str):
        self._source_values: dict[int, Any] = {}
        super().__init__(stream)

    def construct_object(self, node: yaml.Node, deep: bool = False) -> Any:
        """Construct normally while retaining a temporary node/value identity map."""

        value = super().construct_object(node, deep=deep)
        self._source_values[id(node)] = value
        return value


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


def _effective_ordinals(arcs: list[_MapArc], value: Any) -> tuple[int, ...]:
    """Return last-value ordinals in constructed-dict first-key order."""

    if type(value) is not dict:
        return ()
    last_by_key: dict[Any, int] = {}
    order: list[Any] = []
    for arc in arcs:
        key = arc.constructed_key
        if key not in last_by_key:
            order.append(key)
        last_by_key[key] = arc.pair_ordinal
    return tuple(last_by_key[key] for key in order)


def _build_definitions(
    root_node: yaml.Node | None,
    root_value: Any,
    node_values: dict[int, Any],
) -> tuple[tuple[_Definition, ...], int]:
    """Intern every reachable YAML node once into a frozen definition DAG."""

    if root_node is None:
        return (_ScalarDef(root_value, None, None),), 0

    definitions: list[_Definition | None] = []
    ids: dict[int, int] = {}

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
        if value is _MISSING:
            raise TypeError("constructed value missing for retained YAML node")
        source_range = _mark_to_range(node)

        if isinstance(node, yaml.MappingNode):
            arcs: list[_MapArc] = []
            for ordinal, (key_node, value_node) in enumerate(node.value):
                key_definition = intern(key_node)
                value_definition = intern(value_node)
                key_value = node_values[id(key_node)]
                arcs.append(
                    _MapArc(
                        constructed_key=key_value,
                        raw_key=(
                            key_node.value
                            if isinstance(key_node, yaml.ScalarNode)
                            else None
                        ),
                        key_definition=key_definition,
                        value_definition=value_definition,
                        pair_ordinal=ordinal,
                    )
                )
            definitions[definition_id] = _MappingDef(
                value=value,
                range=source_range,
                arcs=tuple(arcs),
                effective_ordinals=_effective_ordinals(arcs, value),
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
            )
        except (yaml.YAMLError, yaml.reader.ReaderError) as exc:
            raise SourceSyntaxError(str(exc), _problem_range(exc, text)) from exc
        finally:
            if loader is not None:
                loader._source_values = {}
                loader.dispose()

        return cls(
            text=text,
            label=label,
            _value=value,
            _root_node=root_node,
            _definitions=definitions,
            _root_definition=root_definition,
        )
