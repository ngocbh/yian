"""NodeReader — a span-aware cursor over one constructed YAML mapping.

The compose front end reads a flow's nested `dict`s field by field. A plain `dict`
access loses the source location of a bad or missing field; `NodeReader` wraps the
triple `(constructed_value, spans, yaml_node)` so every field access can raise a
`LoadError` carrying the precise `SourceRange` of the offending field.

The three parts must travel together (see `load_with_spans(..., return_root=True)`):

- `value`     — the constructed Python value for THIS node (a `dict` for a mapping).
- `spans`     — `dict[id(yaml_node) -> SourceRange]` for the whole document; shared by
                every `NodeReader` descended from the same parse. Keyed by `id()`, so it
                is only meaningful while the yaml node tree is held alive.
- `yaml_node` — the composed `yaml.Node` for THIS value (a `MappingNode` for a mapping),
                whose identity indexes into `spans`. Holding it keeps the tree reachable.

`child(key)` descends to a sub-`NodeReader` over the child's value node, so the whole
tree can be walked while every accessor stays located.

A **spanless** reader (`NodeReader.spanless(value)`, `yaml_node=None`) wraps a value that
has no source text to point at — a compact flow's in-memory node, a `defs:` sub-flow's
already-normalized nodes, or a test's inline dict. Field presence and typing still work off
`value`; every range accessor returns `None`. Descendants of a spanless reader are spanless.
"""

from __future__ import annotations

from typing import Any

import yaml

from agent_composer.compose.errors import LoadError
from agent_composer.events import SourceRange


class NodeReader:
    """A located cursor over one constructed YAML mapping value.

    Wraps `(value, spans, yaml_node)` (see the module docstring). Accessors descend the
    `yaml_node` — a `MappingNode` whose `.value` is a list of `(key_node, value_node)`
    pairs — to find the pair whose constructed key equals the requested `key`, then read
    the child's constructed value and resolve its `SourceRange` from `spans`.

    Located-error policy: a missing or wrong-typed field raises a `LoadError` whose
    `range` points at the child VALUE node when the field exists (so the caret sits on the
    bad value), or at the mapping node itself for a MISSING field (the value node does not
    exist to point at).
    """

    def __init__(self, value: Any, spans: dict, yaml_node: yaml.Node) -> None:
        # The constructed Python value for this node — a `dict` for a mapping; the accessors
        # below assume a mapping (they descend `yaml_node.value` pairs).
        self.value = value
        # Shared document span index: id(yaml_node) -> SourceRange. Meaningful only while
        # the yaml node tree (reachable via `self.yaml_node`) is held alive.
        self.spans = spans
        # The composed yaml node for THIS value; its identity (and its children's) indexes
        # into `spans`. A `MappingNode` for a mapping.
        self.yaml_node = yaml_node

    @classmethod
    def spanless(cls, value: Any) -> "NodeReader":
        """A location-free reader over an already-constructed value (no span tree).

        For inputs with no source text to point at — a compact flow's in-memory node, a
        `defs:` sub-flow's nested nodes, or a test's inline dict. All range accessors
        return `None`; field presence/typing still resolve off `value`.
        """
        return cls(value, {}, None)

    # -- internal ---------------------------------------------------------------

    def _own_range(self) -> SourceRange | None:
        """This mapping node's own recorded range (the fallback anchor for a missing field)."""
        if self.yaml_node is None:
            return None
        return self.spans.get(id(self.yaml_node))

    def _pair(self, key: str):
        """Find the `(key_node, value_node)` pair whose constructed key == `key`.

        Returns the pair, or `None` when this node is not a mapping or has no such key.
        Compares against `key_node.value` (the raw scalar text of the key), which for a
        plain string key equals its constructed value. In spanless mode (`yaml_node is
        None`) there are no yaml nodes, so presence is synthesized from `self.value` and
        the pair is `(None, None)` (the range accessors resolve to `None`).
        """
        if self.yaml_node is None:
            if isinstance(self.value, dict) and key in self.value:
                return None, None
            return None
        pairs = getattr(self.yaml_node, "value", None)
        if not isinstance(pairs, list):
            return None
        for key_node, value_node in pairs:
            if getattr(key_node, "value", None) == key:
                return key_node, value_node
        return None

    def _require_pair(self, key: str):
        """`_pair` or a located LoadError anchored at this mapping node (the field is missing)."""
        pair = self._pair(key)
        if pair is None:
            raise LoadError(f"missing required field {key!r}", range=self._own_range())
        return pair

    # -- range ------------------------------------------------------------------

    def range_of(self, key: str) -> SourceRange | None:
        """The `SourceRange` of `key`'s VALUE node (where the field's value lives).

        Value node (not key node) so a diagnostic caret sits on the offending value. `None`
        when the value node has no recorded span (defensive). Raises a located LoadError if
        the field is missing.
        """
        _key_node, value_node = self._require_pair(key)
        if value_node is None:
            return None
        return self.spans.get(id(value_node))

    def item_ranges(self, key: str) -> dict[str, SourceRange]:
        """For a SEQUENCE field, map each scalar item's text -> its own `SourceRange`.

        Descends `key`'s value node (a `SequenceNode`) and, for every scalar element,
        records `element_text -> range`. Used to give each `asserts:` list item its own
        caret (the runtime keys `node.field_ranges` on the raw assert string). Returns
        `{}` in spanless mode, when the field is absent, or when it is not a sequence —
        best-effort, never raising (unlike `range_of`).
        """
        pair = self._pair(key)
        if pair is None:
            return {}
        _key_node, value_node = pair
        if not isinstance(value_node, yaml.SequenceNode):
            return {}
        out: dict[str, SourceRange] = {}
        for item in value_node.value:
            if isinstance(item, yaml.ScalarNode):
                rng = self.spans.get(id(item))
                if rng is not None:
                    out[item.value] = rng
        return out

    # -- typed accessors --------------------------------------------------------

    def str(self, key: str) -> str:
        """The value of `key` as a `str`; located LoadError if missing or not a string."""
        _key_node, value_node = self._require_pair(key)
        val = self.value[key]
        if not isinstance(val, str):
            raise LoadError(
                f"field {key!r} must be a string, got {type(val).__name__}",
                range=self.spans.get(id(value_node)),
            )
        return val

    def int(self, key: str) -> int:
        """The value of `key` as an `int`; located LoadError if missing or not an int.

        `bool` is rejected (a YAML `true`/`false` is a bool, not an int, despite `bool`
        subclassing `int` in Python)."""
        _key_node, value_node = self._require_pair(key)
        val = self.value[key]
        if isinstance(val, bool) or not isinstance(val, int):
            raise LoadError(
                f"field {key!r} must be an integer, got {type(val).__name__}",
                range=self.spans.get(id(value_node)),
            )
        return val

    def mapping(self, key: str) -> dict:
        """The value of `key` as a mapping (`dict`); located LoadError if missing or not a mapping."""
        _key_node, value_node = self._require_pair(key)
        val = self.value[key]
        if not isinstance(val, dict):
            raise LoadError(
                f"field {key!r} must be a mapping, got {type(val).__name__}",
                range=self.spans.get(id(value_node)),
            )
        return val

    def sequence(self, key: str) -> list:
        """The value of `key` as a sequence (`list`); located LoadError if missing or not a sequence."""
        _key_node, value_node = self._require_pair(key)
        val = self.value[key]
        if not isinstance(val, list):
            raise LoadError(
                f"field {key!r} must be a sequence, got {type(val).__name__}",
                range=self.spans.get(id(value_node)),
            )
        return val

    def get(self, key: str, default: Any = None) -> Any:
        """The raw constructed value of `key`, or `default` if absent (no type check, no raise)."""
        pair = self._pair(key)
        if pair is None:
            return default
        return self.value[key]

    def child(self, key: str) -> "NodeReader":
        """A sub-`NodeReader` over `key`'s value node — descend one level, staying located.

        Raises a located LoadError if the field is missing. The child shares this reader's
        `spans` (one document-wide index) and carries the child's own value node, so its
        accessors resolve ranges from the SAME parse.
        """
        _key_node, value_node = self._require_pair(key)
        return NodeReader(self.value[key], self.spans, value_node)
