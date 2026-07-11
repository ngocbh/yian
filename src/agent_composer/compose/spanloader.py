"""A span-recording YAML loader: construct values AND record source ranges in ONE parse.

The compose front end needs both the constructed Python value of a flow document and
the source range of every yaml node (for diagnostics). Rather than double-parse (once
with `yaml.safe_load` for values, again with `yaml.compose` for line marks), this
module subclasses `yaml.SafeLoader` so a single parse yields both: SafeConstructor
builds the value (anchors, `<<:` merge, scalar coercion), and the overrides here record
each yaml *node*'s SourceRange keyed by `id(node)`.
"""

from __future__ import annotations

import yaml

from agent_composer.events import SourceRange


def _mark_to_range(start_mark, end_mark) -> SourceRange:
    """Convert a PyYAML (start_mark, end_mark) pair to a 1-based half-open SourceRange.

    PyYAML marks are 0-based (`mark.line`, `mark.column`); SourceRange is 1-based, so
    each coordinate is offset by +1. PyYAML's `end_mark` already points one-past the
    last char, matching SourceRange's exclusive (half-open) end, so end maps directly
    after the same +1 shift.
    """
    return SourceRange(
        start_line=start_mark.line + 1,
        start_col=start_mark.column + 1,
        end_line=end_mark.line + 1,
        end_col=end_mark.column + 1,
    )


class SpanLoader(yaml.SafeLoader):
    """A SafeLoader that constructs values AND records per-node source ranges in ONE parse.

    Inherits all of SafeConstructor's value construction (anchors, `<<:` merge via
    flatten_mapping, scalar coercion), and additionally records each yaml *node*'s
    SourceRange into `self._span_marks` keyed by `id(node)` — a private name chosen to
    avoid colliding with the PyYAML Parser's internal `self.marks` stack (S-new-1).
    The tree is keyed by yaml-node identity, not constructed value, because
    construct_yaml_map yields a discarded dict (S-new-2).
    """

    def __init__(self, stream):
        super().__init__(stream)
        self._span_marks = {}  # id(node) -> SourceRange

    # Each override calls super() first (preserving SafeConstructor's value construction:
    # anchors, `<<:` merge via flatten_mapping, scalar coercion) then records the node's
    # own range. super() constructs every child via construct_object, which routes back
    # into these three overrides, so each key/value/element records itself — the mapping
    # and sequence overrides only need to record their own container node. Merged/anchored
    # keys are recorded at their anchor definition site (anchors share node identity).

    def construct_mapping(self, node, deep=False):
        data = super().construct_mapping(node, deep=deep)
        self._span_marks[id(node)] = _mark_to_range(node.start_mark, node.end_mark)
        return data

    def construct_sequence(self, node, deep=False):
        data = super().construct_sequence(node, deep=deep)
        self._span_marks[id(node)] = _mark_to_range(node.start_mark, node.end_mark)
        return data

    def construct_scalar(self, node):
        data = super().construct_scalar(node)
        self._span_marks[id(node)] = _mark_to_range(node.start_mark, node.end_mark)
        return data


def load_with_spans(text, return_root=False):
    """Parse a single YAML document, returning both the value and its span tree.

    Mirrors PyYAML's `yaml.load` (construct via `get_single_data`, dispose in a
    `finally`) but holds the loader instance so `_span_marks` survives — the one-shot
    `yaml.load(text, SpanLoader)` would discard the loader and its recorded spans.

    Returns `(value, spans)` where `spans` is `dict[id(node) -> SourceRange]`.

    `spans` is keyed by `id(node)` of *this loader's* yaml nodes. To resolve any range
    from `spans` a caller must ALSO hold the yaml node tree alive (else the ids belong to
    a garbage-collected tree and `id()` values are recycled). Pass `return_root=True` to
    additionally return the composed root node so the tree stays reachable:

    - `return_root=False` (default): `(value, spans)` — the 2-tuple for back-compat.
    - `return_root=True`:            `(value, spans, root)` — `root` is the composed
      `yaml.Node` tree; keep it alive alongside `spans` to look up `spans[id(node)]`.

    The root branch drives the loader as `get_single_node()` -> `construct_document(root)`
    (the same construction `get_single_data` performs internally), so the constructed
    `value` and the recorded `spans` are identical to the default path; only the live root
    node is additionally surfaced.
    """
    loader = SpanLoader(text)
    try:
        if return_root:
            # Compose the node tree first (keeps its node identities live), THEN construct
            # the value against those same nodes so `spans[id(node)]` stays resolvable via
            # the returned root. `construct_document` is what `get_single_data` calls under
            # the hood after `get_single_node`.
            root = loader.get_single_node()
            value = loader.construct_document(root)
            return value, loader._span_marks, root
        value = loader.get_single_data()
        return value, loader._span_marks
    finally:
        loader.dispose()
