"""Parse-free source-location primitives for the Compose front end.

The parse-independent slice of the loader's render-only source mapping: the
`SourceFrame` carrier plus best-effort location-map helpers over an existing source cursor.
They never parse or call `parse_nodes`/`parse_file`. Kept at the bottom
of the ladder (`helpers <- parse`) so `parse/` can depend on it without a cycle; the
frame-range helpers that DO call the parser (`_frame_ranges`, `top_frame_ranges`) stay in
`loader.py`, which sits above `parse`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from agent_composer.events import SourceRange
from agent_composer.nodes.base import FieldRanges
from agent_composer.compose.source import SourceAccessError, SourceCursor


def _boundary_field_ranges(root: SourceCursor) -> FieldRanges:
    """The flow-boundary source map stamped onto the top-level StartNode's `field_ranges`.

    Keys: each top-level `input:`/`inputs:` decl name, tagged `("input", name)` -> its
    declaration value `SourceRange`. (The boundary input-coercion failure carries a
    `SourceSpan(kind="input", key=name)` locator, so the runtime funnel resolves the caret by
    that `(kind, key)`.) Flow-level `asserts:` are NOT here — they are located by position (see
    `_boundary_assert_ranges`, threaded into the lowered START/END assert-range lists). Empty for a
    spanless / in-memory flow — best-effort, boundary errors there fall back to the node range.
    Mirrors `_frame_ranges`: render metadata, never raises."""
    try:
        value = root._value_view()
        if not isinstance(value, Mapping):
            return FieldRanges()
        out: FieldRanges = FieldRanges()
        for key in ("input", "inputs"):
            if isinstance(value.get(key), Mapping):
                out.update(
                    {
                        ("input", name): rng
                        for name, rng in root.mapping_value_ranges(key).items()
                    }
                )
        return out
    except SourceAccessError:
        return FieldRanges()


def _boundary_assert_ranges(root: SourceCursor) -> list[SourceRange | None]:
    """The POSITIONAL ranges of the flow-level `asserts:` list, aligned 1:1 with it in author order.

    The flow-boundary twin of the node-local `assert_ranges`: `lower_flow_asserts` consumes this to
    stamp each lowered START/END assert's own `SourceRange` (a false boundary assert then carets its
    own line), and to caret a bad-ref `LoadError` at the offending assert. Positional — NOT keyed by
    text — so two identical flow asserts get distinct carets. `[]` for a spanless / uncomposable
    flow; render metadata, never raises (mirrors `_boundary_field_ranges`)."""
    try:
        if not isinstance(root._value_view(), Mapping):
            return []
        return root.item_range_list("asserts")
    except SourceAccessError:
        return []


def _section_lines(root: SourceCursor) -> dict[str, int]:
    """Map each top-level section key -> its 1-based header line (best-effort).

    Replaces the retired `parser.section_lines`: locates a section-level error (a bad
    `typedefs:`, an eager `uses:` failure, a dangling `outputs:`/`asserts:` ref) at the
    section's header line. Derived from each section key's `SourceRange` on the supplied
    cursor, consistent with the range-native front end — raw keys are
    kept (`output`/`outputs` both surface; the loader coalesces). `{}` when the text is
    absent or PyYAML can't compose it. Render metadata, never the error path — never raises.
    """
    try:
        value = root._value_view()
        if not isinstance(value, Mapping):
            return {}
        out: dict[str, int] = {}
        for key in value:
            rng = root.key_range_of(key)
            if rng is not None:
                out[str(key)] = rng.start_line
        return out
    except SourceAccessError:
        return {}


@dataclass(frozen=True)
class SourceFrame:
    """Render-only source-location metadata for one flow (a def or an external file).

    Carried on a `ComposedFlow` and copied onto the built `CallNode`/`MapNode` so the CLI
    can box a `.yaml` frame for the *child* a call descends into — the basis of the
    nested-error call traceback. Never read by the engine's `run()`; purely for error
    rendering.

    Attributes:
        label (`str`):
            The frame title — `"defs:<name>"` for an in-file callable, or the resolved
            filename (e.g. `"24-uses-external.yaml"`) for an external `uses:` flow.
        text (`str`):
            The YAML source whose lines the maps below index. For a def this is the
            **parent** file's text (the def's nodes live at absolute lines there); for an
            external flow it is that file's own text.
        node_ranges (`dict[str, SourceRange]`):
            `node_id -> the node body's `SourceRange` in `text`` for the child's nodes.
            A source-backed compact or single-node form projects its synthesized
            node header onto the corresponding authored `id:` or `defs:` key.
        field_ranges (`dict[str, dict[tuple[str, str], SourceRange]]`):
            `node_id -> {(kind, key) -> `SourceRange`}` — the per-field precision, keyed by the
            location's `(SourceSpan.kind, key)` (`kind ∈ {input, field}`, never
            `assert`), used when a failing locator resolves against this frame.
    """

    label: str
    text: str
    node_ranges: dict[str, SourceRange]
    field_ranges: dict[str, dict[tuple[str, str], SourceRange]]

    def __deepcopy__(self, memo):
        # Frozen + render-only: every per-callsite `clone_child` deep-copies each child
        # node, which would otherwise copy the full YAML `text` per clone. Sharing the one
        # frozen instance is safe (it is never mutated) and avoids O(elements x filesize).
        return self
