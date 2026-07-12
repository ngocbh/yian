"""Parse-free source-location primitives for the Compose front end.

The parse-independent slice of the loader's render-only source mapping: the
`SourceFrame` carrier plus two best-effort span-map helpers, all built directly on
`load_with_spans` + `NodeReader` (never `parse_nodes`/`parse_file`). Kept at the bottom
of the ladder (`helpers <- parse`) so `parse/` can depend on it without a cycle; the
frame-range helpers that DO call the parser (`_frame_ranges`, `top_frame_ranges`) stay in
`loader.py`, which sits above `parse`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import yaml

from agent_composer.events import SourceRange
from agent_composer.compose.helpers.reader import NodeReader
from agent_composer.compose.helpers.spanloader import load_with_spans


def _boundary_field_ranges(text: Optional[str]) -> dict[str, SourceRange]:
    """The flow-boundary source map stamped onto the top-level StartNode.

    Keys: each top-level `input:`/`inputs:` decl name -> its declaration value `SourceRange`,
    plus each flow-level `asserts:` expr string -> its `SourceRange`. Feeds StartNode.field_ranges
    so a boundary input-type failure (`input <name> — ...`) or a false boundary assert resolves a
    precise caret. Empty for a spanless / uncomposable flow (compact / in-memory / a def, whose
    `input:` lives under `defs.<name>` — best-effort, boundary errors there fall back to the node
    range). Mirrors `_frame_ranges`: render metadata, never raises."""
    if not text:
        return {}
    try:
        value, spans, root = load_with_spans(text, return_root=True)
    except yaml.YAMLError:
        return {}
    if not isinstance(value, dict):
        return {}
    reader = NodeReader(value, spans, root)
    out: dict[str, SourceRange] = {}
    for key in ("input", "inputs"):
        if isinstance(value.get(key), dict):
            out.update(reader.mapping_value_ranges(key))
    out.update(reader.item_ranges("asserts"))
    return out


def _section_lines(text: Optional[str]) -> dict[str, int]:
    """Map each top-level section key -> its 1-based header line (best-effort).

    Replaces the retired `parser.section_lines`: locates a section-level error (a bad
    `typedefs:`, an eager `uses:` failure, a dangling `outputs:`/`asserts:` ref) at the
    section's header line. Derived from the span parse (`load_with_spans` + each section
    key's own `SourceRange`), consistent with the range-native front end — raw keys are
    kept (`output`/`outputs` both surface; the loader coalesces). `{}` when the text is
    absent or PyYAML can't compose it. Render metadata, never the error path — never raises.
    """
    if not text:
        return {}
    try:
        value, spans, root = load_with_spans(text, return_root=True)
    except yaml.YAMLError:
        return {}
    if not isinstance(value, dict):
        return {}
    reader = NodeReader(value, spans, root)
    out: dict[str, int] = {}
    for key in value:
        rng = reader.key_range_of(key)
        if rng is not None:
            out[str(key)] = rng.start_line
    return out


@dataclass(frozen=True)
class SourceFrame:
    """Render-only source-location metadata for one flow (a def or an external file).

    Carried on a `LoadedFlow` and copied onto the built `CallNode`/`MapNode` so the CLI
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
            Empty (frame collapses to the call line) for a compact single-node def with no
            authored inner `nodes:` mapping.
        field_ranges (`dict[str, dict[str, SourceRange]]`):
            `node_id -> {field -> `SourceRange`}` — the kind-fallback / `output:` precision
            used when the failing locator is `kind=field`.
    """

    label: str
    text: str
    node_ranges: dict[str, SourceRange]
    field_ranges: dict[str, dict[str, SourceRange]]

    def __deepcopy__(self, memo):
        # Frozen + render-only: every per-callsite `clone_child` deep-copies each child
        # node, which would otherwise copy the full YAML `text` per clone. Sharing the one
        # frozen instance is safe (it is never mutated) and avoids O(elements x filesize).
        return self
