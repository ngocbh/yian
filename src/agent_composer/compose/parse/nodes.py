"""Graph-stage: parse the `nodes:` mapping into typed per-kind descriptors.

The first cross-node assembly pass. Consumes a source cursor over the flow's
`nodes:` mapping and dispatches each node body to its kind's `parse` via the
`KINDS` table.

Only the generic, kind-independent pre-checks live here (body-is-mapping, reserved node ids,
the `input:`/`output:` -> plural back-map, `kind:` resolution); each kind's own `parse`
owns its field legality (allowed/required/asserts/env) and its descriptor-local
`SourceRange` recording (each field tagged `("field", name)`). Two kind-blind enrichments
run after every kind parse: each `input:` binding's own range is stamped into the
descriptor's `field_ranges` under `("input", name)`, and each `asserts:` item range is
appended positionally to its `assert_ranges`. This metadata remains inside Compose;
`_finish()` copies node/field locations to `BuildResult` for validation, while runtime
diagnosis uses only opaque semantic-carrier origins. Source-order origin registration keeps
duplicate assertions distinct without an aligned range list on the semantic node.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

from agent_composer.compile.model import END_ID, START_ID
from agent_composer.compose.helpers.errors import LoadDiagnostics
from agent_composer.compose.nodes import KINDS
from agent_composer.compose.source import SourceAccessError, SourceCursor

# Singular source keys are projected onto the descriptor dataclasses' plural
# field names while retaining their original child sources.
_INPUT_OUTPUT_REMAP = {"input": "inputs", "output": "outputs"}


def parse_nodes(nodes_cursor: SourceCursor) -> dict[str, Any]:
    """Parse one source cursor into typed per-kind descriptors.

    The source is physical for a full graph, source-aligned for compact/def
    projections, or virtual for a programmatic/test mapping. Returns descriptors
    in source order with Compose-only ranges whenever an authored source exists.
    """
    diagnostics = LoadDiagnostics.from_cursor(nodes_cursor)
    out: dict[str, Any] = {}
    try:
        nodes = nodes_cursor._value_view()
        if not isinstance(nodes, Mapping):
            return out
        for node_id in list(nodes):
            desc = _parse_node(node_id, nodes_cursor.child(node_id))
            # Anchor the descriptor's node-level `.range` on the node's HEADER line (its
            # `key:` token), not its body/value block. The header is the node-level
            # diagnostic anchor a cycle, an unresolved ref, or a namespaced runtime failure
            # boxes.
            header = nodes_cursor.key_range_of(node_id)
            if header is not None and getattr(desc, "range", None) is not None:
                desc = dataclasses.replace(desc, range=header)
            out[node_id] = desc
    except SourceAccessError as exc:
        raise diagnostics.error(str(exc), range=exc.range) from exc
    return out


def _parse_node(node_id: str, cursor: SourceCursor) -> Any:
    """Read one keyed-map node body into its typed per-kind descriptor (located)."""
    diagnostics = LoadDiagnostics.from_cursor(cursor)
    body = cursor._value_view()
    if not isinstance(body, Mapping):
        raise diagnostics.error(
            f"node {node_id!r}: body must be a mapping, got {type(body).__name__}",
            range=cursor.range,
        )
    _reject_reserved_node_id(node_id, cursor)
    cursor = normalize_node_signature_cursor(cursor)

    kind = cursor.get("kind")
    if kind is None:
        raise diagnostics.error(
            f"node {node_id!r}: missing `kind`",
            range=cursor.range,
        )
    cls = KINDS.get(kind)
    if cls is None:
        raise diagnostics.error(
            f"node {node_id!r}: unknown kind {kind!r} "
            f"(allowed: {', '.join(sorted(KINDS))})",
            range=cursor.range_of("kind"),
        )
    desc = cls.parse(node_id, cursor)
    # Compose-only range enrichments retained on parsed descriptors, keyed by `(kind, key)`
    # rather than a bare name so namespaces cannot collide (the kind already stamped
    # its own fields as `("field", name)` in `cls.parse`):
    #   - each `input:` binding -> its own range, tagged `("input", name)` and distinct
    #     from a like-named field.
    #   - each `asserts:` list item -> its POSITIONAL range, appended to the descriptor's
    #     `assert_ranges` (aligned 1:1 with the authored `asserts:` list). Asserts are located by
    #     INDEX, not by their (non-unique) text, so two identical asserts get distinct carets — hence
    #     they live in a list, NOT in `field_ranges`.
    # Runtime diagnosis does not consume these ranges, and the semantic Node never receives
    # them. Both no-op for a source-free cursor or a descriptor without the field.
    # `input:` is read AFTER
    # `normalize_node_signature_cursor` has renamed the author's singular
    # `input:` key to plural.
    field_ranges = getattr(desc, "field_ranges", None)
    if isinstance(field_ranges, dict):
        field_ranges.update(
            {
                ("input", name): rng
                for name, rng in cursor.mapping_value_ranges("inputs").items()
            }
        )
    assert_ranges = getattr(desc, "assert_ranges", None)
    if isinstance(assert_ranges, list):
        assert_ranges.extend(cursor.item_range_list("asserts"))
    return desc


def _reject_reserved_node_id(node_id: str, cursor: SourceCursor) -> None:
    """Reject a node id that collides with a reserved separator, boundary, or resolver head.

    Mirrors the legacy `parser._parse_node` head guards; anchors each error at the node
    body's `SourceRange` (the closest located anchor for a bad node key).
    """
    diagnostics = LoadDiagnostics.from_cursor(cursor)
    rng = cursor.range
    if "#" in node_id or "/" in node_id or "." in node_id:
        raise diagnostics.error(
            f"node id {node_id!r} uses a reserved separator ('#' and '/' are reserved "
            f"for runtime graph expansion; '.' is reserved for reference paths)",
            range=rng,
        )
    if node_id in (START_ID, END_ID):
        raise diagnostics.error(
            f"node id {node_id!r} is reserved for the synthesized START_ID/END_ID boundary"
            f"; authors write `input:`/`output:`, not a boundary node",
            range=rng,
        )
    # Reserve the four singular head literals so the node-first ref `${<node>.output.k}` is
    # unambiguous, and the plural forms as migration typo-catchers.
    if node_id in ("input", "output", "system", "item"):
        raise diagnostics.error(
            f"node id {node_id!r} is reserved (used as `${{{node_id}.X}}` as a "
            f"resolver head — pick a different node id)",
            range=rng,
        )
    if node_id in ("inputs", "outputs"):
        raise diagnostics.error(
            f"node id {node_id!r} is reserved (the plural head is retired — "
            f"rename to `{node_id[:-1]}` or pick a different node id)",
            range=rng,
        )


def normalize_node_signature_cursor(cursor: SourceCursor) -> SourceCursor:
    """Project authored node-signature keys onto descriptor field names.

    The projection is the single crossing from the singular Compose surface to
    the plural Python carrier fields. Later phases that correlate a descriptor
    with its source must reuse this cursor so collision and source-selection
    semantics cannot drift from parsing.

    Args:
        cursor (`SourceCursor`):
            Cursor over one node body mapping in authored or projected form.

    Returns:
        `SourceCursor`:
            Pure projection with signature keys normalized to carrier names.

    Raises:
        `SourceAccessError`:
            Raised when `cursor` is not a traversable mapping.
    """

    return cursor.renamed(_INPUT_OUTPUT_REMAP)
