"""Graph-stage: parse the `nodes:` mapping into typed per-kind descriptors.

The first cross-node assembly pass (#70 Step 7). Consumes a `NodeReader` over the flow's
`nodes:` mapping (located, from the span tree) and dispatches each node body to its kind's
`parse` via the `KINDS` table — replacing the legacy line-based
`parser.parse_nodes`/`_parse_node`/`_KIND_SPECS` trio.

Only the generic, kind-independent pre-checks live here (body-is-mapping, reserved node ids,
the `input:`/`output:` -> plural back-map, `kind:` resolution); each kind's own `parse`
owns its field legality (allowed/required/asserts/env) and its `SourceRange` recording.
"""

from __future__ import annotations

from typing import Any

from agent_composer.compile.model import END_ID, START_ID
from agent_composer.compose.errors import LoadError
from agent_composer.compose.kinds import KINDS
from agent_composer.compose.reader import NodeReader

# Singular source keys the descriptor dataclasses carry in their plural form. The span tree
# (a raw parse) preserves the author's singular `input:`/`output:`; ComposeFile normalizes
# them for the dict path, but the span path bypasses ComposeFile, so we back-map here.
_INPUT_OUTPUT_REMAP = {"input": "inputs", "output": "outputs"}


def parse_nodes(nodes_reader: NodeReader) -> dict[str, Any]:
    """Parse a `nodes:` mapping (via a `NodeReader`) into typed per-kind descriptors.

    `nodes_reader` is located over the flow's `nodes:` value — span-aware over the source
    (top-level flow) or spanless over an already-constructed dict (a compact flow's in-memory
    node, a `defs:` sub-flow, a test). Returns `{node id -> descriptor}` in source order; each
    descriptor carries `range`/`field_ranges` when the reader is span-aware.
    """
    out: dict[str, Any] = {}
    nodes = nodes_reader.value
    if not isinstance(nodes, dict):
        return out
    for node_id in list(nodes):
        out[node_id] = _parse_node(node_id, nodes_reader.child(node_id))
    return out


def _parse_node(node_id: str, reader: NodeReader) -> Any:
    """Read one keyed-map node body into its typed per-kind descriptor (located)."""
    if not isinstance(reader.value, dict):
        raise LoadError(
            f"node {node_id!r}: body must be a mapping, got {type(reader.value).__name__}",
            range=reader._own_range(),
        )
    _reject_reserved_node_id(node_id, reader)
    _back_map_input_output(reader)

    kind = reader.value.get("kind")
    if kind is None:
        raise LoadError(f"node {node_id!r}: missing `kind`", range=reader._own_range())
    spec = KINDS.get(kind)
    if spec is None:
        raise LoadError(
            f"node {node_id!r}: unknown kind {kind!r} "
            f"(allowed: {', '.join(sorted(KINDS))})",
            range=reader.range_of("kind"),
        )
    return spec.parse(node_id, reader)


def _reject_reserved_node_id(node_id: str, reader: NodeReader) -> None:
    """Reject a node id that collides with a reserved separator, boundary, or resolver head.

    Mirrors the legacy `parser._parse_node` head guards; anchors each error at the node
    body's `SourceRange` (the closest located anchor for a bad node key).
    """
    rng = reader._own_range()
    if "#" in node_id or "/" in node_id:
        raise LoadError(
            f"node id {node_id!r} uses a reserved separator ('#'/'/' are reserved for "
            f"runtime graph expansion)",
            range=rng,
        )
    if node_id in (START_ID, END_ID):
        raise LoadError(
            f"node id {node_id!r} is reserved for the synthesized START_ID/END_ID boundary"
            f"; authors write `input:`/`output:`, not a boundary node",
            range=rng,
        )
    # Reserve the four singular head literals so the node-first ref `${<node>.output.k}` is
    # unambiguous, and the plural forms as migration typo-catchers.
    if node_id in ("input", "output", "system", "item"):
        raise LoadError(
            f"node id {node_id!r} is reserved (used as `${{{node_id}.X}}` as a "
            f"resolver head — pick a different node id)",
            range=rng,
        )
    if node_id in ("inputs", "outputs"):
        raise LoadError(
            f"node id {node_id!r} is reserved (the plural head is retired — "
            f"rename to `{node_id[:-1]}` or pick a different node id)",
            range=rng,
        )


def _back_map_input_output(reader: NodeReader) -> None:
    """Rewrite singular `input:`/`output:` to the plural `inputs:`/`outputs:` the kind
    descriptors use, on BOTH the constructed value dict AND the yaml key scalars (so
    `range_of` still resolves — it matches raw key text).

    A no-op when the keys are already plural (a spanless reader whose value came through
    ComposeFile normalization) or absent. Mutates the reader's private span-tree value in
    place — safe: it is a fresh per-parse tree, and value-node identity (what `range_of`
    keys on) is preserved.
    """
    value = reader.value
    if isinstance(value, dict):
        for singular, plural in _INPUT_OUTPUT_REMAP.items():
            if singular in value:
                value[plural] = value.pop(singular)
    pairs = getattr(reader.yaml_node, "value", None)
    if isinstance(pairs, list):
        for key_node, _value_node in pairs:
            raw = getattr(key_node, "value", None)
            if raw in _INPUT_OUTPUT_REMAP:
                key_node.value = _INPUT_OUTPUT_REMAP[raw]
