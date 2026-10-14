"""Occurrence-local extraction for inline composite child-flow bodies.

Extraction projects immutable source cursors. Each alias occurrence therefore
gets its own masked view while the document graph stays unchanged, and retained
child cursors keep their original paths and ranges. A tagged `nodes:` entry is a
value, never an inline body, even when its mapping has a `kind:` key.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from yian.compose.helpers.errors import LoadDiagnostics
from yian.compose.source import SourceCursor
from yian.diagnostics import SourceRange


INLINE_FLOW_PREFIX = "__inline_flow_"
"""Callable-name prefix reserved for compiler-generated inline child flows."""

_INLINE_COMPOSITE_KINDS = frozenset({"call", "map", "loop"})


@dataclass(frozen=True)
class InlineBody:
    """One retained inline child and its occurrence-local private identity.

    Attributes:
        node_id: The enclosing authored CALL/MAP/LOOP node id.
        owner: The complete lexical owner path for this anonymous child.
        flow_id: The stable private callable id derived from ``owner``.
        nodes_cursor: The original located cursor for the child's ``nodes:`` value.
        output_cursor: The original located cursor for the child's ``output:`` value.
        input_ranges: Outer input name to authored binding-value range.
    """

    node_id: str
    owner: tuple[str, ...]
    flow_id: str
    nodes_cursor: SourceCursor
    output_cursor: SourceCursor
    input_ranges: dict[str, SourceRange]

    @property
    def output(self) -> Any:
        """Return an immutable snapshot of the authored child output value."""
        return self.output_cursor.value

    @property
    def output_range(self) -> SourceRange | None:
        """Return the authored child output value range when source-backed."""
        return self.output_cursor.range


def inline_flow_id(owner: tuple[str, ...]) -> str:
    """Return the stable private callable id for one complete lexical owner path."""
    encoded = json.dumps(
        list(owner),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return INLINE_FLOW_PREFIX + hashlib.sha256(encoded).hexdigest()


def extract_inline_bodies(
    nodes_cursor: SourceCursor,
    *,
    owner: tuple[str, ...],
) -> tuple[SourceCursor, dict[str, InlineBody]]:
    """Validate, retain, and mask inline CALL/MAP/LOOP bodies.

    The returned cursor is an occurrence-local projection. Named composites remain
    unchanged. An inline composite becomes an ordinary private ``call:`` in that
    overlay, while the returned body record retains its original child/output
    values and source locations. The supplied cursor and its YAML anchors remain
    read-only.
    """
    diagnostics = LoadDiagnostics.from_cursor(nodes_cursor)
    nodes = nodes_cursor._value_view()
    if not isinstance(nodes, Mapping):
        return nodes_cursor, {}
    nodes_snapshot = nodes_cursor.value

    _reject_authored_private_call_targets(nodes_cursor)
    bodies: dict[str, InlineBody] = {}
    projected: dict[Any, tuple[Any, SourceCursor | None]] = {}

    for node_id in list(nodes):
        body_cursor = nodes_cursor.child(node_id)
        raw_body = body_cursor._value_view()
        # A tagged entry is a value, even a mapping with `kind: call`.
        if not isinstance(raw_body, Mapping) or body_cursor.tag is not None:
            projected[node_id] = (nodes_snapshot[node_id], body_cursor)
            continue
        if raw_body.get("kind") not in _INLINE_COMPOSITE_KINDS:
            projected[node_id] = (nodes_snapshot[node_id], body_cursor)
            continue

        has_call = "call" in raw_body
        has_nodes = "nodes" in raw_body
        if has_call == has_nodes:
            conflict_range = (
                body_cursor.range_of("nodes") if has_nodes else body_cursor.range
            )
            raise diagnostics.error(
                f"node {node_id!r} (kind={raw_body['kind']}): exactly one of "
                "`call:` or inline `nodes:` is required",
                range=conflict_range,
            )

        if has_call:
            if "output" in raw_body:
                raise diagnostics.error(
                    f"node {node_id!r} (kind={raw_body['kind']}): `output:` is only "
                    "valid with inline `nodes:`",
                    range=body_cursor.range_of("output"),
                )
            projected[node_id] = (nodes_snapshot[node_id], body_cursor)
            continue

        child_nodes = raw_body["nodes"]
        if not isinstance(child_nodes, Mapping):
            raise diagnostics.error(
                f"node {node_id!r} (kind={raw_body['kind']}): inline `nodes:` must "
                "be a non-empty mapping",
                range=body_cursor.range_of("nodes"),
            )
        if not child_nodes:
            raise diagnostics.error(
                f"node {node_id!r} (kind={raw_body['kind']}): inline `nodes:` must "
                "be a non-empty mapping",
                range=body_cursor.range_of("nodes"),
            )
        if "output" not in raw_body:
            raise diagnostics.error(
                f"node {node_id!r} (kind={raw_body['kind']}): inline `nodes:` "
                "requires `output:`",
                range=body_cursor.range,
            )

        outer_input = raw_body.get("input", {})
        if not isinstance(outer_input, Mapping) or not all(
            isinstance(key, str) for key in outer_input
        ):
            raise diagnostics.error(
                f"node {node_id!r} (kind={raw_body['kind']}): inline `input:` must "
                "be a mapping with string keys",
                range=body_cursor.range_of("input"),
            )

        child_owner = (*owner, "node", node_id, "inline")
        flow_id = inline_flow_id(child_owner)
        body = InlineBody(
            node_id=node_id,
            owner=child_owner,
            flow_id=flow_id,
            nodes_cursor=body_cursor.child("nodes"),
            output_cursor=body_cursor.child("output"),
            input_ranges=body_cursor.mapping_value_ranges("input"),
        )
        bodies[node_id] = body
        masked, masked_value = _mask_inline_body(
            body_cursor,
            flow_id,
            nodes_snapshot[node_id],
        )
        projected[node_id] = (masked_value, masked)

    return nodes_cursor.project_mapping(projected), bodies


def _reject_authored_private_call_targets(nodes_cursor: SourceCursor) -> None:
    """Reject authored composite calls into the compiler-private namespace.

    This preflight deliberately completes before any inline body is retained or
    masked. Otherwise an authored call later in source order could target the
    predictable id of a sibling that extraction had already synthesized.
    """
    diagnostics = LoadDiagnostics.from_cursor(nodes_cursor)
    nodes = nodes_cursor._value_view()
    if not isinstance(nodes, Mapping):
        return
    for node_id in nodes:
        body_cursor = nodes_cursor.child(node_id)
        raw_body = body_cursor._value_view()
        if not isinstance(raw_body, Mapping) or body_cursor.tag is not None:
            continue
        if raw_body.get("kind") not in _INLINE_COMPOSITE_KINDS:
            continue
        call_target = raw_body.get("call")
        if not isinstance(call_target, str) or not call_target.startswith(
            INLINE_FLOW_PREFIX
        ):
            continue
        raise diagnostics.error(
            f"node {node_id!r} (kind={raw_body['kind']}): call targets beginning "
            f"with {INLINE_FLOW_PREFIX!r} are reserved for compiler-generated "
            "inline child flows",
            range=body_cursor.range_of("call"),
        )


def _mask_inline_body(
    cursor: SourceCursor,
    flow_id: str,
    body_value: dict[Any, Any],
) -> tuple[SourceCursor, dict[Any, Any]]:
    """Project one ``nodes:``/``output:`` pair into a private ``call:``."""

    masked: dict[Any, Any] = {}
    entries: dict[Any, tuple[Any, SourceCursor | None]] = {}
    for key, value in body_value.items():
        if key == "nodes":
            masked["call"] = flow_id
            entries["call"] = (None, cursor.child("nodes"))
        elif key != "output":
            masked[key] = value
            entries[key] = (None, cursor.child(key))
    return cursor.project_mapping(entries), masked
