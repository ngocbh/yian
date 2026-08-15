"""Occurrence-local extraction for inline composite child-flow bodies.

YAML aliases share constructed mappings and composed nodes. Merge expansion can
also reuse key/value nodes from an anchor. Extraction first gives each node
occurrence its own shallow mapping wrappers, then masks inline composite fields
only on those wrappers while retaining the original child values and locations.

The overlay copies constructed mappings, YAML `MappingNode` wrappers, pair lists,
and key scalar nodes. Value nodes remain shared and read-only. Copied-node ranges
live in a small lookup layered over the original span table through `NodeReader`.
"""

from __future__ import annotations

import hashlib
import json
from copy import copy
from dataclasses import dataclass
from typing import Any

import yaml

from agent_composer.compose.helpers.errors import LoadError
from agent_composer.compose.helpers.reader import NodeReader
from agent_composer.events import SourceRange


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
        nodes_reader: The original located reader for the child's ``nodes:`` value.
        output: The authored child ``output:`` value.
        output_range: The authored ``output:`` value range, when located.
        input_ranges: Outer input name to authored binding-value range.
    """

    node_id: str
    owner: tuple[str, ...]
    flow_id: str
    nodes_reader: NodeReader
    output: Any
    output_range: SourceRange | None
    input_ranges: dict[str, SourceRange]


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
    nodes_reader: NodeReader,
    *,
    owner: tuple[str, ...],
) -> tuple[NodeReader, dict[str, InlineBody]]:
    """Validate, retain, and mask inline CALL/MAP/LOOP bodies.

    The returned reader is an occurrence-local overlay. Named composites remain
    unchanged. An inline composite becomes an ordinary private ``call:`` in that
    overlay, while the returned body record retains its original child/output
    values and source locations. The supplied reader and its YAML anchors remain
    read-only.
    """
    overlay = _overlay_node_occurrences(nodes_reader)
    bodies: dict[str, InlineBody] = {}
    if not isinstance(overlay.value, dict):
        return overlay, bodies

    for node_id in list(overlay.value):
        raw_body = overlay.value[node_id]
        if not isinstance(raw_body, dict):
            continue
        if raw_body.get("kind") not in _INLINE_COMPOSITE_KINDS:
            continue

        body_reader = overlay.child(node_id)
        has_call = "call" in raw_body
        has_nodes = "nodes" in raw_body
        if has_call == has_nodes:
            conflict_range = (
                body_reader.range_of("nodes") if has_nodes else body_reader._own_range()
            )
            raise LoadError(
                f"node {node_id!r} (kind={raw_body['kind']}): exactly one of "
                "`call:` or inline `nodes:` is required",
                range=conflict_range,
            )

        if has_call:
            if "output" in raw_body:
                raise LoadError(
                    f"node {node_id!r} (kind={raw_body['kind']}): `output:` is only "
                    "valid with inline `nodes:`",
                    range=body_reader.range_of("output"),
                )
            continue

        child_nodes = raw_body["nodes"]
        if not isinstance(child_nodes, dict):
            raise LoadError(
                f"node {node_id!r} (kind={raw_body['kind']}): inline `nodes:` must "
                "be a non-empty mapping",
                range=body_reader.range_of("nodes"),
            )
        if not child_nodes:
            raise LoadError(
                f"node {node_id!r} (kind={raw_body['kind']}): inline `nodes:` must "
                "be a non-empty mapping",
                range=body_reader.range_of("nodes"),
            )
        if "output" not in raw_body:
            raise LoadError(
                f"node {node_id!r} (kind={raw_body['kind']}): inline `nodes:` "
                "requires `output:`",
                range=body_reader._own_range(),
            )

        outer_input = raw_body.get("input", {})
        if not isinstance(outer_input, dict) or not all(
            isinstance(key, str) for key in outer_input
        ):
            raise LoadError(
                f"node {node_id!r} (kind={raw_body['kind']}): inline `input:` must "
                "be a mapping with string keys",
                range=body_reader.range_of("input"),
            )

        child_owner = (*owner, "node", node_id, "inline")
        flow_id = inline_flow_id(child_owner)
        body = InlineBody(
            node_id=node_id,
            owner=child_owner,
            flow_id=flow_id,
            nodes_reader=body_reader.child("nodes"),
            output=raw_body["output"],
            output_range=body_reader.range_of("output"),
            input_ranges=body_reader.mapping_value_ranges("input"),
        )
        bodies[node_id] = body
        _mask_inline_body(body_reader, flow_id)

    return overlay, bodies


def _mask_inline_body(reader: NodeReader, flow_id: str) -> None:
    """Replace one copied ``nodes:``/``output:`` pair with its private ``call:``."""
    masked: dict[Any, Any] = {}
    for key, value in reader.value.items():
        if key == "nodes":
            masked["call"] = flow_id
        elif key != "output":
            masked[key] = value
    reader.value.clear()
    reader.value.update(masked)

    pairs = getattr(reader.yaml_node, "value", None)
    if not isinstance(pairs, list):
        return
    masked_pairs: list[tuple[yaml.Node, yaml.Node]] = []
    for key_node, value_node in pairs:
        key = getattr(key_node, "value", None)
        if key == "output":
            continue
        if key == "nodes":
            key_node.value = "call"
        masked_pairs.append((key_node, value_node))
    reader.yaml_node.value = masked_pairs


def _overlay_node_occurrences(reader: NodeReader) -> NodeReader:
    """Return a shallow, occurrence-local overlay of one `nodes:` mapping.

    The outer mapping and every mapping-valued node body get independent
    constructed/YAML wrappers and copied key nodes. Child value nodes are reused.
    A spanless reader receives the same constructed-value isolation without a YAML
    tree. Non-mapping readers are returned unchanged.
    """
    if not isinstance(reader.value, dict):
        return reader

    if not isinstance(reader.yaml_node, yaml.MappingNode):
        values = {
            node_id: dict(body) if isinstance(body, dict) else body
            for node_id, body in reader.value.items()
        }
        return NodeReader.spanless(values)

    range_copies: list[tuple[yaml.Node, yaml.Node]] = []
    values, mapping = _copy_mapping(reader.value, reader.yaml_node, range_copies)

    for index, (copied_key, original_value_node) in enumerate(list(mapping.value)):
        node_id = getattr(copied_key, "value", None)
        body = reader.value.get(node_id)
        if not isinstance(body, dict) or not isinstance(
            original_value_node, yaml.MappingNode
        ):
            continue
        body_value, body_node = _copy_mapping(body, original_value_node, range_copies)
        values[node_id] = body_value
        mapping.value[index] = (copied_key, body_node)

    return reader.with_overlay(values, mapping, range_copies)


def _copy_mapping(
    value: dict,
    node: yaml.MappingNode,
    range_copies: list[tuple[yaml.Node, yaml.Node]],
) -> tuple[dict[Any, Any], yaml.MappingNode]:
    """Shallow-copy one constructed/YAML mapping and record copied range owners."""
    copied_node = copy(node)
    copied_pairs: list[tuple[yaml.Node, yaml.Node]] = []
    copied_node.value = copied_pairs
    range_copies.append((copied_node, node))

    for key_node, value_node in node.value:
        copied_key = copy(key_node)
        copied_pairs.append((copied_key, value_node))
        range_copies.append((copied_key, key_node))

    return dict(value), copied_node
