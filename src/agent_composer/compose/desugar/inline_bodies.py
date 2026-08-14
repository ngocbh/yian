"""Occurrence-local located overlays for inline composite-body lowering.

YAML aliases share constructed mappings and composed nodes. Merge expansion can
also reuse key/value nodes from an anchor. Inline-body lowering must later rename
and remove fields without changing any sibling occurrence or anchor, so this
module first gives each node occurrence its own shallow mapping wrappers.

The overlay copies constructed mappings, YAML `MappingNode` wrappers, pair lists,
and key scalar nodes. Value nodes remain shared and read-only. Copied-node ranges
live in a small lookup layered over the original span table through `NodeReader`.
"""

from __future__ import annotations

from copy import copy
from typing import Any

import yaml

from agent_composer.compose.helpers.reader import NodeReader


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
