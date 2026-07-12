"""kind=tool — the deterministic-tool node's descriptor + parse/build handle (#70).

One of the two-axis `compose/kinds/<kind>.py` modules: it owns the `tool` node's frozen
`ToolDescriptor`, its located `parse(node_id, reader)` (which records each field's
`SourceRange`), and its `build(desc, *, registry, resolver)` (which constructs the runtime
`ToolNode` + wiring and stamps span metadata onto it). Registered into
`KINDS`/`BINDING_FIELDS` at import so the shared dispatch routes here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agent_composer.compose.errors import LoadError
from agent_composer.compose.reader import NodeReader
from agent_composer.compose.kinds.base import NodeDescriptor
from agent_composer.compose.kinds.helpers import (
    _sink_params,
    _sink_wiring,
    check_node_keys,
    read_common_fields,
)
from agent_composer.events import SourceRange
from agent_composer.nodes.tool import ToolNode

# Tool's own field surface (was the `tool` row of parser._KIND_SPECS): `tool_id` is the one
# required flat field; these flat keys are allowed besides the shared `_COMMON_FIELDS` + `kind`.
TOOL_REQUIRED: frozenset[str] = frozenset({"tool_id"})
TOOL_ALLOWED: frozenset[str] = frozenset({"tool_id", "args", "asserts"})
# Descriptor fields carrying inline-call binding strings (the kind-blind walk in bindings.py):
# `args` — a tool's arg map is its sink-binding surface (the tool's `inputs` analogue).
TOOL_BINDING_FIELDS: tuple[str, ...] = ("args",)


@dataclass(frozen=True, kw_only=True)
class ToolDescriptor(NodeDescriptor):
    """kind=tool — a direct, deterministic call into TOOL_REGISTRY (untyped args).

    Inherits the common node fields (`id`/`node_name`/`depends_on`/`runs_after`/`env` + the
    `range`/`field_ranges` parse metadata) from `NodeDescriptor`; only the tool-specific
    fields are declared here.
    """

    tool_id: str
    args: dict[str, Any] = field(default_factory=dict)
    asserts: list[str] = field(default_factory=list)


def parse(node_id: str, reader: NodeReader) -> ToolDescriptor:
    """Read one `kind: tool` node body (via a located `NodeReader`) into a `ToolDescriptor`.

    Records the node body's `SourceRange` (`desc.range`) and each present field's
    `SourceRange` (`desc.field_ranges[field]`). Field legality mirrors the legacy
    `parser._parse_node` tool path (allowed/required flat keys, `asserts:`/`env:` shape).
    """
    check_node_keys(node_id, "tool", reader, TOOL_ALLOWED, TOOL_REQUIRED)
    body = reader.value

    ranges: dict[str, SourceRange] = {}
    kwargs: dict[str, Any] = {"id": node_id}
    for key in TOOL_ALLOWED:
        if key not in body:
            continue
        rng = reader.range_of(key)
        if rng is not None:
            ranges[key] = rng
        kwargs[key] = body[key]

    asserts = kwargs.get("asserts")
    if asserts is not None and not (
        isinstance(asserts, list) and all(isinstance(a, str) for a in asserts)
    ):
        raise LoadError(
            f"node {node_id!r} (kind=tool): `asserts:` must be a list of strings",
            range=ranges.get("asserts"),
        )

    read_common_fields(node_id, "tool", reader, kwargs, ranges)

    return ToolDescriptor(**kwargs, range=reader._own_range(), field_ranges=ranges)


def build(desc: ToolDescriptor, *, registry, resolver=None, **_loader_hints):
    """Construct the runtime `ToolNode` (+ wiring) for one `ToolDescriptor`.

    Mirrors the legacy `build.build_leaf_node` tool path (node-side `params` + flow-owned
    `args` wiring; a tool declares no `outputs:`, so `output_type` stays unset) and
    additionally stamps `.range`/`.field_ranges` from the descriptor. `resolver` is unused
    (leaf kind) and `_loader_hints` absorbs the inline-CODE traceback anchors that only the
    CODE build reads; both ride the shared `KindSpec.build` signature.
    """
    node = ToolNode(desc.id, tool_id=desc.tool_id, title=desc.node_name)
    node.params = _sink_params(desc.args)
    wiring = _sink_wiring(desc.args)
    node.range = desc.range
    node.field_ranges = dict(desc.field_ranges)
    return node, wiring
