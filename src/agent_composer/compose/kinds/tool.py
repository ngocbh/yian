"""kind=tool — the deterministic-tool node's descriptor + parse/build handle (#70).

One of the two-axis `compose/kinds/<kind>.py` modules: it owns the `tool` node's frozen
`ToolDescriptor`, inheriting the template `parse` from `NodeDescriptor` (which records each
field's `SourceRange`), and overriding `build` (which constructs the runtime `ToolNode` +
wiring and stamps span metadata onto it). Registered into `KINDS`/`BINDING_FIELDS` at import
so the shared dispatch routes here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agent_composer.compose.kinds.base import NodeDescriptor
from agent_composer.compose.kinds.helpers import (
    _sink_params,
    _sink_wiring,
)
from agent_composer.nodes.tool import ToolNode


@dataclass(frozen=True, kw_only=True)
class ToolDescriptor(NodeDescriptor):
    """kind=tool — a direct, deterministic call into TOOL_REGISTRY (untyped args).

    Inherits the common node fields (`id`/`node_name`/`depends_on`/`runs_after`/`env` + the
    `range`/`field_ranges` parse metadata) and the template `parse` from `NodeDescriptor` (no
    custom parse hook — tool's field legality is the default allowed/required key check + the
    asserts-list check). Only the tool-specific fields (`tool_id`, `args`, `asserts`) are
    declared here; a tool's sink surface is its OWN `args` map, not the base's `inputs`.

    Class attrs: `kind="tool"`; `allowed` adds `tool_id`/`args`/`asserts` to the commons;
    `required` is `{"tool_id"}`; `binding_fields=("args",)` — a tool's arg map is its
    sink-binding surface (the tool's `inputs` analogue), walked kind-blind by `bindings.py`.
    """

    kind = "tool"
    allowed = NodeDescriptor.allow("tool_id", "args", "asserts")
    required = NodeDescriptor.require("tool_id")
    binding_fields = ("args",)

    tool_id: str
    args: dict[str, Any] = field(default_factory=dict)
    asserts: list[str] = field(default_factory=list)

    def build(self, *, registry=None, resolver=None, **_loader_hints):
        """Construct the runtime `ToolNode` (+ wiring) for this `ToolDescriptor`.

        Mirrors the legacy `build.build_leaf_node` tool path (node-side `params` + flow-owned
        `args` wiring; a tool declares no `outputs:`, so `output_type` stays unset) and stamps
        `.range`/`.field_ranges` from the descriptor via `_finish`. The sink surface is the
        tool's `args` map — NOT the base's `inputs` — so it uses the explicit `_sink_params`/
        `_sink_wiring` helpers over `self.args` rather than `self.sink_params()`/`self.sink_wiring()`
        (those read `self.inputs`). `resolver` is unused (leaf kind) and `_loader_hints` absorbs
        the inline-CODE traceback anchors that only the CODE build reads; both ride the shared
        build signature.
        """
        node = ToolNode(self.id, tool_id=self.tool_id, title=self.node_name)
        node.params = _sink_params(self.args)
        wiring = _sink_wiring(self.args)
        return self._finish(node, wiring)
