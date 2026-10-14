"""kind=tool — the deterministic-tool node's descriptor + parse/build handle.

One of the two-axis `compose/nodes/<kind>.py` modules: it owns the `tool` node's frozen
`ToolDescriptor`, inheriting the template `parse` from `NodeDescriptor` (which records each
field's `SourceRange`), and overriding `build` (which constructs the runtime `ToolNode` +
wiring). `_finish` stamps only the descriptor's opaque origin onto that node and retains
concrete ranges on the Compose-owned `BuildResult`. Registered into
`KINDS`/`BINDING_FIELDS` at import so the shared dispatch routes here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from yian.compose.helpers.errors import LoadDiagnostics
from yian.compose.nodes.base import BuildContext, BuildResult, NodeDescriptor, reject_non_text
from yian.diagnostics import SourceRange
from yian.nodes.tool import ToolNode


@dataclass(frozen=True, kw_only=True)
class ToolDescriptor(NodeDescriptor):
    """kind=tool — a direct, deterministic call to one tool in the run's `ToolSet`.

    Inherits the common descriptor fields (`id`/`node_name`/`depends_on`/`runs_after`/`env` +
    the Compose-only `range`/`field_ranges` parse metadata), the normalized `inputs` binding
    carrier, the optional normalized `outputs` declaration, and the template `parse` from
    `NodeDescriptor`; its `_validate_fields` hook rejects a `tool_id:` YAML read as non-text
    (including `null`), then runs the base asserts-list check. Only the tool-specific fields
    (`tool_id`, `asserts`) are declared here; a tool's authored `input:` is stored in the
    base's `inputs` map, the same binding surface every value-shaped leaf kind uses.

    `output:` is optional. When present it becomes the node's output contract, and the
    engine validates the tool's return value against it on every call.

    Class attrs: `kind="tool"`; `allowed` adds `tool_id`, normalized `inputs`/`outputs`, and
    `asserts` to the commons; `required` is `{"tool_id"}`; `binding_fields=("inputs",)` — a
    tool's input map is its sink-binding surface, walked kind-blind by `bindings.py`.
    """

    kind = "tool"
    allowed = NodeDescriptor.allow("tool_id", "inputs", "outputs", "asserts")
    required = NodeDescriptor.require("tool_id")
    binding_fields = ("inputs",)

    tool_id: str
    asserts: list[str] = field(default_factory=list)

    @classmethod
    def _validate_fields(
        cls,
        node_id: str,
        kwargs: dict[str, Any],
        ranges: dict[tuple[str, str], SourceRange],
        own_range: Optional[SourceRange],
        diagnostics: LoadDiagnostics,
    ) -> None:
        """Reject a non-text `tool_id:`, then run the base checks."""
        reject_non_text(
            cls.kind,
            node_id,
            "tool_id",
            kwargs.get("tool_id"),
            ranges.get(("field", "tool_id")) or own_range,
            diagnostics,
            required=True,
        )
        super()._validate_fields(node_id, kwargs, ranges, own_range, diagnostics)

    def build(self, ctx: BuildContext) -> BuildResult:
        """Construct the runtime `ToolNode` (+ wiring) for this `ToolDescriptor`.

        Node-side `params` (the sink NAMES) + flow-owned `inputs` wiring, and `output_contract`
        from the optional authored `output:` (unset when absent). `_finish` stamps the descriptor
        origin on the node and keeps its ranges on the Compose-owned `BuildResult`. The sink
        surface is the base's `inputs` map, so it uses the inherited
        `self.sink_params()`/`self.sink_wiring()`.
        `ctx.resolver` is unused (leaf kind) and the inline-CODE anchors on `ctx` are read only
        by the CODE build.
        """
        node = ToolNode(self.id, tool_id=self.tool_id, title=self.node_name)
        node.output_contract = self.output_contract(ctx)
        node.params = self.sink_params()
        wiring = self.sink_wiring()
        return self._finish(node, wiring)
