"""kind=model — the ML-inference node's descriptor + parse/build handle.

One of the two-axis `compose/nodes/<kind>.py` modules: it owns the `model` node's frozen
`ModelDescriptor`, inheriting the template `parse` from `NodeDescriptor` (which records each
field's `SourceRange`), and overriding `build` (which constructs the runtime `ModelNode` +
wiring). `_finish` stamps only the descriptor's opaque origin onto that node and retains
concrete ranges on the Compose-owned `BuildResult`. Registered into
`KINDS`/`BINDING_FIELDS` at import so the shared dispatch routes here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from yian.compose.nodes.base import BuildContext, BuildResult, NodeDescriptor
from yian.nodes.model import ModelNode


@dataclass(frozen=True, kw_only=True)
class ModelDescriptor(NodeDescriptor):
    """kind=model — ML inference (a `model_id`, optional weights/runtime).

    Inherits the common descriptor fields (`id`/`node_name`/`depends_on`/`runs_after`/`env` +
    the Compose-only `range`/`field_ranges` parse metadata) from `NodeDescriptor`, along with the `input:`
    call-arg map and the template `parse` (no custom parse hook — model's field legality is the
    default allowed/required key check + the asserts-list check). Only the model-specific fields
    (`model_id`, `weights_uri`, `runtime`, `asserts`) are declared here; `inputs`/`outputs` come
    from the base.

    Class attrs: `kind="model"`; `allowed` adds `model_id`, normalized `inputs`/`outputs`, `weights_uri`,
    `runtime`/`asserts` to the commons; `required` is `{"model_id"}`; `binding_fields=("inputs",)`
    — `model_id`/`weights_uri`/`runtime` are static refs, not `${...}` bindings.
    """

    kind = "model"
    allowed = NodeDescriptor.allow(
        "model_id", "inputs", "outputs", "weights_uri", "runtime", "asserts"
    )
    required = NodeDescriptor.require("model_id")
    binding_fields = ("inputs",)

    model_id: str
    weights_uri: Optional[str] = None
    runtime: Optional[str] = None
    asserts: list[str] = field(default_factory=list)

    def build(self, ctx: BuildContext) -> BuildResult:
        """Construct the runtime `ModelNode` (+ wiring) for this `ModelDescriptor`.

        Builds the model leaf (node-side `params` + flow-owned `inputs` wiring,
        `output_contract` from authored `output:`). `_finish` stamps the descriptor origin on the node
        and keeps its ranges on the Compose-owned `BuildResult`. `ctx.resolver` is unused (leaf
        kind), and the inline-CODE anchors on `ctx` are read only by the CODE build.
        """
        node = ModelNode(
            self.id,
            model_id=self.model_id,
            weights_uri=self.weights_uri,
            runtime=self.runtime,
            title=self.node_name,
        )
        node.output_contract = self.output_contract(ctx)
        node.params = self.sink_params()
        wiring = self.sink_wiring()
        return self._finish(node, wiring)
