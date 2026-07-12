"""kind=wait — the suspend-until-timestamp node's descriptor + parse/build handle (#70).

One of the two-axis `compose/nodes/<kind>.py` modules: it owns the `wait` node's frozen
`WaitDescriptor`, inheriting the template `parse` from `NodeDescriptor` (which records each
field's `SourceRange`), and overriding `build` (which constructs the runtime `WaitNode` + the
reserved `until` wiring and stamps span metadata onto it). Registered into `KINDS`/
`BINDING_FIELDS` at import so the shared dispatch routes here.
"""

from __future__ import annotations

from dataclasses import dataclass

from agent_composer.compose.nodes.base import BuildContext, BuildResult, NodeDescriptor
from agent_composer.nodes.wait import WaitNode


@dataclass(frozen=True, kw_only=True)
class WaitDescriptor(NodeDescriptor):
    """kind=wait — suspend until a timestamp (until:); produces no value.

    Inherits the common node fields (`id`/`node_name`/`depends_on`/`runs_after`/`env` + the
    `range`/`field_ranges` parse metadata) and the template `parse` from `NodeDescriptor` (no
    custom parse hook — a wait authors no `asserts`, so the default asserts check is a no-op).
    Only `until` is declared here; a wait never populates `inputs`/`outputs`.

    Class attrs: `kind="wait"`; `allowed` adds only `until` to the commons; `required` is
    `{"until"}`; `binding_fields=()` — a WAIT's `until` source rides the reserved wiring key,
    not the kind-blind inline-call walk, so there is no author sink-binding surface.
    """

    kind = "wait"
    allowed = NodeDescriptor.allow("until")
    required = NodeDescriptor.require("until")
    binding_fields = ()

    until: str

    def build(self, ctx: BuildContext) -> BuildResult:
        """Construct the runtime `WaitNode` (+ wiring) for this `WaitDescriptor`.

        Mirrors the legacy `build.build_leaf_node` wait path: a timed WAIT's `until` SOURCE rides
        the reserved wiring key (`{"until": self.until}`), never a node-side param, and the node
        carries only the `is_timed` residual. Stamps `.range`/`.field_ranges` from the descriptor
        via `_finish` (which wraps them in a `BuildResult`). `ctx.resolver` is unused (leaf kind)
        and the inline-CODE anchors on `ctx` are read only by the CODE build.
        """
        node = WaitNode(self.id, is_timed=self.until is not None, title=self.node_name)
        node.params = []
        wiring = {"until": self.until} if self.until is not None else {}
        return self._finish(node, wiring)
