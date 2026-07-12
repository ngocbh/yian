"""MAP — the `List.map` driver (`kind: map` + `over:`), internal-only build target.

Charter: this package owns the `MapNode` that maps a callable over a list. It is the MAP half of
the REF/MAP pair re-split out of the unified CALL node — `kind: map` builds a `MapNode`, `kind: call`
builds REF's `CallNode` (`nodes.call`). The two are distinct typed drivers; `MapNode`
discriminates by KIND (`NodeKind.MAP`), carries NO `over` attribute, and the iteration SOURCE rides
`flow.wiring[id]["over"]`. `run` returns a `Grow(Flow)` (the N element clones + a list-mode
END fan-in) for the engine's generic `_apply_grow` to splice into the live graph.

Imports flow one way: `nodes.base` (peer) + a deferred `typesys.seeding` import inside `run`
(matching `nodes.call`, keeping the ladder clean). `compose.nodes.map.build` is the caller.
"""

from agent_composer.nodes.map.node import MapNode

__all__ = ["MapNode"]
