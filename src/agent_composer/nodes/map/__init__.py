"""MAP — the `List.map` driver (`kind: map` + `over:`), internal-only build target.

Charter: this package owns the `MapNode` that maps a callable over a list. It is the MAP half of
the REF/MAP pair re-split out of the unified CALL node — `kind: map` builds a `MapNode`, `kind: call`
builds REF's `CallNode` (`nodes.call`). The two are distinct typed drivers; `MapNode`
discriminates by KIND (`NodeKind.MAP`), carries NO `over` attribute, and the iteration SOURCE rides
`flow.wiring[id]["over"]`. `run` returns a seed-first `Grow`; the compile-owned builder constructs
the N element clones and list-mode END fan-in for the engine to splice.

Imports flow one way through node/expr peers; it never imports `compile`.
`compose.nodes.map.build` is the caller.
"""

from agent_composer.nodes.map.node import MapNode

__all__ = ["MapNode"]
