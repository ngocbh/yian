"""MAP — the authorable `List.map` driver (`kind: map` + `over:`).

Charter: this package owns the `MapNode` that maps a callable over a list. It is the mapped
counterpart re-split from CALL: `kind: map` builds a `MapNode`, while `kind: call`
builds `CallNode` (`nodes.call`). The two are distinct typed drivers; `MapNode`
discriminates by KIND (`NodeKind.MAP`), carries NO `over` attribute, and the iteration SOURCE rides
`flow.wiring[id]["over"]`. `run` returns a seed-first `Grow`; the compile-owned builder constructs
the N element clones and list-mode END fan-in for the engine to splice. It also owns origin
traversal for its cached child signature and exposes its baked child through the source-neutral
node hook.

Knows about: nodes.base, expr, and peer node modules.
Never imports: compile, compose, runtime, the run boundary, CLI, or provider SDKs.
"""

from agent_composer.nodes.map.node import MapNode

__all__ = ["MapNode"]
