"""CALL — the REF driver (`kind: call`), internal-only build target.

Charter: this package owns the `CallNode` that applies a callable ONCE (REF). It is the REF half
of the REF/MAP pair; the MAP half is `nodes.map.MapNode` (`kind: map` + `over:`). The two are
distinct typed drivers. `CallNode` carries no `over`/`parallel`; `run` returns one `Grow(Flow)`
*description* for the engine's generic `_apply_grow` to splice into the live graph.

Imports flow one way: `nodes.base` (peer) + a deferred `typesys.seeding` import inside `run`
(keeping the ladder clean). `compose.kinds.call.build` is the caller.
"""

from agent_composer.nodes.call.node import CallNode

__all__ = ["CallNode"]
