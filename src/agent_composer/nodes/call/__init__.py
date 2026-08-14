"""CALL — the authorable REF driver (`kind: call`).

Charter: this package owns the `CallNode` that applies a callable ONCE (REF). It is the REF half
of the REF/MAP pair; the MAP half is `nodes.map.MapNode` (`kind: map` + `over:`). The two are
distinct typed drivers. `CallNode` carries no `over`/`parallel`; `run` returns a seed-first `Grow`
that the compile-owned builder turns into a fragment for the engine to splice.

Knows about: nodes.base and peer node modules.
Never imports: compile, compose, runtime, the run boundary, CLI, or provider SDKs.
"""

from agent_composer.nodes.call.node import CallNode

__all__ = ["CallNode"]
