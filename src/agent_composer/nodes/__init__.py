"""Pure node contract, per-kind implementations, and node-owned seed schemas.

Nodes consume bound input records and return the closed ``Output | Route | Pause |
Grow`` result sum.  Spawners return only a canonical ``Grow`` seed (plus ids to
prune); they never construct compiled graph fragments themselves.

Knows about:   events, expr, typesys, suspension, tools, llm_clients, and peer
               node modules.
Never imports: compile, compose, runtime, the package-root run boundary, or CLI.
"""

from agent_composer.nodes.base import Grow, Node, NodeKind, NodeResult, Output, Pause, Route

__all__ = ["Node", "NodeKind", "NodeResult", "Output", "Route", "Pause", "Grow"]
