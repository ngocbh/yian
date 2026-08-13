"""Kind-blind scheduling state and the engine drain.

Runtime executes the compiled topology and reduces only the closed node-result
sum.  Graph-growth construction arrives as an injected callable from the run
boundary, so runtime interprets no node kind and owns no compile-time expansion.
Pooled drives own their workers through a full join and signal cooperative stop
before that join when active dispatch exits abnormally.

Knows about:   events, typesys, nodes, compile.model topology, and suspension.
Never imports: compile.expand, compose, the package-root run boundary, CLI, or provider
               SDKs.
"""

from agent_composer.runtime.engine import FlowEngine
from agent_composer.runtime.state_manager import StateManager

__all__ = ["FlowEngine", "StateManager"]
