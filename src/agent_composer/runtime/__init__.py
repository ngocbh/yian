"""Kind-blind scheduling state and the engine drain.

Runtime executes the compiled topology and reduces only the closed node-result
sum.  Graph-growth construction arrives as an injected callable from the run
boundary, so runtime interprets no node kind and owns no compile-time expansion.

Knows about:   events, typesys, nodes, compile.model topology, suspension, and llm_clients.
Never imports: compile.expand, compose, the package-root run boundary, CLI, or provider
               SDKs.
"""

from agent_composer.runtime.engine import FlowEngine
from agent_composer.runtime.state_manager import StateManager

__all__ = ["FlowEngine", "StateManager"]
