"""LOOP — the authorable while/until/times fixpoint driver.

Owns the self-respawning ``LoopNode`` policy and seed shape; the compile-owned
builder constructs each body fragment and runtime splices it kind-blindly.

Knows about: nodes.base, expr, and peer node modules.
Never imports: compile, compose, runtime, suspension, the run boundary, CLI, or provider SDKs.
"""

from agent_composer.nodes.loop.node import LoopNode

__all__ = ["LoopNode"]
