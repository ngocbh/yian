"""LOOP — the authorable while/until/times fixpoint driver.

Owns the self-respawning ``LoopNode`` policy and seed shape; the compile-owned
builder constructs each body fragment and runtime splices it kind-blindly. It
also owns origin traversal for its cached body signature and exposes the baked
body through the source-neutral node hook.

Knows about: nodes.base, expr, and peer node modules.
Never imports: compile, compose, runtime, the run boundary, CLI, or provider SDKs.
"""

from yian.nodes.loop.node import LoopNode

__all__ = ["LoopNode"]
