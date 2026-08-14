"""WAIT node — block one worker until an absolute UTC date/datetime.

Owns absolute-deadline parsing and cooperative stop-event waiting. Its author
source remains in flow wiring; the node returns ``Output(None)`` and never owns
run-log or scheduler state.

Knows about: nodes.base, expr, typesys, and the Python standard library.
Never imports: compile, compose, runtime, the run boundary, CLI, or
               provider SDKs.
"""

from agent_composer.nodes.wait.node import WaitNode

__all__ = ["WaitNode"]
