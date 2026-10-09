"""END_ID — the flow's return boundary, internal-only.

Charter: this package owns the synthesized END_ID node — the IR realization of a flow's `output:`
section as its single terminal. The engine's run result is END_ID's committed value. Two modes:
RECORD (a flow's output:) and LIST (a MAP fan-in — the join over each element's child END_ID).

Imports flow one way: `nodes.base`/`nodes.binding` (peer/lower) only. The loader synthesizes it
and the engine reads it as the terminal.
"""

from yian.nodes.end.node import EndNode

__all__ = ["EndNode"]
