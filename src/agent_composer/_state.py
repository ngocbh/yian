"""Dependency-free scheduler-state vocabulary shared across engine layers.

This internal leaf lets compiled topology and durable checkpoints agree on node
and edge state without making the lower suspension package import compile.
"""

from enum import Enum

__all__ = ["NodeState"]


class NodeState(str, Enum):
    """Per-node and per-edge scheduling state tracked by the runtime.

    ``UNKNOWN`` is undecided, ``TAKEN`` is committed, ``SKIPPED`` will not run,
    and ``EXPANDED`` marks a spawner whose returned growth was applied.
    """

    UNKNOWN = "unknown"
    TAKEN = "taken"
    SKIPPED = "skipped"
    EXPANDED = "expanded"
