"""Dependency-free scheduler-state vocabulary shared across engine layers.

This internal leaf lets compiled topology and the authoritative run-log reducer
agree on node and edge state without introducing an import between those layers.
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
