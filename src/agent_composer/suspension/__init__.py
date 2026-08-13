"""Durable suspend/resume machinery.

A node that awaits something (a person, an external event) emits `PauseRequested`
carrying a typed `PauseReason`. The engine parks that node, records it, and the
run can be serialized to a `RunCheckpoint`. The host satisfies the pause by
delivering the awaited value as the parked leaf's `Output` via a
`DeliverAnswerCommand` and re-invoking `resume()` on the live engine.
"""

from agent_composer.suspension.checkpoint import RunCheckpoint
from agent_composer.suspension.commands import (
    AbortCommand,
    Command,
    DeliverAnswerCommand,
)
from agent_composer.suspension.expansions import GrowRecord
from agent_composer.suspension.pause import (
    EventAwaited,
    HumanInputRequired,
    PauseReason,
    ScheduledPause,
)

__all__ = [
    "AbortCommand",
    "Command",
    "DeliverAnswerCommand",
    "EventAwaited",
    "GrowRecord",
    "HumanInputRequired",
    "PauseReason",
    "RunCheckpoint",
    "ScheduledPause",
]
