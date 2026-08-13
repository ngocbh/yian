"""Temporary generic durable suspend/resume compatibility.

A node that returns the legacy ``Pause`` variant emits ``PauseRequested`` carrying
a typed ``PauseReason``. The engine parks that node and can serialize the run to a
``RunCheckpoint``; the host later delivers the value as the parked leaf's
``Output``. Authorable HUMAN_INPUT and AGENT ``ask_user`` calls now block inside
live node execution instead. This compatibility package is removed by the
subsequent run-log cutover.
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
