"""The per-call context yian passes to a host tool that asks for it.

A tool function receives a ``ToolContext`` when its first parameter is annotated
``ToolContext`` or ``ToolContext[T]``.

Uses only the Python standard library.
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
import threading
from typing import Generic, TypeVar

T = TypeVar("T")

# Set by `ToolSet` around each call, so a tool reads its own run's event even when
# several runs share the tool.
_STOP_EVENT: ContextVar[threading.Event | None] = ContextVar("yian_tool_stop_event", default=None)


def _current_stop() -> threading.Event | None:
    """Return the stop event of the run whose tool call is running on this thread.

    For tools that wait and must end early when the run stops (used by MCP tools).

    Returns:
        `threading.Event | None`:
            The calling run's stop event, or `None` outside a tool call or when the
            calling tool set has no stop event.
    """
    return _STOP_EVENT.get()


@dataclass(frozen=True)
class ToolContext(Generic[T]):
    """Who is calling a tool, and the host's app object.

    Args:
        call_id (`str`):
            An opaque id for this call. For a TOOL node it is the run id plus the
            expanded node path, for example `run-7f3a/loop#3/place`, and stays the
            same when a resumed run re-runs the same call. For a call from an
            AGENT's tool loop it is unique per call but not stable across resume
            until AGENT steps are recorded. Outside a run (an unbound tool set)
            each call gets a fresh id.
        attempt (`int`):
            The attempt number, starting at 1. For a TOOL node it counts the
            attempts at the same call: 2 on its first retry, and the next number
            when a resumed run re-runs an interrupted attempt. Every other call,
            including one from an AGENT's tool loop, sees 1.
        tool_name (`str`):
            The id the tool is registered under.
        run_id (`str`):
            The run's id, or `""` outside a run.
        node (`str`):
            The expanded id of the calling node, or `""` when the caller names
            none. A node called directly, outside a run, still passes its id.
        context (`T`):
            The host's app object from `run_flow(..., context=...)` or
            `resume_flow(..., context=...)`. It is never recorded.

    Example:
        ```python
        def place(ctx: ToolContext[Broker], symbol: str) -> str:
            return ctx.context.submit(symbol, idempotency_key=ctx.call_id)
        ```
    """

    call_id: str
    attempt: int
    tool_name: str
    run_id: str
    node: str
    context: T
