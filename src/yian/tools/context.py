"""The per-call context yian passes to a host tool that asks for it.

A tool function receives a ``ToolContext`` when its first parameter is annotated
``ToolContext`` or ``ToolContext[T]``.

Uses only the Python standard library.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")


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
            Starts at 1 and goes up by one on each retry of the same call.
        tool_name (`str`):
            The id the tool is registered under.
        run_id (`str`):
            The run's id, or `""` outside a run.
        node (`str`):
            The expanded id of the calling node, or `""` outside a run.
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
