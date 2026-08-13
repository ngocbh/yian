"""WAIT — block one worker until an absolute UTC deadline.

WAIT accepts one resolved ISO date or datetime through its reserved ``until``
wiring. A date and a naive datetime are interpreted in UTC; an aware datetime
is normalized to UTC. The node waits cooperatively on the run-owned stop event,
so engine failure or host shutdown can wake and drain its worker.
"""

from datetime import datetime, timezone
from threading import TIMEOUT_MAX
from typing import Any

from agent_composer.nodes.base import Node, NodeKind, Output, _HostStop
from agent_composer.typesys.pool import VariablePool


def resolve_until(until_src: Any, pool: VariablePool) -> str:
    """Resolve a WAIT ``until`` source to an ISO date or datetime string.

    A ``${...}`` binding or literal is evaluated against ``pool``. Typed
    ``date``/``datetime`` values resolve to ISO strings; a raw object exposing
    ``isoformat()`` is formatted defensively. The WAIT body parses and
    normalizes the resulting absolute deadline.
    """
    from agent_composer.expr import eval_binding
    from agent_composer.expr.expressions import resolve_reference

    value = eval_binding(until_src, lambda path: resolve_reference(path, pool))
    if isinstance(value, str):
        return value
    if hasattr(value, "isoformat"):
        return value.isoformat()
    raise ValueError(
        f"wait `until` ({until_src!r}) did not resolve to a date/datetime: {value!r}"
    )


def _utc_now() -> datetime:
    """Return the current aware UTC wall-clock instant."""
    return datetime.now(timezone.utc)


def _deadline_utc(value: str) -> datetime:
    """Parse one ISO date/datetime and return its aware UTC instant."""
    deadline = datetime.fromisoformat(value)
    if deadline.tzinfo is None:
        deadline = deadline.replace(tzinfo=timezone.utc)
    return deadline.astimezone(timezone.utc)


class WaitNode(Node):
    """Block until the resolved absolute ``until`` deadline.

    The ``until`` source lives in ``CompiledFlow.wiring[id]["until"]`` and is
    pre-resolved by :meth:`bind_reserved`. ``_run_impl`` receives that concrete
    ISO value plus the exact run-owned stop event. A normal timeout returns
    ``Output(None)``; a lifecycle wake raises the private ``_HostStop`` signal so
    the engine discards the node outcome while draining.

    Args:
        node_id: The node's unique id.
        title: Optional display title.
    """

    kind = NodeKind.WAIT
    needs_stop_event = True

    def __init__(self, node_id: str, *, title: str | None = None) -> None:
        super().__init__(node_id, title=title)

    def bind_reserved(self, node_wiring: dict, pool: VariablePool) -> dict[str, str]:
        """Resolve the required ``until`` wiring into the node's input record."""
        return {"until": resolve_until(node_wiring["until"], pool)}

    def reserved_wiring_keys(self) -> set[str]:
        """Declare the required author-owned ``until`` wiring key."""
        return {"until"}

    def _run_impl(self, inputs: dict, *, stop_event) -> Output:
        """Wait for the remaining absolute duration or unwind on lifecycle stop."""
        deadline = _deadline_utc(inputs["until"])
        remaining = (deadline - _utc_now()).total_seconds()
        if remaining <= 0:
            return Output(None)

        # Event.wait raises OverflowError above TIMEOUT_MAX. Re-read wall time
        # only after each maximum-sized chunk; ordinary waits retain one call.
        while remaining > TIMEOUT_MAX:
            if stop_event.wait(timeout=TIMEOUT_MAX):
                raise _HostStop(primary=False)
            remaining = (deadline - _utc_now()).total_seconds()

        if remaining <= 0:
            return Output(None)
        if stop_event.wait(timeout=remaining):
            raise _HostStop(primary=False)
        return Output(None)
