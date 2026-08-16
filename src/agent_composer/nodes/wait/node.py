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
from agent_composer.nodes.binding import BindingError, _resolve_source, _source_origin
from agent_composer.typesys.pool import VariablePool


def resolve_until(until_src: Any, pool: VariablePool) -> str:
    """Resolve a raw or carrier-wrapped WAIT source to an ISO date/datetime.

    Exactly one semantic ``Binding`` layer is removed before a ``${...}``
    expression or literal is evaluated against ``pool``. Typed
    ``date``/``datetime`` values resolve to ISO strings; a raw object exposing
    ``isoformat()`` is formatted defensively. The WAIT body parses and
    normalizes the resulting absolute deadline.
    """
    origin = _source_origin(until_src)
    value = _resolve_source(until_src, pool)
    if isinstance(value, str):
        resolved = value
    elif hasattr(value, "isoformat"):
        resolved = value.isoformat()
    else:
        raise BindingError(
            f"wait `until` did not resolve to a date/datetime: {value!r}",
            failure_code="contract.input",
            origin=origin,
        )
    try:
        _deadline_utc(resolved)
    except (TypeError, ValueError, OverflowError) as exc:
        raise BindingError(
            f"wait `until` is not an ISO date/datetime: {resolved!r}",
            failure_code="contract.input",
            origin=origin,
        ) from exc
    return resolved


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
