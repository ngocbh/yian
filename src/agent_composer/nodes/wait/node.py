"""WAIT — the internal suspend-until primitive (two modes).

Not exposed to flow authors directly: the WATCH composite is built from the event
mode, and a timed pause is built from the timed mode. The mode is selected by
which field is authored:

- `event_spec` set -> EVENT mode: suspend with `EventAwaited(node_id=self.id)` until
  the host watcher delivers the event payload as this node's `Output` (a
  `DeliverAnswerCommand`) and resumes the run.
- `until` set -> TIMED mode: resolve the `${...}` binding to an ISO timestamp and
  suspend with `ScheduledPause(resume_at=ts, node_id=self.id)`. Resume is
  command-driven — the host releases the wait by delivering `value=None` as this
  node's `Output`; the engine never reads wall-clock. `date`/`datetime` segments
  resolve to ISO strings, so the resolver accepts a str directly.

Both mirror HUMAN_INPUT's deliver-as-Output model: the node ALWAYS pauses on
its single run; the engine delivers the answer — no re-run.
"""

from typing import Any, Optional

from agent_composer.nodes.base import Node, NodeKind, Pause
from agent_composer.typesys.pool import VariablePool
from agent_composer.suspension.pause import EventAwaited, ScheduledPause


def resolve_until(until_src: Any, pool: VariablePool) -> str:
    """Resolve a WAIT `until` source to an ISO timestamp string.

    A `${...}` binding (or a literal) is evaluated against the pool; `date`/`datetime` segments
    resolve to ISO strings (validated + returned), a raw `.isoformat()` object is formatted
    defensively, anything else is a loud `ValueError`. Lifted out of `WaitNode` so the engine bind
    seam can pre-resolve `inputs["until"]` for a pure `WaitNode.run`."""
    from datetime import datetime as _datetime

    from agent_composer.expr import eval_binding
    from agent_composer.expr.expressions import resolve_reference

    val = eval_binding(until_src, lambda path: resolve_reference(path, pool))
    if isinstance(val, str):
        # A str reaches `ScheduledPause.resume_at` unchecked, so a malformed literal (e.g.
        # `until: "not-a-date"`) would pause forever. Validate the ISO shape here; `fromisoformat`
        # (3.11+) covers date, datetime, and a trailing `Z`.
        try:
            _datetime.fromisoformat(val)
        except ValueError:
            raise ValueError(
                f"wait `until` ({until_src!r}) is not an ISO timestamp: {val!r}"
            )
        return val
    if hasattr(val, "isoformat"):  # defensive: a raw date/datetime object
        return val.isoformat()
    raise ValueError(
        f"wait `until` ({until_src!r}) did not resolve to a date/datetime: {val!r}"
    )


class WaitNode(Node):
    """
    The internal suspend-until primitive, in two modes (not authored directly).

    EVENT mode (`event_spec` set) parks with `EventAwaited` until the host watcher delivers the
    payload as this node's Output; TIMED mode parks with `ScheduledPause(resume_at=...)` (the
    `until` source is pre-resolved into `inputs["until"]` by the engine bind seam). Both mirror
    HUMAN_INPUT's deliver-as-Output model — the node always pauses on its single run.

    Args:
        node_id (`str`):
            The node's unique id.
        is_timed (`bool`, *optional*, defaults to `False`):
            The timed/event discriminator; the timed `until` source rides `flow.wiring`.
        event_spec (`dict`, *optional*, defaults to `None`):
            EVENT mode: what to watch; defaults to `{}`.
        poll (`dict`, *optional*, defaults to `None`):
            EVENT mode: how often to re-check; defaults to `{}`.
        title (`str`, *optional*, defaults to `None`):
            Display title.
    """

    kind = NodeKind.WAIT

    def __init__(
        self,
        node_id: str,
        *,
        is_timed: bool = False,
        event_spec: Optional[dict[str, Any]] = None,
        poll: Optional[dict[str, Any]] = None,
        title: Optional[str] = None,
    ) -> None:
        # The timed/event discriminator: a timed WAIT's `until` SOURCE lives on
        # `CompiledFlow.wiring[id]["until"]` (the node/flow split), not the node — `is_timed` is the
        # only residual; the engine pre-resolves the source into inputs["until"] before `run`.
        super().__init__(node_id, title=title)
        self.is_timed = is_timed
        self.event_spec = event_spec or {}
        self.poll = poll or {}

    def bind_reserved(self, node_wiring: dict, pool) -> dict:
        """Pre-resolve the timed WAIT `until` source into a concrete ISO ts for `run`.

        Returns `{"until": <ISO ts>}` when `self.is_timed` (the `until` source rides
        `node_wiring["until"]`); `{}` for the event mode. Its presence in the record is the
        timed/event discriminator `run` reads. The timed resolution is located to the `until:`
        line — a bad ref or non-date value carets `until:`, not the node header. Stamp here (which
        has `self`), not in the free `resolve_until`. The event branch reserves nothing."""
        if self.is_timed:
            with self.locating_reserved("until"):
                return {"until": resolve_until(node_wiring["until"], pool)}
        return {}

    def reserved_wiring_keys(self) -> set[str]:
        """A timed WAIT reserves `until` (its author-wiring source); event mode reserves nothing."""
        return {"until"} if self.is_timed else set()

    def _run_impl(self, inputs: dict):
        # The engine pre-resolves a timed `until` into inputs["until"] (a concrete ISO ts);
        # its presence is the timed/event discriminator. `resolve_until` is the bind's job now.
        if "until" in inputs:  # timed mode
            return Pause(ScheduledPause(resume_at=inputs["until"], node_id=self.id))
        # event mode (unauthorable; WATCH) — the watcher delivers the payload as this node's Output
        return Pause(EventAwaited(event_spec=self.event_spec, poll=self.poll, node_id=self.id))
