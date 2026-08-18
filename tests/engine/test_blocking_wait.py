"""Blocking WAIT semantics and cooperative lifecycle wakeups.

WAIT owns only an absolute UTC deadline. It blocks one worker on the run-owned
stop event, returns ``Output(None)`` when the deadline is reached, and unwinds
through the private host-stop signal when run shutdown wakes it.
"""

from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone

import pytest

from agent_composer.compile.model import END_ID, START_ID
from agent_composer.events import NodeFailed, NodeStarted, NodeSucceeded, RunFailed
from agent_composer.nodes.base import Node, NodeKind, Output, _HostStop
from agent_composer.nodes.wait import WaitNode
from tests.engine._graph_builder import _engine, _graph


_NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)
_FUTURE = "2026-01-02T00:00:00+00:00"
_JOIN_TIMEOUT = 2.0


class _FakeEvent:
    """Record WAIT timeouts without sleeping; optionally report a host wake."""

    def __init__(self, *, wakes: bool = False) -> None:
        self.wakes = wakes
        self.timeouts: list[float | None] = []

    def wait(self, timeout: float | None = None) -> bool:
        self.timeouts.append(timeout)
        return self.wakes


class _WakeOnSecondWait(_FakeEvent):
    """Time out once, then report a host wake on the next wait."""

    def wait(self, timeout: float | None = None) -> bool:
        self.timeouts.append(timeout)
        return len(self.timeouts) == 2


class _ObservedEvent(threading.Event):
    """A real stop event that exposes when WAIT has entered ``wait()``."""

    def __init__(self) -> None:
        super().__init__()
        self.wait_entered = threading.Event()

    def wait(self, timeout: float | None = None) -> bool:
        self.wait_entered.set()
        return super().wait(timeout)


class _FailAfterWaitStarts(Node):
    """Fail only after the sibling WAIT is observably blocked."""

    kind = NodeKind.CODE

    def __init__(self, node_id: str, wait_entered: threading.Event) -> None:
        super().__init__(node_id)
        self._wait_entered = wait_entered

    def forward(self, inputs: dict) -> Output:
        assert self._wait_entered.wait(timeout=_JOIN_TIMEOUT)
        raise RuntimeError("sibling failed")


class _CompleteAfterWaitStarts(Node):
    """Succeed only after the sibling WAIT is observably blocked."""

    kind = NodeKind.CODE

    def __init__(self, node_id: str, wait_entered: threading.Event) -> None:
        super().__init__(node_id)
        self._wait_entered = wait_entered

    def forward(self, inputs: dict) -> Output:
        assert self._wait_entered.wait(timeout=_JOIN_TIMEOUT)
        return Output("done")


class _ObserverError(RuntimeError):
    """Identity-bearing callback error for first-cause assertions."""


def _patch_now(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("agent_composer.nodes.wait.node._utc_now", lambda: _NOW)


def _run_direct(until: str, monkeypatch: pytest.MonkeyPatch) -> tuple[Output, _FakeEvent]:
    _patch_now(monkeypatch)
    event = _FakeEvent()
    result = WaitNode("wait").run({"until": until}, stop_event=event)
    assert isinstance(result, Output)
    return result, event


def _wait_graph(*nodes: Node):
    wait = WaitNode("wait")
    body = [wait, *nodes]
    edges = [(START_ID, node.id) for node in body]
    edges += [(node.id, END_ID) for node in body]
    return _graph(body, edges, wiring={"wait": {"until": _FUTURE}})


def test_date_means_midnight_utc(monkeypatch: pytest.MonkeyPatch) -> None:
    result, event = _run_direct("2026-01-02", monkeypatch)

    assert result.value is None
    assert event.timeouts == [86_400.0]


def test_naive_datetime_means_utc(monkeypatch: pytest.MonkeyPatch) -> None:
    result, event = _run_direct("2026-01-01T01:02:03", monkeypatch)

    assert result.value is None
    assert event.timeouts == [3_723.0]


def test_aware_datetime_is_normalized_to_utc(monkeypatch: pytest.MonkeyPatch) -> None:
    result, event = _run_direct("2026-01-02T05:30:00+05:30", monkeypatch)

    assert result.value is None
    assert event.timeouts == [86_400.0]


@pytest.mark.parametrize(
    "deadline",
    ["2025-12-31T23:59:59+00:00", "2026-01-01T00:00:00+00:00"],
    ids=["past", "exact-boundary"],
)
def test_deadline_at_or_before_now_returns_immediately(
    deadline: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, event = _run_direct(deadline, monkeypatch)

    assert result.value is None
    assert event.timeouts == []


def test_wait_uses_remaining_absolute_duration(monkeypatch: pytest.MonkeyPatch) -> None:
    result, event = _run_direct("2026-01-01T00:00:12.500000+00:00", monkeypatch)

    assert result.value is None
    assert event.timeouts == [12.5]
    assert WaitNode.needs_stop_event is True


def test_far_future_deadline_waits_in_timeout_max_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    event = _FakeEvent()
    deadline = _NOW + timedelta(seconds=threading.TIMEOUT_MAX + 12.5)
    now_values = iter((_NOW, _NOW + timedelta(seconds=threading.TIMEOUT_MAX)))
    clock_reads: list[tuple[float | None, ...]] = []

    def utc_now() -> datetime:
        clock_reads.append(tuple(event.timeouts))
        return next(now_values)

    monkeypatch.setattr("agent_composer.nodes.wait.node._utc_now", utc_now)
    result = WaitNode("wait").run(
        {"until": deadline.isoformat()}, stop_event=event
    )

    assert result.value is None
    assert event.timeouts == [threading.TIMEOUT_MAX, 12.5]
    assert clock_reads == [(), (threading.TIMEOUT_MAX,)]


def test_host_stop_between_timeout_max_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    event = _WakeOnSecondWait()
    deadline = _NOW + timedelta(seconds=2 * threading.TIMEOUT_MAX + 12.5)
    now_values = iter((_NOW, _NOW + timedelta(seconds=threading.TIMEOUT_MAX)))
    monkeypatch.setattr(
        "agent_composer.nodes.wait.node._utc_now", lambda: next(now_values)
    )

    with pytest.raises(_HostStop) as caught:
        WaitNode("wait").run({"until": deadline.isoformat()}, stop_event=event)

    assert caught.value.primary is False
    assert event.timeouts == [threading.TIMEOUT_MAX, threading.TIMEOUT_MAX]


@pytest.mark.parametrize("num_workers", [0, 1])
def test_host_stop_wakes_wait_and_discards_its_outcome(
    num_workers: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_now(monkeypatch)
    stop_event = _ObservedEvent()
    engine = _engine(_wait_graph(), num_workers=num_workers, stop_event=stop_event)
    events: list[object] = []
    errors: list[BaseException] = []

    def drive() -> None:
        try:
            events.extend(engine.run())
        except BaseException as exc:  # noqa: BLE001 - assertion captures driver control flow
            errors.append(exc)

    driver = threading.Thread(target=drive, name="test-wait-host-stop")
    driver.start()
    assert stop_event.wait_entered.wait(timeout=_JOIN_TIMEOUT)

    stop_event.set()
    driver.join(timeout=_JOIN_TIMEOUT)

    assert not driver.is_alive()
    assert len(errors) == 1 and isinstance(errors[0], _HostStop)
    assert errors[0].primary is False
    assert not any(
        isinstance(event, NodeSucceeded) and event.node_id == "wait" for event in events
    )
    assert not any(
        isinstance(event, NodeFailed) and event.failure.node_id == "wait"
        for event in events
    )
    assert not any(thread.name.startswith("ac-worker-") for thread in threading.enumerate())


def test_observer_failure_wakes_wait_and_preserves_callback_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_now(monkeypatch)
    stop_event = _ObservedEvent()
    sibling = _CompleteAfterWaitStarts("sibling", stop_event.wait_entered)
    original = _ObserverError("observer failed")
    observed: list[object] = []

    def observer(event: object) -> None:
        observed.append(event)
        if isinstance(event, NodeSucceeded) and event.node_id == "sibling":
            raise original

    engine = _engine(
        _wait_graph(sibling),
        num_workers=2,
        on_event=observer,
        stop_event=stop_event,
    )
    with pytest.raises(_ObserverError) as caught:
        list(engine.run())

    assert caught.value is original
    assert stop_event.is_set()
    assert not any(
        isinstance(event, NodeFailed) and event.failure.node_id == "wait"
        for event in observed
    )
    assert not any(thread.name.startswith("ac-worker-") for thread in threading.enumerate())


def test_sibling_failure_wakes_wait_and_remains_terminal_cause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_now(monkeypatch)
    stop_event = _ObservedEvent()
    fail = _FailAfterWaitStarts("fail", stop_event.wait_entered)
    events = list(
        _engine(_wait_graph(fail), num_workers=2, stop_event=stop_event).run()
    )

    assert isinstance(events[-1], RunFailed)
    assert events[-1].failure.message == "sibling failed"
    assert stop_event.is_set()
    assert not any(
        isinstance(event, NodeSucceeded) and event.node_id == "wait" for event in events
    )
    assert not any(
        isinstance(event, NodeFailed) and event.failure.node_id == "wait"
        for event in events
    )
    assert not any(thread.name.startswith("ac-worker-") for thread in threading.enumerate())
