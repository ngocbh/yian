"""Host-stop propagation across the node seam, worker pool, and run boundary."""

from __future__ import annotations

import inspect
import queue
import threading
from types import SimpleNamespace

import pytest

from agent_composer.compile.model import END_ID, START_ID
from agent_composer.events import (
    NodeFailed,
    NodeStarted,
    NodeSucceeded,
    RunAborted,
    RunFailed,
    RunPaused,
)
from agent_composer.nodes.base import Node, NodeKind, Output, _HostStop
from agent_composer.run import run_flow
from agent_composer.runtime.engine import FlowEngine
from agent_composer.runtime.eval_node import eval_node
from agent_composer.suspension.commands import AbortCommand, DeliverAnswerCommand
from agent_composer.typesys.pool import VariablePool
from tests.engine._fakes import FailNode, PauseNode
from tests.engine._graph_builder import _graph


class _CapNode(Node):
    """Record the capabilities supplied to one direct node evaluation."""

    kind = NodeKind.CODE

    def __init__(self, node_id: str, *, opted_in: bool) -> None:
        super().__init__(node_id)
        self.needs_stop_event = opted_in
        self.caps: dict = {}

    def _run_impl(self, inputs, **caps):
        self.caps = caps
        return Output("done")


class _NeverRunsNode(Node):
    """Expose whether a pre-set stop event allowed the node body to start."""

    kind = NodeKind.CODE

    def __init__(self, node_id: str) -> None:
        super().__init__(node_id)
        self.ran = False

    def _run_impl(self, inputs):
        self.ran = True
        return Output("unexpected")


class _RaisesHostStop(Node):
    """Raise the private host-control signal from inside the node boundary."""

    kind = NodeKind.CODE

    def _run_impl(self, inputs):
        raise _HostStop


def test_stop_cap_is_exact_event_and_only_reaches_opted_in_node():
    stop_event = threading.Event()
    opted_in = _CapNode("aware", opted_in=True)
    ordinary = _CapNode("ordinary", opted_in=False)

    assert isinstance(list(eval_node(opted_in, None, VariablePool(), stop_event=stop_event))[-1],
                      NodeSucceeded)
    assert isinstance(list(eval_node(ordinary, None, VariablePool(), stop_event=stop_event))[-1],
                      NodeSucceeded)

    assert opted_in.caps == {"stop_event": stop_event}
    assert opted_in.caps["stop_event"] is stop_event
    assert ordinary.caps == {}


@pytest.mark.parametrize("num_workers", [0, 2])
def test_engine_passes_its_exact_event_only_to_opted_in_nodes(num_workers):
    stop_event = threading.Event()
    opted_in = _CapNode("aware", opted_in=True)
    ordinary = _CapNode("ordinary", opted_in=False)
    graph = _graph(
        [opted_in, ordinary],
        [
            (START_ID, "aware"),
            (START_ID, "ordinary"),
            ("aware", END_ID),
            ("ordinary", END_ID),
        ],
    )

    list(FlowEngine(graph, num_workers=num_workers, stop_event=stop_event).run())

    assert opted_in.caps == {"stop_event": stop_event}
    assert opted_in.caps["stop_event"] is stop_event
    assert ordinary.caps == {}


def test_direct_opted_in_eval_gets_private_unset_event_when_host_omits_one():
    node = _CapNode("aware", opted_in=True)

    list(eval_node(node, None, VariablePool()))

    assert set(node.caps) == {"stop_event"}
    assert isinstance(node.caps["stop_event"], threading.Event)
    assert not node.caps["stop_event"].is_set()


def test_preset_stop_starts_event_stream_but_never_runs_or_emits_node_failed():
    stop_event = threading.Event()
    stop_event.set()
    node = _NeverRunsNode("n")
    stream = eval_node(node, None, VariablePool(), stop_event=stop_event)

    assert isinstance(next(stream), NodeStarted)
    with pytest.raises(_HostStop):
        next(stream)

    assert not node.ran


@pytest.mark.parametrize("num_workers", [0, 1])
def test_preset_engine_stop_unwinds_without_node_failed_or_public_abort(num_workers):
    stop_event = threading.Event()
    stop_event.set()
    node = _NeverRunsNode("n")
    graph = _graph([node], [(START_ID, "n"), ("n", END_ID)])
    stream = FlowEngine(graph, num_workers=num_workers, stop_event=stop_event).run()
    events = []

    with pytest.raises(_HostStop):
        while True:
            events.append(next(stream))

    assert not node.ran
    assert not any(isinstance(event, (NodeFailed, RunFailed, RunAborted)) for event in events)
    assert not any(thread.name.startswith("ac-worker-") for thread in threading.enumerate())


def test_stop_set_while_node_started_prevents_serial_body_and_terminal_events():
    stop_event = threading.Event()
    node = _NeverRunsNode("n")
    graph = _graph([node], [(START_ID, "n"), ("n", END_ID)])
    stream = FlowEngine(graph, stop_event=stop_event).run()
    events = []

    while True:
        event = next(stream)
        events.append(event)
        if isinstance(event, NodeStarted) and event.node_id == "n":
            stop_event.set()
            break

    with pytest.raises(_HostStop):
        next(stream)

    assert not node.ran
    assert not any(isinstance(event, (NodeFailed, RunFailed, RunAborted)) for event in events)


def test_host_stop_bypasses_direct_eval_node_failed_funnel():
    stream = eval_node(_RaisesHostStop("n"), None, VariablePool())
    events = [next(stream)]

    with pytest.raises(_HostStop):
        next(stream)

    assert not any(isinstance(event, NodeFailed) for event in events)


def test_host_stop_bypasses_pooled_worker_node_failed_and_run_aborted():
    graph = _graph(
        [_RaisesHostStop("n")],
        [(START_ID, "n"), ("n", END_ID)],
    )
    stream = FlowEngine(graph, num_workers=1).run()
    events = []

    with pytest.raises(_HostStop):
        while True:
            events.append(next(stream))

    assert not any(isinstance(event, (NodeFailed, RunFailed, RunAborted)) for event in events)
    assert not any(thread.name.startswith("ac-worker-") for thread in threading.enumerate())


class _CooperativeSibling(Node):
    """Block until a failing sibling sets the shared host stop event."""

    kind = NodeKind.CODE
    needs_stop_event = True

    def __init__(self, node_id: str, entered: threading.Event, saw_stop: threading.Event) -> None:
        super().__init__(node_id)
        self._entered = entered
        self._saw_stop = saw_stop

    def _run_impl(self, inputs, *, stop_event):
        self._entered.set()
        stop_event.wait()
        if stop_event.is_set():
            self._saw_stop.set()
        return Output("late-success-must-be-discarded")


class _AbortBlockingNode(Node):
    """Wait briefly for an abort to wake the exact injected stop event."""

    kind = NodeKind.CODE
    needs_stop_event = True

    def __init__(self, node_id: str, entered: threading.Event, saw_stop: threading.Event) -> None:
        super().__init__(node_id)
        self._entered = entered
        self._saw_stop = saw_stop

    def _run_impl(self, inputs, *, stop_event):
        self._entered.set()
        if stop_event.wait(timeout=0.5):
            self._saw_stop.set()
        return Output("late-abort-output")


class _FailsAfterHostStop(Node):
    """Raise an ordinary node error only after an external host stop wins."""

    kind = NodeKind.CODE
    needs_stop_event = True

    def __init__(self, node_id: str, entered: threading.Event) -> None:
        super().__init__(node_id)
        self._entered = entered

    def _run_impl(self, inputs, *, stop_event):
        self._entered.set()
        assert stop_event.wait(timeout=1.0)
        raise RuntimeError("late node failure")


class _FreezeGetAfterNodeEntered(queue.Queue):
    """Hold the pooled dispatcher inside ``get`` until the late failure arrives."""

    def __init__(self, entered: threading.Event) -> None:
        super().__init__()
        self._entered = entered
        self.blocked = threading.Event()

    def get(self, block=True, timeout=None):
        if block and self._entered.is_set() and self.empty():
            self.blocked.set()
            return super().get(block=True)
        return super().get(block=block, timeout=timeout)


class _FailAfterSiblingStarts(FailNode):
    """Make the failure deterministic by waiting until its sibling is blocked."""

    def __init__(self, node_id: str, entered: threading.Event) -> None:
        super().__init__(node_id, "primary failure")
        self._entered = entered

    def _run_impl(self, inputs):
        assert self._entered.wait(timeout=1.0)
        return super()._run_impl(inputs)


def test_closing_active_pooled_stream_wakes_waiter_and_joins() -> None:
    """Generator close is an active-drive stop, not an unbounded worker join."""
    stop_event = threading.Event()
    entered = threading.Event()
    saw_stop = threading.Event()
    graph = _graph(
        [_CooperativeSibling("blocked", entered, saw_stop)],
        [(START_ID, "blocked"), ("blocked", END_ID)],
    )
    stream = FlowEngine(graph, num_workers=1, stop_event=stop_event).run()

    while True:
        event = next(stream)
        if isinstance(event, NodeStarted) and event.node_id == "blocked":
            assert entered.wait(timeout=1.0)
            break

    close_errors = []

    def close_stream() -> None:
        try:
            stream.close()
        except BaseException as exception:
            close_errors.append(exception)

    closer = threading.Thread(target=close_stream, name="test-close-pooled-stream")
    closer.start()
    closer.join(timeout=0.2)
    completed_without_rescue = not closer.is_alive()
    if not completed_without_rescue:
        stop_event.set()
        closer.join(timeout=1.0)

    assert completed_without_rescue
    assert not closer.is_alive()
    assert close_errors == []
    assert stop_event.is_set()
    assert saw_stop.is_set()
    assert not any(thread.name.startswith("ac-worker-") for thread in threading.enumerate())


def test_closing_stream_after_run_paused_does_not_poison_resume_event() -> None:
    """A terminal-yield close happens after worker drain and preserves resumability."""
    stop_event = threading.Event()
    graph = _graph(
        [PauseNode("pause")],
        [(START_ID, "pause"), ("pause", END_ID)],
    )
    stream = FlowEngine(graph, num_workers=1, stop_event=stop_event).run()

    while not isinstance(next(stream), RunPaused):
        pass
    stream.close()

    assert not stop_event.is_set()


def test_node_failure_sets_event_before_cooperative_drain_and_joins_workers():
    stop_event = threading.Event()
    entered = threading.Event()
    saw_stop = threading.Event()
    graph = _graph(
        [
            _FailAfterSiblingStarts("fail", entered),
            _CooperativeSibling("blocked", entered, saw_stop),
        ],
        [
            (START_ID, "fail"),
            (START_ID, "blocked"),
            ("fail", END_ID),
            ("blocked", END_ID),
        ],
    )

    events = list(FlowEngine(graph, num_workers=2, stop_event=stop_event).run())

    assert isinstance(events[-1], RunFailed)
    assert events[-1].error == "primary failure"
    assert stop_event.is_set()
    assert saw_stop.is_set()
    assert not any(
        isinstance(event, NodeSucceeded) and event.node_id == "blocked" for event in events
    )
    assert not any(thread.name.startswith("ac-worker-") for thread in threading.enumerate())


@pytest.mark.parametrize("num_workers", [0, 1])
def test_request_abort_wakes_blocking_node_but_remains_public_run_aborted(num_workers):
    """Legacy abort is the primary outcome while its event drains cooperative work."""

    entered = threading.Event()
    saw_stop = threading.Event()
    node = _AbortBlockingNode("blocked", entered, saw_stop)
    graph = _graph([node], [(START_ID, "blocked"), ("blocked", END_ID)])
    engine = FlowEngine(graph, num_workers=num_workers)
    events = []
    errors = []

    def drive() -> None:
        try:
            events.extend(engine.run())
        except BaseException as exc:
            errors.append(exc)

    driver = threading.Thread(target=drive, name="test-abort-driver")
    driver.start()
    assert entered.wait(timeout=1.0)

    engine.request_abort()
    driver.join(timeout=1.0)

    assert not driver.is_alive()
    assert errors == []
    assert isinstance(events[-1], RunAborted)
    assert engine.stop_event.is_set()
    assert saw_stop.is_set()
    assert not any(thread.name.startswith("ac-worker-") for thread in threading.enumerate())


def test_abort_command_sets_shared_stop_event() -> None:
    """The command form uses the same wakeup path as request_abort."""

    engine = FlowEngine(_graph([], []))

    engine._apply_command(AbortCommand(reason="operator stopped"))

    assert engine._cancel
    assert engine.stop_event.is_set()


def test_serial_late_node_failure_cannot_replace_external_host_stop() -> None:
    """A failure produced after stop is set is discarded before it becomes observable."""

    entered = threading.Event()
    stop_event = threading.Event()
    graph = _graph(
        [_FailsAfterHostStop("late", entered)],
        [(START_ID, "late"), ("late", END_ID)],
    )
    engine = FlowEngine(graph, stop_event=stop_event)
    events = []
    errors = []

    def drive() -> None:
        try:
            events.extend(engine.run())
        except BaseException as exc:
            errors.append(exc)

    driver = threading.Thread(target=drive, name="test-late-failure-serial")
    driver.start()
    assert entered.wait(timeout=1.0)
    stop_event.set()
    driver.join(timeout=1.0)

    assert not driver.is_alive()
    assert len(errors) == 1 and isinstance(errors[0], _HostStop)
    assert not any(isinstance(event, (NodeFailed, RunFailed)) for event in events)


def test_pooled_check_get_race_cannot_promote_late_node_failure() -> None:
    """The dispatcher rechecks stop after ``get`` before forwarding a queued failure."""

    entered = threading.Event()
    stop_event = threading.Event()
    graph = _graph(
        [_FailsAfterHostStop("late", entered)],
        [(START_ID, "late"), ("late", END_ID)],
    )
    engine = FlowEngine(graph, num_workers=1, stop_event=stop_event)
    event_q = _FreezeGetAfterNodeEntered(entered)
    engine.event_q = event_q
    events = []
    errors = []

    def drive() -> None:
        try:
            events.extend(engine.run())
        except BaseException as exc:
            errors.append(exc)

    driver = threading.Thread(target=drive, name="test-late-failure-pooled")
    driver.start()
    assert entered.wait(timeout=1.0)
    assert event_q.blocked.wait(timeout=1.0)
    stop_event.set()
    driver.join(timeout=1.0)

    assert not driver.is_alive()
    assert len(errors) == 1 and isinstance(errors[0], _HostStop)
    assert not any(isinstance(event, (NodeFailed, RunFailed)) for event in events)
    assert not any(thread.name.startswith("ac-worker-") for thread in threading.enumerate())


class _ObserverBlockingNode(Node):
    """Class-owned synchronization survives the run boundary's graph deepcopy."""

    kind = NodeKind.CODE
    needs_stop_event = True
    entered = threading.Event()
    saw_stop = threading.Event()

    def _run_impl(self, inputs, *, stop_event):
        type(self).entered.set()
        stop_event.wait()
        if stop_event.is_set():
            type(self).saw_stop.set()
        raise _HostStop


class _ObserverError(RuntimeError):
    """Identity-bearing observer error used to prove first-cause preservation."""


def test_observer_error_sets_stop_closes_pooled_iterator_and_preserves_identity():
    from agent_composer.run import _run_flow_controlled

    _ObserverBlockingNode.entered.clear()
    _ObserverBlockingNode.saw_stop.clear()
    graph = _graph(
        [_ObserverBlockingNode("blocked")],
        [(START_ID, "blocked"), ("blocked", END_ID)],
    )
    loaded = SimpleNamespace(input=[], compiled=graph)
    stop_event = threading.Event()
    original = _ObserverError("observer failed")

    def observer(event):
        if isinstance(event, NodeStarted) and event.node_id == "blocked":
            assert _ObserverBlockingNode.entered.wait(timeout=1.0)
            raise original

    with pytest.raises(_ObserverError) as caught:
        _run_flow_controlled(
            loaded,
            {},
            on_event=observer,
            num_workers=1,
            stop_event=stop_event,
        )

    assert caught.value is original
    assert stop_event.is_set()
    assert _ObserverBlockingNode.saw_stop.is_set()
    assert not any(thread.name.startswith("ac-worker-") for thread in threading.enumerate())


def test_resume_observer_error_sets_stop_before_closing_pooled_iterator() -> None:
    """The public resume observer path wakes blockers before its full worker join."""

    from agent_composer.run import resume_flow

    _ObserverBlockingNode.entered.clear()
    _ObserverBlockingNode.saw_stop.clear()
    graph = _graph(
        [PauseNode("pause"), _ObserverBlockingNode("blocked")],
        [
            (START_ID, "pause"),
            ("pause", "blocked"),
            ("blocked", END_ID),
        ],
    )
    engine = FlowEngine(graph, num_workers=1)
    assert isinstance(list(engine.run())[-1], RunPaused)
    loaded = SimpleNamespace(input=[], compiled=graph)
    original = _ObserverError("resume observer failed")
    observer_called = threading.Event()
    errors = []

    def observer(event):
        if isinstance(event, NodeStarted) and event.node_id == "blocked":
            observer_called.set()
            raise original

    def resume() -> None:
        try:
            resume_flow(
                loaded,
                engine=engine,
                commands=[DeliverAnswerCommand(node_id="pause", value="continue")],
                on_event=observer,
            )
        except BaseException as exc:
            errors.append(exc)

    driver = threading.Thread(target=resume, name="test-resume-observer")
    driver.start()
    assert observer_called.wait(timeout=1.0)
    driver.join(timeout=0.2)
    completed_without_rescue = not driver.is_alive()
    if not completed_without_rescue:
        engine.stop_event.set()
        driver.join(timeout=1.0)

    assert completed_without_rescue
    assert errors == [original]
    assert engine.stop_event.is_set()
    assert _ObserverBlockingNode.saw_stop.is_set()
    assert not any(thread.name.startswith("ac-worker-") for thread in threading.enumerate())


def test_public_run_flow_signature_has_no_stop_event():
    signature = inspect.signature(run_flow)

    assert list(signature.parameters) == [
        "loaded",
        "inputs",
        "run_id",
        "on_event",
        "llm_config",
        "num_workers",
    ]
    assert "stop_event" not in signature.parameters


def test_restore_preserves_exact_host_stop_event():
    graph = _graph([], [])
    checkpoint = FlowEngine(graph).snapshot()
    stop_event = threading.Event()

    restored = FlowEngine.restore(graph, checkpoint, stop_event=stop_event)

    assert restored.stop_event is stop_event
