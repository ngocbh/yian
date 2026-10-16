"""Node steps through the engine: the ``caps['steps']`` capability, serial and pooled."""

from __future__ import annotations

import queue
import threading
from copy import deepcopy
from types import SimpleNamespace
from typing import Any, Callable

import pytest

from yian.compile.expand import build_grow
from yian.compile.model import END_ID, START_ID
from yian.events import NodeStarted, NodeSucceeded, RunFailed, RunSucceeded
from yian.nodes.base import Node, NodeKind, Output, _HostStop
from yian.nodes.contracts import NodeSteps
from yian.runtime import engine as engine_module
from yian.runtime.engine import FlowEngine
from yian.runtime.eval_node import eval_node
from yian.runtime.log import (
    NodeFailedRecord,
    NodeStartedRecord,
    NodeStepRecord,
    NodeSucceededRecord,
)
from yian.runtime.state import (
    RunState,
    _replay_records,
    commit_prepared,
    prepare_run_resumed,
)
from yian.typesys.pool import VariablePool
from tests.engine._graph_builder import _engine, _graph


_TIMEOUT = 10.0


class _StepNode(Node):
    """Test node that opts into node steps and runs a scripted body."""

    kind = NodeKind.CODE
    needs_steps = True

    def __init__(self, node_id: str, body: Callable[[dict[str, Any]], Any]) -> None:
        """Keep the body that receives the node's capabilities."""
        super().__init__(node_id)
        self.body = body

    def forward(self, inputs: dict[str, Any], **caps: Any):
        """Run the scripted body over this evaluation's capabilities."""
        return self.body(caps)


class _PlainNode(_StepNode):
    """The same scripted node without the steps trait."""

    needs_steps = False


class _Crash(BaseException):
    """Stands in for a process crash in the middle of a node."""


def _single(node: Node, *, num_workers: int, **kwargs) -> FlowEngine:
    """Return an engine for START -> node -> END."""
    flow = _graph([node], [(START_ID, node.id), (node.id, END_ID)])
    return _engine(flow, num_workers=num_workers, **kwargs)


def _pair(first: Node, second: Node, *, num_workers: int, **kwargs) -> FlowEngine:
    """Return an engine running two nodes side by side after START."""
    flow = _graph(
        [first, second],
        [(START_ID, first.id), (START_ID, second.id), (first.id, END_ID), (second.id, END_ID)],
    )
    return _engine(flow, num_workers=num_workers, **kwargs)


def _records(engine: FlowEngine, node_id: str) -> list[Any]:
    """Return the log records about ``node_id``, in log order."""
    return [
        envelope.record for envelope in engine.log
        if getattr(envelope.record, "node_id", None) == node_id
    ]


def _run_bounded(engine: FlowEngine) -> tuple[list[Any], BaseException | None]:
    """Drive ``engine`` on a helper thread and fail the test if it hangs."""
    events: list[Any] = []
    error: list[BaseException] = []

    def drive() -> None:
        try:
            events.extend(engine.run())
        except BaseException as exc:
            error.append(exc)

    thread = threading.Thread(target=drive, daemon=True)
    thread.start()
    thread.join(_TIMEOUT)
    assert not thread.is_alive(), "engine did not stop"
    return events, (error[0] if error else None)


@pytest.mark.parametrize("num_workers", [0, 2])
def test_steps_are_durable_before_record_returns(num_workers):
    holder: dict[str, FlowEngine] = {}
    seen: list[Any] = []

    def body(caps):
        steps = caps["steps"]
        seen.append((type(steps), steps.starts, steps.prior))
        for n in (1, 2):
            steps.record({"event": "started", "attempt": n})
            last = [
                envelope.record for envelope in holder["engine"].log
                if isinstance(envelope.record, NodeStepRecord)
            ][-1]
            seen.append(last.step)
        return Output("done")

    engine = _single(_StepNode("work", body), num_workers=num_workers)
    holder["engine"] = engine
    events = list(engine.run())

    assert seen == [
        (NodeSteps, 1, ()),
        {"attempt": 1, "event": "started"},
        {"attempt": 2, "event": "started"},
    ]
    assert _records(engine, "work") == [
        NodeStartedRecord("work"),
        NodeStepRecord("work", {"event": "started", "attempt": 1}),
        NodeStepRecord("work", {"event": "started", "attempt": 2}),
        _records(engine, "work")[-1],
    ]
    assert isinstance(_records(engine, "work")[-1], NodeSucceededRecord)
    assert isinstance(engine.state.result, RunSucceeded)
    assert "work" not in engine.state.node_steps
    # A step publishes no observer: the node's observers are its start and its outcome.
    work_events = [event for event in events if getattr(event, "node_id", None) == "work"]
    assert [type(event) for event in work_events] == [NodeStarted, NodeSucceeded]


@pytest.mark.parametrize("num_workers", [0, 2])
def test_a_node_without_the_trait_receives_no_steps(num_workers):
    seen: list[bool] = []

    def body(caps):
        seen.append("steps" in caps)
        return Output("done")

    engine = _single(_PlainNode("work", body), num_workers=num_workers)
    list(engine.run())

    assert seen == [False]
    assert isinstance(engine.state.result, RunSucceeded)


def test_eval_node_passes_steps_only_to_opted_in_nodes_and_checks_the_type():
    steps = NodeSteps(starts=1, prior=(), record=lambda step: None)
    seen: list[Any] = []

    def body(caps):
        seen.append(caps.get("steps"))
        return Output("done")

    flow = SimpleNamespace(wiring={})
    pool = VariablePool(start_id=START_ID)
    eval_node(_StepNode("a", body), flow, pool, steps=steps)
    eval_node(_PlainNode("b", body), flow, pool, steps=steps)
    eval_node(_StepNode("c", body), flow, pool)

    assert seen == [steps, None, None]
    with pytest.raises(TypeError, match="steps"):
        eval_node(_StepNode("d", body), flow, pool, steps={"starts": 1})


def test_node_steps_checks_its_fields():
    record = lambda step: None  # noqa: E731

    with pytest.raises(TypeError):
        NodeSteps(starts=True, prior=(), record=record)
    with pytest.raises(ValueError):
        NodeSteps(starts=0, prior=(), record=record)
    with pytest.raises(TypeError):
        NodeSteps(starts=1, prior=[{}], record=record)
    with pytest.raises(TypeError):
        NodeSteps(starts=1, prior=("x",), record=record)
    with pytest.raises(TypeError):
        NodeSteps(starts=1, prior=(), record=None)


@pytest.mark.parametrize("num_workers", [0, 2])
def test_resume_sees_the_second_start_and_the_prior_steps(num_workers):
    seen: list[Any] = []

    def body(caps):
        steps = caps["steps"]
        seen.append((steps.starts, steps.prior))
        if steps.starts == 1:
            steps.record({"event": "failed", "attempt": 1, "nested": {"retry_in": 1}})
            raise _Crash()
        return Output("done")

    first = _single(_StepNode("work", body), num_workers=num_workers)
    with pytest.raises(_Crash):
        list(first.run())

    flow = first.state.flow
    state = _replay_records(RunState(deepcopy(flow)), tuple(first.log), grow_builder=build_grow)
    assert state.in_flight == {"work": None}
    commit_prepared(first.log, state, prepare_run_resumed(state, num_workers=num_workers))
    resumed = FlowEngine(state, first.log, grow_builder=build_grow, num_workers=num_workers)
    list(resumed.run())

    assert seen == [
        (1, ()),
        (2, ({"attempt": 1, "event": "failed", "nested": {"retry_in": 1}},)),
    ]
    assert isinstance(state.result, RunSucceeded)
    steps = [record.step for record in _records(resumed, "work") if isinstance(record, NodeStepRecord)]
    assert steps == [{"attempt": 1, "event": "failed", "nested": {"retry_in": 1}}]


def test_resume_hands_detached_prior_steps():
    kept: list[Any] = []

    def body(caps):
        steps = caps["steps"]
        if steps.starts == 1:
            steps.record({"event": "failed", "nested": {"retry_in": 1}})
            raise _Crash()
        # The node gets copies: changing them must not reach the engine's state.
        steps.prior[0]["nested"]["retry_in"] = 99
        kept.append(deepcopy(state.node_steps["work"]))
        return Output("done")

    first = _single(_StepNode("work", body), num_workers=0)
    with pytest.raises(_Crash):
        list(first.run())
    state = first.state
    commit_prepared(first.log, state, prepare_run_resumed(state, num_workers=0))
    list(FlowEngine(state, first.log, grow_builder=build_grow, num_workers=0).run())

    assert kept == [({"event": "failed", "nested": {"retry_in": 1}},)]


def test_pooled_workers_never_read_the_step_authority(monkeypatch):
    def body(caps):
        caps["steps"].record({"event": "started", "attempt": 2})
        return Output("done")

    engine = _single(_StepNode("work", body), num_workers=2)
    caller = threading.get_ident()
    reads: list[str] = []
    get_attribute = RunState.__getattribute__

    def guarded_get_attribute(state, key):
        if key in ("node_starts", "node_steps"):
            assert threading.get_ident() == caller, f"worker read RunState.{key}"
            reads.append(key)
        return get_attribute(state, key)

    monkeypatch.setattr(RunState, "__getattribute__", guarded_get_attribute)
    list(engine.run())

    assert {"node_starts", "node_steps"} <= set(reads)
    assert isinstance(engine.state.result, RunSucceeded)


def _signal_queue(monkeypatch, on_put: Callable[[Any], None]) -> None:
    """Make the pooled dispatcher queue call ``on_put`` before each message is queued."""

    class _SignalQueue(queue.Queue):
        def put(self, item, *args, **kwargs):
            on_put(item)
            super().put(item, *args, **kwargs)

    monkeypatch.setattr(
        engine_module, "queue", SimpleNamespace(Queue=_SignalQueue, Empty=queue.Empty),
    )


def test_stop_releases_a_pooled_worker_waiting_for_its_step(monkeypatch):
    stop = threading.Event()
    raised: list[BaseException] = []

    def on_put(item):
        # Stop before the dispatcher can see the step, so the worker is left waiting.
        if isinstance(item, engine_module._RecordedStep):
            stop.set()

    _signal_queue(monkeypatch, on_put)

    def body(caps):
        try:
            caps["steps"].record({"event": "started", "attempt": 2})
        except BaseException as exc:
            raised.append(exc)
            raise
        return Output("done")

    engine = _single(_StepNode("work", body), num_workers=2, stop_event=stop)
    _, error = _run_bounded(engine)

    assert isinstance(error, _HostStop)
    assert [type(exc) for exc in raised] == [_HostStop]
    assert not any(isinstance(record, NodeStepRecord) for record in _records(engine, "work"))
    assert engine.state.result is None


def test_a_sibling_failure_releases_a_pooled_worker_waiting_for_its_step(monkeypatch):
    sibling_done = threading.Event()
    raised: list[BaseException] = []

    def on_put(item):
        if isinstance(item, engine_module._RecordedCompletion) and item.node_id == "boom":
            sibling_done.set()

    _signal_queue(monkeypatch, on_put)

    def waiting(caps):
        # Queue the step behind the sibling's failed completion, so the dispatcher
        # fails the run before it reaches the step.
        assert sibling_done.wait(_TIMEOUT)
        try:
            caps["steps"].record({"event": "started", "attempt": 2})
        except BaseException as exc:
            raised.append(exc)
            raise
        return Output("done")

    def boom(caps):
        raise RuntimeError("boom")

    engine = _pair(
        _StepNode("work", waiting), _PlainNode("boom", boom), num_workers=2,
    )
    _, error = _run_bounded(engine)

    assert error is None
    assert isinstance(engine.state.result, RunFailed)
    assert engine.state.result.failure.node_id == "boom"
    assert [type(exc) for exc in raised] == [_HostStop]
    assert not any(isinstance(record, NodeStepRecord) for record in _records(engine, "work"))


@pytest.mark.parametrize("num_workers", [0, 2])
def test_a_recorder_log_fault_escapes_as_a_host_error(num_workers, monkeypatch):
    fault = OSError("disk full")
    after: list[str] = []

    def body(caps):
        caps["steps"].record({"event": "started", "attempt": 2})
        after.append("continued")
        return Output("done")

    engine = _single(_StepNode("work", body), num_workers=num_workers)
    append = engine.log.append

    def failing_append(record):
        if isinstance(record, NodeStepRecord):
            raise fault
        return append(record)

    monkeypatch.setattr(engine.log, "append", failing_append)
    _, error = _run_bounded(engine)

    assert error is fault
    assert after == []
    assert not any(isinstance(envelope.record, NodeFailedRecord) for envelope in engine.log)
    assert engine.state.result is None


@pytest.mark.parametrize("num_workers", [0, 2])
@pytest.mark.parametrize(
    ("step", "error_type"),
    [({"when": object()}, ValueError), ({"retry_in": float("nan")}, ValueError), (["x"], TypeError)],
    ids=["object", "nan", "list"],
)
def test_an_invalid_step_is_a_host_error(num_workers, step, error_type):
    def body(caps):
        caps["steps"].record(step)
        return Output("done")

    engine = _single(_StepNode("work", body), num_workers=num_workers)
    _, error = _run_bounded(engine)

    assert isinstance(error, error_type)
    assert not any(isinstance(envelope.record, NodeFailedRecord) for envelope in engine.log)
    assert not any(isinstance(envelope.record, NodeStepRecord) for envelope in engine.log)
    assert engine.state.result is None
