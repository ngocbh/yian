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
from yian.nodes.base import Node, NodeKind, Output, _CapabilityFault, _HostStop
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
    events, error = _run_bounded(engine)

    assert error is None

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
    _, error = _run_bounded(engine)

    assert error is None
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
    _, error = _run_bounded(first)
    assert isinstance(error, _Crash)

    flow = first.state.flow
    state = _replay_records(RunState(deepcopy(flow)), tuple(first.log), grow_builder=build_grow)
    assert state.in_flight == {"work": None}
    commit_prepared(first.log, state, prepare_run_resumed(state, num_workers=num_workers))
    resumed = FlowEngine(state, first.log, grow_builder=build_grow, num_workers=num_workers)
    _, error = _run_bounded(resumed)

    assert error is None

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
    reads: list[tuple[str, str]] = []
    get_attribute = RunState.__getattribute__

    def guarded_get_attribute(state, key):
        if key in ("node_starts", "node_steps"):
            reads.append((key, threading.current_thread().name))
        return get_attribute(state, key)

    monkeypatch.setattr(RunState, "__getattribute__", guarded_get_attribute)
    _, error = _run_bounded(engine)

    assert error is None
    assert {key for key, _ in reads} == {"node_starts", "node_steps"}
    # Pooled workers run on the engine's "ac-runtime" threads; only the dispatcher reads.
    assert not [read for read in reads if read[1].startswith("ac-runtime")]
    assert isinstance(engine.state.result, RunSucceeded)


def _signal_queue(
    monkeypatch,
    *,
    before: Callable[[Any], None] = lambda item: None,
    after: Callable[[Any], None] = lambda item: None,
) -> None:
    """Make the pooled dispatcher queue call ``before``/``after`` around each put."""

    class _SignalQueue(queue.Queue):
        def put(self, item, *args, **kwargs):
            before(item)
            super().put(item, *args, **kwargs)
            after(item)

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

    _signal_queue(monkeypatch, before=on_put)

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

    _signal_queue(monkeypatch, after=on_put)

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


def test_serial_skips_a_step_once_stop_is_set():
    stop = threading.Event()
    raised: list[BaseException] = []

    def body(caps):
        stop.set()
        try:
            caps["steps"].record({"event": "started", "attempt": 2})
        except BaseException as exc:
            raised.append(exc)
            raise
        return Output("done")

    engine = _single(_StepNode("work", body), num_workers=0, stop_event=stop)
    _, error = _run_bounded(engine)

    assert isinstance(error, _HostStop)
    assert [type(exc) for exc in raised] == [_HostStop]
    assert not any(isinstance(record, NodeStepRecord) for record in _records(engine, "work"))


@pytest.mark.parametrize("num_workers", [0, 2])
def test_a_kept_recorder_is_closed_after_its_evaluation(num_workers):
    kept: list[NodeSteps] = []
    late: list[BaseException] = []

    def keeper(caps):
        kept.append(caps["steps"])
        return Output("kept")

    def later(caps):
        # A later node's evaluation tries the first node's recorder.
        try:
            kept[0].record({"event": "late"})
        except _CapabilityFault as exc:
            late.append(exc)
        return Output("done")

    flow = _graph(
        [_StepNode("keeper", keeper), _PlainNode("later", later)],
        [(START_ID, "keeper"), ("keeper", "later"), ("later", END_ID)],
    )
    engine = _engine(flow, num_workers=num_workers)
    _, error = _run_bounded(engine)

    assert error is None
    assert isinstance(engine.state.result, RunSucceeded)
    assert [str(exc.original) for exc in late] == ["node steps capability is closed"]

    # After the run, a call fails at once, without touching the log.
    size = len(tuple(engine.log))
    after: list[BaseException] = []
    thread = threading.Thread(
        target=lambda: _capture(after, lambda: kept[0].record({"event": "late"})), daemon=True,
    )
    thread.start()
    thread.join(_TIMEOUT)
    assert not thread.is_alive(), "closed recorder blocked"
    assert [type(exc) for exc in after] == [_CapabilityFault]
    assert len(tuple(engine.log)) == size
    assert not any(isinstance(envelope.record, NodeStepRecord) for envelope in engine.log)


def _capture(errors: list[BaseException], call: Callable[[], Any]) -> None:
    """Run ``call`` and keep the exception it raises."""
    try:
        call()
    except BaseException as exc:
        errors.append(exc)


def test_pooled_nodes_interleave_their_own_steps():
    turns = {"a": threading.Event(), "b": threading.Event()}

    def body_for(node_id: str, other: str):
        def body(caps):
            for attempt in (1, 2):
                # Take turns: a records, then b, then a, then b.
                if node_id == "b" or attempt == 2:
                    assert turns[node_id].wait(_TIMEOUT)
                    turns[node_id].clear()
                caps["steps"].record({"node": node_id, "attempt": attempt})
                turns[other].set()
            return Output(node_id)
        return body

    engine = _pair(
        _StepNode("a", body_for("a", "b")), _StepNode("b", body_for("b", "a")), num_workers=2,
    )
    _, error = _run_bounded(engine)

    assert error is None
    assert isinstance(engine.state.result, RunSucceeded)
    steps = [
        (envelope.record.node_id, envelope.record.step)
        for envelope in engine.log if isinstance(envelope.record, NodeStepRecord)
    ]
    assert steps == [
        ("a", {"attempt": 1, "node": "a"}),
        ("b", {"attempt": 1, "node": "b"}),
        ("a", {"attempt": 2, "node": "a"}),
        ("b", {"attempt": 2, "node": "b"}),
    ]
