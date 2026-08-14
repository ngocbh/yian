"""Bounded dispatch over the append-only run-state reducer.

These tests exercise the private recorded dispatcher before the atomic public
``FlowEngine`` cutover.  They deliberately use real reducer/log transitions so
durability order, unfinished starts, observer order, and drain behavior cannot be
faked by an event-only adapter.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any

import pytest

import agent_composer.runtime.engine as engine_module
from agent_composer.compile.model import CompiledFlow, Edge, END_ID, Flow, START_ID
from agent_composer.events import (
    NodeExpanded,
    NodeFailed,
    NodeRouted,
    NodeStarted,
    NodeSucceeded,
    RunFailed,
    RunSucceeded,
    StreamChunk,
)
from agent_composer.nodes.base import Grow, Node, NodeKind, Output, Route, _HostStop
from agent_composer.nodes.end import EndNode
from agent_composer.nodes.start import StartNode
from agent_composer.runtime.engine import _RecordedDispatcher
from agent_composer.runtime.log import (
    MemoryRunLog,
    NodeExpandedRecord,
    NodeFailedRecord,
    NodeStartedRecord,
    NodeSucceededRecord,
)
from agent_composer.runtime.state import (
    RunStartData,
    RunState,
    commit_prepared,
    prepare_run_started,
)
from agent_composer.typesys import Type, ValueKind, VariablePool


_RUN_ID = "recorded-dispatch"
_FINGERPRINT = "d" * 64


class _ActionNode(Node):
    """Run one injected side effect and return its value."""

    kind = NodeKind.CODE

    def __init__(
        self,
        node_id: str,
        action: Callable[[], Any],
        *,
        output_type: Type | None = None,
    ) -> None:
        super().__init__(node_id, output_type=output_type)
        self._action = action

    def _run_impl(self, inputs: dict[str, Any]) -> Output:
        return Output(self._action())


class _RouteNode(Node):
    """Return one fixed route handle through the raw-outcome arm."""

    kind = NodeKind.CASE

    def _run_impl(self, inputs: dict[str, Any]) -> Route:
        return Route("chosen")


class _GrowNode(Node):
    """Return one seed-first expansion through the raw-outcome arm."""

    kind = NodeKind.CALL
    is_spawner = True
    grow_depth_delta = 1

    def _run_impl(self, inputs: dict[str, Any]) -> Grow:
        return Grow(seed={"canonical": (1, 2)})


class _StopAwareNode(Node):
    """Block until cooperative drain wakes this worker."""

    kind = NodeKind.CODE
    needs_stop_event = True

    def __init__(
        self,
        node_id: str,
        entered: threading.Event,
        released: threading.Event,
    ) -> None:
        super().__init__(node_id)
        self._entered = entered
        self._released = released

    def _run_impl(self, inputs: dict[str, Any], *, stop_event: threading.Event) -> Output:
        self._entered.set()
        stop_event.wait(2)
        self._released.set()
        raise _HostStop(primary=False)


class _PrimaryHostStopNode(Node):
    """Raise one caller-owned host exception object unchanged."""

    kind = NodeKind.CODE

    def __init__(
        self,
        node_id: str,
        exception: _HostStop,
        *,
        wait_for: threading.Event | None = None,
    ) -> None:
        super().__init__(node_id)
        self._exception = exception
        self._wait_for = wait_for

    def _run_impl(self, inputs: dict[str, Any]) -> Output:
        if self._wait_for is not None:
            assert self._wait_for.wait(2)
        raise self._exception


class _LateStreamNode(Node):
    """Emit one live chunk and one chunk after failure-driven stop begins."""

    kind = NodeKind.CODE
    needs_stop_event = True

    def __init__(self, node_id: str, failure_may_run: threading.Event) -> None:
        super().__init__(node_id)
        self._failure_may_run = failure_may_run

    def _run_impl(self, inputs: dict[str, Any], *, stop_event: threading.Event):
        yield StreamChunk(self.id, "output", "early")
        self._failure_may_run.set()
        stop_event.wait(2)
        yield StreamChunk(self.id, "output", "late")
        return Output("discarded")


def _flow(nodes: list[Node]) -> CompiledFlow:
    """Return START fan-out into ``nodes`` followed by one ordering-joined END."""
    all_nodes = {
        START_ID: StartNode(START_ID),
        **{node.id: node for node in nodes},
        END_ID: EndNode.record(END_ID, output_names=[]),
    }
    edges = [
        Edge(
            id=f"start-{node.id}",
            from_=START_ID,
            to=node.id,
            optional=True,
            ordering=True,
        )
        for node in nodes
    ]
    edges.extend(
        Edge(
            id=f"{node.id}-end",
            from_=node.id,
            to=END_ID,
            optional=True,
            ordering=True,
        )
        for node in nodes
    )
    return CompiledFlow.from_parts(nodes=all_nodes, edges=edges)


def _growth_fragment(spawner_id: str) -> Flow:
    """Return one valid dynamic boundary pair committing under ``spawner_id``."""
    start_id = f"{spawner_id}/child/start"
    end_id = f"{spawner_id}/child/end"
    start = StartNode(start_id, input_decls=[])
    end = EndNode.record(end_id, output_names=[])
    end.commit_as = spawner_id
    return Flow(
        nodes={start_id: start, end_id: end},
        edges=[
            Edge(
                id=f"{start_id}-end",
                from_=start_id,
                to=end_id,
                ordering=True,
            )
        ],
        wiring={start_id: {}, end_id: {}},
        start_id=start_id,
        end_id=end_id,
    )


def _initialized(
    flow: CompiledFlow,
    *,
    num_workers: int,
    log: MemoryRunLog | None = None,
) -> tuple[RunState, MemoryRunLog]:
    """Apply one authoritative RunStarted and return its ready START frontier."""
    state = RunState(flow)
    pool = VariablePool(start_id=flow.start_id)
    pool.set(flow.start_id, {})
    pool.add_system("run_id", _RUN_ID)
    log = MemoryRunLog(_RUN_ID) if log is None else log
    prepared = prepare_run_started(
        state,
        RunStartData(
            run_id=_RUN_ID,
            flow_fingerprint=_FINGERPRINT,
            initial_pool=pool,
            start_wiring_override={},
            effective_llm_by_static_path={},
            num_workers=num_workers,
        ),
    )
    commit_prepared(log, state, prepared)
    return state, log


def _dispatcher(
    state: RunState,
    log: MemoryRunLog,
    *,
    num_workers: int,
    **kwargs: Any,
) -> _RecordedDispatcher:
    """Construct the private runtime with a growth seam unused by static tests."""
    return _RecordedDispatcher(
        state,
        log,
        grow_builder=lambda _spawner, _seed: None,
        num_workers=num_workers,
        **kwargs,
    )


class _GuardedLog(MemoryRunLog):
    """Inject append failures or inspect execution at exact durable records."""

    def __init__(self, run_id: str) -> None:
        super().__init__(run_id)
        self.guard: Callable[[Any], None] | None = None
        self.close_calls = 0

    def append(self, record):
        if self.guard is not None:
            self.guard(record)
        return super().append(record)

    def close(self) -> None:
        self.close_calls += 1
        super().close()


def test_started_record_is_durable_before_node_code_runs():
    executed = []
    log = _GuardedLog(_RUN_ID)
    state, log = _initialized(
        _flow([_ActionNode("work", lambda: executed.append("work") or "done")]),
        num_workers=0,
        log=log,
    )

    def guard(record):
        if isinstance(record, NodeStartedRecord) and record.node_id == "work":
            assert executed == []

    log.guard = guard
    list(_dispatcher(state, log, num_workers=0).run())

    records = [envelope.record for envelope in log]
    start_index = records.index(NodeStartedRecord(node_id="work"))
    success_index = next(
        index
        for index, record in enumerate(records)
        if isinstance(record, NodeSucceededRecord) and record.node_id == "work"
    )
    assert executed == ["work"]
    assert start_index < success_index


def test_start_append_failure_prevents_node_execution_and_propagates_exactly():
    executed = []
    expected = OSError("start fsync failed")
    log = _GuardedLog(_RUN_ID)
    state, log = _initialized(
        _flow([_ActionNode("work", lambda: executed.append("work") or "done")]),
        num_workers=0,
        log=log,
    )

    def reject(record):
        if isinstance(record, NodeStartedRecord) and record.node_id == "work":
            raise expected

    log.guard = reject
    with pytest.raises(OSError) as caught:
        list(_dispatcher(state, log, num_workers=0).run())

    assert caught.value is expected
    assert executed == []
    assert "work" not in state.in_flight


def test_outcome_append_failure_leaves_one_unfinished_durable_start():
    executed = []
    expected = OSError("outcome fsync failed")
    log = _GuardedLog(_RUN_ID)
    state, log = _initialized(
        _flow([_ActionNode("work", lambda: executed.append("work") or "done")]),
        num_workers=0,
        log=log,
    )

    def reject(record):
        if isinstance(record, NodeSucceededRecord) and record.node_id == "work":
            raise expected

    log.guard = reject
    with pytest.raises(OSError) as caught:
        list(_dispatcher(state, log, num_workers=0).run())

    assert caught.value is expected
    assert executed == ["work"]
    assert tuple(state.in_flight) == ("work",)
    assert any(
        isinstance(envelope.record, NodeStartedRecord)
        and envelope.record.node_id == "work"
        for envelope in log
    )
    assert not any(
        isinstance(envelope.record, NodeSucceededRecord)
        and envelope.record.node_id == "work"
        for envelope in log
    )


def test_reducer_programming_failure_is_not_rewritten_as_node_failure(monkeypatch):
    expected = RuntimeError("reducer invariant broke")
    state, log = _initialized(
        _flow([_ActionNode("work", lambda: "done")]),
        num_workers=0,
    )
    real_commit = engine_module.commit_prepared

    def broken_commit(target_log, target_state, prepared):
        if (
            isinstance(prepared.record, NodeSucceededRecord)
            and prepared.record.node_id == "work"
        ):
            raise expected
        return real_commit(target_log, target_state, prepared)

    monkeypatch.setattr(engine_module, "commit_prepared", broken_commit)

    with pytest.raises(RuntimeError) as caught:
        list(_dispatcher(state, log, num_workers=0).run())

    assert caught.value is expected
    assert tuple(state.in_flight) == ("work",)
    assert not any(isinstance(envelope.record, NodeFailedRecord) for envelope in log)


def test_success_preparation_runtime_error_remains_exact_host_failure(monkeypatch):
    expected = RuntimeError("success reducer programming fault")
    state, log = _initialized(
        _flow([_ActionNode("work", lambda: "done")]),
        num_workers=0,
    )
    real_prepare = engine_module.prepare_node_succeeded

    def broken_prepare(target_state, node_id, output, *, commit_as=None):
        if node_id == "work":
            raise expected
        return real_prepare(
            target_state,
            node_id,
            output,
            commit_as=commit_as,
        )

    monkeypatch.setattr(engine_module, "prepare_node_succeeded", broken_prepare)

    with pytest.raises(RuntimeError) as caught:
        list(_dispatcher(state, log, num_workers=0).run())

    assert caught.value is expected
    assert tuple(state.in_flight) == ("work",)
    assert not any(isinstance(envelope.record, NodeFailedRecord) for envelope in log)


def test_serial_yields_and_callbacks_follow_post_durable_observer_order():
    state, log = _initialized(
        _flow([_ActionNode("work", lambda: "done")]),
        num_workers=0,
    )
    callbacks = []

    yielded = list(
        _dispatcher(state, log, num_workers=0, on_event=callbacks.append).run()
    )

    assert callbacks == yielded
    assert [type(event) for event in yielded] == [
        NodeStarted,
        NodeSucceeded,
        NodeStarted,
        NodeSucceeded,
        NodeStarted,
        NodeSucceeded,
        RunSucceeded,
    ]
    assert [event.node_id for event in yielded if hasattr(event, "node_id")] == [
        START_ID,
        START_ID,
        "work",
        "work",
        END_ID,
        END_ID,
    ]


def test_invalid_raw_output_preparation_becomes_durable_node_failure():
    node = _ActionNode(
        "work",
        lambda: "not-an-int",
        output_type=Type.scalar(ValueKind.INTEGER),
    )
    state, log = _initialized(_flow([node]), num_workers=0)

    events = list(_dispatcher(state, log, num_workers=0).run())

    assert isinstance(events[-2], NodeFailed)
    assert isinstance(events[-1], RunFailed)
    assert events[-1].error_type == "TypeCheckError"
    assert isinstance(state.result, RunFailed)
    assert isinstance(tuple(log)[-1].record, NodeFailedRecord)


def test_route_raw_outcome_is_prepared_committed_and_reduced():
    ran = []
    router = _RouteNode("route")
    chosen = _ActionNode("chosen", lambda: ran.append("chosen") or "yes")
    skipped = _ActionNode("skipped", lambda: ran.append("skipped") or "no")
    nodes = {
        START_ID: StartNode(START_ID),
        router.id: router,
        chosen.id: chosen,
        skipped.id: skipped,
        END_ID: EndNode.record(END_ID, output_names=[]),
    }
    flow = CompiledFlow.from_parts(
        nodes=nodes,
        edges=[
            Edge("start-route", START_ID, router.id, optional=True, ordering=True),
            Edge("route-chosen", router.id, chosen.id, source_handle="chosen"),
            Edge("route-skipped", router.id, skipped.id, source_handle="other"),
            Edge("chosen-end", chosen.id, END_ID, optional=True, ordering=True),
            Edge("skipped-end", skipped.id, END_ID, optional=True, ordering=True),
        ],
    )
    state, log = _initialized(flow, num_workers=0)

    events = list(_dispatcher(state, log, num_workers=0).run())

    assert ran == ["chosen"]
    assert any(isinstance(event, NodeRouted) for event in events)
    assert isinstance(state.result, RunSucceeded)


def test_grow_raw_outcome_uses_injected_builder_and_flat_record():
    state, log = _initialized(_flow([_GrowNode("spawn")]), num_workers=0)
    dispatcher = _RecordedDispatcher(
        state,
        log,
        grow_builder=lambda spawner, _seed: _growth_fragment(spawner.id),
        num_workers=0,
    )

    events = list(dispatcher.run())

    assert any(isinstance(event, NodeExpanded) for event in events)
    expanded = [
        envelope.record
        for envelope in log
        if isinstance(envelope.record, NodeExpandedRecord)
    ]
    assert len(expanded) == 1
    assert expanded[0].spawner_id == "spawn"
    assert expanded[0].seed == {"canonical": [1, 2]}
    assert isinstance(state.result, RunSucceeded)


def test_injected_grow_builder_runtime_error_is_a_durable_node_failure():
    expected = RuntimeError("builder rejected seed")
    state, log = _initialized(_flow([_GrowNode("spawn")]), num_workers=0)

    def reject_builder(_spawner, _seed):
        raise expected

    events = list(
        _RecordedDispatcher(
            state,
            log,
            grow_builder=reject_builder,
            num_workers=0,
        ).run()
    )

    assert isinstance(events[-2], NodeFailed)
    assert events[-2].node_id == "spawn"
    assert (events[-2].error, events[-2].error_type) == (
        "builder rejected seed",
        "RuntimeError",
    )
    assert isinstance(events[-1], RunFailed)
    assert isinstance(tuple(log)[-1].record, NodeFailedRecord)


def test_positive_worker_count_never_exceeds_the_submission_bound():
    lock = threading.Lock()
    active = 0
    maximum = 0

    def work():
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
        time.sleep(0.04)
        with lock:
            active -= 1
        return "done"

    state, log = _initialized(
        _flow([_ActionNode(f"work-{index}", work) for index in range(5)]),
        num_workers=2,
    )

    list(_dispatcher(state, log, num_workers=2).run())

    assert maximum == 2


def test_parallel_outcomes_commit_in_completion_message_order():
    release_slow = threading.Event()

    def slow():
        assert release_slow.wait(2)
        return "slow"

    def fast():
        release_slow.set()
        return "fast"

    state, log = _initialized(
        _flow([_ActionNode("slow", slow), _ActionNode("fast", fast)]),
        num_workers=2,
    )

    list(_dispatcher(state, log, num_workers=2).run())

    completed = [
        envelope.record.node_id
        for envelope in log
        if isinstance(envelope.record, NodeSucceededRecord)
        and envelope.record.node_id in {"slow", "fast"}
    ]
    assert completed == ["fast", "slow"]


def test_submit_failure_is_exact_and_leaves_the_started_node_recoverable(monkeypatch):
    expected = RuntimeError("submit rejected")
    shutdown = []

    class RejectingExecutor:
        def __init__(self, **_kwargs):
            pass

        def submit(self, *_args, **_kwargs):
            raise expected

        def shutdown(self, *, wait, cancel_futures):
            shutdown.append((wait, cancel_futures))

    monkeypatch.setattr(engine_module, "ThreadPoolExecutor", RejectingExecutor)
    state, log = _initialized(
        _flow([_ActionNode("never", lambda: "unexpected")]),
        num_workers=1,
    )

    with pytest.raises(RuntimeError) as caught:
        list(_dispatcher(state, log, num_workers=1).run())

    assert caught.value is expected
    assert tuple(state.in_flight) == (START_ID,)
    assert shutdown == [(True, False)]


def test_event_callback_failure_signals_drains_and_reraises_same_exception():
    entered = threading.Event()
    released = threading.Event()
    expected = RuntimeError("observer failed")

    def trigger():
        assert entered.wait(2)
        return "triggered"

    state, log = _initialized(
        _flow(
            [
                _ActionNode("trigger", trigger),
                _StopAwareNode("blocked", entered, released),
            ]
        ),
        num_workers=2,
    )

    def reject(event):
        if isinstance(event, NodeSucceeded) and event.node_id == "trigger":
            raise expected

    with pytest.raises(RuntimeError) as caught:
        list(_dispatcher(state, log, num_workers=2, on_event=reject).run())

    assert caught.value is expected
    assert released.is_set()
    assert "blocked" in state.in_flight
    assert state.result is None


def test_cleanup_failure_never_replaces_callback_error_and_drain_is_attempted(
    monkeypatch,
):
    entered = threading.Event()
    released = threading.Event()
    expected = RuntimeError("observer failed first")
    cleanup = RuntimeError("executor shutdown failed afterward")
    real_executor = engine_module.ThreadPoolExecutor

    class HostileShutdownExecutor:
        """Join real workers, then raise to probe first-cause preservation."""

        def __init__(self, **kwargs):
            self._delegate = real_executor(**kwargs)

        def submit(self, *args, **kwargs):
            return self._delegate.submit(*args, **kwargs)

        def shutdown(self, *, wait, cancel_futures):
            self._delegate.shutdown(wait=wait, cancel_futures=cancel_futures)
            raise cleanup

    monkeypatch.setattr(
        engine_module,
        "ThreadPoolExecutor",
        HostileShutdownExecutor,
    )

    def trigger():
        assert entered.wait(2)
        return "triggered"

    state, log = _initialized(
        _flow(
            [
                _ActionNode("trigger", trigger),
                _StopAwareNode("blocked", entered, released),
            ]
        ),
        num_workers=2,
    )

    def reject(event):
        if isinstance(event, NodeSucceeded) and event.node_id == "trigger":
            raise expected

    with pytest.raises(RuntimeError) as caught:
        list(_dispatcher(state, log, num_workers=2, on_event=reject).run())

    assert caught.value is expected
    assert released.is_set()
    assert any(
        "executor shutdown failed afterward" in note
        for note in getattr(expected, "__notes__", ())
    )


def test_started_observer_failure_prevents_submission_of_that_node():
    executed = []
    expected = RuntimeError("started observer failed")
    state, log = _initialized(
        _flow([_ActionNode("work", lambda: executed.append("work") or "done")]),
        num_workers=1,
    )

    def reject(event):
        if isinstance(event, NodeStarted) and event.node_id == "work":
            raise expected

    with pytest.raises(RuntimeError) as caught:
        list(_dispatcher(state, log, num_workers=1, on_event=reject).run())

    assert caught.value is expected
    assert executed == []
    assert tuple(state.in_flight) == ("work",)


def test_primary_host_interruption_is_preserved_after_full_sibling_drain():
    entered = threading.Event()
    released = threading.Event()
    primary = _HostStop("terminal EOF", primary=True)
    state, log = _initialized(
        _flow(
            [
                _StopAwareNode("blocked", entered, released),
                _PrimaryHostStopNode("interrupt", primary, wait_for=entered),
            ]
        ),
        num_workers=2,
    )

    with pytest.raises(_HostStop) as caught:
        list(_dispatcher(state, log, num_workers=2).run())

    assert caught.value is primary
    assert entered.is_set()
    assert released.is_set()
    assert state.result is None


def test_node_failure_drains_sibling_and_discards_late_stream_chunks():
    failure_may_run = threading.Event()
    chunks = []

    def fail():
        assert failure_may_run.wait(2)
        raise RuntimeError("boom")

    state, log = _initialized(
        _flow(
            [
                _LateStreamNode("stream", failure_may_run),
                _ActionNode("fail", fail),
            ]
        ),
        num_workers=2,
    )

    events = list(
        _dispatcher(state, log, num_workers=2, on_stream=chunks.append).run()
    )

    assert [chunk.chunk for chunk in chunks] == ["early"]
    assert isinstance(events[-2], NodeFailed)
    assert events[-2].node_id == "fail"
    assert isinstance(events[-1], RunFailed)
    assert tuple(state.in_flight) == ("stream",)
    assert not any(
        isinstance(envelope.record, NodeSucceededRecord)
        and envelope.record.node_id == "stream"
        for envelope in log
    )


def test_stream_callback_failure_is_host_error_not_durable_node_failure():
    expected = RuntimeError("stream callback failed")
    failure_may_run = threading.Event()
    state, log = _initialized(
        _flow([_LateStreamNode("stream", failure_may_run)]),
        num_workers=1,
    )

    def reject(_chunk):
        raise expected

    with pytest.raises(RuntimeError) as caught:
        list(_dispatcher(state, log, num_workers=1, on_stream=reject).run())

    assert caught.value is expected
    assert tuple(state.in_flight) == ("stream",)
    assert not any(isinstance(envelope.record, NodeFailedRecord) for envelope in log)


def test_authority_mismatch_is_rejected_before_any_node_executes():
    executed = []
    state, _ = _initialized(
        _flow([_ActionNode("work", lambda: executed.append("work"))]),
        num_workers=0,
    )
    other_log = MemoryRunLog("other-run")

    with pytest.raises(ValueError, match="run_id mismatch"):
        list(_dispatcher(state, other_log, num_workers=0).run())

    assert executed == []
    assert other_log.next_sequence == 0


def test_dispatcher_borrows_and_never_closes_its_log():
    log = _GuardedLog(_RUN_ID)
    state, log = _initialized(
        _flow([_ActionNode("work", lambda: "done")]),
        num_workers=0,
        log=log,
    )

    list(_dispatcher(state, log, num_workers=0).run())

    assert log.close_calls == 0
    assert isinstance(state.result, RunSucceeded)
    log.close()
    assert log.close_calls == 1
