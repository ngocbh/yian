"""Transactional dispatch and recovery-frontier tests for the run-state reducer."""

from __future__ import annotations

import ast
import inspect
from collections import deque
from copy import deepcopy
from dataclasses import FrozenInstanceError
from pathlib import Path
from textwrap import dedent
from typing import Any

import pytest

import agent_composer.runtime.state as state_module
from agent_composer.compile.model import CompiledFlow, Edge, END_ID, START_ID
from agent_composer.runtime.common import NodeState
from agent_composer.diagnostics import Failure
from agent_composer.events import NodeStarted, RunFailed, RunResumed
from agent_composer.nodes.end import EndNode
from agent_composer.nodes.start import StartNode
from agent_composer.runtime.log import (
    MemoryRunLog,
    NodeExpandedRecord,
    NodeStartedRecord,
    RecordEnvelope,
    RunResumedRecord,
    RunStartedRecord,
    SCHEMA_VERSION,
)
from agent_composer.runtime.state import (
    RunStartData,
    RunState,
    commit_prepared,
    prepare_node_started,
    prepare_node_succeeded,
    prepare_record,
    prepare_run_resumed,
    prepare_run_started,
)
from agent_composer.typesys import VariablePool
from tests.engine._fakes import FuncNode


_FINGERPRINT = "a" * 64
_RUN_ID = "dispatch-run"


def _flow() -> CompiledFlow:
    """Return a graph with several nodes available for isolated frontier tests."""
    nodes = {
        START_ID: StartNode(START_ID),
        "first": FuncNode("first", lambda inputs: inputs),
        "second": FuncNode("second", lambda inputs: inputs),
        "ordinary": FuncNode("ordinary", lambda inputs: inputs),
        END_ID: EndNode.record(END_ID, output_names=[]),
    }
    return CompiledFlow.from_parts(
        nodes=nodes,
        edges=[
            Edge(id="start-first", from_=START_ID, to="first"),
            Edge(id="first-second", from_="first", to="second"),
            Edge(id="second-ordinary", from_="second", to="ordinary"),
            Edge(id="ordinary-end", from_="ordinary", to=END_ID),
        ],
    )


def _initialized(
    *, num_workers: int = 2, flow: CompiledFlow | None = None
) -> tuple[RunState, MemoryRunLog, CompiledFlow]:
    """Return state and log after their shared authoritative RunStarted record."""
    flow = _flow() if flow is None else flow
    state = RunState(flow)
    pool = VariablePool(start_id=flow.start_id)
    pool.set(flow.start_id, {"labels": ["one", "two"]})
    pool.add_system("run_id", _RUN_ID)
    data = RunStartData(
        run_id=_RUN_ID,
        flow_fingerprint=_FINGERPRINT,
        initial_pool=pool,
        start_wiring_override={"metadata": {"labels": ["one", "two"]}},
        effective_llm_by_static_path={
            "agent": {"options": {"stop": ["END"]}},
        },
        num_workers=num_workers,
    )
    log = MemoryRunLog(_RUN_ID)
    commit_prepared(log, state, prepare_run_started(state, data))
    _assert_committed_path(log)
    return state, log, flow


def _assert_committed_path(log: MemoryRunLog, *tail: Any) -> None:
    """Assert one replay-shaped prefix with RunStarted followed by ``tail``."""
    envelopes = tuple(log)
    assert [envelope.seq for envelope in envelopes] == list(range(len(envelopes)))
    assert all(envelope.run_id == _RUN_ID for envelope in envelopes)
    assert isinstance(envelopes[0].record, RunStartedRecord)
    assert tuple(envelope.record for envelope in envelopes[1:]) == tail


def _start_frontier(state: RunState, log: MemoryRunLog, *node_ids: str) -> None:
    """Accept starts from an isolated test frontier through the real reducer."""
    state.ready = deque(node_ids)
    for node_id in node_ids:
        commit_prepared(log, state, prepare_node_started(state, node_id))


def _state_snapshot(state: RunState) -> dict[str, Any]:
    """Return every reducer coordinate in an equality-friendly detached form."""
    return {
        "flow": state.flow,
        "pool": state.pool.model_dump(mode="json"),
        "pool_identity": id(state.pool),
        "node_state": dict(state.node_state),
        "node_state_identity": id(state.node_state),
        "edge_state": dict(state.edge_state),
        "edge_state_identity": id(state.edge_state),
        "ready": tuple(state.ready),
        "ready_identity": id(state.ready),
        "in_flight": tuple(state.in_flight),
        "in_flight_identity": id(state.in_flight),
        "accepted_starts": dict(state.accepted_starts),
        "accepted_starts_identity": id(state.accepted_starts),
        "conversation_state": state.conversation_state,
        "result": state.result,
        "terminal_unreachable_failure": state.terminal_unreachable_failure,
        "depth": dict(state.depth),
        "depth_identity": id(state.depth),
        "wiring_overrides": deepcopy(state.wiring_overrides),
        "wiring_identity": id(state.wiring_overrides),
        "run_id": state.run_id,
        "flow_fingerprint": state.flow_fingerprint,
        "next_sequence": state.next_sequence,
        "num_workers": state.num_workers,
        "input": deepcopy(state.input),
        "input_identity": id(state.input),
        "effective_llm_by_static_path": deepcopy(
            state.effective_llm_by_static_path
        ),
        "effective_identity": id(state.effective_llm_by_static_path),
    }


def test_node_started_requires_and_removes_the_exact_ready_head():
    state, log, _ = _initialized()
    before = _state_snapshot(state)

    prepared = prepare_node_started(state, START_ID)

    assert _state_snapshot(state) == before
    assert prepared.record == NodeStartedRecord(node_id=START_ID)
    assert prepared.observers == (NodeStarted(node_id=START_ID),)

    observers = commit_prepared(log, state, prepared)

    assert observers == (NodeStarted(node_id=START_ID),)
    assert tuple(state.ready) == ()
    assert tuple(state.in_flight) == (START_ID,)
    assert state.accepted_starts == {START_ID: 1}
    assert state.node_state[START_ID] is NodeState.UNKNOWN
    assert state.next_sequence == 2
    _assert_committed_path(log, NodeStartedRecord(node_id=START_ID))


@pytest.mark.parametrize(
    ("node_id", "ready", "match"),
    [
        ("first", (), "ready head"),
        ("second", ("first", "second"), "ready head"),
        ("missing", ("missing",), "does not exist"),
    ],
)
def test_node_started_rejects_an_empty_wrong_or_unknown_ready_head(
    node_id: str, ready: tuple[str, ...], match: str
):
    state, log, _ = _initialized()
    state.ready = deque(ready)
    before = _state_snapshot(state)

    with pytest.raises(ValueError, match=match):
        prepare_node_started(state, node_id)

    assert _state_snapshot(state) == before
    assert log.next_sequence == 1


@pytest.mark.parametrize("invalid", [True, 1, None, object()])
def test_node_started_requires_a_string_node_id(invalid: Any):
    state, log, _ = _initialized()

    with pytest.raises(TypeError, match="node_id"):
        prepare_node_started(state, invalid)

    assert log.next_sequence == 1


def test_node_started_requires_unknown_disposition_and_no_existing_flight():
    state, log, _ = _initialized()
    state.ready = deque(("first",))
    state.node_state["first"] = NodeState.TAKEN

    with pytest.raises(ValueError, match="UNKNOWN"):
        prepare_node_started(state, "first")

    state.node_state["first"] = NodeState.UNKNOWN
    _start_frontier(state, log, "first")
    state.ready.append("first")
    with pytest.raises(ValueError, match="in-flight"):
        prepare_node_started(state, "first")

    assert log.next_sequence == 2


def test_retried_node_can_start_again_in_a_valid_record_sequence():
    state, log, _ = _initialized()

    commit_prepared(log, state, prepare_node_started(state, START_ID))
    commit_prepared(log, state, prepare_run_resumed(state, num_workers=3))
    commit_prepared(log, state, prepare_node_started(state, START_ID))

    assert tuple(state.ready) == ()
    assert tuple(state.in_flight) == (START_ID,)
    assert state.accepted_starts == {START_ID: 3}
    _assert_committed_path(
        log,
        NodeStartedRecord(node_id=START_ID),
        RunResumedRecord(retry_node_ids=(START_ID,), num_workers=3),
        NodeStartedRecord(node_id=START_ID),
    )


def test_node_started_preparation_keeps_existing_frontier_live():
    state, log, _ = _initialized()
    _start_frontier(state, log, "first", "second")
    state.ready = deque(("ordinary",))

    prepared = prepare_node_started(state, "ordinary")

    assert prepared.record == NodeStartedRecord(node_id="ordinary")
    assert prepared._plan.delta.node_id == "ordinary"
    assert tuple(state.ready) == ("ordinary",)
    assert tuple(state.in_flight) == ("first", "second")
    _assert_committed_path(log, NodeStartedRecord("first"), NodeStartedRecord("second"))


class _InspectingLog(MemoryRunLog):
    """Assert the captured frontier remains live until append returns."""

    def __init__(
        self,
        state: RunState,
        prior: MemoryRunLog,
        expected_ready: tuple[str, ...],
        expected_in_flight: tuple[str, ...],
    ) -> None:
        super().__init__(_RUN_ID)
        for envelope in prior:
            super().append(envelope.record)
        self._state = state
        self._expected_ready = expected_ready
        self._expected_in_flight = expected_in_flight
        self._expected_accepted_starts = dict(state.accepted_starts)
        self.saw_append = False

    def append(self, record):
        assert tuple(self._state.ready) == self._expected_ready
        assert tuple(self._state.in_flight) == self._expected_in_flight
        assert self._state.accepted_starts == self._expected_accepted_starts
        self.saw_append = True
        return super().append(record)


def test_node_started_appends_before_applying_the_frontier_delta():
    state, prior, _ = _initialized()
    log = _InspectingLog(state, prior, (START_ID,), ())

    commit_prepared(log, state, prepare_node_started(state, START_ID))

    assert log.saw_append
    assert tuple(state.ready) == ()
    assert tuple(state.in_flight) == (START_ID,)
    _assert_committed_path(log, NodeStartedRecord(node_id=START_ID))


def test_run_resumed_appends_before_applying_the_frontier_delta():
    state, prior, _ = _initialized(num_workers=2)
    commit_prepared(prior, state, prepare_node_started(state, START_ID))
    log = _InspectingLog(
        state,
        prior,
        (),
        (START_ID,),
    )

    commit_prepared(log, state, prepare_run_resumed(state, num_workers=6))

    assert log.saw_append
    assert tuple(state.ready) == (START_ID,)
    assert state.in_flight == {}
    assert state.accepted_starts == {}
    assert state.num_workers == 6
    _assert_committed_path(
        log,
        NodeStartedRecord(node_id=START_ID),
        RunResumedRecord(retry_node_ids=(START_ID,), num_workers=6),
    )


def test_run_resumed_preparation_preserves_retry_order_ahead_of_ready_work():
    state, log, _ = _initialized(num_workers=2)
    _start_frontier(state, log, "first", "second")
    state.ready = deque(("ordinary", END_ID))
    before = _state_snapshot(state)

    prepared = prepare_run_resumed(state, num_workers=5)

    assert _state_snapshot(state) == before
    assert prepared.record == RunResumedRecord(
        retry_node_ids=("first", "second"), num_workers=5
    )
    assert prepared.observers == (RunResumed(),)

    assert prepared._plan.delta.retry_node_ids == ("first", "second")
    assert prepared._plan.delta.num_workers == 5
    assert tuple(state.ready) == ("ordinary", END_ID)
    assert tuple(state.in_flight) == ("first", "second")
    assert state.num_workers == 2
    assert state.node_state["first"] is NodeState.UNKNOWN
    assert state.node_state["second"] is NodeState.UNKNOWN
    _assert_committed_path(log, NodeStartedRecord("first"), NodeStartedRecord("second"))


@pytest.mark.parametrize("num_workers", [-1, True, 1.5, "3", None])
def test_run_resumed_requires_an_exact_nonnegative_worker_count(num_workers: Any):
    state, log, _ = _initialized()
    before = _state_snapshot(state)

    with pytest.raises((TypeError, ValueError), match="num_workers"):
        prepare_run_resumed(state, num_workers=num_workers)

    assert _state_snapshot(state) == before
    assert log.next_sequence == 1


def test_run_resumed_allows_an_empty_retry_segment_and_changes_workers():
    state, log, _ = _initialized(num_workers=4)

    prepared = prepare_run_resumed(state, num_workers=0)
    commit_prepared(log, state, prepared)

    assert prepared.record == RunResumedRecord(retry_node_ids=(), num_workers=0)
    assert tuple(state.ready) == (START_ID,)
    assert state.in_flight == {}
    assert state.num_workers == 0
    _assert_committed_path(log, RunResumedRecord(retry_node_ids=(), num_workers=0))


def _wide_resume_flow(width: int) -> CompiledFlow:
    """Return a START fan-out whose first successor can be retried over wide ready work."""
    queued = [f"queued-{index}" for index in range(width)]
    nodes = {
        START_ID: StartNode(START_ID),
        "retry": FuncNode("retry", lambda inputs: inputs),
        END_ID: EndNode.record(END_ID, output_names=[]),
    }
    nodes.update(
        {
            node_id: FuncNode(node_id, lambda inputs: inputs)
            for node_id in queued
        }
    )
    edges = [Edge(id="start-retry", from_=START_ID, to="retry")]
    edges.extend(
        Edge(id=f"start-{node_id}", from_=START_ID, to=node_id)
        for node_id in queued
    )
    return CompiledFlow.from_parts(nodes=nodes, edges=edges)


def test_run_resumed_preserves_wide_ready_work_without_copying_it():
    state, log, _ = _initialized(flow=_wide_resume_flow(512))
    commit_prepared(log, state, prepare_node_started(state, START_ID))
    commit_prepared(
        log,
        state,
        prepare_node_succeeded(state, START_ID, {"started": True}),
    )
    commit_prepared(log, state, prepare_node_started(state, "retry"))

    prepared = prepare_run_resumed(state, num_workers=3)
    ready = state.ready
    commit_prepared(log, state, prepared)

    assert state.ready is ready
    assert len(state.ready) == 513
    assert state.ready[0] == "retry"
    assert state.in_flight == {}
    assert prepared.record.retry_node_ids == ("retry",)


def test_run_resumed_validation_does_not_read_the_ready_frontier():
    tree = ast.parse(dedent(inspect.getsource(state_module._validate_run_resumed)))
    ready_reads = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "state"
        and node.attr == "ready"
    ]
    assert ready_reads == []


@pytest.mark.parametrize(
    ("case", "match"),
    [
        ("missing", "does not exist"),
        ("taken", "UNKNOWN"),
    ],
)
def test_run_resumed_rejects_invalid_retry_frontiers(
    case: str,
    match: str,
):
    state, log, _ = _initialized()
    _start_frontier(state, log, "first")
    if case == "missing":
        del state.flow.nodes["first"]
    else:
        state.node_state["first"] = NodeState.TAKEN
    before = _state_snapshot(state)

    with pytest.raises(ValueError, match=match):
        prepare_run_resumed(state, num_workers=3)

    assert _state_snapshot(state) == before
    assert log.next_sequence == 2


@pytest.mark.parametrize(
    "retry_node_ids",
    [
        (),
        ("first",),
        ("second", "first"),
        ("first", "second", "ordinary"),
    ],
)
def test_prepare_record_requires_the_exact_ordered_resume_retry_list(
    retry_node_ids: tuple[str, ...],
):
    state, log, _ = _initialized()
    _start_frontier(state, log, "first", "second")
    before = _state_snapshot(state)

    with pytest.raises(ValueError, match="retry_node_ids"):
        prepare_record(
            state,
            RunResumedRecord(retry_node_ids=retry_node_ids, num_workers=3),
        )

    assert _state_snapshot(state) == before
    assert log.next_sequence == 3


def test_prepare_record_accepts_exact_dispatch_and_resume_records():
    state, log, _ = _initialized()

    started = prepare_record(state, NodeStartedRecord(node_id=START_ID))
    commit_prepared(log, state, started)
    resumed = prepare_record(
        state,
        RunResumedRecord(retry_node_ids=(START_ID,), num_workers=7),
    )
    commit_prepared(log, state, resumed)

    assert tuple(state.ready) == (START_ID,)
    assert state.in_flight == {}
    assert state.num_workers == 7
    _assert_committed_path(
        log,
        NodeStartedRecord(node_id=START_ID),
        RunResumedRecord(retry_node_ids=(START_ID,), num_workers=7),
    )


def test_prepare_record_requires_a_builder_for_expansion_replay():
    state, _, _ = _initialized()

    with pytest.raises(TypeError, match="grow_builder"):
        prepare_record(
            state,
            NodeExpandedRecord(
                spawner_id="first",
                seed={},
                prune_ids=(),
                topology_digest="0" * 64,
            ),
        )


def test_dispatch_and_resume_require_initialized_nonterminal_state():
    pristine = RunState(_flow())
    with pytest.raises(ValueError, match="initialized"):
        prepare_node_started(pristine, START_ID)
    with pytest.raises(ValueError, match="initialized"):
        prepare_run_resumed(pristine, num_workers=1)

    state, _, _ = _initialized()
    state.result = RunFailed(
        failure=Failure("node.execute", "done", "first")
    )
    with pytest.raises(ValueError, match="terminal"):
        prepare_node_started(state, START_ID)
    with pytest.raises(ValueError, match="terminal"):
        prepare_run_resumed(state, num_workers=1)


@pytest.mark.parametrize("transition", ["start", "resume"])
def test_competing_frontier_plans_reject_the_second_sequence_before_append(
    transition: str,
):
    state, log, _ = _initialized()
    if transition == "start":
        first = prepare_node_started(state, START_ID)
        second = prepare_node_started(state, START_ID)
    else:
        commit_prepared(log, state, prepare_node_started(state, START_ID))
        first = prepare_run_resumed(state, num_workers=3)
        second = prepare_run_resumed(state, num_workers=3)

    commit_prepared(log, state, first)
    sequence = log.next_sequence
    with pytest.raises(ValueError, match="run-state sequence mismatch"):
        commit_prepared(log, state, second)

    assert log.next_sequence == sequence


class _CoordinateLog:
    """Non-retaining log double used only for pre-append coordinate rejection."""

    path: Path | None = None

    def __init__(self, *, run_id: str, next_sequence: int) -> None:
        self.run_id = run_id
        self.next_sequence = next_sequence

    def append(self, record):
        raise AssertionError("coordinate mismatch must reject before append")


@pytest.mark.parametrize("mismatch", ["log-run-id", "log-sequence"])
def test_non_start_commit_requires_exact_log_identity_before_append(mismatch: str):
    state, prior, _ = _initialized()
    prepared = prepare_node_started(state, START_ID)
    log = _CoordinateLog(
        run_id="different" if mismatch == "log-run-id" else _RUN_ID,
        next_sequence=2 if mismatch == "log-sequence" else 1,
    )
    before = _state_snapshot(state)
    before_sequence = log.next_sequence

    with pytest.raises(ValueError, match="run-log .*mismatch"):
        commit_prepared(log, state, prepared)

    assert _state_snapshot(state) == before
    assert log.next_sequence == before_sequence
    _assert_committed_path(prior)


def test_resume_rejects_cross_state_before_append():
    state, log, flow = _initialized()
    _start_frontier(state, log, "first", "second")
    state.ready = deque(("ordinary",))
    prepared = prepare_run_resumed(state, num_workers=4)

    other = RunState(flow)
    with pytest.raises(ValueError, match="originating RunState"):
        commit_prepared(log, other, prepared)
    assert log.next_sequence == 3


def test_dispatch_rejects_adopted_flow_or_boundary_replacement_before_append():
    state, log, _ = _initialized()
    state.ready = deque(("first",))
    prepared = prepare_node_started(state, "first")
    state.flow.nodes[START_ID] = StartNode(START_ID)

    with pytest.raises(ValueError, match="originating flow"):
        commit_prepared(log, state, prepared)

    assert log.next_sequence == 1


class _FailingLog(MemoryRunLog):
    """Fail at the append boundary without retaining the candidate record."""

    def __init__(self, prior: MemoryRunLog) -> None:
        super().__init__(prior.run_id)
        for envelope in prior:
            super().append(envelope.record)

    def append(self, record):
        raise OSError("disk full")


@pytest.mark.parametrize("transition", ["start", "resume"])
def test_append_failure_leaves_dispatch_or_resume_state_exactly_unchanged(
    transition: str,
):
    state, prior, _ = _initialized(num_workers=2)
    if transition == "start":
        prepared = prepare_node_started(state, START_ID)
    else:
        commit_prepared(prior, state, prepare_node_started(state, START_ID))
        prepared = prepare_run_resumed(state, num_workers=6)
    before = _state_snapshot(state)

    with pytest.raises(OSError, match="disk full"):
        commit_prepared(_FailingLog(prior), state, prepared)

    assert _state_snapshot(state) == before


@pytest.mark.parametrize("transition", ["start", "resume"])
def test_committed_frontier_deltas_preserve_container_identities(transition: str):
    state, log, _ = _initialized()
    if transition == "start":
        prepared = prepare_node_started(state, START_ID)
    else:
        commit_prepared(log, state, prepare_node_started(state, START_ID))
        prepared = prepare_run_resumed(state, num_workers=3)
    roots = (
        state.pool,
        state.node_state,
        state.edge_state,
        state.ready,
        state.in_flight,
        state.depth,
        state.wiring_overrides,
        state.input,
        state.effective_llm_by_static_path,
    )
    root_ids = tuple(id(root) for root in roots)

    commit_prepared(log, state, prepared)

    assert root_ids == tuple(
        id(root)
        for root in (
            state.pool,
            state.node_state,
            state.edge_state,
            state.ready,
            state.in_flight,
            state.depth,
            state.wiring_overrides,
            state.input,
            state.effective_llm_by_static_path,
        )
    )


@pytest.mark.parametrize("transition", ["start", "resume"])
def test_frontier_commit_uses_only_the_prevalidated_delta(
    monkeypatch: pytest.MonkeyPatch,
    transition: str,
):
    state, log, _ = _initialized()
    if transition == "start":
        prepared = prepare_node_started(state, START_ID)
    else:
        commit_prepared(log, state, prepare_node_started(state, START_ID))
        prepared = prepare_run_resumed(state, num_workers=3)

    def unexpected(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("commit repeated preparation work")

    monkeypatch.setattr(state_module, "_validate_node_started", unexpected)
    monkeypatch.setattr(state_module, "_validate_run_resumed", unexpected)
    monkeypatch.setattr(state_module, "encode_pool", unexpected)
    monkeypatch.setattr(state_module, "decode_pool", unexpected)
    monkeypatch.setattr(state.flow, "incoming", unexpected)
    monkeypatch.setattr(state.flow, "outgoing", unexpected)

    commit_prepared(log, state, prepared)

    assert state.next_sequence == log.next_sequence


def test_frontier_commit_rejects_public_record_mutation_before_append():
    state, log, _ = _initialized()
    prepared = prepare_node_started(state, START_ID)
    object.__setattr__(prepared.record, "node_id", "changed")

    with pytest.raises(ValueError, match="record changed"):
        commit_prepared(log, state, prepared)

    assert log.next_sequence == 1
    assert tuple(state.ready) == (START_ID,)


def test_commit_returns_fresh_observers_derived_from_the_canonical_plan():
    state, log, _ = _initialized()
    prepared = prepare_node_started(state, START_ID)
    prepared.observers[0].node_id = "caller-mutated"

    observers = commit_prepared(log, state, prepared)

    assert observers == (NodeStarted(node_id=START_ID),)
    assert observers[0] is not prepared.observers[0]
    assert tuple(state.in_flight) == (START_ID,)
    _assert_committed_path(log, NodeStartedRecord(node_id=START_ID))


def test_frontier_plan_is_frozen_and_contains_only_an_immutable_delta():
    state, _, flow = _initialized()
    state.ready = deque(("first", "second"))
    prepared = prepare_node_started(state, "first")
    plan = prepared._plan

    assert plan.originating_state is state
    assert plan.originating_flow is flow
    assert plan.delta.node_id == "first"
    assert isinstance(plan.envelope_bytes, bytes)
    assert not hasattr(plan, "expected_ready")
    assert not hasattr(plan, "expected_in_flight")
    assert not hasattr(plan, "authority_signature")
    with pytest.raises(FrozenInstanceError):
        plan.delta = None  # type: ignore[misc]


class _ReturningLog:
    """Minimal protocol violator used to prove non-start append verification."""

    path: Path | None = None

    def __init__(self, record: NodeStartedRecord) -> None:
        self.run_id = _RUN_ID
        self.next_sequence = 1
        self._record = record

    def append(self, record):
        return RecordEnvelope(
            schema=SCHEMA_VERSION,
            run_id=self.run_id,
            seq=self.next_sequence + 1,
            record=self._record,
        )


def test_non_start_append_return_mismatch_leaves_state_unapplied():
    state, _, _ = _initialized()
    prepared = prepare_node_started(state, START_ID)
    before = _state_snapshot(state)

    with pytest.raises(ValueError, match="append result"):
        commit_prepared(_ReturningLog(prepared.record), state, prepared)

    assert _state_snapshot(state) == before
