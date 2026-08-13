"""Transactional node-outcome tests for the authoritative run-state reducer."""

from __future__ import annotations

import gc
import tracemalloc
from copy import deepcopy
from dataclasses import FrozenInstanceError
from statistics import median
from typing import Any

import pytest

import agent_composer.runtime.state as state_module
from agent_composer.compile.model import CompiledFlow, Edge, END_ID, NodeState, START_ID
from agent_composer.events import NodeSucceeded
from agent_composer.nodes.end import EndNode
from agent_composer.nodes.start import StartNode
from agent_composer.runtime.log import (
    MemoryRunLog,
    NodeStartedRecord,
    NodeSucceededRecord,
    RunStartedRecord,
    decode_pool,
    encode_pool,
)
from agent_composer.runtime.state import (
    RunStartData,
    RunState,
    commit_prepared,
    prepare_node_started,
    prepare_node_succeeded,
    prepare_record,
    prepare_run_started,
)
from agent_composer.typesys import Type, ValueKind, VariablePool
from tests.engine._fakes import FuncNode


_FINGERPRINT = "b" * 64
_RUN_ID = "outcome-run"


def _fanout_flow() -> CompiledFlow:
    """Return a complete graph whose START success makes two nodes ready in order."""
    nodes = {
        START_ID: StartNode(START_ID),
        "first": FuncNode("first", lambda inputs: inputs),
        "second": FuncNode("second", lambda inputs: inputs),
        END_ID: EndNode.record(END_ID, output_names=[]),
    }
    return CompiledFlow.from_parts(
        nodes=nodes,
        edges=[
            Edge(id="start-first", from_=START_ID, to="first"),
            Edge(id="start-second", from_=START_ID, to="second"),
        ],
    )


def _initialized_and_started(
    flow: CompiledFlow,
) -> tuple[RunState, MemoryRunLog]:
    """Return a state/log after a replay-valid RunStarted and START dispatch."""
    pool = VariablePool(start_id=flow.start_id)
    pool.set(flow.start_id, {"items": ["seed"]})
    pool.add_system("run_id", _RUN_ID)
    state = RunState(flow)
    log = MemoryRunLog(_RUN_ID)
    commit_prepared(
        log,
        state,
        prepare_run_started(
            state,
            RunStartData(
                run_id=_RUN_ID,
                flow_fingerprint=_FINGERPRINT,
                initial_pool=pool,
                start_wiring_override={},
                effective_llm_by_static_path={},
                num_workers=2,
            ),
        ),
    )
    commit_prepared(log, state, prepare_node_started(state, START_ID))
    return state, log


def _commit_start_success(
    state: RunState,
    log: MemoryRunLog,
    output: Any | None = None,
) -> None:
    """Commit START's success so its outgoing nodes enter the ready FIFO."""
    commit_prepared(
        log,
        state,
        prepare_node_succeeded(
            state,
            START_ID,
            {"items": ["canonical"]} if output is None else output,
        ),
    )


def _state_snapshot(state: RunState) -> dict[str, Any]:
    """Return every reducer coordinate plus mutable-container identities."""
    return {
        "flow": state.flow,
        "pool": encode_pool(state.pool),
        "pool_identity": id(state.pool),
        "node_state": dict(state.node_state),
        "node_state_identity": id(state.node_state),
        "edge_state": dict(state.edge_state),
        "edge_state_identity": id(state.edge_state),
        "ready": tuple(state.ready),
        "ready_identity": id(state.ready),
        "in_flight": tuple(state.in_flight),
        "in_flight_identity": id(state.in_flight),
        "result": state.result,
        "depth": dict(state.depth),
        "wiring_overrides": deepcopy(state.wiring_overrides),
        "run_id": state.run_id,
        "flow_fingerprint": state.flow_fingerprint,
        "next_sequence": state.next_sequence,
        "num_workers": state.num_workers,
        "input": deepcopy(state.input),
        "effective_llm_by_static_path": deepcopy(
            state.effective_llm_by_static_path
        ),
    }


def test_success_commits_canonical_value_before_fifo_successors():
    flow = _fanout_flow()
    state, log = _initialized_and_started(flow)

    prepared = prepare_node_succeeded(
        state,
        START_ID,
        {"items": ("alpha", "beta")},
    )

    assert prepared.record == NodeSucceededRecord(
        node_id=START_ID,
        commit_target=START_ID,
        value={
            "kind": "object",
            "value": {"items": ["alpha", "beta"]},
        },
    )
    assert prepared.observers == (
        NodeSucceeded(
            node_id=START_ID,
            output={"items": ["alpha", "beta"]},
            commit_as=None,
        ),
    )

    observers = commit_prepared(log, state, prepared)

    assert observers == prepared.observers
    assert state.pool.get(START_ID) == {"items": ["alpha", "beta"]}
    assert state.node_state[START_ID] is NodeState.TAKEN
    assert state.edge_state == {
        "start-first": NodeState.TAKEN,
        "start-second": NodeState.TAKEN,
    }
    assert tuple(state.ready) == ("first", "second")
    assert tuple(state.in_flight) == ()
    assert state.next_sequence == 3
    assert [type(envelope.record) for envelope in log] == [
        RunStartedRecord,
        NodeStartedRecord,
        NodeSucceededRecord,
    ]


def _joined_fanout_flow() -> CompiledFlow:
    """Return a graph with fan-out, a two-edge join, and an existing ready node."""
    nodes = {
        START_ID: StartNode(START_ID),
        "first": FuncNode("first", lambda inputs: inputs),
        "second": FuncNode("second", lambda inputs: inputs),
        "existing": FuncNode("existing", lambda inputs: inputs),
        "after-a": FuncNode("after-a", lambda inputs: inputs),
        "after-b": FuncNode("after-b", lambda inputs: inputs),
        "join": FuncNode("join", lambda inputs: inputs),
        END_ID: EndNode.record(END_ID, output_names=[]),
    }
    return CompiledFlow.from_parts(
        nodes=nodes,
        edges=[
            Edge(id="start-first", from_=START_ID, to="first"),
            Edge(id="start-second", from_=START_ID, to="second"),
            Edge(id="start-existing", from_=START_ID, to="existing"),
            Edge(id="first-join", from_="first", to="join", input_group="both"),
            Edge(id="first-after-a", from_="first", to="after-a"),
            Edge(id="first-after-b", from_="first", to="after-b"),
            Edge(id="second-join", from_="second", to="join", input_group="both"),
        ],
    )


def test_success_preserves_sibling_flight_and_appends_new_readiness_fifo():
    flow = _joined_fanout_flow()
    state, log = _initialized_and_started(flow)
    _commit_start_success(state, log)
    commit_prepared(log, state, prepare_node_started(state, "first"))
    commit_prepared(log, state, prepare_node_started(state, "second"))

    commit_prepared(
        log,
        state,
        prepare_node_succeeded(state, "first", {"done": "first"}),
    )

    assert tuple(state.in_flight) == ("second",)
    assert tuple(state.ready) == ("existing", "after-a", "after-b")
    assert state.edge_state["first-join"] is NodeState.TAKEN
    assert state.edge_state["second-join"] is NodeState.UNKNOWN

    commit_prepared(
        log,
        state,
        prepare_node_succeeded(state, "second", {"done": "second"}),
    )

    assert tuple(state.in_flight) == ()
    assert tuple(state.ready) == ("existing", "after-a", "after-b", "join")


def _redirect_flow() -> CompiledFlow:
    """Return a graph whose executing filler has a baked and explicit target."""
    list_type = Type.scalar(ValueKind.LIST_STRING)
    filler = FuncNode("filler", lambda inputs: inputs)
    filler.commit_as = "baked"
    nodes = {
        START_ID: StartNode(START_ID),
        "filler": filler,
        "baked": FuncNode("baked", lambda inputs: inputs, output_type=list_type),
        "explicit": FuncNode(
            "explicit", lambda inputs: inputs, output_type=list_type
        ),
        "after-baked": FuncNode("after-baked", lambda inputs: inputs),
        "after-explicit": FuncNode("after-explicit", lambda inputs: inputs),
        END_ID: EndNode.record(END_ID, output_names=[]),
    }
    return CompiledFlow.from_parts(
        nodes=nodes,
        edges=[
            Edge(id="start-filler", from_=START_ID, to="filler"),
            Edge(id="baked-after", from_="baked", to="after-baked"),
            Edge(id="explicit-after", from_="explicit", to="after-explicit"),
        ],
    )


def _redirect_ready() -> tuple[RunState, MemoryRunLog]:
    """Return a replay-valid state with the redirect filler in flight."""
    state, log = _initialized_and_started(_redirect_flow())
    _commit_start_success(state, log)
    commit_prepared(log, state, prepare_node_started(state, "filler"))
    return state, log


@pytest.mark.parametrize(
    ("supplied_target", "expected_target", "expected_successor"),
    [
        ("explicit", "explicit", "after-explicit"),
        (None, "baked", "after-baked"),
    ],
)
def test_success_resolves_redirect_precedence_and_preserves_target_disposition(
    supplied_target: str | None,
    expected_target: str,
    expected_successor: str,
):
    state, log = _redirect_ready()
    other_target = "baked" if expected_target == "explicit" else "explicit"

    prepared = prepare_node_succeeded(
        state,
        "filler",
        ("alpha", "beta"),
        commit_as=supplied_target,
    )
    observers = commit_prepared(log, state, prepared)

    assert prepared.record.commit_target == expected_target
    assert prepared.record.value == {
        "kind": "list[str]",
        "value": ["alpha", "beta"],
    }
    assert observers == (
        NodeSucceeded(
            node_id="filler",
            output=["alpha", "beta"],
            commit_as=expected_target,
        ),
    )
    assert state.pool.get(expected_target) == ["alpha", "beta"]
    assert state.pool.get_value(other_target) is None
    assert state.node_state["filler"] is NodeState.TAKEN
    assert state.node_state[expected_target] is NodeState.UNKNOWN
    assert state.edge_state[f"{expected_target}-after"] is NodeState.TAKEN
    assert state.edge_state[f"{other_target}-after"] is NodeState.UNKNOWN
    assert tuple(state.ready) == (expected_successor,)


@pytest.mark.parametrize(
    ("output", "commit_as", "match"),
    [
        ("not-a-list", None, "declared list"),
        (("valid",), "missing", "does not exist"),
    ],
)
def test_success_rejects_type_or_missing_target_before_append(
    output: Any,
    commit_as: str | None,
    match: str,
):
    state, log = _redirect_ready()
    before = _state_snapshot(state)
    sequence = log.next_sequence

    with pytest.raises(ValueError, match=match):
        prepare_node_succeeded(
            state,
            "filler",
            output,
            commit_as=commit_as,
        )

    assert _state_snapshot(state) == before
    assert log.next_sequence == sequence


def test_success_rejects_a_previously_settled_target_edge_before_append():
    state, log = _redirect_ready()
    state.edge_state["baked-after"] = NodeState.TAKEN
    before = _state_snapshot(state)
    sequence = log.next_sequence

    with pytest.raises(ValueError, match="outgoing edge .* must be UNKNOWN"):
        prepare_node_succeeded(state, "filler", ("valid",))

    assert _state_snapshot(state) == before
    assert log.next_sequence == sequence


def test_success_rejects_a_settled_newly_ready_successor_before_append():
    state, log = _redirect_ready()
    state.node_state["after-baked"] = NodeState.TAKEN
    before = _state_snapshot(state)
    sequence = log.next_sequence

    with pytest.raises(ValueError, match="must have UNKNOWN disposition"):
        prepare_node_succeeded(state, "filler", ("valid",))

    assert _state_snapshot(state) == before
    assert log.next_sequence == sequence


def _duplicate_successor_flow() -> CompiledFlow:
    """Return a redirect graph with two target edges into the same successor."""
    flow = _redirect_flow()
    edges = [
        edge
        for edge in flow.edges
        if edge.id != "baked-after"
    ]
    edges.extend(
        [
            Edge(id="baked-after-1", from_="baked", to="after-baked"),
            Edge(id="baked-after-2", from_="baked", to="after-baked"),
        ]
    )
    return CompiledFlow.from_parts(nodes=flow.nodes, edges=edges)


def test_success_multiple_edges_to_one_successor_enqueue_it_once():
    state, log = _initialized_and_started(_duplicate_successor_flow())
    _commit_start_success(state, log)
    commit_prepared(log, state, prepare_node_started(state, "filler"))

    commit_prepared(
        log,
        state,
        prepare_node_succeeded(state, "filler", ("valid",)),
    )

    assert state.edge_state["baked-after-1"] is NodeState.TAKEN
    assert state.edge_state["baked-after-2"] is NodeState.TAKEN
    assert tuple(state.ready) == ("after-baked",)


def _interleaved_successor_flow() -> CompiledFlow:
    """Return target edges ordered A1, B, A2 so B becomes ready before A."""
    nodes = {
        START_ID: StartNode(START_ID),
        "target": FuncNode("target", lambda inputs: inputs),
        "after-a": FuncNode("after-a", lambda inputs: inputs),
        "after-b": FuncNode("after-b", lambda inputs: inputs),
        END_ID: EndNode.record(END_ID, output_names=[]),
    }
    return CompiledFlow.from_parts(
        nodes=nodes,
        edges=[
            Edge(id="start-target", from_=START_ID, to="target"),
            Edge(id="target-a-1", from_="target", to="after-a", input_group="a"),
            Edge(id="target-b", from_="target", to="after-b"),
            Edge(id="target-a-2", from_="target", to="after-a", input_group="a"),
        ],
    )


def test_success_preserves_sequential_readiness_for_interleaved_successors():
    state, log = _initialized_and_started(_interleaved_successor_flow())
    _commit_start_success(state, log)
    commit_prepared(log, state, prepare_node_started(state, "target"))

    commit_prepared(
        log,
        state,
        prepare_node_succeeded(state, "target", {"done": True}),
    )

    assert tuple(state.ready) == ("after-b", "after-a")


def _parallel_successor_flow(width: int) -> CompiledFlow:
    """Return two wide joins fed by interleaved edges from one target."""
    nodes = {
        START_ID: StartNode(START_ID),
        "target": FuncNode("target", lambda inputs: inputs),
        "after-a": FuncNode("after-a", lambda inputs: inputs),
        "after-b": FuncNode("after-b", lambda inputs: inputs),
        END_ID: EndNode.record(END_ID, output_names=[]),
    }
    edges = [Edge(id="start-target", from_=START_ID, to="target")]
    for index in range(width):
        edges.extend(
            (
                Edge(
                    id=f"target-a-{index}",
                    from_="target",
                    to="after-a",
                    input_group="a",
                ),
                Edge(
                    id=f"target-b-{index}",
                    from_="target",
                    to="after-b",
                    input_group="b",
                ),
            )
        )
    return CompiledFlow.from_parts(nodes=nodes, edges=edges)


def test_success_scans_each_distinct_affected_join_once(
    monkeypatch: pytest.MonkeyPatch,
):
    state, log = _initialized_and_started(_parallel_successor_flow(64))
    _commit_start_success(state, log)
    commit_prepared(log, state, prepare_node_started(state, "target"))
    incoming = state.flow.incoming
    scanned: list[str] = []

    def counted_incoming(node_id: str) -> list[Edge]:
        scanned.append(node_id)
        return incoming(node_id)

    monkeypatch.setattr(state.flow, "incoming", counted_incoming)

    prepared = prepare_node_succeeded(state, "target", {"done": True})

    assert scanned == ["after-a", "after-b"]
    assert prepared._plan.delta.newly_ready_ids == ("after-a", "after-b")


@pytest.mark.parametrize(
    ("node_id", "match"),
    [("missing", "does not exist"), ("first", "not in-flight")],
)
def test_success_requires_an_existing_unknown_in_flight_node(
    node_id: str,
    match: str,
):
    state, log = _initialized_and_started(_fanout_flow())
    before = _state_snapshot(state)

    with pytest.raises(ValueError, match=match):
        prepare_node_succeeded(state, node_id, {"value": 1})

    assert _state_snapshot(state) == before
    assert log.next_sequence == 2


class _FailingAppendLog(MemoryRunLog):
    """Copy an authoritative prefix and fail only the next append."""

    def __init__(self, prior: MemoryRunLog) -> None:
        super().__init__(prior.run_id)
        self.fail = False
        for envelope in prior:
            super().append(envelope.record)
        self.fail = True

    def append(self, record):
        if self.fail:
            raise OSError("append failed")
        return super().append(record)


def test_success_append_failure_leaves_every_state_coordinate_unchanged():
    state, prior = _initialized_and_started(_fanout_flow())
    prepared = prepare_node_succeeded(state, START_ID, {"value": [1, 2]})
    before = _state_snapshot(state)
    log = _FailingAppendLog(prior)

    with pytest.raises(OSError, match="append failed"):
        commit_prepared(log, state, prepared)

    assert _state_snapshot(state) == before
    assert log.next_sequence == prior.next_sequence


def test_success_delta_preserves_all_reducer_container_identities():
    state, log = _initialized_and_started(_fanout_flow())
    prepared = prepare_node_succeeded(state, START_ID, {"value": "done"})
    roots = (
        state.pool,
        state.pool.store,
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
            state.pool.store,
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


def test_success_plan_contains_only_an_immutable_sparse_delta():
    state, _ = _initialized_and_started(_fanout_flow())
    plan = prepare_node_succeeded(
        state,
        START_ID,
        {"value": "done"},
    )._plan

    assert isinstance(plan.delta.value_bytes, bytes)
    assert plan.delta.outgoing_edge_ids == ("start-first", "start-second")
    assert plan.delta.newly_ready_ids == ("first", "second")
    with pytest.raises(FrozenInstanceError):
        plan.delta.newly_ready_ids = ()  # type: ignore[misc]


def test_success_commit_uses_only_the_prevalidated_delta(
    monkeypatch: pytest.MonkeyPatch,
):
    state, log = _initialized_and_started(_fanout_flow())
    prepared = prepare_node_succeeded(state, START_ID, {"value": "done"})

    def unexpected(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("commit repeated success preparation work")

    monkeypatch.setattr(state_module, "_validate_success_value", unexpected)
    monkeypatch.setattr(state_module, "_prepare_success_delta", unexpected)
    monkeypatch.setattr(state_module, "canonical_typed", unexpected)
    monkeypatch.setattr(state_module, "encode_pool", unexpected)
    monkeypatch.setattr(state_module, "decode_pool", unexpected)
    monkeypatch.setattr(state.flow, "incoming", unexpected)
    monkeypatch.setattr(state.flow, "outgoing", unexpected)

    commit_prepared(log, state, prepared)

    assert state.pool.get(START_ID) == {"value": "done"}
    assert tuple(state.ready) == ("first", "second")


@pytest.mark.parametrize("broken", ["edge-state", "successor", "incoming"])
def test_success_rejects_touched_topology_invariant_breaks(broken: str):
    state, log = _redirect_ready()
    if broken == "edge-state":
        del state.edge_state["baked-after"]
        match = "has no disposition"
    elif broken == "successor":
        del state.node_state["after-baked"]
        match = "unknown successor"
    else:
        state.flow._incoming["after-baked"] = []
        match = "absent from successor"

    with pytest.raises(ValueError, match=match):
        prepare_node_succeeded(state, "filler", ("valid",))

    assert log.next_sequence == 4


def test_success_canonical_copies_isolate_input_record_pool_and_observers():
    state, log = _initialized_and_started(_fanout_flow())
    raw = {"nested": (["original"],)}

    prepared = prepare_node_succeeded(state, START_ID, raw)
    raw["nested"][0].append("raw mutation")
    prepared.observers[0].output["nested"][0].append("prepared mutation")

    observers = commit_prepared(log, state, prepared)
    success_record = tuple(log)[-1].record
    assert isinstance(success_record, NodeSucceededRecord)
    assert success_record.value["value"] == {"nested": [["original"]]}
    assert state.pool.get(START_ID) == {"nested": [["original"]]}
    assert observers == (
        NodeSucceeded(START_ID, {"nested": [["original"]]}, None),
    )

    observers[0].output["nested"][0].append("returned mutation")
    assert state.pool.get(START_ID) == {"nested": [["original"]]}
    assert success_record.value["value"] == {"nested": [["original"]]}
    state.pool.get(START_ID)["nested"][0].append("pool mutation")
    assert success_record.value["value"] == {"nested": [["original"]]}


def test_competing_success_plans_reject_the_second_sequence_before_append():
    state, log = _initialized_and_started(_fanout_flow())
    first = prepare_node_succeeded(state, START_ID, {"value": "done"})
    second = prepare_node_succeeded(state, START_ID, {"value": "done"})

    commit_prepared(log, state, first)
    sequence = log.next_sequence
    with pytest.raises(ValueError, match="run-state sequence mismatch"):
        commit_prepared(log, state, second)

    assert log.next_sequence == sequence


def _reducer_snapshot(state: RunState) -> dict[str, Any]:
    """Return value-only reducer coordinates for comparing independent replays."""
    snapshot = _state_snapshot(state)
    snapshot.pop("flow")
    for key in tuple(snapshot):
        if key.endswith("_identity"):
            snapshot.pop(key)
    return snapshot


def test_success_record_replay_reconstructs_identical_state():
    live_state, live_log = _initialized_and_started(_fanout_flow())
    commit_prepared(
        live_log,
        live_state,
        prepare_node_succeeded(
            live_state,
            START_ID,
            {"items": ("alpha", "beta")},
        ),
    )
    records = tuple(envelope.record for envelope in live_log)
    started = records[0]
    assert isinstance(started, RunStartedRecord)

    replay_flow = _fanout_flow()
    replay_state = RunState(replay_flow)
    replay_log = MemoryRunLog(_RUN_ID)
    commit_prepared(
        replay_log,
        replay_state,
        prepare_run_started(
            replay_state,
            RunStartData(
                run_id=_RUN_ID,
                flow_fingerprint=started.flow_fingerprint,
                initial_pool=decode_pool(started.initial_pool),
                start_wiring_override=started.start_wiring_override,
                effective_llm_by_static_path=started.effective_llm_by_static_path,
                num_workers=started.num_workers,
            ),
        ),
    )
    for record in records[1:]:
        commit_prepared(replay_log, replay_state, prepare_record(replay_state, record))

    assert _reducer_snapshot(replay_state) == _reducer_snapshot(live_state)
    assert tuple(replay_log) == tuple(live_log)


def _constant_degree_success_state(size: int) -> RunState:
    """Return a target-in-flight state with ``size`` unrelated degree-one nodes."""
    nodes = {
        START_ID: StartNode(START_ID),
        "target": FuncNode("target", lambda inputs: inputs),
        "after": FuncNode("after", lambda inputs: inputs),
        END_ID: EndNode.record(END_ID, output_names=[]),
    }
    nodes.update(
        {
            f"unused-{index}": FuncNode(
                f"unused-{index}",
                lambda inputs: inputs,
            )
            for index in range(size)
        }
    )
    edges = [
        Edge(id="start-target", from_=START_ID, to="target"),
        Edge(id="target-after", from_="target", to="after"),
    ]
    edges.extend(
        Edge(
            id=f"unused-edge-{index}",
            from_=f"unused-{index}",
            to=f"unused-{index + 1}",
        )
        for index in range(size - 1)
    )
    state, log = _initialized_and_started(
        CompiledFlow.from_parts(nodes=nodes, edges=edges)
    )
    _commit_start_success(state, log)
    commit_prepared(log, state, prepare_node_started(state, "target"))
    return state


class _AllocationStop(Exception):
    """Stop one allocation probe at the append boundary."""


class _AllocationStopLog:
    """Log double that preserves state while measuring all pre-append work."""

    path = None

    def __init__(self, state: RunState) -> None:
        self.run_id = state.run_id
        self.next_sequence = state.next_sequence

    def append(self, record: Any) -> Any:
        raise _AllocationStop


def _median_success_reducer_peak(state: RunState) -> float:
    """Return median peak bytes through preparation and pre-append commit."""
    log = _AllocationStopLog(state)
    prepared = prepare_node_succeeded(state, "target", {"done": True})
    with pytest.raises(_AllocationStop):
        commit_prepared(log, state, prepared)
    samples: list[int] = []
    for _ in range(5):
        gc.collect()
        tracemalloc.start()
        prepared = prepare_node_succeeded(state, "target", {"done": True})
        try:
            commit_prepared(log, state, prepared)
        except _AllocationStop:
            pass
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        samples.append(peak)
        del prepared
    return median(samples)


def test_success_reducer_allocation_is_independent_of_total_run_size():
    small = _constant_degree_success_state(64)
    large = _constant_degree_success_state(4096)

    small_peak = _median_success_reducer_peak(small)
    large_peak = _median_success_reducer_peak(large)

    assert large_peak / small_peak <= 2.5, (
        f"success reducer allocation scaled with total run state: "
        f"small={small_peak}, large={large_peak}"
    )
