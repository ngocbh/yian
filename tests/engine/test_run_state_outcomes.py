"""Transactional node-outcome tests for the authoritative run-state reducer."""

from __future__ import annotations

import gc
import tracemalloc
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from statistics import median
from typing import Any

import pytest

import agent_composer.runtime.state as state_module
from agent_composer.compile.model import CompiledFlow, Edge, END_ID, NodeState, START_ID
from agent_composer.events import (
    NodeFailed,
    NodeRouted,
    NodeSkipped,
    NodeSucceeded,
    RunFailed,
    RunSucceeded,
    SourceRange,
    SourceSpan,
)
from agent_composer.nodes.contracts import ValueContract
from agent_composer.nodes.end import EndNode
from agent_composer.nodes.start import StartNode
from agent_composer.runtime.log import (
    MemoryRunLog,
    NodeFailedRecord,
    NodeRoutedRecord,
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
    prepare_node_failed,
    prepare_node_routed,
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


def _route_flow() -> CompiledFlow:
    """Return a route whose unselected sibling has a two-node skip chain."""
    nodes = {
        START_ID: StartNode(START_ID),
        "router": FuncNode("router", lambda inputs: inputs),
        "chosen": FuncNode("chosen", lambda inputs: inputs),
        "dead-parent": FuncNode("dead-parent", lambda inputs: inputs),
        "dead-child": FuncNode("dead-child", lambda inputs: inputs),
        END_ID: EndNode.record(END_ID, output_names=[]),
    }
    return CompiledFlow.from_parts(
        nodes=nodes,
        edges=[
            Edge(id="start-router", from_=START_ID, to="router"),
            Edge(
                id="route-dead",
                from_="router",
                to="dead-parent",
                source_handle="no",
            ),
            Edge(
                id="route-chosen",
                from_="router",
                to="chosen",
                source_handle="yes",
            ),
            Edge(id="dead-child", from_="dead-parent", to="dead-child"),
        ],
    )


def _route_in_flight(
    flow: CompiledFlow | None = None,
) -> tuple[RunState, MemoryRunLog]:
    """Return a replay-valid state/log with the generic router in flight."""
    state, log = _initialized_and_started(_route_flow() if flow is None else flow)
    _commit_start_success(state, log)
    commit_prepared(log, state, prepare_node_started(state, "router"))
    return state, log


def test_route_derives_skip_closure_without_skip_record():
    state, log = _route_in_flight()
    before = _state_snapshot(state)

    prepared = prepare_node_routed(state, "router", "yes")

    assert _state_snapshot(state) == before
    assert log.next_sequence == 4
    assert prepared.record == NodeRoutedRecord(node_id="router", handle="yes")
    assert prepared.observers == (
        NodeRouted("router", "yes"),
        NodeSkipped("dead-parent"),
        NodeSkipped("dead-child"),
    )
    assert prepared._plan.delta.edge_updates == (
        ("route-dead", NodeState.SKIPPED),
        ("dead-child", NodeState.SKIPPED),
        ("route-chosen", NodeState.TAKEN),
    )
    with pytest.raises(FrozenInstanceError):
        prepared._plan.delta.edge_updates = ()  # type: ignore[misc]

    observers = commit_prepared(log, state, prepared)

    assert observers == prepared.observers
    assert state.node_state["router"] is NodeState.TAKEN
    assert state.node_state["dead-parent"] is NodeState.SKIPPED
    assert state.node_state["dead-child"] is NodeState.SKIPPED
    assert state.node_state["chosen"] is NodeState.UNKNOWN
    assert tuple(state.ready) == ("chosen",)
    assert tuple(state.in_flight) == ()
    assert [type(envelope.record) for envelope in log] == [
        RunStartedRecord,
        NodeStartedRecord,
        NodeSucceededRecord,
        NodeStartedRecord,
        NodeRoutedRecord,
    ]


@pytest.mark.parametrize(
    ("selected", "matching_handles"),
    [
        ("default", (None, "")),
        ("go", ("go", "go")),
    ],
)
def test_route_takes_every_edge_matching_default_or_named_handle(
    selected: str,
    matching_handles: tuple[str | None, str | None],
):
    nodes = {
        START_ID: StartNode(START_ID),
        "router": FuncNode("router", lambda inputs: inputs),
        "left": FuncNode("left", lambda inputs: inputs),
        "right": FuncNode("right", lambda inputs: inputs),
        "dead": FuncNode("dead", lambda inputs: inputs),
        END_ID: EndNode.record(END_ID, output_names=[]),
    }
    flow = CompiledFlow.from_parts(
        nodes=nodes,
        edges=[
            Edge(id="start-router", from_=START_ID, to="router"),
            Edge(
                id="match-left",
                from_="router",
                to="left",
                source_handle=matching_handles[0],
            ),
            Edge(
                id="match-right",
                from_="router",
                to="right",
                source_handle=matching_handles[1],
            ),
            Edge(
                id="other",
                from_="router",
                to="dead",
                source_handle="other",
            ),
        ],
    )
    state, log = _route_in_flight(flow)

    commit_prepared(log, state, prepare_node_routed(state, "router", selected))

    assert state.edge_state["match-left"] is NodeState.TAKEN
    assert state.edge_state["match-right"] is NodeState.TAKEN
    assert state.edge_state["other"] is NodeState.SKIPPED
    assert state.node_state["dead"] is NodeState.SKIPPED
    assert tuple(state.ready) == ("left", "right")


def test_route_preserves_dfs_observers_and_existing_fifo_readiness():
    nodes = {
        START_ID: StartNode(START_ID),
        "router": FuncNode("router", lambda inputs: inputs),
        "existing": FuncNode("existing", lambda inputs: inputs),
        "dead-a": FuncNode("dead-a", lambda inputs: inputs),
        "ready-a": FuncNode("ready-a", lambda inputs: inputs),
        "chosen": FuncNode("chosen", lambda inputs: inputs),
        "dead-b": FuncNode("dead-b", lambda inputs: inputs),
        "ready-b": FuncNode("ready-b", lambda inputs: inputs),
        END_ID: EndNode.record(END_ID, output_names=[]),
    }
    flow = CompiledFlow.from_parts(
        nodes=nodes,
        edges=[
            Edge(id="start-router", from_=START_ID, to="router"),
            Edge(id="start-existing", from_=START_ID, to="existing"),
            Edge(
                id="route-a", from_="router", to="dead-a", source_handle="no-a"
            ),
            Edge(
                id="route-chosen",
                from_="router",
                to="chosen",
                source_handle="yes",
            ),
            Edge(
                id="route-b", from_="router", to="dead-b", source_handle="no-b"
            ),
            Edge(
                id="a-ready", from_="dead-a", to="ready-a", optional=True
            ),
            Edge(
                id="b-ready", from_="dead-b", to="ready-b", optional=True
            ),
        ],
    )
    state, log = _route_in_flight(flow)

    observers = commit_prepared(
        log,
        state,
        prepare_node_routed(state, "router", "yes"),
    )

    assert observers == (
        NodeRouted("router", "yes"),
        NodeSkipped("dead-a"),
        NodeSkipped("dead-b"),
    )
    assert tuple(state.ready) == ("existing", "ready-a", "chosen", "ready-b")


def test_route_skip_join_matches_control_data_ordering_and_wait_semantics():
    node_ids = (
        "router",
        "dead-parent",
        "control-dead",
        "data-dead",
        "ordering-dead",
        "waiting",
        "optional-ready",
        "runs-after-ready",
        "unused",
    )
    nodes = {
        START_ID: StartNode(START_ID),
        **{node_id: FuncNode(node_id, lambda inputs: inputs) for node_id in node_ids},
        END_ID: EndNode.record(END_ID, output_names=[]),
    }
    flow = CompiledFlow.from_parts(
        nodes=nodes,
        edges=[
            Edge(id="start-router", from_=START_ID, to="router"),
            Edge(
                id="start-control-data",
                from_=START_ID,
                to="control-dead",
                input_group="value",
            ),
            Edge(
                id="route-parent",
                from_="router",
                to="dead-parent",
                source_handle="no",
            ),
            Edge(
                id="route-live",
                from_="router",
                to="unused",
                source_handle="yes",
            ),
            Edge(
                id="parent-control",
                from_="dead-parent",
                to="control-dead",
                source_handle="gate",
            ),
            Edge(
                id="parent-data",
                from_="dead-parent",
                to="data-dead",
                input_group="required",
            ),
            Edge(
                id="parent-ordering",
                from_="dead-parent",
                to="ordering-dead",
                ordering=True,
            ),
            Edge(
                id="parent-ordering-again",
                from_="dead-parent",
                to="ordering-dead",
                ordering=True,
            ),
            Edge(
                id="parent-wait",
                from_="dead-parent",
                to="waiting",
                input_group="both",
            ),
            Edge(
                id="unused-wait",
                from_="unused",
                to="waiting",
                input_group="both",
            ),
            Edge(
                id="parent-optional",
                from_="dead-parent",
                to="optional-ready",
                optional=True,
            ),
            Edge(
                id="parent-runs-after",
                from_="dead-parent",
                to="runs-after-ready",
                optional=True,
                ordering=True,
            ),
        ],
    )
    state, log = _route_in_flight(flow)

    observers = commit_prepared(
        log,
        state,
        prepare_node_routed(state, "router", "yes"),
    )

    assert observers == (
        NodeRouted("router", "yes"),
        NodeSkipped("dead-parent"),
        NodeSkipped("control-dead"),
        NodeSkipped("data-dead"),
        NodeSkipped("ordering-dead"),
    )
    assert state.node_state["waiting"] is NodeState.UNKNOWN
    assert state.edge_state["start-control-data"] is NodeState.TAKEN
    assert state.edge_state["unused-wait"] is NodeState.UNKNOWN
    assert state.edge_state["parent-ordering-again"] is NodeState.SKIPPED
    assert tuple(state.ready) == (
        "optional-ready",
        "runs-after-ready",
        "unused",
    )


@pytest.mark.parametrize(
    ("node_id", "handle", "match", "node_data"),
    [
        ("missing", "yes", "does not exist", False),
        ("chosen", "yes", "not in-flight", False),
        ("router", "missing", "matches no outgoing edge", True),
    ],
)
def test_route_rejects_unknown_router_frontier_or_handle(
    node_id: str,
    handle: str,
    match: str,
    node_data: bool,
):
    state, log = _route_in_flight()
    before = _state_snapshot(state)

    with pytest.raises(ValueError, match=match) as caught:
        prepare_node_routed(state, node_id, handle)

    assert isinstance(caught.value, state_module._NodeDataError) is node_data
    assert _state_snapshot(state) == before
    assert log.next_sequence == 4


@pytest.mark.parametrize("settled", [NodeState.TAKEN, NodeState.SKIPPED])
def test_route_rejects_a_previously_settled_outgoing_edge(settled: NodeState):
    state, log = _route_in_flight()
    state.edge_state["route-dead"] = settled
    before = _state_snapshot(state)

    with pytest.raises(ValueError, match="route edge .* must be UNKNOWN"):
        prepare_node_routed(state, "router", "yes")

    assert _state_snapshot(state) == before
    assert log.next_sequence == 4


@pytest.mark.parametrize(
    ("source_handle", "error_type"),
    [(7, TypeError), (False, TypeError)],
)
def test_route_rejects_malformed_source_handle_as_a_host_invariant(
    source_handle: Any,
    error_type: type[Exception],
):
    state, log = _route_in_flight()
    outgoing = state.flow._outgoing["router"]
    selected = next(edge for edge in outgoing if edge.source_handle == "yes")
    outgoing[outgoing.index(selected)] = replace(
        selected,
        source_handle=source_handle,
    )
    before = _state_snapshot(state)

    with pytest.raises(error_type, match="source_handle") as caught:
        prepare_node_routed(state, "router", "yes")

    assert not isinstance(caught.value, state_module._NodeDataError)
    assert _state_snapshot(state) == before
    assert log.next_sequence == 4


@pytest.mark.parametrize("field", ["node", "handle"])
def test_route_requires_exact_nonempty_strings(field: str):
    state, _ = _route_in_flight()
    args = ("", "yes") if field == "node" else ("router", "")
    with pytest.raises(ValueError, match="non-empty"):
        prepare_node_routed(state, *args)

    invalid = 1
    args = (invalid, "yes") if field == "node" else ("router", invalid)
    with pytest.raises(TypeError, match=field):
        prepare_node_routed(state, *args)


def test_route_append_failure_leaves_every_state_coordinate_unchanged():
    state, prior = _route_in_flight()
    prepared = prepare_node_routed(state, "router", "yes")
    before = _state_snapshot(state)
    log = _FailingAppendLog(prior)

    with pytest.raises(OSError, match="append failed"):
        commit_prepared(log, state, prepared)

    assert _state_snapshot(state) == before
    assert log.next_sequence == prior.next_sequence


def test_route_delta_preserves_all_reducer_container_identities():
    state, log = _route_in_flight()
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

    commit_prepared(log, state, prepare_node_routed(state, "router", "yes"))

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


def test_competing_route_plans_reject_the_second_sequence_before_append():
    state, log = _route_in_flight()
    first = prepare_node_routed(state, "router", "yes")
    second = prepare_node_routed(state, "router", "yes")

    commit_prepared(log, state, first)
    sequence = log.next_sequence
    with pytest.raises(ValueError, match="run-state sequence mismatch"):
        commit_prepared(log, state, second)

    assert log.next_sequence == sequence


def test_route_observers_are_fresh_and_isolated_from_prepared_mutation():
    state, log = _route_in_flight()
    prepared = prepare_node_routed(state, "router", "yes")
    prepared.observers[0].handle = "mutated"
    prepared.observers[1].node_id = "mutated"

    observers = commit_prepared(log, state, prepared)

    assert observers == (
        NodeRouted("router", "yes"),
        NodeSkipped("dead-parent"),
        NodeSkipped("dead-child"),
    )


def test_route_record_replay_reconstructs_identical_state():
    live_state, live_log = _route_in_flight()
    commit_prepared(
        live_log,
        live_state,
        prepare_node_routed(live_state, "router", "yes"),
    )
    records = tuple(envelope.record for envelope in live_log)
    started = records[0]
    assert isinstance(started, RunStartedRecord)

    replay_flow = _route_flow()
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


def test_route_commit_uses_only_materialized_delta_after_append(
    monkeypatch: pytest.MonkeyPatch,
):
    state, prior = _route_in_flight()
    prepared = prepare_node_routed(state, "router", "yes")
    pool_before = encode_pool(state.pool)

    def unexpected(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("route commit repeated preparation work")

    monkeypatch.setattr(state_module, "_prepare_route_delta", unexpected)
    monkeypatch.setattr(state_module, "_scan_join", unexpected)
    monkeypatch.setattr(state_module, "_settle_join_edge", unexpected)
    monkeypatch.setattr(state_module, "_classify_join", unexpected)
    monkeypatch.setattr(state.flow, "incoming", unexpected)
    monkeypatch.setattr(state.flow, "outgoing", unexpected)

    class PoisonAfterAppendLog(MemoryRunLog):
        """Copy the prefix, then poison observer constructors after durable append."""

        def __init__(self, source: MemoryRunLog) -> None:
            super().__init__(source.run_id)
            for envelope in source:
                super().append(envelope.record)

        def append(self, record: Any):
            envelope = super().append(record)
            monkeypatch.setattr(state_module, "NodeRouted", unexpected)
            monkeypatch.setattr(state_module, "NodeSkipped", unexpected)
            return envelope

    log = PoisonAfterAppendLog(prior)

    observers = commit_prepared(log, state, prepared)

    assert observers == (
        NodeRouted("router", "yes"),
        NodeSkipped("dead-parent"),
        NodeSkipped("dead-child"),
    )
    assert encode_pool(state.pool) == pool_before


def test_route_scans_each_touched_head_incoming_adjacency_once(
    monkeypatch: pytest.MonkeyPatch,
):
    nodes = {
        START_ID: StartNode(START_ID),
        "router": FuncNode("router", lambda inputs: inputs),
        "head": FuncNode("head", lambda inputs: inputs),
        END_ID: EndNode.record(END_ID, output_names=[]),
    }
    edges = [Edge(id="start-router", from_=START_ID, to="router")]
    edges.extend(
        Edge(
            id=f"route-{index}",
            from_="router",
            to="head",
            source_handle=f"no-{index}",
        )
        for index in range(64)
    )
    edges.append(
        Edge(id="route-live", from_="router", to="head", source_handle="yes")
    )
    flow = CompiledFlow.from_parts(nodes=nodes, edges=edges)
    state, _ = _route_in_flight(flow)
    original_incoming = flow.incoming
    scans: dict[str, int] = {}

    def counted_incoming(node_id: str):
        scans[node_id] = scans.get(node_id, 0) + 1
        return original_incoming(node_id)

    monkeypatch.setattr(flow, "incoming", counted_incoming)

    prepared = prepare_node_routed(state, "router", "yes")

    assert scans == {"head": 1}
    assert prepared._plan.delta.newly_ready_ids == ("head",)


def test_route_skip_flood_is_iterative_for_a_deep_chain():
    depth = 1500
    nodes = {
        START_ID: StartNode(START_ID),
        "router": FuncNode("router", lambda inputs: inputs),
        "chosen": FuncNode("chosen", lambda inputs: inputs),
        END_ID: EndNode.record(END_ID, output_names=[]),
    }
    nodes.update(
        {
            f"dead-{index}": FuncNode(f"dead-{index}", lambda inputs: inputs)
            for index in range(depth)
        }
    )
    edges = [
        Edge(id="start-router", from_=START_ID, to="router"),
        Edge(
            id="route-dead",
            from_="router",
            to="dead-0",
            source_handle="no",
        ),
        Edge(
            id="route-live",
            from_="router",
            to="chosen",
            source_handle="yes",
        ),
    ]
    edges.extend(
        Edge(
            id=f"dead-edge-{index}",
            from_=f"dead-{index}",
            to=f"dead-{index + 1}",
        )
        for index in range(depth - 1)
    )
    state, log = _route_in_flight(
        CompiledFlow.from_parts(nodes=nodes, edges=edges)
    )

    observers = commit_prepared(
        log,
        state,
        prepare_node_routed(state, "router", "yes"),
    )

    assert len(observers) == depth + 1
    assert observers[1] == NodeSkipped("dead-0")
    assert observers[-1] == NodeSkipped(f"dead-{depth - 1}")
    assert tuple(state.ready) == ("chosen",)


def _wide_ready_route_state(size: int) -> RunState:
    """Return a constant-degree route frontier with unrelated queued work."""
    state, _ = _route_in_flight()
    state.ready.extend(f"unrelated-{index}" for index in range(size))
    return state


def _median_route_reducer_peak(state: RunState) -> float:
    """Return median peak bytes through route preparation and pre-append commit."""
    log = _AllocationStopLog(state)
    prepared = prepare_node_routed(state, "router", "yes")
    with pytest.raises(_AllocationStop):
        commit_prepared(log, state, prepared)
    samples: list[int] = []
    for _ in range(5):
        gc.collect()
        tracemalloc.start()
        prepared = prepare_node_routed(state, "router", "yes")
        try:
            commit_prepared(log, state, prepared)
        except _AllocationStop:
            pass
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        samples.append(peak)
        del prepared
    return median(samples)


def test_route_reducer_allocation_is_independent_of_wide_ready_queue():
    small = _wide_ready_route_state(64)
    large = _wide_ready_route_state(16_384)

    small_peak = _median_route_reducer_peak(small)
    large_peak = _median_route_reducer_peak(large)

    assert large_peak / small_peak <= 2.5, (
        "route reducer allocation scaled with the unrelated ready queue: "
        f"small={small_peak}, large={large_peak}"
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
        "baked": FuncNode(
            "baked",
            lambda inputs: inputs,
            output_contract=ValueContract(list_type),
        ),
        "explicit": FuncNode(
            "explicit",
            lambda inputs: inputs,
            output_contract=ValueContract(list_type),
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


def test_success_validates_ordinary_target_against_output_contract_type():
    """A non-redirected commit uses its own contract's semantic Type."""
    state, log = _redirect_ready()
    filler = state.flow.nodes["filler"]
    filler.commit_as = None
    filler.output_contract = ValueContract(Type.scalar(ValueKind.NUMBER))

    prepared = prepare_node_succeeded(state, "filler", 3)
    commit_prepared(log, state, prepared)

    assert prepared.record.commit_target == "filler"
    assert prepared.record.value == {"kind": "float", "value": 3.0}
    assert state.pool.get("filler") == 3.0


def test_success_rejects_malformed_target_contract_as_a_host_invariant():
    """A corrupt compiled carrier is not node-produced failure data."""
    state, log = _redirect_ready()
    state.flow.nodes["baked"].output_contract = Type.scalar(  # type: ignore[assignment]
        ValueKind.STRING
    )
    before = _state_snapshot(state)
    sequence = log.next_sequence

    with pytest.raises(TypeError, match="exact ValueContract") as caught:
        prepare_node_succeeded(state, "filler", ("valid",))

    assert not isinstance(caught.value, state_module._NodeDataError)
    assert _state_snapshot(state) == before
    assert log.next_sequence == sequence


def test_success_replay_revalidates_target_output_contract_type():
    """Replay accepts the same contract and rejects a changed incompatible contract."""
    live_state, live_log = _redirect_ready()
    commit_prepared(
        live_log,
        live_state,
        prepare_node_succeeded(live_state, "filler", ("valid",)),
    )

    replay_state, replay_log, _ = _replay_outcome_log(_redirect_flow(), live_log)
    assert _reducer_snapshot(replay_state) == _reducer_snapshot(live_state)
    assert tuple(replay_log) == tuple(live_log)

    incompatible = _redirect_flow()
    incompatible.nodes["baked"].output_contract = ValueContract(
        Type.scalar(ValueKind.INTEGER)
    )
    with pytest.raises(ValueError, match="declared type int"):
        _replay_outcome_log(incompatible, live_log)

    malformed = _redirect_flow()
    malformed.nodes["baked"].output_contract = Type.scalar(  # type: ignore[assignment]
        ValueKind.STRING
    )
    with pytest.raises(TypeError, match="exact ValueContract") as caught:
        _replay_outcome_log(malformed, live_log)
    assert not isinstance(caught.value, state_module._NodeDataError)


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

    with pytest.raises(ValueError, match=match) as caught:
        prepare_node_succeeded(
            state,
            "filler",
            output,
            commit_as=commit_as,
        )

    assert isinstance(caught.value, state_module._NodeDataError)
    assert _state_snapshot(state) == before
    assert log.next_sequence == sequence


def test_success_rejects_a_previously_settled_target_edge_before_append():
    state, log = _redirect_ready()
    state.edge_state["baked-after"] = NodeState.TAKEN
    before = _state_snapshot(state)
    sequence = log.next_sequence

    with pytest.raises(ValueError, match="outgoing edge .* must be UNKNOWN") as caught:
        prepare_node_succeeded(state, "filler", ("valid",))

    assert not isinstance(caught.value, state_module._NodeDataError)
    assert _state_snapshot(state) == before
    assert log.next_sequence == sequence


def test_success_treats_an_invalid_baked_target_as_a_host_invariant_error():
    state, log = _redirect_ready()
    state.flow.nodes["filler"].commit_as = "missing"
    before = _state_snapshot(state)

    with pytest.raises(ValueError, match="does not exist") as caught:
        prepare_node_succeeded(state, "filler", ("valid",))

    assert not isinstance(caught.value, state_module._NodeDataError)
    assert _state_snapshot(state) == before
    assert log.next_sequence == 4


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


def _end_and_sibling_flow() -> CompiledFlow:
    """Return a graph whose END, sibling, and queued leaf run concurrently."""
    nodes = {
        START_ID: StartNode(START_ID),
        "sibling": FuncNode("sibling", lambda inputs: inputs),
        "queued": FuncNode("queued", lambda inputs: inputs),
        END_ID: EndNode.record(END_ID, output_names=[]),
    }
    return CompiledFlow.from_parts(
        nodes=nodes,
        edges=[
            Edge(id="start-end", from_=START_ID, to=END_ID),
            Edge(id="start-sibling", from_=START_ID, to="sibling"),
            Edge(id="start-queued", from_=START_ID, to="queued"),
        ],
    )


def _end_committed_with_sibling_in_flight() -> tuple[RunState, MemoryRunLog]:
    """Return a live run whose END value exists while its last sibling is running."""
    state, log = _initialized_and_started(_end_and_sibling_flow())
    _commit_start_success(state, log)
    commit_prepared(log, state, prepare_node_started(state, END_ID))
    commit_prepared(log, state, prepare_node_started(state, "sibling"))
    commit_prepared(
        log,
        state,
        prepare_node_succeeded(state, END_ID, {"terminal": ["stable"]}),
    )
    return state, log


def _end_only_in_flight() -> tuple[RunState, MemoryRunLog]:
    """Return a live run with END as the sole in-flight node and no ready work."""
    nodes = {
        START_ID: StartNode(START_ID),
        END_ID: EndNode.record(END_ID, output_names=[]),
    }
    flow = CompiledFlow.from_parts(
        nodes=nodes,
        edges=[Edge(id="start-end", from_=START_ID, to=END_ID)],
    )
    state, log = _initialized_and_started(flow)
    _commit_start_success(state, log)
    commit_prepared(log, state, prepare_node_started(state, END_ID))
    return state, log


def _terminal_skip_route_flow() -> CompiledFlow:
    """Return a route whose selected control still co-skips two END groups."""
    nodes = {
        START_ID: StartNode(START_ID),
        "router": FuncNode("router", lambda inputs: inputs),
        "dead-optional": FuncNode("dead-optional", lambda inputs: inputs),
        "dead-z": FuncNode("dead-z", lambda inputs: inputs),
        "dead-a": FuncNode("dead-a", lambda inputs: inputs),
        END_ID: EndNode.record(END_ID, output_names=["zeta", "alpha"]),
    }
    return CompiledFlow.from_parts(
        nodes=nodes,
        edges=[
            Edge(id="start-router", from_=START_ID, to="router"),
            Edge(
                id="router-end",
                from_="router",
                to=END_ID,
                source_handle="go",
            ),
            Edge(
                id="router-dead-optional",
                from_="router",
                to="dead-optional",
                source_handle="skip-optional",
            ),
            Edge(
                id="dead-optional-end",
                from_="dead-optional",
                to=END_ID,
                input_group="optional-first",
                optional=True,
            ),
            Edge(
                id="router-dead-z",
                from_="router",
                to="dead-z",
                source_handle="skip-z",
            ),
            Edge(
                id="dead-z-end",
                from_="dead-z",
                to=END_ID,
                input_group="zeta",
            ),
            Edge(
                id="router-dead-a",
                from_="router",
                to="dead-a",
                source_handle="skip-a",
            ),
            Edge(
                id="dead-a-end",
                from_="dead-a",
                to=END_ID,
                input_group="alpha",
            ),
        ],
    )


def _terminal_skip_router_in_flight() -> tuple[RunState, MemoryRunLog]:
    """Return a live run with the terminal-skipping router as its sole work."""
    state, log = _initialized_and_started(_terminal_skip_route_flow())
    _commit_start_success(state, log)
    commit_prepared(log, state, prepare_node_started(state, "router"))
    return state, log


def _replay_outcome_log(
    flow: CompiledFlow,
    source: MemoryRunLog,
) -> tuple[RunState, MemoryRunLog, tuple[tuple[Any, ...], ...]]:
    """Replay one outcome log and retain each transition's derived observers."""
    records = tuple(envelope.record for envelope in source)
    started = records[0]
    assert isinstance(started, RunStartedRecord)
    state = RunState(flow)
    log = MemoryRunLog(_RUN_ID)
    observers = [
        commit_prepared(
            log,
            state,
            prepare_run_started(
                state,
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
    ]
    for record in records[1:]:
        prepared = prepare_record(state, record)
        observers.append(commit_prepared(log, state, prepared))
    return state, log, tuple(observers)


def test_end_output_waits_for_full_quiescence():
    state, log = _end_committed_with_sibling_in_flight()

    assert state.pool.get(END_ID) == {"terminal": ["stable"]}
    assert state.result is None
    assert tuple(state.ready) == ("queued",)
    assert tuple(state.in_flight) == ("sibling",)

    observers = commit_prepared(
        log,
        state,
        prepare_node_succeeded(state, "sibling", {"done": True}),
    )

    assert observers == (NodeSucceeded("sibling", {"done": True}),)
    assert state.result is None
    assert tuple(state.ready) == ("queued",)
    assert tuple(state.in_flight) == ()

    commit_prepared(log, state, prepare_node_started(state, "queued"))
    observers = commit_prepared(
        log,
        state,
        prepare_node_succeeded(state, "queued", {"done": "last"}),
    )

    assert observers == (
        NodeSucceeded("queued", {"done": "last"}),
        RunSucceeded({"terminal": ["stable"]}),
    )
    assert state.result == RunSucceeded({"terminal": ["stable"]})
    assert tuple(state.ready) == ()
    assert tuple(state.in_flight) == ()


def test_newly_ready_successor_prevents_quiescence():
    nodes = {
        START_ID: StartNode(START_ID),
        "worker": FuncNode("worker", lambda inputs: inputs),
        "after": FuncNode("after", lambda inputs: inputs),
        END_ID: EndNode.record(END_ID, output_names=[]),
    }
    flow = CompiledFlow.from_parts(
        nodes=nodes,
        edges=[
            Edge(id="start-end", from_=START_ID, to=END_ID),
            Edge(id="start-worker", from_=START_ID, to="worker"),
            Edge(id="worker-after", from_="worker", to="after"),
        ],
    )
    state, log = _initialized_and_started(flow)
    _commit_start_success(state, log)
    commit_prepared(log, state, prepare_node_started(state, END_ID))
    commit_prepared(log, state, prepare_node_started(state, "worker"))
    commit_prepared(log, state, prepare_node_succeeded(state, END_ID, "done"))

    observers = commit_prepared(
        log,
        state,
        prepare_node_succeeded(state, "worker", "worker-done"),
    )

    assert observers == (NodeSucceeded("worker", "worker-done"),)
    assert state.result is None
    assert tuple(state.ready) == ("after",)
    assert tuple(state.in_flight) == ()


def test_late_sibling_failure_wins():
    state, log = _end_committed_with_sibling_in_flight()

    observers = commit_prepared(
        log,
        state,
        prepare_node_failed(
            state,
            "sibling",
            "late boom",
            error_type="RuntimeError",
        ),
    )

    assert observers == (
        NodeFailed("sibling", "late boom", "RuntimeError"),
        RunFailed("late boom", "RuntimeError"),
    )
    assert state.pool.get(END_ID) == {"terminal": ["stable"]}
    assert state.result == RunFailed("late boom", "RuntimeError")


def test_typed_none_end_membership_succeeds_at_quiescence():
    state, log = _end_only_in_flight()

    prepared = prepare_node_succeeded(state, END_ID, None)

    assert isinstance(prepared._plan.terminal.output_bytes, bytes)
    with pytest.raises(FrozenInstanceError):
        prepared._plan.terminal.output_bytes = b"changed"  # type: ignore[misc]
    assert prepared.observers == (
        NodeSucceeded(END_ID, None),
        RunSucceeded(None),
    )
    observers = commit_prepared(log, state, prepared)

    assert END_ID in state.pool.store
    assert state.pool.get_value(END_ID) is not None
    assert state.pool.get(END_ID) is None
    assert state.result == RunSucceeded(None)
    assert observers == prepared.observers


def test_route_terminal_skip_uses_first_required_end_group_and_no_extra_record():
    state, log = _terminal_skip_router_in_flight()

    prepared = prepare_node_routed(state, "router", "go")

    expected_terminal = RunFailed(
        error="terminal output 'zeta' skipped",
        error_type="TerminalSkipped",
    )
    assert prepared.observers == (
        NodeRouted("router", "go"),
        NodeSkipped("dead-optional"),
        NodeSkipped("dead-z"),
        NodeSkipped(END_ID),
        NodeSkipped("dead-a"),
        expected_terminal,
    )
    observers = commit_prepared(log, state, prepared)

    assert observers == prepared.observers
    assert state.result == expected_terminal
    assert tuple(state.ready) == ()
    assert tuple(state.in_flight) == ()
    assert [type(envelope.record) for envelope in log][-1:] == [NodeRoutedRecord]
    assert len(tuple(log)) == 5


def test_terminal_skip_diagnosis_scans_only_end_beyond_touched_route_joins(
    monkeypatch: pytest.MonkeyPatch,
):
    state, _ = _terminal_skip_router_in_flight()
    incoming = state.flow.incoming
    scanned: list[str] = []

    def counted_incoming(node_id: str) -> list[Edge]:
        scanned.append(node_id)
        return incoming(node_id)

    monkeypatch.setattr(state.flow, "incoming", counted_incoming)

    prepare_node_routed(state, "router", "go")

    assert scanned == [
        END_ID,
        "dead-optional",
        "dead-z",
        "dead-a",
        END_ID,
    ]


@pytest.mark.parametrize("outcome", ["success", "route"])
def test_terminal_derivation_replays_from_causal_records_only(outcome: str):
    if outcome == "success":
        live_state, live_log = _end_only_in_flight()
        expected_observers = commit_prepared(
            live_log,
            live_state,
            prepare_node_succeeded(live_state, END_ID, {"value": [1, 2]}),
        )
        replay_flow = CompiledFlow.from_parts(
            nodes={
                START_ID: StartNode(START_ID),
                END_ID: EndNode.record(END_ID, output_names=[]),
            },
            edges=[Edge(id="start-end", from_=START_ID, to=END_ID)],
        )
    else:
        live_state, live_log = _terminal_skip_router_in_flight()
        expected_observers = commit_prepared(
            live_log,
            live_state,
            prepare_node_routed(live_state, "router", "go"),
        )
        replay_flow = _terminal_skip_route_flow()

    replay_state, replay_log, replay_observers = _replay_outcome_log(
        replay_flow,
        live_log,
    )

    assert replay_observers[-1] == expected_observers
    assert _reducer_snapshot(replay_state) == _reducer_snapshot(live_state)
    assert tuple(replay_log) == tuple(live_log)
    assert all(
        not type(envelope.record).__name__.startswith("RunSucceeded")
        and not type(envelope.record).__name__.startswith("RunFailed")
        for envelope in replay_log
    )


@pytest.mark.parametrize("outcome", ["success", "route"])
def test_terminal_append_failure_leaves_result_frontier_and_pool_unchanged(
    outcome: str,
):
    if outcome == "success":
        state, prior = _end_only_in_flight()
        prepared = prepare_node_succeeded(state, END_ID, {"value": [1, 2]})
    else:
        state, prior = _terminal_skip_router_in_flight()
        prepared = prepare_node_routed(state, "router", "go")
    before = _state_snapshot(state)
    log = _FailingAppendLog(prior)

    with pytest.raises(OSError, match="append failed"):
        commit_prepared(log, state, prepared)

    assert _state_snapshot(state) == before
    assert log.next_sequence == prior.next_sequence


def test_terminal_success_values_are_mutation_isolated():
    state, log = _end_only_in_flight()
    raw = {"nested": [["original"]]}
    prepared = prepare_node_succeeded(state, END_ID, raw)
    raw["nested"][0].append("raw mutation")
    prepared.observers[0].output["nested"][0].append("prepared node mutation")
    prepared.observers[1].output["nested"][0].append("prepared run mutation")

    observers = commit_prepared(log, state, prepared)

    expected = {"nested": [["original"]]}
    assert state.pool.get(END_ID) == expected
    assert state.result == RunSucceeded(expected)
    assert observers == (
        NodeSucceeded(END_ID, expected),
        RunSucceeded(expected),
    )
    assert observers[0].output is not observers[1].output
    assert observers[1].output is not state.result.output
    observers[0].output["nested"][0].append("returned node mutation")
    observers[1].output["nested"][0].append("returned run mutation")
    state.result.output["nested"][0].append("state mutation")
    assert state.pool.get(END_ID) == expected


@pytest.mark.parametrize("outcome", ["success", "route"])
def test_terminal_commit_performs_no_materialization_after_append(
    monkeypatch: pytest.MonkeyPatch,
    outcome: str,
):
    if outcome == "success":
        state, prior = _end_only_in_flight()
        prepared = prepare_node_succeeded(state, END_ID, {"value": [1, 2]})
    else:
        state, prior = _terminal_skip_router_in_flight()
        prepared = prepare_node_routed(state, "router", "go")

    def unexpected(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("terminal commit materialized data after append")

    class PoisonAfterAppendLog(MemoryRunLog):
        """Copy a prefix, then poison terminal work after durable append."""

        def __init__(self, source: MemoryRunLog) -> None:
            super().__init__(source.run_id)
            for envelope in source:
                super().append(envelope.record)

        def append(self, record: Any):
            envelope = super().append(record)
            monkeypatch.setattr(state_module, "RunSucceeded", unexpected)
            monkeypatch.setattr(state_module, "RunFailed", unexpected)
            monkeypatch.setattr(state_module, "_materialize_terminal", unexpected)
            monkeypatch.setattr(state.flow, "incoming", unexpected)
            return envelope

    log = PoisonAfterAppendLog(prior)

    observers = commit_prepared(log, state, prepared)

    assert len(observers) >= 2
    assert state.result is not None


def _failure_flow() -> CompiledFlow:
    """Return a graph with END and two ordinary nodes concurrently in flight."""
    nodes = {
        START_ID: StartNode(START_ID),
        "failing": FuncNode("failing", lambda inputs: inputs),
        "sibling": FuncNode("sibling", lambda inputs: inputs),
        "queued": FuncNode("queued", lambda inputs: inputs),
        END_ID: EndNode.record(END_ID, output_names=[]),
    }
    return CompiledFlow.from_parts(
        nodes=nodes,
        edges=[
            Edge(id="start-end", from_=START_ID, to=END_ID),
            Edge(id="start-failing", from_=START_ID, to="failing"),
            Edge(id="start-sibling", from_=START_ID, to="sibling"),
            Edge(id="start-queued", from_=START_ID, to="queued"),
        ],
    )


def _failure_in_flight() -> tuple[RunState, MemoryRunLog]:
    """Return a state where END has output and one failing node has a live sibling."""
    state, log = _initialized_and_started(_failure_flow())
    _commit_start_success(state, log)
    commit_prepared(log, state, prepare_node_started(state, END_ID))
    commit_prepared(log, state, prepare_node_started(state, "failing"))
    commit_prepared(log, state, prepare_node_started(state, "sibling"))
    commit_prepared(log, state, prepare_node_succeeded(state, END_ID, {}))
    return state, log


def test_failure_is_terminal_and_preserves_diagnostics():
    state, log = _failure_in_flight()
    locator = SourceSpan(node="failing", kind="assert", key="${value > 0}")
    source_range = SourceRange(
        start_line=11,
        start_col=7,
        end_line=11,
        end_col=19,
    )
    pool_before = encode_pool(state.pool)
    edges_before = dict(state.edge_state)
    ready_before = tuple(state.ready)
    sequence_before = state.next_sequence

    prepared = prepare_node_failed(
        state,
        "failing",
        "value must be positive",
        error_type="NodeAssertFailed",
        locator=locator,
        range=source_range,
        traceback="Traceback (most recent call last):\n  boom",
    )

    expected_record = NodeFailedRecord(
        node_id="failing",
        error="value must be positive",
        error_type="NodeAssertFailed",
        locator=locator,
        range=source_range,
        traceback="Traceback (most recent call last):\n  boom",
    )
    expected_result = RunFailed(
        error="value must be positive",
        error_type="NodeAssertFailed",
        locator=locator,
        range=source_range,
        traceback="Traceback (most recent call last):\n  boom",
    )
    assert prepared.record == expected_record
    assert prepared.observers == (
        NodeFailed(
            node_id="failing",
            error=expected_result.error,
            error_type=expected_result.error_type,
            locator=locator,
            range=source_range,
            traceback=expected_result.traceback,
        ),
        expected_result,
    )
    assert state.result is None
    assert state.pool.get(END_ID) == {}

    prepared.observers[0].error = "mutated prepared observer"
    prepared.observers[1].error = "mutated prepared observer"
    observers = commit_prepared(log, state, prepared)

    assert observers == (
        NodeFailed(
            node_id="failing",
            error="value must be positive",
            error_type="NodeAssertFailed",
            locator=locator,
            range=source_range,
            traceback="Traceback (most recent call last):\n  boom",
        ),
        expected_result,
    )
    assert state.result == expected_result
    assert observers[1] is not state.result
    observers[1].error = "mutated returned observer"
    assert state.result.error == "value must be positive"
    assert state.node_state["failing"] is NodeState.TAKEN
    assert state.node_state["sibling"] is NodeState.UNKNOWN
    assert tuple(state.in_flight) == ("sibling",)
    assert tuple(state.ready) == ready_before == ("queued",)
    assert encode_pool(state.pool) == pool_before
    assert state.edge_state == edges_before
    assert state.next_sequence == sequence_before + 1
    assert tuple(log)[-1].record == expected_record


def test_failure_record_replay_reconstructs_identical_state():
    live_state, live_log = _failure_in_flight()
    commit_prepared(
        live_log,
        live_state,
        prepare_node_failed(
            live_state,
            "failing",
            "boom",
            error_type="ValueError",
            locator=SourceSpan(node="failing", kind="field", key="code"),
            range=SourceRange(4, 3, 5, 1),
            traceback="trace",
        ),
    )
    records = tuple(envelope.record for envelope in live_log)
    started = records[0]
    assert isinstance(started, RunStartedRecord)

    replay_state = RunState(_failure_flow())
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


def test_failure_append_failure_leaves_every_state_coordinate_unchanged():
    state, prior = _failure_in_flight()
    prepared = prepare_node_failed(state, "failing", "boom")
    before = _state_snapshot(state)
    log = _FailingAppendLog(prior)

    with pytest.raises(OSError, match="append failed"):
        commit_prepared(log, state, prepared)

    assert _state_snapshot(state) == before
    assert log.next_sequence == prior.next_sequence


def test_failure_requires_exact_diagnostic_carriers_before_append():
    state, log = _failure_in_flight()
    invalid_calls = (
        lambda: prepare_node_failed(state, 1, "boom"),
        lambda: prepare_node_failed(state, "failing", 1),
        lambda: prepare_node_failed(state, "failing", "boom", error_type=1),
        lambda: prepare_node_failed(state, "failing", "boom", locator={}),
        lambda: prepare_node_failed(state, "failing", "boom", range=(1, 1, 1, 2)),
        lambda: prepare_node_failed(state, "failing", "boom", traceback=[]),
    )
    before = _state_snapshot(state)

    for call in invalid_calls:
        with pytest.raises(TypeError):
            call()

    assert _state_snapshot(state) == before
    assert log.next_sequence == state.next_sequence


@pytest.mark.parametrize(
    ("node_id", "match"),
    [("missing", "does not exist"), ("queued", "not in-flight")],
)
def test_failure_requires_an_existing_unknown_in_flight_node(
    node_id: str,
    match: str,
):
    state, log = _failure_in_flight()
    before = _state_snapshot(state)

    with pytest.raises(ValueError, match=match):
        prepare_node_failed(state, node_id, "boom")

    assert _state_snapshot(state) == before
    assert log.next_sequence == state.next_sequence


def test_failure_requires_initialized_nonterminal_unknown_state():
    pristine = RunState(_failure_flow())
    with pytest.raises(ValueError, match="initialized"):
        prepare_node_failed(pristine, "failing", "boom")

    state, log = _failure_in_flight()
    state.node_state["failing"] = NodeState.TAKEN
    before = _state_snapshot(state)
    with pytest.raises(ValueError, match="UNKNOWN disposition"):
        prepare_node_failed(state, "failing", "boom")
    assert _state_snapshot(state) == before
    assert log.next_sequence == state.next_sequence

    state.node_state["failing"] = NodeState.UNKNOWN
    state.result = RunFailed("already terminal")
    before = _state_snapshot(state)
    with pytest.raises(ValueError, match="terminal"):
        prepare_node_failed(state, "failing", "boom")
    assert _state_snapshot(state) == before


def test_failure_replay_rejects_noncanonical_diagnostic_carriers():
    state, log = _failure_in_flight()
    before = _state_snapshot(state)
    record = NodeFailedRecord(
        node_id="failing",
        error="boom",
        locator={"node": "failing", "kind": "field", "key": "code"},  # type: ignore[arg-type]
    )

    with pytest.raises(TypeError, match="locator"):
        prepare_record(state, record)

    assert _state_snapshot(state) == before
    assert log.next_sequence == state.next_sequence


def test_competing_failure_plan_rejects_after_another_same_sequence_outcome():
    state, log = _failure_in_flight()
    failure = prepare_node_failed(state, "failing", "boom")
    sibling_success = prepare_node_succeeded(state, "sibling", {"ok": True})

    commit_prepared(log, state, sibling_success)
    sequence = log.next_sequence
    with pytest.raises(ValueError, match="run-state sequence mismatch"):
        commit_prepared(log, state, failure)

    assert log.next_sequence == sequence
    assert state.result is None
    assert tuple(state.in_flight) == ("failing",)


def test_failure_commit_uses_only_materialized_data_after_append(
    monkeypatch: pytest.MonkeyPatch,
):
    state, prior = _failure_in_flight()
    prepared = prepare_node_failed(
        state,
        "failing",
        "boom",
        error_type="RuntimeError",
    )

    def unexpected(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("failure commit repeated preparation work")

    monkeypatch.setattr(state_module, "_validate_failure_record", unexpected)
    monkeypatch.setattr(state.flow, "incoming", unexpected)
    monkeypatch.setattr(state.flow, "outgoing", unexpected)

    class PoisonAfterAppendLog(MemoryRunLog):
        """Copy the prefix, then poison observer constructors after durable append."""

        def __init__(self, source: MemoryRunLog) -> None:
            super().__init__(source.run_id)
            for envelope in source:
                super().append(envelope.record)

        def append(self, record: Any):
            envelope = super().append(record)
            monkeypatch.setattr(state_module, "NodeFailed", unexpected)
            monkeypatch.setattr(state_module, "RunFailed", unexpected)
            return envelope

    log = PoisonAfterAppendLog(prior)

    observers = commit_prepared(log, state, prepared)

    assert observers == (
        NodeFailed("failing", "boom", "RuntimeError"),
        RunFailed("boom", "RuntimeError"),
    )
    assert state.result == RunFailed("boom", "RuntimeError")
