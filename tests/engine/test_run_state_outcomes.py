"""Transactional node-outcome tests for the authoritative run-state reducer."""

from __future__ import annotations

import gc
import tracemalloc
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from statistics import median
from typing import Any

import pytest

import yian.runtime.state as state_module
from yian.compile.model import (
    CompiledFlow,
    Edge,
    END_ID,
    Flow,
    START_ID,
)
from yian.runtime.common import NodeState
from yian.diagnostics import Failure, OriginRef
from yian.events import (
    NodeFailed,
    NodeRouted,
    NodeSkipped,
    NodeSucceeded,
    RunFailed,
    RunSucceeded,
)
from yian.nodes.base import Grow, Node, NodeKind
from yian.nodes.code import CodeNode
from yian.nodes.contracts import PartialRecordContract, ValueContract
from yian.nodes.end import EndNode
from yian.nodes.start import StartNode
from yian.runtime.log import (
    MemoryRunLog,
    NodeFailedRecord,
    NodeRoutedRecord,
    NodeStartedRecord,
    NodeSucceededRecord,
    RunStartedRecord,
    decode_pool,
    encode_pool,
)
from yian.runtime.state import (
    RunStartData,
    RunState,
    commit_prepared,
    prepare_expanded,
    prepare_node_failed,
    prepare_node_routed,
    prepare_node_started,
    prepare_node_succeeded,
    prepare_record,
    prepare_run_started,
)
from yian.typesys import Type, ValueKind, VariablePool
from tests.engine._fakes import FuncNode
from tests.engine._graph_builder import _forward_capture


_FINGERPRINT = "b" * 64
_RUN_ID = "outcome-run"


class _SpecialOrigin(OriginRef):
    """Valid front-end-specific origin used to test durable normalization."""


def _node_failure(
    node_id: str,
    message: str = "boom",
    *,
    cause_type: str | None = None,
    traceback: str | None = None,
) -> Failure:
    """Return one canonical node-execution failure for reducer tests."""
    return Failure(
        code="node.execute",
        message=message,
        node_id=node_id,
        cause_type=cause_type,
        traceback=traceback,
    )


def _fanout_flow() -> CompiledFlow:
    """Return a complete graph whose START success makes two nodes ready in order."""
    nodes = {
        START_ID: StartNode(START_ID),
        "first": FuncNode("first", lambda inputs: inputs),
        "second": FuncNode("second", lambda inputs: inputs),
        END_ID: EndNode.value(END_ID, output_name=None),
    }
    return CompiledFlow.from_parts(
        nodes=nodes,
        edges=[
            Edge(id="start-first", from_=START_ID, to="first"),
            Edge(id="start-second", from_=START_ID, to="second"),
        ],
    )


def _initialized_and_started(
    flow: CompiledFlow, *, learnable: bool = False,
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
                learnable=learnable,
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
            forward_capture=_forward_capture(state.flow.nodes[START_ID], {}) if state.learnable else None,
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
        END_ID: EndNode.value(END_ID, output_name=None),
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
        ("default", (None, None)),
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
        END_ID: EndNode.value(END_ID, output_name=None),
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
        END_ID: EndNode.value(END_ID, output_name=None),
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
        END_ID: EndNode.value(END_ID, output_name=None),
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
    [("", ValueError), (7, TypeError), (False, TypeError)],
)
def test_route_rejects_malformed_source_handle_as_a_host_invariant(
    source_handle: Any,
    error_type: type[Exception],
):
    state, log = _route_in_flight()
    outgoing = state.flow._outgoing["router"]
    selected = next(edge for edge in outgoing if edge.source_handle == "yes")
    malformed = replace(
        selected,
        source_handle=source_handle,
    )
    outgoing[outgoing.index(selected)] = malformed
    state.flow.edges[state.flow.edges.index(selected)] = malformed
    before = _state_snapshot(state)

    with pytest.raises(error_type, match="source_handle") as caught:
        prepare_node_routed(state, "router", "yes")

    assert not isinstance(caught.value, state_module._NodeDataError)
    assert _state_snapshot(state) == before
    assert log.next_sequence == 4


def test_route_rejects_stale_outgoing_adjacency_as_a_host_invariant():
    state, log = _route_in_flight()
    state.flow._outgoing["router"] = []
    before = _state_snapshot(state)

    with pytest.raises(ValueError, match="invalid outgoing adjacency") as caught:
        prepare_node_routed(state, "router", "missing")

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
        END_ID: EndNode.value(END_ID, output_name=None),
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
        END_ID: EndNode.value(END_ID, output_name=None),
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
        "terminal_unreachable_failure": state.terminal_unreachable_failure,
        "depth": dict(state.depth),
        "growth_owner": dict(state.growth_owner),
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


def test_success_replay_revalidates_partial_record_contract() -> None:
    """A forged durable success cannot bypass field-local output guarantees."""
    flow = _fanout_flow()
    flow.nodes[START_ID].output_contract = PartialRecordContract(
        fields=(
            ("items", ValueContract(Type(kind=ValueKind.LIST_INTEGER))),
            ("note", None),
        )
    )
    state, _log = _initialized_and_started(flow)
    record = NodeSucceededRecord(
        node_id=START_ID,
        commit_target=START_ID,
        value={
            "kind": "object",
            "value": {"items": ["bad"], "note": "opaque"},
        },
    )

    with pytest.raises(ValueError, match="declared"):
        prepare_record(state, record)


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
        END_ID: EndNode.value(END_ID, output_name=None),
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
        END_ID: EndNode.value(END_ID, output_name=None),
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
    flow = CompiledFlow.from_parts(
        nodes={
            START_ID: StartNode(START_ID),
            "filler": FuncNode(
                "filler",
                lambda inputs: inputs,
                output_contract=ValueContract(Type.scalar(ValueKind.NUMBER)),
            ),
            "after": FuncNode("after", lambda inputs: inputs),
            END_ID: EndNode.value(END_ID, output_name=None),
        },
        edges=[
            Edge(id="start-filler", from_=START_ID, to="filler"),
            Edge(id="filler-after", from_="filler", to="after"),
        ],
    )
    state, log = _initialized_and_started(flow)
    _commit_start_success(state, log)
    commit_prepared(log, state, prepare_node_started(state, "filler"))

    prepared = prepare_node_succeeded(state, "filler", 3)
    commit_prepared(log, state, prepared)

    assert prepared.record.commit_target == "filler"
    assert prepared.record.value == {"kind": "float", "value": 3.0}
    assert state.pool.get("filler") == 3.0
    assert tuple(state.ready) == ("after",)


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
        END_ID: EndNode.value(END_ID, output_name=None),
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
        END_ID: EndNode.value(END_ID, output_name=None),
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
        END_ID: EndNode.value(END_ID, output_name=None),
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
        END_ID: EndNode.value(END_ID, output_name=None),
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
        END_ID: EndNode.value(END_ID, output_name=None),
    }
    flow = CompiledFlow.from_parts(
        nodes=nodes,
        edges=[Edge(id="start-end", from_=START_ID, to=END_ID)],
    )
    state, log = _initialized_and_started(flow)
    _commit_start_success(state, log)
    commit_prepared(log, state, prepare_node_started(state, END_ID))
    return state, log


def _terminal_skip_route_flow(
    *,
    router_origin: OriginRef | None = None,
    sibling: bool = False,
) -> CompiledFlow:
    """Return a route whose selected control still co-skips two END groups."""
    nodes = {
        START_ID: StartNode(START_ID),
        "router": FuncNode(
            "router",
            lambda inputs: inputs,
            origin=router_origin,
        ),
        "dead-optional": FuncNode("dead-optional", lambda inputs: inputs),
        "dead-z": FuncNode("dead-z", lambda inputs: inputs),
        "dead-a": FuncNode("dead-a", lambda inputs: inputs),
        END_ID: EndNode.record(END_ID, output_names=["zeta", "alpha"]),
    }
    edges = [
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
    ]
    if sibling:
        nodes["unrelated"] = FuncNode("unrelated", lambda inputs: inputs)
        edges.append(Edge(id="start-unrelated", from_=START_ID, to="unrelated"))
    return CompiledFlow.from_parts(nodes=nodes, edges=edges)


def _terminal_skip_router_in_flight() -> tuple[RunState, MemoryRunLog]:
    """Return a live run with the terminal-skipping router as its sole work."""
    state, log = _initialized_and_started(_terminal_skip_route_flow())
    _commit_start_success(state, log)
    commit_prepared(log, state, prepare_node_started(state, "router"))
    return state, log


def _terminal_skip_with_sibling_in_flight() -> tuple[RunState, MemoryRunLog]:
    """Return a causal router and unrelated sibling durably in flight."""
    flow = _terminal_skip_route_flow(
        router_origin=OriginRef("router-site"),
        sibling=True,
    )
    state, log = _initialized_and_started(flow)
    _commit_start_success(state, log)
    commit_prepared(log, state, prepare_node_started(state, "router"))
    commit_prepared(log, state, prepare_node_started(state, "unrelated"))
    return state, log


class _OutcomeSpawner(Node):
    """Minimal spawner for chronological terminal-latch pruning coverage."""

    kind = NodeKind.CALL
    is_spawner = True
    grow_depth_delta = 0


def _terminal_prune_parent_flow() -> CompiledFlow:
    """Return static creator/pruner work with one required global output."""
    nodes = {
        START_ID: StartNode(START_ID),
        "creator": _OutcomeSpawner("creator"),
        "pruner": _OutcomeSpawner("pruner"),
        END_ID: EndNode.value(END_ID, output_name="result"),
    }
    return CompiledFlow.from_parts(
        nodes=nodes,
        edges=[
            Edge(id="start-creator", from_=START_ID, to="creator"),
            Edge(id="start-pruner", from_=START_ID, to="pruner"),
            Edge(
                id="creator-end",
                from_="creator",
                to=END_ID,
                input_group="result",
            ),
        ],
        wiring={node_id: {} for node_id in nodes},
    )


def _terminal_router_fragment(spawner_id: str) -> Flow:
    """Return a dynamic route whose completion skip reaches global END."""
    start_id = f"{spawner_id}/child/start"
    router_id = f"{spawner_id}/child/router"
    dead_id = f"{spawner_id}/child/dead"
    end_id = f"{spawner_id}/child/end"
    end = EndNode.value(end_id, output_name="zeta")
    end.commit_as = spawner_id
    router = CodeNode(router_id, code="return inputs")
    router.origin = OriginRef("dynamic-router-site")
    nodes = {
        start_id: StartNode(start_id),
        router_id: router,
        dead_id: CodeNode(dead_id, code="return inputs"),
        end_id: end,
    }
    return Flow(
        nodes=nodes,
        edges=[
            Edge(id=f"{start_id}-router", from_=start_id, to=router_id),
            Edge(
                id=f"{router_id}-end",
                from_=router_id,
                to=end_id,
                source_handle="go",
            ),
            Edge(
                id=f"{router_id}-dead",
                from_=router_id,
                to=dead_id,
                source_handle="skip",
            ),
            Edge(
                id=f"{dead_id}-end",
                from_=dead_id,
                to=end_id,
                input_group="zeta",
            ),
        ],
        wiring={node_id: {} for node_id in nodes},
        start_id=start_id,
        end_id=end_id,
    )


def _terminal_pruner_fragment(spawner_id: str) -> Flow:
    """Return a minimal completion fragment for the unrelated pruner."""
    start_id = f"{spawner_id}/child/start"
    end_id = f"{spawner_id}/child/end"
    end = EndNode.value(end_id, output_name=None)
    end.commit_as = spawner_id
    nodes = {start_id: StartNode(start_id), end_id: end}
    return Flow(
        nodes=nodes,
        edges=[
            Edge(
                id=f"{start_id}-end",
                from_=start_id,
                to=end_id,
                ordering=True,
            )
        ],
        wiring={node_id: {} for node_id in nodes},
        start_id=start_id,
        end_id=end_id,
    )


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
        END_ID: EndNode.value(END_ID, output_name=None),
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
    failure = _node_failure(
        "sibling",
        "late boom",
        cause_type="RuntimeError",
    )

    observers = commit_prepared(
        log,
        state,
        prepare_node_failed(state, failure),
    )

    assert observers == (
        NodeFailed(failure),
        RunFailed(failure),
    )
    assert state.pool.get(END_ID) == {"terminal": ["stable"]}
    assert state.result == RunFailed(failure)


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

    expected_failure = Failure(
        code="graph.terminal_unreachable",
        message="terminal output 'zeta' skipped",
        node_id="router",
    )
    failure = prepared.record.terminal_failure
    assert failure is prepared._plan.delta.terminal_unreachable_failure
    assert failure == expected_failure
    expected_terminal = RunFailed(expected_failure)
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
    assert state.terminal_unreachable_failure == expected_failure
    assert state.result.failure is state.terminal_unreachable_failure
    assert tuple(state.ready) == ()
    assert tuple(state.in_flight) == ()
    assert [type(envelope.record) for envelope in log][-1:] == [NodeRoutedRecord]
    assert tuple(log)[-1].record.terminal_failure == expected_failure
    assert len(tuple(log)) == 5


def test_route_latches_causal_terminal_failure_until_sibling_quiescence():
    state, log = _terminal_skip_with_sibling_in_flight()
    outer_origin = OriginRef("outer-site")
    state.flow.nodes["router"].origin_context = (outer_origin,)

    prepared = prepare_node_routed(state, "router", "go")
    failure = prepared._plan.delta.terminal_unreachable_failure

    assert failure == Failure(
        code="graph.terminal_unreachable",
        message="terminal output 'zeta' skipped",
        node_id="router",
        trace=(outer_origin, OriginRef("router-site")),
    )
    assert failure.trace == (
        outer_origin,
        state.flow.nodes["router"].origin,
    )
    assert prepared.record.terminal_failure is failure
    assert prepared._plan.terminal is None
    assert all(not isinstance(event, RunFailed) for event in prepared.observers)
    assert state.terminal_unreachable_failure is None

    route_observers = commit_prepared(log, state, prepared)

    assert route_observers == prepared.observers
    assert state.terminal_unreachable_failure is failure
    assert state.result is None

    final = prepare_node_succeeded(state, "unrelated", {"done": True})
    assert final._plan.terminal.failure is failure
    assert final.observers == (
        NodeSucceeded("unrelated", {"done": True}),
        RunFailed(failure),
    )

    observers = commit_prepared(log, state, final)

    assert observers[-1].failure is failure
    assert state.result.failure is failure
    assert state.terminal_unreachable_failure is failure
    assert not any(
        isinstance(envelope.record, NodeFailedRecord) for envelope in log
    )
    assert [type(envelope.record) for envelope in log][-2:] == [
        NodeRoutedRecord,
        NodeSucceededRecord,
    ]


def test_causal_route_canonicalizes_origin_subclass_for_wire():
    flow = _terminal_skip_route_flow(
        router_origin=_SpecialOrigin("special-router-site")
    )
    state, log = _initialized_and_started(flow)
    _commit_start_success(state, log)
    commit_prepared(log, state, prepare_node_started(state, "router"))

    prepared = prepare_node_routed(state, "router", "go")
    failure = prepared.record.terminal_failure

    assert failure is not None
    assert failure.trace == (OriginRef("special-router-site"),)
    assert type(failure.trace[0]) is OriginRef
    assert failure is prepared._plan.delta.terminal_unreachable_failure

    commit_prepared(log, state, prepared)

    assert state.result is not None
    assert state.result.failure is failure
    logged = tuple(log)[-1].record.terminal_failure
    assert logged == failure
    assert type(logged.trace[0]) is OriginRef


def test_explicit_failure_wins_after_terminal_unreachable_is_latched():
    state, log = _terminal_skip_with_sibling_in_flight()
    route = prepare_node_routed(state, "router", "go")
    route_failure = route._plan.delta.terminal_unreachable_failure
    commit_prepared(log, state, route)
    explicit = _node_failure(
        "unrelated",
        "explicit failure",
        cause_type="RuntimeError",
    )

    prepared = prepare_node_failed(state, explicit)
    failure = prepared.record.failure
    observers = commit_prepared(log, state, prepared)

    assert observers == (NodeFailed(failure), RunFailed(failure))
    assert observers[0].failure is failure
    assert observers[1].failure is failure
    assert state.result.failure is failure
    assert failure == explicit
    assert failure is not route_failure
    assert state.terminal_unreachable_failure is route_failure


def test_terminal_unreachable_latch_is_installed_only_after_route_append():
    state, prior = _terminal_skip_with_sibling_in_flight()
    prepared = prepare_node_routed(state, "router", "go")
    before = _state_snapshot(state)
    log = _FailingAppendLog(prior)

    with pytest.raises(OSError, match="append failed"):
        commit_prepared(log, state, prepared)

    assert _state_snapshot(state) == before
    assert state.terminal_unreachable_failure is None
    assert log.next_sequence == prior.next_sequence


def test_causal_route_replay_requires_its_persisted_terminal_failure():
    state, log = _terminal_skip_router_in_flight()
    before = _state_snapshot(state)

    with pytest.raises(ValueError, match="requires a persisted terminal failure"):
        prepare_record(state, NodeRoutedRecord(node_id="router", handle="go"))

    assert _state_snapshot(state) == before
    assert log.next_sequence == state.next_sequence


def test_noncausal_route_replay_rejects_a_terminal_failure():
    state, log = _route_in_flight()
    failure = Failure(
        code="graph.terminal_unreachable",
        message="terminal output None skipped",
        node_id="router",
    )
    before = _state_snapshot(state)

    with pytest.raises(ValueError, match="present on a noncausal route"):
        prepare_record(
            state,
            NodeRoutedRecord(
                node_id="router",
                handle="yes",
                terminal_failure=failure,
            ),
        )

    assert _state_snapshot(state) == before
    assert log.next_sequence == state.next_sequence


@pytest.mark.parametrize(
    ("failure", "match"),
    [
        (
            Failure("node.execute", "terminal output 'zeta' skipped", "router"),
            "wrong failure code",
        ),
        (
            Failure(
                "graph.terminal_unreachable",
                "terminal output 'zeta' skipped",
                "other",
            ),
            "wrong causal node ID",
        ),
        (
            Failure("graph.terminal_unreachable", "wrong", "router"),
            "wrong message",
        ),
        (
            Failure(
                "graph.terminal_unreachable",
                "terminal output 'zeta' skipped",
                "router",
                cause_type="RuntimeError",
            ),
            "exception diagnostics",
        ),
        (
            Failure(
                "graph.terminal_unreachable",
                "terminal output 'zeta' skipped",
                "router",
                traceback="trace",
            ),
            "exception diagnostics",
        ),
    ],
    ids=["code", "node", "message", "cause", "traceback"],
)
def test_causal_route_replay_rejects_mismatched_terminal_failure(
    failure: Failure,
    match: str,
):
    state, log = _terminal_skip_router_in_flight()
    before = _state_snapshot(state)

    with pytest.raises(ValueError, match=match):
        prepare_record(
            state,
            NodeRoutedRecord(
                node_id="router",
                handle="go",
                terminal_failure=failure,
            ),
        )

    assert _state_snapshot(state) == before
    assert log.next_sequence == state.next_sequence


def test_quiescence_without_retained_causal_failure_is_a_graph_fault():
    state, log = _terminal_skip_with_sibling_in_flight()
    commit_prepared(log, state, prepare_node_routed(state, "router", "go"))
    state.terminal_unreachable_failure = None
    before = _state_snapshot(state)

    with pytest.raises(
        ValueError,
        match="quiesced without END output or a causal terminal failure",
    ):
        prepare_node_succeeded(state, "unrelated", {"done": True})

    assert _state_snapshot(state) == before
    assert log.next_sequence == state.next_sequence


def test_terminal_unreachable_latch_replays_before_and_at_quiescence():
    live_state, live_log = _terminal_skip_with_sibling_in_flight()
    live_state.flow.nodes["router"].origin_context = (OriginRef("outer-site"),)
    commit_prepared(
        live_log,
        live_state,
        prepare_node_routed(live_state, "router", "go"),
    )
    replay_flow = _terminal_skip_route_flow(
        router_origin=OriginRef("rebased-router-site"),
        sibling=True,
    )
    replay_flow.nodes["router"].origin_context = (
        OriginRef("rebased-outer-site"),
    )

    prefix_state, prefix_log, prefix_observers = _replay_outcome_log(
        replay_flow,
        live_log,
    )

    assert prefix_observers[-1][-1] == NodeSkipped("dead-a")
    assert prefix_state.result is None
    assert prefix_state.terminal_unreachable_failure == (
        live_state.terminal_unreachable_failure
    )
    assert _reducer_snapshot(prefix_state) == _reducer_snapshot(live_state)
    assert tuple(prefix_log) == tuple(live_log)

    live_observers = commit_prepared(
        live_log,
        live_state,
        prepare_node_succeeded(live_state, "unrelated", {"done": True}),
    )
    final_replay_flow = _terminal_skip_route_flow(
        router_origin=OriginRef("rebased-router-site"),
        sibling=True,
    )
    final_replay_flow.nodes["router"].origin_context = (
        OriginRef("rebased-outer-site"),
    )
    final_state, final_log, final_observers = _replay_outcome_log(
        final_replay_flow,
        live_log,
    )

    assert final_observers[-1] == live_observers
    assert _reducer_snapshot(final_state) == _reducer_snapshot(live_state)
    assert tuple(final_log) == tuple(live_log)
    assert final_state.result.failure is final_state.terminal_unreachable_failure
    assert final_state.result.failure.node_id == "router"
    assert final_state.result.failure.trace == (
        OriginRef("outer-site"),
        OriginRef("router-site"),
    )


@pytest.mark.parametrize("learnable", [False, True])
def test_terminal_unreachable_latch_survives_pruning_its_dynamic_cause(learnable):
    state, log = _initialized_and_started(_terminal_prune_parent_flow(), learnable=learnable)
    _commit_start_success(state, log)
    commit_prepared(log, state, prepare_node_started(state, "creator"))
    commit_prepared(log, state, prepare_node_started(state, "pruner"))
    commit_prepared(
        log,
        state,
        prepare_expanded(
            state,
            "creator",
            Grow(seed={}),
            lambda spawner, _seed: _terminal_router_fragment(spawner.id),
            forward_capture=_forward_capture(state.flow.nodes["creator"], {}) if learnable else None,
        ),
    )
    router_id = "creator/child/router"
    commit_prepared(
        log,
        state,
        prepare_node_started(state, "creator/child/start"),
    )
    commit_prepared(
        log,
        state,
        prepare_node_succeeded(state, "creator/child/start", {},
                               forward_capture=_forward_capture(state.flow.nodes["creator/child/start"], {}) if learnable else None),
    )
    commit_prepared(log, state, prepare_node_started(state, router_id))

    route = prepare_node_routed(state, router_id, "go",
                               forward_capture=_forward_capture(state.flow.nodes[router_id], {}) if learnable else None)
    failure = route._plan.delta.terminal_unreachable_failure
    commit_prepared(log, state, route)

    assert failure == Failure(
        code="graph.terminal_unreachable",
        message="terminal output 'result' skipped",
        node_id=router_id,
        trace=(OriginRef("dynamic-router-site"),),
    )
    assert state.result is None
    assert state.terminal_unreachable_failure is failure
    assert router_id not in state.initial_static_ids

    commit_prepared(
        log,
        state,
        prepare_expanded(
            state,
            "pruner",
            Grow(seed={}, prune=frozenset({router_id})),
            lambda spawner, _seed: _terminal_pruner_fragment(spawner.id),
            forward_capture=_forward_capture(state.flow.nodes["pruner"], {}) if learnable else None,
        ),
    )

    assert router_id not in state.flow.nodes
    assert state.terminal_unreachable_failure is failure
    assert state.result is None

    commit_prepared(
        log,
        state,
        prepare_node_started(state, "pruner/child/start"),
    )
    commit_prepared(
        log,
        state,
        prepare_node_succeeded(state, "pruner/child/start", {},
                               forward_capture=_forward_capture(state.flow.nodes["pruner/child/start"], {}) if learnable else None),
    )
    commit_prepared(
        log,
        state,
        prepare_node_started(state, "pruner/child/end"),
    )
    observers = commit_prepared(
        log,
        state,
        prepare_node_succeeded(state, "pruner/child/end", {},
                               forward_capture=_forward_capture(state.flow.nodes["pruner/child/end"], {}) if learnable else None),
    )

    assert observers[-1] == RunFailed(failure)
    assert observers[-1].failure is failure
    assert state.result.failure is failure
    assert state.terminal_unreachable_failure is failure

    if learnable:
        from yian.runtime.backward import compute_backward

        causal = next(item for item in state.trace.occurrences if item.node_id == router_id)
        assert state.trace.terminal.source == causal.source
        root, results = compute_backward(state.trace, "review pruned failure")
        assert root.input_feedback == {}
        assert next(item for item in results if item.source == causal.source).invoked


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
                END_ID: EndNode.value(END_ID, output_name=None),
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
        END_ID: EndNode.value(END_ID, output_name=None),
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
    failure = _node_failure(
        "failing",
        "value must be positive",
        cause_type="NodeAssertFailed",
        traceback="Traceback (most recent call last):\n  boom",
    )
    pool_before = encode_pool(state.pool)
    edges_before = dict(state.edge_state)
    ready_before = tuple(state.ready)
    sequence_before = state.next_sequence

    prepared = prepare_node_failed(state, failure)

    expected_record = NodeFailedRecord(failure=failure)
    authority = prepared.record.failure
    expected_result = RunFailed(authority)
    assert prepared.record == expected_record
    assert prepared._plan.failure is authority
    assert prepared.observers == (
        NodeFailed(authority),
        expected_result,
    )
    assert prepared.observers[0].failure is authority
    assert prepared.observers[1].failure is authority
    assert state.result is None
    assert state.pool.get(END_ID) == {}

    prepared.observers[0].failure = _node_failure("failing", "mutated")
    prepared.observers[1].failure = _node_failure("failing", "mutated")
    observers = commit_prepared(log, state, prepared)

    assert observers == (
        NodeFailed(authority),
        expected_result,
    )
    assert state.result == expected_result
    assert observers[1] is not state.result
    assert observers[0].failure is authority
    assert observers[1].failure is authority
    assert state.result.failure is authority
    observers[1].failure = _node_failure("failing", "mutated returned observer")
    assert state.result.failure.message == "value must be positive"
    assert state.node_state["failing"] is NodeState.TAKEN
    assert state.node_state["sibling"] is NodeState.UNKNOWN
    assert tuple(state.in_flight) == ("sibling",)
    assert tuple(state.ready) == ready_before == ("queued",)
    assert encode_pool(state.pool) == pool_before
    assert state.edge_state == edges_before
    assert state.next_sequence == sequence_before + 1
    assert tuple(log)[-1].record == expected_record


def test_failure_transition_canonicalizes_origin_subclass_for_wire():
    state, log = _failure_in_flight()
    supplied = Failure(
        code="node.execute",
        message="boom",
        node_id="failing",
        trace=(_SpecialOrigin("special-failure-site"),),
    )

    prepared = prepare_node_failed(state, supplied)
    failure = prepared.record.failure

    assert failure.trace == (OriginRef("special-failure-site"),)
    assert type(failure.trace[0]) is OriginRef
    assert failure is prepared._plan.failure
    assert prepared.observers[0].failure is failure
    assert prepared.observers[1].failure is failure

    commit_prepared(log, state, prepared)

    assert state.result is not None
    assert state.result.failure is failure
    logged = tuple(log)[-1].record.failure
    assert logged == failure
    assert type(logged.trace[0]) is OriginRef


def test_failure_record_replay_reconstructs_identical_state():
    live_state, live_log = _failure_in_flight()
    commit_prepared(
        live_log,
        live_state,
        prepare_node_failed(
            live_state,
            _node_failure(
                "failing",
                cause_type="ValueError",
                traceback="trace",
            ),
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
    prepared = prepare_node_failed(state, _node_failure("failing"))
    before = _state_snapshot(state)
    log = _FailingAppendLog(prior)

    with pytest.raises(OSError, match="append failed"):
        commit_prepared(log, state, prepared)

    assert _state_snapshot(state) == before
    assert log.next_sequence == prior.next_sequence


def test_failure_requires_exact_diagnostic_carriers_before_append():
    state, log = _failure_in_flight()
    invalid_calls = (
        lambda: prepare_node_failed(state, 1),
        lambda: prepare_node_failed(
            state,
            Failure("node.execute", "boom", None),
        ),
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
        prepare_node_failed(state, _node_failure(node_id))

    assert _state_snapshot(state) == before
    assert log.next_sequence == state.next_sequence


def test_failure_requires_initialized_nonterminal_unknown_state():
    pristine = RunState(_failure_flow())
    with pytest.raises(ValueError, match="initialized"):
        prepare_node_failed(pristine, _node_failure("failing"))

    state, log = _failure_in_flight()
    state.node_state["failing"] = NodeState.TAKEN
    before = _state_snapshot(state)
    with pytest.raises(ValueError, match="UNKNOWN disposition"):
        prepare_node_failed(state, _node_failure("failing"))
    assert _state_snapshot(state) == before
    assert log.next_sequence == state.next_sequence

    state.node_state["failing"] = NodeState.UNKNOWN
    state.result = RunFailed(_node_failure("failing", "already terminal"))
    before = _state_snapshot(state)
    with pytest.raises(ValueError, match="terminal"):
        prepare_node_failed(state, _node_failure("failing"))
    assert _state_snapshot(state) == before


def test_failure_replay_rejects_a_non_failure_authority():
    state, log = _failure_in_flight()
    before = _state_snapshot(state)
    record = NodeFailedRecord(
        failure=object(),  # type: ignore[arg-type]
    )

    with pytest.raises(TypeError, match="failure"):
        prepare_record(state, record)

    assert _state_snapshot(state) == before
    assert log.next_sequence == state.next_sequence


def test_failure_record_rejects_derived_terminal_unreachable_code():
    state, log = _failure_in_flight()
    failure = Failure(
        code="graph.terminal_unreachable",
        message="terminal output 'result' skipped",
        node_id="failing",
    )
    before = _state_snapshot(state)
    calls = (
        lambda: prepare_node_failed(state, failure),
        lambda: prepare_record(state, NodeFailedRecord(failure=failure)),
    )

    for call in calls:
        with pytest.raises(ValueError, match="derived from a route record"):
            call()

    assert _state_snapshot(state) == before
    assert log.next_sequence == state.next_sequence


def test_competing_failure_plan_rejects_after_another_same_sequence_outcome():
    state, log = _failure_in_flight()
    failure = prepare_node_failed(state, _node_failure("failing"))
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
    failure = _node_failure("failing", cause_type="RuntimeError")
    prepared = prepare_node_failed(state, failure)

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
        NodeFailed(failure),
        RunFailed(failure),
    )
    assert state.result == RunFailed(failure)
