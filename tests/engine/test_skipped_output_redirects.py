"""Skip propagation through node-owned logical output redirects."""

from __future__ import annotations

from typing import Any

import pytest

from yian.compile.model import (
    CompiledFlow,
    Edge,
    END_ID,
    START_ID,
)
from yian.runtime.common import NodeState
from yian.events import RunFailed, RunSucceeded
from yian.nodes.call import CallNode
from yian.nodes.end import EndNode
from yian.nodes.loop import LoopNode
from yian.nodes.map import MapNode
from yian.nodes.start import StartNode
from yian.runtime.log import (
    MemoryRunLog,
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
    prepare_node_routed,
    prepare_node_started,
    prepare_node_succeeded,
    prepare_record,
    prepare_run_started,
)
from yian.typesys import VariablePool
from tests.engine._fakes import FuncNode


_FINGERPRINT = "8" * 64
_RUN_ID = "skipped-output-redirects"
_CHILD_END = "call/__end__"


def _call_redirect_flow(*, fallback: bool = False) -> CompiledFlow:
    """Return a CALL-shaped child END redirecting output to its logical parent."""
    child_end = EndNode.value(_CHILD_END, output_name=None)
    child_end.commit_as = "call"
    nodes = {
        START_ID: StartNode(START_ID),
        "router": FuncNode("router", lambda inputs: inputs),
        _CHILD_END: child_end,
        "call": CallNode("call", flow_id="child"),
        END_ID: EndNode.value(END_ID, output_name=None),
    }
    edges = [
        Edge(id="start-router", from_=START_ID, to="router"),
        Edge(
            id="router-child-end",
            from_="router",
            to=_CHILD_END,
            source_handle="child",
        ),
        Edge(
            id="call-end",
            from_="call",
            to=END_ID,
            input_group="result",
        ),
    ]
    if fallback:
        nodes["fallback"] = FuncNode("fallback", lambda inputs: inputs)
        edges.insert(1, Edge(id="start-fallback", from_=START_ID, to="fallback"))
        edges.append(
            Edge(
                id="fallback-end",
                from_="fallback",
                to=END_ID,
                input_group="result",
            )
        )
    return CompiledFlow.from_parts(nodes=nodes, edges=edges)


def _map_redirect_flow() -> CompiledFlow:
    """Return one failed MAP item feeding a redirected list collector."""
    child_end_id = "map#0/__end__"
    collector_id = "map/__end__"
    collector = EndNode.list_(collector_id, n=1)
    collector.commit_as = "map"
    return CompiledFlow.from_parts(
        nodes={
            START_ID: StartNode(START_ID),
            "router": FuncNode("router", lambda inputs: inputs),
            child_end_id: EndNode.value(child_end_id, output_name=None),
            collector_id: collector,
            "map": MapNode("map", flow_id="child"),
            END_ID: EndNode.value(END_ID, output_name=None),
        },
        edges=[
            Edge(id="start-router", from_=START_ID, to="router"),
            Edge(
                id="router-map-child-end",
                from_="router",
                to=child_end_id,
                source_handle="item",
            ),
            Edge(
                id="map-child-collector",
                from_=child_end_id,
                to=collector_id,
                input_group="e0",
            ),
            Edge(
                id="map-end",
                from_="map",
                to=END_ID,
                input_group="result",
            ),
        ],
    )


def _loop_redirect_flow() -> CompiledFlow:
    """Return one failed LOOP body feeding a respawned driver."""
    body_end_id = "loop#0/__end__"
    driver_id = "loop~1"
    return CompiledFlow.from_parts(
        nodes={
            START_ID: StartNode(START_ID),
            "router": FuncNode("router", lambda inputs: inputs),
            body_end_id: EndNode.value(body_end_id, output_name=None),
            driver_id: LoopNode(
                driver_id,
                flow_id="body",
                predicate_kind="times",
                times=2,
                max_iters=2,
                iteration=1,
                origin_id="loop",
            ),
            "loop": LoopNode(
                "loop",
                flow_id="body",
                predicate_kind="times",
                times=2,
                max_iters=2,
                origin_id="loop",
            ),
            END_ID: EndNode.value(END_ID, output_name=None),
        },
        edges=[
            Edge(id="start-router", from_=START_ID, to="router"),
            Edge(
                id="router-loop-body-end",
                from_="router",
                to=body_end_id,
                source_handle="body",
            ),
            Edge(
                id="loop-body-driver",
                from_=body_end_id,
                to=driver_id,
                input_group="carried",
            ),
            Edge(
                id="loop-end",
                from_="loop",
                to=END_ID,
                input_group="result",
            ),
        ],
    )


def _nested_call_redirect_flow() -> CompiledFlow:
    """Return two CALL output redirects separated by ordinary child wiring."""
    inner_end_id = "outer/inner/__end__"
    outer_end_id = "outer/__end__"
    inner_end = EndNode.value(inner_end_id, output_name=None)
    inner_end.commit_as = "outer/inner"
    outer_end = EndNode.value(outer_end_id, output_name=None)
    outer_end.commit_as = "outer"
    return CompiledFlow.from_parts(
        nodes={
            START_ID: StartNode(START_ID),
            "router": FuncNode("router", lambda inputs: inputs),
            inner_end_id: inner_end,
            "outer/inner": CallNode("outer/inner", flow_id="inner"),
            outer_end_id: outer_end,
            "outer": CallNode("outer", flow_id="outer"),
            END_ID: EndNode.value(END_ID, output_name=None),
        },
        edges=[
            Edge(id="start-router", from_=START_ID, to="router"),
            Edge(
                id="router-inner-end",
                from_="router",
                to=inner_end_id,
                source_handle="inner",
            ),
            Edge(
                id="inner-call-outer-end",
                from_="outer/inner",
                to=outer_end_id,
                input_group="inner-result",
            ),
            Edge(
                id="outer-call-end",
                from_="outer",
                to=END_ID,
                input_group="result",
            ),
        ],
    )


def _initialized(flow: CompiledFlow) -> tuple[RunState, MemoryRunLog]:
    """Return an initialized state with START durably in flight."""
    pool = VariablePool(start_id=flow.start_id)
    pool.set(flow.start_id, {"seed": True})
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


def _router_in_flight(
    flow: CompiledFlow,
) -> tuple[RunState, MemoryRunLog]:
    """Commit START and durably start the router."""
    state, log = _initialized(flow)
    commit_prepared(
        log,
        state,
        prepare_node_succeeded(state, START_ID, {"seed": True}),
    )
    commit_prepared(log, state, prepare_node_started(state, "router"))
    return state, log


def _state_snapshot(state: RunState) -> tuple[Any, ...]:
    """Return value-only reducer coordinates used by failed preparation tests."""
    return (
        encode_pool(state.pool),
        dict(state.node_state),
        dict(state.edge_state),
        tuple(state.ready),
        tuple(state.in_flight),
        state.result,
        state.next_sequence,
    )


def _replay(
    flow: CompiledFlow,
    source: MemoryRunLog,
) -> tuple[RunState, MemoryRunLog]:
    """Fold one live prefix through the same public reducer preparation APIs."""
    records = tuple(envelope.record for envelope in source)
    started = records[0]
    assert isinstance(started, RunStartedRecord)
    state = RunState(flow)
    log = MemoryRunLog(_RUN_ID)
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
    for record in records[1:]:
        commit_prepared(log, state, prepare_record(state, record))
    return state, log


def test_required_call_child_end_redirect_skips_parent_output_path() -> None:
    state, log = _router_in_flight(_call_redirect_flow())

    commit_prepared(log, state, prepare_node_routed(state, "router", "default"))

    assert state.node_state[_CHILD_END] is NodeState.SKIPPED
    assert state.node_state["call"] is NodeState.UNKNOWN
    assert state.edge_state["call-end"] is NodeState.SKIPPED
    assert state.node_state[END_ID] is NodeState.SKIPPED
    assert isinstance(state.result, RunFailed)


def test_redirected_skip_completes_coalesced_fallback_and_run() -> None:
    state, log = _router_in_flight(_call_redirect_flow(fallback=True))
    commit_prepared(log, state, prepare_node_started(state, "fallback"))
    commit_prepared(
        log,
        state,
        prepare_node_succeeded(state, "fallback", "fallback-value"),
    )

    commit_prepared(log, state, prepare_node_routed(state, "router", "default"))

    assert state.edge_state["call-end"] is NodeState.SKIPPED
    assert state.edge_state["fallback-end"] is NodeState.TAKEN
    assert state.node_state[END_ID] is NodeState.UNKNOWN
    assert tuple(state.ready) == (END_ID,)
    assert state.result is None

    commit_prepared(log, state, prepare_node_started(state, END_ID))
    commit_prepared(
        log,
        state,
        prepare_node_succeeded(state, END_ID, "fallback-value"),
    )

    assert isinstance(state.result, RunSucceeded)


@pytest.mark.parametrize(
    ("target", "error"),
    [
        ("missing", ValueError),
        ("", ValueError),
        (0, TypeError),
    ],
)
def test_malformed_skipped_output_target_leaves_state_unchanged(
    target: Any,
    error: type[Exception],
) -> None:
    state, log = _router_in_flight(_call_redirect_flow())
    state.flow.nodes[_CHILD_END].commit_as = target
    before = _state_snapshot(state)
    sequence = log.next_sequence

    with pytest.raises(error):
        prepare_node_routed(state, "router", "default")

    assert _state_snapshot(state) == before
    assert log.next_sequence == sequence


def test_multiple_redirect_owners_are_rejected_before_skip() -> None:
    """One skipped alias cannot settle a target another alias may still produce."""
    first = FuncNode("first", lambda inputs: inputs)
    first.commit_as = "target"
    second = FuncNode("second", lambda inputs: inputs)
    second.commit_as = "target"
    flow = CompiledFlow.from_parts(
        nodes={
            START_ID: StartNode(START_ID),
            "router": FuncNode("router", lambda inputs: inputs),
            "first": first,
            "second": second,
            "target": FuncNode("target", lambda inputs: inputs),
            END_ID: EndNode.value(END_ID, output_name=None),
        },
        edges=[
            Edge(id="start-router", from_=START_ID, to="router"),
            Edge(
                id="router-first",
                from_="router",
                to="first",
                source_handle="first",
            ),
            Edge(
                id="router-second",
                from_="router",
                to="second",
                source_handle="second",
            ),
            Edge(
                id="target-end",
                from_="target",
                to=END_ID,
                input_group="result",
            ),
        ],
    )
    state, log = _router_in_flight(flow)
    before = _state_snapshot(state)
    sequence = log.next_sequence

    with pytest.raises(ValueError, match="multiple skipped output owners"):
        prepare_node_routed(state, "router", "second")

    assert _state_snapshot(state) == before
    assert log.next_sequence == sequence


def test_live_redirect_target_is_rejected_before_skip() -> None:
    """A logical anchor cannot race a redirected producer for the same output."""
    owner = FuncNode("owner", lambda inputs: inputs)
    owner.commit_as = "target"
    flow = CompiledFlow.from_parts(
        nodes={
            START_ID: StartNode(START_ID),
            "router": FuncNode("router", lambda inputs: inputs),
            "owner": owner,
            "target": FuncNode("target", lambda inputs: inputs),
            END_ID: EndNode.value(END_ID, output_name=None),
        },
        edges=[
            Edge(id="start-router", from_=START_ID, to="router"),
            Edge(id="start-target", from_=START_ID, to="target"),
            Edge(
                id="router-owner",
                from_="router",
                to="owner",
                source_handle="owner",
            ),
            Edge(
                id="target-end",
                from_="target",
                to=END_ID,
                input_group="result",
            ),
        ],
    )
    state, log = _router_in_flight(flow)
    assert tuple(state.ready) == ("target",)
    before = _state_snapshot(state)
    sequence = log.next_sequence

    with pytest.raises(ValueError, match="also executable"):
        prepare_node_routed(state, "router", "default")

    assert _state_snapshot(state) == before
    assert log.next_sequence == sequence


def test_settled_redirect_residue_does_not_own_a_future_output() -> None:
    """Pruned nested-growth residue is inert even if its old anchor is gone."""
    flow = _call_redirect_flow()
    residue = FuncNode("settled-residue", lambda inputs: inputs)
    residue.commit_as = "pruned-anchor"
    flow.nodes[residue.id] = residue
    state, log = _router_in_flight(flow)
    state.node_state[residue.id] = NodeState.TAKEN

    commit_prepared(log, state, prepare_node_routed(state, "router", "default"))

    assert state.edge_state["call-end"] is NodeState.SKIPPED
    assert state.node_state[END_ID] is NodeState.SKIPPED
    assert isinstance(state.result, RunFailed)


def test_map_collector_redirects_skipped_output_to_spawner() -> None:
    state, log = _router_in_flight(_map_redirect_flow())

    commit_prepared(log, state, prepare_node_routed(state, "router", "default"))

    assert state.node_state["map#0/__end__"] is NodeState.SKIPPED
    assert state.edge_state["map-child-collector"] is NodeState.SKIPPED
    assert state.node_state["map/__end__"] is NodeState.SKIPPED
    assert state.node_state["map"] is NodeState.UNKNOWN
    assert state.edge_state["map-end"] is NodeState.SKIPPED
    assert state.node_state[END_ID] is NodeState.SKIPPED


def test_loop_driver_redirects_skipped_output_to_origin() -> None:
    state, log = _router_in_flight(_loop_redirect_flow())

    commit_prepared(log, state, prepare_node_routed(state, "router", "default"))

    assert state.node_state["loop#0/__end__"] is NodeState.SKIPPED
    assert state.edge_state["loop-body-driver"] is NodeState.SKIPPED
    assert state.node_state["loop~1"] is NodeState.SKIPPED
    assert state.node_state["loop"] is NodeState.UNKNOWN
    assert state.edge_state["loop-end"] is NodeState.SKIPPED
    assert state.node_state[END_ID] is NodeState.SKIPPED


def test_nested_redirects_propagate_to_the_top_level_end() -> None:
    state, log = _router_in_flight(_nested_call_redirect_flow())

    commit_prepared(log, state, prepare_node_routed(state, "router", "default"))

    assert state.node_state["outer/inner/__end__"] is NodeState.SKIPPED
    assert state.node_state["outer/inner"] is NodeState.UNKNOWN
    assert state.edge_state["inner-call-outer-end"] is NodeState.SKIPPED
    assert state.node_state["outer/__end__"] is NodeState.SKIPPED
    assert state.node_state["outer"] is NodeState.UNKNOWN
    assert state.edge_state["outer-call-end"] is NodeState.SKIPPED
    assert state.node_state[END_ID] is NodeState.SKIPPED


def test_redirected_skip_replays_without_a_terminal_record() -> None:
    live_state, live_log = _router_in_flight(_nested_call_redirect_flow())
    commit_prepared(
        live_log,
        live_state,
        prepare_node_routed(live_state, "router", "default"),
    )

    record_types = [type(envelope.record) for envelope in live_log]
    assert record_types == [
        RunStartedRecord,
        NodeStartedRecord,
        NodeSucceededRecord,
        NodeStartedRecord,
        NodeRoutedRecord,
    ]

    replay_state, replay_log = _replay(_nested_call_redirect_flow(), live_log)

    assert _state_snapshot(replay_state) == _state_snapshot(live_state)
    assert tuple(replay_log) == tuple(live_log)
