"""Resume of unfinished agent invocations through the append-before-apply reducer."""

from copy import deepcopy
from dataclasses import replace

import pytest

import yian.memory.contracts as c
from yian.compile.model import CompiledFlow, Edge, END_ID, START_ID
from yian.nodes.end import EndNode
from yian.nodes.start import StartNode
from yian.runtime.contracts import ForwardCapture
from yian.runtime import state as reducer
from yian.runtime.log import MemoryRunLog
from yian.expr.lineage import ValueRoute
from yian.typesys import VariablePool
from tests.engine._fakes import FuncNode


def _flow(*, sibling=False):
    node = FuncNode("agent", lambda _: pytest.fail("replay invoked a node"))
    node.memory_scope = c.MemoryScope(("agent",), ("session",))
    flow = CompiledFlow.from_parts(
        nodes={
            START_ID: StartNode(START_ID),
            "agent": node,
            END_ID: EndNode.value(END_ID, output_name=None),
        },
        edges=[
            Edge(id="start-agent", from_=START_ID, to="agent"),
            Edge(id="agent-end", from_="agent", to=END_ID),
        ],
    )
    if sibling:
        flow.add_subgraph(
            {"sibling": FuncNode("sibling", lambda _: pytest.fail("node called"))},
            [Edge(id="start-sibling", from_=START_ID, to="sibling")],
            {"sibling": {}},
        )
    return flow


def _capture(state):
    return ForwardCapture({}, ValueRoute.record(())) if state.learnable else None


def _started(*, learnable=False, sibling=False, enabled=True):
    flow = _flow(sibling=sibling)
    state = reducer.RunState(flow)
    pool = VariablePool(start_id=flow.start_id)
    pool.set(flow.start_id, {})
    pool.add_system("run_id", "run")
    log = MemoryRunLog("run")
    reducer.commit_prepared(
        log,
        state,
        reducer.prepare_run_started(
            state,
            reducer.RunStartData(
                run_id="run",
                flow_fingerprint="a" * 64,
                initial_pool=pool,
                start_wiring_override={},
                effective_llm_by_static_path={},
                num_workers=0,
                learnable=learnable,
                memories=(
                    c.MemorySpec("session", "session", {}, (c.DeclSite((), ("root",), "agent"),), True),
                ) if enabled else None,
                memory_stacks=(c.MemoryStack(("agent",), ("session",)),) if enabled else None,
                memory_snapshots=(c.MemorySnapshotEnvelope("session", {}),) if enabled else None,
            ),
        ),
    )
    reducer.commit_prepared(log, state, reducer.prepare_node_started(state, START_ID))
    reducer.commit_prepared(
        log,
        state,
        reducer.prepare_node_succeeded(
            state, START_ID, {}, forward_capture=_capture(state)
        ),
    )
    reducer.commit_prepared(log, state, reducer.prepare_node_started(state, "agent"))
    return state, log


def _snapshot(state):
    return (
        deepcopy(state.pool),
        dict(state.node_state),
        dict(state.edge_state),
        tuple(state.ready),
        dict(state.in_flight),
        dict(state.accepted_starts),
        state.result,
        state.next_sequence,
        state.backward_state,
        dict(state.depth),
        dict(state.growth_owner),
        deepcopy(state.wiring_overrides),
        deepcopy(state.input),
        deepcopy(state.effective_llm_by_static_path),
        state.memories,
        state.memory_stacks,
        state.memory_snapshots,
        state.learnable,
        state.terminal_unreachable_failure,
        state.num_workers,
        state.run_id,
        state.flow_fingerprint,
        tuple((name, id(node)) for name, node in state.flow.nodes.items()),
    )


def _trace_snapshot(trace):
    if trace is None:
        return None
    return {
        name: dict(value) if name == "_backward_handlers" else deepcopy(value)
        for name, value in trace.__dict__.items()
    }


def _replay(state, log, *, sibling=False):
    replayed = reducer.RunState(_flow(sibling=sibling))
    reducer._replay_records(
        replayed, tuple(log), grow_builder=lambda *_: pytest.fail("builder called")
    )
    # The final snapshot entry holds live node identities, not replay semantics.
    assert _snapshot(replayed)[:-1] == _snapshot(state)[:-1]
    assert replayed.initial_static_ids == state.initial_static_ids
    assert replayed.flow.start_id == state.flow.start_id
    assert replayed.flow.end_id == state.flow.end_id
    assert replayed.flow.edges == state.flow.edges
    assert replayed.flow.wiring == state.flow.wiring
    assert replayed.flow._incoming == state.flow._incoming
    assert replayed.flow._outgoing == state.flow._outgoing
    assert {
        name: (type(node), node.memory_scope, node.execution_scope, node.commit_as)
        for name, node in replayed.flow.nodes.items()
    } == {
        name: (type(node), node.memory_scope, node.execution_scope, node.commit_as)
        for name, node in state.flow.nodes.items()
    }
    traces = (_trace_snapshot(replayed.trace), _trace_snapshot(state.trace))
    for trace in traces:
        if trace is not None:
            trace["_backward_handlers"] = {
                source: (type(handler.__self__), handler.__self__.id, handler.__func__)
                for source, handler in trace["_backward_handlers"].items()
            }
    assert traces[0] == traces[1]
    return replayed


def test_resume_allows_memory_free_unfinished_invocations():
    state, log = _started(enabled=False)
    first_sequence = state.accepted_starts["agent"]
    reducer.commit_prepared(
        log, state, reducer.prepare_run_resumed(state, num_workers=2)
    )
    assert state.accepted_starts == {}
    reducer.commit_prepared(log, state, reducer.prepare_node_started(state, "agent"))
    assert state.accepted_starts["agent"] > first_sequence
    assert not hasattr(state, "conversation_state")
    _replay(state, log)


@pytest.mark.parametrize("learnable", [False, True])
def test_resume_retries_every_unfinished_invocation(learnable):
    state, log = _started(learnable=learnable, sibling=True)
    reducer.commit_prepared(log, state, reducer.prepare_node_started(state, "sibling"))
    before = tuple(state.in_flight)
    prepared = reducer.prepare_run_resumed(state, num_workers=0)
    assert prepared.record.retry_node_ids == before == ("agent", "sibling")
    reducer.commit_prepared(log, state, prepared)
    assert tuple(state.ready) == before
    assert state.in_flight == state.accepted_starts == {}
    _replay(state, log, sibling=True)


def test_forged_resume_plan_cannot_change_the_retry_frontier():
    state, log = _started()
    prepared = reducer.prepare_run_resumed(state, num_workers=0)
    prepared = replace(
        prepared,
        _plan=replace(prepared._plan, delta=replace(prepared._plan.delta, retry_node_ids=())),
    )
    before = _snapshot(state)
    with pytest.raises(ValueError):
        reducer.commit_prepared(log, state, prepared)
    assert _snapshot(state) == before


def test_active_conversation_preparation_is_not_exported():
    assert not hasattr(reducer, "prepare_conversation_record")
    assert "prepare_conversation_record" not in reducer.__all__
