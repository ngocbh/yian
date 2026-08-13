"""The generic `_apply_grow` core splices a self-describing `Flow`.

Drives `FlowEngine._apply_grow` directly with a tiny one-node child `Flow` (a leaf carrying a
baked `commit_as`), asserting the generic core's contract: the node is spliced into `flow.nodes`,
registered in the state manager, and its `start_id` scheduled. The spawner stand-in is a plain
`FuncNode` carrying the default node traits (`grow_depth_delta=None`, `is_loop=False`,
`grow_restamps_self=False`), so the trait-driven growth bookkeeping is a no-op — the core's
splice/register/schedule/prune contract is exercised in isolation."""

from agent_composer.compile.model import Edge, END_ID, Flow, START_ID
from agent_composer.nodes.base import Grow
from agent_composer.runtime.engine import FlowEngine
from tests.engine._fakes import FuncNode
from tests.engine._graph_builder import _graph


def _parent_engine() -> FlowEngine:
    # A minimal parent flow with one spawner-standin body node. The engine is constructed but
    # not driven — `_apply_grow` is called directly (paused is empty, so scheduled roots land in
    # self.ready).
    flow = _graph([FuncNode("s", lambda p: {"output": "x"})],
                  [(START_ID, "s"), ("s", END_ID)])
    return FlowEngine(flow, num_workers=0, grow_builder=lambda spawner, seed: seed)


def _one_node_grow(spawner_id: str) -> Grow:
    # A single-node child Flow: a leaf whose Output commits under the spawner id (baked
    # commit_as); it is both the entry (start_id) and the exit (end_id).
    child = FuncNode(f"{spawner_id}/leaf", lambda p: {"output": "y"})
    child.commit_as = spawner_id
    sg = Flow(nodes={child.id: child}, edges=[], wiring={child.id: {}},
              start_id=child.id, end_id=child.id)
    return Grow(seed=sg)


def test_apply_grow_splices_registers_and_schedules():
    eng = _parent_engine()
    grow = _one_node_grow("s")
    child_id = "s/leaf"

    eng._apply_grow("s", grow)

    # Spliced into the live topology.
    assert child_id in eng.flow.nodes
    # Registered in the state manager overlay.
    assert child_id in eng.sm.node_state
    # Its start_id scheduled (paused is empty -> lands in the serial ready deque).
    assert child_id in list(eng.ready)


def test_apply_grow_schedule_false_suppresses_scheduling():
    eng = _parent_engine()
    grow = _one_node_grow("s")
    child_id = "s/leaf"

    eng._apply_grow("s", grow, schedule=False)

    # Splice + register happen on replay, but nothing is scheduled.
    assert child_id in eng.flow.nodes
    assert child_id in eng.sm.node_state
    assert child_id not in list(eng.ready)


def _two_node_grow(a_id: str, b_id: str) -> Grow:
    # A two-node child Flow (a -> b): `a` is the entry (start_id), `b` the exit (end_id). Reused to
    # splice a subgraph whose ids a later Grow can name in its `prune` set.
    a = FuncNode(a_id, lambda p: {"output": "a"})
    b = FuncNode(b_id, lambda p: {"output": "b"})
    sg = Flow(
        nodes={a.id: a, b.id: b},
        edges=[Edge(id=f"{a_id}->{b_id}", from_=a_id, to=b_id)],
        wiring={a.id: {}, b.id: {}},
        start_id=a.id,
        end_id=b.id,
    )
    return Grow(seed=sg)


def test_apply_grow_applies_prune_removing_named_ids():
    eng = _parent_engine()

    # Splice S1 (nodes a, b) and give them some overlay bookkeeping to verify it is all reclaimed.
    eng._apply_grow("s", _two_node_grow("a", "b"))
    assert "a" in eng.flow.nodes and "b" in eng.flow.nodes
    eng.pool.store["a"] = {"output": "a"}
    eng.depth["a"] = 1
    eng._spawner_expansion["a"] = object()

    # Splice S2 whose Grow.prune names S1's ids: the generic core retires them after the splice.
    s2 = _two_node_grow("c", "d")
    s2 = Grow(seed=s2.seed, prune=frozenset({"a", "b"}))
    eng._apply_grow("s", s2)

    # The named ids are gone from topology, state, pool, depth, and _spawner_expansion.
    assert "a" not in eng.flow.nodes and "b" not in eng.flow.nodes
    assert "a" not in eng.sm.node_state and "b" not in eng.sm.node_state
    assert not any(e.from_ in {"a", "b"} or e.to in {"a", "b"} for e in eng.flow.edges)
    assert "a" not in eng.pool.store
    assert "a" not in eng.depth
    assert "a" not in eng._spawner_expansion
    # The just-spliced S2 survives (prune only names S1).
    assert "c" in eng.flow.nodes and "d" in eng.flow.nodes
