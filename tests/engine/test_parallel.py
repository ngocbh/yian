"""Unit tests for the parallel (worker-pool) engine.

Two things to prove:
1. Scheduling stays correct under threads — the single-writer dispatcher gives
   the same outcomes as the single-threaded drain (join-once, branch/skip).
2. Independent branches actually overlap (the point of parallelism).
"""

import time

from agent_composer.events import RunFailed, RunSucceeded
from agent_composer.compile.model import END_ID, START_ID, Edge, CompiledFlow, NodeState, FlowOutput
from agent_composer.nodes.end import EndNode
from agent_composer.nodes.start import StartNode
from tests.engine._fakes import (
    BranchNode,
    FailNode,
    FuncNode,
    RecordNode,
)
from tests.engine._graph_builder import _engine, _graph


def _run(engine):
    return list(engine.run())


def _sleepy(node_id, seconds, log=None):
    def fn(_pool):
        time.sleep(seconds)
        if log is not None:
            log.append(node_id)
        return {"output": node_id}

    return FuncNode(node_id, fn)


def test_linear_is_deterministic_under_workers():
    log: list = []
    g = _graph(
        [RecordNode("a", log), RecordNode("b", log), RecordNode("c", log)],
        [(START_ID, "a"), ("a", "b"), ("b", "c"), ("c", END_ID)],
    )
    events = _run(_engine(g, num_workers=4))
    assert log == ["a", "b", "c"]  # a chain serializes regardless of pool size
    assert isinstance(events[-1], RunSucceeded)


def test_diamond_join_once_under_workers():
    log: list = []
    g = _graph(
        [RecordNode("a", log), RecordNode("b", log), RecordNode("c", log), RecordNode("d", log)],
        [(START_ID, "a"), ("a", "b"), ("a", "c"), ("b", "d"), ("c", "d"), ("d", END_ID)],
    )
    _run(_engine(g, num_workers=4))
    assert set(log) == {"a", "b", "c", "d"}
    assert log.count("d") == 1  # exact-once join held under concurrency


def test_case_skip_under_workers():
    log: list = []
    g = _graph(
        [BranchNode("cond", "yes"), RecordNode("b", log), RecordNode("c", log)],
        [(START_ID, "cond"), ("cond", "b", "yes"), ("cond", "c", "default"), ("b", END_ID), ("c", END_ID)],
    )
    engine = _engine(g, num_workers=4)
    _run(engine)
    assert log == ["b"]
    assert engine.state.node_state["c"] == NodeState.SKIPPED


def test_veto_under_workers_data_edged_branch_skips():
    # veto under the pool: a branch target with a gate-upstream data edge must still
    # skip-flood when its branch loses (shared disposition/_skip_edge, inherited by the pool).
    log: list = []
    nodes = [
        RecordNode("synth", log, output="pro"),
        BranchNode("gate", "pro_note"),
        RecordNode("pro_note", log, output="B"),
        RecordNode("con_note", log, output="b"),
    ]
    edges = [
        Edge("s0", START_ID, "synth"),
        Edge("synth->gate#0", "synth", "gate", input_group="cond"),
        Edge("gate->pro_note#0", "gate", "pro_note", source_handle="pro_note"),
        Edge("gate->con_note#0", "gate", "con_note", source_handle="con_note"),
        Edge("synth->pro_note#0", "synth", "pro_note", input_group="base"),
        Edge("synth->con_note#0", "synth", "con_note", input_group="base"),
        Edge("pro_note->end", "pro_note", END_ID),
        Edge("con_note->end", "con_note", END_ID),
    ]
    node_map = {n.id: n for n in nodes}
    node_map[START_ID] = StartNode(START_ID, params=[])
    node_map[END_ID] = EndNode.record(END_ID, output_names=[])
    engine = _engine(CompiledFlow.from_parts(node_map, edges), num_workers=4)
    _run(engine)
    assert "pro_note" in log and "con_note" not in log
    assert engine.state.node_state["con_note"] == NodeState.SKIPPED


def test_terminal_skipped_fails_under_workers():
    # the terminal-skipped failure fires on the parallel engine too (its run() uses the
    # shared _emit_terminal helper).
    g = _graph(
        [BranchNode("gate", "a"), RecordNode("a", []), RecordNode("t", [])],
        [(START_ID, "gate"), ("gate", "a", "a"), ("gate", "t", "t"), ("a", END_ID), ("t", END_ID)],
        outputs=[FlowOutput("result", "${t.output}")],
    )
    last = _run(_engine(g, num_workers=4))[-1]
    assert isinstance(last, RunFailed)
    assert last.failure.code == "graph.terminal_unreachable"


def test_fan_out_actually_overlaps():
    log: list = []
    nodes = [FuncNode("a", lambda p: {})]
    edges = [(START_ID, "a")]
    for i in range(4):
        nodes.append(_sleepy(f"s{i}", 0.1, log))
        edges.append(("a", f"s{i}"))
        edges.append((f"s{i}", "join"))
    nodes.append(FuncNode("join", lambda p: {"output": "ok"}))
    edges.append(("join", END_ID))

    g = _graph(nodes, edges)
    start = time.perf_counter()
    events = _run(_engine(g, num_workers=4))
    elapsed = time.perf_counter() - start

    assert isinstance(events[-1], RunSucceeded)
    assert len(log) == 4
    # four 0.1s sleeps run concurrently -> well under the 0.4s sequential cost
    assert elapsed < 0.3, f"expected overlap, took {elapsed:.2f}s"


def test_failure_under_workers():
    g = _graph(
        [FuncNode("a", lambda p: {}), FailNode("b", "nope")],
        [(START_ID, "a"), ("a", "b"), ("b", END_ID)],
    )
    events = _run(_engine(g, num_workers=4))
    assert isinstance(events[-1], RunFailed)
    assert events[-1].failure.message == "nope"


def test_terminal_output_under_workers():
    # The terminal node's produced value reaches RunSucceeded. With a single
    # declared output bound to node `t`'s whole value, terminal_output() yields
    # the bare value (== the node's object).
    g = _graph(
        [FuncNode("a", lambda p: {}), FuncNode("t", lambda p: {"output": "final"})],
        [(START_ID, "a"), ("a", "t"), ("t", END_ID)],
        outputs=[FlowOutput(name="result", binding="${t.output}")],
    )
    events = _run(_engine(g, num_workers=4))
    assert events[-1].output == {"output": "final"}


def test_no_declared_outputs_yields_none_under_workers():
    # the raw-terminal dict fallback is retired. A flow with no
    # declared outputs returns None from terminal_output().
    g = _graph(
        [FuncNode("a", lambda p: {}), FuncNode("t", lambda p: {"output": "final"})],
        [(START_ID, "a"), ("a", "t"), ("t", END_ID)],
    )
    events = _run(_engine(g, num_workers=4))
    assert isinstance(events[-1], RunSucceeded)
    assert events[-1].output is None
