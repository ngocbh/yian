"""Unit tests for the single-threaded FlowEngine drain.

These exercise scheduling semantics with fake nodes (no LLMs/tools):
ordering, fan-out, exact-once diamond join, CASE branch + skip-flood,
outputs-before-successors, failure, abort, and pause.
"""

from agent_composer.events import (
    RunAborted,
    RunFailed,
    RunPaused,
    RunSucceeded,
)
from agent_composer.compile.model import END_ID, START_ID, Edge, CompiledFlow, FlowOutput, NodeState
from agent_composer.nodes.end import EndNode
from agent_composer.nodes.start import StartNode
from agent_composer.runtime.engine import FlowEngine
from agent_composer.typesys.pool import VariablePool
from agent_composer.typesys.values import ValueKind, Type
from tests.engine._fakes import (
    BranchNode,
    FailNode,
    FuncNode,
    PauseOnceNode,
    RecordNode,
    derive_wiring,
)
from tests.engine._graph_builder import _graph


def _run(engine):
    return list(engine.run())


# --- ordering / linear ------------------------------------------------------ #


def test_linear_runs_in_order():
    log: list = []
    g = _graph(
        [RecordNode("a", log), RecordNode("b", log), RecordNode("c", log)],
        [(START_ID, "a"), ("a", "b"), ("b", "c"), ("c", END_ID)],
    )
    events = _run(FlowEngine(g))
    assert log == ["a", "b", "c"]
    assert isinstance(events[-1], RunSucceeded)


def test_terminal_output_surface():
    # The flow's declared codomain is assembled from the pool: two outputs bound to
    # the terminal node's value -> an object keyed by output name (>=2 arity rule).
    g = _graph(
        [FuncNode("a", lambda p: {}), FuncNode("t", lambda p: {"output": "final", "n": 7})],
        [(START_ID, "a"), ("a", "t"), ("t", END_ID)],
        outputs=[
            FlowOutput(name="output", from_="${t.output.output}"),
            FlowOutput(name="n", from_="${t.output.n}"),
        ],
    )
    events = _run(FlowEngine(g))
    assert events[-1].output == {"output": "final", "n": 7}


def test_no_declared_outputs_yields_none():
    # A-NOFALLBACK: with no declared `outputs:`, the raw-terminal dict fallback is
    # retired -> terminal_output() (and the success event) is None.
    g = _graph(
        [FuncNode("a", lambda p: {}), FuncNode("t", lambda p: {"output": "final", "n": 7})],
        [(START_ID, "a"), ("a", "t"), ("t", END_ID)],
    )
    events = _run(FlowEngine(g))
    assert events[-1].output is None


# --- terminal_output via the binding layer ---------------------------------- #


def test_terminal_output_coalesce_returns_first_non_none():
    # A coalesce output `${a | b}` ("return whichever branch ran"): only `b`
    # produced a value (the `a` branch was skipped) -> resolve to b's value.
    g = _graph(
        [FuncNode("b", lambda p: "B-RAN")],
        [(START_ID, "b"), ("b", END_ID)],
        outputs=[FlowOutput(name="result", from_="${a.output | b.output}")],
    )
    events = _run(FlowEngine(g))
    assert events[-1].output == "B-RAN"


def test_terminal_output_default_when_ref_none():
    # `${x:-lit}` falls back to the literal default when x is unbound (None).
    g = _graph(
        [FuncNode("a", lambda p: {})],
        [(START_ID, "a"), ("a", END_ID)],
        outputs=[FlowOutput(name="result", from_='${missing.output:-"fallback"}')],
    )
    events = _run(FlowEngine(g))
    assert events[-1].output == "fallback"


def test_terminal_output_literal_string_passthrough():
    # A non-${...} string flow-output value passes through unchanged.
    g = _graph(
        [FuncNode("a", lambda p: {})],
        [(START_ID, "a"), ("a", END_ID)],
        outputs=[FlowOutput(name="result", from_="plain literal")],
    )
    events = _run(FlowEngine(g))
    assert events[-1].output == "plain literal"


def test_terminal_output_literal_numeric_passthrough():
    # A non-string literal output value is kept as-is (no stringification).
    g = _graph(
        [FuncNode("a", lambda p: {})],
        [(START_ID, "a"), ("a", END_ID)],
        outputs=[FlowOutput(name="result", from_=7)],
    )
    events = _run(FlowEngine(g))
    assert events[-1].output == 7
    assert isinstance(events[-1].output, int)


def test_terminal_output_flowoutput_carrier_like_iofield():
    # FlowOutput behaves exactly like IOField in terminal_output (reads .name/.from_):
    # whole-string ref preserves the typed value; >=2 -> object keyed by name.
    g = _graph(
        [FuncNode("a", lambda p: {}), FuncNode("t", lambda p: {"output": "final", "n": 7})],
        [(START_ID, "a"), ("a", "t"), ("t", END_ID)],
        outputs=[
            FlowOutput(name="output", from_="${t.output.output}"),
            FlowOutput(name="n", from_="${t.output.n}"),
        ],
    )
    events = _run(FlowEngine(g))
    assert events[-1].output == {"output": "final", "n": 7}


def test_terminal_output_iofield_still_resolves():
    # Legacy IOField-typed outputs list still resolves (whole-string ${...}).
    g = _graph(
        [FuncNode("t", lambda p: {"output": "final"})],
        [(START_ID, "t"), ("t", END_ID)],
        outputs=[FlowOutput(name="result", from_="${t.output.output}")],
    )
    events = _run(FlowEngine(g))
    assert events[-1].output == "final"


# --- fan-out / fan-in ------------------------------------------------------- #


def test_fan_out_runs_all_branches():
    log: list = []
    g = _graph(
        [RecordNode("a", log), RecordNode("b", log), RecordNode("c", log), RecordNode("d", log)],
        [(START_ID, "a"), ("a", "b"), ("a", "c"), ("b", "d"), ("c", "d"), ("d", END_ID)],
    )
    _run(FlowEngine(g))
    assert set(log) == {"a", "b", "c", "d"}


def test_multiple_roots_all_start():
    # two parallel entry nodes fanning out from __start__ (e.g. multi-reviewer flows)
    log: list = []
    g = _graph(
        [RecordNode("a", log), RecordNode("b", log), RecordNode("j", log)],
        [(START_ID, "a"), (START_ID, "b"), ("a", "j"), ("b", "j"), ("j", END_ID)],
    )
    engine = FlowEngine(g)
    events = _run(engine)
    assert set(log) == {"a", "b", "j"}
    assert log.count("j") == 1  # join still fires exactly once
    # the single root is the synthesized START_ID; a/b are its out-edge targets.
    assert engine.flow.start_id == START_ID
    assert {e.to for e in engine.flow.outgoing(START_ID)} == {"a", "b"}
    assert isinstance(events[-1], RunSucceeded)


def test_diamond_join_fires_exactly_once():
    log: list = []
    g = _graph(
        [RecordNode("a", log), RecordNode("b", log), RecordNode("c", log), RecordNode("d", log)],
        [(START_ID, "a"), ("a", "b"), ("a", "c"), ("b", "d"), ("c", "d"), ("d", END_ID)],
    )
    _run(FlowEngine(g))
    assert log.count("d") == 1  # join node runs once, not once-per-incoming-edge


# --- outputs visible to successors ----------------------------------------- #


def test_outputs_written_before_successor_runs():
    seen = {}

    def reader(inputs: dict) -> dict:
        seen["upstream"] = inputs["upstream"]
        return {}

    g = _graph(
        [
            FuncNode("up", lambda p: {"output": "payload"}),
            FuncNode("down", reader, reads={"upstream": "${up.output.output}"}),
        ],
        [(START_ID, "up"), ("up", "down"), ("down", END_ID)],
    )
    _run(FlowEngine(g))
    assert seen["upstream"] == "payload"


# --- CASE branch + skip -------------------------------------------------- #


def test_case_takes_branch_and_skips_other():
    log: list = []
    g = _graph(
        [BranchNode("cond", "yes"), RecordNode("b", log), RecordNode("c", log)],
        [(START_ID, "cond"), ("cond", "b", "yes"), ("cond", "c", "default"), ("b", END_ID), ("c", END_ID)],
    )
    engine = FlowEngine(g)
    events = _run(engine)
    assert log == ["b"]  # only the selected branch ran
    assert engine.sm.node_state["c"] == NodeState.SKIPPED
    assert isinstance(events[-1], RunSucceeded)


def test_case_default_fallback():
    log: list = []
    g = _graph(
        [BranchNode("cond", "default"), RecordNode("b", log), RecordNode("c", log)],
        [(START_ID, "cond"), ("cond", "b", "yes"), ("cond", "c", "default"), ("b", END_ID), ("c", END_ID)],
    )
    _run(FlowEngine(g))
    assert log == ["c"]


def test_skip_floods_through_unreachable_chain():
    # cond -default-> b -> c ; selecting "yes" (a dead-end to END_ID) skips b AND c.
    log: list = []
    g = _graph(
        [BranchNode("cond", "yes"), RecordNode("b", log), RecordNode("c", log)],
        [(START_ID, "cond"), ("cond", "b", "default"), ("cond", END_ID, "yes"), ("b", "c"), ("c", END_ID)],
    )
    engine = FlowEngine(g)
    _run(engine)
    assert log == []
    assert engine.sm.node_state["b"] == NodeState.SKIPPED
    assert engine.sm.node_state["c"] == NodeState.SKIPPED


def test_data_edged_branch_target_does_not_run_when_skipped():
    # The control-veto shape: pro_note/con_note each carry a CONTROL edge from gate
    # AND a DATA edge from synth (UPSTREAM of the gate). Routing to pro_note must skip-flood
    # con_note even though synth->con_note is TAKEN. (Pre-veto, con_note RAN — the bug.)
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
    node_map[START_ID] = StartNode(START_ID, input_decls=[])
    node_map[END_ID] = EndNode.record(END_ID, output_names=[])
    g = CompiledFlow.from_parts(node_map, edges)
    engine = FlowEngine(g)
    events = _run(engine)
    assert "pro_note" in log and "con_note" not in log
    assert engine.sm.node_state["pro_note"] == NodeState.TAKEN
    assert engine.sm.node_state["con_note"] == NodeState.SKIPPED
    assert isinstance(events[-1], RunSucceeded)


# --- E7: a co-skipped terminal fails the run ------------------------------- #


def _gate_flow(log, output_from):
    # gate routes to "a"; "t" (the other branch) is skip-flooded -> SKIPPED.
    return _graph(
        [BranchNode("gate", "a"), RecordNode("a", log, output="A"), RecordNode("t", log, output="T")],
        [(START_ID, "gate"), ("gate", "a", "a"), ("gate", "t", "t"), ("a", END_ID), ("t", END_ID)],
        outputs=[FlowOutput("result", output_from)],
    )


def test_skipped_terminal_producer_fails_run():
    last = _run(FlowEngine(_gate_flow([], "${t.output}")))[-1]
    assert isinstance(last, RunFailed) and "skipped" in last.error
    assert last.error_type == "TerminalSkipped"


def test_coalesce_terminal_one_branch_skipped_succeeds():
    # ${a.output | t.output}: a ran, t skipped -> legitimate success (taken branch's value).
    last = _run(FlowEngine(_gate_flow([], "${a.output | t.output}")))[-1]
    assert isinstance(last, RunSucceeded)


def test_does_not_misfire_on_optional_terminal():
    # ${t.output:-null}: t skipped, but the literal escape -> succeeds binding null (not E7).
    last = _run(FlowEngine(_gate_flow([], "${t.output:-null}")))[-1]
    assert isinstance(last, RunSucceeded) and last.output is None


def test_genuine_none_terminal_succeeds():
    # terminal RAN and returned None (TAKEN, value None) -> RunSucceeded(None), NOT E7.
    g = _graph(
        [FuncNode("n", lambda p: None)],
        [(START_ID, "n"), ("n", END_ID)],
        outputs=[FlowOutput("result", "${n.output}")],
    )
    last = _run(FlowEngine(g))[-1]
    assert isinstance(last, RunSucceeded) and last.output is None


# --- failure / abort / pause ------------------------------------------------ #


def test_node_failure_produces_graph_failed():
    g = _graph(
        [FuncNode("a", lambda p: {}), FailNode("b", "nope"), FuncNode("c", lambda p: {})],
        [(START_ID, "a"), ("a", "b"), ("b", "c"), ("c", END_ID)],
    )
    events = _run(FlowEngine(g))
    assert isinstance(events[-1], RunFailed)
    assert events[-1].error == "nope"


def test_abort_stops_the_drain():
    log: list = []

    class AbortAfterA(FlowEngine):
        pass

    g = _graph(
        [RecordNode("a", log), RecordNode("b", log)],
        [(START_ID, "a"), ("a", "b"), ("b", END_ID)],
    )
    engine = FlowEngine(g)
    gen = engine.run()
    next(gen)  # RunStarted -> root 'a' enqueued
    engine.request_abort()
    events = [e for e in gen]
    assert any(isinstance(e, RunAborted) for e in events)


def test_pause_suspends_run_with_reason():
    g = _graph(
        [
            FuncNode("a", lambda p: {}),
            PauseOnceNode("gate", reason="approve?"),
            FuncNode("after", lambda p: {"output": "done"}),
        ],
        [(START_ID, "a"), ("a", "gate"), ("gate", "after"), ("after", END_ID)],
    )
    engine = FlowEngine(g)
    events = _run(engine)
    assert isinstance(events[-1], RunPaused)
    assert events[-1].reasons == ["approve?"]
    # downstream did NOT run (no answer yet); the parked gate stays TAKEN (deliver-as-Output:
    # no UNKNOWN reset / re-run — the engine delivers its answer on resume).
    assert engine.sm.node_state["after"] == NodeState.UNKNOWN
    assert engine.sm.node_state["gate"] == NodeState.TAKEN


# --- write-boundary output enforcement (slice 2) ---------------------------- #


def test_output_enforced_rejects_type_mismatch():
    g = _graph(
        [FuncNode("a", lambda p: "oops", output_type=Type.scalar(ValueKind.NUMBER))],
        [(START_ID, "a"), ("a", END_ID)],
    )
    events = _run(FlowEngine(g))
    assert isinstance(events[-1], RunFailed)


def test_output_enforced_coerces_and_types_storage():
    pool = VariablePool()
    g = _graph(
        [FuncNode("a", lambda p: 3, output_type=Type.scalar(ValueKind.NUMBER))],
        [(START_ID, "a"), ("a", END_ID)],
    )
    events = _run(FlowEngine(g, pool))
    assert isinstance(events[-1], RunSucceeded)
    seg = pool.get_value("a")
    assert seg.kind.value == "float" and seg.value == 3.0  # int coerced to float, typed NUMBER


def test_multi_output_record_is_a_closed_type():
    # >=2 declared outputs -> a CLOSED record Type ("several outputs = one object"):
    # all fields required, no extras. A node returning a MISSING or an EXTRA field fails
    # at the write boundary (NodeExecutionError -> RunFailed); the exact object stores whole.
    rec = Type(
        kind=ValueKind.OBJECT,
        fields={"a": Type.scalar(ValueKind.STRING), "b": Type.scalar(ValueKind.STRING)},
        required=frozenset({"a", "b"}),
    )
    g_missing = _graph([FuncNode("n", lambda p: {"a": "x"}, output_type=rec)], [(START_ID, "n"), ("n", END_ID)])
    assert isinstance(_run(FlowEngine(g_missing))[-1], RunFailed)
    g_extra = _graph(
        [FuncNode("n", lambda p: {"a": "x", "b": "y", "c": "z"}, output_type=rec)], [(START_ID, "n"), ("n", END_ID)]
    )
    assert isinstance(_run(FlowEngine(g_extra))[-1], RunFailed)
    pool = VariablePool()
    g_ok = _graph([FuncNode("n", lambda p: {"a": "x", "b": "y"}, output_type=rec)], [(START_ID, "n"), ("n", END_ID)])
    assert isinstance(_run(FlowEngine(g_ok, pool))[-1], RunSucceeded)
    assert pool.get("n") == {"a": "x", "b": "y"}


def test_unresolvable_declared_type_is_not_enforced():
    # 'Policy' is unresolvable against the empty registry -> the compiler stamps
    # output_type=None (unenforced); the node's value is stored raw.
    pool = VariablePool()
    g = _graph(
        [FuncNode("a", lambda p: {"k": 1}, output_type=None)],
        [(START_ID, "a"), ("a", END_ID)],
    )
    events = _run(FlowEngine(g, pool))
    assert isinstance(events[-1], RunSucceeded)
    assert pool.get("a") == {"k": 1}


def test_undeclared_output_is_not_enforced():
    # No declared output_type -> the node's value is stored whole, unenforced;
    # any object key is reachable via object-walk.
    pool = VariablePool()
    g = _graph(
        [FuncNode("a", lambda p: {"other": "x"}, output_type=None)],
        [(START_ID, "a"), ("a", END_ID)],
    )
    events = _run(FlowEngine(g, pool))
    assert isinstance(events[-1], RunSucceeded)
    assert pool.resolve("a", ["output", "other"]) == "x"


# --- one engine, the num_workers knob --------------------------------------- #


def test_num_workers_0_and_4_same_terminal():
    def build(log):
        return _graph(
            [RecordNode("a", log), RecordNode("b", log), RecordNode("c", log)],
            [(START_ID, "a"), ("a", "b"), ("b", "c"), ("c", END_ID)],
        )

    log0, log4 = [], []
    ev0 = list(FlowEngine(build(log0), num_workers=0).run())
    ev4 = list(FlowEngine(build(log4), num_workers=4).run())
    assert log0 == ["a", "b", "c"]  # a chain serializes under either knob
    assert set(log4) == {"a", "b", "c"}
    assert isinstance(ev0[-1], RunSucceeded) and isinstance(ev4[-1], RunSucceeded)


def test_bare_num_workers_defaults_serial():
    # The default knob is serial (num_workers=0) — run.py's run_flow relies on it.
    eng = FlowEngine(_graph([RecordNode("a", [])], [(START_ID, "a"), ("a", END_ID)]))
    assert eng.num_workers == 0


def test_pause_works_under_both_knobs():
    def build():
        return _graph(
            [FuncNode("a", lambda p: {}),
             PauseOnceNode("gate", reason="approve?")],
            [(START_ID, "a"), ("a", "gate"), ("gate", END_ID)],
        )

    for nw in (0, 4):
        ev = list(FlowEngine(build(), num_workers=nw).run())
        assert isinstance(ev[-1], RunPaused) and ev[-1].reasons == ["approve?"]


def test_ready_snapshot_branches_on_num_workers():
    # _ready_snapshot must branch strictly on num_workers==0
    # (self.ready) vs the pooled ready_q — so snapshot() is correct under either path.
    serial = FlowEngine(_graph([RecordNode("a", [])], [(START_ID, "a"), ("a", END_ID)]), num_workers=0)
    serial.ready.append("queued")
    assert serial._ready_snapshot() == ["queued"]
    pooled = FlowEngine(_graph([RecordNode("a", [])], [(START_ID, "a"), ("a", END_ID)]), num_workers=4)
    pooled.ready_q.put("queued")
    assert pooled._ready_snapshot() == ["queued"]  # reads ready_q, not the empty serial deque


def test_pooled_snapshot_captures_paused_state():
    # Pooled run pauses, then snapshot() goes through _ready_snapshot's POOLED arm — proves the
    # single-engine design does not silently break the snapshot path under num_workers>=1.
    # A test-only Pause node carries a structured reason accepted by the checkpoint model.
    from agent_composer.suspension.pause import HumanInputRequired

    eng = FlowEngine(
        _graph([FuncNode("a", lambda p: {}),
                PauseOnceNode(
                    "gate",
                    reason=HumanInputRequired(prompt="approve?", node_id="gate"),
                )],
               [(START_ID, "a"), ("a", "gate"), ("gate", END_ID)]),
        num_workers=4,
    )
    ev = list(eng.run())
    assert isinstance(ev[-1], RunPaused)
    ckpt = eng.snapshot()  # exercises the pooled _ready_snapshot arm
    assert [r.prompt for r in ckpt.pause_reasons] == ["approve?"]


def test_parallel_module_and_export_are_gone():
    # ParallelFlowEngine is deleted; the only pooled path is FlowEngine(num_workers>=1).
    import importlib

    import agent_composer as ac

    assert not hasattr(ac, "ParallelFlowEngine")
    import pytest

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("agent_composer.runtime.parallel")
