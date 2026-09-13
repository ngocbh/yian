"""Unit tests for the serial and pooled append-only engine drains.

These exercise scheduling semantics with fake nodes (no LLMs/tools):
ordering, fan-out, exact-once diamond join, CASE branch + skip-flood,
outputs-before-successors, failure, and typed output commits.
"""

from agent_composer.events import (
    RunFailed,
    RunSucceeded,
)
from agent_composer.compile.model import END_ID, START_ID, Edge, CompiledFlow, FlowOutput
from agent_composer.runtime.common import NodeState
from agent_composer.nodes.contracts import ValueContract
from agent_composer.nodes.end import EndNode
from agent_composer.nodes.start import StartNode
from agent_composer.typesys.values import ValueKind, Type
from tests.engine._fakes import (
    BranchNode,
    FailNode,
    FuncNode,
    RecordNode,
    derive_wiring,
)
from tests.engine._graph_builder import _engine, _graph


def _run(engine):
    return list(engine.run())


# --- ordering / linear ------------------------------------------------------ #


def test_linear_runs_in_order():
    log: list = []
    g = _graph(
        [RecordNode("a", log), RecordNode("b", log), RecordNode("c", log)],
        [(START_ID, "a"), ("a", "b"), ("b", "c"), ("c", END_ID)],
    )
    events = _run(_engine(g))
    assert log == ["a", "b", "c"]
    assert isinstance(events[-1], RunSucceeded)


def test_end_assembles_declared_output_record():
    # The flow's declared codomain is assembled from the pool: two outputs bound to
    # the terminal node's value -> an object keyed by output name (>=2 arity rule).
    g = _graph(
        [FuncNode("a", lambda p: {}), FuncNode("t", lambda p: {"output": "final", "n": 7})],
        [(START_ID, "a"), ("a", "t"), ("t", END_ID)],
        outputs=[
            FlowOutput(name="output", binding="${t.output.output}"),
            FlowOutput(name="n", binding="${t.output.n}"),
        ],
    )
    events = _run(_engine(g))
    assert events[-1].output == {"output": "final", "n": 7}


def test_no_declared_outputs_yields_none():
    # With no declared `output:`, END and the success event carry None.
    g = _graph(
        [FuncNode("a", lambda p: {}), FuncNode("t", lambda p: {"output": "final", "n": 7})],
        [(START_ID, "a"), ("a", "t"), ("t", END_ID)],
    )
    events = _run(_engine(g))
    assert events[-1].output is None


# --- END output assembly via the binding layer ----------------------------- #


def test_end_output_coalesce_returns_first_non_none():
    # A coalesce output `${a | b}` ("return whichever branch ran"): only `b`
    # produced a value (the `a` branch was skipped) -> resolve to b's value.
    g = _graph(
        [FuncNode("b", lambda p: "B-RAN")],
        [(START_ID, "b"), ("b", END_ID)],
        outputs=[FlowOutput(name="result", binding="${a.output | b.output}")],
    )
    events = _run(_engine(g))
    assert events[-1].output == "B-RAN"


def test_end_output_default_when_ref_none():
    # `${x:-lit}` falls back to the literal default when x is unbound (None).
    g = _graph(
        [FuncNode("a", lambda p: {})],
        [(START_ID, "a"), ("a", END_ID)],
        outputs=[FlowOutput(name="result", binding='${missing.output:-"fallback"}')],
    )
    events = _run(_engine(g))
    assert events[-1].output == "fallback"


def test_end_output_literal_string_passthrough():
    # A non-${...} string flow-output value passes through unchanged.
    g = _graph(
        [FuncNode("a", lambda p: {})],
        [(START_ID, "a"), ("a", END_ID)],
        outputs=[FlowOutput(name="result", binding="plain literal")],
    )
    events = _run(_engine(g))
    assert events[-1].output == "plain literal"


def test_end_output_literal_numeric_passthrough():
    # A non-string literal output value is kept as-is (no stringification).
    g = _graph(
        [FuncNode("a", lambda p: {})],
        [(START_ID, "a"), ("a", END_ID)],
        outputs=[FlowOutput(name="result", binding=7)],
    )
    events = _run(_engine(g))
    assert events[-1].output == 7
    assert isinstance(events[-1].output, int)


def test_end_output_flowoutput_carrier_builds_record():
    # FlowOutput reads each name/binding pair: a whole-string ref preserves the typed
    # value, and two or more outputs form an object keyed by name.
    g = _graph(
        [FuncNode("a", lambda p: {}), FuncNode("t", lambda p: {"output": "final", "n": 7})],
        [(START_ID, "a"), ("a", "t"), ("t", END_ID)],
        outputs=[
            FlowOutput(name="output", binding="${t.output.output}"),
            FlowOutput(name="n", binding="${t.output.n}"),
        ],
    )
    events = _run(_engine(g))
    assert events[-1].output == {"output": "final", "n": 7}


def test_end_output_whole_ref_preserves_typed_value():
    g = _graph(
        [FuncNode("t", lambda p: {"output": "final"})],
        [(START_ID, "t"), ("t", END_ID)],
        outputs=[FlowOutput(name="result", binding="${t.output.output}")],
    )
    events = _run(_engine(g))
    assert events[-1].output == "final"


# --- fan-out / fan-in ------------------------------------------------------- #


def test_fan_out_runs_all_branches():
    log: list = []
    g = _graph(
        [RecordNode("a", log), RecordNode("b", log), RecordNode("c", log), RecordNode("d", log)],
        [(START_ID, "a"), ("a", "b"), ("a", "c"), ("b", "d"), ("c", "d"), ("d", END_ID)],
    )
    _run(_engine(g))
    assert set(log) == {"a", "b", "c", "d"}


def test_multiple_roots_all_start():
    # two parallel entry nodes fanning out from __start__ (e.g. multi-reviewer flows)
    log: list = []
    g = _graph(
        [RecordNode("a", log), RecordNode("b", log), RecordNode("j", log)],
        [(START_ID, "a"), (START_ID, "b"), ("a", "j"), ("b", "j"), ("j", END_ID)],
    )
    engine = _engine(g)
    events = _run(engine)
    assert set(log) == {"a", "b", "j"}
    assert log.count("j") == 1  # join still fires exactly once
    # the single root is the synthesized START_ID; a/b are its out-edge targets.
    assert engine.state.flow.start_id == START_ID
    assert {e.to for e in engine.state.flow.outgoing(START_ID)} == {"a", "b"}
    assert isinstance(events[-1], RunSucceeded)


def test_diamond_join_fires_exactly_once():
    log: list = []
    g = _graph(
        [RecordNode("a", log), RecordNode("b", log), RecordNode("c", log), RecordNode("d", log)],
        [(START_ID, "a"), ("a", "b"), ("a", "c"), ("b", "d"), ("c", "d"), ("d", END_ID)],
    )
    _run(_engine(g))
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
    _run(_engine(g))
    assert seen["upstream"] == "payload"


# --- CASE branch + skip -------------------------------------------------- #


def test_case_takes_branch_and_skips_other():
    log: list = []
    g = _graph(
        [BranchNode("cond", "yes"), RecordNode("b", log), RecordNode("c", log)],
        [(START_ID, "cond"), ("cond", "b", "yes"), ("cond", "c", "default"), ("b", END_ID), ("c", END_ID)],
    )
    engine = _engine(g)
    events = _run(engine)
    assert log == ["b"]  # only the selected branch ran
    assert engine.state.node_state["c"] == NodeState.SKIPPED
    assert isinstance(events[-1], RunSucceeded)


def test_case_default_fallback():
    log: list = []
    g = _graph(
        [BranchNode("cond", "default"), RecordNode("b", log), RecordNode("c", log)],
        [(START_ID, "cond"), ("cond", "b", "yes"), ("cond", "c", "default"), ("b", END_ID), ("c", END_ID)],
    )
    _run(_engine(g))
    assert log == ["c"]


def test_skip_floods_through_unreachable_chain():
    # cond -default-> b -> c ; selecting "yes" (a dead-end to END_ID) skips b AND c.
    log: list = []
    g = _graph(
        [BranchNode("cond", "yes"), RecordNode("b", log), RecordNode("c", log)],
        [(START_ID, "cond"), ("cond", "b", "default"), ("cond", END_ID, "yes"), ("b", "c"), ("c", END_ID)],
    )
    engine = _engine(g)
    _run(engine)
    assert log == []
    assert engine.state.node_state["b"] == NodeState.SKIPPED
    assert engine.state.node_state["c"] == NodeState.SKIPPED


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
    node_map[START_ID] = StartNode(START_ID, params=[])
    node_map[END_ID] = EndNode.record(END_ID, output_names=[])
    g = CompiledFlow.from_parts(node_map, edges)
    engine = _engine(g)
    events = _run(engine)
    assert "pro_note" in log and "con_note" not in log
    assert engine.state.node_state["pro_note"] == NodeState.TAKEN
    assert engine.state.node_state["con_note"] == NodeState.SKIPPED
    assert isinstance(events[-1], RunSucceeded)


# --- a co-skipped terminal fails the run ----------------------------------- #


def _gate_flow(log, output_from):
    # gate routes to "a"; "t" (the other branch) is skip-flooded -> SKIPPED.
    return _graph(
        [BranchNode("gate", "a"), RecordNode("a", log, output="A"), RecordNode("t", log, output="T")],
        [(START_ID, "gate"), ("gate", "a", "a"), ("gate", "t", "t"), ("a", END_ID), ("t", END_ID)],
        outputs=[FlowOutput("result", output_from)],
    )


def test_skipped_terminal_producer_fails_run():
    last = _run(_engine(_gate_flow([], "${t.output}")))[-1]
    assert isinstance(last, RunFailed) and "skipped" in last.failure.message
    assert last.failure.code == "graph.terminal_unreachable"


def test_coalesce_terminal_one_branch_skipped_succeeds():
    # ${a.output | t.output}: a ran, t skipped -> legitimate success (taken branch's value).
    last = _run(_engine(_gate_flow([], "${a.output | t.output}")))[-1]
    assert isinstance(last, RunSucceeded)


def test_does_not_misfire_on_optional_terminal():
    # ${t.output:-null}: t skipped, but the literal escape succeeds by binding null.
    last = _run(_engine(_gate_flow([], "${t.output:-null}")))[-1]
    assert isinstance(last, RunSucceeded) and last.output is None


def test_genuine_none_terminal_succeeds():
    # A terminal that ran and returned None succeeds with that genuine value.
    g = _graph(
        [FuncNode("n", lambda p: None)],
        [(START_ID, "n"), ("n", END_ID)],
        outputs=[FlowOutput("result", "${n.output}")],
    )
    last = _run(_engine(g))[-1]
    assert isinstance(last, RunSucceeded) and last.output is None


# --- failure ---------------------------------------------------------------- #


def test_node_failure_produces_graph_failed():
    g = _graph(
        [FuncNode("a", lambda p: {}), FailNode("b", "nope"), FuncNode("c", lambda p: {})],
        [(START_ID, "a"), ("a", "b"), ("b", "c"), ("c", END_ID)],
    )
    events = _run(_engine(g))
    assert isinstance(events[-1], RunFailed)
    assert events[-1].failure.message == "nope"


# --- write-boundary output enforcement -------------------------------------- #


def test_output_enforced_rejects_type_mismatch():
    g = _graph(
        [
            FuncNode(
                "a",
                lambda p: "oops",
                output_contract=ValueContract(Type.scalar(ValueKind.NUMBER)),
            )
        ],
        [(START_ID, "a"), ("a", END_ID)],
    )
    events = _run(_engine(g))
    assert isinstance(events[-1], RunFailed)


def test_output_enforced_coerces_and_types_storage():
    g = _graph(
        [
            FuncNode(
                "a",
                lambda p: 3,
                output_contract=ValueContract(Type.scalar(ValueKind.NUMBER)),
            )
        ],
        [(START_ID, "a"), ("a", END_ID)],
    )
    engine = _engine(g)
    events = _run(engine)
    assert isinstance(events[-1], RunSucceeded)
    seg = engine.state.pool.get_value("a")
    assert seg.kind.value == "float" and seg.value == 3.0  # int coerced to float, typed NUMBER


def test_multi_output_record_is_a_closed_type():
    # >=2 declared outputs -> a CLOSED record Type ("several outputs = one object"):
    # all fields required, no extras. A node returning a MISSING or an EXTRA field fails
    # at success preparation (durable NodeFailed -> RunFailed); the exact object stores whole.
    rec = Type(
        kind=ValueKind.OBJECT,
        fields={"a": Type.scalar(ValueKind.STRING), "b": Type.scalar(ValueKind.STRING)},
        required=frozenset({"a", "b"}),
    )
    g_missing = _graph(
        [
            FuncNode(
                "n",
                lambda p: {"a": "x"},
                output_contract=ValueContract(rec),
            )
        ],
        [(START_ID, "n"), ("n", END_ID)],
    )
    assert isinstance(_run(_engine(g_missing))[-1], RunFailed)
    g_extra = _graph(
        [
            FuncNode(
                "n",
                lambda p: {"a": "x", "b": "y", "c": "z"},
                output_contract=ValueContract(rec),
            )
        ],
        [(START_ID, "n"), ("n", END_ID)],
    )
    assert isinstance(_run(_engine(g_extra))[-1], RunFailed)
    g_ok = _graph(
        [
            FuncNode(
                "n",
                lambda p: {"a": "x", "b": "y"},
                output_contract=ValueContract(rec),
            )
        ],
        [(START_ID, "n"), ("n", END_ID)],
    )
    engine = _engine(g_ok)
    assert isinstance(_run(engine)[-1], RunSucceeded)
    assert engine.state.pool.get("n") == {"a": "x", "b": "y"}


def test_unresolvable_declared_type_is_not_enforced():
    # 'Policy' is unresolvable against the empty registry -> the compiler stamps
    # output_contract=None (unenforced); the node's value is stored raw.
    g = _graph(
        [FuncNode("a", lambda p: {"k": 1}, output_contract=None)],
        [(START_ID, "a"), ("a", END_ID)],
    )
    engine = _engine(g)
    events = _run(engine)
    assert isinstance(events[-1], RunSucceeded)
    assert engine.state.pool.get("a") == {"k": 1}


def test_undeclared_output_is_not_enforced():
    # No declared output contract -> the node's value is stored whole, unenforced;
    # any object key is reachable via object-walk.
    g = _graph(
        [FuncNode("a", lambda p: {"other": "x"}, output_contract=None)],
        [(START_ID, "a"), ("a", END_ID)],
    )
    engine = _engine(g)
    events = _run(engine)
    assert isinstance(events[-1], RunSucceeded)
    assert engine.state.pool.resolve("a", ["output", "other"]) == "x"


# --- one engine, the num_workers knob --------------------------------------- #


def test_num_workers_0_and_4_same_terminal():
    def build(log):
        return _graph(
            [RecordNode("a", log), RecordNode("b", log), RecordNode("c", log)],
            [(START_ID, "a"), ("a", "b"), ("b", "c"), ("c", END_ID)],
        )

    log0, log4 = [], []
    ev0 = list(_engine(build(log0), num_workers=0).run())
    ev4 = list(_engine(build(log4), num_workers=4).run())
    assert log0 == ["a", "b", "c"]  # a chain serializes under either knob
    assert set(log4) == {"a", "b", "c"}
    assert isinstance(ev0[-1], RunSucceeded) and isinstance(ev4[-1], RunSucceeded)


def test_bare_num_workers_defaults_serial():
    # The default knob is serial (num_workers=0) — run.py's run_flow relies on it.
    eng = _engine(_graph([RecordNode("a", [])], [(START_ID, "a"), ("a", END_ID)]))
    assert eng.num_workers == 0


def test_parallel_module_and_export_are_gone():
    # ParallelFlowEngine is deleted; the only pooled path is FlowEngine(num_workers>=1).
    import importlib

    import agent_composer as ac

    assert not hasattr(ac, "ParallelFlowEngine")
    import pytest

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("agent_composer.runtime.parallel")
