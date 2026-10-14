"""The boundary flip: the loader synthesizes real START_ID/END_ID NODES into every
flow (the __start__/__end__ sentinels retire — the strings persist as the reserved node ids), the
engine reads END_ID's committed value as the run result, START_ID is seeded at engine init, and
${input.X} resolves to store[START_ID]. Assertions check the NODES exist + edges carry real
adjacency, never that the strings are absent (they ARE the node ids).
"""

from yian.compile.model import END_ID, START_ID
from yian.compose import load_flow
from yian.run import run_flow
from yian.nodes.base import NodeKind
from tests.engine._graph_builder import _engine

_FLOW = """
id: f
name: f
input:
  x: !str
nodes:
  n:
    kind: code
    input:
      x: ${input.x}
    output: str
    code: tests.engine._compose_codefns:echo_x
output: ${n.output}
"""

# A terminal output whose only producer, `kept`, is a CASE target that may be skipped. The
# loader rewrites the output to the null default `${kept.output:-null}`, so END_ID's group is
# optional: when `gate` routes to `cold`, `kept` is skipped and the run returns null.
_SKIPPED_TERMINAL = """
id: e7
name: e7
input:
  seed: !float
nodes:
  score:
    kind: code
    code: tests.engine._compose_codefns:score
    input:
      seed: ${input.seed}
    output: float
  gate:
    kind: case
    cases:
      - when: "score.output >= 1"
        then: kept
    else: cold
  kept:
    kind: code
    code: tests.engine._compose_codefns:cautious
    input:
      topic: ${input.seed}
    output: str
  cold:
    kind: code
    code: tests.engine._compose_codefns:cautious
    input:
      topic: ${input.seed}
    output: str
output: ${kept.output}
"""
_E7_INPUTS = {"seed": 0.0}   # score=0 -> when `>= 1` false -> else `cold`; `kept` skip-flooded


def test_loader_synthesizes_start_and_end_nodes():
    flow = load_flow(_FLOW).compiled
    # START_ID and END_ID identify real synthesized nodes.
    assert flow.nodes[START_ID].kind is NodeKind.START
    assert flow.nodes[END_ID].kind is NodeKind.END
    # START_ID is the single root; END_ID is the terminal (via the start_id/end_id accessors).
    assert flow.start_id == START_ID
    assert flow.terminal_id == END_ID
    # ordinary edges to/from the real boundary nodes (n -> END_ID is a real adjacency).
    pairs = {(e.from_, e.to) for e in flow.edges}
    assert (START_ID, "n") in pairs and ("n", END_ID) in pairs
    # the n -> END_ID edge carries the output's input_group (a real data edge, not a pseudo-edge).
    assert any(e.from_ == "n" and e.to == END_ID and e.input_group is not None for e in flow.edges)


def test_run_result_is_end_node_value():
    res = run_flow(load_flow(_FLOW), {"x": "ACME"})
    assert res.status == "succeeded"
    assert res.output == "ACME"            # = store[END_ID], the END_ID node's committed value


def test_terminal_helpers_retired():
    from yian.runtime.engine import FlowEngine
    # the three terminal helpers are gone (END_ID commits the value; the disposition handles skip).
    assert not hasattr(FlowEngine, "terminal_output")
    assert not hasattr(FlowEngine, "_emit_terminal")
    assert not hasattr(FlowEngine, "_terminal_coskip")


def test_terminal_output_of_a_skipped_producer_is_null():
    # END_ID's group for an output the loader null-defaults is optional, so a skipped producer
    # commits null. A required group that dies still fails the run (test_engine.py covers it).
    loaded = load_flow(_SKIPPED_TERMINAL)
    assert loaded.compiled.outputs[0].from_ == "${kept.output:-null}"
    res = run_flow(loaded, _E7_INPUTS)
    assert res.status == "succeeded", res.failure
    assert res.output is None


# --- START_ID seeded at engine init + ${input.X} -> store[START_ID]; inputs namespace retired -- #
def test_inputs_resolve_to_start_node():
    from yian.typesys.pool import VariablePool
    pool = VariablePool()
    pool.set(START_ID, {"x": "ACME"})                 # START_ID commits the bound input record
    assert pool.resolve("input", ["x"]) == "ACME"    # ${input.x} == store[START_ID].x


def test_pool_inputs_namespace_retired():
    from yian.typesys.pool import VariablePool
    pool = VariablePool()
    assert not hasattr(pool, "add_inputs")
    assert "inputs" not in VariablePool.model_fields   # the `inputs` field is gone


def test_inputs_read_mints_producer_edge_from_start():
    # ${input.x} on node `n` now depends on START_ID -> a START_ID->n data edge tagged input_group=x.
    flow = load_flow(_FLOW).compiled
    assert any(e.from_ == START_ID and e.to == "n" and e.input_group == "x" for e in flow.edges)


def test_input_reader_is_not_also_a_bare_root():
    # a node reading ${input.x} gets the data edge ONLY — NOT a second bare START_ID->n root edge.
    flow = load_flow(_FLOW).compiled
    start_to_n = [e for e in flow.edges if e.from_ == START_ID and e.to == "n"]
    assert len(start_to_n) == 1 and start_to_n[0].input_group == "x"


def test_engine_schedules_start_and_emits_succeeded():
    # START is now an ORDINARY scheduled node: the engine primes store[START_ID] with the raw
    # run-args, schedules START_ID, and START runs through eval_node — emitting NodeStarted +
    # NodeSucceeded like any node and committing its COERCED bound record. The run succeeds and
    # the value flows from the scheduled START, not a hand-seed bypass.
    from yian.events import NodeStarted, NodeSucceeded
    loaded = load_flow(_FLOW)
    eng = _engine(loaded.compiled, run_inputs={"x": "ACME"})
    events = list(eng.run())
    assert any(isinstance(e, NodeStarted) and e.node_id == START_ID for e in events)
    assert any(isinstance(e, NodeSucceeded) and e.node_id == START_ID for e in events)
    assert eng.state.pool.get(START_ID) == {"x": "ACME"}  # scheduled START commit
    res = run_flow(load_flow(_FLOW), {"x": "ACME"})
    assert res.status == "succeeded" and res.output == "ACME"
