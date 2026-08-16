"""Step 6: the top-level START runs as an ordinary scheduled node.

Before this step the engine bypassed START — it invoked `StartNode.run(run_inputs)`
once at init, committed `store[START_ID]`, and advanced WITHOUT enqueuing START or
emitting NodeStarted/NodeSucceeded, and the flow's boundary `asserts:` fired from a
bespoke `boundary_asserts` slot inside `StartNode._run_impl`.

After this step START is scheduled like any other node: it runs through `eval_node`
(so it emits NodeStarted/NodeSucceeded) and the flow's boundary asserts are rerouted
into START's ordinary `post_asserts`, firing against START's COERCED output. The
`${input.X}` refs a boundary assert reads are lowered to `${output.X}` so they see the
coerced record, not the raw pre-coercion run args.
"""

import pytest

from agent_composer.compile.model import END_ID, START_ID
from agent_composer.compose import load_flow
from agent_composer.events import NodeStarted, NodeSucceeded, RunFailed
from agent_composer.run import run_flow
from tests.engine._graph_builder import _engine


# --- boundary assert still gates the run (same behavior), now via START post_assert -------- #
_AGE_FLOW = """
id: age
name: age
input:
  age: int
nodes:
  n:
    kind: code
    input:
      x: ${input.age}
    output: int
    code: tests.engine._compose_codefns:echo_x
output: ${n.output}
asserts:
  - "${input.age} >= 18"
"""


def test_boundary_assert_violating_value_fails():
    res = run_flow(load_flow(_AGE_FLOW), {"age": 10})
    assert res.status == "failed"
    assert "assert failed" in res.failure.message


def test_boundary_assert_satisfying_value_succeeds():
    res = run_flow(load_flow(_AGE_FLOW), {"age": 20})
    assert res.status == "succeeded"
    assert res.output == 20


def test_start_emits_started_and_succeeded():
    # BEHAVIOR CHANGE: START is now an ordinary scheduled node, so it emits
    # NodeStarted + NodeSucceeded (the old bypass emitted neither).
    loaded = load_flow(_AGE_FLOW)
    eng = _engine(loaded.compiled, run_inputs={"age": 20})
    events = list(eng.run())
    assert any(isinstance(e, NodeStarted) and e.node_id == START_ID for e in events)
    assert any(isinstance(e, NodeSucceeded) and e.node_id == START_ID for e in events)


# --- C-2: a typed default reaches START output through the full engine path ---------------- #
_DEFAULT_FLOW = """
id: dflt
name: dflt
input:
  n: int = 5
nodes:
  echo:
    kind: code
    input:
      x: ${input.n}
    output: int
    code: tests.engine._compose_codefns:echo_x
output: ${echo.output}
"""


def test_typed_int_default_is_used_when_omitted():
    # C-2 regression: the caller omits `n`; ordinary parameter binding supplies the
    # already-validated integer default before START runs through `eval_node`.
    loaded = load_flow(_DEFAULT_FLOW)
    eng = _engine(loaded.compiled, run_inputs={})
    events = list(eng.run())
    succ = next(e for e in events if isinstance(e, NodeSucceeded) and e.node_id == START_ID)
    assert succ.output == {"n": 5}
    res = run_flow(loaded, {})
    assert res.status == "succeeded" and res.output == 5


# --- S-a: a boundary assert over a typed default reads START's bound output ---------------- #
_DEFAULT_ASSERT_FLOW = """
id: da
name: da
input:
  n: int = 5
nodes:
  echo:
    kind: code
    input:
      x: ${input.n}
    output: int
    code: tests.engine._compose_codefns:echo_x
output: ${echo.output}
asserts:
  - "${input.n} >= 5"
"""


def test_boundary_assert_reads_typed_default():
    # S-a regression: `n` is omitted, so the boundary assert reads START's bound
    # integer output after `${input.n}` is lowered to `${output.n}`.
    res = run_flow(load_flow(_DEFAULT_ASSERT_FLOW), {})
    assert res.status == "succeeded"
    assert res.output == 5


# --- S-1: a def's boundary assert fires from the cloned child START, fails the run --------- #
_CHILD_ASSERT_FLOW = """
id: d
name: d
input:
  v: int
defs:
  checked:
    input:
      n: int
    nodes:
      x:
        kind: code
        code: tests.engine._compose_codefns:double
        input:
          n: ${input.n}
        output: int
    output: ${x.output}
    asserts:
      - "${input.n} >= 0"
nodes:
  call_it:
    kind: call
    call: checked
    input:
      n: ${input.v}
output: ${call_it.output}
"""


@pytest.mark.parametrize("num_workers", [0, 4])
def test_child_boundary_assert_violation_fails_run(num_workers):
    res = run_flow(load_flow(_CHILD_ASSERT_FLOW), {"v": -1}, num_workers=num_workers)
    assert res.status == "failed"
    assert "assert failed" in res.failure.message


@pytest.mark.parametrize("num_workers", [0, 4])
def test_child_boundary_assert_satisfied_succeeds(num_workers):
    res = run_flow(load_flow(_CHILD_ASSERT_FLOW), {"v": 5}, num_workers=num_workers)
    assert res.status == "succeeded"
    assert res.output == 10


# --- fix-B guard: a ComposedFlow is REUSABLE across runs with different input key sets --------- #
# START's per-run `${input.X}` narrowing rides a per-ENGINE `wiring_overrides` dict, NOT the
# shared compiled `flow.wiring[START_ID]`. This test would FAIL if the engine reverted to
# mutating the shared IR (run 1 would narrow it to {topic,window}, then run 2 with `window`
# omitted would leave it narrowed to {topic}, discarding the loader's full wiring).
_REUSE_FLOW = """
id: reuse
name: reuse
input:
  topic: str
  window: int = 30
nodes:
  emit:
    kind: code
    input:
      topic: ${input.topic}
      window: ${input.window}
    output:
      report: str
      n: int
    code: tests.engine._compose_codefns:make_report
output:
  report: ${emit.output.report}
  n: ${emit.output.n}
"""


def test_loadedflow_reusable_across_runs_shared_ir_untouched():
    loaded = load_flow(_REUSE_FLOW)
    full_wiring = dict(loaded.compiled.wiring[START_ID])  # {topic, window} sources
    assert set(full_wiring) == {"topic", "window"}

    # Run 1: BOTH keys present (an explicit non-default window).
    r1 = run_flow(loaded, {"topic": "A", "window": 10})
    assert r1.status == "succeeded"
    assert r1.input == {"topic": "A", "window": 10}
    # The shared compiled START wiring is UNCHANGED (no per-run narrowing leaked into the IR).
    assert loaded.compiled.wiring[START_ID] == full_wiring

    # Run 2: a DIFFERENT key set (window omitted -> START fills the default 30, so the run still
    # succeeds even though `emit` binds `${input.window}`).
    r2 = run_flow(loaded, {"topic": "B"})
    assert r2.status == "succeeded"
    assert r2.input == {"topic": "B"}          # caller-passed, coerced (default not re-surfaced)
    assert r2.output == {"report": "report for B", "n": 1}  # make_report: n = len(topic)
    # Still unchanged after a second run with an omitted key — the fix-B guarantee.
    assert loaded.compiled.wiring[START_ID] == full_wiring
