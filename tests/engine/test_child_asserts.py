"""A def's / child's `asserts:` enforced through the child's boundary nodes.

A `def` may declare `asserts:`. At build these are lowered onto the def's synthesized
START/END nodes (PRE/boundary -> `START.post_asserts` with each `${input.X}` rewritten to
`${output.X}`, POST -> `END.post_asserts`). When the def runs via a `call` (REF once, MAP per
element), the child START/END are cloned into the parent and fire their asserts through the
ordinary `eval_node` path — the child START fires the boundary asserts (as post-asserts against
its COERCED output), the child END fires the post asserts. A failed child assert fails the child
node -> the parent run fails. This is the SAME firing path as a top-level flow's asserts (the
whole point of the merge).
"""

import pytest

from agent_composer.compose import LoadError, load_flow
from agent_composer.run import run_flow

# `checked` def: doubles its int input; boundary asserts n>=0, post asserts the output < 100.
_CHECKED_DEF = """
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
      - "${x.output} < 100"
"""

_REF_FLOW = f"""
id: d
name: d
input:
  v: int
{_CHECKED_DEF}
nodes:
  call_it:
    kind: call
    call: checked
    input:
      n: ${{input.v}}
output: ${{call_it.output}}
"""

_MAP_FLOW = f"""
id: dm
name: dm
input:
  vs: list[int]
{_CHECKED_DEF}
nodes:
  map_it:
    kind: map
    call: checked
    over: ${{input.vs}}
    input:
      n: ${{item}}
output: ${{map_it.output}}
"""


def test_def_with_asserts_loads():
    load_flow(_REF_FLOW)


# --- REF: non-vacuous boundary + post pairs, run through expansion. The same parent flow
# stays `succeeded` with a satisfying value AND fails with a violating one (the eager
# boundary eval in _apply_grow / the child END_ID pool-scoped post-assert). #
def test_ref_child_boundary_assert_satisfying_value_succeeds():
    out = run_flow(load_flow(_REF_FLOW), {"v": 5})           # 5 >= 0 -> boundary holds
    assert out.status == "succeeded"
    assert out.output == 10


def test_ref_child_boundary_assert_violating_value_fails():
    res = run_flow(load_flow(_REF_FLOW), {"v": -1})          # -1 violates ${input.n} >= 0
    assert res.status == "failed"                            # eager boundary eval in _apply_grow
    assert res.failure is not None
    assert "assert failed" in res.failure.message


def test_ref_child_post_assert_satisfying_value_succeeds():
    out = run_flow(load_flow(_REF_FLOW), {"v": 5})           # x=10 < 100 -> post holds
    assert out.status == "succeeded"


def test_ref_child_post_assert_violating_value_fails():
    res = run_flow(load_flow(_REF_FLOW), {"v": 60})          # x=120 violates ${x.output} < 100
    assert res.status == "failed"                            # child END_ID pool-scoped post-assert
    assert res.failure is not None
    assert "assert failed" in res.failure.message


# --- MAP: non-vacuous per-element boundary pair, run through expansion. The boundary
# asserts fire PER ELEMENT in the MAP arm's eager eval (each element's baked record); the
# same flow succeeds when all elements satisfy AND fails on one violating one. #
def test_map_child_asserts_all_elements_satisfy_succeeds():
    res = run_flow(load_flow(_MAP_FLOW), {"vs": [1, 2, 3]})          # every element >= 0
    assert res.status == "succeeded" and res.output == [2, 4, 6]     # over-order join


def test_map_child_assert_one_violating_element_fails():
    res = run_flow(load_flow(_MAP_FLOW), {"vs": [1, -1, 3]})         # element -1 violates ${input.n} >= 0
    assert res.status == "failed"                                    # per-element eager boundary eval
    assert res.failure is not None
    assert "assert failed" in res.failure.message


# --- a failing child boundary assert fails the run through the cloned child START ---------- #
# A def's boundary asserts are lowered onto its START node's `post_asserts` (each `${input.X}`
# rewritten to `${output.X}`); the cloned child START fires them through `eval_node` against its
# COERCED output (AFTER the subgraph is spliced + the GrowRecord attached, not eagerly inside
# `_apply_grow`). So a boundary failure now surfaces as an ordinary NodeFailed on the child START
# -> RunFailed, and the expansion IS attached (the child ran far enough to fire its START). The
# user-facing guarantee is the failure + message; the ledger attach is normal. Driven via
# FlowEngine with `run_inputs=` (the parent flow declares `v`/`vs`) so the top-level START is an
# ordinary scheduled node.
from agent_composer.events import RunFailed
from agent_composer.compile.expand import build_grow
from tests.engine._graph_builder import _engine


@pytest.mark.parametrize("num_workers", [0, 4])
def test_ref_boundary_failure_fails_run(num_workers):
    loaded = load_flow(_REF_FLOW)
    eng = _engine(loaded.compiled, run_inputs={"v": -1},   # violates n >= 0 on the cloned child START
                     num_workers=num_workers, grow_builder=build_grow)
    events = list(eng.run())
    assert isinstance(events[-1], RunFailed)
    # Fired by the cloned child StartNode post-asserts.
    assert "post-assert failed" in events[-1].failure.message


@pytest.mark.parametrize("num_workers", [0, 4])
def test_map_boundary_failure_fails_run(num_workers):
    loaded = load_flow(_MAP_FLOW)
    eng = _engine(loaded.compiled, run_inputs={"vs": [1, -1]},  # element 1 violates n >= 0 on its child START
                     num_workers=num_workers, grow_builder=build_grow)
    events = list(eng.run())
    assert isinstance(events[-1], RunFailed)
    assert "post-assert failed" in events[-1].failure.message
