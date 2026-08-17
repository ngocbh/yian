"""Regression tests for flow assertions lowered onto synthesized boundaries.

Flow-level `asserts:` run through ordinary node-local assertion machinery on the
synthesized START and END nodes. These tests cover child assertions over internal
producers across call, map, and loop; boundary assertion reference forms; post-assert
ordering behind a case branch; and load-time rejection of the ambient `system`
namespace. Record-scoped bare boundary references remain deferred to issue #93.
"""

import pytest

from agent_composer.compose import LoadError, load_flow
from agent_composer.run import run_flow


# =========================================================================== #
# Child bare assertions over internal producers across spawners
#
# A def declares `asserts: - "emit.output.n >= 1"` in bare form and reads its own
# internal producer. Lowering must retain that dependency when each spawner clones
# and namespaces the child graph.
# =========================================================================== #

# A def whose top-level assert reads an internal producer (`emit`) in BARE form.
_INTERNAL_ASSERT_CHILD_DEF = """
defs:
  child:
    input:
      seed: int
    nodes:
      emit:
        kind: code
        input:
          seed: ${input.seed}
        output: {n: int}
        code: |
          return {"n": inputs["seed"] + 1}
    output: ${emit.output}
    asserts:
      - "emit.output.n >= 1"
"""


def test_call_bare_internal_assert_succeeds():
    """A call succeeds when its child's internal-producer assertion holds."""
    flow = f"""
id: c2call
name: c2_call
input:
  seed: int
{_INTERNAL_ASSERT_CHILD_DEF}
nodes:
  callee:
    kind: call
    call: child
    input:
      seed: ${{input.seed}}
output: ${{callee.output}}
"""
    res = run_flow(load_flow(flow), {"seed": 5})
    assert res.status == "succeeded", res.failure
    assert res.output == {"n": 6}


def test_map_bare_internal_assert_all_satisfy_succeeds():
    """A map succeeds in input order when every child assertion holds."""
    flow = f"""
id: c2map
name: c2_map
input:
  seeds: list[int]
{_INTERNAL_ASSERT_CHILD_DEF}
nodes:
  fan:
    kind: map
    call: child
    over: ${{input.seeds}}
    input:
      seed: ${{item}}
output: ${{fan.output}}
"""
    res = run_flow(load_flow(flow), {"seeds": [5, 6, 7]})
    assert res.status == "succeeded", res.failure
    assert res.output == [{"n": 6}, {"n": 7}, {"n": 8}]


def test_map_bare_internal_assert_one_element_violates_fails_loudly():
    """A map fails loudly when one child's internal-producer assertion is false."""
    flow = f"""
id: c2mapv
name: c2_map_violate
input:
  seeds: list[int]
{_INTERNAL_ASSERT_CHILD_DEF}
nodes:
  fan:
    kind: map
    call: child
    over: ${{input.seeds}}
    input:
      seed: ${{item}}
output: ${{fan.output}}
"""
    res = run_flow(load_flow(flow), {"seeds": [5, -5, 7]})  # -5 -> n=-4 violates >= 1
    assert res.status == "failed"
    assert res.failure is not None
    assert "assert" in res.failure.message.lower()


def test_loop_bare_internal_assert_succeeds():
    """A loop succeeds when every body's internal-producer assertion holds."""
    flow = """
id: c2loop
name: c2_loop
input:
  start: int
defs:
  body:
    input:
      n: int
    nodes:
      step:
        kind: code
        input:
          n: ${input.n}
        output: {n: int}
        code: |
          return {"n": inputs["n"] + 1}
    output: ${step.output}
    asserts:
      - "step.output.n >= 1"
nodes:
  spin:
    kind: loop
    call: body
    input:
      n: ${input.start}
    times: 3
output: ${spin.output}
"""
    res = run_flow(load_flow(flow), {"start": 5})
    assert res.status == "succeeded", res.failure
    assert res.output == {"n": 8}


# =========================================================================== #
# Boundary assertion reference forms
#
# The braced `${input.age}` form runs through START's lowered assertion. Resolving a
# bare `age` reference against the START record remains deferred to issue #93.
# =========================================================================== #

_BOUNDARY_ASSERT_ECHO_BODY = """
nodes:
  echo:
    kind: code
    input:
      age: ${input.age}
    output: int
    code: |
      return inputs["age"]
output: ${echo.output}
"""


def _boundary_assert_flow(assert_str: str) -> str:
    return f"""
id: c3
name: c3
input:
  age: int
{_BOUNDARY_ASSERT_ECHO_BODY}
asserts:
  - "{assert_str}"
"""


def test_braced_boundary_assert_fail_fast_and_pass():
    """The braced boundary form fails for age 10 and succeeds for age 20."""
    loaded = load_flow(_boundary_assert_flow("${input.age} >= 18"))
    assert run_flow(loaded, {"age": 10}).status == "failed"
    assert run_flow(loaded, {"age": 20}).status == "succeeded"


@pytest.mark.xfail(
    reason="The record-scoped bare boundary form `age >= 18` is an 'unknown "
    "namespace' LoadError; resolving it against the START record is deferred to "
    "ngocbh/agent-composer#93. The braced `${input.age}` form works.",
    strict=False,
)
def test_bare_record_scoped_boundary_assert_fail_fast_and_pass():
    """A bare boundary reference should resolve against the START record."""
    loaded = load_flow(_boundary_assert_flow("age >= 18"))
    assert run_flow(loaded, {"age": 10}).status == "failed"
    assert run_flow(loaded, {"age": 20}).status == "succeeded"


# =========================================================================== #
# Optional-edge and co-skip semantics behind a `case` branch
#
# A top-level post assert referencing a producer behind an UNTAKEN case branch.
# An assert on the TAKEN branch's producer makes END wait for the branch to settle,
# then not co-skip (optional edge), so the run completes rather than deadlocking.
# An ordered assert on a genuinely skipped producer fails loudly instead of passing.
#
# `gate` routes on `input.v`: v>=100 -> `hi`, else -> `lo`. With v=5, `lo` runs and
# `hi` is skip-flooded. Branches join with a coalesce `${hi.output | lo.output}`.
# =========================================================================== #

_BRANCH_ASSERT_BODY = """
nodes:
  gate:
    kind: case
    cases:
      - when: "input.v >= 100"
        then: hi
    else: lo
  hi:
    kind: code
    input:
      n: ${input.v}
    output: int
    code: |
      return inputs["n"] * 2
  lo:
    kind: code
    input:
      n: ${input.v}
    output: int
    code: |
      return inputs["n"] * 2
output: ${hi.output | lo.output}
"""


def _branch_assert_flow(assert_str: str) -> str:
    return f"""
id: c1
name: c1
input:
  v: int
{_BRANCH_ASSERT_BODY}
asserts:
  - "{assert_str}"
"""


def test_optional_edge_taken_branch_completes():
    """A post-assert on the taken branch waits without co-skipping END."""
    res = run_flow(load_flow(_branch_assert_flow("lo.output >= 0")), {"v": 5})
    assert res.status == "succeeded", res.failure
    assert res.output == 10


def test_ordered_assert_on_skipped_producer_fails_loudly():
    """A post-assert on a skipped producer fails instead of silently passing."""
    res = run_flow(load_flow(_branch_assert_flow("hi.output >= 0")), {"v": 5})
    assert res.status == "failed"
    assert res.failure is not None
    assert "assert" in res.failure.message.lower()


# =========================================================================== #
# `${system.X}` in a flow `asserts:` must raise LoadError
#
# A top-level flow assert referencing the `system` namespace must be rejected at
# load time because the ambient namespace has no producer for an assertion edge.
# =========================================================================== #


def test_system_ref_in_flow_assert_is_load_error():
    """A top-level flow assert rejects the ambient `system` namespace at load."""
    flow = f"""
id: sysf
name: sysf
input:
  age: int
{_BOUNDARY_ASSERT_ECHO_BODY}
asserts:
  - "${{system.run_id}} != null"
"""
    with pytest.raises(LoadError):
        load_flow(flow)
