"""Target-behavior regression tests for the merge-asserts refactor.

BIG PICTURE
    We are merging all flow-level `asserts:` into node-level `pre_asserts`/
    `post_asserts` on the flow's synthesized START/END boundary nodes. Today asserts
    live in two grammars across several firing sites; the refactor unifies them so
    `eval_node` fires every assert node-locally.

    This file captures the TARGET behavior. The refactor code lands in later steps,
    so the behaviors that only become correct AFTER the refactor are marked
    `@pytest.mark.xfail(reason=..., strict=False)`: a strict=False xfail that starts
    passing is reported XPASS (not a failure), so these flip cleanly once the code
    lands. Behaviors that already hold today are plain (green) tests that document
    current, must-keep contracts.

    Each test names the concern it locks (C1 / C2 / C3 / system-forbid) and whether
    it is expected to pass-today or xfail-now-pass-after.

CONCERNS
    C2  a `def`'s top-level BARE assert (`emit.output.n >= 1`, no `${}`) reading an
        INTERNAL producer must succeed when the def runs via call / map / loop. Today
        it FAILS ("post-assert failed: emit.output.n >= 1") even when true -> xfail.
    C3  a top-level boundary assert `age >= 18` (record-scoped bare form) fail-fasts
        for {age:10} and succeeds for {age:20}. The braced pool form (`${input.age}`)
        works today; the bare record-scoped form is the refactor target.
    C1  optional-edge / co-skip semantics for a post assert behind a `case` branch.
    system-forbid  a `${system.X}` in a flow `asserts:` must raise LoadError at load.
"""

import pytest

from agent_composer.compose import LoadError, load_flow
from agent_composer.run import run_flow


# =========================================================================== #
# C2 — a def's top-level BARE assert on an INTERNAL producer, run via a spawner
#
# The core bug. A def declares `asserts: - "emit.output.n >= 1"` (BARE, no braces)
# reading its own internal producer `emit`. When the def runs via kind:call / map /
# loop, today the child END fires it as "emit.output.n >= 1" against a pool that
# namespaces internal producers differently -> a TRUE assert fails the run.
#
# The BARE form is load-bearing: the braced form `${emit.output.n} >= 1` gets
# rewritten by `_rens_internal` and passes TODAY, so a braced test proves nothing.
# All C2 assertions below are xfail(strict=False): they FAIL today, XPASS after.
# =========================================================================== #

# A def whose top-level assert reads an internal producer (`emit`) in BARE form.
_C2_CHILD_DEF = """
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


def test_c2_call_bare_internal_assert_succeeds():
    """C2 (call).

    A `call` invoking a def whose top-level BARE assert reads an internal producer.
    seed=5 -> emit.output.n == 6 >= 1 holds, so the run MUST succeed. Before the
    refactor this failed with "node 'callee/__end__' post-assert failed: emit.output.n >= 1".
    """
    flow = f"""
id: c2call
name: c2_call
input:
  seed: int
{_C2_CHILD_DEF}
nodes:
  callee:
    kind: call
    call: child
    input:
      seed: ${{input.seed}}
output: ${{callee.output}}
"""
    res = run_flow(load_flow(flow), {"seed": 5})
    assert res.status == "succeeded", res.error
    assert res.output == {"n": 6}


def test_c2_map_bare_internal_assert_all_satisfy_succeeds():
    """C2 (map, all satisfy).

    A `map` fans the def over a list; every element's internal-producer assert holds
    (seed>=0 -> n>=1), so the run MUST succeed with the over-order join. Before the
    refactor the child END mis-fired the BARE assert on element 0 and failed.
    """
    flow = f"""
id: c2map
name: c2_map
input:
  seeds: list[int]
{_C2_CHILD_DEF}
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
    assert res.status == "succeeded", res.error
    assert res.output == [{"n": 6}, {"n": 7}, {"n": 8}]


def test_c2_map_bare_internal_assert_one_element_violates_fails_loudly():
    """C2 (map, one violating element), pass-today AND pass-after.

    A genuinely-violating element (seed=-5 -> n=-4 < 1) must fail the run LOUDLY.
    This fails today (for the wrong reason: the C2 bug mis-fires on element 0) and
    must still fail after the refactor (for the RIGHT reason: element -5 violates),
    so it stays a plain green test on both sides — we only assert loud failure, not
    the message.
    """
    flow = f"""
id: c2mapv
name: c2_map_violate
input:
  seeds: list[int]
{_C2_CHILD_DEF}
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
    assert "assert" in (res.error or "").lower()


def test_c2_loop_bare_internal_assert_succeeds():
    """C2 (loop).

    A `loop` whose BODY def carries a top-level BARE assert on an internal producer
    (`step.output.n >= 1`). The body threads {n:int}; with start=5 every iteration's
    producer holds, so times:3 MUST succeed (final n == 8). Before the refactor each
    iteration's child END mis-fired the BARE assert and failed.
    """
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
    assert res.status == "succeeded", res.error
    assert res.output == {"n": 8}


# =========================================================================== #
# C3 — a top-level boundary assert, record-scoped
#
# Today's boundary asserts use the braced pool form `${input.age} >= 18`. The
# refactor rewrites them to the record-scoped BARE form `age >= 18` (or `${age}`),
# resolved against the START record. We keep a braced test green (documents today)
# and add the bare record-scoped target (xfail, since `age`/`${age}` is an "unknown
# namespace" LoadError today).
# =========================================================================== #

_C3_ECHO_BODY = """
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


def _c3_flow(assert_str: str) -> str:
    return f"""
id: c3
name: c3
input:
  age: int
{_C3_ECHO_BODY}
asserts:
  - "{assert_str}"
"""


def test_c3_braced_boundary_assert_fail_fast_and_pass():
    """C3 (braced pool form), pass-today.

    Documents the CURRENT top-level boundary spelling `${input.age} >= 18`:
    fail-fast for {age:10} (status "failed"), succeed for {age:20}.
    """
    loaded = load_flow(_c3_flow("${input.age} >= 18"))
    assert run_flow(loaded, {"age": 10}).status == "failed"
    assert run_flow(loaded, {"age": 20}).status == "succeeded"


@pytest.mark.xfail(
    reason="C3: the record-scoped BARE boundary form `age >= 18` is an 'unknown "
    "namespace' LoadError today; resolving it against the START record is deferred "
    "to a follow-up (ngocbh/agent-composer#93). The braced `${input.age}` form works.",
    strict=False,
)
def test_c3_bare_record_scoped_boundary_assert_fail_fast_and_pass():
    """C3 (bare record-scoped target), xfail-now / pass-after.

    The refactor target: `age >= 18` resolved record-scoped against the START
    record. Must fail-fast for {age:10} and succeed for {age:20}. Today `load_flow`
    raises LoadError ("unknown namespace 'age'"), which the xfail records.
    """
    loaded = load_flow(_c3_flow("age >= 18"))
    assert run_flow(loaded, {"age": 10}).status == "failed"
    assert run_flow(loaded, {"age": 20}).status == "succeeded"


# =========================================================================== #
# C1 — optional edge / co-skip semantics for a post assert behind a `case` branch
#
# A top-level post assert referencing a producer behind an UNTAKEN case branch.
# Two sub-behaviors:
#   (A) an assert on the TAKEN branch's producer: END must WAIT for the branch to
#       settle then NOT co-skip (optional edge) — the run completes, not deadlocks.
#   (B) an ORDERED assert on a GENUINELY-SKIPPED producer must fail LOUDLY
#       (documented behavior, not a silent pass).
#
# `gate` routes on `input.v`: v>=100 -> `hi`, else -> `lo`. With v=5, `lo` runs and
# `hi` is skip-flooded. Branches join with a coalesce `${hi.output | lo.output}`.
# =========================================================================== #

_C1_BODY = """
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


def _c1_flow(assert_str: str) -> str:
    return f"""
id: c1
name: c1
input:
  v: int
{_C1_BODY}
asserts:
  - "{assert_str}"
"""


def test_c1_optional_edge_taken_branch_completes():
    """C1 (A) optional edge, pass-today AND pass-after.

    A post assert on the TAKEN branch's producer (`lo.output >= 0`) while the other
    branch (`hi`) is skipped. END must WAIT for the branch to settle and NOT co-skip
    to dead — the run COMPLETES (status "succeeded", output 10), it does not deadlock
    or drop END. Documents the optional-edge contract the refactor must preserve.
    """
    res = run_flow(load_flow(_c1_flow("lo.output >= 0")), {"v": 5})  # -> lo taken
    assert res.status == "succeeded", res.error
    assert res.output == 10


def test_c1_ordered_assert_on_skipped_producer_fails_loudly():
    """C1 (B) skipped-producer ordered assert, pass-today AND pass-after.

    An ORDERED assert (`hi.output >= 0`) on a GENUINELY-SKIPPED producer (`hi` is
    the untaken branch when v=5) must fail LOUDLY — it must NOT silently pass. The
    run status is "failed". This is the documented behavior on both sides.
    """
    res = run_flow(load_flow(_c1_flow("hi.output >= 0")), {"v": 5})  # hi skipped
    assert res.status == "failed"
    assert "assert" in (res.error or "").lower()


# =========================================================================== #
# system-forbid — `${system.X}` in a flow `asserts:` must raise LoadError
#
# A top-level flow assert referencing the `system` namespace must be rejected at
# LOAD time. This is NOT enforced today (the flow loads fine), so the test is
# xfail(strict=False): after the refactor's check lands, load_flow raises LoadError
# and the pytest.raises succeeds -> the xfail flips to XPASS.
# =========================================================================== #


def test_system_ref_in_flow_assert_is_load_error():
    """system-forbid.

    A `${system.X}` ref in a top-level flow `asserts:` must raise LoadError at load —
    it is ambient, so there is no producer to draw a synth END edge from.
    """
    flow = f"""
id: sysf
name: sysf
input:
  age: int
{_C3_ECHO_BODY}
asserts:
  - "${{system.run_id}} != null"
"""
    with pytest.raises(LoadError):
        load_flow(flow)
