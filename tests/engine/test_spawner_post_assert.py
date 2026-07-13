"""A `call` node's node-level POST `asserts:` fire from the CLONED CHILD END.

A `call` is a spawner: `run()` returns a `Grow`, its `${output}` is committed when the
cloned child END filler finishes. This branch LOWERS the spawner's node-level `asserts:`
onto that cloned child END (in the same grow builder that clones it), so they fire through
the ORDINARY `eval_node` post-assert seam — NOT from a bespoke redirect-commit site.

Because the assert now rides the cloned child END, a failing assert emits `NodeFailed`
naming the CHILD END id (e.g. `research/__end__`), not the spawner id. The lowering mirrors
build-time flow-assert lowering:

- `${output}` / `${output.field}` stays as-is (the cloned END's own record value == the
  spawner's committed output, injected by eval_node as `{"output": value}`).
- a declared call input `${X}` maps to a synth END param wired off the cloned child START's
  coerced output (`${<callsite>/__start__.output.X:-null}`), with a structural
  `<callsite>/__start__ -> <callsite>/__end__` edge — the same shape build-time uses.

Resume-safety: `replay_grow` rebuilds via the SAME `call_subgraph`, so a resumed run's
rebuilt child END carries identical lowered asserts and still fires the post-assert.
"""

from agent_composer.compose import load_flow
from agent_composer.run import resume_command, resume_flow, run_flow
from agent_composer.events import RunFailed
from agent_composer.suspension.checkpoint import RunCheckpoint


# A child that doubles its int input and re-exports it as `${output}` (an int).
_ECHO_CHILD = """
id: echo-child
name: echo_child
input:
  v: int
nodes:
  x:
    kind: code
    code: tests.engine._compose_codefns:double
    input:
      n: ${input.v}
    output: int
output: ${x.output}
"""


def _resolver(**children):
    loaded = {fid: load_flow(text) for fid, text in children.items()}

    def resolve(flow_id, version=None):
        return loaded[flow_id]

    return resolve


# =========================================================================== #
# 1. `${output}` post-assert fires from the cloned child END
# =========================================================================== #

def _output_parent(value: int) -> str:
    # A parent that calls echo-child with a fixed value, carrying a node-level
    # `${output} < 100` post-assert. child doubles v -> output == 2*value.
    return f"""
id: out-parent
name: out_parent
uses:
  echo-child: echo-child
nodes:
  callee:
    kind: call
    call: echo-child
    input:
      v: {value}
    asserts:
      - ${{output}} < 100
output: ${{callee.output}}
"""


def _load_output(value: int):
    return load_flow(_output_parent(value),
                     child_resolver=_resolver(**{"echo-child": _ECHO_CHILD}))


def test_output_post_assert_satisfying_succeeds():
    # v=5 -> output 10 < 100 -> the assert holds, the run succeeds.
    res = run_flow(_load_output(5), {})
    assert res.status == "succeeded", res.error
    assert res.output == 10


def test_output_post_assert_violating_fails_on_child_end():
    # v=60 -> output 120, violates ${output} < 100 -> loud failure fired on the CHILD END.
    res = run_flow(_load_output(60), {})
    assert res.status == "failed"
    failed = [e for e in res.events if isinstance(e, RunFailed)]
    assert failed and failed[0].error_type == "NodeAssertFailed"
    assert "post-assert" in (res.error or "").lower()
    # The failing node id identifies the cloned CHILD END (callsite + end id), NOT the spawner.
    loc = res.locator
    assert loc is not None and loc.kind == "assert"
    assert loc.node == "callee/__end__"


# =========================================================================== #
# 2. a declared-input post-assert maps to the cloned child START's output
# =========================================================================== #

def _input_parent(seed_expr: str) -> str:
    # The call declares an input `v` seeded from the run arg; the node-level POST assert reads
    # BOTH that DECLARED INPUT (`${v}`) AND `${output}` (so it is a POST assert, fired on the
    # cloned child END). The lowering maps `${v}` onto the cloned child START's coerced output
    # so the child END can resolve it.
    return f"""
id: in-parent
name: in_parent
uses:
  echo-child: echo-child
input:
  v: int
nodes:
  callee:
    kind: call
    call: echo-child
    input:
      v: {seed_expr}
    asserts:
      - ${{v}} >= 0 and ${{output}} > 0
output: ${{callee.output}}
"""


def _load_input():
    return load_flow(_input_parent("${input.v}"),
                     child_resolver=_resolver(**{"echo-child": _ECHO_CHILD}))


def test_declared_input_post_assert_satisfying_succeeds():
    # v=5 >= 0 -> holds; output 10.
    res = run_flow(_load_input(), {"v": 5})
    assert res.status == "succeeded", res.error
    assert res.output == 10


def test_declared_input_post_assert_violating_fails():
    # v=-3 < 0 -> ${v} >= 0 violated, loud failure on the child END.
    res = run_flow(_load_input(), {"v": -3})
    assert res.status == "failed"
    failed = [e for e in res.events if isinstance(e, RunFailed)]
    assert failed and failed[0].error_type == "NodeAssertFailed"
    assert res.locator is not None and res.locator.node == "callee/__end__"


# =========================================================================== #
# 3. resume-safety: the lowered spawner post-assert survives a checkpoint hop
# =========================================================================== #

# A child that PAUSES before its END (a human_input), so the run suspends mid-child.
_PAUSING_CHILD = """
id: approver
name: approver
input:
  v: int
nodes:
  approve:
    kind: human_input
    input:
      v: ${input.v}
    prompt: "value ${v}?"
    output: int
output: ${approve.output}
"""


def _pausing_parent() -> str:
    # The call carries a node-level ${output} < 100 post-assert; the child pauses on a
    # human_input before its END, so the assert can only fire on the RESUMED leg.
    return """
id: pause-parent
name: pause_parent
uses:
  approver: approver
input:
  v: int
nodes:
  gate:
    kind: call
    call: approver
    input:
      v: ${input.v}
    asserts:
      - ${output} < 100
output: ${gate.output}
"""


def _load_pausing():
    return load_flow(_pausing_parent(),
                     child_resolver=_resolver(**{"approver": _PAUSING_CHILD}))


def test_spawner_post_assert_fires_on_resume():
    # Run -> pause -> cross-process snapshot/dumps -> loads/restore -> resume. The human
    # delivers 120, violating ${output} < 100 -> the rebuilt child END fires the post-assert
    # on the RESUMED leg and the run FAILS. Proves replay_grow re-lowers the same asserts.
    loaded = _load_pausing()
    rec = run_flow(loaded, {"v": 1})
    assert rec.status == "paused"
    reason = rec.pause_reasons[0]
    ckpt = RunCheckpoint.loads(rec.checkpoint.dumps())
    fresh = _load_pausing()
    cmd = resume_command(fresh, reason, 120)
    res = resume_flow(fresh, checkpoint=ckpt, commands=[cmd])
    assert res.status == "failed"
    failed = [e for e in res.events if isinstance(e, RunFailed)]
    assert failed and failed[0].error_type == "NodeAssertFailed"
    assert "post-assert" in (res.error or "").lower()
