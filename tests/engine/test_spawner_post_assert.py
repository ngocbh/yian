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

Recovery-safety: authoritative log replay rebuilds via the SAME `call_subgraph`, so a
recovered run's child END carries identical lowered asserts and still fires the post-assert.
"""

from pathlib import Path

import pytest

from agent_composer.compile.expand import _SYNTH_PREFIX, call_subgraph
from agent_composer.compose import load_flow
from agent_composer.diagnostics import OriginRef, SourceRange
from agent_composer.events import RunFailed
from agent_composer.nodes.base import _HostStop
from agent_composer.nodes.contracts import Assertion
from agent_composer.run import resume_flow, run_flow
from agent_composer.tools import install_ask_user


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
# 3. recovery-safety: the lowered spawner post-assert survives log replay
# =========================================================================== #

# A child whose live HUMAN_INPUT can be interrupted after its durable start.
_BLOCKING_CHILD = """
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


def _blocking_parent() -> str:
    # The call carries a node-level ${output} < 100 post-assert. Recovery retries the
    # unfinished HUMAN_INPUT, then the rebuilt child END evaluates this assertion.
    return """
id: recovery-parent
name: recovery_parent
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


def _load_blocking():
    return load_flow(
        _blocking_parent(),
        child_resolver=_resolver(**{"approver": _BLOCKING_CHILD}),
    )


def test_spawner_post_assert_fires_after_log_recovery(tmp_path: Path):
    # Interrupt the live HUMAN_INPUT after its NodeStarted is durable. Recovery replays the
    # expansion, retries the whole input node, and delivers 120. The rebuilt child END must
    # then reject ${output} < 100 through its ordinary post-assert path.
    path = tmp_path / "spawner-post-assert.jsonl"
    interruption = _HostStop("interrupt before human answer")

    def interrupt(**request):
        raise interruption

    with install_ask_user(interrupt), pytest.raises(_HostStop) as caught:
        run_flow(_load_blocking(), {"v": 1}, log=path)
    assert caught.value is interruption

    with install_ask_user(lambda **request: 120):
        res = resume_flow(_load_blocking(), log=path)

    assert res.status == "failed"
    failed = [e for e in res.events if isinstance(e, RunFailed)]
    assert failed and failed[0].error_type == "NodeAssertFailed"
    assert "post-assert" in (res.error or "").lower()


# =========================================================================== #
# 4. collision-free synth alloc vs a child def's OWN lowered `__assert_input_v`
# =========================================================================== #

# A child def that (a) doubles its input AND (b) carries its OWN flow-level POST assert
# reading `${input.v}`. Because a flow assert is POST (rides the def's END) only when it
# also reads a produced value, it reads `${x.output}` too. Build-time lowers that flow
# assert onto the def's END as a synth param `__assert_input_v` (+ its wiring + a
# producer->END edge). A `call` to this def ALSO carrying a node-level POST assert reading
# the SAME declared input `${v}` used to allocate the IDENTICAL `__assert_input_v` name a
# SECOND time onto the cloned END (a duplicate ParamDecl + a phantom producer->END edge
# sharing the input_group).
_ECHO_CHILD_WITH_INPUT_ASSERT = """
id: echo-child
name: echo_child
input:
  v: int
asserts:
  - "${input.v} >= 0 and ${x.output} >= 0"
nodes:
  x:
    kind: code
    code: tests.engine._compose_codefns:double
    input:
      n: ${input.v}
    output: int
output: ${x.output}
"""


def _collision_parent() -> str:
    # The call declares input `v` seeded from the run arg; its node-level POST assert reads
    # the SAME declared input `${v}` (`< 100`). The child def already lowered `__assert_input_v`
    # for its OWN `${input.v} >= 0` flow assert — so the spawner lowering must NOT reuse that name.
    return """
id: collision-parent
name: collision_parent
uses:
  echo-child: echo-child
input:
  v: int
nodes:
  callee:
    kind: call
    call: echo-child
    input:
      v: ${input.v}
    asserts:
      - "${v} < 100 and ${output} > 0"
output: ${callee.output}
"""


def _load_collision():
    return load_flow(
        _collision_parent(),
        child_resolver=_resolver(**{"echo-child": _ECHO_CHILD_WITH_INPUT_ASSERT}),
    )


def _cloned_child_end(loaded):
    """Build the CALL's grow subgraph directly (via `call_subgraph`, the SAME builder the
    CallNode uses) and return the cloned child END node + the flow's edges, so a test can
    inspect the cloned END's structural integrity."""
    callee = loaded.compiled.nodes["callee"]
    flow = call_subgraph(
        callee.child,
        callsite="callee",
        record={"v": "${input.v}"},
        post_asserts=callee.post_asserts,
        post_assert_ranges=callee.post_assert_ranges,
        input_names={p.name for p in (callee.params or [])},
    )
    return flow.nodes[flow.end_id], flow


def test_collision_free_cloned_end_has_no_duplicate_params_or_edges():
    # Structural regression: the cloned child END must NOT carry a duplicate synth param nor a
    # phantom second producer->END edge sharing an input_group. This FAILS against the pre-fix
    # code (the def's `__assert_input_v` survives the clone, then the spawner lowering allocates
    # `__assert_input_v` AGAIN — duplicate ParamDecl + phantom edge).
    loaded = _load_collision()
    end, flow = _cloned_child_end(loaded)

    names = [p.name for p in end.params]
    assert len(names) == len(set(names)), f"duplicate END param names: {names}"

    # No two edges into the END share an input_group (a phantom duplicate feeds the 3-state join).
    groups = [e.input_group for e in flow.edges
              if e.to == end.id and e.input_group is not None]
    assert len(groups) == len(set(groups)), f"duplicate input_group into END: {groups}"

    # Both the child's own `__assert_input_v` and the spawner's synth (bumped to `_1`) are present.
    synths = [n for n in names if n.startswith(f"{_SYNTH_PREFIX}input_v")]
    assert len(synths) == 2, f"expected two distinct v-synth params, got {synths}"


def test_spawner_lowering_preserves_presentation_origin_and_range():
    """CALL expansion rewrites only executable text and keeps occurrence diagnostics aligned."""
    loaded = _load_output(5)
    callee = loaded.compiled.nodes["callee"]
    origin = OriginRef("nodes/callee/asserts/0")
    assertion_range = SourceRange(8, 7, 8, 38)
    assertion = Assertion(
        "${v} >= 0 and ${output} > 0",
        "callee input and output must be positive",
        origin,
    )

    flow = call_subgraph(
        callee.child,
        callsite="callee",
        record={"v": 5},
        post_asserts=[assertion],
        post_assert_ranges=[assertion_range],
        input_names={"v"},
    )
    lowered = flow.nodes[flow.end_id]

    assert lowered.post_asserts[-1].expression.startswith("${__assert_input_v}")
    assert lowered.post_asserts[-1].display == assertion.display
    assert lowered.post_asserts[-1].origin is origin
    assert lowered.post_assert_ranges[-1] == assertion_range


def test_collision_case_run_satisfying_succeeds():
    # v=5: satisfies BOTH `${input.v} >= 0` (child) and `${v} < 100` (spawner). output == 10.
    res = run_flow(_load_collision(), {"v": 5})
    assert res.status == "succeeded", res.error
    assert res.output == 10


def test_collision_case_run_violating_spawner_assert_fails_on_child_end():
    # v=120: satisfies the CHILD's `>= 0` but violates the SPAWNER's `< 100` -> post-assert
    # failure fired on the cloned CHILD END.
    res = run_flow(_load_collision(), {"v": 120})
    assert res.status == "failed"
    failed = [e for e in res.events if isinstance(e, RunFailed)]
    assert failed and failed[0].error_type == "NodeAssertFailed"
    assert "post-assert" in (res.error or "").lower()
    assert res.locator is not None and res.locator.node == "callee/__end__"


# =========================================================================== #
# 5. repeated refs to the SAME declared input share ONE synth param
# =========================================================================== #

def _repeated_ref_parent() -> str:
    # The node-level POST assert references the SAME declared input `${v}` TWICE — the lowering
    # must allocate ONE synth param for it (the `synth_for_input` per-pass dedup).
    return """
id: repeat-parent
name: repeat_parent
uses:
  echo-child: echo-child
input:
  v: int
nodes:
  callee:
    kind: call
    call: echo-child
    input:
      v: ${input.v}
    asserts:
      - "${v} >= 0 and ${v} < 100 and ${output} > 0"
output: ${callee.output}
"""


def _load_repeated():
    # Reuse the plain `_ECHO_CHILD` (no child-level asserts) so only the spawner's synth appears.
    return load_flow(
        _repeated_ref_parent(),
        child_resolver=_resolver(**{"echo-child": _ECHO_CHILD}),
    )


def test_repeated_input_ref_shares_one_synth_param():
    loaded = _load_repeated()
    end, _flow = _cloned_child_end(loaded)
    synths = [p.name for p in end.params if p.name.startswith(f"{_SYNTH_PREFIX}input_v")]
    assert len(synths) == 1, f"expected ONE v-synth param, got {synths}"


def test_repeated_input_ref_run_succeeds():
    res = run_flow(_load_repeated(), {"v": 5})
    assert res.status == "succeeded", res.error
    assert res.output == 10
