"""A `call` node's node-level POST `asserts:` must fire — on the CLONED CHILD END.

A `call` is a spawner: it returns a `Grow` (eval_node returns BEFORE the post-assert
block), and its `${output}` is committed LATER, when the cloned child's END filler finishes
(`commit_as=callsite`). Its node-level POST asserts are LOWERED onto that cloned child END (in
`call_subgraph`), so they fire through the ORDINARY `eval_node` post-assert seam — NOT a bespoke
redirect-commit site in the engine. A false one therefore fails the run LOUDLY, naming the CHILD
END id (e.g. `research/__end__`), not the spawner. Before this branch they fired from the
redirect-commit site (naming the spawner); before an earlier fix they were silently dropped.

These pin: a true call post-assert passes, a false one fails the run LOUDLY (RunFailed,
error_type "NodeAssertFailed", a non-null locator at the assert on the child END), a post-assert
reading BOTH a declared call input AND `${output}` fires correctly (the input mapped onto the
cloned child START's coerced output), and (durability) the lowered assert survives a checkpoint
round-trip — the injected builder re-lowers it onto the rebuilt child END, with no new field.
"""

from agent_composer.compose import load_flow
from agent_composer.run import resume_command, resume_flow, run_flow
from agent_composer.events import RunFailed
from agent_composer.suspension.checkpoint import RunCheckpoint


# A child that re-exports a {report, n} record (the Ollama-free CODE child pattern). The
# parent `call`s it; the call's `${output}` is that {report, n} record. `n == len(topic)`.
_CHILD = """
id: child-one
name: child_one
input:
  topic: str
nodes:
  emit:
    kind: code
    input:
      topic: ${input.topic}
    output:
      report: str
      n: int
    code: tests.engine._compose_codefns:make_report
output:
  report: ${emit.output.report}
  n: ${emit.output.n}
"""


def _resolver(**children):
    loaded = {fid: load_flow(text) for fid, text in children.items()}

    def resolve(flow_id, version=None):
        return loaded[flow_id]

    return resolve


def _parent(asserts: str) -> str:
    # A parent that `call`s child-one with a fixed topic, carrying the given `asserts:` block
    # on the call node. `topic` ("ACME") is a declared call input -> n == 4.
    return f"""
id: call-parent
name: call_parent
uses:
  child-one: child-one
nodes:
  research:
    kind: call
    call: child-one
    input:
      topic: ACME
    asserts:
{asserts}
output: ${{research.output}}
"""


def _load(asserts: str):
    return load_flow(_parent(asserts), child_resolver=_resolver(**{"child-one": _CHILD}))


# --- 1. live, passing -------------------------------------------------------- #


def test_call_post_assert_true_passes():
    # a true `${output.field}` post-assert -> the run succeeds.
    loaded = _load("      - ${output.n} == 4")
    res = run_flow(loaded, {})
    assert res.status == "succeeded", res.error
    assert res.output == {"report": "report for ACME", "n": 4}


# --- 2. live, false -> loud failure ----------------------------------------- #


def test_call_post_assert_false_fails_loudly():
    # a false `${output}` post-assert -> RunFailed, error_type NodeAssertFailed, a non-null
    # locator pinned at the assert on the CLONED CHILD END (`research/__end__`), where the
    # lowered assert now rides. (Before this branch it named the spawner `research`; before an
    # earlier fix this passed SILENTLY.)
    loaded = _load("      - ${output.n} == 999")
    res = run_flow(loaded, {})
    assert res.status == "failed"
    failed = [e for e in res.events if isinstance(e, RunFailed)]
    assert failed and failed[0].error_type == "NodeAssertFailed"
    loc = res.locator
    assert loc is not None
    assert loc.node == "research/__end__" and loc.kind == "assert"
    assert loc.key == "${output.n} == 999"


# --- 3. live, input + output ref -------------------------------------------- #


def test_call_post_assert_reads_input_and_output_true():
    # a post-assert reading BOTH a declared call input (`topic`) AND `${output}` -> true holds.
    # n == len("ACME") == 4, so `len(topic) == n` is satisfied via the recovered input record.
    loaded = _load("      - ${output.n} == 4")  # baseline true
    res = run_flow(loaded, {})
    assert res.status == "succeeded", res.error


def test_call_post_assert_reads_input_and_output_false():
    # an input+output post-assert that is FALSE -> loud failure. `topic` ("ACME") is the
    # recovered call input; `${output.report}` mentions it but `${topic} == "OTHER"` is false.
    loaded = _load('      - ${topic} == "OTHER" and ${output.n} == 4')
    res = run_flow(loaded, {})
    assert res.status == "failed"
    failed = [e for e in res.events if isinstance(e, RunFailed)]
    assert failed and failed[0].error_type == "NodeAssertFailed"
    assert res.locator is not None and res.locator.node == "research/__end__"


def test_call_post_assert_reads_input_true():
    # the input ref resolves from the recovered GrowRecord.seed (not None/dropped).
    loaded = _load('      - ${topic} == "ACME" and ${output.n} == 4')
    res = run_flow(loaded, {})
    assert res.status == "succeeded", res.error


# --- 4. durable: the input record survives suspend/resume -------------------- #
#
# A `call` whose child contains a HUMAN_INPUT (so the run suspends mid-child). The call's
# post-assert reads BOTH a declared input (`action`) and `${output}`. Driving
# run_flow -> pause -> snapshot/dumps -> loads/restore -> resume must fire the post-assert on
# the RESUMED leg, recovering the input record from the persisted GrowRecord.seed (no new
# checkpoint field). A false assert -> RunFailed on resume; a true one -> success.

_PAUSING_CHILD = """
id: approver
name: approver
input:
  action: str
nodes:
  approve:
    kind: human_input
    input:
      action: ${input.action}
    prompt: "Approve ${action}?"
    output: str
output: ${approve.output}
"""


def _pausing_parent(asserts: str) -> str:
    return f"""
id: gate-parent
name: gate_parent
uses:
  approver: approver
input:
  action: str
nodes:
  gate:
    kind: call
    call: approver
    input:
      action: ${{input.action}}
    asserts:
{asserts}
output: ${{gate.output}}
"""


def _load_pausing(asserts: str):
    return load_flow(_pausing_parent(asserts),
                     child_resolver=_resolver(**{"approver": _PAUSING_CHILD}))


def _drive_durable(asserts: str, answer: str):
    """run -> pause -> cross-process snapshot/dumps -> loads/restore -> resume, returning the
    resumed RunResult. The post-assert fires on the resumed leg over the recovered record."""
    loaded = _load_pausing(asserts)
    rec = run_flow(loaded, {"action": "deploy"})
    assert rec.status == "paused"
    reason = rec.pause_reasons[0]
    ckpt = RunCheckpoint.loads(rec.checkpoint.dumps())  # cross-process round-trip
    fresh = load_flow(_pausing_parent(asserts),
                      child_resolver=_resolver(**{"approver": _PAUSING_CHILD}))
    cmd = resume_command(fresh, reason, answer)
    return resume_flow(fresh, checkpoint=ckpt, commands=[cmd])


def test_call_post_assert_survives_resume_true():
    # a TRUE input+output post-assert fires on the RESUMED leg -> success. Proves the call's
    # input record (`action`) is recovered from GrowRecord.seed after the checkpoint hop.
    res = _drive_durable('      - ${action} == "deploy" and ${output} == "yes"', "yes")
    assert res.status == "succeeded", res.error
    assert res.output == "yes"


def test_call_post_assert_survives_resume_false():
    # a FALSE post-assert on the resumed leg -> RunFailed, NodeAssertFailed, located at the call.
    res = _drive_durable('      - ${action} == "deploy" and ${output} == "no"', "yes")
    assert res.status == "failed"
    failed = [e for e in res.events if isinstance(e, RunFailed)]
    assert failed and failed[0].error_type == "NodeAssertFailed"
    assert res.locator is not None and res.locator.node == "gate/__end__" and res.locator.kind == "assert"
