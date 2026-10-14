"""A `call` node's node-level POST `asserts:` must fire — on the CLONED CHILD END.

A `call` is a spawner: its `run()` returns `Grow`, and its `${output}` is committed when the
cloned child END filler finishes (`commit_as=callsite`). `call_subgraph` lowers the call's
node-level post-asserts onto that END. When `EndNode.forward()` returns `Output`, the inherited
`Node.run` wrapper evaluates them against the END's bound inputs and output. A false assertion
therefore fails the child END (for example `research/__end__`), which owns the committed call
output and its postconditions.

These pin: a true call post-assert passes, a false one fails the run LOUDLY (RunFailed,
failure cause type "NodeAssertFailed", with the assertion origin on the child END), a post-assert
reading BOTH a declared call input AND `${output}` fires correctly (the input mapped onto the
cloned child START's coerced output), and (durability) the lowered assert survives authoritative
log recovery — the injected builder re-lowers it onto the replayed child END.
"""

from pathlib import Path

import pytest

from yian.compose import load_flow
from yian.events import RunFailed
from yian.nodes.base import _HostStop
from yian.run import resume_flow, run_flow


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
    loaded = _load("      - output.n == 4")
    res = run_flow(loaded, {})
    assert res.status == "succeeded", res.failure
    assert res.output == {"report": "report for ACME", "n": 4}


# --- 2. live, false -> loud failure ----------------------------------------- #


def test_call_post_assert_false_fails_loudly():
    # A false `${output}` post-assert fails on the CLONED CHILD END
    # (`research/__end__`), where the lowered assertion now rides.
    loaded = _load("      - output.n == 999")
    res = run_flow(loaded, {})
    assert res.status == "failed"
    failed = [e for e in res.events if isinstance(e, RunFailed)]
    assert failed and failed[0].failure.cause_type == "NodeAssertFailed"
    assert res.failure.node_id == "research/__end__"
    assert loaded.diagnose(res.failure).frames


# --- 3. live, input + output ref -------------------------------------------- #


def test_call_post_assert_reads_input_and_output_true():
    # a post-assert reading BOTH a declared call input (`topic`) AND `${output}` -> true holds.
    # n == len("ACME") == 4, so `len(topic) == n` is satisfied via the recovered input record.
    loaded = _load("      - output.n == 4")  # baseline true
    res = run_flow(loaded, {})
    assert res.status == "succeeded", res.failure


def test_call_post_assert_reads_input_and_output_false():
    # an input+output post-assert that is FALSE -> loud failure. `topic` ("ACME") is the
    # recovered call input; `${output.report}` mentions it but `topic == "OTHER"` is false.
    loaded = _load('      - topic == "OTHER" and output.n == 4')
    res = run_flow(loaded, {})
    assert res.status == "failed"
    failed = [e for e in res.events if isinstance(e, RunFailed)]
    assert failed and failed[0].failure.cause_type == "NodeAssertFailed"
    assert res.failure.node_id == "research/__end__"


def test_call_post_assert_reads_input_true():
    # The input ref resolves from the call's canonical growth seed (not None/dropped).
    loaded = _load('      - topic == "ACME" and output.n == 4')
    res = run_flow(loaded, {})
    assert res.status == "succeeded", res.failure


# --- 4. durable: the input record survives interruption/recovery ------------- #
#
# A `call` whose child contains a live HUMAN_INPUT leaf. The call's post-assert reads BOTH
# a declared input (`action`) and `${output}`. A host interruption leaves the input node's
# durable start unfinished; log recovery rebuilds the expansion, retries that node, and fires
# the post-assert over the recorded call input. A false assert fails; a true one succeeds.

_BLOCKING_CHILD = """
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


def _blocking_parent(asserts: str) -> str:
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


def _load_blocking(asserts: str):
    return load_flow(
        _blocking_parent(asserts),
        child_resolver=_resolver(**{"approver": _BLOCKING_CHILD}),
    )


def _drive_durable(
    tmp_path: Path,
    asserts: str,
    answer: str,
):
    """Interrupt one started HUMAN_INPUT, then recover it from its run log.

    The post-assert fires after replay over the recovered call input record.
    """
    path = tmp_path / "call-post-assert.jsonl"
    interruption = _HostStop("interrupt before human answer")

    def interrupt(**request):
        raise interruption

    with pytest.raises(_HostStop) as caught:
        run_flow(
            _load_blocking(asserts),
            {"action": "deploy"},
            log=path,
            tools={"ask_user": interrupt},
        )
    assert caught.value is interruption

    return resume_flow(
        _load_blocking(asserts), log=path, tools={"ask_user": lambda **request: answer}
    )


def test_call_post_assert_survives_recovery_true(tmp_path: Path):
    # A true input+output post-assert fires after recovery. The call's input record
    # (`action`) comes from the authoritative NodeExpanded seed.
    res = _drive_durable(
        tmp_path,
        '      - action == "deploy" and output == "yes"',
        "yes",
    )
    assert res.status == "succeeded", res.failure
    assert res.output == "yes"


def test_call_post_assert_survives_recovery_false(tmp_path: Path):
    # A false post-assert after recovery fails and remains located at the cloned child END.
    res = _drive_durable(
        tmp_path,
        '      - action == "deploy" and output == "no"',
        "yes",
    )
    assert res.status == "failed"
    failed = [e for e in res.events if isinstance(e, RunFailed)]
    assert failed and failed[0].failure.cause_type == "NodeAssertFailed"
    assert res.failure.node_id == "gate/__end__"
