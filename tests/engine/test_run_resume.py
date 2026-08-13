import pytest

from agent_composer.compose.loader import load_flow
from agent_composer.run import resume_command, run_flow, resume_flow
from agent_composer.suspension.commands import DeliverAnswerCommand
from agent_composer.suspension.checkpoint import RunCheckpoint
from agent_composer.suspension.pause import EventAwaited
from agent_composer.tools import install_ask_user

FLOW = """
id: f
name: f
input:
  settle_at: date
nodes:
  settle:
    kind: wait
    until: ${input.settle_at}
  done:
    kind: code
    depends_on: [settle]
    output: str
    code: tests.seeds.fns:confirm_action
output: ${done.output}
"""


def test_run_flow_pauses_carries_handles():
    loaded = load_flow(FLOW)
    res = run_flow(loaded, {"settle_at": "2026-07-01"})
    assert res.status == "paused"
    assert res.checkpoint is not None
    assert res.engine is not None
    assert len(res.pause_reasons) == 1


def test_resume_via_checkpoint_completes():
    loaded = load_flow(FLOW)
    res = run_flow(loaded, {"settle_at": "2026-07-01"})
    blob = res.checkpoint.dumps()
    ckpt = RunCheckpoint.loads(blob)             # cross-process round-trip
    release = DeliverAnswerCommand(node_id="settle", value=None)   # WAIT release = Output(None)
    res2 = resume_flow(loaded, checkpoint=ckpt, commands=[release])
    assert res2.status == "succeeded"
    assert res2.output == "cancel"               # confirm_action(rec={}) -> "cancel"


def test_resume_via_live_engine_completes():
    loaded = load_flow(FLOW)
    res = run_flow(loaded, {"settle_at": "2026-07-01"})
    release = DeliverAnswerCommand(node_id="settle", value=None)   # WAIT release = Output(None)
    res2 = resume_flow(loaded, engine=res.engine, commands=[release])
    assert res2.status == "succeeded"
    assert res2.output == "cancel"


EFFECTS = """
id: e
name: e
typedefs:
  Approval: Literal[approve, reject]
input:
  settle_at: date
nodes:
  approve:
    kind: human_input
    prompt: "approve? (approve/reject)"
    output: Approval
  gate:
    kind: case
    on: ${approve.output}
    cases:
      - when: approve
        then: settle
    else: abort
  settle:
    kind: wait
    until: ${input.settle_at}
  confirm:
    kind: code
    depends_on: [settle]
    input:
      answer: ${approve.output}
    output: str
    code: tests.seeds.fns:confirm_action
  abort:
    kind: code
    input:
      answer: ${approve.output}
    output: str
    code: tests.seeds.fns:confirm_action
output: ${confirm.output | abort.output}
"""


_RESUME_FANOUT = """
id: rf
name: rf
input:
  settle_at: date
nodes:
  settle:
    kind: wait
    until: ${input.settle_at}
  pro:
    kind: code
    depends_on: [settle]
    input: {topic: "${input.settle_at}"}
    output: str
    code: tests.engine._compose_codefns:positive
  con:
    kind: code
    depends_on: [settle]
    input: {topic: "${input.settle_at}"}
    output: str
    code: tests.engine._compose_codefns:cautious
  merge:
    kind: code
    input: {pro: "${pro.output}", con: "${con.output}"}
    output: str
    code: tests.engine._compose_codefns:join_two
output: ${merge.output}
"""


def test_effects_approve_path():
    loaded = load_flow(EFFECTS)
    with install_ask_user(lambda **request: "approve"):
        paused = run_flow(loaded, {"settle_at": "2026-07-01"})
    assert paused.status == "paused"                   # only the timed WAIT parks
    done = resume_flow(loaded, engine=paused.engine,
                     commands=[DeliverAnswerCommand(node_id="settle", value=None)])
    assert done.status == "succeeded"
    assert done.output == "approve"                    # confirm_action(rec={answer:"approve"})


def test_effects_reject_path():
    loaded = load_flow(EFFECTS)
    with install_ask_user(lambda **request: "reject"):
        result = run_flow(loaded, {"settle_at": "2026-07-01"})
    assert result.status == "succeeded"                # abort branch, no wait
    assert result.output == "reject"                   # confirm_action(rec={answer:"reject"})


def test_effects_typed_answer_rejected():
    loaded = load_flow(EFFECTS)
    with install_ask_user(lambda **request: "maybe"):
        result = run_flow(loaded, {"settle_at": "2026-07-01"})
    assert result.status == "failed"  # not in Literal[approve,reject]


def test_wait_after_blocking_human_input_resumes_via_reason():
    # HUMAN_INPUT answers in the live run; the remaining WAIT reason maps to its
    # compatibility delivery command without a hardcoded coordinate.
    loaded = load_flow(EFFECTS)
    with install_ask_user(lambda **request: "approve"):
        paused = run_flow(loaded, {"settle_at": "2026-07-01"})
    assert paused.status == "paused"
    command = resume_command(loaded, paused.pause_reasons[0], True)
    done = resume_flow(loaded, engine=paused.engine, commands=[command])
    assert done.status == "succeeded"
    assert done.output == "approve"


def test_resume_command_rejects_non_resumable_reason():
    loaded = load_flow(EFFECTS)
    with pytest.raises(ValueError):
        resume_command(loaded, EventAwaited(), "x")  # external-watcher pause, not host-resumable


def test_full_host_round_trip_human_then_wait():
    loaded = load_flow(EFFECTS)
    with install_ask_user(lambda **request: "approve"):
        paused = run_flow(loaded, {"settle_at": "2026-07-01"})
    assert paused.status == "paused"                           # timed WAIT "settle"
    done = resume_flow(
        loaded,
        engine=paused.engine,
        commands=[resume_command(loaded, paused.pause_reasons[0], None)],
    )
    assert done.status == "succeeded"
    assert done.output == "approve"


def test_resume_command_does_not_require_node_in_compiled_flow():
    # the parked id may be a live namespaced id that exists only on engine.flow.nodes;
    # resume_command must dispatch on the reason SHAPE, never deref loaded.compiled.nodes.
    from types import SimpleNamespace

    from agent_composer.suspension.pause import HumanInputRequired

    loaded = SimpleNamespace(compiled=SimpleNamespace(nodes={}))   # EMPTY static graph
    reason = HumanInputRequired(prompt="?", node_id="call_sub/w")  # a live namespaced id
    cmd = resume_command(loaded, reason, "x")
    assert isinstance(cmd, DeliverAnswerCommand)
    assert cmd.node_id == "call_sub/w"                              # passed straight through


def test_checkpoint_persists_num_workers_roundtrip():
    """num_workers survives a dumps()->loads() round-trip; absent it would default to 0
    and a pooled resume would silently downgrade to serial."""
    from agent_composer.typesys.pool import VariablePool
    ckpt = RunCheckpoint(pool=VariablePool(), num_workers=4)
    loaded = RunCheckpoint.loads(ckpt.dumps())
    assert loaded.num_workers == 4


def test_checkpoint_num_workers_defaults_zero():
    from agent_composer.typesys.pool import VariablePool
    ckpt = RunCheckpoint(pool=VariablePool())
    assert ckpt.num_workers == 0


def test_checkpoint_rejects_pre_6_blob():
    import json
    from agent_composer.typesys.pool import VariablePool
    raw = json.loads(RunCheckpoint(pool=VariablePool()).dumps())
    raw["version"] = "5.0"                       # force an older version explicitly
    with pytest.raises(ValueError, match="incompatible checkpoint version"):
        RunCheckpoint.loads(json.dumps(raw))


def test_live_pooled_resume_matches_serial():
    """A run started pooled pauses at the wait; resume() then drives the post-wait fan-out
    under the POOLED driver and reaches the SAME terminal as a serial resume."""
    from agent_composer.run import run_flow, resume_flow
    from agent_composer.suspension.commands import DeliverAnswerCommand
    loaded = load_flow(_RESUME_FANOUT)

    serial = run_flow(loaded, {"settle_at": "2026-07-01"}, num_workers=0)
    assert serial.status == "paused"
    s2 = resume_flow(loaded, engine=serial.engine,
                     commands=[DeliverAnswerCommand(node_id="settle", value=None)])
    assert s2.status == "succeeded", s2.error

    pooled = run_flow(loaded, {"settle_at": "2026-07-01"}, num_workers=4)
    assert pooled.status == "paused"
    p2 = resume_flow(loaded, engine=pooled.engine,
                     commands=[DeliverAnswerCommand(node_id="settle", value=None)])
    assert p2.status == "succeeded", p2.error
    assert p2.output == s2.output


def test_resume_flow_num_workers_passthrough():
    """A durable (checkpoint=) resume accepts a num_workers override and drives the
    restored engine under that mode to terminal."""
    from agent_composer.run import run_flow, resume_flow
    from agent_composer.suspension.commands import DeliverAnswerCommand
    from agent_composer.suspension.checkpoint import RunCheckpoint
    loaded = load_flow(_RESUME_FANOUT)
    paused = run_flow(loaded, {"settle_at": "2026-07-01"})
    ckpt = RunCheckpoint.loads(paused.checkpoint.dumps())
    res = resume_flow(loaded, checkpoint=ckpt, num_workers=4,
                      commands=[DeliverAnswerCommand(node_id="settle", value=None)])
    assert res.status == "succeeded", res.error
