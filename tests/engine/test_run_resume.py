import pytest

from agent_composer.compose.loader import load_flow
from agent_composer.run import resume_command, run_flow, resume_flow
from agent_composer.suspension.commands import DeliverAnswerCommand
from agent_composer.suspension.checkpoint import RunCheckpoint
from agent_composer.suspension.pause import EventAwaited
from agent_composer.tools import install_ask_user
from tests.engine._fakes import replace_human_inputs_with_artificial_pauses

FLOW = """
id: f
name: f
input:
  settle_at: date
nodes:
  settle:
    kind: human_input
    prompt: "continue?"
    output: Optional[str]
  done:
    kind: code
    depends_on: [settle]
    output: str
    code: tests.seeds.fns:confirm_action
output: ${done.output}
"""


def _load_artificial_pause_flow(source: str):
    """Load a flow and replace its human gate for checkpoint-only tests."""
    loaded = load_flow(source)
    replace_human_inputs_with_artificial_pauses(loaded.compiled)
    return loaded


def test_run_flow_pauses_carries_handles():
    loaded = _load_artificial_pause_flow(FLOW)
    res = run_flow(loaded, {"settle_at": "2026-07-01"})
    assert res.status == "paused"
    assert res.checkpoint is not None
    assert res.engine is not None
    assert len(res.pause_reasons) == 1


def test_resume_via_checkpoint_completes():
    loaded = _load_artificial_pause_flow(FLOW)
    res = run_flow(loaded, {"settle_at": "2026-07-01"})
    blob = res.checkpoint.dumps()
    ckpt = RunCheckpoint.loads(blob)             # cross-process round-trip
    release = DeliverAnswerCommand(node_id="settle", value=None)
    res2 = resume_flow(loaded, checkpoint=ckpt, commands=[release])
    assert res2.status == "succeeded"
    assert res2.output == "cancel"               # confirm_action(rec={}) -> "cancel"


def test_resume_via_live_engine_completes():
    loaded = _load_artificial_pause_flow(FLOW)
    res = run_flow(loaded, {"settle_at": "2026-07-01"})
    release = DeliverAnswerCommand(node_id="settle", value=None)
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
    kind: human_input
    prompt: "continue?"
    output: Optional[str]
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
        result = run_flow(loaded, {"settle_at": "2000-01-01"})
    assert result.status == "succeeded"
    assert result.output == "approve"


def test_effects_reject_path():
    loaded = load_flow(EFFECTS)
    with install_ask_user(lambda **request: "reject"):
        result = run_flow(loaded, {"settle_at": "2000-01-01"})
    assert result.status == "succeeded"                # abort branch, no wait
    assert result.output == "reject"                   # confirm_action(rec={answer:"reject"})


def test_effects_typed_answer_rejected():
    loaded = load_flow(EFFECTS)
    with install_ask_user(lambda **request: "maybe"):
        result = run_flow(loaded, {"settle_at": "2000-01-01"})
    assert result.status == "failed"  # not in Literal[approve,reject]


def test_wait_after_blocking_human_input_completes_in_same_run():
    # HUMAN_INPUT answers live, then a past absolute WAIT returns immediately.
    loaded = load_flow(EFFECTS)
    with install_ask_user(lambda **request: "approve"):
        result = run_flow(loaded, {"settle_at": "2000-01-01"})
    assert result.status == "succeeded"
    assert result.output == "approve"


def test_resume_command_rejects_non_resumable_reason():
    loaded = load_flow(EFFECTS)
    with pytest.raises(ValueError):
        resume_command(loaded, EventAwaited(), "x")  # external-watcher pause, not host-resumable


def test_blocking_human_then_wait_has_no_checkpoint_round_trip():
    loaded = load_flow(EFFECTS)
    with install_ask_user(lambda **request: "approve"):
        result = run_flow(loaded, {"settle_at": "2000-01-01"})
    assert result.status == "succeeded"
    assert result.checkpoint is None
    assert result.output == "approve"


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
    """A run started pooled pauses at an artificial gate; resume drives the fan-out
    under the POOLED driver and reaches the SAME terminal as a serial resume."""
    from agent_composer.run import run_flow, resume_flow
    from agent_composer.suspension.commands import DeliverAnswerCommand
    loaded = _load_artificial_pause_flow(_RESUME_FANOUT)

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
    loaded = _load_artificial_pause_flow(_RESUME_FANOUT)
    paused = run_flow(loaded, {"settle_at": "2026-07-01"})
    ckpt = RunCheckpoint.loads(paused.checkpoint.dumps())
    res = resume_flow(loaded, checkpoint=ckpt, num_workers=4,
                      commands=[DeliverAnswerCommand(node_id="settle", value=None)])
    assert res.status == "succeeded", res.error
