"""Durable resume must re-apply the CLI cascade BEFORE `restore` (the C1 fix).

A cross-process resume recompiles the flow and rebuilds the live graph inside
`FlowEngine.restore` (replay re-clones each CALL/MAP child). If the cascade is resolved
*after* restore, the replayed child agents already hold the unresolved config. So
`resume_flow` resolves the cascade on the recompiled flow before `restore`.
"""

import agent_composer.llm_clients as llm_clients_mod
import agent_composer.run as run_mod
from agent_composer.compose.loader import load_flow
from agent_composer.run import resume_command, resume_flow, run_flow
from agent_composer.suspension.checkpoint import RunCheckpoint

# A HUMAN_INPUT before an AGENT inside a CALL child: resume replays the CALL expansion,
# re-cloning the not-yet-run agent. Top flow sets `provider`; the CLI layer sets `model`.
_CALL_WRAPS_AGENT = """
id: cag
name: cag
llm_config: {provider: anthropic}
defs:
  approver:
    nodes:
      approval: {kind: human_input, prompt: "Continue?", output: str}
      agent:
        kind: agent
        mode: plain
        input: {approval: "${approval.output}"}
        prompt: "Approval was ${approval}"
        output: str
    output: ${agent.output}
nodes:
  gate:
    kind: call
    call: approver
output: ${gate.output}
"""

_HUMAN_ONLY = """
id: human-only
name: human_only
nodes:
  answer: {kind: human_input, prompt: "Continue?", output: str}
output: ${answer.output}
"""


def test_cli_cascade_survives_cross_process_resume_through_call(monkeypatch):
    from langchain_core.messages import AIMessage

    captured: list = []

    class _Chat:
        def invoke(self, messages):
            return AIMessage(content="FINAL")

    chat = _Chat()

    def fake_model_from_config(cfg):
        captured.append(dict(cfg))
        return chat

    monkeypatch.setattr(llm_clients_mod, "model_from_config", fake_model_from_config)

    loaded = load_flow(_CALL_WRAPS_AGENT)
    rec = run_flow(loaded, {}, llm_config={"model": "claude-opus-4-8"})
    assert rec.status == "paused"

    # Cross-process: recompile fresh, resume from the round-tripped checkpoint with the SAME
    # CLI layer. The inner agent's first model build after resume must carry provider+model.
    ckpt = RunCheckpoint.loads(rec.checkpoint.dumps())
    fresh = load_flow(_CALL_WRAPS_AGENT)
    captured.clear()
    res = resume_flow(
        fresh,
        checkpoint=ckpt,
        commands=[resume_command(loaded, rec.pause_reasons[0], "yes")],
        llm_config={"model": "claude-opus-4-8"},
    )
    assert res.status == "succeeded" and res.output == "FINAL"
    assert any(
        c.get("provider") == "anthropic" and c.get("model") == "claude-opus-4-8"
        for c in captured
    )


def test_defaults_are_read_once_per_fresh_or_durable_init_but_not_live_resume(
    monkeypatch,
):
    calls = {"provider": 0, "model": 0}

    def provider():
        calls["provider"] += 1
        return "anthropic"

    def model():
        calls["model"] += 1
        return "default-model"

    monkeypatch.setattr(run_mod, "default_llm_provider", provider)
    monkeypatch.setattr(run_mod, "default_llm_model", model)

    loaded = load_flow(_HUMAN_ONLY)
    paused = run_flow(loaded, {})
    assert paused.status == "paused"
    assert calls == {"provider": 1, "model": 1}

    live = resume_flow(
        loaded,
        engine=paused.engine,
        commands=[resume_command(loaded, paused.pause_reasons[0], "live")],
    )
    assert live.status == "succeeded"
    assert calls == {"provider": 1, "model": 1}

    durable_source = load_flow(_HUMAN_ONLY)
    durable_paused = run_flow(durable_source, {})
    checkpoint = RunCheckpoint.loads(durable_paused.checkpoint.dumps())
    assert calls == {"provider": 2, "model": 2}

    fresh = load_flow(_HUMAN_ONLY)
    durable = resume_flow(
        fresh,
        checkpoint=checkpoint,
        commands=[
            resume_command(
                durable_source, durable_paused.pause_reasons[0], "durable"
            )
        ],
    )
    assert durable.status == "succeeded"
    assert calls == {"provider": 3, "model": 3}
