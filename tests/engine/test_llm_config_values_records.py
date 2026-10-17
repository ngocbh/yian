"""The config a call-time `llm_config` agent resolves: its node step, and the resume check."""

import json
import threading

from langchain_core.messages import AIMessage
import pytest

from yian.cli.run import _ResumeKeyError, _check_resume_credentials
import yian.run as run_module
from yian.compose import load_flow
from yian.run import run_flow

_HOST = {
    "provider": "openai",
    "api_key": "sk-host-secret",
    "headers": {"X-Tenant": "tenant-secret"},
}

_PICKED = """id: f
name: f
llm_config: {provider: openai}
nodes:
  pick: {kind: code, output: str, code: "return 'gpt-picked'"}
  chat:
    kind: agent
    prompt: Go.
    llm_config: {model: '${pick.output}'}
output: ${chat.output}
"""


class _HostCrash(BaseException):
    """Stands for the host process dying mid-call."""


@pytest.fixture(autouse=True)
def fake_model(monkeypatch):
    """Record each model built; a model built while ``crash`` is set dies on its call."""
    for name in ("OPENAI_API_KEY", "OPENAI_BASE_URL"):
        monkeypatch.delenv(name, raising=False)

    class Recorder(list):
        crash = False
        # Each model call's messages, and the replies queued for the next calls.
        calls: list = []
        replies: list = []

    built = Recorder()
    built.calls, built.replies = [], []

    class Model:
        def __init__(self, crash):
            self.crash = crash

        def invoke(self, messages):
            if self.crash:
                raise _HostCrash()
            built.calls.append(messages)
            return AIMessage(content=built.replies.pop(0) if built.replies else "ok")

    def model_from_config(config):
        built.append(config)
        return Model(built.crash)

    monkeypatch.setattr("yian.llm_clients.model_from_config", model_from_config)
    return built


def _records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def _steps(path, node_id="chat"):
    return [
        record["payload"]["step"]
        for record in _records(path)
        if record["kind"] == "node_step" and record["payload"]["node_id"] == node_id
    ]


def test_each_call_time_start_logs_its_config_without_secrets(tmp_path, fake_model):
    log = tmp_path / "run.jsonl"
    assert run_flow(load_flow(_PICKED), {}, log=log, llm_config=_HOST).status == "succeeded"
    [step] = _steps(log)
    config = step["effective_llm_config"]
    assert (config["provider"], config["model"]) == ("openai", "gpt-picked")
    assert config["_host_key"] is True and config["_header_names"] == ["X-Tenant"]
    assert "api_key" not in config and "headers" not in config
    data = log.read_bytes()
    assert b"sk-host-secret" not in data and b"tenant-secret" not in data
    [built] = fake_model
    assert built["model"] == "gpt-picked"


def test_an_agent_resolved_at_run_start_logs_no_step(tmp_path):
    text = """id: f
name: f
input: {m: !str }
nodes:
  chat:
    kind: agent
    prompt: Go.
    llm_config: {provider: openai, model: '${input.m}'}
output: ${chat.output}
"""
    log = tmp_path / "run.jsonl"
    assert run_flow(load_flow(text), {"m": "gpt-x"}, log=log).status == "succeeded"
    assert not [record for record in _records(log) if record["kind"] == "node_step"]


def _crashed_in_chat(tmp_path, fake_model):
    log = tmp_path / "run.jsonl"
    fake_model.crash = True
    with pytest.raises(_HostCrash):
        run_flow(load_flow(_PICKED), {}, log=log)
    fake_model.crash = False
    fake_model.clear()
    # The step was written before the model call that died.
    assert len(_steps(log)) == 1
    return log


def test_resume_recomputes_the_recorded_config(tmp_path, fake_model):
    log = _crashed_in_chat(tmp_path, fake_model)
    result = run_module.resume_flow(load_flow(_PICKED), log=log)
    assert result.status == "succeeded"
    first, second = _steps(log)
    assert first == second
    assert [config["model"] for config in fake_model] == ["gpt-picked"]


def test_resume_refuses_a_recorded_config_that_differs(tmp_path, fake_model):
    log = _crashed_in_chat(tmp_path, fake_model)
    lines = log.read_text().splitlines()
    for index, line in enumerate(lines):
        record = json.loads(line)
        if record["kind"] == "node_step":
            record["payload"]["step"]["effective_llm_config"]["model"] = "gpt-tampered"
            lines[index] = json.dumps(record)
    log.write_text("\n".join(lines) + "\n")
    before = len(_records(log))
    with pytest.raises(ValueError, match=(
        r"^node 'chat' llm_config: recorded config differs from the recomputed one"
    )):
        run_module.resume_flow(load_flow(_PICKED), log=log)
    kinds = [record["kind"] for record in _records(log)[before:]]
    assert "node_failed" not in kinds
    assert fake_model == []


_TWO_CALL_TIME = """id: f
name: f
llm_config: {provider: openai}
nodes:
  pick: {kind: code, output: str, code: "return 'anthropic'"}
  chat:
    kind: agent
    prompt: Go.
    llm_config: {provider: '${pick.output}', model: claude-picked}
  later:
    kind: agent
    prompt: Go.
    llm_config: {model: !str '${chat.output}'}
output: ${later.output}
"""


def _resume_checks(log, flow_text):
    """Resume ``log`` and return the map its key check received, appending nothing."""
    seen = []

    class _Stop(Exception):
        pass

    def check(effective):
        seen.append(effective)
        raise _Stop()

    with pytest.raises(_Stop):
        run_module._resume_flow_controlled(
            load_flow(flow_text), log=log, check_llm_configs=check,
            stop_event=threading.Event(),
        )
    [effective] = seen
    return effective


def test_resume_checks_the_provider_an_in_flight_call_time_agent_recorded(
    tmp_path, fake_model, monkeypatch,
):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-run")
    log = tmp_path / "run.jsonl"
    fake_model.crash = True
    with pytest.raises(_HostCrash):
        run_flow(load_flow(_TWO_CALL_TIME), {}, log=log)
    effective = _resume_checks(log, _TWO_CALL_TIME)
    assert effective["chat"]["provider"] == "anthropic"
    assert effective["chat"]["model"] == "claude-picked"
    # `later` has not started, so its provider is checked when it resolves.
    assert "later" not in effective
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    with pytest.raises(_ResumeKeyError, match="ANTHROPIC_API_KEY"):
        _check_resume_credentials(
            effective, ask_secret=lambda _question: None, prompt_available=False,
        )


_NO_FEEDBACK = json.dumps(
    {"input_feedback": {}, "self_feedback": ["Be brief."], "variable_feedback": {}}
)


def test_backward_of_a_call_time_agent_uses_its_bound_config(fake_model):
    flow = load_flow(_PICKED)
    assert "__llm_config" not in flow.compiled.nodes["chat"].backward_input_names()
    result = run_flow(flow, {}, learnable=True, llm_config=_HOST)
    assert result.status == "succeeded"
    fake_model.clear()
    fake_model.replies.append(_NO_FEEDBACK)
    assert result.backward("Shorter.").status == "succeeded"
    [config] = fake_model
    assert (config["provider"], config["model"]) == ("openai", "gpt-picked")
    # Marked for the host key, so backward sends it with its headers.
    assert config["api_key"].get_secret_value() == "sk-host-secret"
    assert config["headers"]["X-Tenant"].get_secret_value() == "tenant-secret"


@pytest.mark.parametrize("flow_llm", [
    "{provider: anthropic}",
    "{provider: azure, base_url: 'https://res.openai.azure.com/openai/v1'}",
], ids=["anthropic", "azure"])
def test_backward_of_a_call_time_agent_off_the_host_provider_gets_no_host_key(
    fake_model, monkeypatch, flow_llm,
):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "azure-env-key")
    text = _PICKED.replace("{provider: openai}", flow_llm)
    result = run_flow(load_flow(text), {}, learnable=True, llm_config=_HOST)
    assert result.status == "succeeded"
    fake_model.clear()
    fake_model.replies.append(_NO_FEEDBACK)
    assert result.backward("Shorter.").status == "succeeded"
    [config] = fake_model
    assert config["model"] == "gpt-picked"
    assert config["provider"] in flow_llm
    assert "api_key" not in config and "headers" not in config


def test_backward_of_a_static_agent_keeps_the_recorded_map(fake_model):
    text = _PICKED.replace("llm_config: {model: '${pick.output}'}", "llm_config: {model: m1}")
    result = run_flow(load_flow(text), {}, learnable=True)
    assert result.status == "succeeded"
    fake_model.clear()
    fake_model.replies.append(_NO_FEEDBACK)
    assert result.backward("Shorter.").status == "succeeded"
    [config] = fake_model
    assert (config["provider"], config["model"]) == ("openai", "m1")


def test_a_learnable_host_key_call_time_run_records_no_secret(tmp_path, monkeypatch):
    from yian.nodes.agent import AgentNode

    bound = []
    original = AgentNode.bind_reserved

    def spy(self, node_wiring, pool, *, bind=None):
        result = original(self, node_wiring, pool, bind=bind)
        bound.append(result)
        return result

    monkeypatch.setattr(AgentNode, "bind_reserved", spy)
    log = tmp_path / "run.jsonl"
    result = run_flow(load_flow(_PICKED), {}, log=log, learnable=True, llm_config=_HOST)
    assert result.status == "succeeded"
    [record] = [entry for entry in bound if entry]
    [succeeded] = [
        entry["payload"] for entry in _records(log)
        if entry["kind"] == "node_succeeded" and entry["payload"]["node_id"] == "chat"
    ]
    capture = json.dumps(succeeded["forward_capture"])
    evidence = json.dumps(succeeded["evidence"])
    assert "gpt-picked" in capture
    for text in (json.dumps(record), capture, evidence, log.read_text()):
        assert "sk-host-secret" not in text and "tenant-secret" not in text
