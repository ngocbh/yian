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


def test_llm_config_is_not_model_visible(tmp_path, fake_model, monkeypatch):
    import yian.runtime.eval_node as evaluator
    from yian.memory.session import SessionMemory

    recall_inputs = []
    real_recall = evaluator.RecallContext

    def recall_spy(base_system_prompt, rendered_prompt, inputs, static_path):
        recall_inputs.append(inputs)
        return real_recall(base_system_prompt, rendered_prompt, inputs, static_path)

    monkeypatch.setattr(evaluator, "RecallContext", recall_spy)
    text = _PICKED.replace("    prompt: Go.\n", "    prompt: Go.\n    long_term_memory: session\n")
    log = tmp_path / "run.jsonl"
    result = run_flow(
        load_flow(text), {}, log=log, learnable=True, memory={"session": SessionMemory()},
    )
    assert result.status == "succeeded"
    assert recall_inputs == [{}]
    [succeeded] = [
        entry["payload"] for entry in _records(log)
        if entry["kind"] == "node_succeeded" and entry["payload"]["node_id"] == "chat"
    ]
    assert succeeded["evidence"]["bound_inputs"] == {}
    # The bound record keeps it, so backward and resume read what ran.
    assert succeeded["forward_capture"]["bound_inputs"]["__llm_config"]["model"] == "gpt-picked"
    fake_model.calls.clear()
    fake_model.replies.append(_NO_FEEDBACK)
    assert result.backward("Shorter.").status == "succeeded"
    [messages] = fake_model.calls
    assert "__llm_config" not in str(messages[-1].content)
    assert "gpt-picked" not in str(messages[-1].content)


def test_backward_never_routes_feedback_to_the_model_value_producer(fake_model, monkeypatch):
    flow = load_flow(_PICKED)
    producer = type(flow.compiled.nodes["pick"])
    reached = []
    original = producer.backward

    def spy(self, context, feedback):
        reached.append(self.id)
        return original(self, context, feedback)

    monkeypatch.setattr(producer, "backward", spy)
    result = run_flow(flow, {}, learnable=True)
    assert result.status == "succeeded"
    fake_model.replies.append(json.dumps({
        "input_feedback": {"__llm_config": ["Use a bigger model."]},
        "self_feedback": [],
        "variable_feedback": {},
    }))
    outcome = result.backward("Shorter.")
    assert outcome.status == "failed"
    assert reached == []


def test_only_an_agent_with_the_key_hides_it_from_the_model():
    from yian.nodes.base import Node
    from yian.nodes.map.node import MapNode
    from yian.nodes.wait.node import WaitNode

    flow = load_flow(_PICKED)
    assert flow.compiled.nodes["chat"].model_hidden_inputs() == frozenset({"__llm_config"})
    static = load_flow(_PICKED.replace("{model: '${pick.output}'}", "{model: m1}"))
    assert static.compiled.nodes["chat"].model_hidden_inputs() == frozenset()
    # WAIT and MAP keep their reserved inputs; neither records evidence.
    for kind in (WaitNode, MapNode):
        assert kind.model_hidden_inputs is Node.model_hidden_inputs
        assert kind.needs_evaluation_trace is False


def _stopped_before_chat(log, text=_PICKED, **options):
    """Run ``text`` to ``log``, stopping the host as ``chat`` starts."""
    from yian.events import NodeStarted

    def stop(event):
        if isinstance(event, NodeStarted) and event.node_id == "chat":
            raise RuntimeError("host stopped before the agent")

    with pytest.raises(RuntimeError, match="host stopped"):
        run_flow(load_flow(text), {}, log=log, on_event=stop, **options)


def test_resume_without_the_host_key_raises_before_a_call_time_agent_calls(
    tmp_path, fake_model,
):
    from yian.llm_clients.auth import AuthError

    log = tmp_path / "run.jsonl"
    _stopped_before_chat(log, llm_config={"provider": "openai", "api_key": "sk-host-secret"})
    before = len(_records(log))
    with pytest.raises(AuthError):
        run_module.resume_flow(load_flow(_PICKED), log=log)
    failed = [
        record for record in _records(log)[before:]
        if record["kind"] == "node_failed" and record["payload"]["node_id"] == "chat"
    ]
    assert failed == []
    assert fake_model == []


def test_resume_refuses_a_recorded_config_that_no_longer_resolves(
    tmp_path, fake_model, monkeypatch,
):
    from yian.nodes.binding import BindingError

    log = _crashed_in_chat(tmp_path, fake_model)

    def drifted(*_args, **_kwargs):
        raise BindingError("node 'chat' llm_config.model: drifted", failure_code="contract.input")

    monkeypatch.setattr("yian.nodes.agent.node.call_time_config", drifted)
    before = len(_records(log))
    with pytest.raises(ValueError, match=(
        r"^node 'chat' llm_config.model: drifted; an earlier start of this node resolved it"
    )):
        run_module.resume_flow(load_flow(_PICKED), log=log)
    kinds = [record["kind"] for record in _records(log)[before:]]
    assert "node_failed" not in kinds
    assert fake_model == []


_DEFS_CALL_TIME = """id: f
name: f
llm_config: {provider: openai}
defs:
  ask:
    input: {m: !str }
    nodes:
      chat: {kind: agent, prompt: Go., llm_config: {model: '${input.m}'}}
    output: ${chat.output}
nodes:
  a: {kind: call, call: ask, input: {m: model-a}}
  b: {kind: call, call: ask, input: {m: '${a.output}'}}
output: ${b.output}
"""


def _crashed_in_second_call(tmp_path, fake_model):
    """Run ``_DEFS_CALL_TIME`` until the model call of ``b``'s agent dies; return the log."""
    log = tmp_path / "run.jsonl"
    original = fake_model.append

    def append(config):
        # The second agent's model dies on its call.
        fake_model.crash = config["model"] == "ok"
        original(config)

    fake_model.append = append
    with pytest.raises(_HostCrash):
        run_flow(load_flow(_DEFS_CALL_TIME), {}, log=log)
    del fake_model.append
    fake_model.crash = False
    fake_model.clear()
    return log


def _instance_steps(log):
    return [
        record["payload"] for record in _records(log) if record["kind"] == "node_step"
    ]


def test_resume_recomputes_the_config_of_each_defs_call_instance(tmp_path, fake_model):
    log = _crashed_in_second_call(tmp_path, fake_model)
    [first, second] = _instance_steps(log)
    assert first["node_id"] != second["node_id"]
    assert first["step"]["effective_llm_config"]["model"] == "model-a"
    assert second["step"]["effective_llm_config"]["model"] == "ok"
    result = run_module.resume_flow(load_flow(_DEFS_CALL_TIME), log=log)
    assert result.status == "succeeded"
    steps = _instance_steps(log)
    assert len(steps) == 3 and steps[2] == second
    assert [config["model"] for config in fake_model] == ["ok"]


def test_resume_refuses_a_defs_call_instance_whose_recorded_config_differs(
    tmp_path, fake_model,
):
    log = _crashed_in_second_call(tmp_path, fake_model)
    lines = log.read_text().splitlines()
    edited = 0
    for index, line in enumerate(lines):
        record = json.loads(line)
        step = record["payload"].get("step") if record["kind"] == "node_step" else None
        if step and step["effective_llm_config"]["model"] == "ok":
            step["effective_llm_config"]["model"] = "gpt-tampered"
            lines[index] = json.dumps(record)
            edited += 1
    assert edited == 1
    log.write_text("\n".join(lines) + "\n")
    before = len(_records(log))
    with pytest.raises(ValueError, match=(
        r"^node 'b/chat' llm_config: recorded config differs from the recomputed one"
    )):
        run_module.resume_flow(load_flow(_DEFS_CALL_TIME), log=log)
    assert "node_failed" not in [record["kind"] for record in _records(log)[before:]]
    assert fake_model == []
