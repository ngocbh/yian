"""`llm_config` values known only when their agent runs: run start, the session check, replay."""

import json

from pydantic import SecretStr
import pytest

import yian.run as run_module
from yian.compile.memory_layout import check_session_compatibility, resolve_memory_layout
from yian.compose import load_flow
from yian.events import NodeStarted
from yian.llm_clients import connection as connection_module
from yian.memory.session import SessionMemory
from yian.memory.contracts import MemorySetupError
from yian.run import _replay_run_log, run_flow
from yian.runtime.log import JsonlRunLog


@pytest.fixture(autouse=True)
def fake_model(monkeypatch):
    """Record every model built; no test here may reach a call-time agent's model."""
    for name in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    built = []

    def model_from_config(config):
        built.append(config)
        raise AssertionError("no model is built in these tests")

    monkeypatch.setattr("yian.llm_clients.model_from_config", model_from_config)
    return built


def _records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


_PICK_FAILS = """id: f
name: f
llm_config: {provider: openai}
nodes:
  pick:
    kind: code
    output: str
    code: "return str(1 // 0)"  # fails on purpose
  chat:
    kind: agent
    prompt: Go.
    llm_config: {model: '${pick.output}'}
output: ${chat.output}
"""


def test_a_model_from_an_upstream_node_puts_the_agent_in_call_time_llm(tmp_path, fake_model):
    log = tmp_path / "run.jsonl"
    result = run_flow(load_flow(_PICK_FAILS), {}, log=log)
    assert result.status == "failed"
    started = _records(log)[0]["payload"]
    assert started["effective_llm_by_static_path"] == {}
    call_time = started["call_time_llm"]
    assert call_time["agents"] == {"chat": {}}
    # The flow's literal provider is the only one the agent can use.
    assert set(call_time["selection"]["connections"]) == {"openai"}
    assert call_time["selection"]["host"] is None
    assert fake_model == []


def test_a_run_without_call_time_agents_records_no_call_time_llm(tmp_path):
    log = tmp_path / "run.jsonl"
    text = _PICK_FAILS.replace("'${pick.output}'", "static-model")
    assert run_flow(load_flow(text), {}, log=log).status == "failed"
    assert "call_time_llm" not in _records(log)[0]["payload"]


def test_the_host_key_reaches_a_call_time_agent_of_its_provider(tmp_path, monkeypatch):
    bound = []
    original = run_module.bind_host_credentials

    def spy(flow, api_key, headers=None):
        original(flow, api_key, headers)
        bound.append(flow.nodes["chat"].host_credentials)

    monkeypatch.setattr(run_module, "bind_host_credentials", spy)
    log = tmp_path / "run.jsonl"
    host = {
        "provider": "openai",
        "api_key": "sk-host-secret",
        "headers": {"X-Tenant": "tenant-secret"},
    }
    assert run_flow(load_flow(_PICK_FAILS), {}, log=log, llm_config=host).status == "failed"
    # No static agent is marked for the key; the call-time agent may still use it.
    assert [credentials.api_key.get_secret_value() for credentials in bound] == [
        "sk-host-secret"
    ]
    selection = _records(log)[0]["payload"]["call_time_llm"]["selection"]
    assert selection["host"] == {"provider": "openai", "header_names": ["X-Tenant"]}
    data = log.read_bytes()
    assert b"sk-host-secret" not in data and b"tenant-secret" not in data


def test_a_host_key_no_call_time_agent_can_use_is_dropped(tmp_path, monkeypatch):
    bound = []
    monkeypatch.setattr(run_module, "bind_host_credentials", lambda *args: bound.append(args))
    log = tmp_path / "run.jsonl"
    text = _PICK_FAILS.replace("{provider: openai}", "{provider: anthropic}")
    host = {"provider": "openai", "api_key": "sk-host-secret"}
    assert run_flow(load_flow(text), {}, log=log, llm_config=host).status == "failed"
    assert bound == []
    assert _records(log)[0]["payload"]["call_time_llm"]["selection"]["host"] is None


_SESSION_FROM_VALUE = """id: f
name: f
defs:
  chat:
    input: {p: !str }
    nodes:
      ask:
        kind: agent
        long_term_memory: session
        prompt: hi
        output: str
        llm_config: {provider: '${input.p}', model: m}
    output: ${ask.output}
nodes:
  boom: {kind: code, output: str, code: "return str(1 // 0)"}
  a: {kind: call, call: chat, input: {p: openai}, depends_on: [boom]}
  b: {kind: call, call: chat, input: {p: anthropic}, depends_on: [boom]}
output: ${[a.output, b.output]}
"""

_SESSION_KNOWN = """id: f
name: f
defs:
  chat:
    input: {m: !str }
    nodes:
      ask:
        kind: agent
        long_term_memory: session
        prompt: hi
        output: str
        llm_config: {model: '${input.m}'}
    output: ${ask.output}
  via_one:
    llm_config: {provider: openai}
    nodes:
      inner: {kind: call, call: chat, input: {m: one}}
    output: ${inner.output}
  via_two:
    llm_config: {provider: PROVIDER}
    nodes:
      inner: {kind: call, call: chat, input: {m: two}}
    output: ${inner.output}
nodes:
  boom: {kind: code, output: str, code: "return str(1 // 0)"}
  a: {kind: call, call: via_one, depends_on: [boom]}
  b: {kind: call, call: via_two, depends_on: [boom]}
output: ${[a.output, b.output]}
"""


def test_a_session_placement_whose_provider_comes_from_a_value_passes_run_start(tmp_path):
    # The session's own check compares the providers when the agents run.
    result = run_flow(load_flow(_SESSION_FROM_VALUE), {}, memory={"session": SessionMemory()})
    assert result.status == "failed"
    assert result.failure.node_id == "boom"


def test_call_time_placements_with_conflicting_known_providers_fail_at_run_start(tmp_path):
    flow = load_flow(_SESSION_KNOWN.replace("PROVIDER", "anthropic"))
    log = tmp_path / "run.jsonl"
    with pytest.raises(MemorySetupError) as caught:
        run_flow(flow, {}, log=log, memory={"session": SessionMemory()})
    message = str(caught.value)
    assert "resolves provider 'openai' at 'a/inner/ask'" in message
    assert "but 'anthropic' at 'b/inner/ask'" in message
    assert not log.exists()


def test_call_time_placements_with_one_known_provider_pass_run_start():
    flow = load_flow(_SESSION_KNOWN.replace("PROVIDER", "openai"))
    result = run_flow(flow, {}, memory={"session": SessionMemory()})
    assert result.status == "failed"
    assert result.failure.node_id == "boom"


def test_the_session_check_skips_only_placements_it_is_told_are_unknown():
    flow = load_flow(_SESSION_KNOWN.replace("PROVIDER", "anthropic"))
    layout = resolve_memory_layout(flow.compiled)
    check_session_compatibility(
        layout, {}, call_time_known={"a/inner/ask": "openai", "b/inner/ask": None}
    )
    check_session_compatibility(
        layout, {"a/inner/ask": {"provider": "openai", "model": "m"}},
        call_time_known={"b/inner/ask": "openai"},
    )
    with pytest.raises(MemorySetupError, match="'anthropic' at 'b/inner/ask'"):
        check_session_compatibility(
            layout, {"a/inner/ask": {"provider": "openai", "model": "m"}},
            call_time_known={"b/inner/ask": "anthropic"},
        )
    with pytest.raises(KeyError):
        check_session_compatibility(layout, {}, call_time_known={"a/inner/ask": "openai"})


_PICKED = """id: f
name: f
llm_config: {provider: openai}
nodes:
  pick: {kind: code, output: str, code: "return 'gpt-picked'"}
  chat:
    kind: agent
    prompt: Go.
    llm_config: {model: '${pick.output}', openai: {temperature: 0.5}}
output: ${chat.output}
"""


def _interrupt_before_chat(flow, path, **options):
    def stop(event):
        if isinstance(event, NodeStarted) and event.node_id == "chat":
            raise RuntimeError("host stopped before the agent")

    with pytest.raises(RuntimeError, match="host stopped"):
        run_flow(flow, {}, log=path, on_event=stop, **options)


def test_replay_applies_the_recorded_call_time_agents_without_ambient_settings(
    tmp_path, monkeypatch, fake_model
):
    flow = load_flow(_PICKED)
    log = tmp_path / "run.jsonl"
    host = {"provider": "openai", "model": "host-model", "api_key": "sk-host-secret"}
    _interrupt_before_chat(flow, log, llm_config=host)
    recorded = _records(log)[0]["payload"]["call_time_llm"]

    def no_snapshot():
        pytest.fail("read the connection snapshot")

    monkeypatch.setattr(connection_module, "connection_snapshot", no_snapshot)
    monkeypatch.setattr(run_module, "connection_snapshot", no_snapshot)
    authority = JsonlRunLog.resume(log)
    try:
        state = _replay_run_log(flow, authority, api_key=SecretStr("sk-host-secret"))
    finally:
        authority.close()
    assert state.call_time_llm == recorded
    chat = state.flow.nodes["chat"]
    assert chat.llm_config is None
    assert chat.call_time.path == "chat"
    assert chat.call_time.base == {"provider": "openai", "model": "host-model"}
    assert chat.call_time.selection.host_key_provider == "openai"
    assert chat.call_time.selection.bind({"provider": "openai", "model": "m"})["_host_key"]
    assert chat.host_credentials.api_key.get_secret_value() == "sk-host-secret"
    assert flow.compiled.nodes["chat"].call_time is None
    assert fake_model == []
