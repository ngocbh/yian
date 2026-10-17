"""`llm_config` values known only when their agent runs: run start, sessions, replay, the call."""

import json
import time

from langchain_core.messages import AIMessage
from pydantic import SecretStr
import pytest

import yian.run as run_module
from yian.compile.memory_layout import check_session_compatibility, resolve_memory_layout
from yian.compose import load_flow
from yian.events import NodeStarted
from yian.llm_clients import connection as connection_module
from yian.llm_clients.auth import ApiKeyCredentials, AuthStore, OAuthCredentials
from yian.memory.session import SessionMemory
from yian.memory.contracts import MemorySetupError
from yian.nodes.agent import AgentNode
from yian.nodes.agent.llm_values import LLM_CONFIG_KEY
from yian.run import _replay_run_log, run_flow
from yian.runtime.log import JsonlRunLog, canonical_seed


@pytest.fixture(autouse=True)
def fake_model(monkeypatch):
    """Record the config of every model built, and answer each call with ``ok``."""
    for name in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    built = []

    class Model:
        def invoke(self, messages):
            return AIMessage(content="ok")

    def model_from_config(config):
        built.append(config)
        return Model()

    monkeypatch.setattr("yian.llm_clients.model_from_config", model_from_config)
    return built


@pytest.fixture
def bound(monkeypatch):
    """Record what each AGENT's ``bind_reserved`` returns, by node id."""
    seen = []
    original = AgentNode.bind_reserved

    def spy(self, node_wiring, pool, *, bind=None):
        result = original(self, node_wiring, pool, bind=bind)
        seen.append((self.id, result))
        return result

    monkeypatch.setattr(AgentNode, "bind_reserved", spy)
    return seen


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


def test_a_session_placement_whose_provider_comes_from_a_value_passes_run_start():
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


# Resolving and checking the config before the model call.


def _picking(pick_output, pick_value, llm_config, *, typedefs="", flow_llm=""):
    """A flow whose ``chat`` agent takes ``llm_config`` and may read ``pick``'s output."""
    return f"""id: f
name: f
{typedefs}{flow_llm}nodes:
  pick: {{kind: code, output: {pick_output}, code: "return {pick_value}"}}
  chat:
    kind: agent
    prompt: Go.
    llm_config: {llm_config}
output: ${{chat.output}}
"""


def _wired(flow, node="chat"):
    return flow.compiled.wiring[node][LLM_CONFIG_KEY].source


def test_a_block_the_resolved_connection_refuses_fails_at_its_field(fake_model):
    AuthStore().save(
        "openai",
        OAuthCredentials("access-secret", "refresh-secret", time.time() + 3600, "account"),
        "gpt-selected",
    )
    flow = load_flow(_picking(
        "str", "'openai'",
        "{provider: '${pick.output}', openai: {max_output_tokens: 100}}",
    ))
    result = run_flow(flow, {})
    assert result.status == "failed"
    failure = result.failure
    assert failure.node_id == "chat"
    assert failure.code == "contract.input"
    assert failure.message.startswith("node 'chat' llm_config.openai: ")
    assert "`openai.max_output_tokens`" in failure.message
    assert "ChatGPT subscription" in failure.message
    origin = _wired(flow).origin_at(("node", "openai"))
    assert origin is not None and failure.trace[-1] == origin
    assert fake_model == []


def test_a_block_field_from_a_value_is_checked_at_call_time(fake_model):
    flow = load_flow(_picking(
        "Thinking", "{'type': 'enabled', 'budget_tokens': 10}",
        "{provider: anthropic, model: claude-x, anthropic: {thinking: '${pick.output}'}}",
        typedefs="typedefs:\n  Thinking: {type: str, budget_tokens: int}\n",
    ))
    result = run_flow(flow, {})
    assert result.status == "failed"
    failure = result.failure
    assert failure.code == "contract.input"
    assert failure.message.startswith(
        "node 'chat' llm_config.anthropic.thinking.budget_tokens: "
    ), failure.message
    assert "`anthropic.thinking.budget_tokens`" not in failure.message
    assert "1024" in failure.message
    assert failure.trace[-1] == _wired(flow).origin_at(("node", "anthropic", "thinking"))
    assert fake_model == []


def test_a_value_that_breaks_its_tag_fails_with_binding_type(fake_model):
    flow = load_flow(_picking(
        "Optional(str)", "None", "{provider: openai, model: !str '${pick.output}'}",
    ))
    result = run_flow(flow, {})
    assert result.status == "failed"
    failure = result.failure
    assert failure.code == "binding.type"
    assert failure.message.startswith("node 'chat' llm_config.model: "), failure.message
    assert failure.trace[-1] == _wired(flow).value["node"]["model"].origin  # the tag
    assert fake_model == []


def test_a_flow_value_that_breaks_its_tag_names_the_flow_field(fake_model):
    flow = load_flow(_picking(
        "Optional(str)", "None", "{provider: openai}",
        flow_llm="llm_config: {model: !str '${pick.output}'}\n",
    ))
    result = run_flow(flow, {})
    assert result.status == "failed"
    failure = result.failure
    assert failure.code == "binding.type"
    assert failure.message.startswith("flow llm_config.model: "), failure.message
    assert failure.trace[-1] == _wired(flow).value["flow"]["model"].origin  # the tag
    assert fake_model == []


def test_a_value_that_fails_to_evaluate_names_the_field(fake_model):
    flow = load_flow(_picking(
        "Optional(str)", "None",
        "{provider: openai, model: !str '${pick.output :? \"no model\"}'}",
    ))
    result = run_flow(flow, {})
    assert result.status == "failed"
    failure = result.failure
    assert failure.message.startswith("node 'chat' llm_config.model: "), failure.message
    assert "no model" in failure.message
    assert failure.trace[-1] == _wired(flow).origin_at(("node", "model"))
    assert fake_model == []


def test_a_non_finite_value_names_the_block_field(fake_model):
    flow = load_flow(_picking(
        "float", "1e308",
        "{provider: openai, model: m, openai: {temperature: !float '${pick.output * 10}'}}",
    ))
    result = run_flow(flow, {})
    assert result.status == "failed"
    failure = result.failure
    assert failure.code == "binding.evaluate"
    assert failure.message.startswith(
        "node 'chat' llm_config.openai.temperature: non-finite number inf"
    ), failure.message
    tagged = _wired(flow).value["node"]["openai"]["temperature"]
    assert failure.trace[-1] == tagged.origin
    assert fake_model == []


def test_a_block_value_out_of_bounds_names_the_field_once(fake_model):
    flow = load_flow(_picking(
        "float", "5.0", "{provider: openai, model: m, openai: {temperature: '${pick.output}'}}",
    ))
    result = run_flow(flow, {})
    assert result.status == "failed"
    failure = result.failure
    assert failure.code == "contract.input"
    assert failure.message == (
        "node 'chat' llm_config.openai.temperature: Input should be less than or equal to 2"
    )
    assert fake_model == []


def test_a_literal_typed_provider_resolving_to_an_unknown_member_names_the_field(fake_model):
    flow = load_flow(_picking(
        "Some", "'nope'", "{provider: '${pick.output}', model: m}",
        typedefs="typedefs:\n  Some: Literal(openai, nope)\n",
    ))
    result = run_flow(flow, {})
    assert result.status == "failed"
    failure = result.failure
    assert failure.code == "contract.input"
    assert failure.message.startswith("node 'chat' llm_config.provider: not a known provider")
    assert failure.trace[-1] == _wired(flow).origin_at(("node", "provider"))
    assert fake_model == []


def test_a_host_only_field_from_a_whole_record_value_is_refused_at_call_time(fake_model):
    record = (
        "{'provider': 'azure', 'model': 'dep', "
        "'base_url': 'https://res.openai.azure.com'}"
    )
    flow = load_flow(_picking("object", record, "'${pick.output}'"))
    result = run_flow(flow, {})
    assert result.status == "failed"
    failure = result.failure
    assert failure.code == "contract.input"
    assert failure.message.startswith("node 'chat' llm_config.base_url: ")
    assert "stay with the host or yian auth" in failure.message
    assert "res.openai.azure.com" not in failure.message
    assert fake_model == []


def _walk(value):
    """Yield every value nested in ``value``, ``value`` included."""
    yield value
    if isinstance(value, dict):
        for item in value.values():
            yield from _walk(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk(item)


_HOST = {
    "provider": "openai",
    "api_key": "sk-host-secret",
    "headers": {"X-Tenant": "tenant-secret"},
}


def test_the_bound_config_is_a_secret_free_map_entry(tmp_path, bound):
    static_log = tmp_path / "static.jsonl"
    static = _picking(
        "str", "'gpt-picked'", "{model: gpt-picked}", flow_llm="llm_config: {provider: openai}\n"
    )
    assert run_flow(load_flow(static), {}, log=static_log, llm_config=_HOST).status == "succeeded"
    entry = _records(static_log)[0]["payload"]["effective_llm_by_static_path"]["chat"]
    assert entry["_header_names"] == ["X-Tenant"]
    bound.clear()

    text = static.replace("{model: gpt-picked}", "{model: '${pick.output}'}")
    assert run_flow(load_flow(text), {}, llm_config=_HOST).status == "succeeded"
    [(node_id, reserved)] = bound
    assert node_id == "chat"
    assert reserved == {LLM_CONFIG_KEY: entry}
    for item in _walk(reserved):
        assert not isinstance(item, SecretStr)
        assert item not in ("sk-host-secret", "tenant-secret")
    assert "api_key" not in reserved[LLM_CONFIG_KEY]
    assert "headers" not in reserved[LLM_CONFIG_KEY]
    assert canonical_seed(reserved)[0] == reserved


def test_an_agent_resolved_at_run_start_binds_no_config(bound):
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
    flow = load_flow(text)
    assert LLM_CONFIG_KEY in flow.compiled.wiring["chat"]
    assert run_flow(flow, {"m": "gpt-x"}).status == "succeeded"
    assert bound == [("chat", {})]


# Calling the model with the resolved config.


def _seen(fake_model):
    """Return the one config the fake model was built from, as plain values."""
    [config] = fake_model
    plain = dict(config)
    if isinstance(plain.get("api_key"), SecretStr):
        plain["api_key"] = plain["api_key"].get_secret_value()
    if plain.get("headers"):
        plain["headers"] = {
            name: value.get_secret_value() if isinstance(value, SecretStr) else value
            for name, value in plain["headers"].items()
        }
    return plain


def test_the_model_comes_from_an_upstream_node(fake_model):
    flow = load_flow(_PICKED)
    assert run_flow(flow, {}).status == "succeeded"
    seen = _seen(fake_model)
    assert (seen["provider"], seen["model"]) == ("openai", "gpt-picked")
    assert seen["openai"] == {"temperature": 0.5}


def test_the_saved_model_of_a_provider_known_only_at_call_time_is_used(fake_model):
    AuthStore().save("anthropic", ApiKeyCredentials("saved-anthropic-key"), "claude-saved")
    flow = load_flow(_picking("str", "'anthropic'", "{provider: '${pick.output}'}"))
    assert run_flow(flow, {}).status == "succeeded"
    seen = _seen(fake_model)
    assert (seen["provider"], seen["model"]) == ("anthropic", "claude-saved")
    assert seen["_connection"]["source"] == "store"


def test_a_null_value_falls_through_to_the_flow_layer(fake_model):
    flow = load_flow(_picking(
        "Optional(str)", "None", "{model: '${pick.output}'}",
        flow_llm="llm_config: {provider: openai, model: flow-model}\n",
    ))
    assert run_flow(flow, {}).status == "succeeded"
    assert _seen(fake_model)["model"] == "flow-model"


def test_a_flow_level_whole_record_from_a_node_works_at_call_time(fake_model):
    text = """id: f
name: f
typedefs:
  Cfg: {provider: str, model: str}
llm_config: !Cfg ${pick.output}
nodes:
  pick: {kind: code, output: Cfg, code: "return {'provider': 'openai', 'model': 'whole'}"}
  chat: {kind: agent, prompt: Go.}
output: ${chat.output}
"""
    assert run_flow(load_flow(text), {}).status == "succeeded"
    seen = _seen(fake_model)
    assert (seen["provider"], seen["model"]) == ("openai", "whole")


def test_a_defs_field_from_a_child_input_works_at_call_time(fake_model):
    text = """id: f
name: f
defs:
  ask:
    input: {m: !str }
    llm_config: {model: '${input.m}'}
    nodes:
      chat: {kind: agent, prompt: Go., llm_config: {provider: openai}}
    output: ${chat.output}
nodes:
  a: {kind: call, call: ask, input: {m: def-model}}
output: ${a.output}
"""
    assert run_flow(load_flow(text), {}).status == "succeeded"
    seen = _seen(fake_model)
    assert (seen["provider"], seen["model"]) == ("openai", "def-model")


def test_the_host_key_authenticates_a_call_time_agent(tmp_path, fake_model):
    log = tmp_path / "run.jsonl"
    assert run_flow(load_flow(_PICKED), {}, log=log, llm_config=_HOST).status == "succeeded"
    seen = _seen(fake_model)
    assert seen["api_key"] == "sk-host-secret"
    assert seen["headers"] == {"X-Tenant": "tenant-secret"}
    assert seen["_host_key"] is True
    data = log.read_bytes()
    assert b"sk-host-secret" not in data and b"tenant-secret" not in data


def test_a_call_time_agent_of_another_provider_gets_no_host_key(fake_model):
    flow = load_flow(_picking("str", "'anthropic'", "{provider: '${pick.output}', model: c}"))
    assert run_flow(flow, {}, llm_config=_HOST).status == "succeeded"
    seen = _seen(fake_model)
    assert seen["provider"] == "anthropic"
    assert "api_key" not in seen and "headers" not in seen and "_host_key" not in seen


def test_an_azure_call_time_agent_gets_no_openai_host_key(fake_model):
    flow = load_flow(_picking(
        "str", "'deployment'", "{model: '${pick.output}'}",
        flow_llm="llm_config: {provider: azure, base_url: 'https://res.openai.azure.com'}\n",
    ))
    assert run_flow(flow, {}, llm_config=_HOST).status == "succeeded"
    seen = _seen(fake_model)
    assert (seen["provider"], seen["model"]) == ("azure", "deployment")
    assert "api_key" not in seen and "headers" not in seen and "_host_key" not in seen


def test_a_composer_call_time_error_names_the_gate(fake_model):
    text = """id: f
name: f
typedefs:
  Some: Literal(openai, nope)
nodes:
  pick: {kind: code, output: Some, code: "return 'nope'"}
  ask:
    kind: human_input
    adaptive_questions:
      prompt: "Ask about the project"
      llm_config: {provider: '${pick.output}', model: m}
output: ${ask.output}
"""
    result = run_flow(load_flow(text), {})
    assert result.status == "failed"
    assert result.failure.code == "contract.input"
    assert result.failure.message.startswith(
        "node 'ask' adaptive_questions.llm_config.provider: not a known provider"
    )
    assert fake_model == []


# A whole run, its replay, and the host key it keeps.


def test_a_finished_run_records_its_call_time_agent(tmp_path, fake_model):
    log = tmp_path / "run.jsonl"
    result = run_flow(load_flow(_PICKED), {}, log=log)
    assert result.status == "succeeded" and result.output == "ok"
    started = _records(log)[0]["payload"]
    assert started["effective_llm_by_static_path"] == {}
    assert started["call_time_llm"]["agents"] == {"chat": {}}


def test_a_finished_run_replays_without_ambient_settings(tmp_path, monkeypatch, fake_model):
    log = tmp_path / "run.jsonl"
    flow = load_flow(_PICKED)
    first = run_flow(flow, {}, log=log, llm_config=_HOST)
    assert first.status == "succeeded"
    fake_model.clear()

    def no_snapshot():
        pytest.fail("read the connection snapshot")

    monkeypatch.setattr(connection_module, "connection_snapshot", no_snapshot)
    monkeypatch.setattr(run_module, "connection_snapshot", no_snapshot)
    for name in ("YIAN_DEFAULT_PROVIDER", "YIAN_DEFAULT_MODEL"):
        monkeypatch.delenv(name, raising=False)
    replayed = run_module.resume_flow(flow, log=log)
    assert (replayed.status, replayed.output) == (first.status, first.output)
    assert fake_model == []


def _learnable_failed(llm_config, text=_PICK_FAILS, tmp_path=None):
    log = None if tmp_path is None else tmp_path / "run.jsonl"
    result = run_flow(load_flow(text), {}, log=log, learnable=True, llm_config=llm_config)
    assert result.status == "failed" and result.failure.node_id == "pick"
    return result


def test_the_result_keeps_the_host_key_a_call_time_agent_may_use(tmp_path):
    result = _learnable_failed(_HOST, tmp_path=tmp_path)
    handle = result._backward_handle
    assert handle.api_key.get_secret_value() == "sk-host-secret"
    assert handle.headers["X-Tenant"].get_secret_value() == "tenant-secret"
    recovered = run_module.resume_flow(
        load_flow(_PICK_FAILS), log=tmp_path / "run.jsonl",
        api_key="sk-host-secret", headers={"X-Tenant": "tenant-secret"},
    )
    assert recovered._backward_handle.api_key.get_secret_value() == "sk-host-secret"


def test_the_result_drops_a_host_key_no_call_time_agent_can_use(tmp_path):
    text = _PICK_FAILS.replace("{provider: openai}", "{provider: anthropic}")
    result = _learnable_failed(_HOST, text, tmp_path)
    assert result._backward_handle.api_key is None
    assert result._backward_handle.headers is None
    recovered = run_module.resume_flow(
        load_flow(text), log=tmp_path / "run.jsonl", api_key="sk-host-secret",
    )
    assert recovered._backward_handle.api_key is None


def test_a_call_time_agent_without_a_host_key_selection_gets_no_credentials(tmp_path):
    text = _PICK_FAILS.replace("{provider: openai}", "{provider: anthropic}")
    log = tmp_path / "run.jsonl"
    flow = load_flow(text)
    assert run_flow(flow, {}, log=log, llm_config=_HOST).status == "failed"
    authority = JsonlRunLog.resume(log)
    try:
        state = _replay_run_log(flow, authority, api_key=SecretStr("sk-host-secret"))
    finally:
        authority.close()
    assert state.flow.nodes["chat"].call_time.selection.host_key_provider is None
    assert state.flow.nodes["chat"].host_credentials is None


def test_resume_sends_the_host_key_to_an_in_flight_call_time_agent(tmp_path, fake_model):
    flow = load_flow(_PICKED)
    log = tmp_path / "run.jsonl"
    _interrupt_before_chat(flow, log, llm_config=_HOST)
    with pytest.raises(ValueError, match="header names this run recorded for its host key"):
        run_module.resume_flow(
            flow, log=log, api_key="sk-host-secret", headers={"X-Other": "v"}
        )
    assert fake_model == []
    resumed = run_module.resume_flow(
        flow, log=log, api_key="sk-host-secret", headers={"X-Tenant": "tenant-secret"}
    )
    assert resumed.status == "succeeded"
    seen = _seen(fake_model)
    assert seen["api_key"] == "sk-host-secret"
    assert seen["headers"] == {"X-Tenant": "tenant-secret"}


def test_backward_checks_given_headers_against_the_call_time_selection(tmp_path):
    result = _learnable_failed(_HOST, tmp_path=tmp_path)
    with pytest.raises(ValueError, match="missing X-Tenant; not recorded X-Other"):
        result.backward("Be brief.", headers={"X-Other": "v"})


def test_the_retained_backward_handle_opens_memory_with_the_call_time_agents(monkeypatch):
    replies = ["ok", json.dumps(
        {"input_feedback": {}, "self_feedback": ["Be brief."], "variable_feedback": {}}
    )]

    class Model:
        def invoke(self, messages):
            return AIMessage(content=replies.pop(0))

    monkeypatch.setattr("yian.llm_clients.model_from_config", lambda config: Model())
    applied = []
    original = run_module.apply_effective_llm_map

    def spy(flow, effective, call_time=None):
        applied.append(call_time)
        return original(flow, effective, call_time)

    text = _PICKED.replace("    prompt: Go.\n", "    prompt: Go.\n    long_term_memory: session\n")
    result = run_flow(load_flow(text), {}, learnable=True, memory={"session": SessionMemory()})
    assert result.status == "succeeded"
    monkeypatch.setattr(run_module, "apply_effective_llm_map", spy)
    outcome = result.backward("Shorter.")
    assert outcome.status == "succeeded" and outcome.learning is not None
    # Replay applies the record, then `open_memory` applies it to the retained program.
    assert len(applied) == 2
    assert [call_time["agents"] for call_time in applied] == [{"chat": {}}, {"chat": {}}]
