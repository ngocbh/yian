"""Run start prepares agents whose `llm_config` resolves only when they run."""

import copy

import pytest
from pydantic import SecretStr

from yian.compile.llm_cascade import (
    apply_effective_llm_map,
    bind_host_credentials,
    resolve_run_llm,
)
from yian.compile.manifest import flow_fingerprint
from yian.compose import load_flow
from yian.llm_clients.config import HostConnection, pop_host_credentials

_DEFAULTS = {"provider": "openai", "model": "default-model"}

_ROOT = """id: f
name: f
llm_config: {provider: openai, openai: {temperature: 0.5}}
nodes:
  pick: {kind: agent, prompt: Pick., output: str, llm_config: {model: pick-model}}
  chat: {kind: agent, prompt: Go., llm_config: {model: '${pick.output}'}}
  solo:
    kind: agent
    prompt: Go.
    llm_config: {inherit: false, provider: anthropic, model: '${pick.output}'}
output: ${chat.output}
"""

_NESTED = """id: f
name: f
llm_config: {provider: anthropic}
defs:
  child:
    input: {m: !str }
    llm_config: {anthropic: {temperature: 0.3}}
    nodes:
      a: {kind: agent, prompt: Go., llm_config: {model: '${input.m}'}}
      b: {kind: agent, prompt: Go., llm_config: {model: b-model}}
    output: ${a.output}
nodes:
  c: {kind: call, call: child, input: {m: x}}
output: ${c.output}
"""


def _resolve(text, host=None):
    flow = load_flow(text).compiled
    private = copy.deepcopy(flow)
    run_llm = resolve_run_llm(private, dict(host or {}), defaults=_DEFAULTS)
    return flow, private, run_llm


def _node(flow, path):
    *callsites, node_id = path.split("/")
    for callsite in callsites:
        flow = flow.nodes[callsite].child
    return flow.nodes[node_id]


def test_a_root_call_time_agent_gets_the_host_layer_as_its_base():
    _, _, run_llm = _resolve(_ROOT, host={"model": "host-model"})
    assert set(run_llm.effective) == {"pick"}
    # The run's flow layer rides in the agent's own reserved binding, not in its base.
    assert run_llm.call_time == {"chat": {"model": "host-model"}, "solo": {}}


def test_a_nested_call_time_agent_gets_the_outer_layers_without_its_own_flow_layer():
    _, _, run_llm = _resolve(_NESTED, host={"model": "host-model"})
    assert set(run_llm.effective) == {"c/b"}
    assert run_llm.call_time == {"c/a": {"provider": "anthropic", "model": "host-model"}}


def test_apply_sets_each_call_time_agent_and_leaves_it_unresolved():
    _, private, run_llm = _resolve(_NESTED, host={"model": "host-model"})
    fresh = copy.deepcopy(load_flow(_NESTED).compiled)
    apply_effective_llm_map(fresh, run_llm.effective, run_llm.call_time)
    a, b = _node(fresh, "c/a"), _node(fresh, "c/b")
    assert a.call_time.path == "c/a"
    assert a.call_time.base == {"provider": "anthropic", "model": "host-model"}
    assert a.llm_config is None
    assert b.call_time is None
    assert b.llm_config == run_llm.effective["c/b"]
    # Each node holds its own copy.
    a.call_time.base["model"] = "changed"
    assert run_llm.call_time["c/a"]["model"] == "host-model"


def test_the_two_maps_must_cover_every_agent_once():
    flow, _, run_llm = _resolve(_ROOT)
    effective, call_time = run_llm.effective, run_llm.call_time
    without_chat = {k: v for k, v in call_time.items() if k != "chat"}
    with pytest.raises(ValueError, match=r"missing paths: \['chat'\]"):
        apply_effective_llm_map(copy.deepcopy(flow), effective, without_chat)
    with pytest.raises(ValueError, match=r"missing paths: \['chat', 'solo'\]"):
        apply_effective_llm_map(copy.deepcopy(flow), effective)
    with pytest.raises(ValueError, match=r"extra paths: \['ghost'\]"):
        apply_effective_llm_map(copy.deepcopy(flow), effective, {**call_time, "ghost": {}})
    overlap = {**effective, "chat": effective["pick"]}
    with pytest.raises(ValueError, match=r"in both maps: \['chat'\]"):
        apply_effective_llm_map(copy.deepcopy(flow), overlap, call_time)


def test_a_call_time_path_must_name_an_agent_with_llm_config_values():
    flow, _, run_llm = _resolve(_ROOT)
    effective = {k: v for k, v in run_llm.effective.items() if k != "pick"}
    with pytest.raises(ValueError, match="'pick'.*no llm_config value"):
        apply_effective_llm_map(
            copy.deepcopy(flow), effective, {**run_llm.call_time, "pick": {}}
        )


def test_a_call_time_base_must_be_a_valid_layer():
    flow, _, run_llm = _resolve(_ROOT)
    with pytest.raises(TypeError, match="'chat' must be a dict"):
        apply_effective_llm_map(
            copy.deepcopy(flow), run_llm.effective, {**run_llm.call_time, "chat": "x"}
        )
    with pytest.raises(ValueError, match="'chat'.*unknown field"):
        apply_effective_llm_map(
            copy.deepcopy(flow), run_llm.effective, {**run_llm.call_time, "chat": {"bogus": 1}}
        )
    with pytest.raises(ValueError, match="'chat'.*api_key"):
        apply_effective_llm_map(
            copy.deepcopy(flow), run_llm.effective,
            {**run_llm.call_time, "chat": {"provider": "openai", "api_key": "k"}},
        )


def test_the_host_key_reaches_call_time_agents_on_the_private_flow_only():
    host = {
        "provider": "openai",
        "model": "host-model",
        "api_key": "sk-secret-call-time",
        "headers": {"X-Tenant": "tenant-secret"},
    }
    connection = pop_host_credentials(host)
    flow, private, run_llm = _resolve(_NESTED, host=host)
    assert "secret" not in repr(run_llm.call_time)
    apply_effective_llm_map(private, run_llm.effective, run_llm.call_time)
    bind_host_credentials(private, connection.api_key, connection.headers)

    joined = _node(private, "c/a").host_credentials
    assert isinstance(joined, HostConnection)
    assert joined.api_key.get_secret_value() == "sk-secret-call-time"
    assert {k: v.get_secret_value() for k, v in joined.headers.items()} == {
        "X-Tenant": "tenant-secret"
    }
    # The key never enters the base, and a static agent gets no host credentials.
    assert "secret" not in repr(_node(private, "c/a").call_time.base)
    assert _node(private, "c/b").host_credentials is None
    assert _node(flow, "c/a").host_credentials is None
    assert _node(flow, "c/a").call_time is None


def test_bind_host_credentials_without_headers_gives_none():
    _, private, run_llm = _resolve(_ROOT)
    apply_effective_llm_map(private, run_llm.effective, run_llm.call_time)
    bind_host_credentials(private, SecretStr("sk-x"))
    assert private.nodes["chat"].host_credentials.headers is None


def test_the_fingerprint_ignores_the_call_time_attributes():
    flow, private, run_llm = _resolve(_ROOT)
    before = flow_fingerprint(flow)
    apply_effective_llm_map(private, run_llm.effective, run_llm.call_time)
    bind_host_credentials(private, SecretStr("sk-x"))
    assert private.nodes["chat"].call_time is not None
    assert flow_fingerprint(private) == before
