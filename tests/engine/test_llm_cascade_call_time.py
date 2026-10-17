"""Run start prepares agents whose `llm_config` resolves only when they run."""

import copy
import json

import pytest
from pydantic import SecretStr

from yian.compile.llm_cascade import (
    apply_effective_llm_map,
    bind_host_credentials,
    call_time_selection,
    known_call_time_providers,
    resolve_run_llm,
)
from yian.compile.manifest import flow_fingerprint
from yian.compose import load_flow
from yian.llm_clients.config import HostConnection, pop_host_credentials
from yian.llm_clients.connection import (
    ConnectionSnapshot,
    record_selection,
    restore_selection,
)

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


# The recorded connection selection.

_STORE = "/tmp/yian-auth/auth.json"
_OPENAI_STORE = {
    "method": "api_key",
    "source": "store",
    "credential_id": "openai-login",
    "endpoint": "https://api.openai.com/v1",
    "store_path": _STORE,
    "organization": "org-saved",
}
_ANTHROPIC_STORE = {
    "method": "api_key",
    "source": "store",
    "credential_id": "anthropic-login",
    "endpoint": None,
    "store_path": _STORE,
}
_ENVIRONMENT = {
    "method": "api_key",
    "source": "environment",
    "credential_id": None,
    "endpoint": None,
    "store_path": None,
}


def _snapshot(**overrides):
    fields = {
        "defaults": {"provider": "openai", "model": "gpt-saved"},
        "provider_models": {"openai": "gpt-saved", "anthropic": "claude-saved"},
        "connections": {
            "openai": dict(_OPENAI_STORE),
            "anthropic": dict(_ANTHROPIC_STORE),
            "google": dict(_ENVIRONMENT),
        },
        "api_endpoint": "https://api.openai.com/v1",
        "stray_model": False,
    }
    fields.update(overrides)
    return ConnectionSnapshot(**fields)


_HOST = HostConnection(
    api_key=SecretStr("sk-secret-host"),
    base_url="https://proxy.example.com/v1",
    organization="org-host",
    project="proj-host",
    headers={"X-Tenant": SecretStr("tenant-secret"), "X-Alpha": SecretStr("alpha-secret")},
)


def _host_record(providers=None, snapshot=None):
    return record_selection(
        snapshot or _snapshot(),
        providers=providers,
        host_key_provider="openai",
        host_base_url=_HOST.base_url,
        host_settings=_HOST,
    )


def test_a_recorded_selection_round_trips_and_binds_as_the_snapshot_does():
    snapshot = _snapshot()
    record = json.loads(json.dumps(_host_record(snapshot=snapshot)))
    restored = restore_selection(record)
    assert restored.snapshot == snapshot
    for config in (
        {"provider": "openai", "model": "gpt-x"},
        {"provider": "anthropic", "model": "claude-x"},
        {"provider": "ollama", "model": "llama"},
    ):
        expected = snapshot.bind(
            {"a": config},
            host_key_provider="openai",
            host_base_url=_HOST.base_url,
            host_settings=_HOST,
        )["a"]
        assert restored.bind(config) == expected
    assert restored.bind({"provider": "openai", "model": "gpt-x"})["_header_names"] == [
        "X-Alpha", "X-Tenant",
    ]


def test_a_selection_without_a_host_key_binds_saved_connections():
    record = record_selection(_snapshot(), providers=None)
    assert record["host"] is None
    restored = restore_selection(json.loads(json.dumps(record)))
    assert restored.host_key_provider is None
    assert restored.bind({"provider": "openai", "model": "gpt-x"}) == {
        "provider": "openai", "model": "gpt-x", "_connection": _OPENAI_STORE,
    }


def test_no_secret_appears_in_a_recorded_selection():
    text = json.dumps(_host_record())
    for secret in ("sk-secret-host", "tenant-secret", "alpha-secret"):
        assert secret not in text


def _tampered(edit):
    record = copy.deepcopy(_host_record())
    edit(record)
    return record


@pytest.mark.parametrize(
    "edit",
    [
        lambda r: r["connections"]["openai"].update(source="bogus"),
        lambda r: r["connections"]["openai"].update(store_path="relative"),
        lambda r: r["connections"].update(ollama=dict(_ENVIRONMENT)),
        lambda r: r["connections"].update(nope=dict(_ENVIRONMENT)),
        lambda r: r.update(extra=1),
        lambda r: r.pop("stray_model"),
        lambda r: r.update(stray_model="no"),
        lambda r: r.update(api_endpoint="ftp://example.com"),
        lambda r: r["defaults"].update(provider="nope"),
        lambda r: r["defaults"].update(temperature=1),
        lambda r: r["provider_models"].update(openai=""),
        lambda r: r["host"].update(provider="nope"),
        lambda r: r["host"].update(base_url="http://example.com/v1"),
        lambda r: r["host"].update(header_names=["X-Tenant", "X-Alpha"]),
        lambda r: r["host"].update(organization=""),
        lambda r: r["host"].update(api_key="sk-x"),
    ],
)
def test_a_tampered_selection_fails_to_restore(edit):
    with pytest.raises(ValueError, match="recorded connection selection"):
        restore_selection(_tampered(edit))


def test_a_selection_limited_to_providers_names_only_their_connections():
    record = _host_record(providers={"anthropic"})
    assert record["connections"] == {"anthropic": _ANTHROPIC_STORE}
    assert record["provider_models"] == {"anthropic": "claude-saved"}
    # The default provider and the host key's provider are not reachable.
    assert record["defaults"] == {}
    assert record["host"] is None
    assert "openai-login" not in json.dumps(record)
    restored = restore_selection(record)
    assert restored.snapshot.connections == {"anthropic": _ANTHROPIC_STORE}


_LITERAL_PROVIDERS = """id: f
name: f
llm_config: {provider: google}
nodes:
  pick: {kind: agent, prompt: Pick., output: str, llm_config: {model: pick-model}}
  chat: {kind: agent, prompt: Go., llm_config: {provider: anthropic, model: '${pick.output}'}}
  flowed: {kind: agent, prompt: Go., llm_config: {model: '${pick.output}'}}
  solo:
    kind: agent
    prompt: Go.
    llm_config: {inherit: false, model: '${pick.output}'}
output: ${chat.output}
"""

_VALUE_PROVIDER = """id: f
name: f
llm_config: {provider: google}
nodes:
  pick: {kind: agent, prompt: Pick., output: str, llm_config: {model: pick-model}}
  chat: {kind: agent, prompt: Go., llm_config: {provider: '${pick.output}', model: m}}
output: ${chat.output}
"""


def _call_time_record(text):
    private = copy.deepcopy(load_flow(text).compiled)
    snapshot = _snapshot()
    run_llm = resolve_run_llm(private, {}, defaults=snapshot.defaults)
    return call_time_selection(
        private, run_llm.call_time, snapshot,
        host_key_provider="openai", host_base_url=_HOST.base_url, host_settings=_HOST,
    )


def test_known_providers_decide_which_connections_are_recorded():
    private = copy.deepcopy(load_flow(_LITERAL_PROVIDERS).compiled)
    run_llm = resolve_run_llm(private, {}, defaults=_snapshot().defaults)
    # Literal node, own-flow, and (under inherit: false) default providers.
    assert known_call_time_providers(private, run_llm.call_time, _snapshot().defaults) == {
        "chat": "anthropic", "flowed": "google", "solo": "openai",
    }
    record = _call_time_record(_LITERAL_PROVIDERS)
    assert set(record["connections"]) == {"anthropic", "google", "openai"}
    assert record["host"]["provider"] == "openai"


def test_only_literal_provider_agents_record_only_their_connections():
    text = _LITERAL_PROVIDERS.replace(
        "  solo:\n    kind: agent\n    prompt: Go.\n"
        "    llm_config: {inherit: false, model: '${pick.output}'}\n",
        "",
    )
    record = _call_time_record(text)
    assert set(record["connections"]) == {"anthropic", "google"}
    assert set(record["provider_models"]) == {"anthropic"}
    assert record["defaults"] == {}
    assert record["host"] is None


def test_a_provider_from_a_value_records_every_connection():
    private = copy.deepcopy(load_flow(_VALUE_PROVIDER).compiled)
    run_llm = resolve_run_llm(private, {}, defaults=_snapshot().defaults)
    assert known_call_time_providers(private, run_llm.call_time, {}) == {"chat": None}
    record = _call_time_record(_VALUE_PROVIDER)
    assert set(record["connections"]) == {"anthropic", "google", "openai"}
    assert record["defaults"] == _snapshot().defaults
    assert record["host"]["provider"] == "openai"


def test_a_nullable_provider_value_above_a_literal_is_not_known():
    text = """id: f
name: f
input: {p: !Optional(str) }
llm_config: {provider: anthropic}
defs:
  child:
    input: {p: !Optional(str) }
    nodes:
      a: {kind: agent, prompt: Go., llm_config: {provider: '${input.p}', model: m}}
    output: ${a.output}
nodes:
  c: {kind: call, call: child, input: {p: '${input.p}'}}
output: ${c.output}
"""
    private = copy.deepcopy(load_flow(text).compiled)
    run_llm = resolve_run_llm(private, {}, defaults={})
    assert known_call_time_providers(private, run_llm.call_time, {}) == {"c/a": None}
