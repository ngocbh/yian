"""Connection settings come only from the host layer and travel with the host key."""

import dataclasses

import pytest
from langchain_core.messages import AIMessage
from pydantic import SecretStr

import yian.llm_clients as clients
from yian.compose import load_flow
from yian.compose.helpers.errors import LoadError
from yian.llm_clients.config import HostConnection, pop_host_credentials
from yian.memory.profile import ProfileMemory
from yian.memory.reflective import ReflectiveMemory
from yian.memory.storage import FileMarkdownStore
from yian.run import run_flow

SENTINEL = "sk-sentinel-5e0c27b4"
HEADER_VALUE = "gw-token-91c3d0aa"
AZURE_RESOURCE = "https://res.openai.azure.com/"

_FLOW = """id: routed
name: routed
nodes:
  agent: {kind: agent, prompt: hi}
output: ${agent.output}
"""
_SETTINGS = {
    "organization": "{organization: org-abc}",
    "project": "{project: proj-abc}",
    "headers": f"{{headers: {{X-Gateway: {HEADER_VALUE}}}}}",
}
_KEYED = {"provider": "openai", "model": "gpt-5.4", "api_key": SENTINEL}


@pytest.fixture(autouse=True)
def clean_env(tmp_path, monkeypatch):
    monkeypatch.setenv("YIAN_AUTH_FILE", str(tmp_path / "auth.json"))
    for name in (
        "YIAN_DEFAULT_PROVIDER", "YIAN_DEFAULT_MODEL", "OPENAI_API_KEY", "OPENAI_BASE_URL",
        "OPENAI_ORG_ID", "OPENAI_PROJECT_ID",
    ):
        monkeypatch.delenv(name, raising=False)


def _fake_models(monkeypatch):
    """Record each agent's model id instead of calling a provider."""
    seen = []

    class Model:
        def invoke(self, messages):
            return AIMessage(content="ok")

    def factory(config):
        seen.append(config["model"])
        return Model()

    monkeypatch.setattr(clients, "model_from_config", factory)
    return seen


def _flow_level(layer):
    return _FLOW.replace("nodes:", f"llm_config: {layer}\nnodes:")


def _node_level(layer):
    return _FLOW.replace("prompt: hi}", f"prompt: hi, llm_config: {layer}}}")


def _defs_level(layer):
    return f"""id: d
name: d
defs:
  body:
    llm_config: {layer}
    nodes:
      nested: {{kind: agent, prompt: hi}}
    output: ${{nested.output}}
nodes:
  child: {{kind: call, call: body}}
output: ${{child.output}}
"""


def _adaptive(layer):
    return f"""id: f
name: f
nodes:
  ask:
    kind: human_input
    adaptive_questions:
      prompt: "Ask about the project"
      llm_config: {layer}
output: ${{ask.output}}
"""


@pytest.mark.parametrize("field", sorted(_SETTINGS))
@pytest.mark.parametrize(
    ("site", "prefix"),
    [
        (_flow_level, "flow: llm_config."),
        (_defs_level, "defs entry 'body': llm_config."),
        (_node_level, "node 'agent': llm_config."),
        (_adaptive, "node 'ask' adaptive_questions.llm_config."),
    ],
    ids=["flow", "defs", "node", "adaptive"],
)
def test_a_flow_file_cannot_set_a_connection_setting(site, prefix, field):
    layer = _SETTINGS[field]
    # The setting is refused even beside a provider that takes it.
    with pytest.raises(LoadError) as caught:
        load_flow(site(layer.replace("{", "{provider: openai, ", 1)))
    message = str(caught.value)
    assert f"{prefix}{field} " in message
    assert "run's llm_config" in message
    assert "yian auth" in message
    assert HEADER_VALUE not in message


@pytest.mark.parametrize("field", ["organization", "project", "headers"])
def test_a_python_built_flow_layer_cannot_set_a_connection_setting(monkeypatch, field):
    # A flow built in Python skips the loader; the cascade still refuses the setting.
    seen = _fake_models(monkeypatch)
    flow = load_flow(_FLOW)
    value = {"X-Gateway": HEADER_VALUE} if field == "headers" else "org-abc"
    flow.compiled.nodes["agent"].own_llm_config = {field: value}
    with pytest.raises(ValueError, match=f"llm_config.{field} ") as caught:
        run_flow(flow, {}, llm_config=_KEYED)
    assert HEADER_VALUE not in str(caught.value)
    assert seen == []


def test_pop_host_credentials_takes_the_settings_with_the_key():
    layer = {
        **_KEYED,
        "organization": "org-abc",
        "project": "proj-abc",
        "headers": {"X-Gateway": HEADER_VALUE},
    }
    host = pop_host_credentials(layer)
    assert layer == {"provider": "openai", "model": "gpt-5.4"}
    assert host.api_key.get_secret_value() == SENTINEL
    assert (host.base_url, host.organization, host.project) == (None, "org-abc", "proj-abc")
    assert {name: value.get_secret_value() for name, value in host.headers.items()} == {
        "X-Gateway": HEADER_VALUE
    }
    assert all(isinstance(value, SecretStr) for value in host.headers.values())
    assert HEADER_VALUE not in repr(host)
    assert SENTINEL not in repr(host)
    with pytest.raises(dataclasses.FrozenInstanceError):
        host.organization = "org-other"


def test_pop_host_credentials_without_settings_returns_none_for_them():
    assert pop_host_credentials({"provider": "openai"}) == HostConnection()
    host = pop_host_credentials({"provider": "anthropic", "api_key": SENTINEL, "headers": {}})
    assert (host.organization, host.project, host.headers) == (None, None, None)


@pytest.mark.parametrize(
    "layer",
    [
        {"provider": "openai", "organization": "org-abc"},
        {"provider": "openai", "project": "proj-abc"},
        {"provider": "openai", "headers": {"X-Gateway": HEADER_VALUE}},
        {"organization": "org-abc"},
        {"provider": "azure", "base_url": AZURE_RESOURCE, "headers": {"X-Gateway": HEADER_VALUE}},
    ],
    ids=["organization", "project", "headers", "no-provider", "keyless-azure"],
)
def test_pop_host_credentials_refuses_settings_without_the_layers_key(layer):
    with pytest.raises(ValueError, match="api_key") as caught:
        pop_host_credentials(layer)
    assert HEADER_VALUE not in str(caught.value)


def test_pop_host_credentials_checks_the_settings_against_the_provider():
    with pytest.raises(ValueError, match="provider openai"):
        pop_host_credentials({"provider": "anthropic", "api_key": SENTINEL, "project": "p"})
    with pytest.raises(ValueError, match="Authorization") as caught:
        pop_host_credentials({**_KEYED, "headers": {"Authorization": HEADER_VALUE}})
    assert HEADER_VALUE not in str(caught.value)


def test_settings_without_a_host_key_fail_before_any_node_runs(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-environment-key")
    seen = _fake_models(monkeypatch)
    for setting in (
        {"organization": "org-abc"},
        {"project": "proj-abc"},
        {"headers": {"X-Gateway": HEADER_VALUE}},
    ):
        llm_config = {"provider": "openai", "model": "gpt-5.4", **setting}
        with pytest.raises(ValueError, match="api_key") as caught:
            run_flow(load_flow(_FLOW), {}, llm_config=llm_config)
        assert HEADER_VALUE not in str(caught.value)
    assert seen == []


def test_a_run_with_settings_beside_its_key_records_no_header_value(tmp_path, monkeypatch):
    seen = _fake_models(monkeypatch)
    path = tmp_path / "run.jsonl"
    llm_config = {**_KEYED, "organization": "org-abc", "headers": {"X-Gateway": HEADER_VALUE}}
    result = run_flow(load_flow(_FLOW), {}, log=path, llm_config=llm_config)
    assert result.status == "succeeded", result.failure
    assert seen == ["gpt-5.4"]
    assert HEADER_VALUE not in path.read_text()
    assert SENTINEL not in path.read_text()


@pytest.mark.parametrize(
    "setting",
    [{"organization": "org-abc"}, {"project": "proj-abc"}, {"headers": {"X-Gateway": "v"}}],
    ids=["organization", "project", "headers"],
)
def test_reflective_memory_refuses_connection_settings(setting):
    # Its own message comes first, not LLMConfig's "needs llm_config.api_key".
    with pytest.raises(ValueError) as caught:
        ReflectiveMemory({"provider": "openai", "model": "m", **setting})
    message = str(caught.value)
    assert f"llm_config.{next(iter(setting))} is not supported by reflective memory" in message
    assert "records its model configuration" in message
    assert "api_key" not in message


def test_profile_memory_keeps_the_settings_beside_its_key(tmp_path):
    store = FileMarkdownStore(tmp_path / "ada.md")
    memory = ProfileMemory(store, llm_config={
        **_KEYED,
        "organization": "org-abc",
        "project": "proj-abc",
        "headers": {"X-Gateway": HEADER_VALUE},
    })
    assert memory._llm_config == {"provider": "openai", "model": "gpt-5.4"}
    host = memory._host
    assert (host.organization, host.project) == ("org-abc", "proj-abc")
    assert host.headers["X-Gateway"].get_secret_value() == HEADER_VALUE
    assert host.api_key.get_secret_value() == SENTINEL


def test_profile_memory_needs_its_key_beside_the_settings(tmp_path):
    store = FileMarkdownStore(tmp_path / "ada.md")
    with pytest.raises(ValueError, match="api_key"):
        ProfileMemory(store, llm_config={"provider": "openai", "organization": "org-abc"})
