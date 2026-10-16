"""Connection settings come only from the host layer and travel with the host key."""

import dataclasses
import json

import httpx
import pytest
from langchain_core.messages import AIMessage
from pydantic import SecretStr

import yian.llm_clients as clients
from yian.compose import load_flow
from yian.compose.helpers.errors import LoadError
from yian.events import NodeStarted
from yian.llm_clients.config import HostConnection, pop_host_credentials
import yian.memory.profile as profile_module
from yian.memory.profile import ProfileMemory
from yian.memory.reflective import ReflectiveMemory
from yian.memory.storage import FileMarkdownStore
from yian.run import _check_recorded_llm_fields, resume_flow, run_flow
from tests.engine.test_run_api_key import (
    _NO_FEEDBACK,
    _assert_hidden,
    _chat_completions,
    _route_openai_client,
)

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


# --- the run records the settings without secrets and binds the values ------- #

OTHER_VALUE = "gw-trace-0b7d41e2"
_HOST_CONNECTION = {
    "method": "api_key", "source": "host", "credential_id": None,
    "endpoint": "https://api.openai.com/v1", "store_path": None,
}
_TWO_PROVIDERS = """id: two
name: two
defs:
  body:
    nodes:
      nested: {kind: agent, prompt: hi, llm_config: {model: nested-model}}
    output: ${nested.output}
nodes:
  agent: {kind: agent, prompt: hi}
  child: {kind: call, call: body, depends_on: [agent]}
  other:
    kind: agent
    prompt: hi
    depends_on: [agent]
    llm_config: {provider: anthropic, model: claude-sonnet-4-5}
output: ${other.output}
"""
_ALL_SETTINGS = {
    **_KEYED,
    "organization": "org-abc",
    "project": "proj-abc",
    "headers": {"X-Gateway": HEADER_VALUE, "X-Trace": OTHER_VALUE},
}


def _effective(path):
    return json.loads(path.read_text().splitlines()[0])["payload"]["effective_llm_by_static_path"]


def _recording_models(monkeypatch, target=clients):
    """Record each agent's connection settings by model id instead of calling a provider."""
    seen = {}

    class Model:
        def invoke(self, messages):
            return AIMessage(content="ok")

    def factory(config):
        headers = config.get("headers")
        assert headers is None or all(isinstance(v, SecretStr) for v in headers.values())
        seen[config["model"]] = (
            config.get("organization"),
            config.get("project"),
            None if headers is None else {k: v.get_secret_value() for k, v in headers.items()},
        )
        return Model()

    monkeypatch.setattr(target, "model_from_config", factory)
    return seen


def test_the_run_records_the_settings_on_the_key_agents_only(tmp_path, monkeypatch):
    seen = _recording_models(monkeypatch)
    path = tmp_path / "run.jsonl"
    result = run_flow(load_flow(_TWO_PROVIDERS), {}, log=path, llm_config=_ALL_SETTINGS)
    assert result.status == "succeeded", result.failure
    effective = _effective(path)
    recorded = {
        "provider": "openai", "organization": "org-abc", "project": "proj-abc",
        "_header_names": ["X-Gateway", "X-Trace"], "_host_key": True,
        "_connection": _HOST_CONNECTION,
    }
    assert effective["agent"] == {**recorded, "model": "gpt-5.4"}
    assert effective["child/nested"] == {**recorded, "model": "nested-model"}
    # Another provider's agent gets none of the key's settings.
    assert effective["other"] == {
        "provider": "anthropic", "model": "claude-sonnet-4-5",
        "_connection": {
            "method": "api_key", "source": "environment", "credential_id": None,
            "endpoint": None, "store_path": None,
        },
    }
    log = path.read_text()
    for secret in (HEADER_VALUE, OTHER_VALUE, SENTINEL):
        assert secret not in log
    # The in-memory node configs carry the values next to the key.
    headers = {"X-Gateway": HEADER_VALUE, "X-Trace": OTHER_VALUE}
    assert seen == {
        "gpt-5.4": ("org-abc", "proj-abc", headers),
        "nested-model": ("org-abc", "proj-abc", headers),
        "claude-sonnet-4-5": (None, None, None),
    }


def test_header_names_are_recorded_sorted_for_any_keyed_provider(tmp_path, monkeypatch):
    seen = _recording_models(monkeypatch)
    path = tmp_path / "run.jsonl"
    llm_config = {
        "provider": "anthropic", "model": "claude-sonnet-4-5", "api_key": SENTINEL,
        "headers": {"x-trace": OTHER_VALUE, "X-Gateway": HEADER_VALUE},
    }
    result = run_flow(load_flow(_FLOW), {}, log=path, llm_config=llm_config)
    assert result.status == "succeeded", result.failure
    assert _effective(path)["agent"] == {
        "provider": "anthropic", "model": "claude-sonnet-4-5",
        "_header_names": ["X-Gateway", "x-trace"], "_host_key": True,
    }
    assert seen["claude-sonnet-4-5"] == (
        None, None, {"x-trace": OTHER_VALUE, "X-Gateway": HEADER_VALUE},
    )
    assert HEADER_VALUE not in path.read_text()


def test_a_key_that_marks_no_agent_records_no_settings(tmp_path, monkeypatch):
    seen = _recording_models(monkeypatch)
    path = tmp_path / "run.jsonl"
    flow = load_flow(_FLOW.replace(
        "prompt: hi}", "prompt: hi, llm_config: {provider: anthropic, model: claude-sonnet-4-5}}"
    ))
    result = run_flow(flow, {}, log=path, llm_config=_ALL_SETTINGS)
    assert result.status == "succeeded", result.failure
    assert seen == {"claude-sonnet-4-5": (None, None, None)}
    agent = _effective(path)["agent"]
    for key in ("organization", "project", "_header_names", "_host_key"):
        assert key not in agent
    assert "org-abc" not in path.read_text()
    assert HEADER_VALUE not in path.read_text()


def test_a_finished_run_with_recorded_settings_replays_without_the_key(tmp_path, monkeypatch):
    _recording_models(monkeypatch)
    path = tmp_path / "run.jsonl"
    first = run_flow(load_flow(_FLOW), {}, log=path, llm_config=_ALL_SETTINGS)
    assert first.status == "succeeded", first.failure
    replayed = resume_flow(load_flow(_FLOW), log=path)
    assert replayed.status == "succeeded", replayed.failure
    assert replayed.output == first.output


def _stop_before_model(event):
    if isinstance(event, NodeStarted) and event.node_id == "agent":
        raise RuntimeError("host stopped before model")


def test_resume_with_the_key_keeps_the_recorded_organization(tmp_path, monkeypatch):
    seen = _recording_models(monkeypatch)
    path = tmp_path / "run.jsonl"
    llm_config = {**_KEYED, "organization": "org-abc", "project": "proj-abc"}
    with pytest.raises(RuntimeError, match="host stopped"):
        run_flow(
            load_flow(_FLOW), {}, log=path, on_event=_stop_before_model, llm_config=llm_config
        )
    result = resume_flow(load_flow(_FLOW), log=path, api_key=SENTINEL)
    assert result.status == "succeeded", result.failure
    assert seen == {"gpt-5.4": ("org-abc", "proj-abc", None)}


_HEADERS = {"X-Gateway": HEADER_VALUE, "X-Trace": OTHER_VALUE}


def _interrupted_header_run(tmp_path, llm_config=None):
    path = tmp_path / "run.jsonl"
    with pytest.raises(RuntimeError, match="host stopped"):
        run_flow(
            load_flow(_FLOW), {}, log=path, on_event=_stop_before_model,
            llm_config=_ALL_SETTINGS if llm_config is None else llm_config,
        )
    return path


def test_resume_sends_the_given_headers_to_the_key_agents(tmp_path, monkeypatch):
    seen = _recording_models(monkeypatch)
    path = _interrupted_header_run(tmp_path)
    # Names compare without case, as they do in HTTP.
    headers = {"x-gateway": HEADER_VALUE, "X-TRACE": SecretStr(OTHER_VALUE)}
    result = resume_flow(load_flow(_FLOW), log=path, api_key=SENTINEL, headers=headers)
    assert result.status == "succeeded", result.failure
    assert seen == {"gpt-5.4": ("org-abc", "proj-abc", {
        "x-gateway": HEADER_VALUE, "X-TRACE": OTHER_VALUE,
    })}
    assert HEADER_VALUE not in path.read_text()


@pytest.mark.parametrize(
    ("headers", "named"),
    [
        (None, "missing X-Gateway, X-Trace"),
        ({"X-Gateway": HEADER_VALUE}, "missing X-Trace"),
        ({**_HEADERS, "X-Extra": "extra-3f1c"}, "not recorded X-Extra"),
    ],
)
def test_resume_needs_exactly_the_recorded_header_names(tmp_path, monkeypatch, headers, named):
    seen = _recording_models(monkeypatch)
    path = _interrupted_header_run(tmp_path)
    before = path.read_bytes()
    with pytest.raises(ValueError) as caught:
        resume_flow(load_flow(_FLOW), log=path, api_key=SENTINEL, headers=headers)
    message = str(caught.value)
    assert message.endswith(named)
    for value in (HEADER_VALUE, OTHER_VALUE, "extra-3f1c", SENTINEL):
        assert value not in message
    assert path.read_bytes() == before
    assert seen == {}


def test_resume_reports_extra_header_names_beside_missing_ones(tmp_path, monkeypatch):
    _recording_models(monkeypatch)
    path = _interrupted_header_run(tmp_path)
    with pytest.raises(ValueError, match="missing X-Trace; not recorded X-Other"):
        resume_flow(
            load_flow(_FLOW), log=path, api_key=SENTINEL,
            headers={"X-Gateway": HEADER_VALUE, "X-Other": OTHER_VALUE},
        )


def test_resume_without_recorded_headers_refuses_given_ones(tmp_path, monkeypatch):
    _recording_models(monkeypatch)
    path = _interrupted_header_run(tmp_path, llm_config=_KEYED)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="not recorded X-Gateway"):
        resume_flow(
            load_flow(_FLOW), log=path, api_key=SENTINEL, headers={"X-Gateway": HEADER_VALUE},
        )
    assert path.read_bytes() == before


def test_resume_checks_the_given_header_values(tmp_path, monkeypatch):
    _recording_models(monkeypatch)
    path = _interrupted_header_run(tmp_path)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="header X-Trace value") as caught:
        resume_flow(
            load_flow(_FLOW), log=path, api_key=SENTINEL,
            headers={"X-Gateway": HEADER_VALUE, "X-Trace": "bad\nvalue-77"},
        )
    assert "value-77" not in str(caught.value)
    assert path.read_bytes() == before


def test_resume_ignores_headers_on_a_run_with_no_key_agents(tmp_path, monkeypatch):
    seen = _recording_models(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    path = _interrupted_header_run(tmp_path, llm_config={"provider": "openai", "model": "gpt-5.4"})
    result = resume_flow(
        load_flow(_FLOW), log=path, api_key=SENTINEL, headers={"X-Anything": HEADER_VALUE},
    )
    assert result.status == "succeeded", result.failure
    assert seen == {"gpt-5.4": (None, None, None)}


def test_a_finished_run_checks_the_given_header_names(tmp_path, monkeypatch):
    _recording_models(monkeypatch)
    path = tmp_path / "run.jsonl"
    first = run_flow(load_flow(_FLOW), {}, log=path, llm_config=_ALL_SETTINGS, learnable=True)
    assert first.status == "succeeded", first.failure
    before = path.read_bytes()
    with pytest.raises(ValueError, match="missing X-Trace"):
        resume_flow(
            load_flow(_FLOW), log=path, api_key=SENTINEL, headers={"X-Gateway": HEADER_VALUE},
        )
    with pytest.raises(ValueError, match="X-Gateway, X-Trace"):
        resume_flow(load_flow(_FLOW), log=path, api_key=SENTINEL)
    replayed = resume_flow(load_flow(_FLOW), log=path, api_key=SENTINEL, headers=_HEADERS)
    assert replayed.output == first.output
    assert path.read_bytes() == before


def test_recorded_header_names_pass_the_resume_field_check():
    marked = {
        "provider": "openai", "model": "gpt-5.4", "organization": "org-abc",
        "_header_names": ["X-Gateway"], "_host_key": True, "_connection": _HOST_CONNECTION,
    }
    _check_recorded_llm_fields({"agent": marked}, {})
    unmarked = {key: value for key, value in marked.items() if key != "_host_key"}
    with pytest.raises(ValueError, match="Agent 'agent'"):
        _check_recorded_llm_fields({"agent": unmarked}, {})


def test_profile_memory_settings_reach_its_model(tmp_path, monkeypatch):
    flow = load_flow(_FLOW.replace("nodes:", "long_term_memory: profile\nnodes:"))
    (tmp_path / "profile").mkdir()
    store = FileMarkdownStore(tmp_path / "profile" / "ada.md")
    store.write("# Ada\n", base_version=store.read()[1])
    memory = ProfileMemory(store, llm_config={**_ALL_SETTINGS, "model": "profiler"})
    built = []

    class Learner:
        def invoke(self, messages):
            return AIMessage(content="<profile>\n# Ada\n- Prefers brief replies.\n</profile>")

    def learner(config):
        built.append(config)
        return Learner()

    monkeypatch.setattr(profile_module, "model_from_config", learner)
    replies = iter(["ok", _NO_FEEDBACK])

    class Model:
        def invoke(self, messages):
            return AIMessage(content=next(replies))

    monkeypatch.setattr(clients, "model_from_config", lambda config: Model())
    result = run_flow(
        flow, {}, learnable=True, memory={"profile": memory},
        llm_config={"provider": "anthropic", "model": "claude-sonnet-4-5"},
    )
    outcome = result.backward("Be brief.")
    assert outcome.status == "succeeded", outcome.failure
    [config] = built
    assert (config["organization"], config["project"]) == ("org-abc", "proj-abc")
    assert config["api_key"].get_secret_value() == SENTINEL
    assert {name: value.get_secret_value() for name, value in config["headers"].items()} == {
        "X-Gateway": HEADER_VALUE, "X-Trace": OTHER_VALUE,
    }


# --- backward keeps and takes the host's headers ----------------------------- #

_VLLM_HEADERS = {
    "provider": "vllm", "model": "served-model", "api_key": SENTINEL, "headers": _HEADERS,
}


def _sent_headers(requests):
    return {
        (r.headers["Authorization"], r.headers["X-Gateway"], r.headers["X-Trace"])
        for r in requests
    }


def test_backward_sends_the_headers_the_result_kept(tmp_path, monkeypatch):
    requests, events = [], []
    path = tmp_path / "run.jsonl"
    with httpx.Client(transport=_chat_completions(requests, "ok", _NO_FEEDBACK)) as transport:
        _route_openai_client(monkeypatch, transport)
        result = run_flow(
            load_flow(_FLOW), {}, log=path, on_event=events.append, learnable=True,
            llm_config=_VLLM_HEADERS,
        )
        assert result.status == "succeeded", result.failure
        outcome = result.backward("Be brief.")
        assert outcome.status == "succeeded", outcome.failure
        recovered = resume_flow(
            load_flow(_FLOW), log=path, api_key=SENTINEL, headers=_HEADERS
        )
        resumed = recovered.backward("Be brief.")
        assert resumed.status == "succeeded", resumed.failure
    assert len(requests) == 3
    assert _sent_headers(requests) == {(f"Bearer {SENTINEL}", HEADER_VALUE, OTHER_VALUE)}
    for handle in (result._backward_handle, recovered._backward_handle):
        assert HEADER_VALUE not in repr(handle)
        assert OTHER_VALUE not in repr(handle)
    _assert_hidden(
        path, events, result, outcome, recovered, resumed,
        fragments=(SENTINEL, HEADER_VALUE, OTHER_VALUE),
    )


def test_backward_sends_the_headers_it_is_given_after_resume(tmp_path, monkeypatch):
    requests = []
    path = tmp_path / "run.jsonl"
    with httpx.Client(transport=_chat_completions(requests, "ok", _NO_FEEDBACK)) as transport:
        _route_openai_client(monkeypatch, transport)
        first = run_flow(load_flow(_FLOW), {}, log=path, learnable=True, llm_config=_VLLM_HEADERS)
        assert first.status == "succeeded", first.failure
        recovered = resume_flow(load_flow(_FLOW), log=path)
        # Names compare without case; values may be plain or secret.
        headers = {"x-gateway": SecretStr(HEADER_VALUE), "X-TRACE": OTHER_VALUE}
        outcome = recovered.backward("Be brief.", api_key=SENTINEL, headers=headers)
        assert outcome.status == "succeeded", outcome.failure
    assert len(requests) == 2
    assert _sent_headers(requests) == {(f"Bearer {SENTINEL}", HEADER_VALUE, OTHER_VALUE)}
    assert HEADER_VALUE not in repr(recovered._backward_handle)
    _assert_hidden(path, recovered, outcome, fragments=(SENTINEL, HEADER_VALUE, OTHER_VALUE))


def test_backward_without_the_headers_fails_naming_them(tmp_path, monkeypatch):
    requests = []
    path = tmp_path / "run.jsonl"
    with httpx.Client(transport=_chat_completions(requests, "ok", _NO_FEEDBACK)) as transport:
        _route_openai_client(monkeypatch, transport)
        run_flow(load_flow(_FLOW), {}, log=path, learnable=True, llm_config=_VLLM_HEADERS)
        recovered = resume_flow(load_flow(_FLOW), log=path)
        outcome = recovered.backward("Be brief.", api_key=SENTINEL)
    assert outcome.status == "failed"
    assert outcome.failure.cause_type == "AuthError"
    assert "connection headers X-Gateway, X-Trace" in outcome.failure.message
    assert "RunResult.backward" in outcome.failure.message
    assert len(requests) == 1
    _assert_hidden(path, recovered, outcome, fragments=(SENTINEL, HEADER_VALUE, OTHER_VALUE))


@pytest.mark.parametrize(
    ("headers", "error"),
    [
        ({"X-Gateway": HEADER_VALUE}, "missing X-Trace"),
        ({**_HEADERS, "X-Extra": "extra-3f1c"}, "not recorded X-Extra"),
        ({"X-Gateway": HEADER_VALUE, "X-Trace": "bad\nvalue-77"}, "header X-Trace value"),
    ],
)
def test_backward_checks_the_given_headers_before_appending(tmp_path, monkeypatch, headers, error):
    seen = _recording_models(monkeypatch)
    path = tmp_path / "run.jsonl"
    first = run_flow(load_flow(_FLOW), {}, log=path, learnable=True, llm_config=_ALL_SETTINGS)
    assert first.status == "succeeded", first.failure
    seen.clear()
    recovered = resume_flow(load_flow(_FLOW), log=path)
    before = path.read_bytes()
    with pytest.raises(ValueError, match=error) as caught:
        recovered.backward("Be brief.", api_key=SENTINEL, headers=headers)
    for value in (HEADER_VALUE, OTHER_VALUE, "extra-3f1c", "value-77", SENTINEL):
        assert value not in str(caught.value)
    assert path.read_bytes() == before
    assert seen == {}
