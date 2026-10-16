"""A run-level `llm_config.base_url` goes only with the host key, and the log records it."""

import json

import httpx
import pytest
from langchain_core.messages import AIMessage

import yian.llm_clients as clients
import yian.llm_clients.factory as factory
from yian.compose import load_flow
from yian.compose.helpers.errors import LoadError
from yian.events import NodeStarted
from yian.memory.profile import ProfileMemory
from yian.memory.reflective import ReflectiveMemory
from yian.memory.storage import FileMarkdownStore
from yian.run import resume_flow, run_flow
from tests.engine.test_run_api_key import (
    _NO_FEEDBACK,
    _assert_hidden,
    _chat_completions,
    _route_openai_client,
)
from tests.fixtures.openai_responses import ResponsesServer

SENTINEL = "sk-sentinel-c40a9e17"
AZURE_V1 = "https://res.openai.azure.com/openai/v1/"
ELSEWHERE = "https://collector.example.com/v1"

_FLOW = """id: routed
name: routed
nodes:
  agent: {kind: agent, prompt: hi}
output: ${agent.output}
"""
_AZURE = {"provider": "openai", "model": "gpt-5.4", "base_url": AZURE_V1, "api_key": SENTINEL}
_VLLM = {
    "provider": "vllm", "model": "served-model",
    "base_url": "https://gateway.example.com/v1", "api_key": SENTINEL,
}


@pytest.fixture(autouse=True)
def clean_env(tmp_path, monkeypatch):
    monkeypatch.setenv("YIAN_AUTH_FILE", str(tmp_path / "auth.json"))
    for name in (
        "YIAN_DEFAULT_PROVIDER", "YIAN_DEFAULT_MODEL",
        "OPENAI_API_KEY", "OPENAI_BASE_URL", "VLLM_API_KEY", "VLLM_BASE_URL",
        "ANTHROPIC_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)


def _effective(path):
    return json.loads(path.read_text().splitlines()[0])["payload"]["effective_llm_by_static_path"]


def _fake_models(monkeypatch):
    """Record each agent's key and base_url by model id instead of calling a provider."""
    seen = {}

    class Model:
        def invoke(self, messages):
            return AIMessage(content="ok")

    def factory(config):
        key = config.get("api_key")
        seen[config["model"]] = (
            None if key is None else key.get_secret_value(), config.get("base_url")
        )
        return Model()

    monkeypatch.setattr(clients, "model_from_config", factory)
    return seen


def _fake_clients(monkeypatch, *replies):
    """Build no real client: record each one's provider, model, endpoint, and key.

    The models answer ``replies`` in call order, repeating the last. Model
    construction still runs through `model_from_config`, so every config check
    applies.
    """
    built, calls = [], []

    class Model:
        def invoke(self, messages):
            calls.append(messages)
            return AIMessage(content=replies[min(len(calls), len(replies)) - 1])

    def create(provider, model, **kwargs):
        key = kwargs.get("api_key")
        built.append((
            provider, model, kwargs.get("base_url"),
            key.get_secret_value() if hasattr(key, "get_secret_value") else key,
        ))
        return type("Client", (), {"get_llm": lambda self: Model()})()

    monkeypatch.setattr(factory, "create_llm_client", create)
    return built


def test_openai_run_sends_the_key_to_the_base_url_and_records_the_url(tmp_path, monkeypatch):
    server, events = ResponsesServer(tool_first=False), []
    path = tmp_path / "run.jsonl"
    with httpx.Client(transport=httpx.MockTransport(server)) as transport:
        _route_openai_client(monkeypatch, transport)
        result = run_flow(load_flow(_FLOW), {}, log=path, on_event=events.append, llm_config=_AZURE)
    assert result.status == "succeeded", result.failure
    request = server.requests[0]
    assert str(request.url) == "https://res.openai.azure.com/openai/v1/responses"
    assert request.headers["Authorization"] == f"Bearer {SENTINEL}"
    assert _effective(path) == {"agent": {
        "provider": "openai", "model": "gpt-5.4", "base_url": AZURE_V1, "_host_key": True,
        "_connection": {
            "method": "api_key", "source": "host", "credential_id": None,
            "endpoint": AZURE_V1, "store_path": None,
        },
    }}
    _assert_hidden(path, events, result)


def test_openai_compatible_run_records_the_base_url_beside_the_marker(tmp_path, monkeypatch):
    requests = []
    path = tmp_path / "run.jsonl"
    with httpx.Client(transport=_chat_completions(requests)) as transport:
        _route_openai_client(monkeypatch, transport)
        result = run_flow(load_flow(_FLOW), {}, log=path, llm_config=_VLLM)
    assert result.status == "succeeded", result.failure
    assert str(requests[0].url) == "https://gateway.example.com/v1/chat/completions"
    assert _effective(path) == {"agent": {
        "provider": "vllm", "model": "served-model",
        "base_url": "https://gateway.example.com/v1", "_host_key": True,
    }}


def test_base_url_goes_only_where_the_key_goes(tmp_path, monkeypatch):
    flow = load_flow("""id: two
name: two
defs:
  body:
    nodes:
      nested: {kind: agent, prompt: hi, llm_config: {model: nested-model}}
    output: ${nested.output}
nodes:
  inherits: {kind: agent, prompt: hi}
  child: {kind: call, call: body, depends_on: [inherits]}
  other:
    kind: agent
    prompt: hi
    depends_on: [inherits]
    llm_config: {provider: anthropic, model: claude-sonnet-4-5}
  optout:
    kind: agent
    prompt: hi
    depends_on: [other]
    llm_config: {inherit: false, provider: openai, model: other-deployment}
output: ${optout.output}
""")
    seen = _fake_models(monkeypatch)
    path = tmp_path / "run.jsonl"
    result = run_flow(flow, {}, log=path, llm_config=_AZURE)
    assert result.status == "succeeded", result.failure
    # A lower layer may pick another deployment of the key's provider; that agent
    # still sends the key only to the host's base_url.
    assert seen == {
        "gpt-5.4": (SENTINEL, AZURE_V1),
        "nested-model": (SENTINEL, AZURE_V1),
        "claude-sonnet-4-5": (None, None),
        "other-deployment": (SENTINEL, AZURE_V1),
    }
    assert "base_url" not in _effective(path)["other"]


def test_a_key_that_marks_no_agent_records_no_base_url(tmp_path, monkeypatch):
    seen = _fake_models(monkeypatch)
    path = tmp_path / "run.jsonl"
    flow = load_flow(_FLOW.replace(
        "prompt: hi}", "prompt: hi, llm_config: {provider: anthropic, model: claude-sonnet-4-5}}"
    ))
    result = run_flow(flow, {}, log=path, llm_config=_AZURE)
    assert result.status == "succeeded", result.failure
    assert seen == {"claude-sonnet-4-5": (None, None)}
    assert AZURE_V1 not in path.read_text()


def test_base_url_without_the_layers_key_is_rejected_before_any_request(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-environment-key")
    seen = _fake_models(monkeypatch)
    for llm_config in (
        {"provider": "openai", "model": "gpt-5.4", "base_url": AZURE_V1},
        {"model": "gpt-5.4", "base_url": AZURE_V1},
    ):
        with pytest.raises(ValueError, match="api_key"):
            run_flow(load_flow(_FLOW), {}, llm_config=llm_config)
    with pytest.raises(ValueError, match="https") as caught:
        run_flow(load_flow(_FLOW), {}, llm_config={**_AZURE, "base_url": "http://collector.example.com"})
    assert "collector" not in str(caught.value)
    assert SENTINEL not in str(caught.value)
    assert seen == {}


_DEF_LEVEL = f"""id: d
name: d
defs:
  body:
    llm_config: {{base_url: "{ELSEWHERE}"}}
    nodes:
      nested: {{kind: agent, prompt: hi}}
    output: ${{nested.output}}
nodes:
  child: {{kind: call, call: body}}
output: ${{child.output}}
"""


@pytest.mark.parametrize(
    "source",
    [
        _FLOW.replace("nodes:", f'llm_config: {{base_url: "{ELSEWHERE}"}}\nnodes:'),
        _FLOW.replace("prompt: hi}", f'prompt: hi, llm_config: {{base_url: "{ELSEWHERE}"}}}}'),
        _FLOW.replace(
            "prompt: hi}",
            f'prompt: hi, llm_config: {{provider: openai, base_url: "{ELSEWHERE}"}}}}',
        ),
        _DEF_LEVEL,
    ],
    ids=["flow", "node", "node-with-provider", "def"],
)
def test_a_flow_file_cannot_set_a_base_url(source):
    with pytest.raises(LoadError, match="base_url") as caught:
        load_flow(source)
    assert "run's llm_config" in str(caught.value)


_CHILD = _DEF_LEVEL.replace(f'    llm_config: {{base_url: "{ELSEWHERE}"}}\n', "")


@pytest.mark.parametrize("layer", ["flow", "node", "child-flow", "child-node", "marked-child-node"])
def test_a_lower_layer_base_url_cannot_redirect_the_host_key(monkeypatch, layer):
    # A flow built in Python skips the loader; the cascade still refuses the URL.
    seen = _fake_models(monkeypatch)
    flow = load_flow(_FLOW if layer in ("flow", "node") else _CHILD)
    child = None if layer in ("flow", "node") else flow.compiled.nodes["child"].child
    if layer == "flow":
        flow.compiled.flow_llm_config = {"base_url": ELSEWHERE}
    elif layer == "node":
        flow.compiled.nodes["agent"].own_llm_config = {"base_url": ELSEWHERE}
    elif layer == "child-flow":
        child.flow_llm_config = {"base_url": ELSEWHERE}
    elif layer == "child-node":
        child.nodes["nested"].own_llm_config = {"base_url": ELSEWHERE}
    else:
        child.nodes["nested"].own_llm_config = {"_host_key": True, "base_url": ELSEWHERE}
    with pytest.raises(ValueError, match="base_url"):
        run_flow(flow, {}, llm_config=_AZURE)
    assert seen == {}


# --- provider azure: the endpoint is llm_config.base_url ---------------------- #

AZURE_RESOURCE = "https://res.openai.azure.com/"
_AZURE_FLOW = f"""id: az
name: az
llm_config: {{provider: azure, base_url: "{AZURE_RESOURCE}"}}
nodes:
  agent: {{kind: agent, prompt: hi, llm_config: {{model: deploy-a}}}}
output: ${{agent.output}}
"""


@pytest.fixture
def azure_env(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "sk-azure-environment")
    monkeypatch.setenv("OPENAI_API_VERSION", "2024-02-01")
    # Neither variable is read: the deployment is the model, the endpoint the base_url.
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", ELSEWHERE)
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT_NAME", "env-deployment")


def test_a_flow_file_sets_the_azure_endpoint_and_the_run_records_it(tmp_path, monkeypatch, azure_env):
    built = _fake_clients(monkeypatch, "ok")
    path = tmp_path / "run.jsonl"
    result = run_flow(load_flow(_AZURE_FLOW), {}, log=path)
    assert result.status == "succeeded", result.failure
    assert built == [("azure", "deploy-a", AZURE_RESOURCE, None)]
    assert _effective(path) == {
        "agent": {
            "provider": "azure",
            "model": "deploy-a",
            "base_url": AZURE_RESOURCE,
            "_connection": {
                "method": "api_key",
                "source": "environment",
                "credential_id": None,
                "endpoint": None,
                "store_path": None,
            },
        },
    }


def test_azure_resume_uses_the_recorded_endpoint(tmp_path, monkeypatch, azure_env):
    built = _fake_clients(monkeypatch, "ok")
    path = tmp_path / "run.jsonl"
    with pytest.raises(RuntimeError, match="host stopped"):
        run_flow(load_flow(_AZURE_FLOW), {}, log=path, on_event=_stop_before_model)
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://moved.openai.azure.com/")
    result = resume_flow(load_flow(_AZURE_FLOW), log=path)
    assert result.status == "succeeded", result.failure
    assert built == [("azure", "deploy-a", AZURE_RESOURCE, None)]


@pytest.mark.parametrize(
    "source, llm_config",
    [
        (_FLOW.replace("prompt: hi}", "prompt: hi, llm_config: {provider: azure, model: d}}"), None),
        (_FLOW, {"provider": "azure", "model": "d"}),
    ],
    ids=["flow", "host"],
)
def test_an_azure_run_without_base_url_fails_before_any_request(monkeypatch, azure_env, source, llm_config):
    built = _fake_clients(monkeypatch, "ok")
    with pytest.raises(ValueError, match="Agent 'agent'.*set llm_config.base_url"):
        run_flow(load_flow(source), {}, llm_config=llm_config)
    assert built == []


def test_a_host_layer_can_give_the_azure_endpoint_without_a_key(tmp_path, monkeypatch, azure_env):
    built = _fake_clients(monkeypatch, "ok")
    path = tmp_path / "run.jsonl"
    result = run_flow(load_flow(_FLOW), {}, log=path, llm_config={
        "provider": "azure", "model": "d", "base_url": AZURE_RESOURCE,
    })
    assert result.status == "succeeded", result.failure
    assert built == [("azure", "d", AZURE_RESOURCE, None)]
    assert _effective(path)["agent"]["base_url"] == AZURE_RESOURCE


def test_a_host_azure_key_needs_its_own_base_url(monkeypatch, azure_env):
    # The flow names an endpoint, but a host key never goes to an authored one.
    built = _fake_clients(monkeypatch, "ok")
    with pytest.raises(ValueError, match="base_url") as caught:
        run_flow(load_flow(_AZURE_FLOW), {}, llm_config={"provider": "azure", "api_key": SENTINEL})
    assert SENTINEL not in str(caught.value)
    assert built == []


def test_a_host_azure_key_goes_to_the_host_base_url(tmp_path, monkeypatch, azure_env):
    built = _fake_clients(monkeypatch, "ok")
    path = tmp_path / "run.jsonl"
    host_url = "https://host.openai.azure.com/"
    result = run_flow(load_flow(_AZURE_FLOW), {}, log=path, llm_config={
        "provider": "azure", "api_key": SENTINEL, "base_url": host_url,
    })
    assert result.status == "succeeded", result.failure
    assert built == [("azure", "deploy-a", host_url, SENTINEL)]
    assert _effective(path) == {"agent": {
        "provider": "azure", "model": "deploy-a", "base_url": host_url, "_host_key": True,
    }}


def test_an_inherited_azure_endpoint_does_not_reach_another_provider(monkeypatch, azure_env):
    built = _fake_clients(monkeypatch, "ok")
    flow = load_flow(_AZURE_FLOW.replace("{model: deploy-a}", "{provider: openai, model: m}"))
    with pytest.raises(ValueError, match="base_url"):
        run_flow(flow, {})
    assert built == []


@pytest.mark.parametrize("provider", ["openai", "vllm"])
def test_an_inherited_azure_endpoint_never_carries_another_providers_host_key(
    monkeypatch, azure_env, provider
):
    built = _fake_clients(monkeypatch, "ok")
    flow = load_flow(_AZURE_FLOW.replace("{model: deploy-a}", f"{{provider: {provider}, model: m}}"))
    with pytest.raises(ValueError, match="base_url") as caught:
        run_flow(flow, {}, llm_config={"provider": provider, "api_key": SENTINEL})
    assert SENTINEL not in str(caught.value)
    assert built == []


def test_a_def_level_azure_endpoint_reaches_its_body(tmp_path, monkeypatch, azure_env):
    built = _fake_clients(monkeypatch, "ok")
    path = tmp_path / "run.jsonl"
    flow = load_flow(_CHILD.replace(
        "  body:\n", f'  body:\n    llm_config: {{provider: azure, model: d, base_url: "{AZURE_RESOURCE}"}}\n'
    ))
    result = run_flow(flow, {}, log=path)
    assert result.status == "succeeded", result.failure
    assert built == [("azure", "d", AZURE_RESOURCE, None)]
    assert _effective(path)["child/nested"]["base_url"] == AZURE_RESOURCE


def test_memories_take_a_keyless_azure_endpoint(tmp_path):
    config = {"provider": "azure", "model": "d", "base_url": AZURE_RESOURCE}
    assert ReflectiveMemory(dict(config))._llm_config == config
    profile = ProfileMemory(FileMarkdownStore(tmp_path / "ada.md"), llm_config=dict(config))
    assert profile._llm_config == config
    assert (profile._host.api_key, profile._host.base_url) == (None, None)


# --- resume and backward reuse the recorded base_url ------------------------- #


def _stop_before_model(event):
    if isinstance(event, NodeStarted) and event.node_id == "agent":
        raise RuntimeError("host stopped before model")


def _interrupted_run(path, llm_config):
    with pytest.raises(RuntimeError, match="host stopped"):
        run_flow(load_flow(_FLOW), {}, log=path, on_event=_stop_before_model, llm_config=llm_config)


def _rewrite_effective(path, change):
    lines = path.read_text().splitlines()
    first = json.loads(lines[0])
    change(first["payload"]["effective_llm_by_static_path"]["agent"])
    path.write_text("\n".join([json.dumps(first), *lines[1:]]) + "\n")


def test_resume_sends_the_key_to_the_recorded_base_url(tmp_path, monkeypatch):
    requests, events = [], []
    path = tmp_path / "run.jsonl"
    _interrupted_run(path, _VLLM)
    # The provider's endpoint variable cannot move a recorded agent.
    monkeypatch.setenv("VLLM_BASE_URL", "http://127.0.0.1:9/v1")
    with httpx.Client(transport=_chat_completions(requests)) as transport:
        _route_openai_client(monkeypatch, transport)
        result = resume_flow(load_flow(_FLOW), log=path, on_event=events.append, api_key=SENTINEL)
    assert result.status == "succeeded", result.failure
    assert str(requests[0].url) == "https://gateway.example.com/v1/chat/completions"
    assert requests[0].headers["Authorization"] == f"Bearer {SENTINEL}"
    _assert_hidden(path, events, result)


def test_openai_resume_sends_the_key_to_the_recorded_azure_url(tmp_path, monkeypatch):
    server = ResponsesServer(tool_first=False)
    path = tmp_path / "run.jsonl"
    _interrupted_run(path, _AZURE)
    keyless = tmp_path / "keyless.jsonl"
    keyless.write_bytes(path.read_bytes())
    monkeypatch.setenv("OPENAI_BASE_URL", ELSEWHERE)
    with httpx.Client(transport=httpx.MockTransport(server)) as transport:
        _route_openai_client(monkeypatch, transport)
        assert resume_flow(load_flow(_FLOW), log=keyless).status == "failed"
        assert server.requests == []
        result = resume_flow(load_flow(_FLOW), log=path, api_key=SENTINEL)
    assert result.status == "succeeded", result.failure
    assert str(server.requests[0].url) == "https://res.openai.azure.com/openai/v1/responses"
    assert server.requests[0].headers["Authorization"] == f"Bearer {SENTINEL}"


def test_backward_sends_the_key_to_the_recorded_base_url(tmp_path, monkeypatch):
    requests, events = [], []
    path = tmp_path / "run.jsonl"
    with httpx.Client(transport=_chat_completions(requests, "ok", _NO_FEEDBACK)) as transport:
        _route_openai_client(monkeypatch, transport)
        result = run_flow(
            load_flow(_FLOW), {}, log=path, on_event=events.append, learnable=True, llm_config=_VLLM
        )
        assert result.status == "succeeded", result.failure
        assert result.backward("Be brief.").status == "succeeded"
        recovered = resume_flow(load_flow(_FLOW), log=path, on_event=events.append)
        outcome = recovered.backward("Be brief.", api_key=SENTINEL)
        assert outcome.status == "succeeded", outcome.failure
    assert len(requests) == 3
    assert {str(r.url) for r in requests} == {"https://gateway.example.com/v1/chat/completions"}
    assert {r.headers["Authorization"] for r in requests} == {f"Bearer {SENTINEL}"}
    _assert_hidden(path, events, result, recovered, outcome)


def test_openai_backward_builds_its_client_with_the_recorded_url_and_key(tmp_path, monkeypatch):
    # The path a host on an Azure OpenAI v1 endpoint uses: the recorded config
    # goes through legacy_connection_config and the host connection's endpoint.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-environment-key")
    monkeypatch.setenv("OPENAI_BASE_URL", ELSEWHERE)
    built = _fake_clients(monkeypatch, "ok", _NO_FEEDBACK)
    path = tmp_path / "run.jsonl"
    result = run_flow(load_flow(_FLOW), {}, log=path, learnable=True, llm_config=_AZURE)
    assert result.status == "succeeded", result.failure
    assert result.backward("Be brief.").status == "succeeded"
    recovered = resume_flow(load_flow(_FLOW), log=path)
    outcome = recovered.backward("Be brief.", api_key=SENTINEL)
    assert outcome.status == "succeeded", outcome.failure
    assert built == [("openai", "gpt-5.4", AZURE_V1, SENTINEL)] * 3


def test_a_recorded_base_url_without_the_marker_is_refused(tmp_path, monkeypatch):
    seen = _fake_models(monkeypatch)
    path = tmp_path / "run.jsonl"
    _interrupted_run(path, _VLLM)
    _rewrite_effective(path, lambda config: config.pop("_host_key"))
    with pytest.raises(ValueError, match="base_url without a host key"):
        resume_flow(load_flow(_FLOW), log=path, api_key=SENTINEL)
    assert seen == {}


@pytest.mark.parametrize(
    "recorded", ["http://collector.example.com/v1", " https://gateway.example.com/v1", 7]
)
def test_a_tampered_recorded_base_url_fails_at_resume_start(tmp_path, monkeypatch, recorded):
    seen = _fake_models(monkeypatch)
    path = tmp_path / "run.jsonl"
    _interrupted_run(path, _VLLM)
    _rewrite_effective(path, lambda config: config.update(base_url=recorded))
    with pytest.raises(ValueError, match="effective LLM config for 'agent': base_url") as caught:
        resume_flow(load_flow(_FLOW), log=path, api_key=SENTINEL)
    assert str(recorded).strip() not in str(caught.value)
    assert SENTINEL not in str(caught.value)
    assert seen == {}


def test_a_recorded_base_url_must_match_the_host_connection(tmp_path, monkeypatch):
    server = ResponsesServer(tool_first=False)
    path = tmp_path / "run.jsonl"
    _interrupted_run(path, _AZURE)
    _rewrite_effective(path, lambda config: config.update(base_url=ELSEWHERE))
    with httpx.Client(transport=httpx.MockTransport(server)) as transport:
        _route_openai_client(monkeypatch, transport)
        result = resume_flow(load_flow(_FLOW), log=path, api_key=SENTINEL)
    assert result.status == "failed"
    assert "host connection endpoint" in result.failure.message
    assert server.requests == []


# --- memory learners ---------------------------------------------------------- #


def test_reflective_memory_refuses_a_base_url_without_its_key():
    # Reflective memory refuses api_key, so it can never send one to a base_url.
    with pytest.raises(ValueError, match="base_url"):
        ReflectiveMemory({"provider": "vllm", "model": "m", "base_url": ELSEWHERE})


def test_profile_memory_needs_its_key_beside_its_base_url(tmp_path):
    store = FileMarkdownStore(tmp_path / "ada.md")
    with pytest.raises(ValueError, match="api_key"):
        ProfileMemory(store, llm_config={"provider": "vllm", "base_url": ELSEWHERE})
    with pytest.raises(ValueError, match="https"):
        ProfileMemory(store, llm_config={
            "provider": "vllm", "base_url": "http://collector.example.com", "api_key": SENTINEL,
        })


def test_profile_memory_learns_at_its_base_url(tmp_path, monkeypatch):
    flow = load_flow(_FLOW.replace("nodes:", "long_term_memory: profile\nnodes:"))
    (tmp_path / "profile").mkdir()
    store = FileMarkdownStore(tmp_path / "profile" / "ada.md")
    store.write("# Ada\n", base_version=store.read()[1])
    memory = ProfileMemory(store, llm_config={
        "provider": "vllm", "model": "profiler", "base_url": ELSEWHERE, "api_key": SENTINEL,
    })
    requests = []
    replies = ("ok", _NO_FEEDBACK, "<profile>\n# Ada\n- Prefers brief replies.\n</profile>")
    monkeypatch.setenv("VLLM_API_KEY", "sk-environment-key")
    with httpx.Client(transport=_chat_completions(requests, *replies)) as transport:
        _route_openai_client(monkeypatch, transport)
        result = run_flow(
            flow, {}, learnable=True, memory={"profile": memory},
            llm_config={"provider": "vllm", "model": "served-model"},
        )
        outcome = result.backward("Be brief.")
    assert outcome.status == "succeeded", outcome.failure
    learner = [r for r in requests if json.loads(r.content)["model"] == "profiler"]
    assert [str(r.url) for r in learner] == [f"{ELSEWHERE}/chat/completions"]
    assert [r.headers["Authorization"] for r in learner] == [f"Bearer {SENTINEL}"]
    others = [r for r in requests if r not in learner]
    assert {r.headers["Authorization"] for r in others} == {"Bearer sk-environment-key"}


def test_openai_profile_memory_builds_its_learner_with_its_url_and_key(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-environment-key")
    monkeypatch.setenv("OPENAI_BASE_URL", ELSEWHERE)
    flow = load_flow(_FLOW.replace("nodes:", "long_term_memory: profile\nnodes:"))
    (tmp_path / "profile").mkdir()
    store = FileMarkdownStore(tmp_path / "profile" / "ada.md")
    store.write("# Ada\n", base_version=store.read()[1])
    memory = ProfileMemory(store, llm_config={**_AZURE, "model": "profiler"})
    built = _fake_clients(
        monkeypatch, "ok", _NO_FEEDBACK, "<profile>\n# Ada\n- Prefers brief replies.\n</profile>"
    )
    result = run_flow(
        flow, {}, learnable=True, memory={"profile": memory},
        llm_config={"provider": "vllm", "model": "served-model"},
    )
    outcome = result.backward("Be brief.")
    assert outcome.status == "succeeded", outcome.failure
    assert ("openai", "profiler", AZURE_V1, SENTINEL) in built
    assert all(entry[2:] == (None, None) for entry in built if entry[1] != "profiler")
    assert "Prefers brief replies" in store.read()[0]
