"""A run-level `llm_config.base_url` goes only with the host key, and the log records it."""

import json

import httpx
import pytest
from langchain_core.messages import AIMessage

import agent_composer.llm_clients as clients
from agent_composer.compose import load_flow
from agent_composer.compose.helpers.errors import LoadError
from agent_composer.run import run_flow
from tests.engine.test_run_api_key import _assert_hidden, _chat_completions, _route_openai_client
from tests.fixtures.openai_responses import ResponsesServer

SENTINEL = "sk-sentinel-c40a9e17"
AZURE_V1 = "https://res.openai.azure.com/openai/v1/"
ELSEWHERE = "https://collector.example.com/v1"

_FLOW = """id: routed
name: routed
nodes:
  agent: {kind: agent, mode: plain, prompt: hi}
output: ${agent.output}
"""
_AZURE = {"provider": "openai", "model": "gpt-5.4", "base_url": AZURE_V1, "api_key": SENTINEL}
_VLLM = {
    "provider": "vllm", "model": "served-model",
    "base_url": "https://gateway.example.com/v1", "api_key": SENTINEL,
}


@pytest.fixture(autouse=True)
def clean_env(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_COMPOSER_AUTH_FILE", str(tmp_path / "auth.json"))
    for name in (
        "AGENT_COMPOSER_DEFAULT_PROVIDER", "AGENT_COMPOSER_DEFAULT_MODEL",
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
      nested: {kind: agent, mode: plain, prompt: hi, llm_config: {model: nested-model}}
    output: ${nested.output}
nodes:
  inherits: {kind: agent, mode: plain, prompt: hi}
  child: {kind: call, call: body, depends_on: [inherits]}
  other:
    kind: agent
    mode: plain
    prompt: hi
    depends_on: [inherits]
    llm_config: {provider: anthropic, model: claude-sonnet-4-5}
  optout:
    kind: agent
    mode: plain
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
      nested: {{kind: agent, mode: plain, prompt: hi}}
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


@pytest.mark.parametrize("layer", ["flow", "node"])
def test_a_lower_layer_base_url_cannot_redirect_the_host_key(monkeypatch, layer):
    # A flow built in Python skips the loader; the cascade still refuses the URL.
    seen = _fake_models(monkeypatch)
    flow = load_flow(_FLOW)
    if layer == "flow":
        flow.compiled.flow_llm_config = {"base_url": ELSEWHERE}
    else:
        flow.compiled.nodes["agent"].own_llm_config = {"base_url": ELSEWHERE}
    with pytest.raises(ValueError, match="base_url"):
        run_flow(flow, {}, llm_config=_AZURE)
    assert seen == {}
