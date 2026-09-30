"""A run-level `llm_config.api_key` authenticates agents and stays out of every record."""

import json

import httpx
import pytest

import agent_composer.llm_clients as clients
from agent_composer.compose import load_flow
from agent_composer.compose.helpers.errors import LoadError
from agent_composer.events import NodeStarted
from agent_composer.llm_clients.openai_client import OpenAIClient
from agent_composer.memory.reflective import ReflectiveMemory
from agent_composer.run import resume_flow, run_flow
from tests.fixtures.openai_responses import ResponsesServer

SENTINEL = "sk-sentinel-5b2e8d41"

_FLOW = """id: keyed
name: keyed
nodes:
  agent: {kind: agent, mode: plain, prompt: hi}
output: ${agent.output}
"""


@pytest.fixture(autouse=True)
def clean_env(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_COMPOSER_AUTH_FILE", str(tmp_path / "auth.json"))
    for name in (
        "AGENT_COMPOSER_DEFAULT_PROVIDER",
        "AGENT_COMPOSER_DEFAULT_MODEL",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "VLLM_API_KEY",
        "VLLM_BASE_URL",
        "ANTHROPIC_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)


def _chat_completions(requests):
    """Answer every chat-completions request with one fixed reply."""

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={
            "id": "c", "object": "chat.completion", "created": 0, "model": "m",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "ok"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        })

    return httpx.MockTransport(handler)


def _route_openai_client(monkeypatch, transport):
    original = OpenAIClient.get_llm

    def construct(wrapper):
        wrapper.kwargs.update(http_client=transport, max_retries=0)
        return original(wrapper)

    monkeypatch.setattr(OpenAIClient, "get_llm", construct)


def _assert_hidden(path, events, result):
    assert SENTINEL not in path.read_text()
    assert SENTINEL not in repr(events)
    assert SENTINEL not in repr(result)


def test_openai_compatible_run_sends_the_key_and_records_none_of_it(tmp_path, monkeypatch):
    requests, events = [], []
    path = tmp_path / "run.jsonl"
    with httpx.Client(transport=_chat_completions(requests)) as transport:
        _route_openai_client(monkeypatch, transport)
        result = run_flow(
            load_flow(_FLOW),
            {},
            log=path,
            on_event=events.append,
            llm_config={"provider": "vllm", "model": "served-model", "api_key": SENTINEL},
        )
    assert result.status == "succeeded", result.failure
    assert requests[0].headers["Authorization"] == f"Bearer {SENTINEL}"
    effective = json.loads(path.read_text().splitlines()[0])["payload"][
        "effective_llm_by_static_path"
    ]
    assert effective == {"agent": {"provider": "vllm", "model": "served-model"}}
    _assert_hidden(path, events, result)


def test_openai_run_records_a_host_connection_and_resume_needs_the_key(tmp_path, monkeypatch):
    server, events = ResponsesServer(tool_first=False), []
    path = tmp_path / "run.jsonl"
    llm_config = {"provider": "openai", "model": "gpt-5.5", "api_key": SENTINEL}
    with httpx.Client(transport=httpx.MockTransport(server)) as transport:
        _route_openai_client(monkeypatch, transport)
        result = run_flow(
            load_flow(_FLOW), {}, log=path, on_event=events.append, llm_config=llm_config
        )
    assert result.status == "succeeded", result.failure
    assert server.requests[0].headers["Authorization"] == f"Bearer {SENTINEL}"
    effective = json.loads(path.read_text().splitlines()[0])["payload"][
        "effective_llm_by_static_path"
    ]
    assert effective["agent"]["_connection"]["source"] == "host"
    _assert_hidden(path, events, result)

    def stop(event):
        if isinstance(event, NodeStarted) and event.node_id == "agent":
            raise RuntimeError("host stopped before model")

    interrupted = tmp_path / "interrupted.jsonl"
    with pytest.raises(RuntimeError, match="host stopped"):
        run_flow(load_flow(_FLOW), {}, log=interrupted, on_event=stop, llm_config=llm_config)
    resumed = resume_flow(load_flow(_FLOW), log=interrupted)
    assert resumed.status == "failed"
    assert "llm_config.api_key" in resumed.failure.message
    assert SENTINEL not in interrupted.read_text()


def test_invalid_config_with_a_key_fails_without_echoing_it(tmp_path):
    events = []
    path = tmp_path / "run.jsonl"
    result = run_flow(
        load_flow(_FLOW),
        {},
        log=path,
        on_event=events.append,
        llm_config={
            "provider": "vllm",
            "model": "served-model",
            "api_key": SENTINEL,
            "temperature": "hot",
        },
    )
    assert result.status == "failed"
    assert "temperature" in result.failure.message
    _assert_hidden(path, events, result)


def test_key_goes_only_to_agents_of_its_provider(monkeypatch):
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
    llm_config: {inherit: false, provider: vllm, model: other-model}
output: ${optout.output}
""")
    from langchain_core.messages import AIMessage

    keys = {}

    class Model:
        def invoke(self, messages):
            return AIMessage(content="ok")

    def factory(config):
        key = config.get("api_key")
        keys[config["model"]] = None if key is None else key.get_secret_value()
        return Model()

    monkeypatch.setattr(clients, "model_from_config", factory)
    result = run_flow(
        flow, {}, llm_config={"provider": "vllm", "model": "served-model", "api_key": SENTINEL}
    )
    assert result.status == "succeeded", result.failure
    assert keys == {
        "served-model": SENTINEL,
        "nested-model": SENTINEL,
        "claude-sonnet-4-5": None,
        "other-model": SENTINEL,
    }


def test_key_without_a_provider_is_rejected():
    with pytest.raises(ValueError, match="provider") as caught:
        run_flow(load_flow(_FLOW), {}, llm_config={"api_key": SENTINEL})
    assert SENTINEL not in str(caught.value)


def test_authored_key_is_rejected_without_echoing_it():
    with pytest.raises(LoadError) as caught:
        load_flow(_FLOW.replace("prompt: hi}", f"prompt: hi, llm_config: {{api_key: {SENTINEL}}}}}"))
    assert "api_key" in str(caught.value)
    assert SENTINEL not in str(caught.value)
    flow = load_flow(_FLOW.replace("nodes:", f"llm_config: {{api_key: {SENTINEL}}}\nnodes:"))
    with pytest.raises(ValueError, match="api_key") as caught:
        run_flow(flow, {}, llm_config={"provider": "vllm", "model": "served-model"})
    assert SENTINEL not in str(caught.value)


def test_reflective_memory_refuses_a_key_it_would_record():
    with pytest.raises(ValueError, match="api_key") as caught:
        ReflectiveMemory({"provider": "vllm", "model": "m", "api_key": SENTINEL})
    assert SENTINEL not in str(caught.value)
