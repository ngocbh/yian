"""Provider blocks cascade through a real run into each node's request body."""

import importlib
import json

import anthropic
import httpx
import pytest

from yian.compose import load_flow
from yian.llm_clients.anthropic_client import AnthropicClient
from yian.llm_clients.openai_client import OpenAIClient
from yian.run import run_flow

_FLOW = """
id: blocks
name: Blocks
llm_config:
  provider: anthropic
  model: claude-sonnet-4-5
  anthropic:
    max_tokens: 2048
    output_config:
      effort: low
  openai:
    reasoning:
      effort: high
nodes:
  a:
    kind: agent
    prompt: Say a.
    output: str
  b:
    kind: agent
    prompt: Say b.
    output: str
    llm_config:
      anthropic:
        max_tokens: 512
  c:
    kind: agent
    prompt: Say c.
    output: str
    llm_config:
      provider: openai
      model: gpt-5.5
output: ${a.output}
"""

_ENV_KEYS = (
    "ANTHROPIC_BASE_URL", "OPENAI_BASE_URL", "YIAN_DEFAULT_PROVIDER", "YIAN_DEFAULT_MODEL",
)


def _anthropic_httpx():
    """Return the httpx package this Anthropic SDK is built on (newer SDKs use `httpx2`)."""
    client = next(c for c in anthropic.DefaultHttpxClient.__mro__ if c.__name__ == "Client")
    return importlib.import_module(client.__module__.split(".")[0])


class _Server:
    """Record request bodies by prompt; answer Messages and Responses with one reply."""

    def __init__(self, httpx_module):
        self.httpx = httpx_module
        self.paths = {}
        self.bodies = {}

    def __call__(self, request):
        body = json.loads(request.content)
        text = json.dumps(body)
        name = next(n for n in ("a", "b", "c") if f"Say {n}." in text)
        self.paths[name] = request.url.path
        self.bodies[name] = body
        if request.url.path.endswith("/responses"):
            return self.httpx.Response(200, json={
                "id": "resp_1", "object": "response", "created_at": 0, "status": "completed",
                "model": "m", "error": None, "incomplete_details": None, "usage": None,
                "output": [{
                    "id": "msg_1", "type": "message", "role": "assistant", "status": "completed",
                    "content": [{"type": "output_text", "text": name, "annotations": []}],
                }],
            })
        return self.httpx.Response(200, json={
            "id": "msg_1", "type": "message", "role": "assistant", "model": "m",
            "content": [{"type": "text", "text": name}],
            "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        })


@pytest.fixture
def server(tmp_path, monkeypatch):
    """Route the Anthropic and OpenAI clients of a run to one recording server."""
    monkeypatch.setenv("YIAN_AUTH_FILE", str(tmp_path / "auth.json"))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test")
    for name in _ENV_KEYS:
        monkeypatch.delenv(name, raising=False)
    anthropic_httpx = _anthropic_httpx()
    anthropic_server = _Server(anthropic_httpx)
    openai_server = _Server(httpx)
    with (
        anthropic_httpx.Client(transport=anthropic_httpx.MockTransport(anthropic_server)) as ant_http,
        httpx.Client(transport=httpx.MockTransport(openai_server)) as openai_http,
    ):
        original_anthropic = AnthropicClient.get_llm
        original_openai = OpenAIClient.get_llm

        def anthropic_llm(wrapper):
            chat = original_anthropic(wrapper)
            # ChatAnthropic builds its SDK client lazily in a cached property.
            chat.__dict__["_client"] = anthropic.Anthropic(
                api_key="k", http_client=ant_http, max_retries=0, timeout=30
            )
            return chat

        def openai_llm(wrapper):
            wrapper.kwargs.update(http_client=openai_http, max_retries=0)
            return original_openai(wrapper)

        monkeypatch.setattr(AnthropicClient, "get_llm", anthropic_llm)
        monkeypatch.setattr(OpenAIClient, "get_llm", openai_llm)
        yield anthropic_server, openai_server


def _run_started(log_path):
    """Return the payload of the run log's `RunStarted` record."""
    records = [json.loads(line) for line in log_path.read_text().splitlines()]
    (started,) = [record for record in records if record["kind"] == "run_started"]
    return started["payload"]


def test_blocks_merge_per_field_and_reach_each_request_body(server, tmp_path):
    anthropic_server, openai_server = server
    log = tmp_path / "run.jsonl"
    result = run_flow(load_flow(_FLOW), {}, log=log)
    assert result.status == "succeeded", result.failure
    assert result.output == "a"

    assert anthropic_server.paths == {"a": "/v1/messages", "b": "/v1/messages"}
    a, b = anthropic_server.bodies["a"], anthropic_server.bodies["b"]
    assert a["max_tokens"] == 2048
    assert a["output_config"] == {"effort": "low"}
    assert b["max_tokens"] == 512
    assert b["output_config"] == {"effort": "low"}
    assert "reasoning" not in a and "reasoning" not in b

    assert openai_server.paths == {"c": "/v1/responses"}
    c = openai_server.bodies["c"]
    assert c["model"] == "gpt-5.5"
    assert c["reasoning"] == {"effort": "high"}
    for anthropic_field in ("max_tokens", "output_config", "thinking", "top_k"):
        assert anthropic_field not in c

    effective = _run_started(log)["effective_llm_by_static_path"]
    openai_block = {"reasoning": {"effort": "high"}}
    assert set(effective) == {"a", "b", "c"}
    assert effective["a"]["provider"] == "anthropic"
    assert effective["a"]["anthropic"] == {"max_tokens": 2048, "output_config": {"effort": "low"}}
    assert effective["a"]["openai"] == openai_block
    assert effective["b"]["provider"] == "anthropic"
    assert effective["b"]["anthropic"] == {"max_tokens": 512, "output_config": {"effort": "low"}}
    assert effective["b"]["openai"] == openai_block
    assert effective["c"]["provider"] == "openai"
    assert effective["c"]["model"] == "gpt-5.5"
    assert effective["c"]["openai"] == openai_block
    assert effective["c"]["anthropic"] == {"max_tokens": 2048, "output_config": {"effort": "low"}}
    for config in effective.values():
        assert "api_key" not in config
