"""The ``anthropic`` block reaches the Messages request body as written."""

import importlib
import inspect
import json

import anthropic
import pytest
from anthropic.resources.messages import Messages
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from yian.llm_clients.anthropic_client import AnthropicClient, NormalizedChatAnthropic
from yian.llm_clients.factory import model_from_config
from yian.llm_clients.provider_blocks import AnthropicMessagesBlock

_ENV_KEYS = (
    "ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL", "YIAN_DEFAULT_PROVIDER", "YIAN_DEFAULT_MODEL",
)
_SAMPLING = {"temperature", "top_k", "top_p"}


def _anthropic_httpx():
    """Return the httpx package this Anthropic SDK is built on (newer SDKs use `httpx2`)."""
    client = next(c for c in anthropic.DefaultHttpxClient.__mro__ if c.__name__ == "Client")
    return importlib.import_module(client.__module__.split(".")[0])


class _Server:
    """Record Messages request bodies and answer each with one text reply."""

    def __init__(self):
        self.text = "ok"
        self.requests = []
        self.bodies = []

    def __call__(self, request):
        self.requests.append(request)
        self.bodies.append(json.loads(request.content))
        return _anthropic_httpx().Response(200, json={
            "id": "msg_1", "type": "message", "role": "assistant", "model": "m",
            "content": [{"type": "text", "text": self.text}],
            "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        })


@pytest.fixture
def server(tmp_path, monkeypatch):
    """Route every chat model `AnthropicClient` builds to a recording server."""
    monkeypatch.setenv("YIAN_AUTH_FILE", str(tmp_path / "auth.json"))
    for name in _ENV_KEYS:
        monkeypatch.delenv(name, raising=False)
    recorder = _Server()
    httpx = _anthropic_httpx()
    with httpx.Client(transport=httpx.MockTransport(recorder)) as http_client:
        original = AnthropicClient.get_llm

        def construct(wrapper):
            chat = original(wrapper)
            # ChatAnthropic builds its SDK client lazily in a cached property.
            chat.__dict__["_client"] = anthropic.Anthropic(
                api_key="k", http_client=http_client, max_retries=0, timeout=30
            )
            return chat

        monkeypatch.setattr(AnthropicClient, "get_llm", construct)
        yield recorder


def _model(block=None, model="claude-sonnet-4-5"):
    config = {"provider": "anthropic", "model": model, "api_key": "k"}
    if block is not None:
        config["anthropic"] = block
    return model_from_config(config)


class Answer(BaseModel):
    answer: str


def test_block_effort_sits_beside_the_structured_output_format(server):
    server.text = '{"answer": "ok"}'
    model = _model({"output_config": {"effort": "high"}})
    result = model.with_structured_output(Answer, method="json_schema").invoke("hi")
    assert result == Answer(answer="ok")
    output_config = server.bodies[0]["output_config"]
    assert output_config["effort"] == "high"
    assert output_config["format"]["type"] == "json_schema"


def test_block_does_not_leak_into_bound_call_options(server):
    server.text = '{"answer": "ok"}'
    structured = _model({"output_config": {"effort": "high"}}).with_structured_output(
        Answer, method="json_schema"
    )
    structured.invoke("hi")
    bound = structured.first.kwargs["output_config"]
    assert set(bound) == {"format"}


def test_sampling_and_length_reach_the_body_unchanged(server):
    _model({"temperature": 0.2, "top_k": 5, "max_tokens": 8000}).invoke([HumanMessage("hi")])
    body = server.bodies[0]
    assert (body["temperature"], body["top_k"], body["max_tokens"]) == (0.2, 5, 8000)


def test_thinking_reaches_the_body_as_written(server):
    block = {"thinking": {"type": "enabled", "budget_tokens": 2048}, "max_tokens": 4096}
    _model(block).invoke([HumanMessage("hi")])
    body = server.bodies[0]
    assert body["thinking"] == {"type": "enabled", "budget_tokens": 2048}
    assert body["max_tokens"] == 4096


@pytest.mark.parametrize("model", ["claude-opus-4-5-20251101", "claude-haiku-4-5"])
def test_block_effort_is_sent_for_any_model_id(server, model):
    _model({"output_config": {"effort": "low"}}, model=model).invoke([HumanMessage("hi")])
    assert server.bodies[0]["output_config"] == {"effort": "low"}


FULL_BLOCK = {
    "cache_control": {"type": "ephemeral", "ttl": "5m"},
    "inference_geo": "us",
    "max_tokens": 4096,
    "metadata": {"user_id": "user-1"},
    "output_config": {"effort": "medium"},
    "service_tier": "auto",
    "stop_sequences": ["END"],
    "temperature": 0.5,
    "thinking": {"type": "enabled", "budget_tokens": 1024},
    "top_k": 10,
    "top_p": 0.9,
}


def test_full_block_covers_every_field():
    assert set(FULL_BLOCK) == set(AnthropicMessagesBlock.model_fields)
    AnthropicMessagesBlock.model_validate(FULL_BLOCK)


def test_every_block_field_is_sent(server):
    _model(FULL_BLOCK).invoke([HumanMessage("hi")])
    body = server.bodies[0]
    for key, value in FULL_BLOCK.items():
        assert body[key] == value, key


def test_every_block_field_is_a_messages_keyword_or_a_relocated_sampling_field():
    # langchain-anthropic moves the sampling fields into extra_body when the SDK lacks them.
    keywords = set(inspect.signature(Messages.create).parameters)
    assert set(AnthropicMessagesBlock.model_fields) <= keywords | _SAMPLING


def test_key_outside_the_sdk_signature_goes_through_extra_body(server):
    chat = AnthropicClient("claude-sonnet-4-5", api_key="k", block={"speed": "fast"}).get_llm()
    payload = chat._get_request_payload([HumanMessage("hi")])
    assert payload["extra_body"] == {"speed": "fast"}
    chat.invoke([HumanMessage("hi")])
    assert server.bodies[0]["speed"] == "fast"


def test_block_merges_into_a_caller_extra_body(server):
    chat = AnthropicClient("claude-sonnet-4-5", api_key="k", block={"speed": "fast"}).get_llm()
    payload = chat._get_request_payload([HumanMessage("hi")], extra_body={"x": 1})
    assert payload["extra_body"] == {"x": 1, "speed": "fast"}


def test_block_field_is_hidden_from_serialization():
    chat = NormalizedChatAnthropic(model="claude-sonnet-4-5", api_key="k", block={"top_k": 5})
    assert "block" not in chat.model_dump()
    assert "top_k" not in repr(chat)


def test_without_a_block_the_body_is_unchanged(server):
    _model().invoke([HumanMessage("hi")])
    assert server.requests[0].url.path == "/v1/messages"
    assert server.bodies[0] == {
        "max_tokens": 64000,
        "messages": [{"content": "hi", "role": "user"}],
        "model": "claude-sonnet-4-5",
    }
