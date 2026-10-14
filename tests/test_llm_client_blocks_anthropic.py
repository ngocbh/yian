"""The ``anthropic`` block reaches the Messages request body as written; a host's HTTP
clients carry the requests and never reach the body."""

import asyncio
import importlib
import inspect
import json
import warnings

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


# provider -> a model id and a block whose every field must reach the request body.
BODY_CASES = {"anthropic": {"model": "claude-sonnet-4-5", "block": FULL_BLOCK}}


@pytest.mark.parametrize("provider", sorted(BODY_CASES))
def test_every_block_field_is_sent(server, provider):
    case = BODY_CASES[provider]
    _model(case["block"], model=case["model"]).invoke([HumanMessage("hi")])
    body = server.bodies[0]
    for key, value in case["block"].items():
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


@pytest.mark.parametrize("thinking", [
    {"type": "adaptive"},
    {"type": "enabled", "budget_tokens": 2048},
])
@pytest.mark.parametrize("include_raw", [False, True])
def test_structured_output_with_block_thinking_uses_the_output_format(
    server, thinking, include_raw
):
    # The Messages API rejects thinking beside a forced tool call.
    server.text = '{"answer": "ok"}'
    model = _model({"thinking": thinking, "max_tokens": 4096})
    result = model.with_structured_output(Answer, include_raw=include_raw).invoke("hi")
    assert (result["parsed"] if include_raw else result) == Answer(answer="ok")
    body = server.bodies[0]
    assert body["thinking"] == thinking
    assert body["output_config"]["format"]["type"] == "json_schema"
    assert "tool_choice" not in body
    assert "tools" not in body


def test_structured_output_with_block_thinking_keeps_an_explicit_method(server):
    model = _model({"thinking": {"type": "adaptive"}})
    model.with_structured_output(Answer, method="function_calling").invoke("hi")
    assert "output_config" not in server.bodies[0]
    assert server.bodies[0]["tools"][0]["name"] == "Answer"


@pytest.mark.parametrize("block", [None, {"top_k": 5}, {"thinking": {"type": "disabled"}}])
def test_structured_output_without_block_thinking_forces_the_tool(server, block):
    _model(block).with_structured_output(Answer).invoke("hi")
    body = server.bodies[0]
    assert body["tool_choice"] == {"type": "tool", "name": "Answer"}
    assert "output_config" not in body


@pytest.fixture
def clean_env(tmp_path, monkeypatch):
    monkeypatch.setenv("YIAN_AUTH_FILE", str(tmp_path / "auth.json"))
    for name in (*_ENV_KEYS, "ANTHROPIC_PROXY"):
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


@pytest.fixture
def host_clients():
    """A recorder and the SDK's own sync and async httpx clients routed to it."""
    httpx = _anthropic_httpx()
    recorder = _Server()
    transport = httpx.MockTransport(recorder)
    sync, async_ = httpx.Client(transport=transport), httpx.AsyncClient(transport=transport)
    yield recorder, sync, async_
    sync.close()
    asyncio.run(async_.aclose())


def test_a_host_http_client_carries_every_request(clean_env, host_clients):
    recorder, sync, async_ = host_clients
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # langchain warns when a key goes to model_kwargs
        chat = AnthropicClient(
            "claude-sonnet-4-5", api_key="k", max_retries=0,
            http_client=sync, http_async_client=async_,
        ).get_llm()
    assert chat.model_kwargs == {}
    assert chat.invoke("hi").content == "ok"
    assert asyncio.run(chat.ainvoke("hi")).content == "ok"
    assert [str(r.url) for r in recorder.requests] == [
        "https://api.anthropic.com/v1/messages"
    ] * 2
    for body in recorder.bodies:
        assert "http_client" not in body and "http_async_client" not in body
    assert not {"http_client", "http_async_client"} & set(chat.model_dump())


def test_a_non_redirecting_host_client_is_used_with_a_base_url(clean_env, host_clients):
    recorder, sync, _ = host_clients
    chat = AnthropicClient(
        "claude-sonnet-4-5", base_url="https://gateway.test", api_key="k",
        max_retries=0, http_client=sync,
    ).get_llm()
    assert chat.invoke("hi").content == "ok"
    assert [str(r.url) for r in recorder.requests] == ["https://gateway.test/v1/messages"]


@pytest.mark.parametrize(
    ("key", "factory"),
    [
        ("http_client", lambda: anthropic.DefaultHttpxClient()),
        ("http_async_client", lambda: anthropic.DefaultAsyncHttpxClient()),
    ],
)
def test_a_redirecting_host_client_is_rejected_beside_a_base_url(clean_env, key, factory):
    client = factory()
    try:
        with pytest.raises(ValueError, match=f"{key} follows redirects.*follow_redirects=False"):
            AnthropicClient(
                "claude-sonnet-4-5", base_url="https://gateway.test", api_key="k",
                **{key: client},
            ).get_llm()
    finally:
        if key == "http_client":
            client.close()
        else:
            asyncio.run(client.aclose())


def test_a_directly_built_model_rejects_a_redirecting_client_beside_a_base_url():
    with anthropic.DefaultHttpxClient() as client:
        chat = NormalizedChatAnthropic(
            model="claude-sonnet-4-5", api_key="k", base_url="https://gateway.test",
            follow_redirects=False, http_client=client,
        )
        with pytest.raises(ValueError, match="http_client follows redirects"):
            chat._client


def test_a_redirecting_host_client_is_allowed_without_a_base_url(clean_env):
    with anthropic.DefaultHttpxClient() as client:
        chat = AnthropicClient("claude-sonnet-4-5", api_key="k", http_client=client).get_llm()
        assert chat._client._client is client


def test_a_proxy_beside_a_host_client_is_rejected(clean_env, host_clients):
    _, sync, _ = host_clients
    clean_env.setenv("ANTHROPIC_PROXY", "http://proxy.test:3128")
    with pytest.raises(ValueError, match="ANTHROPIC_PROXY can't be used with http_client"):
        AnthropicClient("claude-sonnet-4-5", api_key="k", http_client=sync).get_llm()


def test_a_host_client_of_the_wrong_type_fails_when_the_model_is_built(clean_env):
    with pytest.raises(TypeError, match="Invalid `http_client` argument; Expected an instance"):
        AnthropicClient("claude-sonnet-4-5", api_key="k", http_client=object()).get_llm()


@pytest.mark.skipif(
    _anthropic_httpx().__name__ == "httpx", reason="the SDK is built on httpx itself"
)
def test_a_plain_httpx_client_on_an_httpx2_sdk_fails_when_the_model_is_built(clean_env):
    import httpx

    with httpx.Client() as client, pytest.raises(TypeError, match="is from the `httpx` package"):
        AnthropicClient("claude-sonnet-4-5", api_key="k", http_client=client).get_llm()
