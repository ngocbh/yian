"""Coverage for the llm_clients provider features ported from TradingAgents.

Construction-only (no network): builds clients with dummy keys and asserts the
capability table, client construction, and the dual-region providers.
"""

import pytest

from yian.llm_clients import create_llm_client
from yian.llm_clients.api_key_env import get_api_key_env
from yian.llm_clients.capabilities import get_capabilities
from yian.llm_clients.config import LLMConfig, SUPPORTED_LLM_PROVIDERS
from yian.llm_clients.factory import (
    _DEDICATED_PROVIDERS,
    _OPENAI_COMPATIBLE,
)


_FACTORY_SUPPORTED_PROVIDERS = (
    *_OPENAI_COMPATIBLE,
    *_DEDICATED_PROVIDERS,
)


# --- capability table ------------------------------------------------------- #


def test_capabilities_deepseek_reasoner_rejects_tool_choice():
    caps = get_capabilities("deepseek-reasoner")
    assert caps.supports_tool_choice is False


def test_capabilities_minimax_requires_reasoning_split():
    assert get_capabilities("MiniMax-M2.7").requires_reasoning_split is True


def test_capabilities_default_model_allows_tool_choice():
    caps = get_capabilities("gpt-4o")
    assert caps.supports_tool_choice is True
    assert caps.preferred_structured_method == "function_calling"


def test_capabilities_forward_compat_pattern():
    # an unseen deepseek-v5 variant inherits the thinking-mode quirks
    assert get_capabilities("deepseek-v5-pro").supports_tool_choice is False


# --- google: no langchain temperature default ------------------------------ #


def test_google_model_never_carries_a_temperature():
    # A temperature comes only from the google block's generationConfig.
    m = create_llm_client("google", "gemini-2.5-flash", api_key="x", temperature=0.4).get_llm()
    assert m.temperature is None


def test_azure_endpoint_base_stripping(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "x")
    monkeypatch.setenv("OPENAI_API_VERSION", "2024-02-01")
    # A portal's v1 surface is stripped back to the base resource URL.
    m = create_llm_client(
        "azure", "gpt-4o", base_url="https://res.openai.azure.com/openai/v1"
    ).get_llm()
    assert str(m.azure_endpoint) == "https://res.openai.azure.com"


def test_azure_reads_neither_deployment_nor_endpoint_from_the_environment(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "x")
    monkeypatch.setenv("OPENAI_API_VERSION", "2024-02-01")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://env-resource.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT_NAME", "env-deployment")
    for model in ("gpt-5.5", "gpt-4.1"):
        m = create_llm_client("azure", model, base_url="https://res.openai.azure.com/").get_llm()
        assert m.deployment_name == model
        assert str(m.azure_endpoint) == "https://res.openai.azure.com"


def test_azure_without_base_url_fails_and_says_to_set_it(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "x")
    monkeypatch.setenv("OPENAI_API_VERSION", "2024-02-01")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://env-resource.openai.azure.com")
    with pytest.raises(ValueError, match="llm_config.base_url"):
        create_llm_client("azure", "gpt-4o").get_llm()


# --- factory: dual-region providers ----------------------------------------- #


@pytest.mark.parametrize("provider", _FACTORY_SUPPORTED_PROVIDERS)
def test_llm_config_accepts_every_factory_supported_provider(provider):
    assert LLMConfig(provider=provider).provider == provider


def test_llm_config_and_factory_provider_vocabularies_match():
    assert set(SUPPORTED_LLM_PROVIDERS) == set(_FACTORY_SUPPORTED_PROVIDERS)


def test_llm_config_still_rejects_unknown_provider():
    with pytest.raises(ValueError, match="provider"):
        LLMConfig(provider="not-a-provider")


@pytest.mark.parametrize("provider", ["qwen-cn", "glm-cn", "minimax", "minimax-cn"])
def test_factory_resolves_new_dual_region_providers(provider):
    from yian.llm_clients.openai_client import OpenAIClient

    client = create_llm_client(provider, "some-model")
    assert isinstance(client, OpenAIClient)
    assert client.provider == provider


def test_api_key_env_maps_dual_region_keys():
    assert get_api_key_env("qwen-cn") == "DASHSCOPE_CN_API_KEY"
    assert get_api_key_env("glm-cn") == "ZHIPU_CN_API_KEY"
    assert get_api_key_env("ollama") is None  # local runtime, no key


# --- vLLM (OpenAI-compatible, optional key) --------------------------------- #


def test_factory_resolves_vllm_as_openai_compatible():
    from yian.llm_clients.openai_client import OpenAIClient

    client = create_llm_client("vllm", "meta-llama/Llama-3.1-8B-Instruct")
    assert isinstance(client, OpenAIClient)
    assert client.provider == "vllm"


def test_vllm_keyless_sends_dummy_key_and_default_base_url(monkeypatch):
    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    monkeypatch.delenv("VLLM_BASE_URL", raising=False)
    m = create_llm_client("vllm", "some-local-model").get_llm()
    assert str(m.openai_api_base) == "http://localhost:8000/v1"
    assert m.openai_api_key.get_secret_value() == "EMPTY"  # keyless -> dummy, no raise


def test_vllm_uses_api_key_and_base_url_when_set(monkeypatch):
    monkeypatch.setenv("VLLM_API_KEY", "secret-token")
    monkeypatch.setenv("VLLM_BASE_URL", "http://gpu-box:8001/v1")
    m = create_llm_client("vllm", "some-local-model").get_llm()
    assert str(m.openai_api_base) == "http://gpu-box:8001/v1"
    assert m.openai_api_key.get_secret_value() == "secret-token"


def test_vllm_accepts_any_model_name():
    from yian.llm_clients.validators import validate_model

    assert validate_model("vllm", "any/arbitrary-served-model") is True


def test_the_picker_custom_sentinel_is_not_a_known_model():
    from yian.llm_clients.validators import validate_model

    assert validate_model("glm", "custom") is False
    assert validate_model("deepseek", "custom") is False


def test_claude_mythos_5_1_is_a_known_anthropic_model():
    """Mythos 5.1 validates and builds an Anthropic client without an unknown-model warning."""
    import warnings

    from yian.llm_clients.anthropic_client import AnthropicClient
    from yian.llm_clients.validators import validate_model

    assert validate_model("anthropic", "claude-mythos-5-1") is True
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        AnthropicClient("claude-mythos-5-1", api_key="k").warn_if_unknown_model()
    assert not [w for w in caught if issubclass(w.category, RuntimeWarning)]


# --- Ollama (native client, no default think) ------------------------------ #


def test_factory_routes_ollama_to_native_client():
    from yian.llm_clients.ollama_client import OllamaClient

    client = create_llm_client("ollama", "qwen3.5:35b")
    assert isinstance(client, OllamaClient)


def test_ollama_get_llm_sets_no_thinking_default_and_strips_v1(monkeypatch):
    # an OpenAI-style base URL (with /v1) must be normalized to the native one
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://gpu-box:11434/v1")
    m = create_llm_client("ollama", "qwen3.5:35b").get_llm()
    assert m.reasoning is None  # no default: `think` is sent only from the ollama block
    assert str(m.base_url) == "http://gpu-box:11434"


def test_ollama_drops_unsupported_kwargs():
    # model_from_config sets max_retries, which ChatOllama does not accept;
    # the client must filter it rather than crash on construction.
    m = create_llm_client("ollama", "qwen3.5:35b", max_retries=6).get_llm()
    assert m.model == "qwen3.5:35b"


def test_openai_responses_preserve_reasoning_for_next_tool_turn():
    import httpx
    from langchain_core.messages import HumanMessage, ToolMessage
    from yian.nodes.agent.text import text_of
    from tests.fixtures.openai_responses import ResponsesServer

    server = ResponsesServer()
    with httpx.Client(transport=httpx.MockTransport(server)) as client:
        model = create_llm_client(
            "openai", "gpt-5.5", api_key="synthetic-api-key", http_client=client,
        ).get_llm()
        first = model.invoke([HumanMessage("Double 3.")])
        assert isinstance(first.content, list)
        assert any(block.get("encrypted_content") == "synthetic-continuation" for block in first.content)
        final = model.invoke([
            HumanMessage("Double 3."), first, ToolMessage("6", tool_call_id="call_test"),
        ])
        assert text_of(final) == "6"
    assert any(item.get("encrypted_content") == "synthetic-continuation" for item in server.bodies[1]["input"])
    assert server.requests[0].headers["Authorization"] == "Bearer synthetic-api-key"


# --- streaming: normalization, usage, DeepSeek fallback --------------------- #

_AZURE_ENV = {"AZURE_OPENAI_API_KEY": "x", "OPENAI_API_VERSION": "2024-02-01"}


@pytest.mark.parametrize(
    ("provider", "model", "kwargs", "env"),
    [
        ("anthropic", "claude-sonnet-4-6", {"api_key": "x"}, {}),
        ("google", "gemini-2.5-flash", {"api_key": "x"}, {}),
        ("azure", "gpt-4o", {"base_url": "https://res.openai.azure.com/"}, _AZURE_ENV),
        ("vllm", "some-local-model", {}, {}),
    ],
)
def test_normalize_reply_matches_invoke_normalization(monkeypatch, provider, model, kwargs, env):
    from langchain_core.messages import AIMessage

    for name, value in env.items():
        monkeypatch.setenv(name, value)
    chat = create_llm_client(provider, model, **kwargs).get_llm()
    message = AIMessage(content=[
        {"type": "thinking", "thinking": "hidden"},
        {"type": "text", "text": "visible"},
    ])
    assert chat.normalize_reply(message).content == "visible"


def test_openai_responses_normalize_reply_keeps_blocks():
    from langchain_core.messages import AIMessage

    chat = create_llm_client("openai", "gpt-5.5", api_key="x").get_llm()
    blocks = [{"type": "reasoning", "id": "r"}, {"type": "text", "text": "visible"}]
    assert chat.normalize_reply(AIMessage(content=list(blocks))).content == blocks


def _sse_server(bodies):
    """Serve one chat-completions SSE stream whose final chunk carries usage."""
    import httpx
    import json as _json

    def handler(request):
        body = _json.loads(request.content)
        bodies.append(body)
        chunks = [
            {"choices": [{"index": 0, "delta": {"role": "assistant", "content": "hel"}}]},
            {"choices": [{"index": 0, "delta": {"content": "lo"}, "finish_reason": "stop"}]},
            {"choices": [], "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6}},
        ]
        lines = "".join(
            "data: " + _json.dumps({"id": "c", "object": "chat.completion.chunk", "created": 0,
                                    "model": "m", **chunk}) + "\n\n"
            for chunk in chunks
        ) + "data: [DONE]\n\n"
        return httpx.Response(200, text=lines, headers={"content-type": "text/event-stream"})

    return httpx.MockTransport(handler)


def test_openai_compatible_stream_requests_and_reports_usage(monkeypatch):
    import httpx
    from langchain_core.messages import HumanMessage

    monkeypatch.setenv("VLLM_API_KEY", "synthetic-key")
    bodies = []
    with httpx.Client(transport=_sse_server(bodies)) as client:
        model = create_llm_client("vllm", "some-local-model", http_client=client).get_llm()
        chunks = list(model.stream([HumanMessage("hi")]))
    assert model.stream_usage is True
    assert bodies[0]["stream_options"] == {"include_usage": True}
    total = chunks[0]
    for chunk in chunks[1:]:
        total = total + chunk
    assert total.content == "hello"
    assert total.usage_metadata["input_tokens"] == 4
    assert total.usage_metadata["output_tokens"] == 2


@pytest.mark.parametrize("base_url", [None, "https://gateway.example/v1"])
def test_native_openai_uses_responses_api_with_or_without_a_base_url(base_url):
    # Responses API streams report usage in `response.completed`, whatever
    # `stream_usage` says; a saved OpenAI connection sets `base_url`.
    m = create_llm_client("openai", "gpt-5.5", base_url=base_url, api_key="x").get_llm()
    assert m.use_responses_api is True


def test_deepseek_streams_through_its_non_streaming_path(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "synthetic-key")
    m = create_llm_client("deepseek", "deepseek-reasoner").get_llm()
    assert m.disable_streaming is True
    assert m._should_stream(async_api=False, stream=True) is False


# --- anthropic: early close releases the HTTP response --------------------- #


def _anthropic_httpx():
    """Return the httpx package this Anthropic SDK is built on (newer SDKs use `httpx2`)."""
    import importlib

    import anthropic

    client = next(c for c in anthropic.DefaultHttpxClient.__mro__ if c.__name__ == "Client")
    return importlib.import_module(client.__module__.split(".")[0])


def test_anthropic_stream_closed_early_closes_its_http_response():
    import json as _json

    import anthropic
    from langchain_core.messages import HumanMessage

    httpx = _anthropic_httpx()
    state = {"sent": 0, "closed": False}

    def event(name, data):
        return f"event: {name}\ndata: {_json.dumps({'type': name, **data})}\n\n".encode()

    class Body(httpx.SyncByteStream):
        def __iter__(self):
            yield event("message_start", {"message": {
                "id": "m", "type": "message", "role": "assistant", "model": "claude-x",
                "content": [], "stop_reason": None, "stop_sequence": None,
                "usage": {"input_tokens": 10, "output_tokens": 1},
            }})
            yield event("content_block_start",
                        {"index": 0, "content_block": {"type": "text", "text": ""}})
            for index in range(50):
                state["sent"] += 1
                yield event("content_block_delta",
                            {"index": 0, "delta": {"type": "text_delta", "text": f"w{index} "}})
            yield event("content_block_stop", {"index": 0})
            yield event("message_stop", {})

        def close(self):
            state["closed"] = True

    def handler(request):
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=Body())

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        chat = create_llm_client("anthropic", "claude-sonnet-4-6", api_key="x").get_llm()
        # ChatAnthropic builds its SDK client lazily in a cached property.
        chat.__dict__["_client"] = anthropic.Anthropic(api_key="x", http_client=client)
        chunks = chat.stream([HumanMessage("hi")])
        texts = [next(chunks).content for _ in range(4)]
        chunks.close()
        assert state["closed"] is True
        assert "w0 " in texts and state["sent"] < 50
        whole = "".join(chunk.content for chunk in chat.stream([HumanMessage("hi")]))
    assert whole == "".join(f"w{index} " for index in range(50))
