"""Provider blocks reach the request body of the API each client calls, as written."""

import json

import httpx
import pytest
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from tests import (
    test_llm_client_blocks_anthropic,
    test_llm_client_blocks_google,
    test_llm_client_blocks_ollama,
    test_openai_subscription,
)
from yian.llm_clients.azure_client import AzureOpenAIClient
from yian.llm_clients.config import SUPPORTED_LLM_PROVIDERS
from yian.llm_clients.factory import model_from_config
from yian.llm_clients.openai_client import OpenAIClient

# provider -> a model id and a block holding one vendor-specific and one standard field.
BLOCK_CASES = {
    "xai": {"model": "grok-4", "block": {"reasoning_effort": "high", "temperature": 0.3}},
    "deepseek": {"model": "deepseek-chat", "block": {"thinking": {"type": "disabled"}, "max_tokens": 50}},
    "qwen": {"model": "qwen-plus", "block": {"enable_search": True, "top_p": 0.5}},
    "qwen-cn": {"model": "qwen-plus", "block": {"enable_thinking": False, "max_tokens": 64}},
    "glm": {"model": "glm-4.6", "block": {"do_sample": False, "temperature": 0.5}},
    "glm-cn": {"model": "glm-4.6", "block": {"thinking": {"type": "disabled"}, "max_tokens": 64}},
    "minimax": {"model": "MiniMax-M2.7", "block": {"thinking": {"type": "adaptive"}, "top_p": 0.9}},
    "minimax-cn": {"model": "MiniMax-M2.7", "block": {"reasoning_effort": "low", "temperature": 0.7}},
    "openrouter": {"model": "openai/gpt-5.5", "block": {"reasoning": {"effort": "high"}, "top_p": 0.9}},
    "vllm": {"model": "served-model", "block": {"top_k": 20, "max_tokens": 32}},
}

_AZURE_ENDPOINT = "https://res.openai.azure.com/"

# provider -> a model id, a block, and the endpoint path of a client with its own API.
DIRECT_CASES = {
    "openai": {
        "model": "gpt-5.5",
        "block": {"reasoning": {"effort": "high"}, "text": {"verbosity": "low"}},
        "path": "/v1/responses",
    },
    "azure": {
        "model": "my-deployment",
        "block": {"max_tokens": 100, "reasoning_effort": "low"},
        "base_url": _AZURE_ENDPOINT,
        "path": "/openai/deployments/my-deployment/chat/completions",
    },
}

_ENV_KEYS = (
    "OPENAI_API_KEY", "OPENAI_BASE_URL", "AZURE_OPENAI_API_KEY", "VLLM_API_KEY",
    "VLLM_BASE_URL", "YIAN_DEFAULT_PROVIDER", "YIAN_DEFAULT_MODEL",
)


@pytest.fixture
def clean_env(tmp_path, monkeypatch):
    monkeypatch.setenv("YIAN_AUTH_FILE", str(tmp_path / "auth.json"))
    monkeypatch.setenv("OPENAI_API_VERSION", "2025-03-01-preview")
    for name in _ENV_KEYS:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


class _Server:
    """Record request bodies; answer Chat Completions and Responses with one reply."""

    def __init__(self, text="ok"):
        self.text = text
        self.requests = []
        self.bodies = []

    def __call__(self, request):
        self.requests.append(request)
        self.bodies.append(json.loads(request.content))
        if request.url.path.endswith("/responses"):
            return httpx.Response(200, json={
                "id": "resp_1", "object": "response", "created_at": 0, "status": "completed",
                "model": "m", "error": None, "incomplete_details": None, "usage": None,
                "output": [{
                    "id": "msg_1", "type": "message", "role": "assistant", "status": "completed",
                    "content": [{"type": "output_text", "text": self.text, "annotations": []}],
                }],
            })
        return httpx.Response(200, json={
            "id": "c", "object": "chat.completion", "created": 0, "model": "m",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": self.text}}],
        })


@pytest.fixture
def server(clean_env):
    """Route every OpenAI-family client built by `model_from_config` to a recording server."""
    recorder = _Server()
    with httpx.Client(transport=httpx.MockTransport(recorder)) as http_client:
        for client_cls in (OpenAIClient, AzureOpenAIClient):
            original = client_cls.get_llm

            def construct(wrapper, original=original):
                wrapper.kwargs.update(http_client=http_client, max_retries=0)
                return original(wrapper)

            clean_env.setattr(client_cls, "get_llm", construct)
        yield recorder


def _config(provider, model, block=None, **fields):
    config = {"provider": provider, "model": model, "api_key": "sk-test", **fields}
    if block is not None:
        config[provider] = block
    return config


@pytest.mark.parametrize("provider", sorted(DIRECT_CASES))
def test_direct_block_reaches_the_body_unchanged(server, provider):
    case = DIRECT_CASES[provider]
    endpoint = {"base_url": case["base_url"]} if "base_url" in case else {}
    config = _config(provider, case["model"], case["block"], **endpoint)
    model_from_config(config).invoke([HumanMessage("hi")])
    assert server.requests[0].url.path == case["path"]
    for key, value in case["block"].items():
        assert server.bodies[0][key] == value


def test_openai_text_block_sits_beside_the_structured_output_format(server):
    class Answer(BaseModel):
        answer: str

    server.text = '{"answer": "6"}'
    chat = model_from_config(_config("openai", "gpt-5.5", {"text": {"verbosity": "low"}}))
    result = chat.with_structured_output(Answer, method="json_schema").invoke([HumanMessage("hi")])
    assert result == Answer(answer="6")
    text = server.bodies[0]["text"]
    assert text["verbosity"] == "low"
    assert text["format"]["type"] == "json_schema"
    assert text["format"]["name"] == "Answer"


def test_azure_block_is_sent_without_renaming(server):
    config = _config("azure", "my-deployment", {"max_tokens": 100}, base_url=_AZURE_ENDPOINT)
    model_from_config(config).invoke([HumanMessage("hi")])
    body = server.bodies[0]
    assert body["max_tokens"] == 100
    assert "max_completion_tokens" not in body


def test_deepseek_block_sends_thinking_and_max_tokens(server):
    block = {"thinking": {"type": "disabled"}, "max_tokens": 50}
    model_from_config(_config("deepseek", "deepseek-chat", block)).invoke([HumanMessage("hi")])
    body = server.bodies[0]
    assert body["thinking"] == {"type": "disabled"}
    assert body["max_tokens"] == 50
    assert "max_completion_tokens" not in body


def test_block_merge_leaves_model_and_call_extra_body_unchanged(server):
    model = model_from_config(_config("deepseek", "deepseek-chat", {"thinking": {"type": "disabled"}}))
    model.extra_body = {"own": 1}
    call_extra = {"call": 2}
    model.invoke([HumanMessage("hi")], extra_body=call_extra)
    model.invoke([HumanMessage("hi")])
    assert model.extra_body == {"own": 1}
    assert call_extra == {"call": 2}
    assert server.bodies[1]["thinking"] == {"type": "disabled"}
    assert "call" not in server.bodies[1]


def test_minimax_block_keeps_reasoning_split(server):
    block = {"thinking": {"type": "adaptive"}}
    model_from_config(_config("minimax", "MiniMax-M2.7", block)).invoke([HumanMessage("hi")])
    body = server.bodies[0]
    assert body["thinking"] == {"type": "adaptive"}
    assert body["reasoning_split"] is True


@pytest.mark.parametrize("provider", sorted(BLOCK_CASES))
def test_openai_compatible_block_reaches_the_body_unchanged(server, provider):
    case = BLOCK_CASES[provider]
    model_from_config(_config(provider, case["model"], case["block"])).invoke([HumanMessage("hi")])
    assert server.requests[0].url.path.endswith("/chat/completions")
    body = server.bodies[0]
    for key, value in case["block"].items():
        assert body[key] == value


def test_azure_responses_deployment_rejects_a_chat_block(server):
    config = _config("azure", "gpt-5-codex", {"max_tokens": 100}, base_url=_AZURE_ENDPOINT)
    with pytest.raises(ValueError) as caught:
        model_from_config(config).invoke([HumanMessage("hi")])
    message = str(caught.value)
    for part in ("azure", "gpt-5-codex", "Chat Completions", "Responses API"):
        assert part in message
    assert server.requests == []


def test_azure_responses_deployment_without_a_block_still_runs(server):
    config = _config("azure", "gpt-5-codex", base_url=_AZURE_ENDPOINT)
    model_from_config(config).invoke([HumanMessage("hi")])
    assert server.requests[0].url.path.endswith("/responses")


@pytest.mark.parametrize(
    ("provider", "model", "path", "expected"),
    [
        ("vllm", "served-model", "/v1/chat/completions",
         {"messages": [{"content": "hi", "role": "user"}], "model": "served-model", "stream": False}),
        ("openai", "gpt-5.5", "/v1/responses",
         {"input": [{"content": "hi", "role": "user", "type": "message"}], "model": "gpt-5.5",
          "stream": False}),
    ],
)
def test_without_a_block_the_body_is_unchanged(server, provider, model, path, expected):
    model_from_config(_config(provider, model)).invoke([HumanMessage("hi")])
    assert server.requests[0].url.path == path
    assert server.bodies[0] == expected


def test_every_supported_provider_has_a_request_body_case():
    covered = {
        *BLOCK_CASES,
        *DIRECT_CASES,
        *test_llm_client_blocks_anthropic.BODY_CASES,
        *test_llm_client_blocks_google.BODY_CASES,
        *test_llm_client_blocks_ollama.BODY_CASES,
        *test_openai_subscription.BODY_CASES,
    }
    assert covered == set(SUPPORTED_LLM_PROVIDERS)
