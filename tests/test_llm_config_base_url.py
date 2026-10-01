"""`LLMConfig.base_url`: a host endpoint bound to the key and provider beside it."""

import json

import httpx
import pytest
from langchain_core.messages import HumanMessage
from pydantic import ValidationError

from agent_composer.llm_clients.auth import AuthStore, OAuthCredentials
from agent_composer.llm_clients.config import LLMConfig, pop_api_key, pop_base_url
from agent_composer.llm_clients.connection import resolve_model_config, split_model_config
from agent_composer.llm_clients.factory import model_from_config
from agent_composer.llm_clients.openai_client import OpenAIClient
from tests.fixtures.openai_responses import ResponsesServer

SENTINEL = "sk-sentinel-7d1f3b20"
AZURE_V1 = "https://res.openai.azure.com/openai/v1/"


def test_a_set_base_url_dumps_and_an_unset_one_adds_nothing():
    config = LLMConfig(provider="openai", model="gpt-5.4", base_url=AZURE_V1, api_key=SENTINEL)
    assert config.model_dump() == {
        "provider": "openai", "model": "gpt-5.4", "anthropic_effort": None,
        "openai_reasoning_effort": None, "google_thinking_level": None,
        "temperature": None, "base_url": AZURE_V1,
    }
    assert SENTINEL not in config.model_dump_json()
    # Recorded snapshots compare full dumps, so an unset base_url must add nothing.
    for dumped in (
        LLMConfig(provider="openai", model="gpt-5.5").model_dump(exclude_none=False),
        LLMConfig(provider="openai", model="gpt-5.5").model_dump(mode="json"),
    ):
        assert "base_url" not in dumped


@pytest.mark.parametrize(
    "url",
    [
        AZURE_V1,
        "https://gateway.example.com",
        "http://localhost:8000/v1",
        "http://127.0.0.1:11434/v1",
        "http://[::1]:8000/v1",
        "http://[::1]/v1",
        "https://[::1]:8443",
        "https://gateway.example.com:443/v1",
        "https://gateway.example.com:65535/v1",
        "https://xn--gtewy-hra.example.com/v1",
        "https://gateway.example.com/openai/v1/deployments/a-b_c~d%20e",
        pytest.param(
            "https://gateway.example.com/" + "a" * (2048 - len("https://gateway.example.com/")),
            id="2048-chars",
        ),
    ],
)
def test_https_and_local_http_urls_are_accepted(url):
    # The stored string is the validated one, and the client parses the same host.
    assert LLMConfig(provider="vllm", base_url=url, api_key=SENTINEL).base_url == url
    assert httpx.URL(url).host


@pytest.mark.parametrize(
    "url",
    [
        "http://gateway.example.com/v1",
        "http://localhost.example.com/v1",
        "ftp://localhost/v1",
        "res.openai.azure.com/openai/v1/",
        "https://",
        "https://user:pw@res.openai.azure.com/openai/v1/",
        "https://res.openai.azure.com/openai/v1/?api-version=1",
        "https://res.openai.azure.com/openai/v1/#frag",
        "https://[bad/v1",
        "https://res.openai.azure.com/openai/v1/?",
        "https://res.openai.azure.com/openai/v1/#",
        "https://user@res.openai.azure.com/openai/v1/",
        "//res.openai.azure.com/openai/v1/",
        # Whitespace and control characters, which urlsplit strips but httpx keeps.
        " https://res.openai.azure.com/openai/v1/",
        "https://res.openai.azure.com/openai/v1/ ",
        "https://res.openai.azure.com/openai/v1/\n",
        "https://res.openai.azure.com/openai/v1/\r\n",
        "https://res.openai.\tazure.com/openai/v1/",
        "http://local\thost/v1",
        "https://res.openai azure.com/openai/v1/",
        "https://res.openai.azure.com/openai/v1/\x00",
        "https://res.openai.azure.com/openai/v1/\x7f",
        "https://res.openai.azure.com/openai/v1/\u00a0",
        # Ports.
        "https://gateway.example.com:abc/v1",
        "https://gateway.example.com:99999/v1",
        "https://gateway.example.com:65536/v1",
        "https://gateway.example.com:0/v1",
        "https://gateway.example.com:/v1",
        "https://gateway.example.com:+443/v1",
        "https://gateway.example.com:4_43/v1",
        # Hosts: ASCII only, no IDNA normalisation, exact local names.
        "https://gateway\u3002example.com/v1",
        "https://g\u00e4teway.example.com/v1",
        "https://\uff47ateway.example.com/v1",
        "https://%67ateway.example.com/v1",
        "https://gateway.example.com\\@collector.example.com/v1",
        "http://localhost.evil.com/v1",
        "http://127.0.0.1.nip.io/v1",
        "http://127.0.0.1@gateway.example.com/v1",
        "http://[0:0:0:0:0:0:0:1]:8000/v1",
        "https://[::1%25lo0]/v1",
        "https://[gateway.example.com]/v1",
        "http://[::1].example.com/v1",
        # The scheme must be lowercase, as written.
        "HTTPS://res.openai.azure.com/openai/v1/",
        "Https://res.openai.azure.com/openai/v1/",
        "HTTP://localhost:8000/v1",
        # Length.
        pytest.param("https://" + "a" * 100_000 + ".example.com/v1", id="long-host"),
        pytest.param(
            "https://gateway.example.com/" + "a" * (2049 - len("https://gateway.example.com/")),
            id="2049-chars",
        ),
    ],
)
def test_other_urls_are_rejected_without_echoing_them(url):
    with pytest.raises(ValidationError, match="base_url") as caught:
        LLMConfig(provider="openai", base_url=url, api_key=SENTINEL)
    # Pydantic's own help link contains "https://", so only longer values are checked.
    if url != "https://":
        assert url not in str(caught.value)
    assert SENTINEL not in str(caught.value)


@pytest.mark.parametrize(
    "fields",
    [{"provider": "openai"}, {"api_key": SENTINEL}, {}],
    ids=["no-key", "no-provider", "neither"],
)
def test_base_url_needs_a_key_and_provider_in_the_same_config(fields):
    with pytest.raises(ValidationError, match="api_key and llm_config.provider") as caught:
        LLMConfig(base_url=AZURE_V1, **fields)
    assert SENTINEL not in str(caught.value)


@pytest.mark.parametrize("provider", ["azure", "ollama"])
def test_dedicated_clients_without_endpoint_support_reject_base_url(provider):
    with pytest.raises(ValidationError, match=f"provider {provider} does not take"):
        LLMConfig(provider=provider, base_url=AZURE_V1, api_key=SENTINEL)


def test_pop_base_url_takes_the_url_only_with_the_layers_key():
    layer = {"provider": "openai", "model": "gpt-5.4", "base_url": AZURE_V1, "api_key": SENTINEL}
    key = pop_api_key(layer)
    assert pop_base_url(layer, key) == AZURE_V1
    assert layer == {"provider": "openai", "model": "gpt-5.4"}
    assert pop_base_url({"provider": "openai"}, key) is None

    keyless = {"provider": "openai", "base_url": AZURE_V1}
    with pytest.raises(ValueError, match="api_key"):
        pop_base_url(keyless, pop_api_key(keyless))

    unsafe = {"provider": "openai", "base_url": "http://gateway.example.com", "api_key": SENTINEL}
    with pytest.raises(ValueError, match="https") as caught:
        pop_base_url(unsafe, pop_api_key(unsafe))
    assert "gateway.example.com" not in str(caught.value)


# --- clients send the key to the base_url ------------------------------------ #

_ENV = (
    "OPENAI_API_KEY", "OPENAI_BASE_URL", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY",
    "VLLM_API_KEY", "VLLM_BASE_URL", "DEEPSEEK_API_KEY", "OPENROUTER_API_KEY",
    "AGENT_COMPOSER_DEFAULT_PROVIDER", "AGENT_COMPOSER_DEFAULT_MODEL",
)


@pytest.fixture
def clean_env(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_COMPOSER_AUTH_FILE", str(tmp_path / "auth.json"))
    for name in _ENV:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def _route(monkeypatch, http_client):
    """Send every OpenAIClient request through ``http_client``; no network."""
    original = OpenAIClient.get_llm

    def construct(wrapper):
        wrapper.kwargs.update(http_client=http_client, max_retries=0)
        return original(wrapper)

    monkeypatch.setattr(OpenAIClient, "get_llm", construct)


def _chat_completions(requests):
    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={
            "id": "c", "object": "chat.completion", "created": 0, "model": "m",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "ok"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        })

    return httpx.MockTransport(handler)


@pytest.mark.parametrize("provider", ["vllm", "deepseek", "openrouter"])
def test_openai_compatible_clients_send_the_key_to_the_base_url(clean_env, provider):
    # The provider's own endpoint variable and key must not win over the config.
    clean_env.setenv("VLLM_BASE_URL", "http://127.0.0.1:9/v1")
    clean_env.setenv(f"{provider.upper()}_API_KEY", "sk-environment-key")
    requests = []
    with httpx.Client(transport=_chat_completions(requests)) as http_client:
        _route(clean_env, http_client)
        reply = model_from_config({
            "provider": provider, "model": "m",
            "base_url": "https://gateway.example.com/v1", "api_key": SENTINEL,
        }).invoke([HumanMessage("hi")])
    assert reply.content == "ok"
    assert str(requests[0].url) == "https://gateway.example.com/v1/chat/completions"
    assert requests[0].headers["Authorization"] == f"Bearer {SENTINEL}"


def test_openai_sends_the_key_to_an_azure_v1_base_url(clean_env):
    clean_env.setenv("OPENAI_API_KEY", "sk-environment-key")
    clean_env.setenv("OPENAI_BASE_URL", "https://proxy.example.com/v1")
    # A saved subscription must not take over a keyed config either.
    AuthStore().save(
        "openai",
        OAuthCredentials("access-secret", "refresh-secret", 4102444800, "account"),
        "gpt-5.5",
    )
    config = {"provider": "openai", "model": "gpt-5.4", "base_url": AZURE_V1, "api_key": SENTINEL}
    server = ResponsesServer(tool_first=False)
    with httpx.Client(transport=httpx.MockTransport(server)) as http_client:
        _route(clean_env, http_client)
        model_from_config(config).invoke("hi")
    request = server.requests[0]
    assert str(request.url) == "https://res.openai.azure.com/openai/v1/responses"
    assert request.headers["Authorization"] == f"Bearer {SENTINEL}"
    assert json.loads(request.content)["model"] == "gpt-5.4"


def test_an_openai_base_url_pins_a_host_connection_to_that_endpoint(clean_env):
    resolved = resolve_model_config(
        {"provider": "openai", "model": "gpt-5.4", "base_url": AZURE_V1, "api_key": SENTINEL}
    )
    assert resolved["base_url"] == AZURE_V1
    assert resolved["_connection"]["source"] == "host"
    assert resolved["_connection"]["endpoint"] == AZURE_V1
    assert SENTINEL not in json.dumps({k: v for k, v in resolved.items() if k != "api_key"})
    validated, connection = split_model_config(resolved)
    assert validated.base_url == connection["endpoint"] == AZURE_V1


def test_a_base_url_must_match_its_pinned_connection(clean_env):
    resolved = resolve_model_config(
        {"provider": "openai", "model": "gpt-5.4", "base_url": AZURE_V1, "api_key": SENTINEL}
    )
    moved = {**resolved, "_connection": {**resolved["_connection"], "endpoint": "https://api.openai.com/v1"}}
    stored = {**resolved, "_connection": {
        "method": "api_key", "source": "environment", "credential_id": "a" * 64,
        "endpoint": AZURE_V1, "store_path": None,
    }}
    for config in (moved, stored):
        with pytest.raises(ValueError, match="host connection endpoint"):
            split_model_config(config)
        with pytest.raises(ValueError, match="host connection endpoint"):
            model_from_config(config)


@pytest.mark.parametrize(
    "provider, model, attribute",
    [
        ("anthropic", "claude-sonnet-4-5", "anthropic_api_url"),
        ("google", "gemini-2.5-flash", "base_url"),
    ],
)
def test_dedicated_clients_take_the_base_url(clean_env, provider, model, attribute):
    chat = model_from_config({
        "provider": provider, "model": model,
        "base_url": "https://gateway.example.com", "api_key": SENTINEL,
    })
    assert getattr(chat, attribute) == "https://gateway.example.com"
