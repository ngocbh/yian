"""`LLMConfig.base_url`: a host endpoint bound to the key and provider beside it."""

import pytest
from pydantic import ValidationError

from agent_composer.llm_clients.config import LLMConfig, pop_api_key, pop_base_url

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
    ],
)
def test_https_and_local_http_urls_are_accepted(url):
    assert LLMConfig(provider="vllm", base_url=url, api_key=SENTINEL).base_url == url


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
