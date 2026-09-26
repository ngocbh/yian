"""Coverage for the llm_clients provider features ported from TradingAgents.

Construction-only (no network): builds clients with dummy keys and asserts the
new capability table, Anthropic effort gating, temperature passthrough (which was
silently dropped for anthropic/google/azure), and the dual-region providers.
"""

import pytest

from agent_composer.llm_clients import create_llm_client
from agent_composer.llm_clients.anthropic_client import _supports_effort
from agent_composer.llm_clients.api_key_env import get_api_key_env
from agent_composer.llm_clients.capabilities import get_capabilities
from agent_composer.llm_clients.config import LLMConfig, SUPPORTED_LLM_PROVIDERS
from agent_composer.llm_clients.factory import (
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
    assert caps.requires_reasoning_content_roundtrip is True


def test_capabilities_minimax_requires_reasoning_split():
    assert get_capabilities("MiniMax-M2.7").requires_reasoning_split is True


def test_capabilities_default_model_allows_tool_choice():
    caps = get_capabilities("gpt-4o")
    assert caps.supports_tool_choice is True
    assert caps.preferred_structured_method == "function_calling"


def test_capabilities_forward_compat_pattern():
    # an unseen deepseek-v5 variant inherits the thinking-mode quirks
    assert get_capabilities("deepseek-v5-pro").supports_tool_choice is False


# --- anthropic effort gating ------------------------------------------------ #


@pytest.mark.parametrize("model", ["claude-opus-4-5", "claude-sonnet-4-5", "claude-opus-9-9"])
def test_supports_effort_true_for_opus_sonnet(model):
    assert _supports_effort(model) is True


@pytest.mark.parametrize("model", ["claude-haiku-4-5", "claude-3-5-haiku", "gpt-4o"])
def test_supports_effort_false_for_haiku_and_others(model):
    assert _supports_effort(model) is False


def test_anthropic_get_llm_drops_effort_for_haiku_keeps_for_opus():
    opus = create_llm_client("anthropic", "claude-opus-4-5", api_key="x", effort="high").get_llm()
    haiku = create_llm_client("anthropic", "claude-haiku-4-5", api_key="x", effort="high").get_llm()
    assert opus.effort == "high"
    assert getattr(haiku, "effort", None) is None


# --- temperature passthrough (was silently dropped) ------------------------- #


def test_anthropic_temperature_passthrough():
    m = create_llm_client("anthropic", "claude-opus-4-5", api_key="x", temperature=0.3).get_llm()
    assert m.temperature == 0.3


def test_google_temperature_passthrough():
    m = create_llm_client("google", "gemini-2.5-flash", api_key="x", temperature=0.4).get_llm()
    assert m.temperature == 0.4


def test_azure_temperature_passthrough_and_endpoint_base_stripping(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "x")
    monkeypatch.setenv("OPENAI_API_VERSION", "2024-02-01")
    # the OpenAI-v1 suffix must be stripped back to the base resource URL
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://res.openai.azure.com/openai/v1")
    m = create_llm_client("azure", "gpt-4o", temperature=0.1).get_llm()
    assert m.temperature == 0.1
    assert str(m.azure_endpoint) == "https://res.openai.azure.com"


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
    from agent_composer.llm_clients.openai_client import OpenAIClient

    client = create_llm_client(provider, "some-model")
    assert isinstance(client, OpenAIClient)
    assert client.provider == provider


def test_api_key_env_maps_dual_region_keys():
    assert get_api_key_env("qwen-cn") == "DASHSCOPE_CN_API_KEY"
    assert get_api_key_env("glm-cn") == "ZHIPU_CN_API_KEY"
    assert get_api_key_env("ollama") is None  # local runtime, no key


# --- vLLM (OpenAI-compatible, optional key) --------------------------------- #


def test_factory_resolves_vllm_as_openai_compatible():
    from agent_composer.llm_clients.openai_client import OpenAIClient

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
    from agent_composer.llm_clients.validators import validate_model

    assert validate_model("vllm", "any/arbitrary-served-model") is True


# --- Ollama (native client, thinking disabled) ------------------------------ #


def test_factory_routes_ollama_to_native_client():
    from agent_composer.llm_clients.ollama_client import OllamaClient

    client = create_llm_client("ollama", "qwen3.5:35b")
    assert isinstance(client, OllamaClient)


def test_ollama_get_llm_disables_thinking_and_strips_v1(monkeypatch):
    # an OpenAI-style base URL (with /v1) must be normalized to the native one
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://gpu-box:11434/v1")
    m = create_llm_client("ollama", "qwen3.5:35b", temperature=0.0).get_llm()
    assert m.reasoning is False  # thinking off -> answer lands in content
    assert str(m.base_url) == "http://gpu-box:11434"
    assert m.temperature == 0.0


def test_ollama_drops_unsupported_kwargs():
    # model_from_config sets max_retries, which ChatOllama does not accept;
    # the client must filter it rather than crash on construction.
    m = create_llm_client("ollama", "qwen3.5:35b", max_retries=6).get_llm()
    assert m.model == "qwen3.5:35b"


def test_openai_responses_preserve_reasoning_for_next_tool_turn():
    import httpx
    from langchain_core.messages import HumanMessage, ToolMessage
    from agent_composer.nodes.agent.modes.utils import text_of
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
