from typing import Any, Optional

from langchain_core.language_models import BaseChatModel

from .config import LLMConfig, SUPPORTED_LLM_PROVIDERS
from .base_client import BaseLLMClient

# Providers that use the OpenAI-compatible chat completions API. Dual-region
# providers (qwen/glm/minimax) expose separate international and China
# endpoints — see openai_client._PROVIDER_BASE_URL.
# Ollama is intentionally NOT here: it routes to its own native client
# (ollama_client.OllamaClient) so thinking can be disabled — see that module.
_OPENAI_COMPATIBLE = (
    "openai", "xai", "deepseek",
    "qwen", "qwen-cn",
    "glm", "glm-cn",
    "minimax", "minimax-cn",
    "openrouter", "vllm",
)
_DEDICATED_PROVIDERS = ("ollama", "anthropic", "google", "azure")

_ROUTED_PROVIDERS = frozenset((*_OPENAI_COMPATIBLE, *_DEDICATED_PROVIDERS))
if _ROUTED_PROVIDERS != frozenset(SUPPORTED_LLM_PROVIDERS):
    raise RuntimeError(
        "LLM factory routes and LLMConfig provider vocabulary are out of sync"
    )

# provider -> the optional-dependency extra that ships its langchain client. Used
# only to turn a missing-package ImportError into an actionable install hint.
_PROVIDER_EXTRA = {
    "openai": "openai",
    "azure": "openai",
    "anthropic": "anthropic",
    "google": "google",
    "ollama": "ollama",
}


def _missing_provider(provider: str, exc: ImportError) -> ImportError:
    """Re-raise a provider import failure with a `pip install` hint for its extra."""
    extra = _PROVIDER_EXTRA.get(provider, provider)
    return ImportError(
        f"LLM provider {provider!r} needs an optional dependency that isn't installed. "
        f"Install it with:  pip install 'agent-composer[{extra}]'"
    )


def create_llm_client(
    provider: str,
    model: str,
    base_url: Optional[str] = None,
    **kwargs: Any,
) -> BaseLLMClient:
    """Create an LLM client for the specified provider.

    Provider client classes (and their langchain provider packages) are imported
    lazily here, in the matched branch only — so the core package installs without
    any provider extra, and importing a provider whose extra is missing raises a
    clear `pip install 'agent-composer[<extra>]'` hint rather than a bare ImportError.

    Args:
        provider (`str`):
            Provider name; case-insensitive. OpenAI-compatible names (`openai`, `xai`,
            `deepseek`, `qwen`, `qwen-cn`, `glm`, `glm-cn`, `minimax`, `minimax-cn`,
            `openrouter`, and `vllm`) route to the OpenAI client; `ollama`, `anthropic`,
            `google`, and `azure` route to dedicated clients.
        model (`str`):
            The provider-specific model id.
        base_url (`str`, *optional*, defaults to `None`):
            Override the provider's API endpoint; `None` uses the provider default.
        **kwargs:
            Additional provider-specific arguments forwarded to the client.

    Returns:
        `BaseLLMClient`: A configured client whose `get_llm()` yields the chat model.

    Raises:
        ValueError: If `provider` is not supported.
        ImportError: If the provider's optional dependency extra is not installed.
    """
    provider_lower = provider.lower()
    if provider_lower not in SUPPORTED_LLM_PROVIDERS:
        raise ValueError(f"Unsupported LLM provider: {provider}")

    if provider_lower in _OPENAI_COMPATIBLE:
        try:
            from .openai_client import OpenAIClient
        except ImportError as exc:
            raise _missing_provider("openai", exc) from exc
        return OpenAIClient(model, base_url, provider=provider_lower, **kwargs)

    if provider_lower == "ollama":
        try:
            from .ollama_client import OllamaClient
        except ImportError as exc:
            raise _missing_provider("ollama", exc) from exc
        return OllamaClient(model, base_url, **kwargs)

    if provider_lower == "anthropic":
        try:
            from .anthropic_client import AnthropicClient
        except ImportError as exc:
            raise _missing_provider("anthropic", exc) from exc
        return AnthropicClient(model, base_url, **kwargs)

    if provider_lower == "google":
        try:
            from .google_client import GoogleClient
        except ImportError as exc:
            raise _missing_provider("google", exc) from exc
        return GoogleClient(model, base_url, **kwargs)

    if provider_lower == "azure":
        try:
            from .azure_client import AzureOpenAIClient
        except ImportError as exc:
            raise _missing_provider("azure", exc) from exc
        return AzureOpenAIClient(model, base_url, **kwargs)

    raise ValueError(f"Unsupported LLM provider: {provider}")


def model_from_config(config) -> BaseChatModel:
    """Resolve an `LLMConfig` OR a plain `dict` into a ready chat model.

    Accepts `dict | LLMConfig` so both the engine's interior carrier (a dict) AND the
    chat-side carrier (an `LLMConfig` instance from `chat/assistant.py`) work end-to-end.
    Dicts are normalized to `LLMConfig` at the top (this is also where `extra="forbid"`
    triggers if a key is unknown).

    Absorbs the former model-builder seam: unset provider/model fall back to the
    package's env-based defaults (`_settings.default_llm_*`); provider-specific reasoning
    knobs apply only when they match the selected provider; transient errors get
    exponential-backoff retries. The returned model is ready for `.bind_tools(...)`. Shared
    by every LLM-backed engine seam (today the AGENT `llm_client`).

    Args:
        config (`LLMConfig` or `dict`):
            The model selection. A `dict` is validated into an `LLMConfig` (raising on
            unknown keys); unset `provider`/`model` inherit the env-based defaults.

    Returns:
        `BaseChatModel`: A langchain chat model, ready for `.bind_tools(...)`.

    Raises:
        ValueError: If the resolved provider is not supported.
        ImportError: If the resolved provider's optional dependency extra is not installed.
    """
    if isinstance(config, dict):
        config = LLMConfig(**config)
    from agent_composer._settings import default_llm_model, default_llm_provider

    provider = (config.provider or default_llm_provider()).lower()
    model = config.model or default_llm_model()

    kwargs: dict[str, Any] = {}
    if provider == "anthropic" and config.anthropic_effort:
        kwargs["effort"] = config.anthropic_effort
    elif provider == "openai" and config.openai_reasoning_effort:
        kwargs["reasoning_effort"] = config.openai_reasoning_effort
    elif provider == "google" and config.google_thinking_level:
        kwargs["thinking_level"] = config.google_thinking_level
    if config.temperature is not None:
        kwargs["temperature"] = config.temperature

    # Parallel nodes fan out at once; a low-TPM deployment rate-limits easily,
    # so lean on the provider SDK's exponential backoff (notably on HTTP 429).
    kwargs.setdefault("max_retries", 6)

    return create_llm_client(provider=provider, model=model, **kwargs).get_llm()
