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

_ROUTED_PROVIDERS = (*_OPENAI_COMPATIBLE, *_DEDICATED_PROVIDERS)
if (
    frozenset(_ROUTED_PROVIDERS) != frozenset(SUPPORTED_LLM_PROVIDERS)
    or len(_ROUTED_PROVIDERS) != len(set(_ROUTED_PROVIDERS))
):
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
        ValueError: If `provider` is unsupported.
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
    Public fields are validated through `LLMConfig`; host-owned connection metadata
    is validated separately. Unset provider/model use one saved/environment selection
    snapshot. Recorded connections use their pinned method and identity, resolving
    fresh credentials per request without changing accounts after a reconnect.

    A config `api_key` authenticates the request in place of the provider's key
    environment variable and any saved OpenAI connection.

    Provider-specific reasoning knobs apply only to the selected provider. The
    returned model supports ordinary tool binding, structured output, and invocation.

    Args:
        config (`LLMConfig` or `dict`):
            The model selection. A `dict` is validated into an `LLMConfig` (raising on
            unknown keys); unset `provider`/`model` inherit matching provider defaults.

    Returns:
        `BaseChatModel`: A langchain chat model, ready for `.bind_tools(...)`.

    Raises:
        ValueError: If the provider is unsupported.
        ImportError: If the resolved provider's optional dependency extra is not installed.
    """
    from .connection import connection_credentials, resolve_model_config, split_model_config

    resolved = resolve_model_config(config)
    config, connection = split_model_config(resolved)
    provider, model = config.provider, config.model
    if (
        config.api_key is None
        and connection is not None
        and connection["method"] == "subscription"
    ):
        if config.temperature is not None:
            raise ValueError("OpenAI subscription connections do not accept temperature; remove that setting")
        try:
            from .openai_subscription import SubscriptionAccess, create_subscription_model
        except ImportError as exc:
            raise _missing_provider("openai", exc) from exc

        def access_source():
            credentials = connection_credentials(connection)
            return SubscriptionAccess(credentials.access_token, credentials.account_id, credentials.residency)

        return create_subscription_model(
            model, access_source=access_source,
            reasoning_effort=config.openai_reasoning_effort,
        )

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

    if config.api_key is not None:
        # A host-supplied key wins over the environment and any saved connection.
        kwargs["api_key"] = config.api_key
        if connection is not None and connection["method"] == "api_key":
            kwargs["base_url"] = connection["endpoint"]
    elif connection is not None:
        # The SDK calls this per request, so re-login cannot redirect an existing
        # model instance to a different account between tool iterations.
        kwargs["api_key"] = lambda: connection_credentials(connection).api_key
        kwargs["base_url"] = connection["endpoint"]
    return create_llm_client(provider=provider, model=model, **kwargs).get_llm()
