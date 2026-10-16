from typing import Any, Optional

from langchain_core.language_models import BaseChatModel
from pydantic import SecretStr

from .auth import ApiKeyCredentials, AuthError
from .config import LLMConfig, SUPPORTED_LLM_PROVIDERS
from .base_client import BaseLLMClient

# Providers that use the OpenAI-compatible chat completions API. Dual-region
# providers (qwen/glm/minimax) expose separate international and China
# endpoints — see _PROVIDER_BASE_URL.
# Ollama is intentionally NOT here: it routes to its own native client
# (ollama_client.OllamaClient), which sends the `ollama` block (think, options,
# keep_alive, ...) to /api/chat as written — see that module.
_OPENAI_COMPATIBLE = (
    "openai", "xai", "deepseek",
    "qwen", "qwen-cn",
    "glm", "glm-cn",
    "minimax", "minimax-cn",
    "openrouter", "vllm",
)
_DEDICATED_PROVIDERS = ("ollama", "anthropic", "google", "azure")

# Default base URLs of the OpenAI-compatible providers other than `openai`, whose
# SDK default applies. Kept here, beside the routes, so callers that only need the
# endpoint (the OpenAI client, model listing) do not import an optional extra.
# API-key env vars live in api_key_env.PROVIDER_API_KEY_ENV. Dual-region
# providers (qwen/glm/minimax) keep separate endpoints because international and
# China accounts cannot share credentials.
_PROVIDER_BASE_URL = {
    "xai":        "https://api.x.ai/v1",
    "deepseek":   "https://api.deepseek.com",
    "qwen":       "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    "qwen-cn":    "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "glm":        "https://api.z.ai/api/paas/v4/",
    "glm-cn":     "https://open.bigmodel.cn/api/paas/v4/",
    "minimax":    "https://api.minimax.io/v1",
    "minimax-cn": "https://api.minimaxi.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "vllm":       "http://localhost:8000/v1",
}

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
        f"Install it with:  pip install 'yian[{extra}]'"
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
    clear `pip install 'yian[<extra>]'` hint rather than a bare ImportError.

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
            Client arguments forwarded to the client: ``block`` (`dict`), the provider's
            block shaped like its API request body and sent into every request as
            written, plus connection settings such as ``api_key``, ``timeout``,
            ``max_retries`` and ``callbacks``; the OpenAI-compatible, Azure, Anthropic
            and Google clients also take ``http_client`` / ``http_async_client``, the
            SDK's own HTTP clients (beside a ``base_url``, Anthropic's and Google's
            must not follow redirects); the Ollama client rejects them. The
            OpenAI-compatible, Azure, Anthropic and Google clients take
            ``connection_headers`` (`dict[str, str | SecretStr]`), headers sent with
            every request and kept out of the model's fields; for ``openai``,
            ``OpenAI-Organization`` and ``OpenAI-Project`` there carry the
            organization and project ids. Their names and values follow
            ``llm_config.headers``'s rules, and a model with them never follows a
            redirect, so a host ``http_client`` / ``http_async_client`` beside them
            must not follow redirects; the Ollama client rejects them. Generation
            settings (``temperature``, ``max_tokens``, reasoning effort, …) are no
            longer taken as keyword arguments and are ignored; put them in ``block``.

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


def _connection_headers(
    organization: Optional[str],
    project: Optional[str],
    headers: Optional[dict[str, SecretStr]],
) -> dict[str, SecretStr]:
    """Return the headers a set of connection settings sends.

    Args:
        organization (`str | None`): The OpenAI organization id, if set.
        project (`str | None`): The OpenAI project id, if set.
        headers (`dict[str, SecretStr] | None`): The extra headers, if any.

    Returns:
        `dict[str, SecretStr]`: ``OpenAI-Organization`` and ``OpenAI-Project`` from
            ``organization`` and ``project`` when set, then ``headers``.
    """
    result: dict[str, SecretStr] = {}
    for name, value in (("OpenAI-Organization", organization), ("OpenAI-Project", project)):
        if value is not None:
            result[name] = SecretStr(value)
    result.update(headers or {})
    return result


def _saved_headers(
    credentials: Optional[ApiKeyCredentials], connection: dict[str, Any], provider: str
) -> dict[str, SecretStr]:
    """Return a saved login's header values, checked against its recorded names.

    Args:
        credentials (`ApiKeyCredentials | None`): The pinned login's credentials, or
            `None` when they were not read because the connection records no headers.
        connection (`dict[str, Any]`): The recorded ``store`` connection.
        provider (`str`): The config's provider, named in the error.

    Returns:
        `dict[str, SecretStr]`: The saved headers by name.

    Raises:
        AuthError: The saved header names differ from the recorded ``header_names``.
    """
    if credentials is None:
        return {}
    if {name for name, _ in credentials.headers} != set(connection.get("header_names", ())):
        raise AuthError(
            f"The saved {provider} login's connection headers differ from the ones this "
            f"run recorded; run 'yian auth {provider}' and start a new run"
        )
    return {name: SecretStr(value) for name, value in credentials.headers}


def model_from_config(config) -> BaseChatModel:
    """Resolve an `LLMConfig` OR a plain `dict` into a ready chat model.

    Accepts `dict | LLMConfig` so both the engine's interior carrier (a dict) AND the
    chat-side carrier (an `LLMConfig` instance from `chat/assistant.py`) work end-to-end.
    Public fields are validated through `LLMConfig`; host-owned connection metadata
    is validated separately. Unset provider/model use one saved/environment selection
    snapshot. Recorded connections use their pinned method and identity, so a later
    login or removal fails the model instead of changing accounts: OpenAI resolves
    fresh credentials per request, and every other provider's saved key is resolved
    once per model build. A saved login's header values are resolved once per build.

    A config `api_key` authenticates the request in place of any saved login and
    the provider's key environment variable. Otherwise a saved login's key is sent,
    else the provider's client reads its key environment variable. A config
    `base_url`, which needs that key, replaces the provider's default endpoint. For
    `azure`, `base_url` is the required resource endpoint (a saved login supplies
    it when unset) and `model` the deployment name. A config `organization`,
    `project` and `headers`, which also need that key, are sent with every request
    (the ids as `OpenAI-Organization` / `OpenAI-Project`), and the model then never
    follows a redirect. A saved login's key brings the settings saved with it
    instead: the ids its connection recorded and its saved header values, whose
    names must equal the recorded ones. A key from the environment sends none.

    The block named after the provider is passed to its client as ``block``, and every
    client sends it into the request body as written. The returned model supports
    ordinary tool binding, structured output, and invocation.

    Args:
        config (`LLMConfig` or `dict`):
            The model selection. A `dict` is validated into an `LLMConfig` (raising on
            unknown keys); unset `provider`/`model` inherit matching provider defaults.

    Returns:
        `BaseChatModel`: A langchain chat model, ready for `.bind_tools(...)`.

    Raises:
        ValueError: If the provider is unsupported, or `check_model_config` rejects the
            config (e.g. a block the API it reaches refuses).
        AuthError: If the config's pinned saved login changed or was removed, or its
            saved header names differ from the recorded ones.
        ImportError: If the resolved provider's optional dependency extra is not installed.
    """
    from .connection import check_model_config, connection_credentials, resolve_model_config

    config, connection = check_model_config(resolve_model_config(config))
    provider, model = config.provider, config.model
    if (
        config.api_key is None
        and connection is not None
        and connection["method"] == "subscription"
    ):
        try:
            from .openai_subscription import SubscriptionAccess, create_subscription_model
        except ImportError as exc:
            raise _missing_provider("openai", exc) from exc

        def access_source():
            credentials = connection_credentials(connection)
            return SubscriptionAccess(credentials.access_token, credentials.account_id, credentials.residency)

        return create_subscription_model(
            model, access_source=access_source,
            block=config.block_for("openai"),
        )

    kwargs: dict[str, Any] = {"block": config.block_for(provider)}
    # The saved login's credentials, when its header values are needed.
    saved: Optional[ApiKeyCredentials] = None

    # Parallel nodes fan out at once; a low-TPM deployment rate-limits easily,
    # so lean on the provider SDK's exponential backoff (notably on HTTP 429).
    kwargs.setdefault("max_retries", 6)

    if config.api_key is not None:
        # A host-supplied key wins over the environment and any saved connection.
        kwargs["api_key"] = config.api_key
        if provider == "openai" and connection is not None and connection["method"] == "api_key":
            kwargs["base_url"] = connection["endpoint"]
    elif provider == "openai" and connection is not None:
        # The SDK calls this per request, so re-login cannot redirect an existing
        # model instance to a different account between tool iterations.
        kwargs["api_key"] = lambda: connection_credentials(connection).api_key
        kwargs["base_url"] = connection["endpoint"]
        if connection.get("header_names"):
            # Headers take no callable, so their values are read once here.
            saved = connection_credentials(connection)
    elif connection is not None and connection["source"] == "store":
        # Anthropic and Google SDKs take no key callable, so the saved key is
        # resolved once per model; the credential id still pins the login.
        saved = connection_credentials(connection, provider)
        kwargs["api_key"] = SecretStr(saved.api_key)
    if config.base_url is not None:
        # Validation ties a base_url to the config's key, or makes it the Azure
        # resource endpoint, so it is where this request goes.
        kwargs["base_url"] = config.base_url
    if config.api_key is None and connection is not None and connection["source"] == "store":
        # A saved login's settings travel with its key, never with a host or
        # environment key.
        connection_headers = _connection_headers(
            connection.get("organization"),
            connection.get("project"),
            _saved_headers(saved, connection, provider),
        )
    else:
        connection_headers = _connection_headers(
            config.organization, config.project, config.headers
        )
    if connection_headers:
        kwargs["connection_headers"] = connection_headers
    return create_llm_client(provider=provider, model=model, **kwargs).get_llm()
