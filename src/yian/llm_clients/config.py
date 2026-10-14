"""LLMConfig — model selection for an AGENT node.

Lives with the clients that consume it: the AGENT node and the Compose loader's
`llm_config:` field carry an `LLMConfig` (or the plain dict that normalizes to one),
and `factory.model_from_config` turns API selections into ready chat models.
"""

from __future__ import annotations

from collections.abc import Mapping
import ipaddress
import re
from typing import Any, Literal, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    field_validator,
    model_serializer,
    model_validator,
)

from yian.llm_clients.provider_blocks import (
    PROVIDER_BLOCKS,
    AnthropicMessagesBlock,
    AzureChatBlock,
    DeepSeekChatBlock,
    GeminiGenerateContentBlock,
    GLMChatBlock,
    GLMCNChatBlock,
    MiniMaxChatBlock,
    MiniMaxCNChatBlock,
    OllamaChatBlock,
    OpenAIResponsesBlock,
    OpenRouterChatBlock,
    QwenChatBlock,
    QwenCNChatBlock,
    VLLMChatBlock,
    XAIChatBlock,
)


# One provider vocabulary shared by the validated config carrier and the factory that
# consumes it. Keep routing subsets in factory.py; its import-time census verifies that
# every name here has exactly one route and that no factory route bypasses validation.
SUPPORTED_LLM_PROVIDERS: tuple[str, ...] = (
    "openai",
    "xai",
    "deepseek",
    "qwen",
    "qwen-cn",
    "glm",
    "glm-cn",
    "minimax",
    "minimax-cn",
    "openrouter",
    "vllm",
    "ollama",
    "anthropic",
    "google",
    "azure",
)
# `typing.Literal` flattens a tuple argument into its member values. Deriving the type
# from the tuple keeps Pydantic's enum validation/schema and the factory gate in lockstep.
LLMProvider = Literal[SUPPORTED_LLM_PROVIDERS]  # type: ignore[valid-type]

# Dedicated clients that cannot take a host base_url: the native Ollama client sends
# no key.
_NO_BASE_URL_PROVIDERS = {
    "ollama": (
        "provider ollama does not take llm_config.base_url; it sends no api_key, so "
        "set OLLAMA_BASE_URL instead"
    ),
}
AZURE_NEEDS_BASE_URL = (
    "provider azure needs llm_config.base_url, the Azure OpenAI resource endpoint "
    "(https://<resource>.openai.azure.com/); set llm_config.base_url beside "
    "provider: azure. AZURE_OPENAI_ENDPOINT is not read"
)
_LOCAL_HOSTS = ("localhost", "127.0.0.1", "::1")
_MAX_BASE_URL_LENGTH = 2048
# RFC 3986 characters minus "?" and "#", since a query or fragment is refused anyway.
# Everything else (whitespace, controls, non-ASCII) is something urlsplit or httpx
# would strip, normalise, or encode, so the string sent would differ from the one
# checked here.
_URL_CHARACTERS = re.compile(r"[A-Za-z0-9\-._~:/\[\]@!$&'()*+,;=%]*")
# A DNS name or bracketed IPv6 literal, an optional all-digit port, then a path. The
# host part has no "@", so userinfo never matches.
_URL_SHAPE = re.compile(
    r"(?P<scheme>https?)://(?P<host>[A-Za-z0-9.-]+|\[[0-9A-Fa-f:.]+\])"
    r"(?::(?P<port>[0-9]{1,5}))?(?P<path>/.*)?"
)


class LLMConfig(BaseModel):
    """
    LLM selection for an AGENT node.

    Every field is optional: an unset `provider`/`model` falls back to the package's
    snapshotted provider defaults, and only the block named by the selected provider is
    sent. `extra="forbid"` makes a mistyped `llm_config:` key loud at load time, and a
    retired key (`temperature`, `anthropic_effort`, ...) fails with its new form (see
    `retired_key_error`). Validation errors never echo the input values, so a rejected
    config cannot leak `api_key`.

    Attributes:
        provider (`str`, *optional*, defaults to `None`):
            One of `openai`, `xai`, `deepseek`, `qwen`, `qwen-cn`, `glm`, `glm-cn`,
            `minimax`, `minimax-cn`, `openrouter`, `vllm`, `ollama`, `anthropic`,
            `google`, or `azure`. `None` inherits the global default provider.
        model (`str`, *optional*, defaults to `None`):
            Provider-specific model id. `None` inherits that provider's default model.
        api_key (`SecretStr`, *optional*, defaults to `None`):
            Provider API key supplied by the host. It takes precedence over the provider's
            key environment variable and saved OpenAI connection. It is masked in `repr`
            and left out of every `model_dump`; read it with `api_key.get_secret_value()`.
            `None` keeps the environment or saved-connection key. An empty key is
            rejected, since some clients would treat it as unset and read the environment.
        base_url (`str`, *optional*, defaults to `None`):
            API endpoint that replaces the provider's default, such as an Azure OpenAI
            resource's `https://<resource>.openai.azure.com/openai/v1/` with
            `provider="openai"`. It decides where `api_key` is sent, so it needs
            `api_key` and `provider` in the same config. It must be a lowercase
            `https://` URL without credentials, query, or fragment; `http://` is
            allowed only for exactly `localhost`, `127.0.0.1`, and `[::1]`. It is
            checked and stored as written, so whitespace, control characters,
            non-ASCII characters (use a host's `xn--` form), a bad port, or more than
            2048 characters are rejected rather than normalised (see
            `validate_base_url`). The `ollama` provider rejects it. For
            `provider="azure"` it is the Azure OpenAI resource endpoint, such as
            `https://<resource>.openai.azure.com/`, and is required when the model
            is called; it needs no `api_key`, since the key may come from
            `AZURE_OPENAI_API_KEY`, and a flow file may set it beside
            `provider: azure`. Unlike the key, it appears in `model_dump` when set.
        <provider> (`_Block`, *optional*, defaults to `None`):
            One block per provider in `SUPPORTED_LLM_PROVIDERS` (`openai`, `qwen-cn`,
            `anthropic`, ...), named after it: request-body fields for the API that
            provider's calls reach, checked by that provider's class in `PROVIDER_BLOCKS`
            (e.g. `AnthropicMessagesBlock` for `anthropic`). Every block is checked, but
            only the one named by the resolved `provider` is used. The dash names are
            written with their dash; the Python attributes are `qwen_cn`, `glm_cn`, and
            `minimax_cn`. An unset block is left out of every `model_dump`, and a set one
            dumps without its unset fields (see `block_for`).

    Example:
        ```python
        config = LLMConfig(provider="anthropic", model="claude-sonnet-4-5", api_key=key)
        "api_key" in config.model_dump()  # False
        config.api_key.get_secret_value() == key  # True
        LLMConfig(
            provider="openai",
            model="gpt-5.4",
            base_url="https://<resource>.openai.azure.com/openai/v1/",
            api_key=key,
        )
        LLMConfig(**{"provider": "qwen-cn", "qwen-cn": {"enable_thinking": True}})
        ```
    """

    provider: Optional[LLMProvider] = None
    model: Optional[str] = None
    # Excluded from dumps so the key never reaches a record, and so configs without
    # a key dump exactly as they did before the field existed.
    api_key: Optional[SecretStr] = Field(default=None, exclude=True)
    base_url: Optional[str] = None
    openai: Optional[OpenAIResponsesBlock] = None
    xai: Optional[XAIChatBlock] = None
    deepseek: Optional[DeepSeekChatBlock] = None
    qwen: Optional[QwenChatBlock] = None
    qwen_cn: Optional[QwenCNChatBlock] = Field(default=None, alias="qwen-cn")
    glm: Optional[GLMChatBlock] = None
    glm_cn: Optional[GLMCNChatBlock] = Field(default=None, alias="glm-cn")
    minimax: Optional[MiniMaxChatBlock] = None
    minimax_cn: Optional[MiniMaxCNChatBlock] = Field(default=None, alias="minimax-cn")
    openrouter: Optional[OpenRouterChatBlock] = None
    vllm: Optional[VLLMChatBlock] = None
    ollama: Optional[OllamaChatBlock] = None
    anthropic: Optional[AnthropicMessagesBlock] = None
    google: Optional[GeminiGenerateContentBlock] = None
    azure: Optional[AzureChatBlock] = None
    # `model` is a regular field here; don't warn on the model_ namespace.
    # extra="forbid" makes typo'd llm_config: keys (e.g. `porvider` instead of
    # `provider`) loud at LOAD time, not silent. hide_input_in_errors keeps
    # config values, including the key, out of validation error messages. Keys are
    # read and dumped as authored, so the dash blocks are `qwen-cn`, not `qwen_cn`.
    model_config = ConfigDict(
        protected_namespaces=(),
        extra="forbid",
        hide_input_in_errors=True,
        allow_inf_nan=False,
        validate_by_alias=True,
        validate_by_name=False,
        serialize_by_alias=True,
    )

    @model_validator(mode="before")
    @classmethod
    def _reject_retired_options(cls, value):
        """Point native configuration at the unified OpenAI connection."""
        if isinstance(value, dict) and (
            str(value.get("provider", "")).lower() == "codex"
            or "sandbox" in value or "cwd" in value
        ):
            raise ValueError(
                "Native Codex configuration has been removed. Run 'yian auth openai', "
                "use provider: openai, and remove sandbox/cwd. Yian tools are "
                "granted explicitly and run on the host."
            )
        return value

    @model_validator(mode="before")
    @classmethod
    def _reject_retired_keys(cls, value):
        """Point a retired reasoning or temperature key at its provider-block form."""
        message = retired_key_error(value)
        if message is not None:
            raise ValueError(message)
        return value

    @model_validator(mode="before")
    @classmethod
    def _reject_python_block_names(cls, value):
        """Point a Python spelling of a dash block, such as ``qwen_cn``, at ``qwen-cn``."""
        if isinstance(value, Mapping):
            for key, name in _BLOCK_ATTRIBUTES.items():
                if key != name and name in value:
                    raise ValueError(f"`{name}` is not an llm_config key; write `{key}`")
        return value

    @field_validator("api_key")
    @classmethod
    def _reject_empty_key(cls, value: Optional[SecretStr]) -> Optional[SecretStr]:
        """Refuse an empty key, which some clients would replace with an env var."""
        if value is not None and not value.get_secret_value():
            raise ValueError("api_key must not be empty")
        return value

    @field_validator("base_url")
    @classmethod
    def _require_safe_url(cls, value: Optional[str]) -> Optional[str]:
        """Accept an HTTPS endpoint, or plain HTTP only on the local machine."""
        return value if value is None else validate_base_url(value)

    @model_validator(mode="after")
    def _bind_base_url_to_key(self) -> "LLMConfig":
        """Require a base_url to travel with the key and provider it serves.

        An Azure base_url is the resource endpoint and may come without a key.
        """
        if self.base_url is None or self.provider == "azure":
            return self
        if self.api_key is None or self.provider is None:
            raise ValueError(
                "llm_config.base_url needs llm_config.api_key and llm_config.provider in "
                "the same config, since it decides where that key is sent"
            )
        if self.provider in _NO_BASE_URL_PROVIDERS:
            raise ValueError(_NO_BASE_URL_PROVIDERS[self.provider])
        return self

    @model_serializer(mode="wrap")
    def _omit_unset_fields(self, handler):
        """Dump an unset base_url or block as absent, so existing recorded dumps stay identical."""
        data = handler(self)
        if not isinstance(data, dict):
            return data
        if self.base_url is None:
            data.pop("base_url", None)
        for key, name in _BLOCK_ATTRIBUTES.items():
            if getattr(self, name) is None:
                data.pop(key, None)
                data.pop(name, None)
        return data

    def block_for(self, provider: str) -> dict[str, Any]:
        """Return the block named ``provider`` as a plain dict.

        Args:
            provider (`str`):
                A provider name as written in ``llm_config``, e.g. ``"qwen-cn"``.

        Returns:
            `dict[str, Any]`:
                A fresh dump of the block without unset fields; ``{}`` when the block is
                unset.

        Raises:
            `KeyError`:
                Raised when ``provider`` has no block.

        Example:
            ```python
            LLMConfig(**{"glm-cn": {"do_sample": False}}).block_for("glm-cn")
            # {"do_sample": False}
            ```
        """
        block = getattr(self, _BLOCK_ATTRIBUTES[provider])
        return {} if block is None else block.model_dump()


# Block name as written -> `LLMConfig` attribute, e.g. "qwen-cn" -> "qwen_cn".
_BLOCK_ATTRIBUTES: dict[str, str] = {
    provider: provider.replace("-", "_") for provider in PROVIDER_BLOCKS
}

# Every top-level `llm_config` key, spelled as authored.
LLM_CONFIG_KEYS: frozenset[str] = frozenset(
    field.alias or name for name, field in LLMConfig.model_fields.items()
)


# Keys removed from `llm_config` -> the message showing their provider-block form.
# Recorded runs may still carry them as null (see `RETIRED_LLM_CONFIG_KEYS`).
_RETIRED_KEY_MESSAGES: dict[str, str] = {
    "anthropic_effort": (
        "`anthropic_effort` was removed; write `anthropic: {output_config: {effort: <level>}}`"
    ),
    "openai_reasoning_effort": (
        "`openai_reasoning_effort` was removed; write `openai: {reasoning: {effort: <level>}}`"
    ),
    "google_thinking_level": (
        "`google_thinking_level` was removed; write "
        "`google: {generationConfig: {thinkingConfig: {thinkingLevel: <LEVEL>}}}`"
    ),
}
# Where `temperature` lives inside each block whose API does not take it top-level.
_TEMPERATURE_FORMS = {
    "google": "`google: {generationConfig: {temperature: <value>}}`",
    "ollama": "`ollama: {options: {temperature: <value>}}`",
}

RETIRED_LLM_CONFIG_KEYS: frozenset[str] = frozenset({*_RETIRED_KEY_MESSAGES, "temperature"})


def retired_key_error(config: Any) -> Optional[str]:
    """Return the migration message for the first retired key in an ``llm_config`` layer.

    The retired keys are ``anthropic_effort``, ``openai_reasoning_effort``,
    ``google_thinking_level`` and top-level ``temperature``; each now lives in a
    provider block. A key counts whatever its value, and the message never echoes it.

    Args:
        config (`Any`): One ``llm_config`` layer; a non-mapping has no retired key.

    Returns:
        `str | None`: A message showing the provider-block form of the key, naming the
            layer's ``provider`` when it is a supported one, or `None`.

    Example:
        ```python
        retired_key_error({"provider": "ollama", "temperature": 0.2})
        # "`temperature` moved into the provider block: write
        #  `ollama: {options: {temperature: <value>}}`"
        ```
    """
    if not isinstance(config, Mapping):
        return None
    for key, message in _RETIRED_KEY_MESSAGES.items():
        if key in config:
            return message
    if "temperature" not in config:
        return None
    provider = config.get("provider")
    if isinstance(provider, str) and provider in SUPPORTED_LLM_PROVIDERS:
        form = _TEMPERATURE_FORMS.get(provider, f"`{provider}: {{temperature: <value>}}`")
        return f"`temperature` moved into the provider block: write {form}"
    return (
        "`temperature` moved into the provider block: write "
        "`<provider>: {temperature: <value>}` (google: `generationConfig.temperature`; "
        "ollama: `options.temperature`)"
    )


def validate_base_url(value: Any) -> str:
    """Return ``value`` unchanged if it is an allowed ``llm_config.base_url``.

    The check runs on the exact string the client will use: it must be a lowercase
    ``https://`` URL (``http://`` only for exactly ``localhost``, ``127.0.0.1``, or
    ``[::1]``) of a host, an optional port from 1 to 65535, and a path, at most 2048
    characters long. Credentials, a query, a fragment, whitespace, control characters,
    and non-ASCII characters are refused rather than normalised; write an
    internationalised host in its ``xn--`` form. Errors never echo the value.

    Args:
        value (`Any`): The candidate URL.

    Returns:
        `str`: ``value`` itself.

    Raises:
        `ValueError`: ``value`` is not an allowed URL.
    """
    if not isinstance(value, str):
        raise ValueError("base_url must be a string")
    if len(value) > _MAX_BASE_URL_LENGTH:
        raise ValueError(f"base_url must be at most {_MAX_BASE_URL_LENGTH} characters")
    if not _URL_CHARACTERS.fullmatch(value):
        raise ValueError(
            "base_url may contain only ASCII URL characters, with no whitespace or "
            "control characters; write an internationalised host in its xn-- form"
        )
    shape = _URL_SHAPE.fullmatch(value)
    host = shape and shape["host"].lower()
    if host and host.startswith("["):
        host = host[1:-1]
        try:
            ipaddress.IPv6Address(host)
        except ValueError:
            host = None
    if not host:
        raise ValueError(
            "base_url must be a lowercase https:// or http:// URL of a host, an "
            "optional port, and a path, without credentials, query, or fragment"
        )
    if shape["port"] is not None and not 1 <= int(shape["port"]) <= 65535:
        raise ValueError("base_url port must be a number from 1 to 65535")
    if shape["scheme"] != "https" and host not in _LOCAL_HOSTS:
        raise ValueError(
            "base_url must use https; http is allowed only for localhost, "
            "127.0.0.1, and ::1"
        )
    return value


# Fields only the host's run layer may set. Authored layers reach the recorded
# effective map, and a flow-level base_url could send the host key elsewhere.
_HOST_ONLY_FIELDS = {
    "api_key": (
        "llm_config.api_key is a host credential; pass it in the run's llm_config, "
        "not in a flow file"
    ),
    "base_url": (
        "llm_config.base_url decides where the host's api_key is sent; pass it with "
        "that key in the run's llm_config, not in a flow file. A flow file sets "
        "base_url only beside provider: azure, as the Azure resource endpoint"
    ),
}


def host_only_field_error(config: Any) -> Optional[str]:
    """Return why an authored ``llm_config`` layer may not be used, or `None`.

    A ``base_url`` beside ``provider: azure`` is allowed: it is the Azure resource
    endpoint, and the Azure key comes from the environment, which flow code can
    already read, while a host ``api_key`` for Azure must bring its own ``base_url``
    (see `pop_host_credentials`).

    Args:
        config (`Any`): One authored layer; a non-mapping has no host-only field.

    Returns:
        `str | None`: The message for the first host-only field the layer sets.
    """
    if not isinstance(config, Mapping):
        return None
    for field, message in _HOST_ONLY_FIELDS.items():
        if field == "base_url" and config.get("provider") == "azure":
            continue
        if field in config:
            return message
    return None


def pop_api_key(config: dict[str, Any]) -> Optional[SecretStr]:
    """Pop a host ``api_key`` from one config layer and return it as a `SecretStr`.

    The layer must name its ``provider``, so the key authenticates only that
    provider. Errors never echo the key.

    Args:
        config (`dict[str, Any]`): A mutable ``llm_config`` layer; ``api_key`` is removed.

    Returns:
        `SecretStr | None`: The validated key, or `None` when the layer has none.

    Raises:
        `ValueError`: The key is empty or not a string, ``provider`` is missing,
            or ``provider`` is unsupported.
    """
    raw = config.pop("api_key", None)
    if raw is None:
        return None
    if not config.get("provider"):
        raise ValueError(
            "llm_config.api_key needs llm_config.provider, so the key goes only to "
            "that provider"
        )
    return LLMConfig(provider=config["provider"], api_key=raw).api_key


def pop_host_credentials(
    config: dict[str, Any],
) -> tuple[Optional[SecretStr], Optional[str]]:
    """Pop a host ``api_key`` and the ``base_url`` it goes to from one config layer.

    Popping both from the same layer is what binds the URL to the key: a base_url
    decides where the key is sent, so it is accepted only beside that key and the
    layer's ``provider``. For ``provider: azure`` without a key, ``base_url`` is the
    resource endpoint the environment key goes to; it stays in the layer as an
    ordinary setting. An Azure key needs a ``base_url`` beside it, so it is never sent
    to an endpoint another layer names. Errors never echo the URL or the key.

    Args:
        config (`dict[str, Any]`): A mutable ``llm_config`` layer; ``api_key`` and
            ``base_url`` are removed.

    Returns:
        `tuple[SecretStr | None, str | None]`: The validated key and URL; either is
            `None` when the layer has none.

    Raises:
        `ValueError`: The key is invalid or lacks ``provider`` (see `pop_api_key`),
            the layer has a non-Azure ``base_url`` but no ``api_key``, an Azure key
            has no ``base_url``, the URL is not allowed (see `validate_base_url`),
            or the provider does not take a base_url.
    """
    api_key = pop_api_key(config)
    azure = config.get("provider") == "azure"
    if api_key is None and azure:
        if config.get("base_url") is not None:
            validate_base_url(config["base_url"])
        return None, None
    raw = config.pop("base_url", None)
    if raw is None:
        if azure:
            raise ValueError(
                "llm_config.api_key for provider azure needs llm_config.base_url in "
                "the same config, since it decides where that key is sent"
            )
        return api_key, None
    return api_key, LLMConfig(provider=config.get("provider"), base_url=raw, api_key=api_key).base_url


def merge_llm_config(specific: dict, base: dict) -> dict:
    """Merge two fill-the-gap layers without mutating either input.

    A non-``None`` value in ``specific`` wins; ``base`` fills every remaining field.
    Explicit ``None`` has the same meaning as omission at every layer, matching the
    optional ``LLMConfig`` fields' fallback behavior. A provider block (a key of
    `PROVIDER_BLOCKS`, e.g. ``openai``) merges per block field when both layers set it
    to a mapping: ``specific``'s non-``None`` fields win, ``base`` fills the rest, and a
    value inside a field (e.g. ``thinking: {...}``) is replaced whole. A block that is
    not a mapping is kept over the other layer's block, whichever layer sets it (the
    ``specific`` one when both are), so it reaches validation instead of being hidden. Every other key, including
    ``_connection``, is replaced whole. Filtering ``None`` keeps the merge associative,
    so a cascade can be grouped without changing its result.

    Args:
        specific (`dict`):
            The higher-precedence layer, e.g. a node's own ``llm_config``.
        base (`dict`):
            The layer that fills its gaps, e.g. the enclosing flow's.

    Returns:
        `dict`:
            A new merged layer. Block mappings are new dicts; other values are shared
            with the inputs.

    Example:
        ```python
        merge_llm_config(
            {"anthropic": {"temperature": 0.2}},
            {"provider": "anthropic", "anthropic": {"max_tokens": 8000, "temperature": 1}},
        )
        # {"provider": "anthropic", "anthropic": {"max_tokens": 8000, "temperature": 0.2}}
        ```
    """
    out = {key: _unset_dropped(key, value) for key, value in base.items() if value is not None}
    for key, value in specific.items():
        if value is None:
            continue
        current = out.get(key)
        if key in PROVIDER_BLOCKS and current is not None:
            if not isinstance(value, Mapping):
                out[key] = value
            elif isinstance(current, Mapping):
                out[key] = {**current, **_unset_dropped(key, value)}
            # Otherwise the base block is not a mapping and stays for validation to refuse.
        else:
            out[key] = _unset_dropped(key, value)
    return out


def _unset_dropped(key: str, value: Any) -> Any:
    """Return a provider block mapping without its ``None`` fields; other values unchanged."""
    if key in PROVIDER_BLOCKS and isinstance(value, Mapping):
        return {field: item for field, item in value.items() if item is not None}
    return value
