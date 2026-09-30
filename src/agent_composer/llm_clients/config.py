"""LLMConfig — model selection for an AGENT node.

Lives with the clients that consume it: the AGENT node and the Compose loader's
`llm_config:` field carry an `LLMConfig` (or the plain dict that normalizes to one),
and `factory.model_from_config` turns API selections into ready chat models.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator


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


class LLMConfig(BaseModel):
    """
    LLM selection for an AGENT node.

    Every field is optional: an unset `provider`/`model` falls back to the package's
    snapshotted provider defaults, and the reasoning knobs apply only when they match the selected
    provider. `extra="forbid"` makes a mistyped `llm_config:` key loud at load time. Validation
    errors never echo the input values, so a rejected config cannot leak `api_key`.

    Attributes:
        provider (`str`, *optional*, defaults to `None`):
            One of `openai`, `xai`, `deepseek`, `qwen`, `qwen-cn`, `glm`, `glm-cn`,
            `minimax`, `minimax-cn`, `openrouter`, `vllm`, `ollama`, `anthropic`,
            `google`, or `azure`. `None` inherits the global default provider.
        model (`str`, *optional*, defaults to `None`):
            Provider-specific model id. `None` inherits that provider's default model.
        anthropic_effort (`str`, *optional*, defaults to `None`):
            Anthropic reasoning effort (`high`/`medium`/`low`); ignored for other providers.
        openai_reasoning_effort (`str`, *optional*, defaults to `None`):
            OpenAI reasoning effort (`high`/`medium`/`low`); ignored for other providers.
        google_thinking_level (`str`, *optional*, defaults to `None`):
            Google thinking level; ignored for other providers.
        temperature (`float`, *optional*, defaults to `None`):
            Sampling temperature; `None` leaves the provider default.
        api_key (`SecretStr`, *optional*, defaults to `None`):
            Provider API key supplied by the host. It takes precedence over the provider's
            key environment variable and saved OpenAI connection. It is masked in `repr`
            and left out of every `model_dump`; read it with `api_key.get_secret_value()`.
            `None` keeps the environment or saved-connection key. An empty key is
            rejected, since some clients would treat it as unset and read the environment.

    Example:
        ```python
        config = LLMConfig(provider="anthropic", model="claude-sonnet-4-5", api_key=key)
        "api_key" in config.model_dump()  # False
        config.api_key.get_secret_value() == key  # True
        ```
    """

    provider: Optional[LLMProvider] = None
    model: Optional[str] = None
    anthropic_effort: Optional[Literal["high", "medium", "low"]] = None
    openai_reasoning_effort: Optional[Literal["high", "medium", "low"]] = None
    google_thinking_level: Optional[str] = None
    temperature: Optional[float] = None
    # Excluded from dumps so the key never reaches a record, and so configs without
    # a key dump exactly as they did before the field existed.
    api_key: Optional[SecretStr] = Field(default=None, exclude=True)
    # `model` is a regular field here; don't warn on the model_ namespace.
    # extra="forbid" makes typo'd llm_config: keys (e.g. `temparature` instead of
    # `temperature`) loud at LOAD time, not silent. hide_input_in_errors keeps
    # config values, including the key, out of validation error messages.
    model_config = ConfigDict(
        protected_namespaces=(), extra="forbid", hide_input_in_errors=True
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
                "Native Codex configuration has been removed. Run 'composer auth openai', "
                "use provider: openai, and remove sandbox/cwd. Composer tools are "
                "granted explicitly and run on the host."
            )
        return value

    @field_validator("api_key")
    @classmethod
    def _reject_empty_key(cls, value: Optional[SecretStr]) -> Optional[SecretStr]:
        """Refuse an empty key, which some clients would replace with an env var."""
        if value is not None and not value.get_secret_value():
            raise ValueError("api_key must not be empty")
        return value


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


def merge_llm_config(specific: dict, base: dict) -> dict:
    """Merge two fill-the-gap layers without mutating either input.

    A non-``None`` value in ``specific`` wins; ``base`` fills every remaining field.
    Explicit ``None`` has the same meaning as omission at every layer, matching the
    optional ``LLMConfig`` fields' fallback behavior. Filtering both operands keeps the
    merge associative, so a cascade can be grouped without changing its result.
    """
    out = {key: value for key, value in base.items() if value is not None}
    out.update(
        (key, value) for key, value in specific.items() if value is not None
    )
    return out
