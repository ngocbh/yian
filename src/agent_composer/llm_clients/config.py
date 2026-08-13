"""LLMConfig — per-AGENT model selection (the carrier `model_from_config` resolves).

Lives with the clients that consume it: the AGENT node and the Compose loader's
`llm_config:` field carry an `LLMConfig` (or the plain dict that normalizes to one),
and `factory.model_from_config` turns it into a ready chat model. Was in the
agent_composer `common/` grab-bag before this cleanup; moved here because it IS the
clients' input contract.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict


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
    Per-AGENT LLM selection — the carrier `model_from_config` resolves into a chat model.

    Every field is optional: an unset `provider`/`model` falls back to the package's
    env-based defaults, and the reasoning knobs apply only when they match the selected
    provider. `extra="forbid"` makes a mistyped `llm_config:` key loud at load time.

    Attributes:
        provider (`str`, *optional*, defaults to `None`):
            One of `openai`, `xai`, `deepseek`, `qwen`, `qwen-cn`, `glm`, `glm-cn`,
            `minimax`, `minimax-cn`, `openrouter`, `vllm`, `ollama`, `anthropic`,
            `google`, or `azure`. `None` inherits the global default provider.
        model (`str`, *optional*, defaults to `None`):
            Provider-specific model id. `None` inherits the global default model.
        anthropic_effort (`str`, *optional*, defaults to `None`):
            Anthropic reasoning effort (`high`/`medium`/`low`); ignored for other providers.
        openai_reasoning_effort (`str`, *optional*, defaults to `None`):
            OpenAI reasoning effort (`high`/`medium`/`low`); ignored for other providers.
        google_thinking_level (`str`, *optional*, defaults to `None`):
            Google thinking level; ignored for other providers.
        temperature (`float`, *optional*, defaults to `None`):
            Sampling temperature; `None` leaves the provider default.
    """

    provider: Optional[LLMProvider] = None
    model: Optional[str] = None
    anthropic_effort: Optional[Literal["high", "medium", "low"]] = None
    openai_reasoning_effort: Optional[Literal["high", "medium", "low"]] = None
    google_thinking_level: Optional[str] = None
    temperature: Optional[float] = None

    # `model` is a regular field here; don't warn on the model_ namespace.
    # extra="forbid" makes typo'd llm_config: keys (e.g. `temparature` instead of
    # `temperature`) loud at LOAD time, not silent.
    model_config = ConfigDict(protected_namespaces=(), extra="forbid")


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
