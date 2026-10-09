"""Environment overrides and provider-scoped fallback model names.

This standard-library leaf owns no authentication or persistent settings.
``llm_clients.connection`` combines these values with a saved auth selection at
host boundaries, then injects the resulting defaults into the pure compiler.
"""

from __future__ import annotations

import os

DEFAULT_PROVIDER = "openai"
DEFAULT_MODEL = "gpt-5.5"
PROVIDER_DEFAULT_MODELS = {
    "openai": DEFAULT_MODEL,
    "anthropic": "claude-sonnet-4-5",
}


def default_llm_provider() -> str:
    """Return the environment-selected provider or the OpenAI fallback."""
    return os.environ.get("YIAN_DEFAULT_PROVIDER", DEFAULT_PROVIDER).lower()


def default_llm_model(provider: str | None = None) -> str:
    """Return a matching default, requiring an explicit model for other providers."""
    selected = default_llm_provider()
    provider = (provider or selected).lower()
    model = (
        os.environ.get("YIAN_DEFAULT_MODEL") if provider == selected else None
    )
    model = model or PROVIDER_DEFAULT_MODELS.get(provider)
    if not model:
        raise ValueError(
            f"Provider {provider!r} needs an explicit model; set llm_config.model or --model"
        )
    return model
