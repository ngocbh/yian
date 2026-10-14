"""Canonical provider -> API-key env-var mapping.

A single source of truth for which environment variable holds the API
key for each supported LLM provider. Used by the CLI's interactive key
prompt (cli/utils.ensure_env_key) and by anything else that needs to
ask "does this provider require a key, and which env var is it?".

A config `api_key` (`LLMConfig.api_key`) takes precedence over these
variables when the client is built.

When adding a new provider, register its env var here so the CLI flow
prompts for it automatically instead of failing on first API call.
"""

from __future__ import annotations

from typing import Optional


PROVIDER_API_KEY_ENV: dict[str, Optional[str]] = {
    "openai":     "OPENAI_API_KEY",
    "anthropic":  "ANTHROPIC_API_KEY",
    "google":     "GOOGLE_API_KEY",
    "azure":      "AZURE_OPENAI_API_KEY",
    "xai":        "XAI_API_KEY",
    "deepseek":   "DEEPSEEK_API_KEY",
    # Dual-region providers each carry their own account; keys are not
    # interchangeable between the international and China endpoints.
    "qwen":       "DASHSCOPE_API_KEY",
    "qwen-cn":    "DASHSCOPE_CN_API_KEY",
    "glm":        "ZHIPU_API_KEY",
    "glm-cn":     "ZHIPU_CN_API_KEY",
    "minimax":    "MINIMAX_API_KEY",
    "minimax-cn": "MINIMAX_CN_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    # Local runtimes do not authenticate.
    "ollama":     None,
    # vLLM is keyless by default but can be started with `--api-key`; the key is
    # optional (see OPTIONAL_API_KEY_PROVIDERS).
    "vllm":       "VLLM_API_KEY",
}

# Self-hosted runtimes whose key variable may be unset: vLLM runs keyless unless
# its server was started with `--api-key`, so its client sends a dummy key and
# the CLI does not ask for one.
OPTIONAL_API_KEY_PROVIDERS: frozenset[str] = frozenset({"vllm"})


def get_api_key_env(provider: str) -> Optional[str]:
    """Return the env var name for `provider`'s API key, or None if not applicable.

    Unknown providers also return None — callers should treat that as
    "no key check possible" rather than as "no key required".
    """
    return PROVIDER_API_KEY_ENV.get(provider.lower())
