"""Client for a local Ollama runtime, via langchain's native `ChatOllama`.

Ollama also exposes an OpenAI-compatible `/v1` endpoint, but that path drops the
`reasoning` field and gives no way to disable a model's thinking mode — so a
reasoning model (e.g. qwen3.5) routes its final answer into the hidden reasoning
channel and returns empty `content`, which silently breaks the agent tool loop.
The native API takes `reasoning=False` to turn thinking off, so the answer lands
in `content` where the loop expects it. Hence Ollama gets its own client rather
than sharing the OpenAI-compatible one.
"""

import os
from typing import Any, Optional

from langchain_ollama import ChatOllama

from .base_client import BaseLLMClient

# Native endpoint (no `/v1` suffix — that's the OpenAI-compat path).
_DEFAULT_BASE_URL = "http://localhost:11434"

# ChatOllama accepts a focused set of generation knobs; forward only these so a
# stray kwarg from the shared `model_from_config` path (e.g. `max_retries`,
# which ChatOllama doesn't take) doesn't blow up construction.
_PASSTHROUGH_KWARGS = ("temperature", "num_ctx", "num_predict", "top_p", "seed")


def _resolve_base_url(explicit: Optional[str]) -> str:
    """Native base URL: explicit arg > OLLAMA_BASE_URL > default. A trailing
    `/v1` (the OpenAI-compat suffix) is stripped so either form works."""
    url = explicit or os.environ.get("OLLAMA_BASE_URL") or _DEFAULT_BASE_URL
    return url[: -len("/v1")] if url.rstrip("/").endswith("/v1") else url


class OllamaClient(BaseLLMClient):
    """Local Ollama models via the native API, with thinking disabled."""

    provider = "ollama"

    def get_llm(self) -> Any:
        """Return a `ChatOllama` on the native endpoint, with thinking disabled by default."""
        self.warn_if_unknown_model()
        kwargs: dict[str, Any] = {
            "model": self.model,
            "base_url": _resolve_base_url(self.base_url),
            # Off by default: a reasoning model must answer in `content`, not in
            # the hidden reasoning channel, for the agent tool loop to see it.
            "reasoning": self.kwargs.get("reasoning", False),
        }
        for key in _PASSTHROUGH_KWARGS:
            if key in self.kwargs:
                kwargs[key] = self.kwargs[key]
        return ChatOllama(**kwargs)

    def validate_model(self) -> bool:
        """Always `True` — a local Ollama serves whatever the user has pulled (no catalog gate)."""
        return True
