"""Client for a local Ollama runtime, via langchain's native `ChatOllama`.

Ollama also exposes an OpenAI-compatible `/v1` endpoint, but that path drops the
`reasoning` field and cannot carry `think`, so a reasoning model's thinking cannot
be controlled there. The native `/api/chat` takes `think`, `options` and
`keep_alive` directly, so Ollama gets its own client rather than sharing the
OpenAI-compatible one.

The `ollama` block of `llm_config` is merged into the `/api/chat` request as
written. Nothing sets `think` by default: it is sent only when the block sets it,
otherwise the model's own default applies.
"""

import copy
import os
from typing import Any, Optional

from langchain_core.messages import BaseMessage
from langchain_ollama import ChatOllama
from pydantic import Field

from .base_client import BaseLLMClient
from .provider_blocks import merge_request_body

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


class NormalizedChatOllama(ChatOllama):
    """`ChatOllama` that merges the `ollama` block into every `/api/chat` request.

    `reasoning` stays unset, so the request carries `think` only when the block sets
    it. With `think` set in the block alone, langchain-ollama does not copy the reply's
    `message.thinking` into `reasoning_content` (it does so only when its own
    `reasoning` is truthy); the answer still lands in `content`.

    Attributes:
        block (`dict[str, Any]`, *optional*, defaults to `{}`):
            The `ollama` block, merged into the request parameters as written.
    """

    block: dict[str, Any] = Field(default_factory=dict, exclude=True, repr=False)

    def _chat_params(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Return langchain's `/api/chat` parameters with the block deep-merged in.

        Args:
            messages (`list[BaseMessage]`):
                The messages to send.
            stop (`list[str]`, *optional*, defaults to `None`):
                Stop words for this call.
            **kwargs:
                Call-time parameters, passed to `ChatOllama._chat_params`.

        Returns:
            `dict[str, Any]`:
                Keyword arguments for `ollama.Client.chat`.
        """
        params = super()._chat_params(messages, stop, **kwargs)
        # langchain builds `options` afresh per call only when the caller passes none;
        # copy it so the merge never writes into a caller-owned dict.
        if isinstance(params.get("options"), dict):
            params["options"] = copy.copy(params["options"])
        return merge_request_body(params, self.block)


class OllamaClient(BaseLLMClient):
    """Local Ollama models via the native API, sending the `ollama` block as written."""

    provider = "ollama"

    def get_llm(self) -> Any:
        """Return a `NormalizedChatOllama` on the native endpoint carrying the `ollama` block."""
        self.warn_if_unknown_model()
        kwargs: dict[str, Any] = {
            "model": self.model,
            "base_url": _resolve_base_url(self.base_url),
            "block": self.kwargs.get("block") or {},
        }
        for key in _PASSTHROUGH_KWARGS:
            if key in self.kwargs:
                kwargs[key] = self.kwargs[key]
        return NormalizedChatOllama(**kwargs)

    def validate_model(self) -> bool:
        """Always `True` — a local Ollama serves whatever the user has pulled (no catalog gate)."""
        return True
