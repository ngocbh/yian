"""Client for a local Ollama runtime, via langchain's native `ChatOllama`.

Ollama also exposes an OpenAI-compatible `/v1` endpoint, but that path drops the
`reasoning` field and cannot carry `think`, so a reasoning model's thinking cannot
be controlled there. The native `/api/chat` takes `think`, `options` and
`keep_alive` directly, so Ollama gets its own client rather than sharing the
OpenAI-compatible one.

The `ollama` block of `llm_config` is merged into the `/api/chat` request as
written. Nothing sets `think` by default: it is sent only when the block sets it,
otherwise the model's own default applies.

yian sends no key to Ollama: the `Authorization` header the ollama SDK builds from
`OLLAMA_API_KEY` is dropped, while Basic auth from userinfo in `OLLAMA_BASE_URL` stays.
"""

import copy
import os
from typing import Any, Optional, Self

from langchain_core.messages import BaseMessage
from langchain_ollama import ChatOllama
from pydantic import Field, model_validator

from .base_client import BaseLLMClient, warn_once
from .provider_blocks import merge_request_body

# Native endpoint (no `/v1` suffix — that's the OpenAI-compat path).
_DEFAULT_BASE_URL = "http://localhost:11434"


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

    The `Authorization` header the SDK built from `OLLAMA_API_KEY` is dropped, with a
    once-per-process warning, so no key is sent.

    Attributes:
        block (`dict[str, Any]`, *optional*, defaults to `{}`):
            The `ollama` block, merged into the request parameters as written.
    """

    block: dict[str, Any] = Field(default_factory=dict, exclude=True, repr=False)

    @model_validator(mode="after")
    def _drop_environment_key(self) -> Self:
        """Remove the `Authorization` header the SDK built from `OLLAMA_API_KEY`.

        Runs after langchain's `_set_clients`. An `Authorization` header equal to
        `Bearer <OLLAMA_API_KEY>` is dropped from both SDK clients whatever its source, so
        an explicit one with that value would go too (yian never passes one). Basic auth
        from userinfo in the base URL never matches and stays.

        Returns:
            `Self`:
                This model, with the header removed from its SDK clients.
        """
        # The ollama SDK reads the variable when the client is built and sends it as a
        # Bearer token, but yian resolves no Ollama credential. yian never sets langchain's
        # `validate_model_on_init`, which would send a request before this runs.
        key = os.environ.get("OLLAMA_API_KEY")
        if not key:
            return self
        dropped = False
        for client in (self._client, self._async_client):
            headers = client._client.headers
            if headers.get("authorization") == f"Bearer {key}":
                del headers["authorization"]
                dropped = True
        if dropped:
            warn_once(
                "yian sends no key to Ollama, so it dropped the Authorization header read "
                "from OLLAMA_API_KEY. Unset the variable to silence this.",
                stacklevel=2,
            )
        return self

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
    """Local Ollama models via the native API, sending the `ollama` block as written.

    No key is sent: the model drops the header the SDK built from `OLLAMA_API_KEY`.
    """

    provider = "ollama"

    def get_llm(self) -> Any:
        """Return a `NormalizedChatOllama` on the native endpoint carrying the `ollama` block.

        Raises:
            `ValueError`:
                Raised when an ``http_client`` or ``http_async_client`` is given.
        """
        self.warn_if_unknown_model()
        for key in ("http_client", "http_async_client"):
            if self.kwargs.get(key) is not None:
                raise ValueError(
                    f"the ollama client can't use {key}: the Ollama SDK builds its own "
                    "httpx client"
                )
        kwargs: dict[str, Any] = {
            "model": self.model,
            "base_url": _resolve_base_url(self.base_url),
            "block": self.kwargs.get("block") or {},
        }
        # Generation settings come only from the block; other client kwargs such as
        # `max_retries` are not ChatOllama fields and are ignored; the HTTP clients are
        # rejected above.
        return NormalizedChatOllama(**kwargs)

    def validate_model(self) -> bool:
        """Always `True` — a local Ollama serves whatever the user has pulled (no catalog gate)."""
        return True
