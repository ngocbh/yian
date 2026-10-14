import copy
import inspect
from contextvars import ContextVar
from functools import cache, cached_property
from typing import Any, Optional

import anthropic
from anthropic.resources.messages import Messages
from langchain_anthropic import ChatAnthropic
from pydantic import Field

from .base_client import BaseLLMClient, normalize_content
from .provider_blocks import merge_request_body
from .validators import validate_model

_PASSTHROUGH_KWARGS = (
    "timeout", "max_retries", "api_key", "callbacks", "http_client", "http_async_client",
)
_HTTP_CLIENT_KWARGS = ("http_client", "http_async_client")


# langchain-anthropic takes these as call options and moves them into ``extra_body`` itself
# when the installed SDK's ``Messages.create`` lacks them.
_SAMPLING_KEYWORDS = frozenset({"temperature", "top_k", "top_p"})


@cache
def _call_keywords() -> frozenset[str]:
    """Return the block keys ChatAnthropic can take as top-level call options.

    Returns:
        `frozenset[str]`:
            Keyword names of the SDK's ``Messages.create`` (without ``self``) plus the
            sampling fields langchain-anthropic relocates.
    """
    names = set(inspect.signature(Messages.create).parameters) - {"self"}
    return frozenset(names | _SAMPLING_KEYWORDS)


# The raw responses opened by the `_stream` call running in this context.
_OPENED_STREAMS: ContextVar[Optional[list]] = ContextVar("anthropic_opened_streams", default=None)


def _check_no_redirects(name: str, http_client: Any) -> None:
    """Reject a caller's HTTP client that follows redirects.

    Args:
        name (`str`):
            The keyword the client came in, for the message.
        http_client (`Any`):
            The caller's httpx client.

    Raises:
        `ValueError`:
            Raised when the client follows redirects.
    """
    # The key travels in x-api-key, which httpx re-sends on a cross-origin redirect.
    if getattr(http_client, "follow_redirects", False):
        raise ValueError(
            f"{name} follows redirects, which would send the API key past base_url; "
            "build it with follow_redirects=False"
        )


class NormalizedChatAnthropic(ChatAnthropic):
    """ChatAnthropic with normalized content output and streams that close early.

    Claude models with extended thinking or tool use return content as a
    list of typed blocks. This normalizes to string for consistent
    downstream handling.

    Attributes:
        follow_redirects (`bool`, *optional*, defaults to `True`):
            Whether the HTTP client follows redirects. The SDK sends the key in
            ``x-api-key``, which httpx re-sends on a cross-origin redirect, so
            `AnthropicClient` turns this off for a host ``base_url``. When false, a given
            ``http_client`` / ``http_async_client`` must not follow redirects either;
            building the SDK client raises `ValueError` otherwise.
        http_client (`httpx.Client`, *optional*, defaults to `None`):
            The SDK's sync HTTP client, used in place of the default one; it must come
            from the httpx package the SDK is built on. `None` uses the SDK default.
        http_async_client (`httpx.AsyncClient`, *optional*, defaults to `None`):
            The SDK's async HTTP client, used in place of the default one; it must come
            from the httpx package the SDK is built on. `None` uses the SDK default.
        block (`dict[str, Any]`, *optional*, defaults to `{}`):
            The ``anthropic`` block, shaped like the Messages API request body and sent
            into every request as written. Empty sends nothing extra.
    """

    follow_redirects: bool = True
    http_client: Any = Field(default=None, exclude=True, repr=False)
    http_async_client: Any = Field(default=None, exclude=True, repr=False)
    block: dict[str, Any] = Field(default_factory=dict, exclude=True, repr=False)

    @cached_property
    def _client(self) -> anthropic.Client:
        client = super()._client
        if self.http_client is not None:
            if not self.follow_redirects:
                _check_no_redirects("http_client", self.http_client)
            return client.with_options(http_client=self.http_client)
        if self.follow_redirects:
            return client
        return client.with_options(http_client=anthropic.DefaultHttpxClient(**self._http_options()))

    @cached_property
    def _async_client(self) -> anthropic.AsyncClient:
        client = super()._async_client
        if self.http_async_client is not None:
            if not self.follow_redirects:
                _check_no_redirects("http_async_client", self.http_async_client)
            return client.with_options(http_client=self.http_async_client)
        if self.follow_redirects:
            return client
        return client.with_options(
            http_client=anthropic.DefaultAsyncHttpxClient(**self._http_options())
        )

    def _http_options(self) -> dict[str, Any]:
        """Keyword arguments for an HTTP client that never follows a redirect."""
        options: dict[str, Any] = {"follow_redirects": False}
        if self.anthropic_proxy:
            options["proxy"] = self.anthropic_proxy
        return options

    def invoke(self, input, config=None, **kwargs):
        return self.normalize_reply(super().invoke(input, config, **kwargs))

    def with_structured_output(self, schema, *, method=None, **kwargs):
        """Bind `schema` as structured output, in a form the block's thinking allows.

        langchain-anthropic picks its method from the model's own ``thinking`` field,
        which a block's ``thinking`` does not set, so it would force a tool call beside
        thinking, a pair the Messages API rejects. With thinking on in the block and no
        `method` given, this uses the JSON-schema output format (``output_config.format``).

        Args:
            schema (`dict` or `type`):
                The output schema.
            method (`str`, *optional*, defaults to `None`):
                ``"function_calling"`` or ``"json_schema"``; `None` picks
                ``"json_schema"`` when the block turns thinking on, else
                ``"function_calling"``.
            **kwargs:
                Passed to `ChatAnthropic.with_structured_output`.

        Returns:
            `Runnable`:
                A runnable that returns outputs matching `schema`.
        """
        if method is None:
            thinking = self.block.get("thinking")
            thinking_on = isinstance(thinking, dict) and thinking.get("type") != "disabled"
            method = "json_schema" if thinking_on else "function_calling"
        return super().with_structured_output(schema, method=method, **kwargs)

    def _get_request_payload(self, input_, *, stop=None, **kwargs):
        """Return langchain's request payload with `block` merged into the call options.

        Merging before langchain builds the payload lets its own handling see the block:
        moving sampling fields the SDK lacks into ``extra_body``, deriving beta headers,
        and merging ``output_config`` so a block ``effort`` sits beside a structured
        output ``format``. Block keys the SDK method does not take go into ``extra_body``.

        Args:
            input_ (`LanguageModelInput`):
                The messages or prompt being sent.
            stop (`list[str]`, *optional*, defaults to `None`):
                Stop sequences for this call.
            **kwargs:
                Call options langchain folds into the payload.

        Returns:
            `dict`:
                Keyword arguments for the SDK method langchain calls.
        """
        if self.block:
            # Bound call options (e.g. a structured output's ``output_config``) are shared
            # across calls, so merge into copies of the ones the block touches.
            for key in (*self.block, "extra_body"):
                if isinstance(kwargs.get(key), dict):
                    kwargs[key] = copy.deepcopy(kwargs[key])
            merge_request_body(kwargs, self.block, top_level=_call_keywords())
        return super()._get_request_payload(input_, stop=stop, **kwargs)

    def _create(self, payload: dict) -> Any:
        response = super()._create(payload)
        opened = _OPENED_STREAMS.get()
        if opened is not None:
            opened.append(response)
        return response

    def _stream(self, *args: Any, **kwargs: Any):
        """Stream like `ChatAnthropic`, closing the HTTP response when closed early.

        langchain_anthropic iterates the SDK stream without closing it, so a
        caller that stops early would hold the connection until garbage
        collection. `_create` hands the raw response to this call through a
        context variable, set only while the inner generator runs.
        """
        opened: list = []
        chunks = super()._stream(*args, **kwargs)
        try:
            while True:
                token = _OPENED_STREAMS.set(opened)
                try:
                    chunk = next(chunks)
                except StopIteration:
                    return
                finally:
                    _OPENED_STREAMS.reset(token)
                yield chunk
        finally:
            chunks.close()
            for response in opened:
                response.http_response.close()

    def normalize_reply(self, response):
        """Apply the reply normalization `invoke` uses; streamed replies reuse it."""
        return normalize_content(response)


class AnthropicClient(BaseLLMClient):
    """Client for Anthropic Claude models.

    With a ``base_url``, the model does not follow redirects, so the key goes only
    to that URL. ``http_client`` / ``http_async_client`` replace the SDK's HTTP clients;
    beside a ``base_url`` they must not follow redirects.
    """

    def __init__(self, model: str, base_url: Optional[str] = None, **kwargs):
        super().__init__(model, base_url, **kwargs)

    def get_llm(self) -> Any:
        """Return configured ChatAnthropic instance.

        Raises:
            `ValueError`:
                Raised when a given ``http_client`` / ``http_async_client`` follows
                redirects beside a ``base_url``, or ``ANTHROPIC_PROXY`` is set beside one.
            `TypeError`:
                Raised when a given ``http_client`` / ``http_async_client`` is not the
                SDK's httpx client type.
        """
        self.warn_if_unknown_model()
        llm_kwargs = {"model": self.model, "block": self.kwargs.get("block") or {}}

        if self.base_url:
            llm_kwargs["base_url"] = self.base_url
            llm_kwargs["follow_redirects"] = False

        for key in _PASSTHROUGH_KWARGS:
            if key in self.kwargs:
                llm_kwargs[key] = self.kwargs[key]

        given = [key for key in _HTTP_CLIENT_KWARGS if llm_kwargs.get(key) is not None]
        llm = NormalizedChatAnthropic(**llm_kwargs)
        if given and llm.anthropic_proxy:
            raise ValueError(
                f"ANTHROPIC_PROXY can't be used with {given[0]}; set the proxy on that client"
            )
        # Build the SDK clients now so a client of the wrong type, or one that follows
        # redirects beside a base_url, fails here, not on the first request.
        if "http_client" in given:
            llm._client
        if "http_async_client" in given:
            llm._async_client
        return llm

    def validate_model(self) -> bool:
        """Validate model for Anthropic."""
        return validate_model("anthropic", self.model)
