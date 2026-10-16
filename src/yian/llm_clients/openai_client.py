import copy
import inspect
import os
from functools import cache
from typing import Any, ClassVar, Literal, Optional

import httpx
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from langchain_openai.chat_models.base import BaseChatOpenAI
from openai.resources.chat.completions import Completions
from openai.resources.responses import Responses
from pydantic import Field, SecretStr, model_validator

from .api_key_env import OPTIONAL_API_KEY_PROVIDERS, get_api_key_env
from .base_client import (
    HEADERS_LEAK,
    BaseLLMClient,
    check_connection_headers,
    check_no_redirects,
    drop_environment_headers,
    normalize_content,
)
from .capabilities import get_capabilities, has_open_schema
from .factory import _PROVIDER_BASE_URL
from .provider_blocks import merge_request_body
from .validators import validate_model

# SDK keywords that configure the HTTP call rather than name a request-body field.
_TRANSPORT_KEYWORDS = frozenset({"self", "extra_headers", "extra_query", "extra_body", "timeout"})
# The environment variables the OpenAI SDK or langchain read headers from when a client is built.
_OPENAI_HEADER_VARIABLES = (
    "OPENAI_CUSTOM_HEADERS", "OPENAI_ORG_ID", "OPENAI_ORGANIZATION", "OPENAI_PROJECT_ID"
)
_HTTP_CLIENT_KEYWORDS = ("http_client", "http_async_client")


def connection_header_clients(
    kwargs: dict[str, Any], proxy: Optional[str] = None
) -> dict[str, Any]:
    """Return the HTTP client keywords for a model that sends connection headers.

    httpx re-sends custom headers on a cross-origin redirect, so a host client must not
    follow redirects, and a missing one is a fresh client that doesn't. langchain-openai
    caches its default clients per endpoint and shares them with other models, so they
    are replaced rather than changed.

    Args:
        kwargs (`dict[str, Any]`): The client's keyword arguments, holding any host
            ``http_client`` / ``http_async_client``.
        proxy (`str`, *optional*, defaults to `None`): The proxy URL the fresh clients
            use, as langchain's own clients would.

    Returns:
        `dict[str, Any]`: ``http_client`` and ``http_async_client`` (the host's or fresh
            ones), plus ``openai_proxy=None`` when both are fresh, since langchain refuses
            a proxy beside a passed client and the fresh ones carry it already.

    Raises:
        `ValueError`: A host client follows redirects.
    """
    clients: dict[str, Any] = {}
    for key, fresh in zip(_HTTP_CLIENT_KEYWORDS, (httpx.Client, httpx.AsyncClient)):
        if key in kwargs:
            check_no_redirects(key, kwargs[key], reason=HEADERS_LEAK)
            clients[key] = kwargs[key]
        else:
            clients[key] = fresh(follow_redirects=False, proxy=proxy)
    if not any(key in kwargs for key in _HTTP_CLIENT_KEYWORDS):
        clients["openai_proxy"] = None
    return clients


@cache
def _sdk_body_keywords(responses: bool) -> frozenset[str]:
    """Return the request-body fields the SDK method langchain calls takes as keywords.

    Args:
        responses (`bool`):
            `True` for the Responses API (`responses.create`/`parse`), `False` for Chat
            Completions (`chat.completions.create`/`parse`).

    Returns:
        `frozenset[str]`:
            Keyword names of both methods, without the transport options.
    """
    resource = Responses if responses else Completions
    methods = (resource.create, resource.parse)
    names = {name for method in methods for name in inspect.signature(method).parameters}
    return frozenset(names - _TRANSPORT_KEYWORDS)


def _strict_false_for_open_schema(
    llm: BaseChatOpenAI, schema: Any, method: Optional[str], kwargs: dict[str, Any]
) -> None:
    """Set ``kwargs["strict"] = False`` for an open schema bound as a Responses function tool.

    With ``strict`` omitted, the Responses API tries strict mode, which can't express a
    schema with an open part (`has_open_schema`); an explicit ``False`` opts out. Chat Completions tools are non-strict unless asked, and an
    explicit ``strict`` is the caller's choice, so both are left alone.

    Args:
        llm (`BaseChatOpenAI`):
            The model the schema is bound to.
        schema (`Any`):
            The schema given to ``with_structured_output``.
        method (`str`, *optional*):
            The resolved structured-output method.
        kwargs (`dict[str, Any]`):
            The remaining ``with_structured_output`` keyword arguments; updated in place.
    """
    if method != "function_calling" or "strict" in kwargs or not has_open_schema(schema):
        return
    # langchain routes by the model's own fields and name plus any Responses-only
    # argument in the call, so the call's own arguments are what it is asked about.
    if llm._use_responses_api(kwargs):
        kwargs["strict"] = False


class RequestBlockMixin(BaseChatOpenAI):
    """Send a provider block into every request body an OpenAI-family chat model builds.

    Put it before the langchain class in the bases so its payload override runs after
    langchain's own (and before a subclass's post-processing). The block is merged as
    written: keys the SDK method takes stay top-level, the rest go into ``extra_body``.

    The SDK clients keep only the headers passed in ``default_headers`` (plus the
    subclass's `_sdk_header_names`) and `connection_headers`; headers the SDK read from
    the environment are dropped. The organization and project are always dropped, so a
    Python caller's ``organization=`` is dropped too; they are sent only as
    ``OpenAI-Organization`` / ``OpenAI-Project`` connection headers.

    Attributes:
        block (`dict[str, Any]`, *optional*, defaults to `{}`):
            The provider block, shaped like the request body of `block_api`. Empty sends
            nothing extra.
        block_provider (`str`, *optional*, defaults to `""`):
            The provider whose block this is, named in errors.
        block_api (`Literal["chat", "responses"]`, *optional*, defaults to `"chat"`):
            The API whose request body the block mirrors: Chat Completions or Responses.
        connection_headers (`dict[str, SecretStr]`, *optional*, defaults to `{}`):
            Headers sent with every request, added to the SDK clients' custom headers.
            Values are secrets: unlike ``default_headers``, this field is left out of
            ``repr``, dumps and langchain serialization.
    """

    block: dict[str, Any] = Field(default_factory=dict, exclude=True, repr=False)
    block_provider: str = Field(default="", exclude=True, repr=False)
    block_api: Literal["chat", "responses"] = Field(default="chat", exclude=True, repr=False)
    connection_headers: dict[str, SecretStr] = Field(
        default_factory=dict, exclude=True, repr=False
    )
    # Custom headers the langchain class itself passes to the SDK client.
    _sdk_header_names: ClassVar[tuple[str, ...]] = ()

    @model_validator(mode="after")
    def _drop_environment_headers(self):
        """Drop the environment's headers, organization and project from the SDK clients.

        Runs after langchain's validator builds ``root_client``/``root_async_client``,
        then adds `connection_headers` to both clients.

        Returns:
            `RequestBlockMixin`:
                This model.
        """
        self.openai_organization = None
        explicit = (
            *(self.default_headers or {}), *self._sdk_header_names, *self.connection_headers
        )
        values = {
            name: value.get_secret_value() for name, value in self.connection_headers.items()
        }
        # Both clients drop the same headers; `drop_environment_headers` shows that
        # message once per process.
        for client in (self.root_client, self.root_async_client):
            drop_environment_headers(client, explicit, _OPENAI_HEADER_VARIABLES)
            if client is not None:
                # The SDK sends `_custom_headers` last, after its own headers.
                client._custom_headers.update(values)
        return self

    def _get_request_payload(self, input_, *, stop=None, **kwargs):
        """Return langchain's request payload with `block` merged in.

        Merging after langchain means its renames (e.g. ``max_tokens`` to
        ``max_completion_tokens``) never touch a block key.

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

        Raises:
            `ValueError`:
                Raised when the block mirrors Chat Completions but langchain routes this
                model to the Responses API.
        """
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        if not self.block:
            return payload
        responses = self._use_responses_api(payload)
        if responses and self.block_api == "chat":
            raise ValueError(
                f"The {self.block_provider!r} block mirrors the Chat Completions request, "
                f"but model {self.model_name!r} is routed to the Responses API; remove the "
                f"block or use a model served through Chat Completions."
            )
        top_level = _sdk_body_keywords(responses)
        # Nested payload dicts can be the model's own fields or the caller's call
        # options; copy the ones the merge writes into so no call leaks into the next.
        for key in {*self.block, "extra_body"}:
            if isinstance(payload.get(key), dict):
                payload[key] = copy.deepcopy(payload[key])
        return merge_request_body(payload, self.block, top_level=top_level)


class NormalizedChatOpenAI(RequestBlockMixin, ChatOpenAI):
    """ChatOpenAI with capability-aware binding and lossless Responses messages.

    Responses content blocks retain reasoning and tool continuation data for
    subsequent requests. Agent presentation extracts visible text separately.
    Chat Completions responses keep the existing string normalization.

    ``with_structured_output`` consults the per-model capability table
    (``capabilities.get_capabilities``) to pick the method and to decide
    whether ``tool_choice`` may be sent. Models that reject ``tool_choice``
    (e.g. DeepSeek V4 and reasoner — per their official tool-calling
    guide) still bind the schema as a tool, but no ``tool_choice``
    parameter is sent.

    Provider-specific quirks beyond structured-output (e.g. DeepSeek's
    reasoning_content roundtrip) live in subclasses so this base class
    stays small.
    """

    def invoke(self, input, config=None, **kwargs):
        return self.normalize_reply(super().invoke(input, config, **kwargs))

    def normalize_reply(self, response):
        """Apply the reply normalization `invoke` uses; streamed replies reuse it."""
        if self.use_responses_api or response.response_metadata.get("object") == "response":
            return response
        return normalize_content(response)

    def with_structured_output(self, schema, *, method=None, **kwargs):
        """Bind `schema` for structured output with the method the capability table picks.

        On a request langchain routes to the Responses API, where function tools default
        to strict, a schema with an open part (`has_open_schema`) is sent with
        ``strict=False``, since strict mode rejects it. An explicit ``method`` or
        ``strict`` is respected; Chat Completions requests are unchanged.

        Args:
            schema (`dict | type`):
                The output schema: a pydantic class, a TypedDict or a JSON-schema dict.
            method (`str`, *optional*, defaults to `None`):
                The structured-output method; `None` takes the model's preferred one.
            **kwargs:
                Passed to `ChatOpenAI.with_structured_output`.

        Returns:
            `Runnable`:
                The structured-output runnable langchain builds.

        Raises:
            `NotImplementedError`:
                The model has no structured-output method.
        """
        caps = get_capabilities(self.model_name)
        if caps.preferred_structured_method == "none":
            raise NotImplementedError(
                f"{self.model_name} has no structured-output method available; "
                f"agent factories will fall back to free-text generation."
            )
        method = method or caps.preferred_structured_method
        # When the model rejects tool_choice, suppress langchain's hardcoded
        # value. The schema is still bound as a tool — exactly what
        # DeepSeek's official tool-calling examples do.
        if method == "function_calling" and not caps.supports_tool_choice:
            kwargs.setdefault("tool_choice", None)
        _strict_false_for_open_schema(self, schema, method, kwargs)
        return super().with_structured_output(schema, method=method, **kwargs)


def _input_to_messages(input_: Any) -> list:
    """Normalise a langchain LLM input to a list of message objects.

    Accepts a list of messages, a ``ChatPromptValue`` (from a
    ChatPromptTemplate), or anything else (treated as no messages).
    Used by providers that need to walk the outgoing message history;
    in particular DeepSeek thinking-mode propagation must work for
    both bare-list invocations and ChatPromptTemplate-driven ones, so
    treating only ``list`` here would silently skip half the call sites.
    """
    if isinstance(input_, list):
        return input_
    if hasattr(input_, "to_messages"):
        return input_.to_messages()
    return []


class DeepSeekChatOpenAI(NormalizedChatOpenAI):
    """DeepSeek-specific overrides on top of the OpenAI-compatible client.

    Thinking-mode round-trip is the only DeepSeek-specific behavior that
    stays here. When DeepSeek's thinking models return a response with
    ``reasoning_content``, that field must be echoed back as part of the
    assistant message on the next turn or the API fails with HTTP 400.
    ``_create_chat_result`` captures it on receive and
    ``_get_request_payload`` re-attaches it on send.

    Tool-choice handling for V4 and reasoner — those models reject the
    ``tool_choice`` parameter — is handled by the capability dispatch in
    ``NormalizedChatOpenAI.with_structured_output``, not here.

    Streaming is disabled because only the non-streaming result path captures
    ``reasoning_content``; ``stream()`` falls back to one complete reply.
    """

    disable_streaming: bool = True

    def _get_request_payload(self, input_, *, stop=None, **kwargs):
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        outgoing = payload.get("messages", [])
        for message_dict, message in zip(outgoing, _input_to_messages(input_)):
            if not isinstance(message, AIMessage):
                continue
            reasoning = message.additional_kwargs.get("reasoning_content")
            if reasoning is not None:
                message_dict["reasoning_content"] = reasoning
        return payload

    def _create_chat_result(self, response, generation_info=None):
        chat_result = super()._create_chat_result(response, generation_info)
        response_dict = (
            response
            if isinstance(response, dict)
            else response.model_dump(
                exclude={"choices": {"__all__": {"message": {"parsed"}}}}
            )
        )
        for generation, choice in zip(
            chat_result.generations, response_dict.get("choices", [])
        ):
            reasoning = choice.get("message", {}).get("reasoning_content")
            if reasoning is not None:
                generation.message.additional_kwargs["reasoning_content"] = reasoning
        return chat_result


class MinimaxChatOpenAI(NormalizedChatOpenAI):
    """MiniMax-specific overrides on top of the OpenAI-compatible client.

    M2.x reasoning models embed ``<think>...</think>`` blocks directly in
    ``message.content`` by default, which would pollute saved reports.
    Per platform.minimax.io/docs/api-reference/text-openai-api,
    ``reasoning_split=True`` redirects the thinking block into
    ``reasoning_details`` so ``content`` stays clean. It is sent via
    ``extra_body`` (not a top-level kwarg) because the openai SDK validates
    top-level params and rejects unknown ones like reasoning_split (#826).

    The flag is gated by ``ModelCapabilities.requires_reasoning_split`` so
    only M2.x reasoning models receive it; non-reasoning MiniMax endpoints
    (Coding Plan, MiniMax-Text-01) never see it.

    Tool-choice handling for M2.x — those models accept only the string
    enum ``{"none", "auto"}`` and reject langchain's function-spec dict —
    is handled by the capability dispatch in
    ``NormalizedChatOpenAI.with_structured_output``, not here.
    """

    def _get_request_payload(self, input_, *, stop=None, **kwargs):
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        if get_capabilities(self.model_name).requires_reasoning_split:
            # Pass via extra_body, not as a top-level kwarg: the openai SDK
            # (>=1.56) validates top-level params against Completions.create
            # and rejects unknown ones like reasoning_split (#826). extra_body
            # is forwarded into the request body untouched.
            extra_body = payload.setdefault("extra_body", {})
            extra_body.setdefault("reasoning_split", True)
        return payload


# Kwargs forwarded from user config to ChatOpenAI
_PASSTHROUGH_KWARGS = (
    "timeout", "max_retries", "api_key", "callbacks", "http_client",
    "http_async_client",
)

# Per-provider env var that overrides the default base URL (call-time, so tests
# monkeypatching the env after import still take effect).
_BASE_URL_ENV = {
    "vllm": "VLLM_BASE_URL",
}


def _resolve_provider_base_url(provider: str) -> Optional[str]:
    """Default base URL for ``provider``, with env-var overrides where defined."""
    env_var = _BASE_URL_ENV.get(provider)
    if env_var:
        env_url = os.environ.get(env_var)
        if env_url:
            return env_url
    return _PROVIDER_BASE_URL.get(provider)


class OpenAIClient(BaseLLMClient):
    """Client for OpenAI and OpenAI-compatible providers (xAI, OpenRouter,
    vLLM, …). Ollama has its own native client (see ollama_client).

    For native OpenAI models, uses the Responses API (/v1/responses) which
    supports reasoning with function tools across all model families
    (GPT-4.1, GPT-5). Third-party compatible providers (xAI, OpenRouter,
    vLLM) use standard Chat Completions.
    """

    def __init__(
        self,
        model: str,
        base_url: Optional[str] = None,
        provider: str = "openai",
        **kwargs,
    ):
        super().__init__(model, base_url, **kwargs)
        self.provider = provider.lower()

    def get_llm(self) -> Any:
        """Return configured ChatOpenAI instance.

        ``connection_headers`` (`dict[str, str | SecretStr]`, a keyword argument) are sent
        with every request; such a model never follows a redirect.

        Raises:
            `ValueError`: The provider's API key is unset, a connection header is refused
                (`check_connection_headers`), or a host HTTP client beside connection
                headers follows redirects.
        """
        self.warn_if_unknown_model()
        llm_kwargs = {"model": self.model}

        # Provider-specific base URL and auth. An explicit base_url on the
        # client (e.g. a corporate proxy) takes precedence over the
        # provider default so users can route through their own gateway.
        if self.provider in _PROVIDER_BASE_URL:
            llm_kwargs["base_url"] = self.base_url or _resolve_provider_base_url(self.provider)
            api_key_env = get_api_key_env(self.provider)
            if "api_key" in self.kwargs:
                # Forwarded below: a caller-supplied key wins over the env var.
                pass
            elif api_key_env:
                api_key = os.environ.get(api_key_env)
                if api_key:
                    llm_kwargs["api_key"] = api_key
                elif self.provider in OPTIONAL_API_KEY_PROVIDERS:
                    # A keyless server; the OpenAI SDK still needs some string.
                    llm_kwargs["api_key"] = "EMPTY"
                else:
                    raise ValueError(
                        f"API key for provider '{self.provider}' is not set. "
                        f"Please set the {api_key_env} environment variable "
                        f"(e.g. add {api_key_env}=your_key to your .env file)."
                    )
            else:
                llm_kwargs["api_key"] = "ollama"
        elif self.base_url:
            llm_kwargs["base_url"] = self.base_url

        # Forward user-provided kwargs
        for key in _PASSTHROUGH_KWARGS:
            if key in self.kwargs:
                llm_kwargs[key] = self.kwargs[key]
        if self.kwargs.get("connection_headers"):
            llm_kwargs["connection_headers"] = check_connection_headers(
                self.kwargs["connection_headers"], self.provider
            )
            # Only models with headers get their own clients; the rest keep langchain's
            # defaults (and the stream_usage default that depends on them).
            llm_kwargs.update(
                connection_header_clients(self.kwargs, os.environ.get("OPENAI_PROXY") or None)
            )

        # Native OpenAI: use Responses API for consistent behavior across
        # all model families. Third-party providers use Chat Completions.
        if self.provider == "openai":
            llm_kwargs["use_responses_api"] = True
        elif "base_url" in llm_kwargs:
            # LangChain requests streamed usage only for the default endpoint;
            # OpenAI-compatible servers need it asked for explicitly.
            llm_kwargs["stream_usage"] = True

        # Provider-specific quirks live in their own subclasses so the
        # base NormalizedChatOpenAI stays free of provider branches.
        if self.provider == "deepseek":
            chat_cls = DeepSeekChatOpenAI
        elif self.provider in ("minimax", "minimax-cn"):
            chat_cls = MinimaxChatOpenAI
        else:
            chat_cls = NormalizedChatOpenAI
        return chat_cls(
            **llm_kwargs,
            block=self.kwargs.get("block") or {},
            block_provider=self.provider,
            block_api="responses" if self.provider == "openai" else "chat",
        )

    def validate_model(self) -> bool:
        """Validate model for the provider."""
        return validate_model(self.provider, self.model)
