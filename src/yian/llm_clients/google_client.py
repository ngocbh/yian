import builtins
import copy
import typing
from typing import Any, Optional

import pydantic
from google.genai.types import HttpOptions
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import Field, model_validator

from .base_client import BaseLLMClient, check_no_redirects, normalize_content
from .provider_blocks import merge_request_body
from .validators import validate_model

_HTTP_CLIENT_KWARGS = ("http_client", "http_async_client")


def _class_name(cls: type) -> str:
    """Return `cls` as written in an error message.

    Args:
        cls (`type`):
            The class to name.

    Returns:
        `str`:
            The bare name for a builtin class (`object`), else the top-level module and
            qualified name (`httpx.Client`).
    """
    if cls.__module__ == builtins.__name__:
        return cls.__qualname__
    return f"{cls.__module__.split('.')[0]}.{cls.__qualname__}"


def _annotation_classes(annotation: Any) -> list[type]:
    """Return the classes in a type annotation, flattening unions and dropping `None`.

    Args:
        annotation (`Any`):
            A pydantic field annotation, such as `Optional[Union[httpx.Client, httpx2.Client]]`.

    Returns:
        `list[type]`:
            The classes it allows, in order.
    """
    args = typing.get_args(annotation)
    if not args:
        return [] if annotation is type(None) else [annotation]
    return [cls for arg in args for cls in _annotation_classes(arg)]


def _check_http_client_type(name: str, value: Any) -> None:
    """Reject a caller's HTTP client that google-genai does not take.

    Args:
        name (`str`):
            The keyword the client came in (``http_client`` or ``http_async_client``).
        value (`Any`):
            The caller's client.

    Raises:
        `TypeError`:
            Raised when `value` is not one of the httpx client classes google-genai
            accepts for that slot.
    """
    field = "httpx_async_client" if name == "http_async_client" else "httpx_client"
    try:
        HttpOptions(**{field: value})
    except pydantic.ValidationError:
        # google-genai adds httpx2's classes to the annotation only when httpx2 imports.
        accepted = _annotation_classes(HttpOptions.model_fields[field].annotation)
        expected = " or ".join(_class_name(cls) for cls in accepted)
        raise TypeError(f"{name} must be an {expected}, got {_class_name(type(value))}") from None


def _const_as_enum(schema: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of a JSON schema with each ``const`` keyword written as a one-value ``enum``.

    Gemini's ``responseJsonSchema`` supports only a subset of JSON Schema, which lists
    ``enum`` but not ``const`` (google-genai ``GenerateContentConfig.response_json_schema``).
    ``{"const": v}`` and ``{"enum": [v]}`` accept the same values, so the rewrite is exact.
    A non-string ``const`` (a number, boolean or null) becomes ``enum: [v]`` too.

    Args:
        schema (`dict[str, Any]`):
            The schema to rewrite. It is not changed.

    Returns:
        `dict[str, Any]`:
            A deep copy of `schema` where every subschema under ``properties``, ``$defs``,
            ``items``, ``prefixItems``, ``additionalProperties``, ``anyOf``, ``oneOf`` and
            ``allOf`` holding ``const: v`` has ``enum: [v]`` instead. A subschema whose
            existing ``enum`` holds ``v`` gets ``enum: [v]``; one whose ``enum`` lacks
            ``v`` is left as written.
    """
    result = copy.deepcopy(schema)
    stack = [result]
    while stack:
        node = stack.pop()
        if not isinstance(node, dict):
            continue
        if "const" in node:
            value = node["const"]
            if "enum" not in node or value in node["enum"]:
                del node["const"]
                node["enum"] = [value]
        for key in ("properties", "$defs"):
            if isinstance(node.get(key), dict):
                stack.extend(node[key].values())
        for key in ("items", "additionalProperties"):
            if isinstance(node.get(key), dict):
                stack.append(node[key])
        for key in ("prefixItems", "anyOf", "oneOf", "allOf"):
            if isinstance(node.get(key), list):
                stack.extend(node[key])
    return result


class NormalizedChatGoogleGenerativeAI(ChatGoogleGenerativeAI):
    """ChatGoogleGenerativeAI with normalized content output.

    Gemini 3 models return content as list of typed blocks.
    This normalizes to string for consistent downstream handling.

    Attributes:
        block (`dict[str, Any]`, *optional*, defaults to `{}`):
            The ``google`` block, shaped like the REST ``generateContent`` request body
            (camelCase names, REST enum spellings). It is deep-merged into every request
            body as written. Empty sends nothing extra.
        http_client (`Any`, *optional*, defaults to `None`):
            The SDK's sync httpx client, used in place of its default one. `None` uses the
            SDK default.
        http_async_client (`Any`, *optional*, defaults to `None`):
            The SDK's async httpx client, used in place of its default one. `None` uses the
            SDK default.
    """

    block: dict[str, Any] = Field(default_factory=dict, exclude=True, repr=False)
    http_client: Any = Field(default=None, exclude=True, repr=False)
    http_async_client: Any = Field(default=None, exclude=True, repr=False)

    @model_validator(mode="after")
    def _use_http_clients(self) -> "NormalizedChatGoogleGenerativeAI":
        """Hand the given httpx clients to the genai client langchain built.

        Returns:
            `NormalizedChatGoogleGenerativeAI`:
                This model, its genai client now sending through the given clients.
        """
        # langchain builds the genai client itself and has no slot for an httpx client;
        # these are the attributes google-genai sets from `HttpOptions.httpx_client` /
        # `httpx_async_client`. Keeping them in `_http_options` stops the SDK from closing
        # the caller's client and from sending async calls through aiohttp. The default
        # clients the SDK already built are left to their own `__del__`.
        api = self.client._api_client
        if self.http_client is not None:
            api._http_options.httpx_client = self.http_client
            api._httpx_client = self.http_client
        if self.http_async_client is not None:
            api._http_options.httpx_async_client = self.http_async_client
            api._async_httpx_client = self.http_async_client
        return self

    def _prepare_request(self, messages, **kwargs):
        """Return langchain's request with `block` added and ``const`` sent as ``enum``.

        google-genai deep-merges ``http_options.extra_body`` into the REST body after
        building it, aligning key case, so the block's camelCase keys land beside
        langchain's own fields (structured output, tools) and win where both set one.
        The request's ``response_json_schema`` is replaced by a copy whose ``const``
        keywords are one-value ``enum``s (see `_const_as_enum`); the bound schema is
        not changed, and replies still parse against the original output class.

        Args:
            messages (`list[BaseMessage]`):
                The messages being sent.
            **kwargs:
                Call options langchain folds into the request, including a caller's
                ``http_options`` (`dict` or `HttpOptions`), whose ``extra_body`` the block
                is merged into.

        Returns:
            `dict`:
                Keyword arguments for google-genai's ``generate_content`` call.
        """
        if self.block:
            options = kwargs.get("http_options")
            if options is None:
                options = HttpOptions()
            elif isinstance(options, dict):
                options = HttpOptions(**options)
            else:
                options = options.model_copy()
            extra_body = copy.deepcopy(options.extra_body or {})
            options.extra_body = merge_request_body(extra_body, self.block)
            kwargs["http_options"] = options
        request = super()._prepare_request(messages, **kwargs)
        config = request["config"]
        schema = config.response_json_schema
        if isinstance(schema, dict):
            rewritten = _const_as_enum(schema)
            if rewritten != schema:
                request["config"] = config.model_copy(update={"response_json_schema": rewritten})
        return request

    def invoke(self, input, config=None, **kwargs):
        return self.normalize_reply(super().invoke(input, config, **kwargs))

    def normalize_reply(self, response):
        """Apply the reply normalization `invoke` uses; streamed replies reuse it."""
        return normalize_content(response)


class GoogleClient(BaseLLMClient):
    """Client for Google Gemini models.

    With a ``base_url``, the model's httpx client does not follow redirects, so the
    key goes only to that URL. ``http_client`` / ``http_async_client`` replace the SDK's
    httpx clients; beside a ``base_url`` they must not follow redirects.
    """

    def __init__(self, model: str, base_url: Optional[str] = None, **kwargs):
        super().__init__(model, base_url, **kwargs)

    def get_llm(self) -> Any:
        """Return configured ChatGoogleGenerativeAI instance.

        Returns:
            `NormalizedChatGoogleGenerativeAI`:
                The chat model for `self.model`, on the given ``base_url`` and host clients.

        Raises:
            `TypeError`:
                Raised when a given ``http_client`` / ``http_async_client`` is not an httpx
                client the SDK takes.
            `ValueError`:
                Raised when a given client follows redirects beside a ``base_url``.
        """
        self.warn_if_unknown_model()
        llm_kwargs: dict[str, Any] = {"model": self.model}

        if self.base_url:
            llm_kwargs["base_url"] = self.base_url
            # The SDK sends the key in x-goog-api-key, which httpx re-sends on a
            # cross-origin redirect.
            llm_kwargs["client_args"] = {"follow_redirects": False}

        for key in ("timeout", "max_retries", "callbacks"):
            if key in self.kwargs:
                llm_kwargs[key] = self.kwargs[key]
        for key in _HTTP_CLIENT_KWARGS:
            value = self.kwargs.get(key)
            if value is None:
                continue
            _check_http_client_type(key, value)
            if self.base_url:
                check_no_redirects(key, value)
            llm_kwargs[key] = value
        # langchain-google-genai sends temperature=0.7 unless it is None; a temperature
        # comes only from the block's generationConfig.
        llm_kwargs["temperature"] = None

        # Unified api_key maps to provider-specific google_api_key
        google_api_key = self.kwargs.get("api_key") or self.kwargs.get("google_api_key")
        if google_api_key:
            llm_kwargs["google_api_key"] = google_api_key

        return NormalizedChatGoogleGenerativeAI(**llm_kwargs, block=self.kwargs.get("block") or {})

    def validate_model(self) -> bool:
        """Validate model for Google."""
        return validate_model("google", self.model)
