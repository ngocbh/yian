import builtins
import copy
import json
import os
import typing
from typing import Any, Optional

import httpx
import pydantic
from google.genai import Client
from google.genai.types import HttpOptions
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import Field, SecretStr, model_validator

from .base_client import (
    HEADERS_LEAK,
    BaseLLMClient,
    check_connection_headers,
    check_no_redirects,
    normalize_content,
    warn_once,
)
from .provider_blocks import merge_request_body
from .validators import validate_model

_HTTP_CLIENT_KWARGS = ("http_client", "http_async_client")
# google-genai's switches to Vertex AI; GOOGLE_GENAI_USE_ENTERPRISE is the newer name.
_VERTEX_SWITCHES = ("GOOGLE_GENAI_USE_VERTEXAI", "GOOGLE_GENAI_USE_ENTERPRISE")


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
    """Return a copy of a JSON schema with string and number ``const``s as one-value ``enum``s.

    Gemini's ``responseJsonSchema`` supports only a subset of JSON Schema, which lists
    ``enum`` (for strings and numbers) but not ``const`` (google-genai
    ``GenerateContentConfig.response_json_schema``). ``{"const": v}`` and ``{"enum": [v]}``
    accept the same values, so the rewrite is exact. A boolean or null ``const`` is left as
    written, since the list does not cover such an ``enum`` either.

    Args:
        schema (`dict[str, Any]`):
            The schema to rewrite. It is not changed.

    Returns:
        `dict[str, Any]`:
            A deep copy of `schema` where the root and every subschema under
            ``properties``, ``$defs``, ``items``, ``prefixItems``, ``additionalProperties``,
            ``anyOf``, ``oneOf`` and ``allOf`` holding a string or number ``const: v`` has
            ``enum: [v]`` instead. A subschema whose existing ``enum`` holds ``v`` (compared as JSON
            values) gets ``enum: [v]``; one whose ``enum`` lacks ``v`` is left as written.
    """
    result = copy.deepcopy(schema)
    stack = [result]
    while stack:
        node = stack.pop()
        if not isinstance(node, dict):
            continue
        value = node.get("const")
        # `bool` is a subclass of `int`, but a JSON boolean is not a number.
        if isinstance(value, (str, int, float)) and not isinstance(value, bool):
            # Compare as JSON, where `true` and `1` differ; Python has `True == 1`.
            wanted = json.dumps(value, sort_keys=True)
            members = node.get("enum")
            if members is None or any(
                json.dumps(member, sort_keys=True) == wanted for member in members
            ):
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
        connection_headers (`dict[str, SecretStr]`, *optional*, defaults to `{}`):
            Headers sent with every request, added to the genai client's headers. Values
            are secrets: unlike ``additional_headers``, this field is left out of ``repr``,
            dumps, langchain serialization and the request metadata.
    """

    block: dict[str, Any] = Field(default_factory=dict, exclude=True, repr=False)
    http_client: Any = Field(default=None, exclude=True, repr=False)
    http_async_client: Any = Field(default=None, exclude=True, repr=False)
    connection_headers: dict[str, SecretStr] = Field(
        default_factory=dict, exclude=True, repr=False
    )

    # Declared above `_use_http_clients`: pydantic runs after-validators in declaration order,
    # so the caller's httpx clients go onto the client this one may rebuild.
    @model_validator(mode="after")
    def _use_developer_api(self) -> "NormalizedChatGoogleGenerativeAI":
        """Rebuild the genai client on the Gemini Developer API if the environment switched it.

        Returns:
            `NormalizedChatGoogleGenerativeAI`:
                This model, its genai client calling the Gemini Developer API with the
                model's key. A model built with an explicit ``vertexai=True`` is left alone.
        """
        # langchain builds the Developer API client without `vertexai`, so google-genai reads
        # GOOGLE_GENAI_USE_ENTERPRISE / GOOGLE_GENAI_USE_VERTEXAI itself and switches to
        # Vertex AI express mode; langchain has no slot to pass `vertexai=False`.
        if self._use_vertexai or not self.client._api_client.vertexai:
            return self
        api = self.client._api_client
        # google-genai sets Content-Type and x-goog-api-key from its own constructor arguments,
        # and patch_http_options lays caller headers over those defaults, so a copied one would
        # override them.
        headers = {
            name: value for name, value in (api._http_options.headers or {}).items()
            if name.lower() not in ("x-goog-api-key", "content-type")
        }
        base = self.base_url["api_endpoint"] if isinstance(self.base_url, dict) else self.base_url
        options = HttpOptions(
            base_url=base,
            api_version=self.api_version,
            headers=headers,
            client_args=self.client_args,
            async_client_args=self.client_args,
        )
        # langchain's cleanup object closes a replaced client's own transports when it is
        # dropped; the caller's httpx clients are attached only afterwards, so they stay
        # open. Versions before 4.3.7 have none, and the replaced client is left to its
        # transports' own `__del__`. Its type is reused because the class is private.
        cleanup = getattr(self, "_client_cleanup", None)
        self.client = Client(
            vertexai=False, api_key=self.google_api_key.get_secret_value(), http_options=options
        )
        if cleanup is not None:
            self._client_cleanup = type(cleanup)(self.client)
        return self

    # Declared after `_use_developer_api`, which may rebuild the genai client.
    @model_validator(mode="after")
    def _add_connection_headers(self) -> "NormalizedChatGoogleGenerativeAI":
        """Add `connection_headers` to the headers the genai client sends with every request.

        Returns:
            `NormalizedChatGoogleGenerativeAI`:
                This model.
        """
        if self.connection_headers:
            # google-genai lays each request's own headers over these, so they reach every
            # call; langchain's `additional_headers` would also copy them into
            # `default_metadata`, a model field.
            options = self.client._api_client._http_options
            options.headers = {
                **(options.headers or {}),
                **{
                    name: value.get_secret_value()
                    for name, value in self.connection_headers.items()
                },
            }
        return self

    @model_validator(mode="after")
    def _use_http_clients(self) -> "NormalizedChatGoogleGenerativeAI":
        """Hand the given httpx clients to the model's genai client.

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

    The model always calls the Gemini Developer API with the key yian resolved, even when
    ``GOOGLE_GENAI_USE_VERTEXAI`` or ``GOOGLE_GENAI_USE_ENTERPRISE`` is set. With a
    ``base_url`` or ``connection_headers``, the model's httpx clients do not follow
    redirects, so the key and headers go only to that URL. ``http_client`` /
    ``http_async_client`` replace the SDK's httpx clients; beside a ``base_url`` or
    ``connection_headers`` they must not follow redirects.
    """

    def __init__(self, model: str, base_url: Optional[str] = None, **kwargs):
        super().__init__(model, base_url, **kwargs)

    def get_llm(self) -> Any:
        """Return configured ChatGoogleGenerativeAI instance.

        ``connection_headers`` (`dict[str, str | SecretStr]`, a keyword argument) are sent
        with every request; such a model never follows a redirect.

        Returns:
            `NormalizedChatGoogleGenerativeAI`:
                The chat model for `self.model`, on the given ``base_url`` and host clients.

        Raises:
            `TypeError`:
                Raised when a given ``http_client`` / ``http_async_client`` is not an httpx
                client the SDK takes.
            `ValueError`:
                Raised when a connection header is refused (`check_connection_headers`), a
                given client follows redirects beside a ``base_url`` or connection
                headers, or no API key is given or found in the environment.

        Warns:
            `UserWarning`:
                Once per process for each of ``GOOGLE_GENAI_USE_VERTEXAI`` and
                ``GOOGLE_GENAI_USE_ENTERPRISE`` set to a non-empty value, naming the variable:
                the model ignores it and calls the Gemini Developer API.
        """
        self.warn_if_unknown_model()
        # Before the build, so a setup that relied on Vertex AI credentials and now fails
        # for want of an API key sees why.
        for name in _VERTEX_SWITCHES:
            if os.environ.get(name):
                warn_once(
                    f"yian ignores {name}: the google provider always calls the Gemini "
                    "Developer API with the key yian resolved, and Vertex AI is not supported. "
                    "Unset the variable to silence this."
                )
        llm_kwargs: dict[str, Any] = {"model": self.model}

        if self.base_url:
            llm_kwargs["base_url"] = self.base_url
        if self.kwargs.get("connection_headers"):
            llm_kwargs["connection_headers"] = check_connection_headers(
                self.kwargs["connection_headers"], "google"
            )
        # The SDK sends the key in x-goog-api-key, and the connection headers beside it,
        # which httpx re-sends on a cross-origin redirect.
        no_redirects = bool(self.base_url or llm_kwargs.get("connection_headers"))
        leak = {"reason": HEADERS_LEAK} if llm_kwargs.get("connection_headers") else {}
        if no_redirects:
            llm_kwargs["client_args"] = {"follow_redirects": False}

        for key in ("timeout", "max_retries", "callbacks"):
            if key in self.kwargs:
                llm_kwargs[key] = self.kwargs[key]
        for key in _HTTP_CLIENT_KWARGS:
            value = self.kwargs.get(key)
            if value is None:
                continue
            _check_http_client_type(key, value)
            if no_redirects:
                check_no_redirects(key, value, **leak)
            llm_kwargs[key] = value
        if no_redirects and "http_async_client" not in llm_kwargs:
            # Without an httpx async client google-genai sends async calls through aiohttp
            # when it is installed, and aiohttp follows redirects by default.
            llm_kwargs["http_async_client"] = httpx.AsyncClient(follow_redirects=False)
        # langchain-google-genai sends temperature=0.7 unless it is None; a temperature
        # comes only from the block's generationConfig.
        llm_kwargs["temperature"] = None
        # Without it langchain reads GOOGLE_GENAI_USE_VERTEXAI, and its Vertex AI branch hands
        # the key to google-genai only through the environment, where an env key wins.
        llm_kwargs["vertexai"] = False

        # Unified api_key maps to provider-specific google_api_key
        google_api_key = self.kwargs.get("api_key") or self.kwargs.get("google_api_key")
        if google_api_key:
            llm_kwargs["google_api_key"] = google_api_key

        return NormalizedChatGoogleGenerativeAI(**llm_kwargs, block=self.kwargs.get("block") or {})

    def validate_model(self) -> bool:
        """Validate model for Google."""
        return validate_model("google", self.model)
