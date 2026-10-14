"""Provider blocks of ``llm_config``: request-body fields checked against the API yian calls.

Each block mirrors the request body of one provider API with the same names, nesting, and
allowed values, and yian deep-merges it into that body as written. Fields that yian builds
itself (model, messages, tools, structured output, streaming, conversation state) are
reserved: setting one is a load error that says so. Error messages name the block and the
key path and never echo values.
"""

from __future__ import annotations

import typing
from collections.abc import Collection, Mapping
from typing import Any, ClassVar, Dict, Literal, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationInfo,
    model_serializer,
    model_validator,
)
from typing_extensions import Annotated


class _Block(BaseModel):
    """Base for blocks and their nested objects.

    Unknown keys, coercion (e.g. ``"5"`` for an integer), and non-finite numbers are
    refused, and dumps leave out unset (``None``) fields at every level. A class that
    sets ``api`` is a top-level block: it checks the raw mapping before field validation
    and reports reserved and unknown key paths in its own words.
    """

    model_config = ConfigDict(
        extra="forbid",
        hide_input_in_errors=True,
        allow_inf_nan=False,
        strict=True,
    )

    # Provider name used in messages when the validation context names no block.
    default_block: ClassVar[str] = ""
    # How an unknown key is described, e.g. "is not a field of the X request".
    api: ClassVar[str] = ""
    # Dotted key path -> what yian builds there; setting one is an error.
    reserved: ClassVar[Dict[str, str]] = {}

    @model_validator(mode="before")
    @classmethod
    def _check_keys(cls, data: Any, info: ValidationInfo) -> Any:
        """Refuse reserved and unknown key paths, naming each one and what is allowed."""
        if not cls.api or not isinstance(data, Mapping):
            return data
        context = info.context if isinstance(info.context, Mapping) else {}
        block = context.get("block") or cls.default_block
        problems: list[str] = []
        cls._collect_problems(cls, data, block, "", problems)
        if problems:
            raise ValueError("; ".join(problems))
        return data

    @classmethod
    def _collect_problems(
        cls, model: type[_Block], data: Mapping, block: str, prefix: str, problems: list[str]
    ) -> None:
        """Append a message for each reserved or unknown key of ``data``, recursing into parts."""
        fields = {field.alias or name: field for name, field in model.model_fields.items()}
        for key, value in data.items():
            path = f"{prefix}{key}"
            if path in cls.reserved:
                problems.append(
                    f"`{block}.{path}` is built by yian ({cls.reserved[path]}); remove it"
                )
            elif key not in fields:
                problems.append(
                    f"`{block}.{path}` {cls.api}; allowed: {', '.join(sorted(fields))}"
                )
            else:
                part = _part_type(fields[key].annotation)
                if part is not None and isinstance(value, Mapping):
                    cls._collect_problems(part, value, block, f"{path}.", problems)

    @model_serializer(mode="wrap")
    def _drop_unset(self, handler):
        """Dump without ``None`` fields, so an unset field is never recorded or sent."""
        data = handler(self)
        if isinstance(data, dict):
            return {key: value for key, value in data.items() if value is not None}
        return data


def _part_type(annotation: Any) -> Optional[type[_Block]]:
    """Return the nested ``_Block`` class inside a field annotation, if there is one."""
    for candidate in (annotation, *typing.get_args(annotation)):
        if isinstance(candidate, type) and issubclass(candidate, _Block):
            return candidate
    return None


class OpenAIReasoning(_Block):
    """The ``reasoning`` object of an OpenAI Responses API request."""

    effort: Optional[Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"]] = None
    summary: Optional[Literal["auto", "concise", "detailed"]] = None


class OpenAIText(_Block):
    """The ``text`` object of an OpenAI Responses API request (``format`` is reserved)."""

    verbosity: Optional[Literal["low", "medium", "high"]] = None


_MetadataKey = Annotated[str, StringConstraints(max_length=64)]
_MetadataValue = Annotated[str, StringConstraints(max_length=512)]


class _OpenAIResponsesShared(_Block):
    """Responses API request fields that every OpenAI transport yian uses accepts."""

    default_block: ClassVar[str] = "openai"
    reserved: ClassVar[Dict[str, str]] = {
        "model": "llm_config.model",
        "input": "messages",
        "instructions": "system prompt",
        "prompt": "messages",
        "tools": "tools",
        "tool_choice": "tools",
        "parallel_tool_calls": "tools",
        "text.format": "structured output",
        "stream": "streaming",
        "stream_options": "streaming",
        "previous_response_id": "conversation state",
        "conversation": "conversation state",
        "store": "conversation state",
        "include": "response items",
        "background": "request lifecycle",
    }

    max_tool_calls: Optional[int] = Field(default=None, ge=1)
    metadata: Optional[Dict[_MetadataKey, _MetadataValue]] = Field(default=None, max_length=16)
    prompt_cache_key: Optional[str] = None
    prompt_cache_retention: Optional[Literal["in_memory", "24h"]] = None
    reasoning: Optional[OpenAIReasoning] = None
    safety_identifier: Optional[str] = Field(default=None, max_length=64)
    service_tier: Optional[
        Literal["auto", "default", "flex", "scale", "priority", "fast", "ultrafast"]
    ] = None
    text: Optional[OpenAIText] = None
    top_logprobs: Optional[int] = Field(default=None, ge=0, le=20)
    top_p: Optional[float] = Field(default=None, ge=0, le=1)
    truncation: Optional[Literal["auto", "disabled"]] = None
    user: Optional[str] = None


class OpenAIResponsesBlock(_OpenAIResponsesShared):
    """The ``openai`` block: the request body of the OpenAI Responses API.

    Mirrors ``POST /v1/responses``
    (https://platform.openai.com/docs/api-reference/responses/create), checked
    2026-10-14 against that reference and the ``openai`` SDK's
    ``openai.types.responses.response_create_params``.

    Example:
        ```yaml
        openai:
          reasoning: {effort: high, summary: auto}
          text: {verbosity: low}
          max_output_tokens: 4096
        ```
    """

    api: ClassVar[str] = "is not a field of the OpenAI Responses API request"

    max_output_tokens: Optional[int] = Field(default=None, ge=1)
    temperature: Optional[float] = Field(default=None, ge=0, le=2)


class OpenAISubscriptionBlock(_OpenAIResponsesShared):
    """The ``openai`` block when the call goes to the ChatGPT subscription endpoint.

    The endpoint accepts a subset of the Responses API request: everything in
    `OpenAIResponsesBlock` except ``temperature`` and ``max_output_tokens``, which it
    refuses.
    """

    api: ClassVar[str] = (
        "is not accepted by the ChatGPT subscription endpoint (a subset of the Responses API)"
    )


PROVIDER_BLOCKS: Dict[str, type[_Block]] = {
    "openai": OpenAIResponsesBlock,
}


def block_schema(
    provider: str, connection: Optional[Mapping] = None, api_key: Any = None
) -> type[_Block]:
    """Return the block class for the API a call to ``provider`` actually reaches.

    Args:
        provider (`str`):
            A key of `PROVIDER_BLOCKS`, e.g. ``"openai"``.
        connection (`Mapping`, *optional*, defaults to `None`):
            The resolved OpenAI connection, whose ``method`` says how the call is
            authenticated. `None` means no connection.
        api_key (`Any`, *optional*, defaults to `None`):
            The host-supplied key. A key sends the call to the Responses API even when
            the connection is a subscription.

    Returns:
        `type[_Block]`:
            `OpenAISubscriptionBlock` for an ``openai`` call on a ``subscription``
            connection without ``api_key``; otherwise ``PROVIDER_BLOCKS[provider]``.

    Raises:
        `KeyError`:
            Raised when ``provider`` has no block.
    """
    if (
        provider == "openai"
        and isinstance(connection, Mapping)
        and connection.get("method") == "subscription"
        and api_key is None
    ):
        return OpenAISubscriptionBlock
    return PROVIDER_BLOCKS[provider]


def merge_request_body(
    body: dict, block: Mapping, *, top_level: Optional[Collection[str]] = None
) -> dict:
    """Deep-merge a provider block into a request body, in place.

    Mappings merge key by key; lists and scalars from ``block`` replace the body's value
    whole; a ``None`` in ``block`` means unset and leaves the body alone. ``block`` is
    copied, never mutated or shared.

    Args:
        body (`dict`):
            The request body (or client keyword arguments) yian built. Updated in place.
        block (`Mapping`):
            The provider block, as written or dumped.
        top_level (`Collection[str]`, *optional*, defaults to `None`):
            Keys the client accepts at the top level. When given, a block key outside it
            is merged into ``body["extra_body"]`` instead. `None` merges every key at the
            top level.

    Returns:
        `dict`:
            ``body``.

    Example:
        ```python
        body = {"text": {"format": {"type": "text"}}}
        merge_request_body(body, {"text": {"verbosity": "low"}})
        # {"text": {"format": {"type": "text"}, "verbosity": "low"}}
        ```
    """
    for key, value in block.items():
        if top_level is not None and key not in top_level:
            if not isinstance(body.get("extra_body"), dict):
                body["extra_body"] = {}
            _deep_merge(body["extra_body"], {key: value})
        else:
            _deep_merge(body, {key: value})
    return body


def _deep_merge(target: dict, source: Mapping) -> None:
    """Merge ``source`` into ``target`` recursively, copying everything taken from ``source``."""
    for key, value in source.items():
        if value is None:
            continue
        if isinstance(value, Mapping):
            current = target.get(key)
            if not isinstance(current, dict):
                current = target[key] = {}
            _deep_merge(current, value)
        else:
            target[key] = _copy(value)


def _copy(value: Any) -> Any:
    """Copy a block value so the body never shares a container with the block."""
    if isinstance(value, Mapping):
        return {key: _copy(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_copy(item) for item in value]
    return value
