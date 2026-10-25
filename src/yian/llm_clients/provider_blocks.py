"""Provider blocks of ``llm_config``: request-body fields checked against the API yian calls.

Each block mirrors the request body of one provider API with the same names, nesting, and
allowed values, and yian deep-merges it into that body as written. Fields that yian builds
itself (model, messages, tools, structured output, streaming, conversation state) are
reserved: setting one is a load error that says so. Error messages name the block and the
key path and never echo values; the key path can include keys the author wrote in a mapping
field (e.g. a ``metadata`` key), since keys are names rather than values.
`partial_block_problems` checks a block whose leaves may be values resolved later
(`located_block_problems` also gives each problem's key path), and
`block_field_annotation` gives the simple type a value for one field must have.
"""

from __future__ import annotations

import types
import typing
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass
from typing import Annotated, Any, ClassVar, Dict, List, Literal, Optional, Union

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Discriminator,
    Field,
    StringConstraints,
    Tag,
    ValidationError,
    model_serializer,
    model_validator,
)


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

    # Block name used in messages, e.g. "qwen-cn".
    default_block: ClassVar[str] = ""
    # How an unknown key is described, e.g. "is not a supported field of the X request".
    api: ClassVar[str] = ""
    # Dotted key path -> what yian builds there; setting one is an error.
    reserved: ClassVar[Dict[str, str]] = {}

    @model_validator(mode="before")
    @classmethod
    def _check_keys(cls, data: Any) -> Any:
        """Refuse reserved and unknown key paths, naming each one and what is allowed."""
        if not cls.api or not isinstance(data, Mapping):
            return data
        problems: list[str] = []
        _key_problems(cls, data, top=cls, prefix="", problems=problems)
        if problems:
            raise ValueError("; ".join(problems))
        return data

    @model_serializer(mode="wrap")
    def _drop_unset(self, handler):
        """Dump without ``None`` fields, so an unset field is never recorded or sent."""
        data = handler(self)
        if isinstance(data, dict):
            return {key: value for key, value in data.items() if value is not None}
        return data


def _key_problems(
    model: type[_Block], data: Mapping, *, top: type[_Block], prefix: str, problems: list[str]
) -> None:
    """Append a message for each reserved or unknown key of ``data``, recursing into parts.

    ``model`` is the class ``data`` is checked against; ``top`` is the top-level block,
    which supplies the block name, the reserved paths and the unknown-key wording.
    """
    fields = {field.alias or name: field for name, field in model.model_fields.items()}
    for key, value in data.items():
        path = f"{prefix}{key}"
        problem = _bad_key(top, path, key, fields)
        if problem is not None:
            problems.append(problem)
        else:
            _value_problems(fields[key].annotation, value, top=top, path=path, problems=problems)


def _bad_key(top: type[_Block], path: str, key: str, allowed: Mapping) -> Optional[str]:
    """Return why ``key`` at ``path`` is not allowed, or `None` when it is.

    ``top`` is the top-level block, which supplies the block name, the reserved paths and
    the unknown-key wording; ``allowed`` holds the keys the enclosing object accepts.
    """
    if path in top.reserved:
        return f"`{top.default_block}.{path}` is built by yian ({top.reserved[path]}); remove it"
    if key not in allowed:
        return f"`{top.default_block}.{path}` {top.api}; allowed: {', '.join(sorted(allowed))}"
    return None


def _value_problems(
    annotation: Any, value: Any, *, top: type[_Block], path: str, problems: list[str]
) -> None:
    """Check the keys of a nested block, or of each block in a list, at ``path``.

    Blocks inside a mapping field (``Dict[str, SomeBlock]``) are not walked; none exists,
    and pydantic's own ``extra_forbidden`` error would still refuse an unknown key there.
    """
    if isinstance(value, Mapping):
        part = _nested_block(annotation, value)
        if part is not None:
            _key_problems(part, value, top=top, prefix=f"{path}.", problems=problems)
    elif isinstance(value, list):
        items = [typing.get_args(m)[0] for m in _members(annotation) if _is_list(m)]
        if items:
            item = Union[tuple(items)]
            for index, entry in enumerate(value):
                _value_problems(item, entry, top=top, path=f"{path}[{index}]", problems=problems)


def _members(annotation: Any) -> list[Any]:
    """Return the types an annotation allows, through ``Optional``, unions and ``Annotated``."""
    origin = typing.get_origin(annotation)
    if origin is Annotated:
        return _members(typing.get_args(annotation)[0])
    if origin is Union or origin is types.UnionType:
        return [member for arg in typing.get_args(annotation) for member in _members(arg)]
    return [annotation]


def _is_list(annotation: Any) -> bool:
    """Return whether ``annotation`` is a parametrized list type."""
    return typing.get_origin(annotation) is list and bool(typing.get_args(annotation))


def _nested_block(annotation: Any, value: Mapping) -> Optional[type[_Block]]:
    """Return the block class to check the mapping ``value`` against, if one can be chosen.

    A union of several block classes is told apart by their ``type`` field; when ``value``
    names no variant, there is none, and field validation reports the problem instead.
    """
    blocks = [m for m in _members(annotation) if isinstance(m, type) and issubclass(m, _Block)]
    if len(blocks) <= 1:
        return blocks[0] if blocks else None
    tag = value.get("type")
    for candidate in blocks:
        field = candidate.model_fields.get("type")
        if field is not None and tag in typing.get_args(field.annotation):
            return candidate
    return None


def _by_type(*variants: type[_Block]) -> Any:
    """Return a union of block classes discriminated by their ``type`` literal.

    An unknown or missing ``type`` fails with a message listing the accepted ones, without
    echoing the value as pydantic's own tag error would.
    """
    tags = {typing.get_args(v.model_fields["type"].annotation)[0]: v for v in variants}

    def tag_of(value: Any) -> Optional[str]:
        tag = value.get("type") if isinstance(value, Mapping) else getattr(value, "type", None)
        return tag if isinstance(tag, str) and tag in tags else None

    return Annotated[
        Union[tuple(Annotated[variant, Tag(tag)] for tag, variant in tags.items())],
        Discriminator(
            tag_of,
            custom_error_type="unknown_type",
            custom_error_message=f"`type` must be one of: {', '.join(sorted(tags))}",
        ),
    ]


class OpenAIReasoning(_Block):
    """The ``reasoning`` object of an OpenAI Responses API request."""

    context: Optional[Literal["auto", "current_turn", "all_turns"]] = None
    effort: Optional[Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"]] = None
    # The SDK also types `mode` as any string; these are the values it documents.
    mode: Optional[Literal["standard", "pro"]] = None
    summary: Optional[Literal["auto", "concise", "detailed"]] = None


class OpenAIText(_Block):
    """The ``text`` object of an OpenAI Responses API request (``format`` is reserved)."""

    verbosity: Optional[Literal["low", "medium", "high"]] = None


class OpenAIAccessPrograms(_Block):
    """The ``access_programs`` object of an OpenAI Responses API request."""

    cyber: Optional[Literal["standard", "daybreak_blue", "daybreak_red"]] = None


class OpenAIModerationMode(_Block):
    """How moderation treats the input or the output: score it, or block it."""

    mode: Literal["score", "block"]


class OpenAIModerationPolicy(_Block):
    """The ``moderation.policy`` object of an OpenAI Responses API request."""

    input: Optional[OpenAIModerationMode] = None
    output: Optional[OpenAIModerationMode] = None


class OpenAIModeration(_Block):
    """The ``moderation`` object of an OpenAI Responses API request."""

    model: str
    policy: Optional[OpenAIModerationPolicy] = None


class OpenAIPromptCacheOptions(_Block):
    """The ``prompt_cache_options`` object of an OpenAI Responses API request.

    ``comparison_response_id`` is left out because it names one earlier response, and
    ``prewarm`` because it suppresses the reply yian reads.
    """

    mode: Optional[Literal["implicit", "explicit"]] = None
    ttl: Optional[Literal["30m"]] = None


_MetadataKey = Annotated[str, StringConstraints(max_length=64)]
_MetadataValue = Annotated[str, StringConstraints(max_length=512)]
_Metadata = Annotated[Dict[_MetadataKey, _MetadataValue], Field(max_length=16)]


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
        "context_management": "conversation state",
    }

    access_programs: Optional[OpenAIAccessPrograms] = None
    max_tool_calls: Optional[int] = Field(default=None, ge=1)
    metadata: Optional[_Metadata] = None
    moderation: Optional[OpenAIModeration] = None
    prompt_cache_key: Optional[str] = None
    prompt_cache_options: Optional[OpenAIPromptCacheOptions] = None
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
    2026-10-13 against that reference and the ``openai`` SDK's
    ``openai.types.responses.response_create_params`` (``openai`` 3.22.1): every body
    keyword of ``Responses.create`` is a field here or reserved.

    Example:
        ```yaml
        openai:
          reasoning: {effort: high, summary: auto}
          text: {verbosity: low}
          max_output_tokens: 4096
        ```
    """

    api: ClassVar[str] = "is not a supported field of the OpenAI Responses API request"

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


# Chat Completions body fields every vendor below reserves, with what yian builds there.
_CHAT_RESERVED: Dict[str, str] = {
    "model": "llm_config.model",
    "messages": "messages",
    "tools": "tools",
    "tool_choice": "tools",
    "parallel_tool_calls": "tools",
    "response_format": "structured output",
    "stream": "streaming",
    "stream_options": "streaming",
    "n": "one choice per call",
}

_Stop4 = Union[str, Annotated[List[str], Field(max_length=4)]]


def _token_id_key(key: Any) -> Any:
    """Turn an integer token id, as YAML reads ``{50256: -100}``, into the string JSON sends."""
    if isinstance(key, int) and not isinstance(key, bool) and key >= 0:
        return str(key)
    return key


# A ``logit_bias`` key: a token id written as a non-negative integer or a digit string.
_TokenId = Annotated[str, BeforeValidator(_token_id_key), StringConstraints(pattern=r"^[0-9]+$")]


class _ChatSampling(_Block):
    """``temperature`` in [0, 2] and ``top_p`` in [0, 1], as most vendors take them."""

    temperature: Optional[float] = Field(default=None, ge=0, le=2)
    top_p: Optional[float] = Field(default=None, ge=0, le=1)


class _ChatPenalties(_Block):
    """``frequency_penalty`` and ``presence_penalty``, each in [-2, 2]."""

    frequency_penalty: Optional[float] = Field(default=None, ge=-2, le=2)
    presence_penalty: Optional[float] = Field(default=None, ge=-2, le=2)


class _ChatLength(_Block):
    """``max_tokens`` (older) and ``max_completion_tokens`` output limits, each at least 1."""

    max_completion_tokens: Optional[int] = Field(default=None, ge=1)
    max_tokens: Optional[int] = Field(default=None, ge=1)


class _ChatLogprobs(_Block):
    """``logprobs`` and ``top_logprobs`` in [0, 20]."""

    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = Field(default=None, ge=0, le=20)


class XAIChatBlock(_ChatSampling, _ChatPenalties, _ChatLength, _ChatLogprobs):
    """The ``xai`` block: the request body of the xAI Chat Completions API.

    Mirrors ``POST /v1/chat/completions``
    (https://docs.x.ai/developers/rest-api-reference/inference/chat-completions), checked
    2026-10-13. ``logit_bias`` is left out because the reference marks it unsupported, and
    ``reasoning_effort`` is any string because the accepted values depend on the model.

    Example:
        ```yaml
        xai:
          reasoning_effort: high
          max_completion_tokens: 8192
        ```
    """

    default_block: ClassVar[str] = "xai"
    api: ClassVar[str] = "is not a supported field of the xAI Chat Completions request"
    reserved: ClassVar[Dict[str, str]] = {
        **_CHAT_RESERVED,
        "deferred": "request lifecycle",
        "search_parameters": "tools: server-side search",
        "web_search_options": "tools: server-side search",
    }

    prompt_cache_key: Optional[str] = None
    reasoning_effort: Optional[str] = None
    safety_identifier: Optional[str] = None
    seed: Optional[int] = None
    service_tier: Optional[Literal["default", "priority", "fast"]] = None
    stop: Optional[Annotated[List[str], Field(max_length=4)]] = None
    top_logprobs: Optional[int] = Field(default=None, ge=0, le=8)
    user: Optional[str] = None


class GLMThinking(_Block):
    """The ``thinking`` object of a Z.AI GLM chat request."""

    type: Optional[Literal["enabled", "disabled"]] = None
    clear_thinking: Optional[bool] = None


class GLMChatBlock(_Block):
    """The ``glm`` block: the request body of the Z.AI GLM Chat Completion API.

    Mirrors ``POST /api/paas/v4/chat/completions``
    (https://docs.z.ai/api-reference/llm/chat-completion), checked 2026-10-13.

    Example:
        ```yaml
        glm:
          thinking: {type: enabled}
          reasoning_effort: high
          temperature: 0.6
        ```
    """

    default_block: ClassVar[str] = "glm"
    api: ClassVar[str] = "is not a supported field of the Z.AI GLM Chat Completion request"
    reserved: ClassVar[Dict[str, str]] = {**_CHAT_RESERVED, "tool_stream": "streaming"}

    do_sample: Optional[bool] = None
    max_tokens: Optional[int] = Field(default=None, ge=1, le=131072)
    reasoning_effort: Optional[
        Literal["max", "xhigh", "high", "medium", "low", "minimal", "none"]
    ] = None
    stop: Optional[Annotated[List[str], Field(max_length=4)]] = None
    temperature: Optional[float] = Field(default=None, ge=0, le=1)
    thinking: Optional[GLMThinking] = None
    top_p: Optional[float] = Field(default=None, ge=0.01, le=1)
    user_id: Optional[str] = Field(default=None, min_length=6, max_length=128)


class GLMCNChatBlock(GLMChatBlock):
    """The ``glm-cn`` block: `GLMChatBlock` for the mainland China endpoint."""

    default_block: ClassVar[str] = "glm-cn"


class MiniMaxThinking(_Block):
    """The ``thinking`` object of a MiniMax chat request."""

    type: Optional[Literal["disabled", "adaptive"]] = None


class MiniMaxChatBlock(_ChatSampling, _ChatLength):
    """The ``minimax`` block: the request body of MiniMax's OpenAI-compatible API.

    Mirrors the Chat Completions parameters MiniMax lists
    (https://platform.minimax.io/docs/api-reference/text-openai-api), checked 2026-10-13.
    MiniMax ignores the other OpenAI parameters, so they are not fields here.

    Example:
        ```yaml
        minimax:
          thinking: {type: adaptive}
          reasoning_effort: high
        ```
    """

    default_block: ClassVar[str] = "minimax"
    api: ClassVar[str] = "is not a supported field of the MiniMax Chat Completions request"
    reserved: ClassVar[Dict[str, str]] = {
        **_CHAT_RESERVED,
        "reasoning_split": "keeps reasoning out of the message content",
    }

    reasoning_effort: Optional[Literal["low", "medium", "high", "xhigh", "max"]] = None
    service_tier: Optional[Literal["standard", "priority"]] = None
    thinking: Optional[MiniMaxThinking] = None


class MiniMaxCNChatBlock(MiniMaxChatBlock):
    """The ``minimax-cn`` block: `MiniMaxChatBlock` for the mainland China endpoint."""

    default_block: ClassVar[str] = "minimax-cn"


class DeepSeekThinking(_Block):
    """The ``thinking`` object of a DeepSeek chat request."""

    type: Optional[Literal["enabled", "disabled"]] = None


class DeepSeekChatBlock(_ChatLogprobs):
    """The ``deepseek`` block: the request body of the DeepSeek Chat Completion API.

    Mirrors ``POST /chat/completions``
    (https://api-docs.deepseek.com/api/create-chat-completion), checked 2026-10-13.
    ``frequency_penalty`` and ``presence_penalty`` are left out because the reference
    says they no longer take effect.

    Example:
        ```yaml
        deepseek:
          thinking: {type: enabled}
          reasoning_effort: max
        ```
    """

    default_block: ClassVar[str] = "deepseek"
    api: ClassVar[str] = "is not a supported field of the DeepSeek Chat Completion request"
    reserved: ClassVar[Dict[str, str]] = dict(_CHAT_RESERVED)

    max_tokens: Optional[int] = Field(default=None, ge=1, le=393216)
    reasoning_effort: Optional[Literal["none", "low", "high", "max"]] = None
    stop: Optional[Union[str, Annotated[List[str], Field(max_length=16)]]] = None
    temperature: Optional[float] = Field(default=None, ge=0, le=2)
    thinking: Optional[DeepSeekThinking] = None
    top_p: Optional[float] = Field(default=None, gt=0, le=1)
    user_id: Optional[
        Annotated[str, StringConstraints(max_length=512, pattern=r"^[a-zA-Z0-9_-]+$")]
    ] = None


class QwenSearchOptions(_Block):
    """The ``search_options`` object of a Qwen chat request."""

    enable_search_extension: Optional[bool] = None
    forced_search: Optional[bool] = None
    search_strategy: Optional[Literal["turbo", "max", "agent"]] = None


class QwenChatBlock(_ChatLength):
    """The ``qwen`` block: the request body of Model Studio's OpenAI-compatible chat.

    Mirrors Alibaba Cloud Model Studio's OpenAI-compatible Chat Completions request
    (https://www.alibabacloud.com/help/en/model-studio/qwen-api-via-openai-chat-completions),
    checked 2026-10-13. ``thinking`` and ``clear_thinking`` are for the MiniMax and GLM
    models Model Studio also serves.

    Example:
        ```yaml
        qwen:
          enable_thinking: true
          thinking_budget: 4096
          top_k: 20
        ```
    """

    default_block: ClassVar[str] = "qwen"
    api: ClassVar[str] = "is not a supported field of the Model Studio Chat Completions request"
    reserved: ClassVar[Dict[str, str]] = {
        **_CHAT_RESERVED,
        "tool_stream": "streaming",
        "modalities": "output format",
        "audio": "output format",
        "enable_code_interpreter": "tools: server-side code interpreter",
    }

    clear_thinking: Optional[bool] = None
    enable_search: Optional[bool] = None
    enable_thinking: Optional[bool] = None
    logprobs: Optional[bool] = None
    preserve_thinking: Optional[bool] = None
    presence_penalty: Optional[float] = Field(default=None, ge=-2, le=2)
    reasoning_effort: Optional[
        Literal["none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra"]
    ] = None
    repetition_penalty: Optional[float] = Field(default=None, gt=0)
    search_options: Optional[QwenSearchOptions] = None
    seed: Optional[int] = Field(default=None, ge=0, le=2**31 - 1)
    # A stop list holds strings or token ids, never both.
    stop: Optional[Union[str, List[str], List[int]]] = None
    temperature: Optional[float] = Field(default=None, ge=0, lt=2)
    thinking: Optional[MiniMaxThinking] = None
    thinking_budget: Optional[int] = Field(default=None, ge=1)
    top_k: Optional[int] = Field(default=None, ge=0)
    top_logprobs: Optional[int] = Field(default=None, ge=0, le=5)
    top_p: Optional[float] = Field(default=None, gt=0, le=1)
    use_multichannel: Optional[bool] = None
    vl_high_resolution_images: Optional[bool] = None


class QwenCNChatBlock(QwenChatBlock):
    """The ``qwen-cn`` block: `QwenChatBlock` for the mainland China endpoint."""

    default_block: ClassVar[str] = "qwen-cn"


class OpenRouterReasoning(_Block):
    """The ``reasoning`` object of an OpenRouter chat request."""

    effort: Optional[Literal["max", "xhigh", "high", "medium", "low", "minimal", "none"]] = None
    enabled: Optional[bool] = None
    exclude: Optional[bool] = None
    max_tokens: Optional[int] = Field(default=None, ge=0)
    summary: Optional[Literal["auto", "concise", "detailed"]] = None


class OpenRouterCacheControl(_Block):
    """The top-level ``cache_control`` object of an OpenRouter chat request."""

    type: Optional[Literal["ephemeral"]] = None
    ttl: Optional[Literal["5m", "1h"]] = None


class OpenRouterPromptCacheOptions(_Block):
    """The ``prompt_cache_options`` object of an OpenRouter chat request."""

    mode: Optional[Literal["explicit"]] = None
    ttl: Optional[str] = None


class OpenRouterChatBlock(_ChatSampling, _ChatPenalties, _ChatLength, _ChatLogprobs):
    """The ``openrouter`` block: the request body of the OpenRouter Chat Completions API.

    Mirrors ``POST /api/v1/chat/completions``
    (https://openrouter.ai/docs/api/api-reference/chat/create-a-chat-completion), with
    bounds from https://openrouter.ai/docs/api/reference/parameters and the ``reasoning``
    fields from https://openrouter.ai/docs/guides/best-practices/reasoning-tokens,
    checked 2026-10-13. ``provider`` (routing preferences) is passed through as written.
    ``session_id`` and ``trace`` are left out because they identify one conversation or
    trace, so a value fixed in the flow would group every run together.

    Example:
        ```yaml
        openrouter:
          reasoning: {effort: high}
          provider: {order: [anthropic], allow_fallbacks: false}
        ```
    """

    default_block: ClassVar[str] = "openrouter"
    api: ClassVar[str] = "is not a supported field of the OpenRouter Chat Completions request"
    reserved: ClassVar[Dict[str, str]] = {
        **_CHAT_RESERVED,
        "plugins": "tools",
        "web_search_options": "tools: server-side search",
        "stop_server_tools_when": "tools",
        "deferred_tools": "tools",
        "structured_outputs": "structured output",
        "modalities": "output format",
        "image_config": "output format",
        "debug": "streaming",
    }

    cache_control: Optional[OpenRouterCacheControl] = None
    logit_bias: Optional[Dict[_TokenId, Annotated[float, Field(ge=-100, le=100)]]] = None
    metadata: Optional[_Metadata] = None
    min_p: Optional[float] = Field(default=None, ge=0, le=1)
    models: Optional[List[str]] = None
    prompt_cache_key: Optional[str] = None
    prompt_cache_options: Optional[OpenRouterPromptCacheOptions] = None
    provider: Optional[Dict[str, Any]] = None
    reasoning: Optional[OpenRouterReasoning] = None
    reasoning_effort: Optional[
        Literal["max", "xhigh", "high", "medium", "low", "minimal", "none"]
    ] = None
    repetition_penalty: Optional[float] = Field(default=None, ge=0, le=2)
    seed: Optional[int] = None
    service_tier: Optional[
        Literal["auto", "default", "fast", "flex", "priority", "scale", "ultrafast"]
    ] = None
    stop: Optional[_Stop4] = None
    top_a: Optional[float] = Field(default=None, ge=0, le=1)
    top_k: Optional[int] = Field(default=None, ge=0)
    user: Optional[str] = None
    verbosity: Optional[Literal["low", "medium", "high", "xhigh", "max"]] = None


class VLLMChatBlock(_ChatPenalties, _ChatLength, _ChatLogprobs):
    """The ``vllm`` block: the request body of a vLLM server's Chat Completions API.

    Mirrors vLLM's ``ChatCompletionRequest`` with its extra sampling parameters
    (https://docs.vllm.ai/en/latest/serving/online_serving/), with bounds from vLLM's
    ``SamplingParams`` checks, checked 2026-10-13. ``top_logprobs`` is capped at 20,
    the server's default ``max_logprobs``.

    Example:
        ```yaml
        vllm:
          top_k: 20
          min_p: 0.05
          chat_template_kwargs: {enable_thinking: false}
        ```
    """

    default_block: ClassVar[str] = "vllm"
    api: ClassVar[str] = "is not a supported field of the vLLM Chat Completions request"
    reserved: ClassVar[Dict[str, str]] = {
        **_CHAT_RESERVED,
        "guided_json": "structured output",
        "guided_regex": "structured output",
        "guided_choice": "structured output",
        "guided_grammar": "structured output",
        "guided_decoding_backend": "structured output",
        "guided_whitespace_pattern": "structured output",
        "structured_outputs": "structured output",
        "chat_template": "messages",
        "add_generation_prompt": "messages",
        "continue_final_message": "messages",
        "echo": "messages",
    }

    chat_template_kwargs: Optional[Dict[str, Any]] = None
    ignore_eos: Optional[bool] = None
    include_stop_str_in_output: Optional[bool] = None
    length_penalty: Optional[float] = None
    logit_bias: Optional[Dict[_TokenId, float]] = None
    min_p: Optional[float] = Field(default=None, ge=0, le=1)
    min_tokens: Optional[int] = Field(default=None, ge=0)
    priority: Optional[int] = None
    reasoning_effort: Optional[
        Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"]
    ] = None
    repetition_penalty: Optional[float] = Field(default=None, gt=0)
    seed: Optional[int] = Field(default=None, ge=-(2**63), le=2**63 - 1)
    skip_special_tokens: Optional[bool] = None
    spaces_between_special_tokens: Optional[bool] = None
    stop: Optional[Union[str, List[str]]] = None
    stop_token_ids: Optional[List[int]] = None
    temperature: Optional[float] = Field(default=None, ge=0, le=2)
    # -1 means no limit.
    thinking_token_budget: Optional[int] = Field(default=None, ge=-1)
    # -1 and 0 both turn top-k off.
    top_k: Optional[int] = Field(default=None, ge=-1)
    top_p: Optional[float] = Field(default=None, gt=0, le=1)
    user: Optional[str] = None


class AzureUserSecurityContext(_Block):
    """The ``user_security_context`` object of an Azure OpenAI chat request."""

    application_name: Optional[str] = None
    end_user_id: Optional[str] = None
    end_user_tenant_id: Optional[str] = None
    source_ip: Optional[str] = None


class AzureChatBlock(_ChatSampling, _ChatPenalties, _ChatLength, _ChatLogprobs):
    """The ``azure`` block: the request body of the Azure OpenAI Chat Completions API.

    Mirrors the chat completions request in the Azure OpenAI reference
    (https://learn.microsoft.com/azure/ai-foundry/openai/latest) and the matching
    ``openai.types.chat.completion_create_params`` fields, checked 2026-10-13.
    ``service_tier``, ``prompt_cache_options`` and ``moderation`` are left out: the
    ``openai`` SDK sends them, but the Azure reference does not list them.

    Example:
        ```yaml
        azure:
          reasoning_effort: low
          max_completion_tokens: 4096
        ```
    """

    default_block: ClassVar[str] = "azure"
    api: ClassVar[str] = "is not a supported field of the Azure OpenAI Chat Completions request"
    reserved: ClassVar[Dict[str, str]] = {
        **_CHAT_RESERVED,
        "data_sources": "tools: On Your Data retrieval",
        "functions": "tools",
        "function_call": "tools",
        "modalities": "output format",
        "audio": "output format",
    }

    logit_bias: Optional[Dict[_TokenId, Annotated[int, Field(ge=-100, le=100)]]] = None
    metadata: Optional[_Metadata] = None
    prompt_cache_key: Optional[str] = None
    prompt_cache_retention: Optional[Literal["in_memory", "24h"]] = None
    reasoning_effort: Optional[
        Literal["none", "minimal", "low", "medium", "high", "xhigh"]
    ] = None
    safety_identifier: Optional[str] = Field(default=None, max_length=64)
    seed: Optional[int] = None
    stop: Optional[_Stop4] = None
    store: Optional[bool] = None
    user: Optional[str] = None
    user_security_context: Optional[AzureUserSecurityContext] = None
    verbosity: Optional[Literal["low", "medium", "high"]] = None


class AnthropicCacheControl(_Block):
    """A ``CacheControlEphemeral`` object of an Anthropic Messages request."""

    type: Literal["ephemeral"]
    ttl: Optional[Literal["5m", "1h"]] = None


class AnthropicMetadata(_Block):
    """The ``metadata`` object of an Anthropic Messages request."""

    user_id: Optional[str] = Field(default=None, max_length=512)


class AnthropicOutputConfig(_Block):
    """The ``output_config`` object of an Anthropic Messages request (``format`` is reserved)."""

    effort: Optional[Literal["low", "medium", "high", "xhigh", "max"]] = None


_ThinkingDisplay = Optional[Literal["summarized", "omitted"]]


class AnthropicThinkingEnabled(_Block):
    """Manual extended thinking with a token budget of at least 1024."""

    type: Literal["enabled"]
    budget_tokens: int = Field(ge=1024)
    display: _ThinkingDisplay = None


class AnthropicThinkingDisabled(_Block):
    """Thinking turned off."""

    type: Literal["disabled"]


class AnthropicThinkingBetweenTools(_Block):
    """Thinking only between tool calls."""

    type: Literal["between_tools"]


class AnthropicThinkingAdaptive(_Block):
    """Thinking whose amount the model decides."""

    type: Literal["adaptive"]
    display: _ThinkingDisplay = None


class AnthropicMessagesBlock(_Block):
    """The ``anthropic`` block: the request body of the Anthropic Messages API.

    Mirrors ``POST /v1/messages``
    (https://platform.claude.com/docs/en/api/messages), checked 2026-10-13. ``temperature``,
    ``top_k`` and ``top_p`` are deprecated for newer models but still body fields; a model
    that refuses them answers with a provider error. ``max_tokens`` starts at 1 because a
    zero-token call only warms the cache and yian reads the reply.

    Example:
        ```yaml
        anthropic:
          max_tokens: 16000
          thinking: {type: adaptive, display: summarized}
          output_config: {effort: high}
        ```
    """

    default_block: ClassVar[str] = "anthropic"
    api: ClassVar[str] = "is not a supported field of the Anthropic Messages API request"
    reserved: ClassVar[Dict[str, str]] = {
        "model": "llm_config.model",
        "messages": "messages",
        "system": "system prompt",
        "tools": "tools",
        "tool_choice": "tools",
        "stream": "streaming",
        "output_config.format": "structured output",
        "container": "tools",
        "mcp_servers": "tools",
        "diagnostics": "conversation state",
    }

    cache_control: Optional[AnthropicCacheControl] = None
    inference_geo: Optional[str] = None
    max_tokens: Optional[int] = Field(default=None, ge=1)
    metadata: Optional[AnthropicMetadata] = None
    output_config: Optional[AnthropicOutputConfig] = None
    service_tier: Optional[Literal["auto", "standard_only"]] = None
    stop_sequences: Optional[List[str]] = None
    temperature: Optional[float] = Field(default=None, ge=0, le=1)
    thinking: Optional[
        _by_type(
            AnthropicThinkingEnabled,
            AnthropicThinkingDisabled,
            AnthropicThinkingBetweenTools,
            AnthropicThinkingAdaptive,
        )
    ] = None
    top_k: Optional[int] = Field(default=None, ge=0)
    top_p: Optional[float] = Field(default=None, ge=0, le=1)


class GeminiThinkingConfig(_Block):
    """The ``generationConfig.thinkingConfig`` object of a Gemini request."""

    includeThoughts: Optional[bool] = None
    # -1 asks for a dynamic budget and 0 turns thinking off on models that allow it.
    thinkingBudget: Optional[int] = None
    thinkingLevel: Optional[Literal["MINIMAL", "LOW", "MEDIUM", "HIGH"]] = None


class GeminiGenerationConfig(_Block):
    """The ``generationConfig`` object of a Gemini request, without its output-format fields."""

    enableEnhancedCivicAnswers: Optional[bool] = None
    frequencyPenalty: Optional[float] = None
    logprobs: Optional[int] = Field(default=None, ge=0, le=20)
    maxOutputTokens: Optional[int] = Field(default=None, ge=1)
    mediaResolution: Optional[
        Literal["MEDIA_RESOLUTION_LOW", "MEDIA_RESOLUTION_MEDIUM", "MEDIA_RESOLUTION_HIGH"]
    ] = None
    presencePenalty: Optional[float] = None
    responseLogprobs: Optional[bool] = None
    seed: Optional[int] = None
    stopSequences: Optional[Annotated[List[str], Field(max_length=5)]] = None
    temperature: Optional[float] = Field(default=None, ge=0, le=2)
    thinkingConfig: Optional[GeminiThinkingConfig] = None
    topK: Optional[int] = None
    topP: Optional[float] = Field(default=None, ge=0, le=1)


class GeminiSafetySetting(_Block):
    """One ``safetySettings`` entry of a Gemini request: a category and its block threshold."""

    category: Literal[
        "HARM_CATEGORY_HARASSMENT",
        "HARM_CATEGORY_HATE_SPEECH",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "HARM_CATEGORY_DANGEROUS_CONTENT",
        "HARM_CATEGORY_CIVIC_INTEGRITY",
        "HARM_CATEGORY_JAILBREAK",
    ]
    threshold: Literal[
        "BLOCK_LOW_AND_ABOVE", "BLOCK_MEDIUM_AND_ABOVE", "BLOCK_ONLY_HIGH", "BLOCK_NONE", "OFF"
    ]


class GeminiGenerateContentBlock(_Block):
    """The ``google`` block: the request body of the Gemini API ``generateContent`` method.

    Mirrors ``POST /v1beta/{model=models/*}:generateContent``
    (https://ai.google.dev/api/generate-content), checked 2026-10-13, with the REST
    (camelCase) names and enum spellings. Safety categories are the ones the reference
    says ``safetySettings`` supports. Output-modality fields (speech, image, audio,
    translation) are left out because yian reads a text reply, and ``labels`` because
    the Gemini Developer API refuses it.

    Example:
        ```yaml
        google:
          generationConfig:
            thinkingConfig: {thinkingLevel: HIGH}
            maxOutputTokens: 8192
          safetySettings:
            - {category: HARM_CATEGORY_DANGEROUS_CONTENT, threshold: BLOCK_ONLY_HIGH}
        ```
    """

    default_block: ClassVar[str] = "google"
    api: ClassVar[str] = "is not a supported field of the Gemini generateContent request"
    reserved: ClassVar[Dict[str, str]] = {
        "model": "llm_config.model",
        "contents": "messages",
        "systemInstruction": "system prompt",
        "tools": "tools",
        "toolConfig": "tools",
        "cachedContent": "system prompt and tools",
        "continuationToken": "conversation state",
        "generationConfig.responseMimeType": "structured output",
        "generationConfig.responseSchema": "structured output",
        "generationConfig.responseJsonSchema": "structured output",
        "generationConfig._responseJsonSchema": "structured output",
        "generationConfig.responseFormat": "structured output",
        "generationConfig.candidateCount": "yian reads one reply",
    }

    generationConfig: Optional[GeminiGenerationConfig] = None
    safetySettings: Optional[List[GeminiSafetySetting]] = None
    serviceTier: Optional[Literal["standard", "flex", "priority"]] = None
    store: Optional[bool] = None


class OllamaOptions(_Block):
    """The ``options`` object of an Ollama chat request.

    Has the fields of ``ollama.Options`` plus ``min_p``, which the /api/chat reference
    lists; the client sends ``options`` as a plain mapping.
    """

    # Load-time options.
    numa: Optional[bool] = None
    num_ctx: Optional[int] = None
    num_batch: Optional[int] = None
    num_gpu: Optional[int] = None
    main_gpu: Optional[int] = None
    low_vram: Optional[bool] = None
    f16_kv: Optional[bool] = None
    logits_all: Optional[bool] = None
    vocab_only: Optional[bool] = None
    use_mmap: Optional[bool] = None
    use_mlock: Optional[bool] = None
    embedding_only: Optional[bool] = None
    num_thread: Optional[int] = None

    # Runtime options.
    num_keep: Optional[int] = None
    seed: Optional[int] = None
    num_predict: Optional[int] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    tfs_z: Optional[float] = None
    typical_p: Optional[float] = None
    repeat_last_n: Optional[int] = None
    temperature: Optional[float] = None
    repeat_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    mirostat: Optional[int] = None
    mirostat_tau: Optional[float] = None
    mirostat_eta: Optional[float] = None
    min_p: Optional[float] = None
    penalize_newline: Optional[bool] = None
    stop: Optional[List[str]] = None


class OllamaChatBlock(_Block):
    """The ``ollama`` block: the request body of Ollama's native chat API.

    Mirrors ``POST /api/chat``
    (https://github.com/ollama/ollama/blob/main/docs/api.md#generate-a-chat-completion),
    checked 2026-10-13, limited to what ``ollama.Client.chat`` sends (``ollama`` 0.6.3).
    ``think`` takes the levels the reference lists; the client passes any string through.

    Example:
        ```yaml
        ollama:
          think: high
          options: {num_ctx: 8192, temperature: 0.6}
          keep_alive: 10m
        ```
    """

    default_block: ClassVar[str] = "ollama"
    api: ClassVar[str] = "is not a supported field of the Ollama /api/chat request"
    reserved: ClassVar[Dict[str, str]] = {
        "model": "llm_config.model",
        "messages": "messages",
        "tools": "tools",
        "stream": "streaming",
        "format": "structured output",
    }

    # A duration string such as "5m", or a number of seconds; negative keeps the model loaded.
    keep_alive: Optional[Union[float, str]] = None
    logprobs: Optional[bool] = None
    options: Optional[OllamaOptions] = None
    think: Optional[Union[bool, Literal["low", "medium", "high", "max"]]] = None
    top_logprobs: Optional[int] = None


PROVIDER_BLOCKS: Dict[str, type[_Block]] = {
    "openai": OpenAIResponsesBlock,
    "azure": AzureChatBlock,
    "xai": XAIChatBlock,
    "deepseek": DeepSeekChatBlock,
    "qwen": QwenChatBlock,
    "qwen-cn": QwenCNChatBlock,
    "glm": GLMChatBlock,
    "glm-cn": GLMCNChatBlock,
    "minimax": MiniMaxChatBlock,
    "minimax-cn": MiniMaxCNChatBlock,
    "openrouter": OpenRouterChatBlock,
    "vllm": VLLMChatBlock,
    "ollama": OllamaChatBlock,
    "anthropic": AnthropicMessagesBlock,
    "google": GeminiGenerateContentBlock,
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

    Example:
        ```python
        block_schema("openai", {"method": "subscription"})
        # OpenAISubscriptionBlock
        block_schema("qwen-cn")
        # QwenCNChatBlock
        ```
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
        if value is None:
            continue
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


@dataclass(frozen=True)
class BlockFieldType:
    """The type of a provider-block field simple enough to check a value's source against.

    Attributes:
        base (`type`): ``str``, ``int``, ``float`` or ``bool``.
        literals (`tuple`): The values a ``Literal`` field allows; empty for any other.
        ge (`int | float | None`): The inclusive lower bound, if any.
        le (`int | float | None`): The inclusive upper bound, if any.
    """

    base: type
    literals: tuple = ()
    ge: Optional[Union[int, float]] = None
    le: Optional[Union[int, float]] = None


_SIMPLE_TYPES = (str, int, float, bool)


@dataclass(frozen=True)
class BlockProblem:
    """One problem `located_block_problems` finds in a provider block.

    Attributes:
        keys (`tuple[str | int, ...]`): The deepest written key path inside the block the
            problem names, with list indexes as ``int``; empty for the block itself.
        message (`str`): The message, starting with the backticked key path.
    """

    keys: tuple[Union[str, int], ...]
    message: str


def partial_block_problems(
    block: str, tree: Any, is_value: Callable[[Any], bool]
) -> list[str]:
    """Check a provider block whose leaves may be values that resolve later.

    Reserved and unknown key paths are reported as `block_schema`'s class reports them.
    A value is opaque: it is never checked, a required field given as one counts as
    present, and a nested object whose ``type`` (the variant tag) is one is not walked.
    Every other leaf is checked against its field's type and constraints. A list
    holding a value has its object entries walked for key paths and its other entries
    left for when it resolves; a mapping field holding one has its value-free items
    checked and its item count left for then. Messages name the key path and never echo
    a value.

    Args:
        block (`str`):
            A key of `PROVIDER_BLOCKS`, e.g. ``"anthropic"``.
        tree (`Any`):
            The block as written, with values in place of some leaves or subtrees.
        is_value (`Callable[[Any], bool]`):
            Whether a leaf or subtree is a value.

    Returns:
        `list[str]`: One message per problem, in the order written; empty when none.

    Raises:
        `KeyError`:
            Raised when ``block`` has no block class.

    Example:
        ```python
        partial_block_problems(
            "anthropic", {"thinking": {"type": "enabled", "budget_tokens": 10}}, is_value
        )
        # ["`anthropic.thinking.budget_tokens`: Input should be greater than or equal to 1024"]
        ```
    """
    return [problem.message for problem in located_block_problems(block, tree, is_value)]


def located_block_problems(
    block: str, tree: Any, is_value: Callable[[Any], bool]
) -> list[BlockProblem]:
    """Check a block as `partial_block_problems` does; give each problem's key path too.

    Args:
        block (`str`):
            A key of `PROVIDER_BLOCKS`, e.g. ``"anthropic"``.
        tree (`Any`):
            The block as written, with values in place of some leaves or subtrees.
        is_value (`Callable[[Any], bool]`):
            Whether a leaf or subtree is a value.

    Returns:
        `list[BlockProblem]`: One per problem, in the order written; empty when none.

    Raises:
        `KeyError`:
            Raised when ``block`` has no block class.
    """
    top = PROVIDER_BLOCKS[block]
    if is_value(tree):
        return []
    if not isinstance(tree, Mapping):
        return [BlockProblem((), f"`{top.default_block}` must be a mapping")]
    problems: list[BlockProblem] = []
    _partial_problems(top, tree, top=top, prefix="", keys=(), is_value=is_value,
                      problems=problems)
    return problems


def _partial_problems(
    model: type[_Block],
    data: Mapping,
    *,
    top: type[_Block],
    prefix: str,
    keys: tuple[Union[str, int], ...],
    is_value: Callable[[Any], bool],
    problems: list[BlockProblem],
) -> None:
    """Append the problems of the partial object ``data`` checked against ``model``.

    ``prefix`` is the written key path of ``data`` as messages show it, and ``keys`` the
    same path as parts.
    """
    names = {field.alias or name: name for name, field in model.model_fields.items()}
    for key, value in data.items():
        path = f"{prefix}{key}"
        problem = _bad_key(top, path, key, names)
        if problem is not None:
            problems.append(BlockProblem((*keys, key), problem))
        else:
            _partial_field(model, names[key], value, top=top, path=path, keys=(*keys, key),
                           is_value=is_value, problems=problems)
    for key, name in names.items():
        if key not in data and model.model_fields[name].is_required():
            problems.append(BlockProblem(
                (*keys, key), f"`{top.default_block}.{prefix}{key}` is required"
            ))


def _partial_field(
    model: type[_Block],
    name: str,
    value: Any,
    *,
    top: type[_Block],
    path: str,
    keys: tuple[Union[str, int], ...],
    is_value: Callable[[Any], bool],
    problems: list[BlockProblem],
) -> None:
    """Append the problems of one field's partial ``value``; values are never checked."""
    if is_value(value):
        return
    annotation = model.model_fields[name].annotation
    if isinstance(value, Mapping):
        part = _nested_block(annotation, value)
        if part is not None:
            _partial_problems(part, value, top=top, prefix=f"{path}.", keys=keys,
                              is_value=is_value, problems=problems)
            return
        if _holds_value(value, is_value):
            if any(typing.get_origin(m) is dict for m in _members(annotation)):
                # Only the value-free items can be checked; the item count waits for the rest.
                written = {k: v for k, v in value.items() if not _holds_value(v, is_value)}
                _check_literal(model, name, written, top=top, path=path, keys=keys,
                               problems=problems)
            # Several variants and no literal tag: only the tag itself can be checked.
            elif len(_block_members(annotation)) > 1 and not is_value(value.get("type")):
                tag = {"type": value["type"]} if "type" in value else {}
                _check_literal(model, name, tag, top=top, path=path, keys=keys,
                               problems=problems)
            elif not _block_members(annotation) and Any not in _members(annotation):
                # A field that takes no mapping: an empty one gives its own type's message,
                # whatever the values resolve to.
                _check_literal(model, name, {}, top=top, path=path, keys=keys, problems=problems)
            return
    elif isinstance(value, list) and _holds_value(value, is_value):
        items = [typing.get_args(m)[0] for m in _members(annotation) if _is_list(m)]
        if not items:
            # A field that takes no list, as for a mapping above.
            if Any not in _members(annotation):
                _check_literal(model, name, [], top=top, path=path, keys=keys, problems=problems)
            return
        item = Union[tuple(items)]
        for index, entry in enumerate(value):
            if isinstance(entry, Mapping) and not is_value(entry):
                part = _nested_block(item, entry)
                if part is not None:
                    _partial_problems(part, entry, top=top, prefix=f"{path}[{index}].",
                                      keys=(*keys, index), is_value=is_value,
                                      problems=problems)
        return
    _check_literal(model, name, value, top=top, path=path, keys=keys, problems=problems)


def _check_literal(
    model: type[_Block],
    name: str,
    value: Any,
    *,
    top: type[_Block],
    path: str,
    keys: tuple[Union[str, int], ...],
    problems: list[BlockProblem],
) -> None:
    """Append why the value-free ``value`` is not valid for field ``name`` of ``model``.

    Each message is located at the deepest key path of ``value`` its error names; the
    union branch names pydantic adds to a location (``str``, ``list[str]``) are dropped.
    """
    try:
        model.__pydantic_validator__.validate_assignment(model.model_construct(), name, value)
    except ValidationError as exc:
        for error in exc.errors(include_input=False, include_url=False):
            written, parts = _written_path(path, keys, value, error["loc"][1:])
            problem = BlockProblem(parts, f"`{top.default_block}.{written}`: {error['msg']}")
            if problem not in problems:
                problems.append(problem)


def _written_path(
    path: str, keys: tuple[Union[str, int], ...], value: Any, loc: tuple
) -> tuple[str, tuple[Union[str, int], ...]]:
    """Extend ``path`` and its parts ``keys`` by the parts of ``loc`` written in ``value``."""
    node = value
    for part in loc:
        if isinstance(part, int) and isinstance(node, list) and 0 <= part < len(node):
            path += f"[{part}]"
        elif isinstance(part, str) and isinstance(node, Mapping) and part in node:
            path += f".{part}"
        else:
            continue
        keys = (*keys, part)
        node = node[part]
    return path, keys


def _holds_value(tree: Any, is_value: Callable[[Any], bool]) -> bool:
    """Return whether ``tree`` is, or anywhere holds, a value."""
    if is_value(tree):
        return True
    if isinstance(tree, Mapping):
        return any(_holds_value(item, is_value) for item in tree.values())
    if isinstance(tree, list):
        return any(_holds_value(item, is_value) for item in tree)
    return False


def _block_members(annotation: Any) -> list[type[_Block]]:
    """Return the block classes ``annotation`` allows."""
    return [m for m in _members(annotation) if isinstance(m, type) and issubclass(m, _Block)]


def block_field_annotation(
    block: str, path: tuple[str, ...], tree: Any
) -> Optional[BlockFieldType]:
    """Return the simple type of the block field at ``path``, if it has one.

    A nested object with several variants is told apart by the literal ``type`` written
    at its place in ``tree``.

    Args:
        block (`str`):
            A key of `PROVIDER_BLOCKS`, e.g. ``"anthropic"``.
        path (`tuple[str, ...]`):
            The field's key path inside the block, e.g. ``("thinking", "budget_tokens")``.
        tree (`Any`):
            The block as written, which may hold values.

    Returns:
        `BlockFieldType | None`:
            The field's type when it is ``str``, ``int``, ``float`` or ``bool`` (with
            any inclusive bounds) or a ``Literal`` of one of them; `None` for any other
            field (an object, list, mapping or union), an unknown path, or a variant
            that ``tree`` does not name.

    Raises:
        `KeyError`:
            Raised when ``block`` has no block class.

    Example:
        ```python
        block_field_annotation("anthropic", ("thinking", "budget_tokens"),
                               {"thinking": {"type": "enabled"}})
        # BlockFieldType(base=int, literals=(), ge=1024, le=None)
        ```
    """
    model: Optional[type[_Block]] = PROVIDER_BLOCKS[block]
    for index, key in enumerate(path):
        fields = {field.alias or name: field for name, field in model.model_fields.items()}
        field = fields.get(key)
        if field is None:
            return None
        if index == len(path) - 1:
            return _simple_type(field.annotation, field.metadata)
        tree = tree.get(key) if isinstance(tree, Mapping) else None
        model = _nested_block(field.annotation, tree if isinstance(tree, Mapping) else {})
        if model is None:
            return None
    return None


def _simple_type(annotation: Any, metadata: list[Any]) -> Optional[BlockFieldType]:
    """Return the `BlockFieldType` of a field's annotation and metadata, if it is simple.

    Only inclusive bounds (``ge``, ``le``) are carried; exclusive bounds (``gt``, ``lt``)
    and string length and pattern constraints are checked only when the whole block is
    validated after its values resolve.
    """
    members = [m for m in _members(annotation) if m is not type(None)]
    if len(members) != 1:
        return None
    member = members[0]
    # pydantic keeps ``Field(ge=..., le=...)`` as metadata objects carrying that attribute.
    ge = next((m.ge for m in metadata if getattr(m, "ge", None) is not None), None)
    le = next((m.le for m in metadata if getattr(m, "le", None) is not None), None)
    if member in _SIMPLE_TYPES:
        return BlockFieldType(base=member, ge=ge, le=le)
    if typing.get_origin(member) is Literal:
        literals = typing.get_args(member)
        kinds = {type(literal) for literal in literals}
        if len(kinds) == 1 and next(iter(kinds)) in _SIMPLE_TYPES:
            return BlockFieldType(base=next(iter(kinds)), literals=literals)
    return None
