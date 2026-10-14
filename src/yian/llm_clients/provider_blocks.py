"""Provider blocks of ``llm_config``: request-body fields checked against the API yian calls.

Each block mirrors the request body of one provider API with the same names, nesting, and
allowed values, and yian deep-merges it into that body as written. Fields that yian builds
itself (model, messages, tools, structured output, streaming, conversation state) are
reserved: setting one is a load error that says so. Error messages name the block and the
key path and never echo values.
"""

from __future__ import annotations

import types
import typing
from collections.abc import Collection, Mapping
from typing import Any, ClassVar, Dict, List, Literal, Optional, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Discriminator,
    Field,
    StringConstraints,
    Tag,
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
                cls._check_value(fields[key].annotation, value, block, path, problems)

    @classmethod
    def _check_value(
        cls, annotation: Any, value: Any, block: str, path: str, problems: list[str]
    ) -> None:
        """Check the keys of a nested block, or of each block in a list, at ``path``."""
        if isinstance(value, Mapping):
            part = _nested_block(annotation, value)
            if part is not None:
                cls._collect_problems(part, value, block, f"{path}.", problems)
        elif isinstance(value, list):
            items = [typing.get_args(m)[0] for m in _members(annotation) if _is_list(m)]
            if items:
                item = Union[tuple(items)]
                for index, entry in enumerate(value):
                    cls._check_value(item, entry, block, f"{path}[{index}]", problems)

    @model_serializer(mode="wrap")
    def _drop_unset(self, handler):
        """Dump without ``None`` fields, so an unset field is never recorded or sent."""
        data = handler(self)
        if isinstance(data, dict):
            return {key: value for key, value in data.items() if value is not None}
        return data


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
    if len(blocks) == 1:
        return blocks[0]
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
    2026-10-13 against that reference and the ``openai`` SDK's
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
_Metadata = Annotated[Dict[_MetadataKey, _MetadataValue], Field(max_length=16)]


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


class XAIChatBlock(_ChatSampling, _ChatPenalties, _ChatLength):
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
    api: ClassVar[str] = "is not a field of the xAI Chat Completions request"
    reserved: ClassVar[Dict[str, str]] = {
        **_CHAT_RESERVED,
        "deferred": "request lifecycle",
        "search_parameters": "tools: server-side search",
        "web_search_options": "tools: server-side search",
    }

    logprobs: Optional[bool] = None
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
    """The ``glm`` and ``glm-cn`` blocks: the request body of the Z.AI GLM Chat Completion API.

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
    api: ClassVar[str] = "is not a field of the Z.AI GLM Chat Completion request"
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


class MiniMaxThinking(_Block):
    """The ``thinking`` object of a MiniMax chat request."""

    type: Optional[Literal["disabled", "adaptive"]] = None


class MiniMaxChatBlock(_ChatSampling, _ChatLength):
    """The ``minimax`` and ``minimax-cn`` blocks: the body of MiniMax's OpenAI-compatible API.

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
    api: ClassVar[str] = "is not a field of the MiniMax Chat Completions request"
    reserved: ClassVar[Dict[str, str]] = {
        **_CHAT_RESERVED,
        "reasoning_split": "keeps reasoning out of the message content",
    }

    reasoning_effort: Optional[Literal["low", "medium", "high", "xhigh", "max"]] = None
    service_tier: Optional[Literal["standard", "priority"]] = None
    thinking: Optional[MiniMaxThinking] = None


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
    api: ClassVar[str] = "is not a field of the DeepSeek Chat Completion request"
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
    """The ``qwen`` and ``qwen-cn`` blocks: the body of Model Studio's OpenAI-compatible chat.

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
    api: ClassVar[str] = "is not a field of the Model Studio Chat Completions request"
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
    checked 2026-10-13. ``provider`` (routing preferences) and ``trace`` are passed
    through as written.

    Example:
        ```yaml
        openrouter:
          reasoning: {effort: high}
          provider: {order: [anthropic], allow_fallbacks: false}
        ```
    """

    default_block: ClassVar[str] = "openrouter"
    api: ClassVar[str] = "is not a field of the OpenRouter Chat Completions request"
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
    logit_bias: Optional[Dict[str, Annotated[float, Field(ge=-100, le=100)]]] = None
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
    session_id: Optional[str] = Field(default=None, max_length=256)
    stop: Optional[_Stop4] = None
    top_a: Optional[float] = Field(default=None, ge=0, le=1)
    top_k: Optional[int] = Field(default=None, ge=0)
    trace: Optional[Dict[str, Any]] = None
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
    api: ClassVar[str] = "is not a field of the vLLM Chat Completions request"
    reserved: ClassVar[Dict[str, str]] = {
        **_CHAT_RESERVED,
        "guided_json": "structured output",
        "guided_regex": "structured output",
        "guided_choice": "structured output",
        "guided_grammar": "structured output",
        "guided_decoding_backend": "structured output",
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
    logit_bias: Optional[Dict[str, float]] = None
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

    Example:
        ```yaml
        azure:
          reasoning_effort: low
          max_completion_tokens: 4096
        ```
    """

    default_block: ClassVar[str] = "azure"
    api: ClassVar[str] = "is not a field of the Azure OpenAI Chat Completions request"
    reserved: ClassVar[Dict[str, str]] = {
        **_CHAT_RESERVED,
        "data_sources": "tools: On Your Data retrieval",
        "functions": "tools",
        "function_call": "tools",
        "modalities": "output format",
        "audio": "output format",
    }

    logit_bias: Optional[Dict[str, Annotated[int, Field(ge=-100, le=100)]]] = None
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
    api: ClassVar[str] = "is not a field of the Anthropic Messages API request"
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
    api: ClassVar[str] = "is not a field of the Gemini generateContent request"
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
    """The ``options`` object of an Ollama chat request, with the fields of ``ollama.Options``."""

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
    penalize_newline: Optional[bool] = None
    stop: Optional[List[str]] = None


class OllamaChatBlock(_Block):
    """The ``ollama`` block: the request body of Ollama's native chat API.

    Mirrors ``POST /api/chat``
    (https://github.com/ollama/ollama/blob/main/docs/api.md#generate-a-chat-completion),
    checked 2026-10-13, limited to what ``ollama.Client.chat`` sends (``ollama`` 0.6.3).

    Example:
        ```yaml
        ollama:
          think: high
          options: {num_ctx: 8192, temperature: 0.6}
          keep_alive: 10m
        ```
    """

    default_block: ClassVar[str] = "ollama"
    api: ClassVar[str] = "is not a field of the Ollama /api/chat request"
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
    think: Optional[Union[bool, Literal["low", "medium", "high"]]] = None
    top_logprobs: Optional[int] = None


PROVIDER_BLOCKS: Dict[str, type[_Block]] = {
    "openai": OpenAIResponsesBlock,
    "azure": AzureChatBlock,
    "xai": XAIChatBlock,
    "deepseek": DeepSeekChatBlock,
    "qwen": QwenChatBlock,
    "qwen-cn": QwenChatBlock,
    "glm": GLMChatBlock,
    "glm-cn": GLMChatBlock,
    "minimax": MiniMaxChatBlock,
    "minimax-cn": MiniMaxChatBlock,
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
