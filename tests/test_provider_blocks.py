"""Provider request-body blocks: shape checks, error wording, dumps, and merging."""

import copy
import inspect

import pytest
import yaml
from pydantic import ValidationError

from yian.llm_clients.config import SUPPORTED_LLM_PROVIDERS
from yian.llm_clients.provider_blocks import (
    PROVIDER_BLOCKS,
    AnthropicMessagesBlock,
    AzureChatBlock,
    DeepSeekChatBlock,
    GeminiGenerateContentBlock,
    GLMChatBlock,
    GLMCNChatBlock,
    MiniMaxChatBlock,
    MiniMaxCNChatBlock,
    OllamaChatBlock,
    OpenAIResponsesBlock,
    OpenAISubscriptionBlock,
    OpenRouterChatBlock,
    QwenChatBlock,
    QwenCNChatBlock,
    VLLMChatBlock,
    XAIChatBlock,
    block_schema,
    merge_request_body,
)

SENTINEL = "sk-sentinel-7d2e91b4"

RESPONSES_SAMPLE = {
    "access_programs": {"cyber": "standard"},
    "max_output_tokens": 2048,
    "max_tool_calls": 4,
    "metadata": {"team": "research"},
    "moderation": {
        "model": "omni-moderation-latest",
        "policy": {"input": {"mode": "block"}, "output": {"mode": "score"}},
    },
    "prompt_cache_key": "flow-a",
    "prompt_cache_options": {"mode": "explicit", "ttl": "30m"},
    "prompt_cache_retention": "24h",
    "reasoning": {"context": "all_turns", "effort": "high", "mode": "pro", "summary": "auto"},
    "safety_identifier": "user-hash",
    "service_tier": "flex",
    "temperature": 0.2,
    "text": {"verbosity": "low"},
    "top_logprobs": 5,
    "top_p": 0.9,
    "truncation": "auto",
    "user": "someone",
}


def _error(block_cls, data) -> str:
    with pytest.raises(ValidationError) as caught:
        block_cls.model_validate(data)
    return str(caught.value)


def test_every_supported_provider_has_its_own_block_class():
    assert set(PROVIDER_BLOCKS) == set(SUPPORTED_LLM_PROVIDERS)
    assert PROVIDER_BLOCKS == {
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
    for provider, block_cls in PROVIDER_BLOCKS.items():
        assert block_cls.default_block == provider


def test_dump_omits_unset_fields_at_every_level():
    block = OpenAIResponsesBlock.model_validate(
        {"reasoning": {"effort": "low"}, "text": {}, "top_p": 0.5}
    )
    expected = {"reasoning": {"effort": "low"}, "text": {}, "top_p": 0.5}
    assert block.model_dump(exclude_none=False) == expected
    assert block.model_dump() == expected
    assert OpenAIResponsesBlock().model_dump(exclude_none=False) == {}


def test_explicit_none_reads_as_unset():
    block = OpenAIResponsesBlock.model_validate(
        {"temperature": None, "reasoning": {"effort": None, "summary": "concise"}}
    )
    assert block.model_dump(exclude_none=False) == {"reasoning": {"summary": "concise"}}


def test_top_level_typo_names_block_path_and_allowed_keys():
    message = _error(OpenAIResponsesBlock, {"temprature": 0.3})
    expected = "`openai.temprature` is not a supported field of the OpenAI Responses API request"
    assert expected in message
    assert "allowed: access_programs, max_output_tokens, max_tool_calls, metadata," in message
    assert "temperature" in message


def test_nested_typo_names_full_path_and_nested_allowed_keys():
    message = _error(OpenAIResponsesBlock, {"reasoning": {"efort": "high"}})
    expected = "`openai.reasoning.efort` is not a supported field of the OpenAI Responses API"
    assert f"{expected} request" in message
    assert "allowed: context, effort, mode, summary" in message

    message = _error(OpenAIResponsesBlock, {"text": {"verbose": "low"}})
    assert "`openai.text.verbose`" in message
    assert "allowed: verbosity" in message


def test_every_problem_is_reported():
    message = _error(OpenAIResponsesBlock, {"temprature": 1, "store": False})
    assert "`openai.temprature`" in message
    assert "`openai.store`" in message


@pytest.mark.parametrize(
    "data",
    [
        {"max_output_tokens": "x"},
        {"max_output_tokens": "5"},
        {"max_output_tokens": 0},
        {"max_output_tokens": True},
        {"reasoning": {"effort": "hgih"}},
        {"reasoning": {"summary": "brief"}},
        {"reasoning": {"context": "every_turn"}},
        {"reasoning": {"mode": "turbo"}},
        {"reasoning": "high"},
        {"text": {"verbosity": "loud"}},
        {"temperature": 3},
        {"temperature": -0.1},
        {"top_p": 1.5},
        {"top_logprobs": 21},
        {"truncation": "on"},
        {"service_tier": "turbo"},
        {"metadata": {"k": 1}},
        {"metadata": {f"k{i}": "v" for i in range(17)}},
        {"access_programs": {"cyber": "daybreak_green"}},
        {"moderation": {"policy": {"input": {"mode": "block"}}}},
        {"moderation": {"model": "omni-moderation-latest", "policy": {"input": {}}}},
        {"moderation": {"model": "omni-moderation-latest", "policy": {"output": {"mode": "warn"}}}},
        {"prompt_cache_options": {"ttl": "1h"}},
        {"prompt_cache_options": {"mode": "auto"}},
    ],
)
def test_bad_value_is_rejected(data):
    with pytest.raises(ValidationError):
        OpenAIResponsesBlock.model_validate(data)


@pytest.mark.parametrize(
    "data, path",
    [
        ({"input": "hi"}, "input"),
        ({"instructions": "be terse"}, "instructions"),
        ({"tools": []}, "tools"),
        ({"tool_choice": "auto"}, "tool_choice"),
        ({"parallel_tool_calls": False}, "parallel_tool_calls"),
        ({"model": "gpt-5"}, "model"),
        ({"text": {"format": {"type": "text"}}}, "text.format"),
        ({"store": False}, "store"),
        ({"include": []}, "include"),
        ({"previous_response_id": "resp_1"}, "previous_response_id"),
        ({"conversation": "conv_1"}, "conversation"),
        ({"prompt": {"id": "pmpt_1"}}, "prompt"),
        ({"background": True}, "background"),
        ({"stream": True}, "stream"),
        ({"stream_options": {}}, "stream_options"),
        ({"context_management": [{"type": "compaction"}]}, "context_management"),
    ],
)
def test_reserved_path_is_built_by_yian(data, path):
    message = _error(OpenAIResponsesBlock, data)
    assert f"`openai.{path}` is built by yian (" in message
    assert "remove it" in message


def test_reserved_nested_path_keeps_its_reason():
    message = _error(OpenAIResponsesBlock, {"text": {"format": {"type": "json_object"}}})
    assert "`openai.text.format` is built by yian (structured output); remove it" in message


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_numbers_are_rejected(value):
    with pytest.raises(ValidationError):
        OpenAIResponsesBlock.model_validate({"top_p": value})


@pytest.mark.parametrize("key", ["comparison_response_id", "prewarm"])
def test_per_request_prompt_cache_options_are_not_block_fields(key):
    message = _error(OpenAIResponsesBlock, {"prompt_cache_options": {key: True}})
    assert f"`openai.prompt_cache_options.{key}` is not a supported field of the" in message
    assert "allowed: mode, ttl" in message


def test_subscription_block_rejects_fields_the_endpoint_refuses():
    for key, value in (("temperature", 0.5), ("max_output_tokens", 100)):
        message = _error(OpenAISubscriptionBlock, {key: value})
        assert (
            f"`openai.{key}` is not accepted by the ChatGPT subscription endpoint "
            "(a subset of the Responses API); allowed:"
        ) in message
        assert "temperature," not in message.split("allowed:")[1]


def test_subscription_block_keeps_the_rest_of_the_responses_shape():
    sample = {
        k: v for k, v in RESPONSES_SAMPLE.items() if k not in ("temperature", "max_output_tokens")
    }
    assert OpenAISubscriptionBlock.model_validate(sample).model_dump() == sample
    assert set(OpenAIResponsesBlock.model_fields) - set(OpenAISubscriptionBlock.model_fields) == {
        "temperature",
        "max_output_tokens",
    }
    message = _error(OpenAISubscriptionBlock, {"store": True})
    assert "`openai.store` is built by yian" in message


def test_responses_fields_and_reserved_keys_are_keywords_of_responses_create():
    responses = pytest.importorskip("openai.resources.responses")
    keywords = set(inspect.signature(responses.Responses.create).parameters)
    body_keywords = keywords - {"self", "extra_headers", "extra_query", "extra_body", "timeout"}
    fields = set(OpenAIResponsesBlock.model_fields)
    reserved = {path.split(".")[0] for path in OpenAIResponsesBlock.reserved}
    assert fields <= body_keywords
    assert reserved <= body_keywords
    assert fields | reserved == body_keywords


@pytest.mark.parametrize(
    "field, sdk_type",
    [
        ("access_programs", "AccessPrograms"),
        ("moderation", "Moderation"),
        ("prompt_cache_options", "PromptCacheOptions"),
    ],
)
def test_responses_nested_fields_are_keys_of_the_sdk_params(field, sdk_type):
    params = pytest.importorskip("openai.types.responses.response_create_params")
    nested = OpenAIResponsesBlock.model_fields[field].annotation.__args__[0]
    assert set(nested.model_fields) <= set(getattr(params, sdk_type).__annotations__)


def test_block_schema_picks_the_api_actually_called():
    subscription = {"method": "subscription"}
    assert block_schema("openai", subscription) is OpenAISubscriptionBlock
    assert block_schema("openai", subscription, api_key="sk-x") is OpenAIResponsesBlock
    assert block_schema("openai", {"method": "api_key"}) is OpenAIResponsesBlock
    assert block_schema("openai") is OpenAIResponsesBlock


def test_merge_keeps_sibling_keys_and_replaces_lists_and_scalars():
    body = {
        "model": "gpt-5",
        "text": {"format": {"type": "json_schema"}},
        "stop": ["a", "b"],
        "temperature": 1.0,
    }
    block = {"text": {"verbosity": "low"}, "stop": ["c"], "temperature": 0.2}
    result = merge_request_body(body, block)
    assert result is body
    assert body == {
        "model": "gpt-5",
        "text": {"format": {"type": "json_schema"}, "verbosity": "low"},
        "stop": ["c"],
        "temperature": 0.2,
    }


def test_merge_skips_none_values():
    body = {"temperature": 1.0, "reasoning": {"effort": "low"}}
    merge_request_body(body, {"temperature": None, "reasoning": {"effort": None}})
    assert body == {"temperature": 1.0, "reasoning": {"effort": "low"}}


def test_merge_routes_unknown_top_level_keys_into_extra_body():
    body = {"model": "m", "extra_body": {"chat_template_kwargs": {"a": 1}}}
    block = {
        "temperature": 0.3,
        "top_k": 20,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    merge_request_body(body, block, top_level={"model", "temperature"})
    assert body == {
        "model": "m",
        "temperature": 0.3,
        "extra_body": {
            "chat_template_kwargs": {"a": 1, "enable_thinking": False},
            "top_k": 20,
        },
    }


def test_merge_creates_extra_body_when_absent():
    body = {}
    merge_request_body(body, {"top_k": 5}, top_level=())
    assert body == {"extra_body": {"top_k": 5}}


def test_merge_adds_no_extra_body_for_unset_values():
    assert merge_request_body({}, {"top_k": None}, top_level=()) == {}


def test_merge_does_not_mutate_or_alias_the_block():
    block = {"reasoning": {"effort": "high"}, "stop": ["x"], "extra": {"a": {"b": 1}}}
    snapshot = copy.deepcopy(block)
    body = {"reasoning": {"summary": "auto"}}
    merge_request_body(body, block, top_level={"reasoning", "stop"})
    assert block == snapshot
    body["stop"].append("y")
    body["extra_body"]["extra"]["a"]["b"] = 2
    body["reasoning"]["effort"] = "low"
    assert block == snapshot


CHAT_BLOCKS = [
    XAIChatBlock,
    GLMChatBlock,
    GLMCNChatBlock,
    MiniMaxChatBlock,
    MiniMaxCNChatBlock,
    DeepSeekChatBlock,
    QwenChatBlock,
    QwenCNChatBlock,
    OpenRouterChatBlock,
    VLLMChatBlock,
    AzureChatBlock,
]

SAMPLES = {
    OpenAIResponsesBlock: RESPONSES_SAMPLE,
    XAIChatBlock: {
        "frequency_penalty": 0.5,
        "logprobs": True,
        "max_completion_tokens": 4096,
        "presence_penalty": -0.5,
        "prompt_cache_key": "flow-a",
        "reasoning_effort": "high",
        "safety_identifier": "user-hash",
        "seed": 7,
        "service_tier": "priority",
        "stop": ["END"],
        "temperature": 0.7,
        "top_logprobs": 8,
        "top_p": 0.9,
        "user": "someone",
    },
    GLMChatBlock: {
        "do_sample": True,
        "max_tokens": 131072,
        "reasoning_effort": "xhigh",
        "stop": ["END"],
        "temperature": 1,
        "thinking": {"type": "disabled", "clear_thinking": False},
        "top_p": 0.01,
        "user_id": "user-123456",
    },
    MiniMaxChatBlock: {
        "max_completion_tokens": 2048,
        "max_tokens": 4096,
        "reasoning_effort": "max",
        "service_tier": "priority",
        "temperature": 2,
        "thinking": {"type": "adaptive"},
        "top_p": 0.95,
    },
    DeepSeekChatBlock: {
        "logprobs": True,
        "max_tokens": 393216,
        "reasoning_effort": "max",
        "stop": "END",
        "temperature": 1.3,
        "thinking": {"type": "enabled"},
        "top_logprobs": 20,
        "top_p": 0.95,
        "user_id": "team_a-1",
    },
    QwenChatBlock: {
        "enable_search": True,
        "enable_thinking": True,
        "logprobs": True,
        "max_completion_tokens": 8192,
        "presence_penalty": 1.5,
        "reasoning_effort": "medium",
        "repetition_penalty": 1.05,
        "search_options": {"forced_search": True, "search_strategy": "max"},
        "seed": 2147483647,
        "stop": [104307],
        "temperature": 1.99,
        "thinking_budget": 4096,
        "top_k": 20,
        "top_logprobs": 5,
        "top_p": 0.8,
    },
    OpenRouterChatBlock: {
        "cache_control": {"type": "ephemeral", "ttl": "1h"},
        "logit_bias": {"50256": -100},
        "max_completion_tokens": 1000,
        "metadata": {"session": "s-1"},
        "min_p": 0.1,
        "models": ["openai/gpt-5", "anthropic/claude-opus"],
        "provider": {"allow_fallbacks": False, "order": ["anthropic"]},
        "reasoning": {"effort": "high", "max_tokens": 2000, "exclude": True, "enabled": True},
        "repetition_penalty": 1.1,
        "seed": 42,
        "service_tier": "flex",
        "stop": "END",
        "temperature": 0.7,
        "top_a": 0.2,
        "top_k": 40,
        "top_p": 1,
        "verbosity": "xhigh",
    },
    VLLMChatBlock: {
        "chat_template_kwargs": {"enable_thinking": False},
        "ignore_eos": False,
        "include_stop_str_in_output": True,
        "length_penalty": 1.0,
        "max_tokens": 512,
        "min_p": 0.05,
        "min_tokens": 4,
        "priority": -1,
        "repetition_penalty": 1.1,
        "skip_special_tokens": False,
        "spaces_between_special_tokens": True,
        "stop": ["</s>", "END"],
        "stop_token_ids": [2],
        "temperature": 0.6,
        "top_k": 20,
        "top_p": 0.95,
    },
    AzureChatBlock: {
        "frequency_penalty": 0.1,
        "logit_bias": {"50256": -100},
        "max_completion_tokens": 4096,
        "metadata": {"team": "research"},
        "presence_penalty": 0.2,
        "reasoning_effort": "minimal",
        "safety_identifier": "user-hash",
        "seed": 1,
        "stop": ["END"],
        "temperature": 0.5,
        "top_logprobs": 3,
        "top_p": 0.9,
        "user_security_context": {"application_name": "flows"},
        "verbosity": "low",
    },
    AnthropicMessagesBlock: {
        "cache_control": {"type": "ephemeral", "ttl": "1h"},
        "inference_geo": "us",
        "max_tokens": 8192,
        "metadata": {"user_id": "user-hash"},
        "output_config": {"effort": "xhigh"},
        "service_tier": "standard_only",
        "stop_sequences": ["END"],
        "temperature": 1,
        "thinking": {"type": "enabled", "budget_tokens": 2048, "display": "summarized"},
        "top_k": 40,
        "top_p": 0.99,
    },
    GeminiGenerateContentBlock: {
        "generationConfig": {
            "stopSequences": ["END"],
            "maxOutputTokens": 4096,
            "temperature": 0.7,
            "topP": 0.95,
            "topK": 40,
            "seed": 7,
            "presencePenalty": 0.5,
            "frequencyPenalty": -0.5,
            "responseLogprobs": True,
            "logprobs": 5,
            "enableEnhancedCivicAnswers": False,
            "thinkingConfig": {"includeThoughts": True, "thinkingLevel": "HIGH"},
            "mediaResolution": "MEDIA_RESOLUTION_LOW",
        },
        "safetySettings": [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "OFF"},
        ],
        "serviceTier": "flex",
        "store": False,
    },
    OllamaChatBlock: {
        "think": "high",
        "options": {
            "num_ctx": 8192,
            "num_predict": 512,
            "temperature": 0.6,
            "top_k": 20,
            "top_p": 0.95,
            "repeat_penalty": 1.1,
            "seed": 1,
            "stop": ["</s>"],
            "use_mmap": True,
        },
        "keep_alive": "5m",
        "logprobs": True,
        "top_logprobs": 3,
    },
}


@pytest.mark.parametrize("block_cls", list(SAMPLES), ids=lambda cls: cls.__name__)
def test_valid_sample_dumps_unchanged(block_cls):
    sample = SAMPLES[block_cls]
    block = block_cls.model_validate(sample)
    assert block.model_dump() == sample
    assert block.model_dump(exclude_none=False) == sample
    assert block.model_dump(mode="json") == sample


@pytest.mark.parametrize(
    "block_cls, data",
    [
        (DeepSeekChatBlock, {"reasoning_effort": "medium"}),
        (DeepSeekChatBlock, {"max_tokens": 393217}),
        (DeepSeekChatBlock, {"stop": [str(i) for i in range(17)]}),
        (DeepSeekChatBlock, {"top_p": 0}),
        (GLMChatBlock, {"temperature": 1.5}),
        (GLMChatBlock, {"top_p": 0.001}),
        (GLMChatBlock, {"stop": "END"}),
        (GLMChatBlock, {"thinking": {"type": "adaptive"}}),
        (MiniMaxChatBlock, {"thinking": {"type": "enabled"}}),
        (MiniMaxChatBlock, {"reasoning_effort": "none"}),
        (MiniMaxChatBlock, {"service_tier": "flex"}),
        (QwenChatBlock, {"temperature": 2}),
        (QwenChatBlock, {"top_logprobs": 6}),
        (QwenChatBlock, {"enable_thinking": "yes"}),
        (QwenChatBlock, {"search_options": {"search_strategy": "deep"}}),
        (XAIChatBlock, {"top_logprobs": 9}),
        (XAIChatBlock, {"service_tier": "flex"}),
        (XAIChatBlock, {"stop": ["a", "b", "c", "d", "e"]}),
        (OpenRouterChatBlock, {"reasoning": {"effort": "extreme"}}),
        (OpenRouterChatBlock, {"repetition_penalty": 2.5}),
        (VLLMChatBlock, {"top_k": -2}),
        (VLLMChatBlock, {"top_p": 0}),
        (VLLMChatBlock, {"temperature": float("nan")}),
        (AzureChatBlock, {"reasoning_effort": "extreme"}),
        (AzureChatBlock, {"logit_bias": {"1": 101}}),
        (AzureChatBlock, {"temperature": 2.1}),
    ],
)
def test_value_outside_the_vendor_set_is_rejected(block_cls, data):
    with pytest.raises(ValidationError):
        block_cls.model_validate(data)


@pytest.mark.parametrize(
    "provider, parent, typo",
    [
        ("qwen-cn", QwenChatBlock, "enable_thinkng"),
        ("glm-cn", GLMChatBlock, "do_sampel"),
        ("minimax-cn", MiniMaxChatBlock, "reasoning_efort"),
    ],
)
def test_cn_block_shares_the_schema_and_names_itself(provider, parent, typo):
    block_cls = PROVIDER_BLOCKS[provider]
    assert block_cls is not parent and issubclass(block_cls, parent)
    assert block_cls.model_fields.keys() == parent.model_fields.keys()
    assert block_cls.reserved == parent.reserved
    message = _error(block_cls, {typo: True})
    assert f"`{provider}.{typo}` is not a supported field of the" in message
    assert f"`{parent.default_block}.{typo}`" not in message


def test_nested_chat_typo_names_the_full_path():
    message = _error(GLMChatBlock, {"thinking": {"typ": "enabled"}})
    assert "`glm.thinking.typ` is not a supported field of the" in message
    assert "allowed: clear_thinking, type" in message


@pytest.mark.parametrize("block_cls", CHAT_BLOCKS, ids=lambda cls: cls.__name__)
@pytest.mark.parametrize(
    "data, path",
    [
        ({"response_format": {"type": "json_object"}}, "response_format"),
        ({"n": 2}, "n"),
        ({"messages": []}, "messages"),
        ({"model": "m"}, "model"),
        ({"tools": []}, "tools"),
        ({"tool_choice": "auto"}, "tool_choice"),
        ({"parallel_tool_calls": False}, "parallel_tool_calls"),
        ({"stream": True}, "stream"),
        ({"stream_options": {"include_usage": True}}, "stream_options"),
    ],
)
def test_common_chat_reserved_path_is_built_by_yian(block_cls, data, path):
    message = _error(block_cls, data)
    assert f"`{block_cls.default_block}.{path}` is built by yian (" in message


@pytest.mark.parametrize(
    "block_cls, data, path",
    [
        (MiniMaxChatBlock, {"reasoning_split": True}, "minimax.reasoning_split"),
        (VLLMChatBlock, {"guided_json": {}}, "vllm.guided_json"),
        (VLLMChatBlock, {"guided_whitespace_pattern": "x"}, "vllm.guided_whitespace_pattern"),
        (VLLMChatBlock, {"structured_outputs": {}}, "vllm.structured_outputs"),
        (VLLMChatBlock, {"chat_template": "x"}, "vllm.chat_template"),
        (AzureChatBlock, {"data_sources": []}, "azure.data_sources"),
        (XAIChatBlock, {"search_parameters": {"mode": "on"}}, "xai.search_parameters"),
        (XAIChatBlock, {"deferred": True}, "xai.deferred"),
        (GLMChatBlock, {"tool_stream": True}, "glm.tool_stream"),
        (QwenChatBlock, {"tool_stream": True}, "qwen.tool_stream"),
        (OpenRouterChatBlock, {"plugins": []}, "openrouter.plugins"),
    ],
)
def test_vendor_reserved_path_is_built_by_yian(block_cls, data, path):
    message = _error(block_cls, data)
    assert f"`{path}` is built by yian (" in message
    assert "remove it" in message


@pytest.mark.parametrize(
    "block_cls, data, path",
    [
        (AzureChatBlock, {"prediction": {"type": "content", "content": "x"}}, "azure.prediction"),
        (
            OpenRouterChatBlock,
            {"prediction": {"type": "content", "content": "x"}},
            "openrouter.prediction",
        ),
        (GLMChatBlock, {"request_id": "req-123456"}, "glm.request_id"),
        (OpenRouterChatBlock, {"session_id": "session-1"}, "openrouter.session_id"),
        (OpenRouterChatBlock, {"trace": {"trace_id": "t-1"}}, "openrouter.trace"),
    ],
)
def test_per_request_fields_are_not_block_fields(block_cls, data, path):
    assert f"`{path}` is not a supported field of the" in _error(block_cls, data)


@pytest.mark.parametrize(
    "data",
    [
        {"output_config": {"effort": "high"}},
        {"thinking": {"type": "enabled", "budget_tokens": 2048}},
        {"thinking": {"type": "adaptive", "display": "omitted"}},
        {"thinking": {"type": "disabled"}},
        {"thinking": {"type": "between_tools"}},
        {"cache_control": {"type": "ephemeral"}},
    ],
)
def test_anthropic_field_is_accepted(data):
    assert PROVIDER_BLOCKS["anthropic"].model_validate(data).model_dump() == data


@pytest.mark.parametrize(
    "data",
    [
        {"output_config": {"effort": "hgih"}},
        {"thinking": {"type": "enabled", "budget_tokens": 100}},
        {"thinking": {"type": "enabled"}},
        {"thinking": {"type": "adaptive", "budget_tokens": 2048}},
        {"thinking": {"type": "disabled", "display": "omitted"}},
        {"thinking": {"type": "enabled", "budget_tokens": 2048, "display": "full"}},
        {"thinking": {"type": "sometimes"}},
        {"thinking": {}},
        {"thinking": "enabled"},
        {"max_tokens": 0},
        {"temperature": 1.5},
        {"top_p": -0.1},
        {"top_k": -1},
        {"service_tier": "priority"},
        {"cache_control": {"type": "ephemeral", "ttl": "2h"}},
        {"cache_control": {"ttl": "5m"}},
        {"metadata": {"user_id": "u" * 513}},
    ],
)
def test_anthropic_bad_value_is_rejected(data):
    with pytest.raises(ValidationError):
        AnthropicMessagesBlock.model_validate(data)


def test_thinking_variant_is_picked_by_type_for_key_checks():
    message = _error(
        AnthropicMessagesBlock, {"thinking": {"type": "adaptive", "budget_tokens": 2048}}
    )
    assert "`anthropic.thinking.budget_tokens` is not a supported field of the" in message
    assert "allowed: display, type" in message

    message = _error(
        AnthropicMessagesBlock, {"thinking": {"type": "enabled", "budget_token": 2048}}
    )
    assert "`anthropic.thinking.budget_token` is not a supported field of the" in message
    assert "allowed: budget_tokens, display, type" in message


def test_thinking_without_a_known_type_is_left_to_validation():
    message = _error(AnthropicMessagesBlock, {"thinking": {"type": SENTINEL, "budget": 1}})
    assert "`anthropic.thinking.budget`" not in message
    assert "adaptive, between_tools, disabled, enabled" in message
    assert SENTINEL not in message


@pytest.mark.parametrize(
    "data, path, reason",
    [
        (
            {"output_config": {"format": {"type": "json_schema"}}},
            "output_config.format",
            "structured output",
        ),
        ({"mcp_servers": []}, "mcp_servers", "tools"),
        ({"container": "c-1"}, "container", "tools"),
        ({"system": "be terse"}, "system", "system prompt"),
        ({"diagnostics": {}}, "diagnostics", "conversation state"),
        ({"messages": []}, "messages", "messages"),
        ({"model": "claude"}, "model", "llm_config.model"),
        ({"tools": []}, "tools", "tools"),
        ({"tool_choice": {"type": "auto"}}, "tool_choice", "tools"),
        ({"stream": True}, "stream", "streaming"),
    ],
)
def test_anthropic_reserved_path_is_built_by_yian(data, path, reason):
    message = _error(AnthropicMessagesBlock, data)
    assert f"`anthropic.{path}` is built by yian ({reason}); remove it" in message


@pytest.mark.parametrize("key", ["user_profile_id", "workspace_id", "temprature"])
def test_anthropic_fields_outside_the_body_list_are_unknown(key):
    message = _error(AnthropicMessagesBlock, {key: "x"})
    expected = f"`anthropic.{key}` is not a supported field of the Anthropic Messages API request"
    assert expected in message


@pytest.mark.parametrize(
    "data",
    [
        {"generationConfig": {"thinkingConfig": {"thinkingLevel": "HIGH"}}},
        {"generationConfig": {"thinkingConfig": {"thinkingBudget": 1024}}},
        {"generationConfig": {"thinkingConfig": {"thinkingBudget": -1}}},
        {"generationConfig": {"thinkingConfig": {"thinkingLevel": "MINIMAL"}}},
        {"safetySettings": [{"category": "HARM_CATEGORY_JAILBREAK", "threshold": "BLOCK_NONE"}]},
        {"serviceTier": "priority"},
    ],
)
def test_gemini_field_is_accepted(data):
    assert PROVIDER_BLOCKS["google"].model_validate(data).model_dump() == data


@pytest.mark.parametrize(
    "data",
    [
        {"generationConfig": {"thinkingConfig": {"thinkingLevel": "high"}}},
        {"generationConfig": {"thinkingConfig": {"thinkingLevel": "THINKING_LEVEL_UNSPECIFIED"}}},
        {"generationConfig": {"thinkingConfig": {"thinkingBudget": "1024"}}},
        {"generationConfig": {"temperature": 2.5}},
        {"generationConfig": {"logprobs": 21}},
        {"generationConfig": {"maxOutputTokens": 0}},
        {"generationConfig": {"stopSequences": ["a", "b", "c", "d", "e", "f"]}},
        {"generationConfig": {"mediaResolution": "LOW"}},
        {"safetySettings": [{"category": "HARM_CATEGORY_TOXICITY", "threshold": "OFF"}]},
        {"safetySettings": [{"category": "HARM_CATEGORY_HARASSMENT", "threshold": "block_none"}]},
        {"safetySettings": [{"category": "HARM_CATEGORY_HARASSMENT"}]},
        {"safetySettings": {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "OFF"}},
        {"serviceTier": "FLEX"},
        {"serviceTier": "unspecified"},
        {"store": "no"},
    ],
)
def test_gemini_bad_value_is_rejected(data):
    with pytest.raises(ValidationError):
        GeminiGenerateContentBlock.model_validate(data)


@pytest.mark.parametrize(
    "key",
    ["generation_config", "safety_settings", "service_tier", "labels", "temperature"],
)
def test_gemini_snake_case_and_other_keys_are_unknown(key):
    message = _error(GeminiGenerateContentBlock, {key: {}})
    expected = f"`google.{key}` is not a supported field of the Gemini generateContent request"
    assert expected in message
    assert "allowed: generationConfig, safetySettings, serviceTier, store" in message


@pytest.mark.parametrize(
    "key",
    ["responseModalities", "speechConfig", "imageConfig", "max_output_tokens", "thinking_config"],
)
def test_gemini_generation_config_rejects_unknown_keys(key):
    message = _error(GeminiGenerateContentBlock, {"generationConfig": {key: {}}})
    assert f"`google.generationConfig.{key}` is not a supported field of the" in message


def test_gemini_list_item_typo_names_its_index():
    message = _error(
        GeminiGenerateContentBlock,
        {
            "safetySettings": [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "OFF"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshhold": "OFF"},
            ]
        },
    )
    assert "`google.safetySettings[1].threshhold` is not a supported field of the" in message
    assert "allowed: category, threshold" in message
    assert "safetySettings[0]" not in message


def test_gemini_nested_typo_names_the_full_path():
    message = _error(
        GeminiGenerateContentBlock,
        {"generationConfig": {"thinkingConfig": {"thinking_level": "HIGH"}}},
    )
    path = "google.generationConfig.thinkingConfig.thinking_level"
    assert f"`{path}` is not a supported field of the" in message
    assert "allowed: includeThoughts, thinkingBudget, thinkingLevel" in message


@pytest.mark.parametrize(
    "data, path, reason",
    [
        ({"contents": []}, "contents", "messages"),
        ({"systemInstruction": {}}, "systemInstruction", "system prompt"),
        ({"tools": []}, "tools", "tools"),
        ({"toolConfig": {}}, "toolConfig", "tools"),
        ({"cachedContent": "cachedContents/1"}, "cachedContent", "system prompt and tools"),
        ({"continuationToken": "abc"}, "continuationToken", "conversation state"),
    ],
)
def test_gemini_reserved_path_is_built_by_yian(data, path, reason):
    message = _error(GeminiGenerateContentBlock, data)
    assert f"`google.{path}` is built by yian ({reason}); remove it" in message


@pytest.mark.parametrize(
    "key, value, reason",
    [
        ("responseSchema", {}, "structured output"),
        ("responseMimeType", "application/json", "structured output"),
        ("responseJsonSchema", {}, "structured output"),
        ("_responseJsonSchema", {}, "structured output"),
        ("responseFormat", {}, "structured output"),
        ("candidateCount", 2, "yian reads one reply"),
    ],
)
def test_gemini_generation_config_reserved_path_is_built_by_yian(key, value, reason):
    message = _error(GeminiGenerateContentBlock, {"generationConfig": {key: value}})
    assert f"`google.generationConfig.{key}` is built by yian ({reason}); remove it" in message


@pytest.mark.parametrize(
    "data",
    [
        {"options": {"num_ctx": 8192}},
        {"think": "high"},
        {"think": "max"},
        {"think": True},
        {"think": False},
        {"keep_alive": "5m"},
        {"keep_alive": 300},
        {"keep_alive": -1},
        {"options": {"mirostat": 2, "mirostat_tau": 5.0, "numa": False}},
        {"options": {"min_p": 0.05}},
    ],
)
def test_ollama_field_is_accepted(data):
    assert PROVIDER_BLOCKS["ollama"].model_validate(data).model_dump() == data


@pytest.mark.parametrize(
    "data",
    [
        {"think": "ultra"},
        {"think": "HIGH"},
        {"think": 1},
        {"keep_alive": True},
        {"keep_alive": ["5m"]},
        {"options": {"num_ctx": "8192"}},
        {"options": {"stop": "END"}},
        {"options": {"min_p": "0.05"}},
        {"top_logprobs": 1.5},
    ],
)
def test_ollama_bad_value_is_rejected(data):
    with pytest.raises(ValidationError):
        OllamaChatBlock.model_validate(data)


def test_ollama_block_fields_are_keywords_of_client_chat():
    ollama = pytest.importorskip("ollama")
    keywords = set(inspect.signature(ollama.Client.chat).parameters)
    assert set(OllamaChatBlock.model_fields) <= keywords


@pytest.mark.parametrize(
    "data, path, reason",
    [
        ({"format": "json"}, "format", "structured output"),
        ({"model": "llama3"}, "model", "llm_config.model"),
        ({"messages": []}, "messages", "messages"),
        ({"tools": []}, "tools", "tools"),
        ({"stream": False}, "stream", "streaming"),
    ],
)
def test_ollama_reserved_path_is_built_by_yian(data, path, reason):
    message = _error(OllamaChatBlock, data)
    assert f"`ollama.{path}` is built by yian ({reason}); remove it" in message


def test_ollama_option_at_top_level_is_unknown():
    message = _error(OllamaChatBlock, {"num_ctx": 8192})
    assert "`ollama.num_ctx` is not a supported field of the Ollama /api/chat request" in message
    assert "allowed: keep_alive, logprobs, options, think, top_logprobs" in message
    message = _error(OllamaChatBlock, {"options": {"num_ctxx": 8192}})
    assert "`ollama.options.num_ctxx` is not a supported field of the" in message


ALL_BLOCKS = sorted(
    set(PROVIDER_BLOCKS.values()) | {OpenAISubscriptionBlock}, key=lambda cls: cls.__name__
)


@pytest.mark.parametrize("block_cls", ALL_BLOCKS, ids=lambda cls: cls.__name__)
def test_errors_never_echo_values(block_cls):
    long_value = SENTINEL * 30
    fields = list(block_cls.model_fields)
    for data in (
        {"typo": SENTINEL, "model": SENTINEL, "stream": [SENTINEL]},
        {name: long_value for name in fields},
        {name: [long_value] for name in fields},
        {name: {"typo": long_value} for name in fields},
        {name: {"type": long_value} for name in fields},
    ):
        assert SENTINEL not in _error(block_cls, data)


@pytest.mark.parametrize("block_cls", ALL_BLOCKS, ids=lambda cls: cls.__name__)
def test_every_reserved_path_is_refused_with_its_reason(block_cls):
    for path, reason in block_cls.reserved.items():
        data = "x"
        for key in reversed(path.split(".")):
            data = {key: data}
        message = _error(block_cls, data)
        expected = f"`{block_cls.default_block}.{path}` is built by yian ({reason}); remove it"
        assert expected in message


@pytest.mark.parametrize("block_cls", [OpenRouterChatBlock, VLLMChatBlock, AzureChatBlock])
def test_logit_bias_takes_yaml_integer_or_digit_string_token_ids(block_cls):
    data = yaml.safe_load("logit_bias: {50256: -100, '1734': 5}")
    assert block_cls.model_validate(data).model_dump(mode="json") == {
        "logit_bias": {"50256": -100, "1734": 5}
    }
    for bad_key in ("abc", -1, True, 1.5):
        with pytest.raises(ValidationError):
            block_cls.model_validate({"logit_bias": {bad_key: 1}})
