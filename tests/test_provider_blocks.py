"""Provider request-body blocks: shape checks, error wording, dumps, and merging."""

import copy
import inspect

import pytest
from pydantic import ValidationError

from yian.llm_clients.config import SUPPORTED_LLM_PROVIDERS
from yian.llm_clients.provider_blocks import (
    PROVIDER_BLOCKS,
    AzureChatBlock,
    DeepSeekChatBlock,
    GLMChatBlock,
    MiniMaxChatBlock,
    OpenAIResponsesBlock,
    OpenAISubscriptionBlock,
    OpenRouterChatBlock,
    QwenChatBlock,
    VLLMChatBlock,
    XAIChatBlock,
    block_schema,
    merge_request_body,
)

SENTINEL = "sk-sentinel-7d2e91b4"

RESPONSES_SAMPLE = {
    "max_output_tokens": 2048,
    "max_tool_calls": 4,
    "metadata": {"team": "research"},
    "prompt_cache_key": "flow-a",
    "prompt_cache_retention": "24h",
    "reasoning": {"effort": "high", "summary": "auto"},
    "safety_identifier": "user-hash",
    "service_tier": "flex",
    "temperature": 0.2,
    "text": {"verbosity": "low"},
    "top_logprobs": 5,
    "top_p": 0.9,
    "truncation": "auto",
    "user": "someone",
}


def _error(block_cls, data, **kwargs) -> str:
    with pytest.raises(ValidationError) as caught:
        block_cls.model_validate(data, **kwargs)
    return str(caught.value)


def test_registry_maps_openai_to_the_responses_block():
    assert PROVIDER_BLOCKS["openai"] is OpenAIResponsesBlock
    assert set(PROVIDER_BLOCKS) == {
        "openai",
        "azure",
        "xai",
        "deepseek",
        "qwen",
        "qwen-cn",
        "glm",
        "glm-cn",
        "minimax",
        "minimax-cn",
        "openrouter",
        "vllm",
    }
    assert set(PROVIDER_BLOCKS) <= set(SUPPORTED_LLM_PROVIDERS)


def test_valid_responses_sample_dumps_unchanged():
    block = OpenAIResponsesBlock.model_validate(RESPONSES_SAMPLE)
    assert block.model_dump() == RESPONSES_SAMPLE
    assert block.model_dump(exclude_none=False) == RESPONSES_SAMPLE
    assert block.model_dump(mode="json") == RESPONSES_SAMPLE


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
    assert "`openai.temprature` is not a field of the OpenAI Responses API request" in message
    assert "allowed: max_output_tokens, max_tool_calls, metadata," in message
    assert "temperature" in message


def test_nested_typo_names_full_path_and_nested_allowed_keys():
    message = _error(OpenAIResponsesBlock, {"reasoning": {"efort": "high"}})
    assert "`openai.reasoning.efort` is not a field of the OpenAI Responses API request" in message
    assert "allowed: effort, summary" in message

    message = _error(OpenAIResponsesBlock, {"text": {"verbose": "low"}})
    assert "`openai.text.verbose`" in message
    assert "allowed: verbosity" in message


def test_every_problem_is_reported():
    message = _error(OpenAIResponsesBlock, {"temprature": 1, "store": False})
    assert "`openai.temprature`" in message
    assert "`openai.store`" in message


def test_block_name_comes_from_validation_context():
    message = _error(
        OpenAIResponsesBlock, {"top_k": 3}, context={"block": "openai_alias"}
    )
    assert "`openai_alias.top_k` is not a field" in message


@pytest.mark.parametrize(
    "data",
    [
        {"max_output_tokens": "x"},
        {"max_output_tokens": "5"},
        {"max_output_tokens": 0},
        {"max_output_tokens": True},
        {"reasoning": {"effort": "hgih"}},
        {"reasoning": {"summary": "brief"}},
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


@pytest.mark.parametrize(
    "data",
    [
        {"temperature": SENTINEL},
        {"metadata": {"k": SENTINEL * 40}},
        {"reasoning": {"effort": SENTINEL}},
        {"store": SENTINEL},
        {"text": {"format": SENTINEL}},
        {"user": [SENTINEL]},
        {"temprature": SENTINEL},
    ],
)
def test_errors_never_echo_values(data):
    assert SENTINEL not in _error(OpenAIResponsesBlock, data)


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


def test_subscription_fields_are_keywords_of_responses_create():
    responses = pytest.importorskip("openai.resources.responses")
    keywords = set(inspect.signature(responses.Responses.create).parameters)
    assert set(OpenAISubscriptionBlock.model_fields) <= keywords


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


CHAT_SAMPLES = {
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
        "request_id": "req-123456",
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
        "session_id": "session-1",
        "stop": "END",
        "temperature": 0.7,
        "top_a": 0.2,
        "top_k": 40,
        "top_p": 1,
        "trace": {"trace_id": "t-1", "custom": 3},
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
}


@pytest.mark.parametrize("block_cls", list(CHAT_SAMPLES), ids=lambda cls: cls.__name__)
def test_valid_chat_sample_dumps_unchanged(block_cls):
    sample = CHAT_SAMPLES[block_cls]
    block = block_cls.model_validate(sample)
    assert block.model_dump(exclude_none=False) == sample
    assert block.model_dump(mode="json") == sample


@pytest.mark.parametrize(
    "provider, data",
    [
        ("xai", {"reasoning_effort": "low"}),
        ("deepseek", {"thinking": {"type": "disabled"}}),
        ("deepseek", {"reasoning_effort": "max"}),
        ("qwen", {"enable_thinking": False}),
        ("qwen-cn", {"enable_thinking": True}),
        ("glm", {"thinking": {"type": "enabled"}}),
        ("glm-cn", {"thinking": {"type": "disabled"}}),
        ("minimax", {"thinking": {"type": "adaptive"}}),
        ("minimax-cn", {"reasoning_effort": "xhigh"}),
        ("openrouter", {"reasoning": {"effort": "minimal"}}),
        ("vllm", {"top_k": 50}),
        ("azure", {"reasoning_effort": "high"}),
    ],
)
def test_vendor_field_validates_through_the_registry(provider, data):
    assert PROVIDER_BLOCKS[provider].model_validate(data).model_dump() == data


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
        (GLMChatBlock, {"request_id": "abc"}),
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
        (OpenRouterChatBlock, {"session_id": "s" * 257}),
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


def test_shared_class_names_the_block_from_context():
    assert PROVIDER_BLOCKS["qwen"] is PROVIDER_BLOCKS["qwen-cn"] is QwenChatBlock
    assert PROVIDER_BLOCKS["glm"] is PROVIDER_BLOCKS["glm-cn"] is GLMChatBlock
    assert PROVIDER_BLOCKS["minimax"] is PROVIDER_BLOCKS["minimax-cn"] is MiniMaxChatBlock
    message = _error(QwenChatBlock, {"enable_thinkng": True}, context={"block": "qwen-cn"})
    assert "`qwen-cn.enable_thinkng` is not a field of the" in message
    assert "enable_thinking" in message.split("allowed:")[1]
    assert "`qwen.enable_thinkng`" in _error(QwenChatBlock, {"enable_thinkng": True})


def test_nested_chat_typo_names_the_full_path():
    message = _error(GLMChatBlock, {"thinking": {"typ": "enabled"}})
    assert "`glm.thinking.typ` is not a field of the" in message
    assert "allowed: clear_thinking, type" in message


@pytest.mark.parametrize("block_cls", list(CHAT_SAMPLES), ids=lambda cls: cls.__name__)
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


@pytest.mark.parametrize("block_cls", list(CHAT_SAMPLES), ids=lambda cls: cls.__name__)
def test_chat_block_docstring_names_its_reference(block_cls):
    assert "https://" in block_cls.__doc__
    assert "checked 2026-10-13" in " ".join(block_cls.__doc__.split())
    assert block_cls.api.startswith("is not a field of the ")


@pytest.mark.parametrize("block_cls", list(CHAT_SAMPLES), ids=lambda cls: cls.__name__)
def test_chat_errors_never_echo_values(block_cls):
    assert SENTINEL not in _error(block_cls, {"temperature": SENTINEL, "stop": {SENTINEL: 1}})
    assert SENTINEL not in _error(block_cls, {"typo": SENTINEL, "n": SENTINEL})
