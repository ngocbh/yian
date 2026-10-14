"""Provider request-body blocks: shape checks, error wording, dumps, and merging."""

import copy
import inspect

import pytest
from pydantic import ValidationError

from yian.llm_clients.provider_blocks import (
    PROVIDER_BLOCKS,
    OpenAIResponsesBlock,
    OpenAISubscriptionBlock,
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
    assert PROVIDER_BLOCKS == {"openai": OpenAIResponsesBlock}


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
