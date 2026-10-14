"""Provider blocks on `LLMConfig`: validation, aliases, dumps, and cascade merging."""

import copy
import typing

import pytest
from pydantic import ValidationError

from yian.compile.llm_cascade import _validate_layer
from yian.llm_clients import LLM_CONFIG_KEYS
from yian.llm_clients.config import LLMConfig, SUPPORTED_LLM_PROVIDERS, merge_llm_config
from yian.llm_clients.provider_blocks import PROVIDER_BLOCKS

SENTINEL = "sk-sentinel-5f0c2a9e"


def _error(data) -> str:
    with pytest.raises(ValidationError) as caught:
        LLMConfig.model_validate(data)
    return str(caught.value)


def test_every_provider_has_a_block_field_of_its_block_class():
    fields = {field.alias or name: field for name, field in LLMConfig.model_fields.items()}
    for provider in SUPPORTED_LLM_PROVIDERS:
        assert typing.get_args(fields[provider].annotation) == (PROVIDER_BLOCKS[provider], type(None))


def test_llm_config_keys_are_spelled_as_authored():
    assert LLM_CONFIG_KEYS == frozenset(
        {
            "provider",
            "model",
            "api_key",
            "base_url",
            "anthropic_effort",
            "openai_reasoning_effort",
            "google_thinking_level",
            "temperature",
            *SUPPORTED_LLM_PROVIDERS,
        }
    )
    assert "qwen_cn" not in LLM_CONFIG_KEYS


def test_block_of_the_selected_provider_is_valid():
    config = LLMConfig.model_validate(
        {"provider": "anthropic", "anthropic": {"max_tokens": 8000, "temperature": 0.2}}
    )
    assert config.anthropic.max_tokens == 8000
    assert config.block_for("anthropic") == {"max_tokens": 8000, "temperature": 0.2}


def test_block_of_another_provider_is_still_checked():
    message = _error({"provider": "anthropic", "openai": {"temprature": 0.2}})
    assert "`openai.temprature` is not a supported field" in message


def test_block_contents_are_checked_by_type():
    message = _error({"anthropic": {"max_tokens": "8000"}})
    assert "anthropic.max_tokens" in message


def test_dash_block_is_accepted_and_dumped_by_its_authored_name():
    config = LLMConfig(**{"provider": "qwen-cn", "qwen-cn": {"enable_thinking": True}})
    assert config.model_dump()["qwen-cn"] == {"enable_thinking": True}
    assert "qwen_cn" not in config.model_dump()
    assert config.block_for("qwen-cn") == {"enable_thinking": True}


def test_dash_block_typo_names_the_block():
    message = _error({"qwen-cn": {"enable_thinkng": True}})
    assert "`qwen-cn.enable_thinkng` is not a supported field" in message


@pytest.mark.parametrize(
    ("python_name", "authored"),
    [("qwen_cn", "qwen-cn"), ("glm_cn", "glm-cn"), ("minimax_cn", "minimax-cn")],
)
def test_python_spelling_of_a_dash_block_is_rejected(python_name, authored):
    assert f"write `{authored}`" in _error({python_name: {}})
    with pytest.raises(ValidationError, match=f"write `{authored}`"):
        LLMConfig(**{python_name: {}})


def test_keyword_construction_still_works():
    config = LLMConfig(provider="openai", model="gpt-5.4", api_key="sk-x")
    assert config.provider == "openai"
    assert config.api_key.get_secret_value() == "sk-x"


def test_unset_blocks_are_absent_from_every_dump():
    config = LLMConfig(provider="openai", model="gpt-5.4")
    for dump in (config.model_dump(), config.model_dump(exclude_none=False)):
        assert not set(dump) & set(SUPPORTED_LLM_PROVIDERS)


def test_set_block_dumps_without_none_fields_at_any_depth():
    config = LLMConfig.model_validate(
        {"openai": {"reasoning": {"effort": "high", "summary": None}, "temperature": None}}
    )
    assert config.model_dump(exclude_none=False)["openai"] == {"reasoning": {"effort": "high"}}
    assert config.model_dump(mode="json")["openai"] == {"reasoning": {"effort": "high"}}


def test_block_for_an_unset_block_is_empty():
    assert LLMConfig(provider="openai").block_for("openai") == {}
    assert LLMConfig(provider="openai").block_for("glm-cn") == {}


def test_block_for_returns_a_fresh_dict():
    config = LLMConfig.model_validate({"glm-cn": {"thinking": {"type": "enabled"}}})
    config.block_for("glm-cn")["thinking"]["type"] = "disabled"
    assert config.block_for("glm-cn") == {"thinking": {"type": "enabled"}}


def test_dump_round_trips():
    config = LLMConfig.model_validate(
        {
            "provider": "google",
            "model": "gemini-3-pro",
            "temperature": 0.5,
            "google": {"generationConfig": {"thinkingConfig": {"thinkingLevel": "HIGH"}}},
            "minimax-cn": {"thinking": {"type": "adaptive"}},
            "ollama": {"think": True},
        }
    )
    assert LLMConfig(**config.model_dump()) == config
    assert LLMConfig(**config.model_dump(exclude_none=True)) == config


@pytest.mark.parametrize(
    "data",
    [
        {"openai": {"metadata": {"team": 1}, "user": SENTINEL}},
        {"openai": {"service_tier": SENTINEL}},
        {"anthropic": {"max_tokens": SENTINEL}},
        {"qwen-cn": {"reasoning_effort": SENTINEL}},
        {"glm": SENTINEL},
        {"google": {"safetySettings": [{"category": SENTINEL, "threshold": "OFF"}]}},
    ],
)
def test_errors_do_not_echo_block_values(data):
    assert SENTINEL not in _error(data)


# -- cascade merge ----------------------------------------------------------------


def test_blocks_merge_per_field():
    specific = {"anthropic": {"temperature": 0.2}}
    base = {"anthropic": {"max_tokens": 8000, "temperature": 1.0}, "provider": "anthropic"}
    assert merge_llm_config(specific, base) == {
        "provider": "anthropic",
        "anthropic": {"max_tokens": 8000, "temperature": 0.2},
    }


def test_nested_block_values_are_replaced_whole():
    specific = {"anthropic": {"thinking": {"type": "adaptive"}}}
    base = {"anthropic": {"thinking": {"type": "enabled", "budget_tokens": 2048}, "top_k": 5}}
    assert merge_llm_config(specific, base) == {
        "anthropic": {"thinking": {"type": "adaptive"}, "top_k": 5},
    }


def test_none_block_fields_are_dropped():
    specific = {"openai": {"temperature": None, "top_p": 0.5}}
    base = {"openai": {"temperature": 0.7, "user": None}}
    assert merge_llm_config(specific, base) == {"openai": {"temperature": 0.7, "top_p": 0.5}}


def test_none_block_keeps_the_base_block():
    base = {"openai": {"temperature": 0.7}}
    assert merge_llm_config({"openai": None}, base) == base


def test_non_mapping_specific_block_replaces_whole():
    assert merge_llm_config({"openai": "high"}, {"openai": {"temperature": 0.7}}) == {
        "openai": "high"
    }


def test_non_mapping_base_block_is_kept_for_validation():
    # A malformed host block must reach validation, not be hidden by an authored one.
    assert merge_llm_config({"openai": {"temperature": 0.7}}, {"openai": "high"}) == {
        "openai": "high"
    }


def test_connection_is_replaced_whole():
    specific = {"_connection": {"method": "api_key"}}
    base = {"_connection": {"method": "subscription", "source": "store"}}
    assert merge_llm_config(specific, base) == specific


def test_dash_block_merges_across_layers():
    specific = {"qwen-cn": {"enable_thinking": False}}
    base = {"qwen-cn": {"enable_thinking": True, "thinking_budget": 512}}
    assert merge_llm_config(specific, base) == {
        "qwen-cn": {"enable_thinking": False, "thinking_budget": 512}
    }


def test_merge_does_not_mutate_inputs():
    specific = {"openai": {"reasoning": {"effort": "low"}, "top_p": None}, "model": "m"}
    base = {"openai": {"reasoning": {"summary": "auto"}, "temperature": 0.2}, "model": None}
    snapshot = copy.deepcopy((specific, base))
    merged = merge_llm_config(specific, base)
    merged["openai"]["temperature"] = 1.5
    assert (specific, base) == snapshot


_LAYERS = [
    {},
    {"provider": "openai", "model": None},
    {"openai": None},
    {"openai": {"temperature": 0.2, "user": None}},
    {"openai": {"reasoning": {"effort": "high"}, "temperature": None}},
    {"openai": {"reasoning": {"summary": "auto"}, "top_p": 0.5}},
    {"openai": {}},
    {"openai": "not-a-block"},
    {"openai": ["also", "not"]},
    {"anthropic": {"thinking": {"type": "enabled", "budget_tokens": 2048}}},
    {"anthropic": {"thinking": {"type": "adaptive"}, "max_tokens": 100}},
    {"qwen-cn": {"enable_thinking": True}, "_connection": {"method": "api_key"}},
    {"qwen-cn": {"enable_thinking": None, "top_k": 4}, "_connection": {"method": "x"}},
    {"temperature": 0.3, "model": "m"},
]


@pytest.mark.parametrize("node", _LAYERS)
@pytest.mark.parametrize("flow", _LAYERS)
@pytest.mark.parametrize("host", _LAYERS)
def test_merge_is_associative(node, flow, host):
    left = merge_llm_config(node, merge_llm_config(flow, host))
    right = merge_llm_config(merge_llm_config(node, flow), host)
    assert left == right


# -- host layer keys --------------------------------------------------------------


def test_host_layer_rejects_the_python_spelling_of_a_dash_block():
    with pytest.raises(ValueError, match=r"unknown field\(s\) \['qwen_cn'\]") as caught:
        _validate_layer({"qwen_cn": {"enable_thinking": True}})
    assert "qwen-cn" in str(caught.value)
    assert ", ".join(sorted(LLM_CONFIG_KEYS)) in str(caught.value)


def test_host_layer_accepts_a_dash_block():
    _validate_layer({"qwen-cn": {"enable_thinking": True}})
