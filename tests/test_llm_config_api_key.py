"""`LLMConfig.api_key`: a host-supplied provider key that no dump, repr, or error shows."""

import pytest
from pydantic import SecretStr, ValidationError

from agent_composer.llm_clients.config import LLMConfig
from agent_composer.llm_clients.connection import resolve_model_config, split_model_config

SENTINEL = "sk-sentinel-3f9c1a7e"


def test_repr_str_and_dumps_hide_the_key():
    config = LLMConfig(provider="anthropic", model="claude-sonnet-4-5", api_key=SENTINEL)
    assert config.api_key.get_secret_value() == SENTINEL
    for text in (
        repr(config),
        str(config),
        repr(config.model_dump()),
        repr(config.model_dump(mode="json")),
        config.model_dump_json(),
    ):
        assert SENTINEL not in text
    assert "api_key" not in config.model_dump()
    assert "api_key" not in config.model_dump(mode="json", exclude_none=False)


def test_the_key_field_does_not_change_dumps_of_configs_without_it():
    # Recorded snapshots compare full dumps, so an unset key must add nothing.
    dumped = LLMConfig(provider="openai", model="gpt-5.5").model_dump(exclude_none=False)
    assert "api_key" not in dumped


@pytest.mark.parametrize(
    "fields",
    [
        {"temperature": "hot"},
        {"temparature": 0.2},
        {"provider": "nope"},
        {"api_key": 123},
    ],
)
def test_validation_errors_do_not_echo_config_values(fields):
    with pytest.raises(ValidationError) as caught:
        LLMConfig(**{"api_key": SENTINEL, **fields})
    assert SENTINEL not in str(caught.value)
    assert SENTINEL not in repr(caught.value)
    for value in fields.values():
        assert str(value) not in str(caught.value).replace("temparature", "")


def test_split_and_resolve_keep_the_key_as_a_secret():
    config = LLMConfig(provider="anthropic", model="claude-sonnet-4-5", api_key=SENTINEL)
    validated, _ = split_model_config(config)
    assert validated.api_key.get_secret_value() == SENTINEL
    resolved = resolve_model_config({**config.model_dump(exclude_none=True), "api_key": SENTINEL})
    assert isinstance(resolved["api_key"], SecretStr)
    assert resolved["api_key"].get_secret_value() == SENTINEL
    assert SENTINEL not in repr(resolved)
