"""Per-agent `llm_config` resolution in `llm_clients.layers`, as the cascade applies it."""

import pytest

from yian.llm_clients.connection import HOST_KEY_MARKER, missing_model_message

_DEFAULTS = {"provider": "anthropic", "model": "default-model"}


def _resolve(authored, *, defaults=None, provider_models=None, stray_model=False, label=None):
    from yian.llm_clients.layers import resolve_agent_config

    return resolve_agent_config(
        authored,
        defaults=_DEFAULTS if defaults is None else defaults,
        provider_models=provider_models or {},
        stray_model=stray_model,
        path="review/chat",
        label=label,
    )


def test_only_the_default_provider_gives_the_default_model():
    assert _resolve({}) == {"provider": "anthropic", "model": "default-model"}


def test_defaults_other_than_model_fill_gaps_under_the_authored_layer():
    defaults = {**_DEFAULTS, "anthropic": {"temperature": 0.1, "max_tokens": 10}}

    config = _resolve({"anthropic": {"temperature": 0.5}}, defaults=defaults)

    assert config == {
        "provider": "anthropic",
        "model": "default-model",
        "anthropic": {"temperature": 0.5, "max_tokens": 10},
    }


def test_an_authored_model_wins_over_saved_and_default_models():
    config = _resolve(
        {"provider": "openai", "model": "gpt-authored"},
        provider_models={"openai": "gpt-saved"},
    )

    assert config == {"provider": "openai", "model": "gpt-authored"}


def test_a_saved_model_serves_an_authored_provider():
    config = _resolve({"provider": "openai"}, provider_models={"openai": "gpt-saved"})

    assert config == {"provider": "openai", "model": "gpt-saved"}


def test_the_default_model_never_serves_another_provider():
    with pytest.raises(ValueError) as caught:
        _resolve({"provider": "openai"})

    assert str(caught.value) == (
        "Agent 'review/chat' has no model for provider 'openai'; set "
        "llm_config.provider/model, pass --provider/--model, or run 'yian auth openai'"
    )


def test_a_missing_model_names_the_label():
    with pytest.raises(ValueError) as caught:
        _resolve({"provider": "openai"}, label="adaptive_questions.llm_config of node 'ask'")

    assert str(caught.value).startswith(
        "Adaptive_questions.llm_config of node 'ask' has no model for provider 'openai'"
    )


def test_a_missing_provider_names_the_path():
    with pytest.raises(ValueError) as caught:
        _resolve({}, defaults={})

    assert str(caught.value).startswith("Agent 'review/chat' has no provider or model;")


def test_a_stray_default_model_is_named():
    with pytest.raises(ValueError) as caught:
        _resolve({}, defaults={}, stray_model=True)

    assert str(caught.value) == missing_model_message("review/chat", None, stray_model=True)
    assert "YIAN_DEFAULT_MODEL" in str(caught.value)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (HOST_KEY_MARKER, True, "internal to recorded runs"),
        ("base_url", "https://example.test/v1", "base_url"),
    ],
)
def test_a_host_marker_or_base_url_is_rejected(field, value, message):
    with pytest.raises(ValueError, match=message):
        _resolve({"provider": "openai", "model": "m", field: value})


def test_the_result_matches_the_cascade_and_never_aliases_its_inputs():
    from yian.compile.llm_cascade import resolve_llm_cascade
    from yian.compose.loader import load_flow

    text = """
id: f
name: f
llm_config: {provider: openai, openai: {temperature: 0.3}}
nodes:
  chat: {kind: agent, prompt: hi, llm_config: {openai: {max_output_tokens: 5}}}
output: ${chat.output}
"""
    flow = load_flow(text).compiled
    expected = resolve_llm_cascade(
        flow, {}, defaults=_DEFAULTS, provider_models={"openai": "gpt-saved"}
    )["chat"]
    authored = {"provider": "openai", "openai": {"temperature": 0.3, "max_output_tokens": 5}}

    config = _resolve(authored, provider_models={"openai": "gpt-saved"})

    assert config == expected
    config["openai"]["temperature"] = 9
    assert authored["openai"]["temperature"] == 0.3
