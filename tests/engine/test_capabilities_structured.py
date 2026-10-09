"""`supports_native_structured` — provider+model gate for native structured output."""

from yian.llm_clients.capabilities import supports_native_structured


def test_anthropic_openai_google_support_native():
    assert supports_native_structured("anthropic", "claude-sonnet-4-5") is True
    assert supports_native_structured("openai", "gpt-5.5") is True
    assert supports_native_structured("google", "gemini-3-pro") is True


def test_known_no_native_model_returns_false():
    # the catalog sentinel — a single entry with preferred_structured_method="none"
    # so the prompt-injection fallback is reachable through the real source.
    assert supports_native_structured("vllm", "no-structured-sentinel") is False


def test_unknown_defaults_to_native():
    assert supports_native_structured("some-future-provider", "some-model") is True
