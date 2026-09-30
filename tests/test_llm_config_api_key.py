"""`LLMConfig.api_key`: a host-supplied provider key that no dump, repr, or error shows."""

import json
import sys

import httpx
import pytest
from langchain_core.messages import HumanMessage
from pydantic import SecretStr, ValidationError

from agent_composer.llm_clients.auth import AuthError, AuthStore, OAuthCredentials
from agent_composer.llm_clients.config import LLMConfig
from agent_composer.llm_clients.connection import (
    connection_credentials,
    resolve_model_config,
    split_model_config,
    validate_connection,
)
from agent_composer.llm_clients.factory import model_from_config
from tests.fixtures.openai_responses import ResponsesServer

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


@pytest.mark.parametrize("empty", ["", SecretStr("")])
def test_an_empty_key_is_rejected_instead_of_falling_back(clean_env, empty):
    clean_env.setenv("GOOGLE_API_KEY", "sk-environment-key")
    with pytest.raises(ValidationError, match="api_key"):
        LLMConfig(provider="google", api_key=empty)
    with pytest.raises(ValidationError, match="api_key"):
        model_from_config({"provider": "google", "model": "gemini-2.5-flash", "api_key": empty})


def test_split_and_resolve_keep_the_key_as_a_secret():
    config = LLMConfig(provider="anthropic", model="claude-sonnet-4-5", api_key=SENTINEL)
    validated, _ = split_model_config(config)
    assert validated.api_key.get_secret_value() == SENTINEL
    resolved = resolve_model_config({**config.model_dump(exclude_none=True), "api_key": SENTINEL})
    assert isinstance(resolved["api_key"], SecretStr)
    assert resolved["api_key"].get_secret_value() == SENTINEL
    assert SENTINEL not in repr(resolved)


# --- clients prefer the config key ------------------------------------------ #

_ENV_KEYS = (
    "OPENAI_API_KEY", "OPENAI_BASE_URL", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY",
    "AZURE_OPENAI_API_KEY", "VLLM_API_KEY", "VLLM_BASE_URL", "DEEPSEEK_API_KEY",
    "OPENROUTER_API_KEY", "AGENT_COMPOSER_DEFAULT_PROVIDER", "AGENT_COMPOSER_DEFAULT_MODEL",
)


@pytest.fixture
def clean_env(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_COMPOSER_AUTH_FILE", str(tmp_path / "auth.json"))
    for name in _ENV_KEYS:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def _chat_completions(requests):
    """Answer every chat-completions request with one fixed reply."""

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={
            "id": "c", "object": "chat.completion", "created": 0, "model": "m",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "ok"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        })

    return httpx.MockTransport(handler)


def _inject_http_client(monkeypatch, client_cls, http_client):
    original = client_cls.get_llm

    def construct(wrapper):
        wrapper.kwargs.update(http_client=http_client, max_retries=0)
        return original(wrapper)

    monkeypatch.setattr(client_cls, "get_llm", construct)


@pytest.mark.parametrize(
    "provider, model, env, attribute",
    [
        ("openai", "gpt-5.5", "OPENAI_API_KEY", "openai_api_key"),
        ("vllm", "served-model", "VLLM_API_KEY", "openai_api_key"),
        ("deepseek", "deepseek-chat", "DEEPSEEK_API_KEY", "openai_api_key"),
        ("openrouter", "some/model", "OPENROUTER_API_KEY", "openai_api_key"),
        ("anthropic", "claude-sonnet-4-5", "ANTHROPIC_API_KEY", "anthropic_api_key"),
        ("google", "gemini-2.5-flash", "GOOGLE_API_KEY", "google_api_key"),
        ("azure", "gpt-4o", "AZURE_OPENAI_API_KEY", "openai_api_key"),
    ],
)
@pytest.mark.parametrize("env_value", [None, "sk-environment-key"])
def test_every_keyed_client_prefers_the_config_key(clean_env, provider, model, env, attribute, env_value):
    if provider == "azure":
        clean_env.setenv("OPENAI_API_VERSION", "2024-02-01")
        clean_env.setenv("AZURE_OPENAI_ENDPOINT", "https://res.openai.azure.com")
    if env_value is not None:
        clean_env.setenv(env, env_value)
    chat = model_from_config({"provider": provider, "model": model, "api_key": SENTINEL})
    assert getattr(chat, attribute).get_secret_value() == SENTINEL


def test_openai_compatible_endpoint_receives_the_config_key(clean_env):
    from agent_composer.llm_clients.openai_client import OpenAIClient

    clean_env.setenv("VLLM_BASE_URL", "http://127.0.0.1:9/v1")
    requests = []
    with httpx.Client(transport=_chat_completions(requests)) as http_client:
        _inject_http_client(clean_env, OpenAIClient, http_client)
        reply = model_from_config(
            {"provider": "vllm", "model": "served-model", "api_key": SENTINEL}
        ).invoke([HumanMessage("hi")])
    assert reply.content == "ok"
    assert requests[0].headers["Authorization"] == f"Bearer {SENTINEL}"


def test_openai_config_key_wins_a_saved_subscription(clean_env):
    from agent_composer.llm_clients.openai_client import OpenAIClient

    AuthStore().save(
        "openai",
        OAuthCredentials("access-secret", "refresh-secret", 4102444800, "account"),
        "gpt-5.5",
    )
    resolved = resolve_model_config({"provider": "openai", "api_key": SENTINEL})
    assert resolved["_connection"]["source"] == "host"
    assert resolved["_connection"]["credential_id"] is None
    assert SENTINEL not in json.dumps(resolved["_connection"])
    server = ResponsesServer(tool_first=False)
    with httpx.Client(transport=httpx.MockTransport(server)) as http_client:
        _inject_http_client(clean_env, OpenAIClient, http_client)
        model_from_config(resolved).invoke("hi")
    assert server.requests[0].url.host == "api.openai.com"
    assert server.requests[0].headers["Authorization"] == f"Bearer {SENTINEL}"


def test_host_connection_needs_the_config_key(clean_env):
    connection = validate_connection({
        "method": "api_key", "source": "host", "credential_id": None,
        "endpoint": "https://api.openai.com/v1", "store_path": None,
    })
    with pytest.raises(AuthError, match="api_key"):
        connection_credentials(connection)
    with pytest.raises(ValueError):
        validate_connection({**connection, "credential_id": "a" * 64})
    with pytest.raises(ValueError):
        validate_connection({**connection, "method": "subscription"})


@pytest.mark.parametrize("provider, model", [("vllm", "served-model"), ("anthropic", "claude-sonnet-4-5")])
def test_a_host_key_marker_without_the_key_fails_before_the_environment(clean_env, provider, model):
    clean_env.setenv("VLLM_API_KEY", "sk-environment-key")
    clean_env.setenv("ANTHROPIC_API_KEY", "sk-environment-key")
    marked = {"provider": provider, "model": model, "_host_key": True}
    with pytest.raises(AuthError, match="llm_config.api_key"):
        model_from_config(marked)
    validated, _ = split_model_config({**marked, "api_key": SENTINEL})
    assert validated.api_key.get_secret_value() == SENTINEL
    with pytest.raises(ValueError, match="marker"):
        split_model_config({**marked, "_host_key": "yes", "api_key": SENTINEL})


def _raise_chained(key):
    try:
        raise RuntimeError(f"Incorrect API key provided: {key[:8]}****{key[-4:]}.")
    except RuntimeError as inner:
        raise ValueError(f"wrapped: {key}") from inner


def test_scrub_key_removes_the_key_and_its_masked_fragments_from_a_chain():
    import traceback

    from agent_composer.llm_clients.redact import scrub_key

    key = "sk-sentinel-scrub-zqxw"
    with pytest.raises(KeyError) as caught:
        try:
            _raise_chained(key)
        finally:
            raise KeyError(key)
    scrub_key(caught.value, SecretStr(key))
    text = "".join(traceback.format_exception(caught.value))
    for fragment in (key, key[:8], "*" + key[-4:]):
        assert fragment not in text
    assert "Incorrect API key provided: [REDACTED]." in text
    assert "wrapped: [REDACTED]" in text


@pytest.mark.skipif(sys.version_info < (3, 11), reason="exception groups and notes need 3.11")
def test_scrub_key_reaches_exception_group_members_and_notes():
    import traceback

    from agent_composer.llm_clients.redact import scrub_key

    key = "sk-sentinel-scrub-zqxw"
    member = RuntimeError(f"member {key}")
    member.add_note(f"note {key[:8]}***")
    group = ExceptionGroup("group", [member])  # noqa: F821 - builtin on 3.11+
    scrub_key(group, key)
    text = "".join(traceback.format_exception(group))
    assert key not in text and key[:8] not in text
    assert "note [REDACTED]" in text


def test_scrub_key_leaves_unrelated_masks_alone():
    from agent_composer.llm_clients.redact import redact_key

    assert redact_key("token ab****yz and **bold**", "sk-sentinel-zqxw") == (
        "token ab****yz and **bold**"
    )
    assert redact_key("anything", None) == "anything"
