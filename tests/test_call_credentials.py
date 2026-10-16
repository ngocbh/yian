"""Every credential a model call sends is found for scrubbing, and nothing else."""

import time

import pytest
from pydantic import SecretStr

from yian.llm_clients.api_key_env import PROVIDER_API_KEY_ENV
from yian.llm_clients.auth import ApiKeyCredentials, AuthStore, OAuthCredentials
from yian.llm_clients.config import LLMConfig
from yian.llm_clients.connection import (
    call_credentials,
    legacy_connection_config,
    resolve_model_config,
    scrub_call_credentials,
)

AZURE_URL = "https://r.openai.azure.com/"


@pytest.fixture(autouse=True)
def clean_env(tmp_path, monkeypatch):
    monkeypatch.setenv("YIAN_AUTH_FILE", str(tmp_path / "auth" / "auth.json"))
    names = [name for name in PROVIDER_API_KEY_ENV.values() if name]
    names += [
        "GEMINI_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "AZURE_OPENAI_AD_TOKEN",
        "YIAN_DEFAULT_PROVIDER",
        "YIAN_DEFAULT_MODEL",
    ]
    for name in names:
        monkeypatch.delenv(name, raising=False)


def _resolved(provider, model, **fields):
    config = resolve_model_config({"provider": provider, "model": model, **fields})
    assert "_connection" in config
    return config


def test_host_key_is_the_only_key_read(monkeypatch):
    monkeypatch.setenv("VLLM_API_KEY", "sk-env-vllm")
    host = "sk-host-sentinel-zqxw"
    assert call_credentials({"provider": "vllm", "model": "m", "api_key": host}) == (host,)
    assert call_credentials(
        {"provider": "vllm", "model": "m", "api_key": SecretStr(host)}
    ) == (host,)
    assert call_credentials(
        LLMConfig(provider="vllm", model="m", api_key=host)
    ) == (host,)


def test_provider_key_variable(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-env-anthropic")
    config = _resolved("anthropic", "claude-x")
    assert config["_connection"]["source"] == "environment"
    assert call_credentials(config) == ("sk-env-anthropic",)


def test_openai_key_variable(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env-openai")
    config = _resolved("openai", "gpt-x")
    assert config["_connection"]["source"] == "environment"
    assert call_credentials(config) == ("sk-env-openai",)
    legacy = legacy_connection_config({"provider": "openai", "model": "gpt-x"})
    assert legacy["_connection"]["source"] == "legacy_environment"
    assert call_credentials(legacy) == ("sk-env-openai",)


def test_saved_api_key_login_wins_the_variable(monkeypatch):
    AuthStore().save("anthropic", ApiKeyCredentials("sk-saved-anthropic"), "claude-x")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-env-anthropic")
    config = _resolved("anthropic", "claude-x")
    assert config["_connection"]["source"] == "store"
    assert call_credentials(config) == ("sk-saved-anthropic",)


@pytest.mark.parametrize("expires_in", [3600, -3600], ids=["fresh", "expired"])
def test_saved_subscription_tokens(expires_in):
    AuthStore().save(
        "openai",
        OAuthCredentials("access-secret", "refresh-secret", time.time() + expires_in, "acct"),
        "gpt-x",
    )
    config = _resolved("openai", "gpt-x")
    assert config["_connection"]["method"] == "subscription"
    assert set(call_credentials(config)) == {"access-secret", "refresh-secret"}


def test_sdk_fallback_variables(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "sk-env-gemini")
    assert call_credentials(_resolved("google", "gemini-x")) == ("sk-env-gemini",)
    monkeypatch.setenv("GOOGLE_API_KEY", "sk-env-google")
    assert call_credentials(_resolved("google", "gemini-x")) == ("sk-env-google",)

    monkeypatch.setenv("OPENAI_API_KEY", "sk-env-openai")
    azure = _resolved("azure", "dep", base_url=AZURE_URL)
    assert azure["_connection"]["source"] == "environment"
    assert call_credentials(azure) == ("sk-env-openai",)


def test_always_sent_variables(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "sk-env-azure")
    monkeypatch.setenv("AZURE_OPENAI_AD_TOKEN", "ad-token-secret")
    azure = _resolved("azure", "dep", base_url=AZURE_URL)
    assert set(call_credentials(azure)) == {"sk-env-azure", "ad-token-secret"}
    hosted = {"provider": "azure", "model": "dep", "base_url": AZURE_URL, "api_key": "sk-host"}
    assert set(call_credentials(hosted)) == {"sk-host", "ad-token-secret"}

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-env-anthropic")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "auth-token-secret")
    assert set(call_credentials(_resolved("anthropic", "claude-x"))) == {
        "sk-env-anthropic", "auth-token-secret",
    }


def test_environment_header_values_are_not_listed(monkeypatch):
    # The clients drop SDK header variables, so their values are never sent.
    monkeypatch.setenv("OPENAI_CUSTOM_HEADERS", "Authorization: Bearer custom-secret")
    monkeypatch.setenv("ANTHROPIC_CUSTOM_HEADERS", "Authorization: Bearer custom-secret")
    for provider in ("openai", "anthropic"):
        config = {"provider": provider, "model": "m", "api_key": "sk-host"}
        assert call_credentials(config) == ("sk-host",)


def test_unresolved_config_is_resolved_first():
    AuthStore().save("anthropic", ApiKeyCredentials("sk-saved-anthropic"), "claude-x")
    assert call_credentials({"provider": "anthropic", "model": "c"}) == ("sk-saved-anthropic",)


def _malformed_store(tmp_path):
    path = tmp_path / "broken" / "auth.json"
    path.parent.mkdir(mode=0o700)
    path.write_text("{not json")
    path.chmod(0o600)
    return path


def test_a_malformed_store_gives_no_store_secrets(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-env-anthropic")
    path = _malformed_store(tmp_path)
    config = {
        "provider": "anthropic",
        "model": "c",
        "_connection": {
            "method": "api_key",
            "source": "store",
            "credential_id": "abc",
            "endpoint": None,
            "store_path": str(path),
        },
    }
    assert call_credentials(config) == ()
    scrub_call_credentials(RuntimeError("boom"), config)

    monkeypatch.setenv("YIAN_AUTH_FILE", str(path))
    unresolved = {"provider": "anthropic", "model": "c"}
    call_credentials(unresolved)
    scrub_call_credentials(RuntimeError("boom"), unresolved)


def test_an_invalid_provider_still_gives_the_host_key():
    config = {"provider": "nope", "api_key": "sk-x"}
    assert call_credentials(config) == ("sk-x",)
    scrub_call_credentials(RuntimeError("boom"), config)


@pytest.mark.parametrize("config", [None, {}])
def test_an_empty_config_uses_the_saved_default(config):
    assert call_credentials(config) == ()
    scrub_call_credentials(RuntimeError("boom"), config)
    AuthStore().save("anthropic", ApiKeyCredentials("sk-saved-anthropic"), "claude-x")
    assert call_credentials(config) == ("sk-saved-anthropic",)
    scrub_call_credentials(RuntimeError("boom"), config)


@pytest.mark.parametrize("config", ["not a config", 5, {"_connection": "junk"}, {"provider": 3}])
def test_garbage_configs_never_raise(config):
    assert call_credentials(config) == ()
    scrub_call_credentials(RuntimeError("boom"), config)


def test_scrub_call_credentials_removes_a_masked_saved_key():
    key = "sk-saved-sentinel-zqxw"
    AuthStore().save("anthropic", ApiKeyCredentials(key), "claude-x")
    config = _resolved("anthropic", "claude-x")
    try:
        try:
            raise RuntimeError(f"invalid x-api-key: {key[:8]}****{key[-4:]}")
        except RuntimeError as inner:
            raise ValueError("model call failed") from inner
    except ValueError as caught:
        exc = caught
    scrub_call_credentials(exc, config)
    message = str(exc.__cause__)
    assert message == "invalid x-api-key: [REDACTED]"
    assert key[:8] not in message and key[-4:] not in message


def test_a_keyless_provider_sends_nothing():
    assert call_credentials({"provider": "ollama", "model": "m"}) == ()


def test_a_missing_openai_connection_sends_nothing():
    config = _resolved("openai", "gpt-x")
    assert config["_connection"]["source"] == "missing"
    assert call_credentials(config) == ()
