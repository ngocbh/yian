"""Pinned connection metadata on every keyed provider, without network."""

import pytest

from yian.compile.llm_cascade import _validate_layer
from yian.llm_clients import factory
from yian.llm_clients.config import azure_resource_endpoint
from yian.llm_clients.connection import (
    HOST_KEY_MARKER,
    legacy_connection_config,
    resolve_model_config,
    split_model_config,
    validate_connection,
)

STORE = {
    "method": "api_key",
    "source": "store",
    "credential_id": "abc",
    "endpoint": None,
    "store_path": "/abs/auth.json",
}
ENVIRONMENT = {
    "method": "api_key",
    "source": "environment",
    "credential_id": None,
    "endpoint": None,
    "store_path": None,
}


def _with(base, **changes):
    return {**base, **changes}


def test_connection_metadata_validation_per_provider():
    assert validate_connection(STORE, "anthropic") == STORE
    assert validate_connection(ENVIRONMENT, "anthropic") == ENVIRONMENT
    rejected = [
        _with(STORE, method="subscription"),
        _with(ENVIRONMENT, source="host"),
        _with(ENVIRONMENT, source="missing"),
        _with(ENVIRONMENT, source="legacy_environment"),
        _with(STORE, endpoint="https://api.anthropic.com"),
        _with(ENVIRONMENT, credential_id="abc"),
        _with(STORE, store_path="auth.json"),
        _with(STORE, credential_id=""),
        {key: value for key, value in STORE.items() if key != "endpoint"},
    ]
    for value in rejected:
        with pytest.raises(ValueError):
            validate_connection(value, "anthropic")

    azure_store = _with(STORE, endpoint="https://r.openai.azure.com/")
    assert validate_connection(azure_store, "azure") == azure_store
    assert validate_connection(ENVIRONMENT, "azure") == ENVIRONMENT
    for value in (
        STORE,
        _with(STORE, endpoint="http://r.openai.azure.com/"),
        _with(ENVIRONMENT, endpoint="https://r.openai.azure.com/"),
    ):
        with pytest.raises(ValueError):
            validate_connection(value, "azure")

    for provider in ("ollama", "unknown", None, ["openai"]):
        with pytest.raises(ValueError, match="not used for provider"):
            validate_connection(ENVIRONMENT, provider)


def test_split_and_layer_accept_non_openai_metadata():
    _, metadata = split_model_config({"provider": "anthropic", "model": "m", "_connection": STORE})
    assert metadata == STORE
    with pytest.raises(ValueError):
        split_model_config({"provider": "ollama", "model": "m", "_connection": STORE})

    _validate_layer({"provider": "anthropic", "model": "m", "_connection": STORE})
    with pytest.raises(ValueError):
        _validate_layer({"provider": "ollama", "model": "m", "_connection": STORE})

    with pytest.raises(ValueError, match="another Azure resource"):
        split_model_config({
            "provider": "azure",
            "model": "deployment",
            "base_url": "https://r.openai.azure.com/",
            "_connection": _with(STORE, endpoint="https://other.openai.azure.com/"),
        })
    _, metadata = split_model_config({
        "provider": "azure",
        "model": "deployment",
        "base_url": "https://r.openai.azure.com",
        "_connection": _with(STORE, endpoint="https://r.openai.azure.com/openai/v1/"),
    })
    assert metadata["endpoint"] == "https://r.openai.azure.com/openai/v1/"


def test_legacy_adds_environment_connection_except_host_key_and_keyless():
    assert legacy_connection_config({"provider": "anthropic", "model": "m"}) == {
        "provider": "anthropic",
        "model": "m",
        "_connection": ENVIRONMENT,
    }
    assert "_connection" not in legacy_connection_config(
        {"provider": "anthropic", "model": "m", HOST_KEY_MARKER: True}
    )
    assert "_connection" not in legacy_connection_config({"provider": "ollama", "model": "m"})
    assert (
        legacy_connection_config({"provider": "openai", "model": "m"})["_connection"]["source"]
        == "legacy_environment"
    )
    assert (
        legacy_connection_config({"provider": "anthropic", "model": "m", "_connection": STORE})[
            "_connection"
        ]
        == STORE
    )


def test_azure_resource_endpoint():
    for url in (
        "https://r.openai.azure.com/",
        "https://r.openai.azure.com/openai",
        "https://r.openai.azure.com/openai/v1/",
    ):
        assert azure_resource_endpoint(url) == "https://r.openai.azure.com"


def test_resolve_model_config_names_stray_default_model(monkeypatch):
    monkeypatch.delenv("YIAN_DEFAULT_PROVIDER", raising=False)
    monkeypatch.setenv("YIAN_DEFAULT_MODEL", "m")
    with pytest.raises(ValueError, match="YIAN_DEFAULT_MODEL is set but names no provider"):
        resolve_model_config({})


def test_non_openai_connection_adds_no_openai_route(monkeypatch):
    captured = []

    class _Client:
        def get_llm(self):
            return object()

    def fake_client(**kwargs):
        captured.append(kwargs)
        return _Client()

    monkeypatch.setattr(factory, "create_llm_client", fake_client)
    factory.model_from_config({"provider": "anthropic", "model": "m", "_connection": ENVIRONMENT})
    factory.model_from_config(
        {"provider": "anthropic", "model": "m", "api_key": "host-key", "_connection": ENVIRONMENT}
    )
    assert "api_key" not in captured[0] and "base_url" not in captured[0]
    assert captured[1]["api_key"].get_secret_value() == "host-key" and "base_url" not in captured[1]
