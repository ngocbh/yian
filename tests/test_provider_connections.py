"""Pinned connection metadata on every keyed provider, without network."""

import json
import time

import pytest

from yian.compile.llm_cascade import _validate_layer
from yian.compose.loader import load_flow
from yian.llm_clients import connection as connection_module
from yian.llm_clients import factory
from yian.llm_clients.auth import ApiKeyCredentials, AuthStore, OAuthCredentials
from yian.llm_clients.config import azure_resource_endpoint
from yian.llm_clients.connection import (
    HOST_KEY_MARKER,
    ConnectionSnapshot,
    connection_snapshot,
    legacy_connection_config,
    resolve_model_config,
    split_model_config,
    validate_connection,
)
from yian.run import run_flow

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

    with pytest.raises(ValueError, match="not used for provider ollama"):
        validate_connection(ENVIRONMENT, "ollama")
    for provider in ("unknown-secret", None, ["openai"]):
        with pytest.raises(ValueError, match="not used for an unsupported provider$"):
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


def _no_provider_keys(monkeypatch):
    for name in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(name, raising=False)


def _store_connection(selection, endpoint=None):
    return {
        "method": "api_key",
        "source": "store",
        "credential_id": selection.credential_id,
        "endpoint": endpoint,
        "store_path": str(AuthStore().path),
    }


def test_snapshot_connections_per_provider(monkeypatch):
    _no_provider_keys(monkeypatch)
    saved = AuthStore().save("anthropic", ApiKeyCredentials("saved-secret"), "m")

    connections = connection_snapshot().connections

    assert connections["anthropic"] == _store_connection(saved)
    assert connections["google"] == ENVIRONMENT
    assert "ollama" not in connections
    assert connections["openai"] == {
        "method": "api_key",
        "source": "missing",
        "credential_id": None,
        "endpoint": "https://api.openai.com/v1",
        "store_path": None,
    }


@pytest.mark.parametrize(
    ("provider", "credentials", "endpoint"),
    [
        ("openai", None, None),
        ("openai", ApiKeyCredentials("k"), "https://api.example/v1"),
        ("anthropic", OAuthCredentials("a", "r", time.time() + 3600, "acct"), None),
        ("google", ApiKeyCredentials("k"), "https://example.com/"),
        ("azure", ApiKeyCredentials("k"), None),
        ("azure", None, "https://r.openai.azure.com/"),
        ("azure", ApiKeyCredentials("k"), "http://r.openai.azure.com/"),
        ("ollama", ApiKeyCredentials("k"), None),
        ("anthropic", None, None),
    ],
)
def test_snapshot_rejects_impossible_saved_logins(provider, credentials, endpoint):
    AuthStore().save(provider, credentials, "m", endpoint=endpoint, make_default=False)
    with pytest.raises(ValueError, match=f"The saved {provider} login is invalid; run 'yian auth {provider}' again"):
        connection_snapshot()


def test_snapshot_accepts_keyless_logins_and_skips_unsupported_ones(monkeypatch):
    store = AuthStore()
    store.save("ollama", None, "l", make_default=False)
    store.save("vllm", None, "v", make_default=False)
    store.save("claude", ApiKeyCredentials("k"), "c", make_default=False)

    snapshot = connection_snapshot()

    assert snapshot.provider_models["ollama"] == "l"
    assert snapshot.connections["vllm"] == ENVIRONMENT
    assert "claude" not in snapshot.provider_models
    assert "claude" not in snapshot.connections

    monkeypatch.delenv("YIAN_DEFAULT_PROVIDER")
    store.set_default("claude")
    with pytest.raises(ValueError, match="saved default provider 'claude' is not supported"):
        connection_snapshot()


def test_bind_records_store_and_environment_connections(monkeypatch):
    _no_provider_keys(monkeypatch)
    saved = AuthStore().save("anthropic", ApiKeyCredentials("saved-secret"), "m")
    snapshot = connection_snapshot()
    effective = {
        "a": {"provider": "anthropic", "model": "m"},
        "g": {"provider": "google", "model": "g"},
        "o": {"provider": "ollama", "model": "l"},
    }

    bound = snapshot.bind(effective)

    assert bound["a"]["_connection"] == _store_connection(saved)
    assert bound["g"]["_connection"] == ENVIRONMENT
    assert "_connection" not in bound["o"]
    assert "_connection" not in effective["a"]

    hosted = snapshot.bind(effective, host_key_provider="anthropic")
    assert hosted["a"][HOST_KEY_MARKER] is True
    assert "_connection" not in hosted["a"]
    assert hosted["g"]["_connection"] == ENVIRONMENT


def test_bind_azure_endpoint_rules(monkeypatch):
    effective = {"z": {"provider": "azure", "model": "dep"}}

    bound = connection_snapshot().bind(effective)
    assert "base_url" not in bound["z"]
    assert bound["z"]["_connection"] == ENVIRONMENT

    saved = AuthStore().save(
        "azure", ApiKeyCredentials("saved-secret"), "dep", endpoint="https://r.openai.azure.com/"
    )
    snapshot = connection_snapshot()
    store = _store_connection(saved, "https://r.openai.azure.com/")

    bound = snapshot.bind(effective)["z"]
    assert bound["base_url"] == "https://r.openai.azure.com/"
    assert bound["_connection"] == store

    same = {"provider": "azure", "model": "dep", "base_url": "https://r.openai.azure.com/openai/v1/"}
    bound = snapshot.bind({"z": same})["z"]
    assert bound["base_url"] == "https://r.openai.azure.com/openai/v1/"
    assert bound["_connection"] == store

    other = {"provider": "azure", "model": "dep", "base_url": "https://other.openai.azure.com/"}
    bound = snapshot.bind({"z": other})["z"]
    assert bound["base_url"] == "https://other.openai.azure.com/"
    assert bound["_connection"] == ENVIRONMENT


def test_resolve_model_config_attaches_without_rereading_on_replay(monkeypatch):
    _no_provider_keys(monkeypatch)
    real_snapshot = connection_snapshot

    def no_snapshot():
        pytest.fail("read the connection snapshot")

    monkeypatch.setattr(connection_module, "connection_snapshot", no_snapshot)
    assert resolve_model_config({"provider": "ollama", "model": "l"}) == {
        "provider": "ollama",
        "model": "l",
    }
    hosted = resolve_model_config({"provider": "anthropic", "model": "m", "api_key": "k"})
    assert "_connection" not in hosted
    pinned = {"provider": "anthropic", "model": "m", "_connection": STORE}
    assert resolve_model_config(pinned) == pinned
    host = ConnectionSnapshot({}, {}, {}).host_connection()
    recorded = {"provider": "openai", "model": "m", "api_key": "k", "_connection": host}
    assert resolve_model_config(recorded)["_connection"] == host
    monkeypatch.setattr(connection_module, "connection_snapshot", real_snapshot)

    assert resolve_model_config({"provider": "anthropic", "model": "m"})["_connection"] == ENVIRONMENT
    saved = AuthStore().save("anthropic", ApiKeyCredentials("saved-secret"), "m")
    assert resolve_model_config({"provider": "anthropic", "model": "m"})[
        "_connection"
    ] == _store_connection(saved)
    assert (
        resolve_model_config({"provider": "openai", "model": "m", "api_key": "k"})["_connection"]
        == host
    )
    azure = AuthStore().save(
        "azure", ApiKeyCredentials("saved-secret"), "dep", endpoint="https://r.openai.azure.com/"
    )
    resolved = resolve_model_config({"provider": "azure", "model": "dep"})
    assert resolved["base_url"] == "https://r.openai.azure.com/"
    assert resolved["_connection"] == _store_connection(azure, "https://r.openai.azure.com/")


def test_run_records_connections(monkeypatch, tmp_path):
    _no_provider_keys(monkeypatch)
    saved = AuthStore().save("anthropic", ApiKeyCredentials("saved-secret"), "m")

    class _Model:
        def invoke(self, messages):
            class _Response:
                content = "ok"

            return _Response()

    monkeypatch.setattr("yian.llm_clients.model_from_config", lambda config: _Model())
    flow = load_flow(
        "id: f\nname: f\nnodes:\n"
        "  a: {kind: agent, prompt: hi, llm_config: {provider: anthropic, model: m}}\n"
        "  g: {kind: agent, prompt: hi, llm_config: {provider: google, model: g}}\n"
        "output: ${a.output}\n"
    )
    log = tmp_path / "r.jsonl"

    assert run_flow(flow, {}, log=log).status == "succeeded"

    started = json.loads(log.read_text().splitlines()[0])
    assert started["kind"] == "run_started"
    recorded = started["payload"]["effective_llm_by_static_path"]
    assert recorded["a"]["_connection"] == _store_connection(saved)
    assert recorded["g"]["_connection"] == ENVIRONMENT
    assert b"saved-secret" not in log.read_bytes()


def _capture_clients(monkeypatch):
    """Record each client's provider, model, and kwargs instead of building it."""
    captured = []
    model = object()

    class _Client:
        def get_llm(self):
            return model

    def fake_client(provider, model, **kwargs):
        captured.append((provider, model, kwargs))
        return _Client()

    monkeypatch.setattr(factory, "create_llm_client", fake_client)
    return captured, model


def _plain(key):
    return key.get_secret_value() if hasattr(key, "get_secret_value") else key


def test_saved_key_reaches_the_client(monkeypatch):
    AuthStore().save("anthropic", ApiKeyCredentials("saved-anth-key"), "claude-x", make_default=False)
    captured, model = _capture_clients(monkeypatch)

    assert factory.model_from_config({"provider": "anthropic", "model": "claude-x"}) is model
    monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")
    factory.model_from_config({"provider": "anthropic", "model": "claude-x"})

    for provider, name, kwargs in captured:
        assert (provider, name) == ("anthropic", "claude-x")
        assert _plain(kwargs["api_key"]) == "saved-anth-key"
        assert "base_url" not in kwargs


def test_env_key_used_without_saved_login(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")
    captured, _ = _capture_clients(monkeypatch)

    factory.model_from_config({"provider": "anthropic", "model": "claude-x"})

    assert "api_key" not in captured[0][2]


def test_host_key_beats_saved_login_and_sets_no_endpoint(monkeypatch):
    AuthStore().save("anthropic", ApiKeyCredentials("saved-anth-key"), "m", make_default=False)
    captured, _ = _capture_clients(monkeypatch)

    factory.model_from_config({"provider": "anthropic", "model": "m", "api_key": "host"})
    recorded = legacy_connection_config(
        {"provider": "anthropic", "model": "m", HOST_KEY_MARKER: True}
    )
    factory.model_from_config({**recorded, "api_key": "host"})

    for _, _, kwargs in captured:
        assert _plain(kwargs["api_key"]) == "host"
        assert "base_url" not in kwargs


def test_azure_saved_login_reaches_client(monkeypatch):
    AuthStore().save(
        "azure", ApiKeyCredentials("az-key"), "dep",
        endpoint="https://r.openai.azure.com/", make_default=False,
    )
    captured, _ = _capture_clients(monkeypatch)

    factory.model_from_config({"provider": "azure"})
    factory.model_from_config(
        {"provider": "azure", "model": "dep", "base_url": "https://other.openai.azure.com/"}
    )

    provider, model, kwargs = captured[0]
    assert (provider, model) == ("azure", "dep")
    assert _plain(kwargs["api_key"]) == "az-key"
    assert kwargs["base_url"] == "https://r.openai.azure.com/"
    assert "api_key" not in captured[1][2]
    assert captured[1][2]["base_url"] == "https://other.openai.azure.com/"


def test_connection_credentials_per_provider():
    from yian.llm_clients.auth import AuthError
    from yian.llm_clients.connection import connection_credentials

    saved = AuthStore().save("anthropic", ApiKeyCredentials("saved-anth-key"), "m", make_default=False)
    pinned = _store_connection(saved)

    assert connection_credentials(pinned, "anthropic").api_key == "saved-anth-key"

    AuthStore().save("anthropic", ApiKeyCredentials("other-key"), "m", make_default=False)
    with pytest.raises(AuthError, match="yian auth anthropic"):
        connection_credentials(pinned, "anthropic")
    AuthStore().remove("anthropic")
    with pytest.raises(AuthError, match="yian auth anthropic"):
        connection_credentials(pinned, "anthropic")
    with pytest.raises(ValueError, match="read by its client"):
        connection_credentials(ENVIRONMENT, "anthropic")
