"""Pinned connection metadata on every keyed provider, without network."""

import json

import pytest
from pydantic import SecretStr

from yian.compose.loader import load_flow
from yian.llm_clients import connection as connection_module
from yian.llm_clients import factory
from yian.llm_clients.auth import ApiKeyCredentials, AuthError, AuthStore
from yian.llm_clients.config import HostConnection, LLMConfig, azure_resource_endpoint
from yian.llm_clients.connection import (
    HOST_KEY_MARKER,
    ConnectionSnapshot,
    connection_snapshot,
    header_name_diff,
    legacy_connection_config,
    resolve_model_config,
    split_model_config,
    validate_connection,
)
from yian.llm_clients.layers import require_concrete_selection, validate_layer
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

    validate_layer({"provider": "anthropic", "model": "m", "_connection": STORE})
    with pytest.raises(ValueError):
        validate_layer({"provider": "ollama", "model": "m", "_connection": STORE})

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


_KEY = {"api_key": "k"}
_OAUTH = {
    "access_token": "a",
    "refresh_token": "r",
    "expires_at": 9999999999,
    "account_id": "acct",
    "residency": None,
}


@pytest.mark.parametrize(
    ("provider", "method", "credentials", "endpoint"),
    [
        ("openai", "none", None, None),
        ("openai", "api_key", _KEY, "https://api.example/v1"),
        ("anthropic", "subscription", _OAUTH, None),
        ("google", "api_key", _KEY, "https://example.com/"),
        ("azure", "api_key", _KEY, None),
        ("azure", "none", None, "https://r.openai.azure.com/"),
        ("azure", "api_key", _KEY, "http://r.openai.azure.com/"),
        ("ollama", "api_key", _KEY, None),
        ("anthropic", "none", None, None),
    ],
)
def test_snapshot_rejects_impossible_saved_logins(provider, method, credentials, endpoint):
    # `save` refuses these logins, so each is written to the store file as another writer could.
    entry = {"method": method, "model": "m", "credential_id": "a" * 32, "credentials": credentials}
    if endpoint is not None:
        entry["endpoint"] = endpoint
    path = AuthStore().path
    path.parent.mkdir(mode=0o700, exist_ok=True)
    store = {"version": 1, "default_provider": None, "providers": {provider: entry}}
    path.write_text(json.dumps(store))
    path.chmod(0o600)
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


# --- host connection settings: recorded names, carried values --------------- #

HEADER_VALUE = "gw-token-3f9a20c1"
_MARKED = {"provider": "anthropic", "model": "m", HOST_KEY_MARKER: True, "api_key": "k"}


def _plain_headers(headers):
    assert all(isinstance(value, SecretStr) for value in headers.values())
    return {name: value.get_secret_value() for name, value in headers.items()}


def test_bind_records_host_settings_on_marked_agents_only(monkeypatch):
    _no_provider_keys(monkeypatch)
    host = HostConnection(
        api_key=SecretStr("sk-host-6d1e"),
        organization="org-abc",
        project="proj-abc",
        headers={"X-Trace": SecretStr("t"), "X-Gateway": SecretStr(HEADER_VALUE)},
    )
    effective = {
        "o": {"provider": "openai", "model": "m"},
        "a": {"provider": "anthropic", "model": "m"},
    }
    bound = connection_snapshot().bind(
        effective, host_key_provider="openai", host_settings=host
    )
    assert bound["o"]["organization"] == "org-abc"
    assert bound["o"]["project"] == "proj-abc"
    assert bound["o"]["_header_names"] == ["X-Gateway", "X-Trace"]
    assert bound["a"] == {"provider": "anthropic", "model": "m", "_connection": ENVIRONMENT}
    assert HEADER_VALUE not in json.dumps(bound)
    assert "sk-host-6d1e" not in json.dumps(bound)
    assert "api_key" not in bound["o"]
    assert "organization" not in effective["o"]

    plain = connection_snapshot().bind(
        effective, host_key_provider="openai", host_settings=HostConnection(api_key=SecretStr("k"))
    )
    for key in ("organization", "project", "_header_names"):
        assert key not in plain["o"]


def test_split_model_config_needs_the_recorded_headers():
    names = ["X-Gateway", "X-Trace"]
    with pytest.raises(AuthError) as caught:
        split_model_config({**_MARKED, "_header_names": names})
    message = str(caught.value)
    assert "X-Gateway" in message and "X-Trace" in message
    assert "resume_flow" in message
    with pytest.raises(AuthError) as caught:
        split_model_config({
            **_MARKED, "_header_names": names,
            "headers": {"X-Gateway": HEADER_VALUE, "X-Other": HEADER_VALUE},
        })
    message = str(caught.value)
    assert "X-Trace" in message and "X-Other" in message
    assert HEADER_VALUE not in message
    with pytest.raises(AuthError, match="X-Gateway"):
        split_model_config({**_MARKED, "headers": {"X-Gateway": HEADER_VALUE}})

    # Names compare without case.
    validated, _ = split_model_config({
        **_MARKED, "_header_names": names,
        "headers": {"x-gateway": HEADER_VALUE, "X-TRACE": "t"},
    })
    assert _plain_headers(validated.headers) == {"x-gateway": HEADER_VALUE, "X-TRACE": "t"}


@pytest.mark.parametrize(
    "names",
    [[], ["X-Trace", "X-Gateway"], ["X-Gateway", "x-gateway"], ["bad name"],
     ["Authorization"], "X-Gateway", [1]],
    ids=["empty", "unsorted", "case-duplicate", "not-a-token", "refused", "not-a-list",
         "not-text"],
)
def test_split_model_config_rejects_malformed_header_names(names):
    with pytest.raises(ValueError, match="_header_names"):
        split_model_config({**_MARKED, "_header_names": names, "headers": {"X-Gateway": "v"}})


def test_split_model_config_refuses_header_names_without_the_marker():
    unmarked = {key: value for key, value in _MARKED.items() if key != HOST_KEY_MARKER}
    with pytest.raises(ValueError, match="_header_names"):
        split_model_config(
            {**unmarked, "_header_names": ["X-Gateway"], "headers": {"X-Gateway": "v"}}
        )


def test_split_and_resolve_carry_headers_beside_the_key():
    config = LLMConfig(
        provider="anthropic", model="m", api_key="k", headers={"X-Gateway": HEADER_VALUE}
    )
    validated, _ = split_model_config(config)
    assert _plain_headers(validated.headers) == {"X-Gateway": HEADER_VALUE}
    resolved = resolve_model_config(config)
    assert _plain_headers(resolved["headers"]) == {"X-Gateway": HEADER_VALUE}
    assert resolved["api_key"].get_secret_value() == "k"
    resolved = resolve_model_config(
        {**_MARKED, "_header_names": ["X-Gateway"], "headers": {"X-Gateway": HEADER_VALUE}}
    )
    assert _plain_headers(resolved["headers"]) == {"X-Gateway": HEADER_VALUE}
    assert "_header_names" not in resolved
    # The OpenAI route resolves through the snapshot and still keeps them.
    resolved = resolve_model_config({
        "provider": "openai", "model": "m", "api_key": "k", "organization": "org-abc",
        "headers": {"X-Gateway": HEADER_VALUE},
    })
    assert resolved["organization"] == "org-abc"
    assert _plain_headers(resolved["headers"]) == {"X-Gateway": HEADER_VALUE}


def test_layers_refuse_recorded_header_names():
    with pytest.raises(ValueError, match="_header_names is internal to recorded runs"):
        validate_layer({"provider": "anthropic", "_header_names": ["X-Gateway"]})


def test_a_recorded_map_takes_settings_only_beside_the_marker():
    marked = {
        "provider": "openai", "model": "m", HOST_KEY_MARKER: True,
        "organization": "org-abc", "project": "proj-abc", "_header_names": ["X-Gateway"],
    }
    require_concrete_selection(marked, path="agent")
    for key in ("organization", "project", "_header_names"):
        unmarked = {"provider": "openai", "model": "m", key: marked[key]}
        with pytest.raises(ValueError, match=f"'agent' has {key} without a host key"):
            require_concrete_selection(unmarked, path="agent")
    for names in ([], ["X-B", "X-A"], ["Authorization"], "X-A"):
        with pytest.raises(ValueError, match="'agent'.*_header_names"):
            require_concrete_selection({**marked, "_header_names": names}, path="agent")
    with pytest.raises(ValueError, match="'agent'.*organization"):
        require_concrete_selection({**marked, "organization": "has space"}, path="agent")
    for key in ("organization", "project"):
        anthropic = {**marked, "provider": "anthropic", key: marked[key]}
        anthropic.pop("organization" if key == "project" else "project")
        with pytest.raises(
            ValueError, match=f"'agent': {key} applies only to provider openai"
        ):
            require_concrete_selection(anthropic, path="agent")
    with pytest.raises(ValueError, match="headers"):
        require_concrete_selection({**marked, "headers": {"X-Gateway": "v"}}, path="agent")


# --- saved-login connection settings: recorded in _connection, sent from the store --- #

SAVED_HEADER = "saved-gw-7c2e91"
_OPENAI_STORE = {**STORE, "endpoint": "https://api.openai.com/v1"}


def _save_openai_with_settings(**options):
    return AuthStore().save(
        "openai", ApiKeyCredentials("sk-saved-4b1f", {"X-Gateway": SAVED_HEADER}), "gpt-x",
        organization="org-saved", project="proj-saved", **options,
    )


def test_validate_connection_takes_saved_login_settings():
    openai = {
        **_OPENAI_STORE, "organization": "org-a", "project": "proj-a",
        "header_names": ["X-Gateway", "X-Trace"],
    }
    assert validate_connection(openai) == openai
    anthropic = {**STORE, "header_names": ["X-Gateway"]}
    assert validate_connection(anthropic, "anthropic") == anthropic
    azure = {**STORE, "endpoint": "https://r.openai.azure.com/", "header_names": ["X-Gateway"]}
    assert validate_connection(azure, "azure") == azure
    # Old records carry none of them.
    assert validate_connection(_OPENAI_STORE) == _OPENAI_STORE


@pytest.mark.parametrize(
    ("provider", "value", "message"),
    [
        ("openai", {**_OPENAI_STORE, "source": "environment", "credential_id": "a" * 64,
                    "store_path": None, "header_names": ["X-Gateway"]}, "saved API-key login"),
        ("openai", {**_OPENAI_STORE, "source": "host", "credential_id": None,
                    "store_path": None, "organization": "org-a"}, "saved API-key login"),
        ("openai", {**_OPENAI_STORE, "method": "subscription",
                    "endpoint": "https://chatgpt.com/backend-api/codex",
                    "header_names": ["X-Gateway"]}, "saved API-key login"),
        ("anthropic", {**ENVIRONMENT, "header_names": ["X-Gateway"]}, "saved API-key login"),
        ("anthropic", {**STORE, "organization": "org-a"},
         "organization is only for provider openai"),
        ("google", {**STORE, "project": "proj-a"}, "project is only for provider openai"),
        ("openai", {**_OPENAI_STORE, "organization": "has space"}, "connection organization"),
        ("openai", {**_OPENAI_STORE, "project": None}, "connection project"),
        ("openai", {**_OPENAI_STORE, "header_names": []}, "connection header_names"),
        ("openai", {**_OPENAI_STORE, "header_names": ["X-B", "X-A"]}, "connection header_names"),
        ("openai", {**_OPENAI_STORE, "header_names": ["OpenAI-Project"]},
         "connection header_names"),
        ("anthropic", {**STORE, "header_names": ["Authorization"]}, "connection header_names"),
        ("anthropic", {**STORE, "header_names": "X-Gateway"}, "connection header_names"),
        ("anthropic", {**STORE, "headers": {"X-Gateway": "v"}}, "metadata fields"),
    ],
)
def test_validate_connection_refuses_misplaced_settings(provider, value, message):
    with pytest.raises(ValueError, match=message) as error:
        validate_connection(value, provider)
    assert "has space" not in str(error.value)


def test_snapshot_records_saved_settings_without_values(monkeypatch):
    _no_provider_keys(monkeypatch)
    openai = _save_openai_with_settings()
    anthropic = AuthStore().save(
        "anthropic", ApiKeyCredentials("sk-ant-saved", {"X-Trace": "t", "X-Gateway": "g"}), "m",
        make_default=False,
    )
    plain = AuthStore().save("google", ApiKeyCredentials("g-saved"), "g", make_default=False)

    connections = connection_snapshot().connections

    assert connections["openai"] == {
        **_store_connection(openai, "https://api.openai.com/v1"),
        "organization": "org-saved", "project": "proj-saved", "header_names": ["X-Gateway"],
    }
    assert connections["anthropic"] == {
        **_store_connection(anthropic), "header_names": ["X-Gateway", "X-Trace"],
    }
    assert connections["google"] == _store_connection(plain)
    assert SAVED_HEADER not in json.dumps(connections)


def _plain_connection_headers(kwargs):
    return _plain_headers(kwargs["connection_headers"])


def test_saved_openai_settings_reach_the_client(monkeypatch):
    _no_provider_keys(monkeypatch)
    _save_openai_with_settings()
    captured, _ = _capture_clients(monkeypatch)

    factory.model_from_config({"provider": "openai", "model": "gpt-x"})

    kwargs = captured[0][2]
    assert _plain_connection_headers(kwargs) == {
        "OpenAI-Organization": "org-saved", "OpenAI-Project": "proj-saved",
        "X-Gateway": SAVED_HEADER,
    }
    # The key is still resolved per request.
    assert kwargs["api_key"]() == "sk-saved-4b1f"


def test_saved_openai_ids_without_headers_read_no_credentials_at_build(monkeypatch):
    _no_provider_keys(monkeypatch)
    AuthStore().save("openai", ApiKeyCredentials("sk-saved"), "gpt-x", organization="org-saved")
    captured, _ = _capture_clients(monkeypatch)

    def no_store(*args, **kwargs):
        pytest.fail("read the saved login at build")

    monkeypatch.setattr(AuthStore, "credentials", no_store)
    factory.model_from_config({"provider": "openai", "model": "gpt-x"})

    assert _plain_connection_headers(captured[0][2]) == {"OpenAI-Organization": "org-saved"}


def test_saved_anthropic_headers_reach_the_client(monkeypatch):
    _no_provider_keys(monkeypatch)
    AuthStore().save("anthropic", ApiKeyCredentials("sk-ant", {"X-Gateway": SAVED_HEADER}), "m")
    captured, _ = _capture_clients(monkeypatch)

    factory.model_from_config({"provider": "anthropic", "model": "m"})

    kwargs = captured[0][2]
    assert _plain(kwargs["api_key"]) == "sk-ant"
    assert _plain_connection_headers(kwargs) == {"X-Gateway": SAVED_HEADER}


def test_environment_and_host_keys_never_get_saved_settings(monkeypatch):
    _no_provider_keys(monkeypatch)
    _save_openai_with_settings(make_default=False)
    AuthStore().save(
        "azure", ApiKeyCredentials("az-saved", {"X-Gateway": SAVED_HEADER}), "dep",
        endpoint="https://r.openai.azure.com/", make_default=False,
    )
    monkeypatch.setenv("GOOGLE_API_KEY", "env-key")
    captured, _ = _capture_clients(monkeypatch)

    factory.model_from_config({"provider": "google", "model": "g"})
    # An Azure base_url on another resource reads the environment key.
    factory.model_from_config(
        {"provider": "azure", "model": "dep", "base_url": "https://other.openai.azure.com/"}
    )
    factory.model_from_config({"provider": "openai", "model": "gpt-x", "api_key": "sk-host"})
    factory.model_from_config({
        "provider": "azure", "model": "dep", "api_key": "az-host",
        "base_url": "https://r.openai.azure.com/",
    })

    for _, _, kwargs in captured:
        assert "connection_headers" not in kwargs


def test_resaving_the_login_fails_the_build(monkeypatch):
    _no_provider_keys(monkeypatch)
    _save_openai_with_settings()
    recorded = resolve_model_config({"provider": "openai", "model": "gpt-x"})
    _save_openai_with_settings()
    _capture_clients(monkeypatch)

    with pytest.raises(AuthError, match="yian auth openai"):
        factory.model_from_config(recorded)


@pytest.mark.parametrize("names", [["X-Other"], ["X-Gateway", "X-Trace"], None])
def test_saved_headers_must_match_the_recorded_names(monkeypatch, names):
    _no_provider_keys(monkeypatch)
    AuthStore().save("anthropic", ApiKeyCredentials("sk-ant", {"X-Gateway": SAVED_HEADER}), "m")
    recorded = resolve_model_config({"provider": "anthropic", "model": "m"})
    if names is None:
        del recorded["_connection"]["header_names"]
    else:
        recorded["_connection"]["header_names"] = names
    captured, _ = _capture_clients(monkeypatch)

    with pytest.raises(AuthError, match="yian auth anthropic") as error:
        factory.model_from_config(recorded)
    assert SAVED_HEADER not in str(error.value)
    assert captured == []


def test_header_name_diff_names_missing_and_unrecorded_headers_ignoring_case():
    assert header_name_diff(["X-Trace", "X-Gateway"], ["x-gateway", "X-TRACE"]) is None
    assert header_name_diff([], []) is None
    assert header_name_diff(["X-Trace", "X-Gateway"], ["x-gateway", "X-Extra", "A-New"]) == (
        "missing X-Trace; not recorded A-New, X-Extra"
    )
    assert header_name_diff(["X-Trace"], []) == "missing X-Trace"
