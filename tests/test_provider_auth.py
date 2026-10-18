"""Provider login and credential persistence use only simulated external effects."""

import base64
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import multiprocessing
import os
import socket
import ssl
import stat
import threading
import time
from urllib.parse import parse_qs, urlencode, urlparse

import pytest

from yian.llm_clients.auth import (
    ApiKeyCredentials,
    AuthError,
    AuthStore,
    HttpResponse,
    OAuthCredentials,
    OpenAIAuth,
    ProviderSelection,
    is_auth_failure,
)


def _jwt(account="account-test", **claims):
    value = {"https://api.openai.com/auth": {"chatgpt_account_id": account}}
    value.update(claims)
    payload = base64.urlsafe_b64encode(json.dumps(value).encode()).decode().rstrip("=")
    return f"header.{payload}.signature"


def _tokens(account="account-test", **overrides):
    return {
        "access_token": _jwt(account),
        "refresh_token": "refresh-secret",
        "expires_in": 3600,
        **overrides,
    }


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class HTTP:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        return self.responses.pop(0)


def test_store_read_does_not_create_files_and_save_commits_selection(tmp_path):
    path = tmp_path / "private" / "auth.json"
    store = AuthStore(path)
    assert store.read().selections == ()
    assert not path.parent.exists()
    selection = store.save("openai", ApiKeyCredentials("secret-key"), "gpt-test")
    assert selection.provider == "openai"
    assert selection.method == "api_key"
    assert selection.model == "gpt-test"
    assert store.read().default_provider == "openai"
    assert store.selection() == selection
    assert store.credentials("openai", selection.credential_id).api_key == "secret-key"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(path.with_suffix(".json.lock").stat().st_mode) == 0o600
    assert "secret-key" not in repr(store.read())


def test_reconnection_replaces_method_and_invalidates_previous_identity(tmp_path):
    store = AuthStore(tmp_path / "auth" / "auth.json")
    first = store.save("openai", ApiKeyCredentials("old-secret"), "old-model")
    second = store.save(
        "openai", OAuthCredentials("access", "refresh", 9000, "acct"), "new-model"
    )
    assert first.credential_id != second.credential_id
    assert store.selection() == second
    assert "old-secret" not in store.path.read_text()
    with pytest.raises(AuthError, match="changed"):
        store.credentials("openai", first.credential_id)


def test_refresh_is_serialized_and_preserves_identity(tmp_path):
    store = AuthStore(tmp_path / "auth" / "auth.json", clock=lambda: 1000)
    selection = store.save(
        "openai", OAuthCredentials("old", "old-refresh", 900, "acct"), "model"
    )
    refreshed = []

    def refresh(credentials):
        refreshed.append(credentials)
        return OAuthCredentials("new", "new-refresh", 5000, "acct")

    def access(_):
        return store.credentials("openai", selection.credential_id, refresh=refresh)

    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(access, range(12)))
    assert len(refreshed) == 1
    assert all(result.access_token == "new" for result in results)
    assert store.selection() == selection


def _saved_kinds(store):
    store.save("anthropic", ApiKeyCredentials("k-anth"), "claude-x")
    store.save("ollama", None, "llama3")
    store.save(
        "azure",
        ApiKeyCredentials("k-az"),
        "dep",
        endpoint="https://r.openai.azure.com/",
    )
    store.save("openai", OAuthCredentials("access", "refresh", 9000, "acct"), "gpt")
    return {item.provider: item for item in store.read().selections}


def _write_store(path, value):
    path.parent.mkdir(mode=0o700, exist_ok=True)
    path.write_text(json.dumps(value))
    path.chmod(0o600)


def test_store_saves_each_credential_kind(tmp_path):
    store = AuthStore(tmp_path / "auth" / "auth.json")
    saved = _saved_kinds(store)
    assert len(store.read().selections) == 4
    assert {name: item.method for name, item in saved.items()} == {
        "anthropic": "api_key",
        "ollama": "none",
        "azure": "api_key",
        "openai": "subscription",
    }
    assert {name: item.endpoint for name, item in saved.items()} == {
        "anthropic": None,
        "ollama": None,
        "azure": "https://r.openai.azure.com/",
        "openai": None,
    }
    assert saved["ollama"].model == "llama3"
    assert stat.S_IMODE(store.path.stat().st_mode) == 0o600
    assert "k-anth" not in repr(store.read())


def test_credentials_for_saved_kinds(tmp_path):
    store = AuthStore(tmp_path / "auth" / "auth.json")
    saved = _saved_kinds(store)
    anthropic = store.credentials("anthropic", saved["anthropic"].credential_id)
    assert anthropic.api_key == "k-anth"
    assert store.credentials("azure", saved["azure"].credential_id).api_key == "k-az"
    with pytest.raises(AuthError, match="has no saved credential"):
        store.credentials("ollama", saved["ollama"].credential_id)


def test_saved_secrets_returns_each_kind_without_refresh(tmp_path, monkeypatch):
    def no_refresh(*args, **kwargs):
        raise AssertionError("saved_secrets must not refresh")

    monkeypatch.setattr(OpenAIAuth, "refresh", no_refresh)
    store = AuthStore(tmp_path / "auth" / "auth.json", clock=lambda: 10_000.0)
    store.save("anthropic", ApiKeyCredentials("k-anth"), "claude-x")
    store.save("openai", OAuthCredentials("access", "refresh", 900, "acct"), "gpt")
    store.save("ollama", None, "llama3")
    before = (store.path.stat().st_mtime_ns, store.path.read_bytes())
    assert store.saved_secrets("anthropic") == ("k-anth",)
    assert store.saved_secrets("openai") == ("access", "refresh")
    assert store.saved_secrets("ollama") == ()
    assert store.saved_secrets("google") == ()
    assert (store.path.stat().st_mtime_ns, store.path.read_bytes()) == before

    missing = AuthStore(tmp_path / "missing" / "auth.json")
    assert missing.saved_secrets("anthropic") == ()
    assert not missing.path.parent.exists()


def test_remove_provider_and_default(tmp_path):
    store = AuthStore(tmp_path / "auth" / "auth.json")
    store.save("openai", ApiKeyCredentials("k-open"), "gpt")
    store.save("anthropic", ApiKeyCredentials("k-anth"), "claude-x")
    assert store.read().default_provider == "anthropic"
    store.remove("anthropic")
    state = store.read()
    assert state.default_provider is None
    assert [item.provider for item in state.selections] == ["openai"]
    assert "k-anth" not in store.path.read_text()
    with pytest.raises(AuthError, match="no saved anthropic login"):
        store.remove("anthropic")

    store.save("ollama", None, "llama3", make_default=False)
    store.save(
        "azure",
        ApiKeyCredentials("k-az"),
        "dep",
        endpoint="https://r.openai.azure.com/",
    )
    store.remove("ollama")
    store.remove("azure")
    state = store.read()
    assert state.default_provider is None
    assert [item.provider for item in state.selections] == ["openai"]

    missing = AuthStore(tmp_path / "missing" / "auth.json")
    with pytest.raises(AuthError, match="no saved openai login"):
        missing.remove("openai")
    assert not missing.path.parent.exists()

    legacy = tmp_path / "legacy" / "auth.json"
    entry = {
        "method": "api_key",
        "model": "m",
        "credential_id": "c" * 32,
        "credentials": {"api_key": "k-legacy"},
    }
    _write_store(
        legacy,
        {
            "version": 1,
            "default_provider": "legacy-x",
            "providers": {"legacy-x": entry, "openai": entry},
        },
    )
    AuthStore(legacy).remove("legacy-x")
    state = AuthStore(legacy).read()
    assert state.default_provider is None
    assert [item.provider for item in state.selections] == ["openai"]


def test_set_default_requires_saved_provider(tmp_path):
    store = AuthStore(tmp_path / "auth" / "auth.json")
    first = store.save("openai", ApiKeyCredentials("k-open"), "gpt")
    second = store.save("anthropic", ApiKeyCredentials("k-anth"), "claude-x")
    store.set_default("openai")
    assert store.read().default_provider == "openai"
    with pytest.raises(AuthError, match="no saved google login"):
        store.set_default("google")
    assert store.read().default_provider == "openai"
    assert store.selection("openai") == first
    assert store.selection("anthropic") == second

    store.save("ollama", None, "llama3", make_default=False)
    store.save(
        "azure",
        ApiKeyCredentials("k-az"),
        "dep",
        endpoint="https://r.openai.azure.com/",
        make_default=False,
    )
    store.set_default("ollama")
    assert store.selection().provider == "ollama"
    store.set_default("azure")
    assert store.selection().endpoint == "https://r.openai.azure.com/"

    missing = AuthStore(tmp_path / "missing" / "auth.json")
    with pytest.raises(AuthError, match="no saved openai login"):
        missing.set_default("openai")
    assert not missing.path.parent.exists()


def test_v1_store_without_new_fields_still_loads(tmp_path):
    path = tmp_path / "auth" / "auth.json"
    path.parent.mkdir(mode=0o700)
    path.write_text(
        '{"default_provider": "openai", "providers": {"openai": {"credential_id": '
        '"0123456789abcdef0123456789abcdef", "credentials": {"api_key": "sk-old"}, '
        '"method": "api_key", "model": "gpt-4o"}}, "version": 1}\n'
    )
    path.chmod(0o600)
    store = AuthStore(path)
    selection = store.selection()
    assert selection.provider == "openai"
    assert selection.method == "api_key"
    assert selection.endpoint is None
    assert store.credentials("openai", selection.credential_id).api_key == "sk-old"


def _entry_store(provider="p", **entry):
    value = {
        "method": "api_key",
        "model": "m",
        "credential_id": "c" * 32,
        "credentials": {"api_key": "do-not-echo"},
    }
    value.update(entry)
    return json.dumps(
        {"version": 1, "default_provider": None, "providers": {provider: value}}
    )


@pytest.mark.parametrize(
    "contents",
    [
        "{",
        "[]",
        '{"version":99}',
        '{"secret":"do-not-echo"}',
        _entry_store(method="none"),
        _entry_store(credentials=None),
        _entry_store(endpoint=["do-not-echo"]),
        _entry_store(secret="do-not-echo"),
        _entry_store(header_names=["X-Gateway"]),
        _entry_store("openai", organization=["do-not-echo"]),
        _entry_store("openai", organization=""),
        _entry_store("openai", project="do not echo"),
        _entry_store(organization="do-not-echo"),
        _entry_store("openai", organization=None),
        _entry_store("openai", project=None),
        _entry_store("ollama", credentials={"api_key": "k", "headers": {"X": "do-not-echo"}}),
        _entry_store(
            "openai",
            method="subscription",
            project="do-not-echo",
            credentials={
                "access_token": "a",
                "refresh_token": "r",
                "expires_at": 9000,
                "account_id": "acct",
                "residency": None,
            },
        ),
        _entry_store(credentials={"api_key": "k", "headers": [["X", "do-not-echo"]]}),
        _entry_store(credentials={"api_key": "k", "headers": {}}),
        _entry_store(credentials={"api_key": "k", "headers": {"X": 1}}),
        _entry_store(credentials={"api_key": "k", "headers": {"Authorization": "do-not-echo"}}),
        _entry_store(credentials={"api_key": "k", "headers": {"X": "do-not-echo\n"}}),
        _entry_store(credentials={"api_key": "k", "headers": {"X": "do-not-echo", "x": "v"}}),
        _entry_store(credentials={"api_key": "k", "secret": "do-not-echo"}),
        _entry_store(
            "openai", credentials={"api_key": "k", "headers": {"OpenAI-Project": "do-not-echo"}}
        ),
    ],
)
def test_malformed_store_fails_without_echoing_payload(tmp_path, contents):
    path = tmp_path / "auth" / "auth.json"
    path.parent.mkdir(mode=0o700)
    path.write_text(contents)
    path.chmod(0o600)
    with pytest.raises(AuthError) as caught:
        AuthStore(path).read()
    assert "do-not-echo" not in str(caught.value)


def test_store_reads_a_null_endpoint(tmp_path):
    path = tmp_path / "auth" / "auth.json"
    path.parent.mkdir(mode=0o700)
    path.write_text(_entry_store(endpoint=None))
    path.chmod(0o600)
    assert AuthStore(path).selection("p").endpoint is None


def _gateway_credentials(key="sk-open"):
    return ApiKeyCredentials(key, headers={"X-Gateway": "gw-secret", "A-Tag": "tag-secret"})


def test_store_saves_connection_settings_with_a_login(tmp_path):
    store = AuthStore(tmp_path / "auth" / "auth.json")
    selection = store.save(
        "openai", _gateway_credentials(), "gpt", organization="org-1", project="proj-1"
    )
    assert (selection.organization, selection.project) == ("org-1", "proj-1")
    assert selection.header_names == ("A-Tag", "X-Gateway")
    assert store.selection("openai") == selection
    credentials = store.credentials("openai", selection.credential_id)
    assert credentials == _gateway_credentials()
    assert credentials.headers == (("A-Tag", "tag-secret"), ("X-Gateway", "gw-secret"))
    entry = json.loads(store.path.read_text())["providers"]["openai"]
    assert (entry["organization"], entry["project"]) == ("org-1", "proj-1")
    assert entry["credentials"] == {
        "api_key": "sk-open",
        "headers": {"A-Tag": "tag-secret", "X-Gateway": "gw-secret"},
    }

    anthropic = store.save("anthropic", ApiKeyCredentials("k", headers={"X-G": "v"}), "c")
    assert anthropic.header_names == ("X-G",)
    assert (anthropic.organization, anthropic.project) == (None, None)
    assert store.selection("openai") == selection

    plain = store.save("openai", ApiKeyCredentials("sk-open"), "gpt")
    assert (plain.organization, plain.project, plain.header_names) == (None, None, ())
    assert store.credentials("openai", plain.credential_id).headers == ()
    entry = json.loads(store.path.read_text())["providers"]["openai"]
    assert set(entry) == {"method", "model", "credential_id", "credentials"}
    assert entry["credentials"] == {"api_key": "sk-open"}


def test_store_without_connection_settings_rewrites_byte_identically(tmp_path):
    path = tmp_path / "auth" / "auth.json"
    value = {
        "version": 1,
        "default_provider": "openai",
        "providers": {
            "anthropic": {
                "method": "api_key",
                "model": "claude-x",
                "credential_id": "a" * 32,
                "credentials": {"api_key": "k-anth"},
            },
            "azure": {
                "method": "api_key",
                "model": "dep",
                "credential_id": "b" * 32,
                "credentials": {"api_key": "k-az"},
                "endpoint": "https://r.openai.azure.com/",
            },
            "ollama": {
                "method": "none",
                "model": "llama3",
                "credential_id": "c" * 32,
                "credentials": None,
            },
            "openai": {
                "method": "subscription",
                "model": "gpt",
                "credential_id": "d" * 32,
                "credentials": {
                    "access_token": "access",
                    "refresh_token": "refresh",
                    "expires_at": 9000,
                    "account_id": "acct",
                    "residency": None,
                },
            },
        },
    }
    raw = (json.dumps(value, sort_keys=True) + "\n").encode()
    _write_store(path, value)
    path.write_bytes(raw)
    store = AuthStore(path)
    store.set_default("anthropic")
    store.set_default("openai")
    assert path.read_bytes() == raw
    assert all(
        (item.organization, item.project, item.header_names) == (None, None, ())
        for item in store.read().selections
    )


def test_connection_settings_values_stay_out_of_repr(tmp_path):
    store = AuthStore(tmp_path / "auth" / "auth.json")
    selection = store.save("openai", _gateway_credentials(), "gpt", organization="org-1")
    credentials = store.credentials("openai", selection.credential_id)
    for text in (repr(credentials), repr(selection), repr(store.read())):
        assert "gw-secret" not in text
        assert "tag-secret" not in text
        assert "sk-open" not in text
    assert "X-Gateway" in repr(selection)


def test_saved_secrets_lists_header_values(tmp_path):
    store = AuthStore(tmp_path / "auth" / "auth.json")
    store.save("openai", _gateway_credentials(), "gpt")
    assert store.saved_secrets("openai") == ("sk-open", "tag-secret", "gw-secret")


@pytest.mark.parametrize(
    "provider, credentials, settings",
    [
        ("anthropic", ApiKeyCredentials("k"), {"organization": "org-1"}),
        ("azure", ApiKeyCredentials("k"), {"project": "proj-1"}),
        ("ollama", None, {"organization": "org-1"}),
        ("openai", OAuthCredentials("a", "r", 9000, "acct"), {"project": "proj-1"}),
        ("openai", ApiKeyCredentials("k"), {"organization": "has space"}),
        ("openai", ApiKeyCredentials("k"), {"project": ""}),
        ("openai", ApiKeyCredentials("k"), {"organization": "o" * 257}),
        ("openai", ApiKeyCredentials("k", headers={"OpenAI-Organization": "o"}), {}),
    ],
)
def test_store_refuses_connection_settings_a_login_cannot_send(
    tmp_path, provider, credentials, settings
):
    store = AuthStore(tmp_path / "auth" / "auth.json")
    with pytest.raises(AuthError) as caught:
        store.save(provider, credentials, "m", **settings)
    for value in settings.values():
        if value:
            assert value not in str(caught.value)
    assert not store.path.exists()


@pytest.mark.parametrize(
    "headers",
    [
        {"Authorization": "do-not-echo"},
        {"x-api-key": "do-not-echo"},
        {"Content-Type": "do-not-echo"},
        {"bad name": "do-not-echo"},
        {"X": "do-not-echo\r\nY: z"},
        {"X": " do-not-echo"},
        {"X": ""},
        {"X": 1},
        {"X-G": "do-not-echo", "x-g": "other"},
        (("X-G", "do-not-echo"), ("X-G", "other")),
        {f"X-{n}": "v" for n in range(33)},
        "X-G",
    ],
)
def test_api_key_credentials_refuse_headers_that_cannot_be_sent(headers):
    with pytest.raises(AuthError) as caught:
        ApiKeyCredentials("k", headers=headers)
    assert "do-not-echo" not in str(caught.value)


def test_api_key_credentials_hold_sorted_immutable_headers():
    credentials = ApiKeyCredentials("k", headers={"X-B": "2", "X-A": "1"})
    assert credentials.headers == (("X-A", "1"), ("X-B", "2"))
    assert credentials == ApiKeyCredentials("k", headers=(("X-B", "2"), ("X-A", "1")))
    assert hash(credentials) == hash(ApiKeyCredentials("k", headers={"X-A": "1", "X-B": "2"}))
    assert ApiKeyCredentials("k").headers == ()
    assert "OpenAI-Project" in dict(
        ApiKeyCredentials("k", headers={"OpenAI-Project": "p"}).headers
    )


def test_provider_selection_checks_connection_settings():
    selection = ProviderSelection(
        "openai", "api_key", "m", "c" * 32, None, "org-1", "proj-1", ("X-B", "X-A")
    )
    assert selection.header_names == ("X-A", "X-B")
    assert ProviderSelection("p", "api_key", "m", "c" * 32).header_names == ()
    for fields in (
        {"method": "subscription", "header_names": ("X-A",)},
        {"method": "none", "header_names": ("X-A",)},
        {"header_names": ("Authorization",)},
        {"header_names": ("OpenAI-Project",)},
        {"header_names": ("X-A", "x-a")},
        {"header_names": ["X-A"]},
        {"method": "subscription", "organization": "org-1"},
        {"provider": "anthropic", "project": "proj-1"},
        {"organization": "has space"},
        {"provider": "ollama", "header_names": ("X-A",)},
    ):
        values = {"provider": "openai", "method": "api_key", **fields}
        with pytest.raises(AuthError):
            ProviderSelection(model="m", credential_id="c" * 32, **values)


@pytest.mark.parametrize("target", ["directory", "file", "symlink"])
def test_store_rejects_insecure_paths(tmp_path, target):
    path = tmp_path / "auth" / "auth.json"
    store = AuthStore(path)
    store.save("openai", ApiKeyCredentials("secret"), "model")
    if target == "directory":
        path.parent.chmod(0o755)
    elif target == "file":
        path.chmod(0o644)
    else:
        other = path.with_name("other.json")
        path.rename(other)
        path.symlink_to(other)
    with pytest.raises(AuthError, match="private|regular|symbolic"):
        store.read()


def test_device_protocol_uses_json_poll_and_server_pkce():
    clock = FakeClock()
    http = HTTP(
        HttpResponse(
            200,
            {"device_auth_id": "device-id", "user_code": "ABC-DEF", "interval": "2"},
        ),
        HttpResponse(403, {"error": "pending"}),
        HttpResponse(404, {}),
        HttpResponse(
            200, {"authorization_code": "auth-code", "code_verifier": "server-verifier"}
        ),
        HttpResponse(200, _tokens()),
    )
    display = []
    auth = OpenAIAuth(request=http, clock=clock, monotonic=clock, sleep=clock.sleep)
    credentials = auth.device_login(lambda url, code: display.append((url, code)))
    assert display == [("https://auth.openai.com/codex/device", "ABC-DEF")]
    assert credentials.account_id == "account-test"
    assert credentials.expires_at == clock.now + 3600
    assert http.calls[0][2]["json_body"] == {
        "client_id": "app_EMoamEEZ73f0CkXaXp7hrann"
    }
    assert http.calls[1][2]["json_body"] == {
        "device_auth_id": "device-id",
        "user_code": "ABC-DEF",
    }
    assert http.calls[-1][2]["form"] == {
        "grant_type": "authorization_code",
        "code": "auth-code",
        "code_verifier": "server-verifier",
        "client_id": "app_EMoamEEZ73f0CkXaXp7hrann",
        "redirect_uri": "https://auth.openai.com/deviceauth/callback",
    }
    assert "refresh-secret" not in repr(credentials)


def test_browser_login_validates_state_and_uses_pkce():
    http = HTTP(HttpResponse(200, _tokens()))
    urls = []

    def callback(redirect_uri, timeout, check_stopped):
        query = parse_qs(urlparse(urls[0]).query)
        return (
            redirect_uri
            + "?"
            + urlencode({"code": "browser-code", "state": query["state"][0]})
        )

    auth = OpenAIAuth(request=http)
    credentials = auth.browser_login(urls.append, await_callback=callback)
    query = parse_qs(urlparse(urls[0]).query)
    verifier = http.calls[0][2]["form"]["code_verifier"]
    challenge = (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
        .decode()
        .rstrip("=")
    )
    assert query["code_challenge"] == [challenge]
    assert query["code_challenge_method"] == ["S256"]
    assert query["redirect_uri"] == ["http://localhost:1455/auth/callback"]
    assert credentials.account_id == "account-test"


def test_browser_state_mismatch_never_exchanges_code():
    http = HTTP()
    with pytest.raises(AuthError, match="state"):
        OpenAIAuth(request=http).browser_login(
            lambda url: None,
            await_callback=lambda uri, timeout, check: uri
            + "?code=secret-code&state=wrong",
        )
    assert http.calls == []


def test_refresh_preserves_missing_metadata_and_rejects_account_switch():
    old = OAuthCredentials("old", "old-refresh", 1000, "acct", "eu")
    http = HTTP(HttpResponse(200, {"access_token": "new", "expires_in": 3600}))
    new = OpenAIAuth(request=http, clock=lambda: 1000).refresh(old)
    assert new == OAuthCredentials("new", "old-refresh", 4600, "acct", "eu")
    assert http.calls[0][2]["form"]["refresh_token"] == "old-refresh"
    with pytest.raises(AuthError, match="account"):
        OpenAIAuth(request=HTTP(HttpResponse(200, _tokens("different")))).refresh(old)


def test_api_key_validation_lists_models_through_official_endpoint():
    http = HTTP(
        HttpResponse(
            200, {"data": [{"id": "z-model"}, {"id": "a-model"}, {"id": "z-model"}]}
        )
    )
    assert OpenAIAuth(request=http).api_models(ApiKeyCredentials("secret")) == (
        "a-model",
        "z-model",
    )
    assert http.calls[0][0:2] == ("GET", "https://api.openai.com/v1/models")
    assert http.calls[0][2]["headers"]["Authorization"] == "Bearer secret"


GATEWAY = "gw-SENTINEL-token-9876"
ORG = "org-SENTINEL-1234"
PROJECT = "proj_SENTINEL-5678"


def _assert_no_settings(error):
    """Check the error and the chain a traceback prints for every setting value."""
    while error is not None:
        for value in (GATEWAY, ORG, PROJECT):
            assert value not in str(error) and value not in repr(error)
        error = error.__cause__ or (None if error.__suppress_context__ else error.__context__)


def test_api_key_validation_sends_organization_project_and_headers():
    http = HTTP(HttpResponse(200, {"data": [{"id": "m"}]}))
    credentials = ApiKeyCredentials("secret", {"X-Gateway": GATEWAY})
    auth = OpenAIAuth(request=http)
    assert auth.api_models(credentials, organization=ORG, project=PROJECT) == ("m",)
    assert http.calls[0][2]["headers"] == {
        "User-Agent": "yian",
        "X-Gateway": GATEWAY,
        "OpenAI-Organization": ORG,
        "OpenAI-Project": PROJECT,
        "Authorization": "Bearer secret",
    }


def test_organization_mismatch_is_an_http_error_without_values():
    body = {"error": {"code": "mismatched_organization", "message": f"{ORG} {PROJECT}"}}
    http = HTTP(HttpResponse(401, body))
    credentials = ApiKeyCredentials("secret", {"X-Gateway": GATEWAY})
    failed = r"^OpenAI authentication request failed \(HTTP 401\)$"
    with pytest.raises(AuthError, match=failed) as caught:
        OpenAIAuth(request=http).api_models(credentials, organization=ORG, project=PROJECT)
    _assert_no_settings(caught.value)
    assert len(http.calls) == 1


@pytest.mark.parametrize(
    "credentials,settings",
    [
        (ApiKeyCredentials("secret"), {"organization": ORG + " x"}),
        (ApiKeyCredentials("secret"), {"project": ""}),
        (ApiKeyCredentials("secret"), {"organization": 5}),
        # OpenAI takes these through organization / project, not as headers.
        (ApiKeyCredentials("secret", {"OpenAI-Organization": ORG}), {}),
        (ApiKeyCredentials("secret", {"openai-project": PROJECT}), {}),
    ],
)
def test_invalid_connection_settings_are_refused_before_any_request(credentials, settings):
    http = HTTP()
    with pytest.raises(AuthError, match="^Invalid OpenAI connection settings") as caught:
        OpenAIAuth(request=http).api_models(credentials, **settings)
    _assert_no_settings(caught.value)
    assert caught.value.__suppress_context__
    assert http.calls == []


def test_transport_error_never_echoes_connection_settings():
    def broken(*args, **kwargs):
        raise OSError(f"{GATEWAY} {ORG} {PROJECT}")

    credentials = ApiKeyCredentials("secret", {"X-Gateway": GATEWAY})
    with pytest.raises(AuthError) as caught:
        OpenAIAuth(request=broken).api_models(credentials, organization=ORG, project=PROJECT)
    _assert_no_settings(caught.value)
    assert caught.value.__suppress_context__


def test_subscription_catalog_uses_account_routing_and_visibility_order():
    http = HTTP(
        HttpResponse(
            200,
            {
                "models": [
                    {"slug": "third", "visibility": "list", "priority": 3},
                    {"slug": "hidden", "visibility": "hide", "priority": 0},
                    {"slug": "first", "visibility": "list", "priority": 1},
                    {"slug": "unavailable", "visibility": "none", "priority": -1},
                ]
            },
        )
    )
    credentials = OAuthCredentials("access", "refresh", 9000, "acct", "eu")
    assert OpenAIAuth(request=http).subscription_models(credentials) == (
        "first",
        "third",
    )
    assert http.calls[0][0:2] == (
        "GET",
        "https://chatgpt.com/backend-api/codex/models?client_version=0.155.1",
    )
    assert http.calls[0][2]["headers"] == {
        "User-Agent": "yian",
        "Authorization": "Bearer access",
        "ChatGPT-Account-Id": "acct",
        "originator": "yian",
        "x-openai-internal-codex-residency": "eu",
    }


@pytest.mark.parametrize(
    "kind,payload",
    [
        ("api", {"data": []}),
        ("api", {"data": [{"id": ""}]}),
        ("subscription", {"models": []}),
        (
            "subscription",
            {"models": [{"slug": "hidden", "visibility": "hide", "priority": 0}]},
        ),
        (
            "subscription",
            {"models": [{"slug": "bad", "visibility": "list", "priority": True}]},
        ),
    ],
)
def test_model_catalog_rejects_malformed_or_empty_lists(kind, payload):
    auth = OpenAIAuth(request=HTTP(HttpResponse(200, payload)))
    with pytest.raises(AuthError):
        if kind == "api":
            auth.api_models(ApiKeyCredentials("key"))
        else:
            auth.subscription_models(
                OAuthCredentials("access", "refresh", 9000, "acct")
            )


@pytest.mark.parametrize("method", ["api", "subscription", "refresh", "device"])
def test_http_failures_do_not_echo_secrets_or_fallback(method):
    http = HTTP(HttpResponse(401, {"error": "secret-do-not-print"}))
    auth = OpenAIAuth(request=http)
    credentials = OAuthCredentials("access", "refresh", 9000, "acct")
    with pytest.raises(AuthError, match="HTTP 401") as caught:
        if method == "api":
            auth.api_models(ApiKeyCredentials("key"))
        elif method == "subscription":
            auth.subscription_models(credentials)
        elif method == "refresh":
            auth.refresh(credentials)
        else:
            auth.device_login(lambda *args: None)
    assert "secret" not in str(caught.value)
    assert len(http.calls) == 1


def test_transport_error_is_sanitized_and_has_no_printed_exception_chain():
    def broken(*args, **kwargs):
        raise OSError("sensitive-token-in-url")

    with pytest.raises(AuthError) as caught:
        OpenAIAuth(request=broken).api_models(ApiKeyCredentials("key"))
    assert "sensitive" not in str(caught.value)
    assert caught.value.__suppress_context__


class Cancelled(Exception):
    pass


def test_cancellation_before_network_or_file_creation(tmp_path):
    def cancel():
        raise Cancelled

    http = HTTP()
    with pytest.raises(Cancelled):
        OpenAIAuth(request=http, check_stopped=cancel).device_login(lambda *args: None)
    assert not http.calls
    path = tmp_path / "missing" / "auth.json"
    with pytest.raises(Cancelled):
        AuthStore(path, check_stopped=cancel).save(
            "openai", ApiKeyCredentials("key"), "model"
        )
    assert not path.parent.exists()


@pytest.mark.parametrize("cancel", [False, True])
def test_device_polling_has_bounded_cancellable_wait(cancel):
    clock = FakeClock()
    http = HTTP(
        HttpResponse(
            200, {"device_auth_id": "id", "usercode": "code", "interval": "5"}
        ),
        HttpResponse(404, {}),
    )

    def check():
        if cancel and clock.now >= 1000.5:
            raise Cancelled

    auth = OpenAIAuth(
        request=http,
        clock=clock,
        monotonic=clock,
        sleep=clock.sleep,
        check_stopped=check,
    )
    with pytest.raises(
        Cancelled if cancel else AuthError, match=None if cancel else "timed out"
    ):
        auth.device_login(lambda *args: None, timeout=1)
    assert len(http.calls) == 2
    assert clock.now <= 1001


@pytest.mark.parametrize(
    "query",
    ["state=bad", "state=bad&state=also-bad", "error=denied", "code=one&code=two"],
)
def test_browser_callback_rejects_invalid_or_denied_responses(query):
    http = HTTP()
    urls = []

    def callback(uri, timeout, check):
        state = parse_qs(urlparse(urls[0]).query)["state"][0]
        return uri + "?" + query + ("&state=" + state if "state" not in query else "")

    with pytest.raises(AuthError):
        OpenAIAuth(request=http).browser_login(urls.append, await_callback=callback)
    assert not http.calls


def test_browser_listener_is_closed_after_success():
    http = HTTP(HttpResponse(200, _tokens()))
    threads = []
    errors = []

    def open_url(url):
        state = parse_qs(urlparse(url).query)["state"][0]

        def send():
            try:
                with socket.create_connection(
                    ("127.0.0.1", 1455), timeout=2
                ) as connection:
                    connection.sendall(
                        f"GET /auth/callback?code=code&state={state} HTTP/1.1\r\nHost: localhost:1455\r\n\r\n".encode()
                    )
                    connection.recv(4096)
            except Exception as error:
                errors.append(error)

        thread = threading.Thread(target=send)
        threads.append(thread)
        thread.start()

    credentials = OpenAIAuth(request=http).browser_login(open_url, timeout=3)
    for thread in threads:
        thread.join(3)
    assert not errors
    assert credentials.account_id == "account-test"
    with socket.socket() as check:
        check.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        check.bind(("127.0.0.1", 1455))


def test_browser_listener_is_closed_when_user_cancels():
    def cancelled(url):
        raise Cancelled

    with pytest.raises(Cancelled):
        OpenAIAuth(request=HTTP()).browser_login(cancelled)
    with socket.socket() as check:
        check.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        check.bind(("127.0.0.1", 1455))


@pytest.mark.parametrize("cancel", [False, True])
def test_lock_wait_is_bounded_and_cancellable(tmp_path, cancel):
    clock = FakeClock()
    path = tmp_path / "auth" / "auth.json"
    AuthStore(path).save("openai", ApiKeyCredentials("key"), "model")

    def check():
        if cancel and clock.now >= 1000.2:
            raise Cancelled

    with path.with_suffix(".json.lock").open("rb") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        store = AuthStore(
            path,
            monotonic=clock,
            sleep=clock.sleep,
            check_stopped=check,
            lock_timeout=0.5,
        )
        with pytest.raises(
            Cancelled if cancel else AuthError, match=None if cancel else "Timed out"
        ):
            store.save("openai", ApiKeyCredentials("new"), "model")
    assert AuthStore(path).selection().model == "model"


def test_atomic_write_failure_retains_previous_credentials_and_default(
    tmp_path, monkeypatch
):
    store = AuthStore(tmp_path / "auth" / "auth.json")
    before = store.save("openai", ApiKeyCredentials("old-key"), "old-model")
    raw = store.path.read_bytes()

    def fail(*args, **kwargs):
        raise OSError("secret-value")

    monkeypatch.setattr(os, "replace", fail)
    with pytest.raises(AuthError) as caught:
        store.save("anthropic", ApiKeyCredentials("new-key"), "new-model")
    assert "secret-value" not in str(caught.value)
    assert store.path.read_bytes() == raw
    assert store.selection() == before
    assert not list(store.path.parent.glob(".auth-*"))


def test_lock_failure_is_not_ignored(tmp_path, monkeypatch):
    store = AuthStore(tmp_path / "auth" / "auth.json")

    def broken(*args):
        raise OSError("unsupported locking")

    monkeypatch.setattr(fcntl, "flock", broken)
    with pytest.raises(AuthError):
        store.save("openai", ApiKeyCredentials("key"), "model")
    assert not store.path.exists()


def test_changed_connection_prevents_refresh_and_account_switch_prevents_save(tmp_path):
    store = AuthStore(tmp_path / "auth" / "auth.json", clock=lambda: 1000)
    first = store.save(
        "openai", OAuthCredentials("old", "refresh", 900, "acct"), "model"
    )
    second = store.save(
        "openai", OAuthCredentials("new", "refresh", 900, "acct"), "model"
    )
    calls = []

    def refresh(credentials):
        calls.append(credentials)
        return OAuthCredentials("switched", "other-refresh", 5000, "other-account")

    with pytest.raises(AuthError, match="changed"):
        store.credentials("openai", first.credential_id, refresh=refresh)
    assert not calls
    raw = store.path.read_bytes()
    with pytest.raises(AuthError, match="account"):
        store.credentials("openai", second.credential_id, refresh=refresh)
    assert store.path.read_bytes() == raw


def test_default_path_uses_composer_environment_only(tmp_path, monkeypatch):
    monkeypatch.delenv("YIAN_AUTH_FILE", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "native"))
    assert AuthStore().path == tmp_path / "config" / "yian" / "auth.json"
    override = tmp_path / "specific" / "tokens.json"
    monkeypatch.setenv("YIAN_AUTH_FILE", str(override))
    assert AuthStore().path == override
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("interval", ["nan", "inf", True])
def test_device_rejects_nonfinite_or_boolean_interval(interval):
    http = HTTP(
        HttpResponse(
            200, {"device_auth_id": "id", "user_code": "code", "interval": interval}
        )
    )
    with pytest.raises(AuthError, match="interval"):
        OpenAIAuth(request=http).device_login(lambda *args: None)
    assert len(http.calls) == 1


def test_browser_callback_deadline_survives_idle_local_connection():
    threads = []

    def open_url(url):
        def idle():
            with socket.create_connection(("127.0.0.1", 1455), timeout=2):
                time.sleep(1)

        thread = threading.Thread(target=idle)
        threads.append(thread)
        thread.start()

    start = time.monotonic()
    try:
        with pytest.raises(AuthError, match="timed out"):
            OpenAIAuth(request=HTTP()).browser_login(open_url, timeout=0.3)
        elapsed = time.monotonic() - start
    finally:
        for thread in threads:
            thread.join(3)
    assert elapsed < 0.8


@pytest.mark.parametrize("cancel", [False, True])
def test_browser_callback_checks_deadline_and_cancellation_during_slow_headers(cancel):
    threads = []
    stopped = threading.Event()

    def open_url(url):
        def trickle():
            with socket.create_connection(("127.0.0.1", 1455), timeout=2) as connection:
                try:
                    for _ in range(20):
                        connection.sendall(b"G")
                        time.sleep(0.05)
                        stopped.set()
                except OSError:
                    pass

        thread = threading.Thread(target=trickle)
        threads.append(thread)
        thread.start()

    def check():
        if cancel and stopped.is_set():
            raise Cancelled

    start = time.monotonic()
    try:
        with pytest.raises(Cancelled if cancel else AuthError):
            OpenAIAuth(request=HTTP(), check_stopped=check).browser_login(
                open_url, timeout=0.25
            )
        elapsed = time.monotonic() - start
    finally:
        for thread in threads:
            thread.join(3)
    assert elapsed < 0.7
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 1455))


@pytest.mark.parametrize("expiry", [10**1000, float("nan"), float("inf"), True])
def test_invalid_expiry_is_rejected_without_unhandled_conversion_errors(expiry):
    with pytest.raises(AuthError):
        OAuthCredentials("access", "refresh", expiry, "acct")
    with pytest.raises(AuthError):
        OpenAIAuth(request=HTTP(HttpResponse(200, _tokens(expires_in=expiry)))).refresh(
            OAuthCredentials("old", "refresh", 1000, "account-test")
        )


def test_open_snapshot_survives_concurrent_atomic_replacement(tmp_path, monkeypatch):
    store = AuthStore(tmp_path / "auth" / "auth.json")
    original = store.save("openai", ApiKeyCredentials("first"), "first-model")
    fd = os.open(store.path, os.O_RDONLY)
    newer = store.save("openai", ApiKeyCredentials("second"), "second-model")
    open_file = os.open

    def stale_descriptor(path, flags, *args, **kwargs):
        if path == "auth.json":
            return os.dup(fd)
        return open_file(path, flags, *args, **kwargs)

    try:
        assert os.fstat(fd).st_nlink == 0
        with monkeypatch.context() as patch:
            patch.setattr(os, "open", stale_descriptor)
            assert store.selection() == original
    finally:
        os.close(fd)
    assert store.selection() == newer


@pytest.mark.parametrize("target", ["lock", "hardlink", "fifo"])
def test_store_rejects_unsafe_file_kinds_and_lock(tmp_path, target):
    store = AuthStore(tmp_path / "auth" / "auth.json")
    selection = store.save("openai", ApiKeyCredentials("secret"), "model")
    if target == "lock":
        store.path.with_suffix(".json.lock").chmod(0o644)
        with pytest.raises(AuthError, match="private"):
            store.save("openai", ApiKeyCredentials("new"), "model")
        assert store.selection() == selection
    elif target == "hardlink":
        os.link(store.path, store.path.with_name("alias"))
        with pytest.raises(AuthError, match="regular"):
            store.read()
    else:
        store.path.unlink()
        os.mkfifo(store.path, 0o600)
        with pytest.raises(AuthError, match="regular"):
            store.read()


@pytest.mark.parametrize("option", ["lock_timeout", "refresh_margin"])
def test_store_rejects_nonfinite_timings(tmp_path, option):
    with pytest.raises(ValueError, match="timing"):
        AuthStore(tmp_path / "auth.json", **{option: float("nan")})


def test_refresh_is_serialized_across_processes(tmp_path):
    path = tmp_path / "auth" / "auth.json"
    selection = AuthStore(path).save(
        "openai", OAuthCredentials("old", "refresh", 900, "acct"), "model"
    )
    context = multiprocessing.get_context("fork")
    count = context.Value("i", 0)
    barrier = context.Barrier(3)

    def access():
        def refresh(credentials):
            with count.get_lock():
                count.value += 1
            return OAuthCredentials("new", "rotated", 5000, "acct")

        barrier.wait(timeout=5)
        result = AuthStore(path, clock=lambda: 1000).credentials(
            "openai", selection.credential_id, refresh=refresh
        )
        assert result.access_token == "new"

    processes = [context.Process(target=access) for _ in range(3)]
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(5)
        assert all(process.exitcode == 0 for process in processes)
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(5)
    assert count.value == 1
    assert AuthStore(path).selection() == selection


@pytest.mark.parametrize("phase", ["headers", "body", "idle_headers", "idle_body"])
@pytest.mark.parametrize("cancel", [False, True])
@pytest.mark.parametrize("scheme", ["http", "https"])
def test_http_trickle_respects_deadline_cancellation_and_closes_socket(
    phase, cancel, scheme, monkeypatch
):
    from yian.llm_clients.auth import openai as openai_auth
    from yian.llm_clients.auth.openai import _request

    if scheme == "https":
        # Exercise the HTTPS connection/parser wiring; cryptographic negotiation
        # remains stdlib-owned and is replaced only for this loopback server.
        def wrap_socket(context, connection, *, server_hostname):
            assert server_hostname == "127.0.0.1"
            return connection

        monkeypatch.setattr(ssl.SSLContext, "wrap_socket", wrap_socket)

    started = threading.Event()
    peer_closed = threading.Event()
    server_errors = []
    cancellation = Cancelled()
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        server.settimeout(2)
        port = server.getsockname()[1]

        def serve():
            try:
                connection, _ = server.accept()
                with connection:
                    connection.settimeout(2)
                    request = bytearray()
                    while b"\r\n\r\n" not in request:
                        request.extend(connection.recv(4096))
                    if phase.endswith("headers"):
                        connection.sendall(b"HTTP/1.1 200 OK\r\nX-Slow: ")
                    else:
                        connection.sendall(
                            b"HTTP/1.1 200 OK\r\nContent-Length: 4096\r\n\r\n"
                        )
                    started.set()
                    try:
                        if not phase.startswith("idle"):
                            for _ in range(20):
                                connection.sendall(b" ")
                                time.sleep(0.05)
                        while connection.recv(4096):
                            pass
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                    peer_closed.set()
            except Exception as error:
                server_errors.append(error)

        def check():
            if cancel and started.is_set():
                raise cancellation

        def request(method, url, **kwargs):
            return _request(
                method,
                f"{scheme}://127.0.0.1:{port}",
                **kwargs,
            )

        monkeypatch.setattr(openai_auth, "_request", request)
        thread = threading.Thread(target=serve)
        thread.start()
        start = time.monotonic()
        try:
            with pytest.raises(Cancelled if cancel else AuthError) as caught:
                OpenAIAuth(check_stopped=check, request_timeout=0.25).api_models(
                    ApiKeyCredentials("test-only-key")
                )
            if cancel:
                assert caught.value is cancellation
            elapsed = time.monotonic() - start
            assert elapsed < 0.7
            assert peer_closed.wait(1)
        finally:
            thread.join(3)
    assert not thread.is_alive()
    assert not server_errors


@pytest.mark.parametrize("status", [200, 401, 302])
def test_builtin_http_decodes_responses_and_never_follows_redirects(status):
    from yian.llm_clients.auth.http import _request

    errors = []
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        server.settimeout(2)
        port = server.getsockname()[1]

        def serve():
            try:
                connection, _ = server.accept()
                with connection:
                    connection.settimeout(2)
                    request = bytearray()
                    while b"\r\n\r\n" not in request:
                        request.extend(connection.recv(4096))
                    body = b'{"data":[{"id":"gpt-test"}]}'
                    headers = (
                        f"HTTP/1.1 {status} Test\r\nContent-Length: {len(body)}\r\n"
                        f"Location: http://127.0.0.1:{port}/redirected\r\nConnection: close\r\n\r\n"
                    ).encode()
                    connection.sendall(headers + body)
                    assert connection.recv(1) == b""
            except Exception as error:
                errors.append(error)

        thread = threading.Thread(target=serve)
        thread.start()
        try:
            response = _request(
                "GET", f"http://127.0.0.1:{port}", headers={}, timeout=1
            )
        finally:
            thread.join(3)
    assert not thread.is_alive()
    assert not errors
    assert response.status_code == status
    assert response.data == {"data": [{"id": "gpt-test"}]}


def _serve_once(body):
    """Serve one HTTP response with ``body`` on a loopback port; return (port, thread, errors)."""
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    server.settimeout(2)
    errors = []

    def serve():
        try:
            connection, _ = server.accept()
            with connection:
                connection.settimeout(2)
                request = bytearray()
                while b"\r\n\r\n" not in request:
                    request.extend(connection.recv(4096))
                connection.sendall(
                    f"HTTP/1.1 200 OK\r\nContent-Length: {len(body)}\r\n"
                    "Connection: close\r\n\r\n".encode() + body
                )
        except Exception as error:
            errors.append(error)
        finally:
            server.close()

    thread = threading.Thread(target=serve)
    thread.start()
    return server.getsockname()[1], thread, errors


@pytest.mark.parametrize(("body_limit", "fits"), [(1024, False), (4096, True)])
def test_builtin_http_body_limit_raises_value_error(body_limit, fits):
    from yian.llm_clients.auth.http import _request

    body = json.dumps({"pad": "x" * 2048}).encode()
    port, thread, errors = _serve_once(body)
    url = f"http://127.0.0.1:{port}"
    try:
        if fits:
            response = _request("GET", url, headers={}, timeout=2, body_limit=body_limit)
            assert response.data == {"pad": "x" * 2048}
        else:
            with pytest.raises(ValueError, match="too large"):
                _request("GET", url, headers={}, timeout=2, body_limit=body_limit)
    finally:
        thread.join(3)
    assert not errors


def test_builtin_http_deadline_during_connect_raises_timeout_error():
    from yian.llm_clients.auth.http import _request

    # The deadline expires once urllib starts connecting, where it wraps OSErrors.
    readings = iter([0.0, 0.0])
    with pytest.raises(TimeoutError):
        _request(
            "GET", "http://127.0.0.1:9", headers={}, timeout=1,
            monotonic=lambda: next(readings, 100.0),
        )


class _StatusError(Exception):
    """A provider error carrying an HTTP ``status_code``, like the OpenAI and Anthropic SDKs'."""

    def __init__(self, message, status_code):
        super().__init__(message)
        self.status_code = status_code


class _CodeError(Exception):
    """A provider error carrying an HTTP ``code``, like google-genai's ``ClientError``."""

    def __init__(self, message, code):
        super().__init__(message)
        self.code = code


def _raised_from(wrapper, cause):
    """Return ``wrapper`` after raising it ``from`` ``cause``."""
    try:
        raise wrapper from cause
    except BaseException as exc:
        return exc


class _UnreadableError(_StatusError):
    """A 400 whose text can't be read."""

    def __str__(self):
        raise RuntimeError("no text")


class _RaisingStatus(Exception):
    """An error whose ``status_code`` property raises something other than AttributeError."""

    @property
    def status_code(self):
        raise RuntimeError("no status")


def _cycle():
    first, second = ValueError("first"), ValueError("second")
    first.__cause__, second.__cause__ = second, first
    return first


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (AuthError("Subscription credentials expired"), True),
        (_StatusError("unauthorized", 401), True),
        (_StatusError("forbidden", 403), True),
        (_StatusError("bad request", 400), False),
        (_StatusError("not found", 404), False),
        (_StatusError("rate limited", 429), False),
        (_StatusError("server error", 500), False),
        (_raised_from(RuntimeError("wrapped"), _StatusError("unauthorized", 401)), True),
        (_CodeError("unauthenticated", 401), True),
        (_raised_from(RuntimeError("model build"), AuthError("no credentials")), True),
        (_StatusError("API key not valid. reason: API_KEY_INVALID", 400), True),
        (_CodeError("API key not valid. reason: API_KEY_INVALID", 400), True),
        (_StatusError("invalid argument", 400), False),
        (_StatusError("unauthorized", True), False),
        (ValueError("bad value"), False),
        (_UnreadableError("unused", 400), False),
        (_RaisingStatus("unreadable status"), False),
        (_cycle(), False),
    ],
    ids=[
        "auth-error", "401", "403", "400", "404", "429", "500", "wrapped-401",
        "code-401", "wrapped-auth-error", "400-api-key-invalid", "code-400-api-key-invalid",
        "400-other", "bool-status", "value-error", "unreadable-text", "raising-status",
        "cause-cycle",
    ],
)
def test_is_auth_failure(exc, expected):
    assert is_auth_failure(exc) is expected
