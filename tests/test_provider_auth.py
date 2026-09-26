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

from agent_composer.llm_clients.auth import (
    ApiKeyCredentials,
    AuthError,
    AuthStore,
    HttpResponse,
    OAuthCredentials,
    OpenAIAuth,
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


@pytest.mark.parametrize(
    "contents", ["{", "[]", '{"version":99}', '{"secret":"do-not-echo"}']
)
def test_malformed_store_fails_without_echoing_payload(tmp_path, contents):
    path = tmp_path / "auth" / "auth.json"
    path.parent.mkdir(mode=0o700)
    path.write_text(contents)
    path.chmod(0o600)
    with pytest.raises(AuthError) as caught:
        AuthStore(path).read()
    assert "do-not-echo" not in str(caught.value)


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
        "User-Agent": "agent-composer",
        "Authorization": "Bearer access",
        "ChatGPT-Account-Id": "acct",
        "originator": "agent-composer",
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
    monkeypatch.delenv("AGENT_COMPOSER_AUTH_FILE", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "native"))
    assert AuthStore().path == tmp_path / "config" / "agent-composer" / "auth.json"
    override = tmp_path / "specific" / "tokens.json"
    monkeypatch.setenv("AGENT_COMPOSER_AUTH_FILE", str(override))
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
    from agent_composer.llm_clients.auth import openai as openai_auth
    from agent_composer.llm_clients.auth.openai import _request

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
    from agent_composer.llm_clients.auth.http import _request

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
