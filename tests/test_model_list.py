"""Model listing per provider host, through an injected request (no network)."""

import math
import socket
import threading
import time
from urllib.error import URLError
from urllib.parse import parse_qs, urlsplit

import pytest

from agent_composer.llm_clients import model_list
from agent_composer.llm_clients.auth.common import HttpResponse
from agent_composer.llm_clients.factory import _OPENAI_COMPATIBLE
from agent_composer.llm_clients.model_list import ModelListError, list_models

KEY = "sk-test-SENTINEL-0123456789"
BODY_MARKER = "Incorrect API key provided"
AUTH_HEADERS = ("authorization", "x-api-key", "x-goog-api-key")


class Server:
    """Fake `request`: records each call and answers it with `handler`."""

    def __init__(self, handler):
        self.handler = handler
        self.calls = []

    def __call__(self, method, url, *, headers, timeout):
        assert method == "GET"
        call = {
            "url": url,
            "base": url.split("?")[0],
            "params": {k: v[0] for k, v in parse_qs(urlsplit(url).query).items()},
            "headers": {k.lower(): v for k, v in headers.items()},
            "timeout": timeout,
        }
        self.calls.append(call)
        return self.handler(call)


def _ok(data):
    return Server(lambda call: HttpResponse(200, data))


def _never_called(*args, **kwargs):
    raise AssertionError("no request expected")


def _chain(error):
    """Yield `error` and every exception chained to it."""
    while error is not None:
        yield error
        error = error.__cause__ or error.__context__


def _assert_clean(error, message):
    assert str(error) == message
    assert error.__cause__ is None and error.__context__ is None
    for link in _chain(error):
        for text in (str(link), repr(link)):
            assert KEY not in text
            assert BODY_MARKER not in text


def test_anthropic_pages_with_after_id_and_key_headers():
    def handler(call):
        if "after_id" not in call["params"]:
            return HttpResponse(200, {
                "data": [{"id": "claude-b"}, {"id": "claude-a"}],
                "has_more": True, "last_id": "claude-a",
            })
        return HttpResponse(200, {
            "data": [{"id": "claude-c"}, {"id": "claude-a"}],
            "has_more": False, "last_id": "claude-c",
        })

    server = Server(handler)
    assert list_models("anthropic", KEY, request=server) == ("claude-a", "claude-b", "claude-c")
    assert [c["base"] for c in server.calls] == ["https://api.anthropic.com/v1/models"] * 2
    assert server.calls[0]["params"] == {"limit": "1000"}
    assert server.calls[1]["params"] == {"limit": "1000", "after_id": "claude-a"}
    for call in server.calls:
        assert call["headers"]["x-api-key"] == KEY
        assert call["headers"]["anthropic-version"] == "2023-06-01"
        assert "authorization" not in call["headers"]
        assert 0 < call["timeout"] <= 8


def test_anthropic_custom_base_url():
    server = _ok({"data": [{"id": "m"}], "has_more": False})
    assert list_models("anthropic", KEY, "https://gateway.example.com/", request=server) == ("m",)
    assert server.calls[0]["base"] == "https://gateway.example.com/v1/models"


def test_google_uses_key_header_filters_generate_content_and_strips_prefix():
    def handler(call):
        if "pageToken" not in call["params"]:
            return HttpResponse(200, {
                "models": [
                    {"name": "models/gemini-2.5-pro",
                     "supportedGenerationMethods": ["generateContent", "countTokens"]},
                    {"name": "models/text-embedding-004", "supportedGenerationMethods": ["embedContent"]},
                ],
                "nextPageToken": "tok2",
            })
        return HttpResponse(200, {"models": [
            {"name": "models/gemini-2.5-flash", "supportedGenerationMethods": ["generateContent"]},
            {"name": "models/aqa"},
        ]})

    server = Server(handler)
    assert list_models("google", KEY, request=server) == ("gemini-2.5-flash", "gemini-2.5-pro")
    assert len(server.calls) == 2
    assert server.calls[1]["params"]["pageToken"] == "tok2"
    for call in server.calls:
        assert call["base"] == "https://generativelanguage.googleapis.com/v1beta/models"
        assert call["headers"]["x-goog-api-key"] == KEY
        assert "key" not in call["params"]
        assert "key=" not in call["url"]
        assert KEY not in call["url"]


_DEFAULT_LISTING_URL = {
    "openai": "https://api.openai.com/v1/models",
    "xai": "https://api.x.ai/v1/models",
    "deepseek": "https://api.deepseek.com/models",
    "qwen": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/models",
    "qwen-cn": "https://dashscope.aliyuncs.com/compatible-mode/v1/models",
    "glm": "https://api.z.ai/api/paas/v4/models",
    "glm-cn": "https://open.bigmodel.cn/api/paas/v4/models",
    "minimax": "https://api.minimax.io/v1/models",
    "minimax-cn": "https://api.minimaxi.com/v1/models",
    "openrouter": "https://openrouter.ai/api/v1/models",
    "vllm": "http://localhost:8000/v1/models",
}


def test_every_openai_compatible_route_has_a_listing_url():
    assert set(_DEFAULT_LISTING_URL) == set(_OPENAI_COMPATIBLE)


@pytest.mark.parametrize("provider", _OPENAI_COMPATIBLE)
def test_openai_compatible_default_host(provider):
    server = _ok({"object": "list", "data": [{"id": "z"}, {"id": "a"}, {"id": "z"}]})
    assert list_models(provider, KEY, request=server) == ("a", "z")
    assert [c["url"] for c in server.calls] == [_DEFAULT_LISTING_URL[provider]]
    assert server.calls[0]["headers"]["authorization"] == f"Bearer {KEY}"


@pytest.mark.parametrize("base_url", ["https://gateway.example.com/v1", "https://gateway.example.com/v1/"])
def test_openai_custom_base_url(base_url):
    server = _ok({"data": [{"id": "m"}]})
    assert list_models("openai", KEY, base_url, request=server) == ("m",)
    assert [c["url"] for c in server.calls] == ["https://gateway.example.com/v1/models"]


def test_routed_provider_without_an_endpoint_never_falls_back_to_openai(monkeypatch):
    monkeypatch.setattr(model_list, "_OPENAI_COMPATIBLE", (*_OPENAI_COMPATIBLE, "newcompat"))
    with pytest.raises(ModelListError) as info:
        list_models("newcompat", KEY, request=_never_called)
    _assert_clean(info.value, "unsupported provider")


@pytest.mark.parametrize(
    ("base_url", "url"),
    [
        (None, "http://localhost:11434/api/tags"),
        ("http://127.0.0.1:11434/v1", "http://127.0.0.1:11434/api/tags"),
        ("http://127.0.0.1:11434/v1/", "http://127.0.0.1:11434/api/tags"),
    ],
)
def test_ollama_lists_tags(base_url, url):
    server = _ok({"models": [{"name": "qwen3:8b"}, {"name": "llama3.2:latest"}]})
    assert list_models("ollama", base_url=base_url, request=server) == ("llama3.2:latest", "qwen3:8b")
    assert [c["url"] for c in server.calls] == [url]


def test_ollama_sends_no_key_even_when_given_one():
    # No key is sent, so a LAN http host is not held to the base_url rules.
    server = _ok({"models": [{"name": "m"}]})
    assert list_models("ollama", KEY, "http://192.168.1.5:11434", request=server) == ("m",)
    call = server.calls[0]
    assert call["url"] == "http://192.168.1.5:11434/api/tags"
    assert not set(AUTH_HEADERS) & set(call["headers"])
    assert KEY not in repr(call)


def test_azure_has_no_listing():
    assert list_models("azure", KEY, request=_never_called) is None


@pytest.mark.parametrize("provider", ["nope", None])
def test_unknown_provider_raises(provider):
    with pytest.raises(ModelListError) as info:
        list_models(provider, KEY, request=_never_called)
    _assert_clean(info.value, "unsupported provider")


@pytest.mark.parametrize(
    ("provider", "body"),
    [
        ("openai", {"data": []}),
        ("anthropic", {"data": [], "has_more": False}),
        ("google", {"models": []}),
        ("ollama", {"models": []}),
    ],
)
def test_no_key_sends_no_auth_header_even_with_env_keys(monkeypatch, provider, body):
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.setenv(name, "env-key-must-not-be-sent")
    server = _ok(body)
    assert list_models(provider, request=server) == ()
    assert not set(AUTH_HEADERS) & set(server.calls[0]["headers"])
    assert "env-key-must-not-be-sent" not in repr(server.calls)


@pytest.mark.parametrize(
    ("provider", "base_url", "api_key", "timeout"),
    [
        # A key is sent, so the base_url must pass validate_base_url.
        ("openai", "http://gateway.example.com/v1", KEY, 8),
        ("openai", "https://gateway.example.com/v1?x=1", KEY, 8),
        ("anthropic", "https://user:pw@gateway.example.com", KEY, 8),
        ("google", "https://gateway.example.com/\n", KEY, 8),
        # Not a URL at all, even without a key.
        ("openai", "gateway.example.com/v1", None, 8),
        ("vllm", "ftp://localhost/v1", None, 8),
        # A key http.client would refuse in a header.
        ("openai", None, KEY + "\n", 8),
        ("openai", None, "sk-é", 8),
        ("openai", None, b"bytes-key", 8),
        # The timeout.
        ("openai", None, KEY, 0),
        ("openai", None, KEY, -1),
        ("openai", None, KEY, math.nan),
        ("openai", None, KEY, math.inf),
        ("openai", None, KEY, "8"),
    ],
)
def test_invalid_request_sends_nothing(provider, base_url, api_key, timeout):
    with pytest.raises(ModelListError) as info:
        list_models(provider, api_key, base_url, timeout=timeout, request=_never_called)
    _assert_clean(info.value, "invalid request")


def test_keyless_http_base_url_is_allowed():
    server = _ok({"data": [{"id": "m"}]})
    assert list_models("vllm", base_url="http://10.0.0.5:8000/v1", request=server) == ("m",)


@pytest.mark.parametrize("provider", ["openai", "anthropic", "google", "openrouter"])
def test_http_error_carries_only_the_status(provider):
    server = Server(lambda call: HttpResponse(401, {"error": {"message": f"{BODY_MARKER}: {KEY}"}}))
    with pytest.raises(ModelListError) as info:
        list_models(provider, KEY, request=server)
    _assert_clean(info.value, "HTTP 401")


def test_refused_redirect_is_an_http_status():
    server = Server(lambda call: HttpResponse(302, None))
    with pytest.raises(ModelListError) as info:
        list_models("anthropic", KEY, request=server)
    _assert_clean(info.value, "HTTP 302")
    assert len(server.calls) == 1


@pytest.mark.parametrize(
    ("raised", "message"),
    [
        (TimeoutError(f"slow {KEY}"), "timed out"),
        (URLError(TimeoutError(KEY)), "can't reach host"),
        (URLError(f"refused {KEY}"), "can't reach host"),
        (ConnectionResetError(KEY), "can't reach host"),
        (RuntimeError(f"{BODY_MARKER} {KEY}"), "can't reach host"),
        (ValueError(f"too large {KEY}"), "invalid response"),
    ],
)
def test_transport_failures_map_to_a_clean_reason(raised, message):
    def request(*args, **kwargs):
        raise raised

    with pytest.raises(ModelListError) as info:
        list_models("openai", KEY, request=request)
    _assert_clean(info.value, message)


def test_interrupt_is_not_swallowed():
    def request(*args, **kwargs):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        list_models("openai", KEY, request=request)


@pytest.mark.parametrize(
    "response",
    [
        HttpResponse(200, None),
        HttpResponse(200, [{"id": "m"}]),
        HttpResponse(200, {"data": "m"}),
        HttpResponse(200, {"data": [{"id": 5}]}),
        HttpResponse(200, {"data": [{"id": {"x": 1}}]}),
        HttpResponse(200, {"models": []}),
        {"data": [{"id": "m"}]},
    ],
)
def test_malformed_response_is_invalid(response):
    with pytest.raises(ModelListError) as info:
        list_models("openai", KEY, request=lambda *a, **k: response)
    _assert_clean(info.value, "invalid response")


def test_timeout_is_shared_by_every_page():
    # Each page takes 0.15s and always has more; a 0.4s budget allows about three.
    def handler(call):
        time.sleep(0.15)
        return HttpResponse(200, {"data": [{"id": "m"}], "has_more": True, "last_id": "m"})

    server = Server(handler)
    with pytest.raises(ModelListError, match=r"^timed out$"):
        list_models("anthropic", KEY, timeout=0.4, request=server)
    assert 2 <= len(server.calls) <= 3
    budgets = [call["timeout"] for call in server.calls]
    assert budgets[0] <= 0.4
    assert budgets == sorted(budgets, reverse=True) and len(set(budgets)) == len(budgets)


def test_slow_drip_body_stops_at_the_deadline(monkeypatch):
    # Real transport over loopback: the server sends headers, then a byte every 50ms.
    monkeypatch.setenv("no_proxy", "*")
    monkeypatch.setenv("NO_PROXY", "*")
    closed = threading.Event()
    errors = []
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        server.settimeout(5)
        port = server.getsockname()[1]

        def serve():
            try:
                connection, _ = server.accept()
                with connection:
                    connection.settimeout(5)
                    request = bytearray()
                    while b"\r\n\r\n" not in request:
                        request.extend(connection.recv(4096))
                    connection.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 100000\r\n\r\n{")
                    try:
                        for _ in range(100):
                            connection.sendall(b" ")
                            time.sleep(0.05)
                    except OSError:
                        pass
                    closed.set()
            except Exception as error:
                errors.append(error)

        thread = threading.Thread(target=serve)
        thread.start()
        started = time.monotonic()
        try:
            with pytest.raises(ModelListError) as info:
                list_models("openai", KEY, f"http://127.0.0.1:{port}/v1", timeout=0.5)
            elapsed = time.monotonic() - started
        finally:
            thread.join(6)
    _assert_clean(info.value, "timed out")
    assert elapsed < 1.0
    assert closed.is_set()
    assert not errors
