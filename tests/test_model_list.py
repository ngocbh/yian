"""Model listing per provider host, over `httpx.MockTransport` (no network)."""

import time

import httpx
import pytest

from agent_composer.llm_clients.factory import _OPENAI_COMPATIBLE
from agent_composer.llm_clients.model_list import ModelListError, list_models

KEY = "sk-test-SENTINEL-0123456789"
BODY_MARKER = "Incorrect API key provided"


def _serve(handler):
    """Return a transport that records each request before `handler` answers it."""
    seen = []

    def record(request):
        seen.append(request)
        return handler(request)

    return httpx.MockTransport(record), seen


def _never_called(request):
    raise AssertionError("no request expected")


def _chain(error):
    """Yield `error` and every exception chained to it."""
    while error is not None:
        yield error
        error = error.__cause__ or error.__context__


def test_anthropic_pages_with_after_id_and_key_headers():
    def handler(request):
        if "after_id" not in request.url.params:
            return httpx.Response(200, json={
                "data": [{"id": "claude-b"}, {"id": "claude-a"}],
                "has_more": True, "last_id": "claude-a",
            })
        return httpx.Response(200, json={
            "data": [{"id": "claude-c"}, {"id": "claude-a"}],
            "has_more": False, "last_id": "claude-c",
        })

    transport, seen = _serve(handler)
    assert list_models("anthropic", KEY, transport=transport) == ("claude-a", "claude-b", "claude-c")
    assert [str(r.url.copy_with(query=None)) for r in seen] == ["https://api.anthropic.com/v1/models"] * 2
    assert seen[0].url.params["limit"] == "1000"
    assert "after_id" not in seen[0].url.params
    assert seen[1].url.params["after_id"] == "claude-a"
    for request in seen:
        assert request.headers["x-api-key"] == KEY
        assert request.headers["anthropic-version"] == "2023-06-01"
        assert "authorization" not in request.headers


def test_anthropic_custom_base_url():
    transport, seen = _serve(lambda r: httpx.Response(200, json={"data": [{"id": "m"}], "has_more": False}))
    assert list_models("anthropic", KEY, "https://gateway.example.com/", transport=transport) == ("m",)
    assert seen[0].url.copy_with(query=None) == "https://gateway.example.com/v1/models"


def test_google_uses_key_header_filters_generate_content_and_strips_prefix():
    def handler(request):
        if "pageToken" not in request.url.params:
            return httpx.Response(200, json={
                "models": [
                    {"name": "models/gemini-2.5-pro",
                     "supportedGenerationMethods": ["generateContent", "countTokens"]},
                    {"name": "models/text-embedding-004", "supportedGenerationMethods": ["embedContent"]},
                ],
                "nextPageToken": "tok2",
            })
        return httpx.Response(200, json={"models": [
            {"name": "models/gemini-2.5-flash", "supportedGenerationMethods": ["generateContent"]},
            {"name": "models/aqa"},
        ]})

    transport, seen = _serve(handler)
    assert list_models("google", KEY, transport=transport) == ("gemini-2.5-flash", "gemini-2.5-pro")
    assert len(seen) == 2
    assert seen[1].url.params["pageToken"] == "tok2"
    for request in seen:
        assert request.url.copy_with(query=None) == "https://generativelanguage.googleapis.com/v1beta/models"
        assert request.headers["x-goog-api-key"] == KEY
        assert "key" not in request.url.params
        assert "key=" not in str(request.url)
        assert KEY not in str(request.url)


@pytest.mark.parametrize(
    ("provider", "base_url", "url"),
    [
        ("openrouter", None, "https://openrouter.ai/api/v1/models"),
        ("openai", None, "https://api.openai.com/v1/models"),
        ("openai", "https://gateway.example.com/v1", "https://gateway.example.com/v1/models"),
        ("openai", "https://gateway.example.com/v1/", "https://gateway.example.com/v1/models"),
        ("glm", None, "https://api.z.ai/api/paas/v4/models"),
    ],
)
def test_openai_compatible_lists_data_ids(provider, base_url, url):
    transport, seen = _serve(lambda r: httpx.Response(200, json={
        "object": "list", "data": [{"id": "z-model"}, {"id": "a-model"}, {"id": "z-model"}],
    }))
    assert list_models(provider, KEY, base_url, transport=transport) == ("a-model", "z-model")
    assert [str(r.url) for r in seen] == [url]
    assert seen[0].headers["authorization"] == f"Bearer {KEY}"


@pytest.mark.parametrize("provider", _OPENAI_COMPATIBLE)
def test_every_openai_compatible_route_lists_models(provider):
    transport, seen = _serve(lambda r: httpx.Response(200, json={"data": [{"id": "m"}]}))
    assert list_models(provider, KEY, transport=transport) == ("m",)
    assert seen[0].url.path.endswith("/models")


@pytest.mark.parametrize(
    ("base_url", "url"),
    [
        (None, "http://localhost:11434/api/tags"),
        ("http://127.0.0.1:11434/v1", "http://127.0.0.1:11434/api/tags"),
        ("http://127.0.0.1:11434/v1/", "http://127.0.0.1:11434/api/tags"),
    ],
)
def test_ollama_lists_tags(base_url, url):
    transport, seen = _serve(lambda r: httpx.Response(200, json={
        "models": [{"name": "qwen3:8b"}, {"name": "llama3.2:latest"}],
    }))
    assert list_models("ollama", base_url=base_url, transport=transport) == ("llama3.2:latest", "qwen3:8b")
    assert [str(r.url) for r in seen] == [url]
    assert "authorization" not in seen[0].headers


def test_azure_has_no_listing():
    assert list_models("azure", KEY, transport=httpx.MockTransport(_never_called)) is None


def test_unknown_provider_raises():
    with pytest.raises(ModelListError) as info:
        list_models("nope", KEY, transport=httpx.MockTransport(_never_called))
    assert str(info.value) == "unsupported provider"


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
    transport, seen = _serve(lambda r: httpx.Response(200, json=body))
    assert list_models(provider, transport=transport) == ()
    for header in ("authorization", "x-api-key", "x-goog-api-key"):
        assert header not in seen[0].headers
    assert "env-key-must-not-be-sent" not in str(seen[0].url)


@pytest.mark.parametrize("provider", ["openai", "anthropic", "google", "openrouter"])
def test_http_error_carries_only_the_status(provider):
    transport, _ = _serve(lambda r: httpx.Response(401, json={"error": {"message": f"{BODY_MARKER}: {KEY}"}}))
    with pytest.raises(ModelListError) as info:
        list_models(provider, KEY, transport=transport)
    error = info.value
    assert str(error) == "HTTP 401"
    assert error.__cause__ is None and error.__context__ is None
    for link in _chain(error):
        for text in (str(link), repr(link)):
            assert KEY not in text
            assert BODY_MARKER not in text


def test_redirect_is_not_followed():
    transport, seen = _serve(lambda r: httpx.Response(302, headers={"location": "https://collector.example.com/"}))
    with pytest.raises(ModelListError, match=r"^HTTP 302$"):
        list_models("anthropic", KEY, transport=transport)
    assert len(seen) == 1


def test_connection_error_is_cant_reach_host():
    def handler(request):
        raise httpx.ConnectError(f"refused {KEY}", request=request)

    with pytest.raises(ModelListError) as info:
        list_models("openai", KEY, transport=httpx.MockTransport(handler))
    assert str(info.value) == "can't reach host"
    assert info.value.__cause__ is None and info.value.__context__ is None
    assert all(KEY not in repr(link) for link in _chain(info.value))


def test_transport_timeout_is_a_timeout():
    def handler(request):
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(ModelListError, match=r"^timed out$") as info:
        list_models("openai", KEY, transport=httpx.MockTransport(handler))
    assert info.value.__context__ is None


def test_malformed_body_is_an_invalid_response():
    transport, _ = _serve(lambda r: httpx.Response(200, text=f"<html>{KEY}</html>"))
    with pytest.raises(ModelListError) as info:
        list_models("openai", KEY, transport=transport)
    assert str(info.value) == "invalid response"
    assert info.value.__cause__ is None and info.value.__context__ is None


def test_timeout_covers_all_pages_together():
    # Each page takes 0.15s and always has more; a 0.4s budget allows about three.
    def handler(request):
        time.sleep(0.15)
        return httpx.Response(200, json={"data": [{"id": "m"}], "has_more": True, "last_id": "m"})

    transport, seen = _serve(handler)
    started = time.monotonic()
    with pytest.raises(ModelListError, match=r"^timed out$"):
        list_models("anthropic", KEY, timeout=0.4, transport=transport)
    assert time.monotonic() - started < 1.0
    assert 2 <= len(seen) <= 3
    # Each request gets only what is left of the shared budget.
    budgets = [request.extensions["timeout"]["read"] for request in seen]
    assert budgets[0] <= 0.4
    assert budgets == sorted(budgets, reverse=True) and len(set(budgets)) == len(budgets)
