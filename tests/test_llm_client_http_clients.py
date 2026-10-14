"""A host's ``http_client`` / ``http_async_client`` become the SDK's transport or are rejected."""

import asyncio
import json
import warnings

import httpx
import pytest

from yian.llm_clients.factory import create_llm_client

_GEMINI_REPLY = {
    "candidates": [{
        "content": {"role": "model", "parts": [{"text": "ok"}]},
        "finishReason": "STOP",
        "index": 0,
    }],
    "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1, "totalTokenCount": 2},
}


@pytest.fixture
def clean_env(tmp_path, monkeypatch):
    monkeypatch.setenv("YIAN_AUTH_FILE", str(tmp_path / "auth.json"))
    for name in (
        "GOOGLE_API_KEY", "GEMINI_API_KEY", "GOOGLE_GEMINI_BASE_URL",
        "GOOGLE_GENAI_USE_VERTEXAI", "LANGSMITH_GATEWAY", "OLLAMA_BASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


class _Gemini:
    """Record each Gemini request (URL, key header, JSON body) and answer one reply."""

    def __init__(self):
        self.requests = []

    def __call__(self, request):
        self.requests.append(
            (str(request.url), request.headers.get("x-goog-api-key"), json.loads(request.content))
        )
        if ":streamGenerateContent" in request.url.path:
            return httpx.Response(200, text=f"data: {json.dumps(_GEMINI_REPLY)}\n\n")
        return httpx.Response(200, json=_GEMINI_REPLY)


@pytest.fixture
def gemini():
    """A recorder and sync and async httpx clients routed to it."""
    recorder = _Gemini()
    transport = httpx.MockTransport(recorder)
    sync, async_ = httpx.Client(transport=transport), httpx.AsyncClient(transport=transport)
    yield recorder, sync, async_
    sync.close()
    asyncio.run(async_.aclose())


def _google(**kwargs):
    return create_llm_client("google", "gemini-2.5-flash", api_key="k", **kwargs).get_llm()


def test_google_sends_every_request_through_the_host_clients(clean_env, gemini):
    recorder, sync, async_ = gemini
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)  # langchain warns when a key goes to model_kwargs
        chat = _google(http_client=sync, http_async_client=async_)
    assert chat.model_kwargs == {}
    assert chat.invoke("hi").content == "ok"
    assert asyncio.run(chat.ainvoke("hi")).content == "ok"
    assert "".join(chunk.content for chunk in chat.stream("hi")) == "ok"
    base = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash"
    assert [(url.split("?")[0], key) for url, key, _ in recorder.requests] == [
        (f"{base}:generateContent", "k"),
        (f"{base}:generateContent", "k"),
        (f"{base}:streamGenerateContent", "k"),
    ]
    for _, _, body in recorder.requests:
        assert "http_client" not in body and "http_async_client" not in body
    assert not {"http_client", "http_async_client"} & set(chat.model_dump())


def test_google_does_not_close_the_host_clients(clean_env, gemini):
    _, sync, async_ = gemini
    chat = _google(http_client=sync, http_async_client=async_)
    asyncio.run(chat.aclose())
    assert not sync.is_closed and not async_.is_closed


def test_google_uses_a_non_redirecting_host_client_with_a_base_url(clean_env, gemini):
    recorder, sync, _ = gemini
    chat = _google(base_url="https://gateway.test", http_client=sync)
    assert chat.invoke("hi").content == "ok"
    assert [url.split("?")[0] for url, _, _ in recorder.requests] == [
        "https://gateway.test/v1beta/models/gemini-2.5-flash:generateContent"
    ]


@pytest.mark.parametrize(
    ("key", "factory"),
    [
        ("http_client", lambda: httpx.Client(follow_redirects=True)),
        ("http_async_client", lambda: httpx.AsyncClient(follow_redirects=True)),
    ],
)
def test_google_rejects_a_redirecting_host_client_beside_a_base_url(clean_env, key, factory):
    client = factory()
    try:
        with pytest.raises(ValueError, match=f"{key} follows redirects.*follow_redirects=False"):
            _google(base_url="https://gateway.test", **{key: client})
    finally:
        if key == "http_client":
            client.close()
        else:
            asyncio.run(client.aclose())


def test_google_allows_a_redirecting_host_client_without_a_base_url(clean_env):
    with httpx.Client(follow_redirects=True) as client:
        _google(http_client=client)


@pytest.mark.parametrize(
    ("key", "kind"), [("http_client", "Client"), ("http_async_client", "AsyncClient")]
)
def test_google_rejects_a_host_client_of_the_wrong_type(clean_env, key, kind):
    # The accepted classes depend on whether httpx2 is installed; httpx's is always one.
    with pytest.raises(TypeError, match=rf"{key} must be an httpx\.{kind}\b.*, got object$"):
        _google(**{key: object()})
