"""`LLMConfig.base_url`: a host endpoint bound to the key and provider beside it."""

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import openai
import pytest
from langchain_core.messages import HumanMessage
from pydantic import ValidationError

from yian.llm_clients.auth import AuthStore, OAuthCredentials
from yian.llm_clients.config import LLMConfig, host_only_field_error, pop_host_credentials
from yian.llm_clients.connection import (
    check_model_config,
    resolve_model_config,
    split_model_config,
)
from yian.llm_clients.factory import create_llm_client, model_from_config
from yian.llm_clients.openai_client import OpenAIClient
from tests.fixtures.openai_responses import ResponsesServer

SENTINEL = "sk-sentinel-7d1f3b20"
AZURE_V1 = "https://res.openai.azure.com/openai/v1/"
AZURE_RESOURCE = "https://res.openai.azure.com/"


def test_a_set_base_url_dumps_and_an_unset_one_adds_nothing():
    config = LLMConfig(provider="openai", model="gpt-5.4", base_url=AZURE_V1, api_key=SENTINEL)
    assert config.model_dump() == {
        "provider": "openai", "model": "gpt-5.4", "base_url": AZURE_V1,
    }
    assert SENTINEL not in config.model_dump_json()
    # Recorded snapshots compare full dumps, so an unset base_url must add nothing.
    for dumped in (
        LLMConfig(provider="openai", model="gpt-5.5").model_dump(exclude_none=False),
        LLMConfig(provider="openai", model="gpt-5.5").model_dump(mode="json"),
    ):
        assert "base_url" not in dumped


@pytest.mark.parametrize(
    "url",
    [
        AZURE_V1,
        "https://gateway.example.com",
        "http://localhost:8000/v1",
        "http://127.0.0.1:11434/v1",
        "http://[::1]:8000/v1",
        "http://[::1]/v1",
        "https://[::1]:8443",
        "https://gateway.example.com:443/v1",
        "https://gateway.example.com:65535/v1",
        "https://xn--gtewy-hra.example.com/v1",
        "https://gateway.example.com/openai/v1/deployments/a-b_c~d%20e",
        pytest.param(
            "https://gateway.example.com/" + "a" * (2048 - len("https://gateway.example.com/")),
            id="2048-chars",
        ),
    ],
)
def test_https_and_local_http_urls_are_accepted(url):
    # The stored string is the validated one, and the client parses the same host.
    assert LLMConfig(provider="vllm", base_url=url, api_key=SENTINEL).base_url == url
    assert httpx.URL(url).host


@pytest.mark.parametrize(
    "url",
    [
        "http://gateway.example.com/v1",
        "http://localhost.example.com/v1",
        "ftp://localhost/v1",
        "res.openai.azure.com/openai/v1/",
        "https://",
        "https://user:pw@res.openai.azure.com/openai/v1/",
        "https://res.openai.azure.com/openai/v1/?api-version=1",
        "https://res.openai.azure.com/openai/v1/#frag",
        "https://[bad/v1",
        "https://res.openai.azure.com/openai/v1/?",
        "https://res.openai.azure.com/openai/v1/#",
        "https://user@res.openai.azure.com/openai/v1/",
        "//res.openai.azure.com/openai/v1/",
        # Whitespace and control characters, which urlsplit strips but httpx keeps.
        " https://res.openai.azure.com/openai/v1/",
        "https://res.openai.azure.com/openai/v1/ ",
        "https://res.openai.azure.com/openai/v1/\n",
        "https://res.openai.azure.com/openai/v1/\r\n",
        "https://res.openai.\tazure.com/openai/v1/",
        "http://local\thost/v1",
        "https://res.openai azure.com/openai/v1/",
        "https://res.openai.azure.com/openai/v1/\x00",
        "https://res.openai.azure.com/openai/v1/\x7f",
        "https://res.openai.azure.com/openai/v1/\u00a0",
        # Ports.
        "https://gateway.example.com:abc/v1",
        "https://gateway.example.com:99999/v1",
        "https://gateway.example.com:65536/v1",
        "https://gateway.example.com:0/v1",
        "https://gateway.example.com:/v1",
        "https://gateway.example.com:+443/v1",
        "https://gateway.example.com:4_43/v1",
        # Hosts: ASCII only, no IDNA normalisation, exact local names.
        "https://gateway\u3002example.com/v1",
        "https://g\u00e4teway.example.com/v1",
        "https://\uff47ateway.example.com/v1",
        "https://%67ateway.example.com/v1",
        "https://gateway.example.com\\@collector.example.com/v1",
        "http://localhost.evil.com/v1",
        "http://127.0.0.1.nip.io/v1",
        "http://127.0.0.1@gateway.example.com/v1",
        "http://[0:0:0:0:0:0:0:1]:8000/v1",
        "https://[::1%25lo0]/v1",
        "https://[gateway.example.com]/v1",
        "http://[::1].example.com/v1",
        # The scheme must be lowercase, as written.
        "HTTPS://res.openai.azure.com/openai/v1/",
        "Https://res.openai.azure.com/openai/v1/",
        "HTTP://localhost:8000/v1",
        # Length.
        pytest.param("https://" + "a" * 100_000 + ".example.com/v1", id="long-host"),
        pytest.param(
            "https://gateway.example.com/" + "a" * (2049 - len("https://gateway.example.com/")),
            id="2049-chars",
        ),
    ],
)
def test_other_urls_are_rejected_without_echoing_them(url):
    with pytest.raises(ValidationError, match="base_url") as caught:
        LLMConfig(provider="openai", base_url=url, api_key=SENTINEL)
    # Pydantic's own help link contains "https://", so only longer values are checked.
    if url != "https://":
        assert url not in str(caught.value)
    assert SENTINEL not in str(caught.value)


@pytest.mark.parametrize(
    "fields",
    [{"provider": "openai"}, {"api_key": SENTINEL}, {}],
    ids=["no-key", "no-provider", "neither"],
)
def test_base_url_needs_a_key_and_provider_in_the_same_config(fields):
    with pytest.raises(ValidationError, match="api_key and llm_config.provider") as caught:
        LLMConfig(base_url=AZURE_V1, **fields)
    assert SENTINEL not in str(caught.value)


def test_ollama_rejects_base_url():
    with pytest.raises(ValidationError, match="provider ollama does not take"):
        LLMConfig(provider="ollama", base_url=AZURE_V1, api_key=SENTINEL)


def test_azure_takes_its_resource_endpoint_as_base_url_with_or_without_a_key():
    config = LLMConfig(provider="azure", model="gpt-4o", base_url=AZURE_RESOURCE)
    assert config.model_dump(exclude_none=True) == {
        "provider": "azure", "model": "gpt-4o", "base_url": AZURE_RESOURCE,
    }
    assert LLMConfig(provider="azure", base_url=AZURE_RESOURCE, api_key=SENTINEL).base_url == AZURE_RESOURCE
    with pytest.raises(ValidationError, match="https"):
        LLMConfig(provider="azure", base_url="http://res.openai.azure.com/")


def test_an_authored_layer_may_set_base_url_only_beside_provider_azure():
    assert host_only_field_error({"provider": "azure", "base_url": AZURE_RESOURCE}) is None
    for layer in ({"base_url": AZURE_RESOURCE}, {"provider": "openai", "base_url": AZURE_V1}):
        assert "provider: azure" in host_only_field_error(layer)
    assert host_only_field_error({"provider": "azure", "api_key": SENTINEL}) is not None


def test_azure_config_without_base_url_fails_the_model_check():
    with pytest.raises(ValueError, match="set llm_config.base_url"):
        check_model_config({"provider": "azure", "model": "gpt-4o"})
    with pytest.raises(ValueError, match="set llm_config.base_url"):
        check_model_config({"provider": "azure", "model": "gpt-4o", "api_key": SENTINEL})
    validated, _ = check_model_config(
        {"provider": "azure", "model": "gpt-4o", "base_url": AZURE_RESOURCE}
    )
    assert validated.base_url == AZURE_RESOURCE


def test_pop_host_credentials_takes_the_url_only_with_the_layers_key():
    layer = {"provider": "openai", "model": "gpt-5.4", "base_url": AZURE_V1, "api_key": SENTINEL}
    key, url = pop_host_credentials(layer)
    assert (key.get_secret_value(), url) == (SENTINEL, AZURE_V1)
    assert layer == {"provider": "openai", "model": "gpt-5.4"}
    key, url = pop_host_credentials({"provider": "openai", "api_key": SENTINEL})
    assert (key.get_secret_value(), url) == (SENTINEL, None)
    assert pop_host_credentials({"provider": "openai"}) == (None, None)

    with pytest.raises(ValueError, match="api_key"):
        pop_host_credentials({"provider": "openai", "base_url": AZURE_V1})

    # Without a host key, an Azure endpoint is an ordinary layer setting.
    layer = {"provider": "azure", "model": "gpt-4o", "base_url": AZURE_RESOURCE}
    assert pop_host_credentials(layer) == (None, None)
    assert layer == {"provider": "azure", "model": "gpt-4o", "base_url": AZURE_RESOURCE}
    # A host Azure key never goes to an endpoint another layer names.
    with pytest.raises(ValueError, match="base_url") as caught:
        pop_host_credentials({"provider": "azure", "model": "gpt-4o", "api_key": SENTINEL})
    assert SENTINEL not in str(caught.value)
    key, url = pop_host_credentials(
        {"provider": "azure", "base_url": AZURE_RESOURCE, "api_key": SENTINEL}
    )
    assert (key.get_secret_value(), url) == (SENTINEL, AZURE_RESOURCE)

    unsafe = {"provider": "openai", "base_url": "http://gateway.example.com", "api_key": SENTINEL}
    with pytest.raises(ValueError, match="https") as caught:
        pop_host_credentials(unsafe)
    assert "gateway.example.com" not in str(caught.value)
    assert SENTINEL not in str(caught.value)


# --- clients send the key to the base_url ------------------------------------ #

_ENV = (
    "OPENAI_API_KEY", "OPENAI_BASE_URL", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY",
    "VLLM_API_KEY", "VLLM_BASE_URL", "DEEPSEEK_API_KEY", "OPENROUTER_API_KEY",
    "YIAN_DEFAULT_PROVIDER", "YIAN_DEFAULT_MODEL",
    # AzureChatOpenAI sends this token instead of the configured `api-key` when it is set.
    "AZURE_OPENAI_AD_TOKEN",
)


@pytest.fixture
def clean_env(tmp_path, monkeypatch):
    monkeypatch.setenv("YIAN_AUTH_FILE", str(tmp_path / "auth.json"))
    for name in _ENV:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def _route(monkeypatch, http_client):
    """Send every OpenAIClient request through ``http_client``; no network."""
    original = OpenAIClient.get_llm

    def construct(wrapper):
        wrapper.kwargs.update(http_client=http_client, max_retries=0)
        return original(wrapper)

    monkeypatch.setattr(OpenAIClient, "get_llm", construct)


def _chat_completions(requests):
    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={
            "id": "c", "object": "chat.completion", "created": 0, "model": "m",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "ok"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        })

    return httpx.MockTransport(handler)


@pytest.mark.parametrize("provider", ["vllm", "deepseek", "openrouter"])
def test_openai_compatible_clients_send_the_key_to_the_base_url(clean_env, provider):
    # The provider's own endpoint variable and key must not win over the config.
    clean_env.setenv("VLLM_BASE_URL", "http://127.0.0.1:9/v1")
    clean_env.setenv(f"{provider.upper()}_API_KEY", "sk-environment-key")
    requests = []
    with httpx.Client(transport=_chat_completions(requests)) as http_client:
        _route(clean_env, http_client)
        reply = model_from_config({
            "provider": provider, "model": "m",
            "base_url": "https://gateway.example.com/v1", "api_key": SENTINEL,
        }).invoke([HumanMessage("hi")])
    assert reply.content == "ok"
    assert str(requests[0].url) == "https://gateway.example.com/v1/chat/completions"
    assert requests[0].headers["Authorization"] == f"Bearer {SENTINEL}"


def test_openai_sends_the_key_to_an_azure_v1_base_url(clean_env):
    clean_env.setenv("OPENAI_API_KEY", "sk-environment-key")
    clean_env.setenv("OPENAI_BASE_URL", "https://proxy.example.com/v1")
    # A saved subscription must not take over a keyed config either.
    AuthStore().save(
        "openai",
        OAuthCredentials("access-secret", "refresh-secret", 4102444800, "account"),
        "gpt-5.5",
    )
    config = {"provider": "openai", "model": "gpt-5.4", "base_url": AZURE_V1, "api_key": SENTINEL}
    server = ResponsesServer(tool_first=False)
    with httpx.Client(transport=httpx.MockTransport(server)) as http_client:
        _route(clean_env, http_client)
        model_from_config(config).invoke("hi")
    request = server.requests[0]
    assert str(request.url) == "https://res.openai.azure.com/openai/v1/responses"
    assert request.headers["Authorization"] == f"Bearer {SENTINEL}"
    assert json.loads(request.content)["model"] == "gpt-5.4"


def test_an_openai_base_url_pins_a_host_connection_to_that_endpoint(clean_env):
    resolved = resolve_model_config(
        {"provider": "openai", "model": "gpt-5.4", "base_url": AZURE_V1, "api_key": SENTINEL}
    )
    assert resolved["base_url"] == AZURE_V1
    assert resolved["_connection"]["source"] == "host"
    assert resolved["_connection"]["endpoint"] == AZURE_V1
    assert SENTINEL not in json.dumps({k: v for k, v in resolved.items() if k != "api_key"})
    validated, connection = split_model_config(resolved)
    assert validated.base_url == connection["endpoint"] == AZURE_V1


def test_a_base_url_must_match_its_pinned_connection(clean_env):
    resolved = resolve_model_config(
        {"provider": "openai", "model": "gpt-5.4", "base_url": AZURE_V1, "api_key": SENTINEL}
    )
    moved = {**resolved, "_connection": {**resolved["_connection"], "endpoint": "https://api.openai.com/v1"}}
    stored = {**resolved, "_connection": {
        "method": "api_key", "source": "environment", "credential_id": "a" * 64,
        "endpoint": AZURE_V1, "store_path": None,
    }}
    for config in (moved, stored):
        with pytest.raises(ValueError, match="host connection endpoint"):
            split_model_config(config)
        with pytest.raises(ValueError, match="host connection endpoint"):
            model_from_config(config)


class _Endpoint:
    """A local HTTP server that records each request, then answers 400 or redirects."""

    def __init__(self, redirect_to: str | None = None):
        self.requests: list[tuple[str, dict[str, str]]] = []
        endpoint = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length") or 0))
                endpoint.requests.append(
                    (self.path, {name.lower(): value for name, value in self.headers.items()})
                )
                if redirect_to is not None:
                    self.send_response(307)
                    self.send_header("Location", redirect_to + self.path)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                body = json.dumps({
                    "type": "error",
                    "error": {"type": "invalid_request_error", "message": "refused", "code": 400,
                              "status": "INVALID_ARGUMENT"},
                }).encode()
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self._server.server_port}"

    def close(self):
        self._server.shutdown()
        self._server.server_close()


@pytest.fixture
def endpoints():
    opened = []

    def open_endpoint(redirect_to=None):
        opened.append(_Endpoint(redirect_to))
        return opened[-1]

    yield open_endpoint
    for endpoint in opened:
        endpoint.close()


_DEDICATED = [
    ("anthropic", "claude-sonnet-4-5", "x-api-key", "/v1/messages"),
    ("google", "gemini-2.5-flash", "x-goog-api-key", "/v1beta/models/gemini-2.5-flash:generateContent"),
]


@pytest.mark.parametrize("provider, model, header, path", _DEDICATED)
def test_dedicated_clients_send_the_key_to_the_base_url(
    clean_env, endpoints, provider, model, header, path
):
    configured, decoy = endpoints(), endpoints()
    # Neither the provider's own key nor its endpoint variables may win over the config.
    for name in ("ANTHROPIC_BASE_URL", "ANTHROPIC_API_URL", "GOOGLE_GEMINI_BASE_URL"):
        clean_env.setenv(name, decoy.url)
    clean_env.setenv(f"{provider.upper()}_API_KEY", "sk-environment-key")
    model = model_from_config(
        {"provider": provider, "model": model, "base_url": configured.url, "api_key": SENTINEL}
    )
    with pytest.raises(Exception):
        model.invoke("hi")
    assert [(p, h[header]) for p, h in configured.requests] == [(path, SENTINEL)]
    assert decoy.requests == []


@pytest.mark.parametrize("provider, model, header, path", _DEDICATED)
def test_dedicated_clients_do_not_follow_a_redirect_with_the_key(
    clean_env, endpoints, provider, model, header, path
):
    # These SDKs send the key in a custom header, which httpx keeps on a
    # cross-origin redirect; only the configured URL may receive it.
    elsewhere = endpoints()
    configured = endpoints(redirect_to=elsewhere.url)
    model = model_from_config(
        {"provider": provider, "model": model, "base_url": configured.url, "api_key": SENTINEL}
    )
    with pytest.raises(Exception):
        model.invoke("hi")
    assert [(p, h[header]) for p, h in configured.requests] == [(path, SENTINEL)]
    assert elsewhere.requests == []


@pytest.mark.parametrize(
    "key, factory",
    [
        (None, None),
        ("http_client", lambda: httpx.Client(follow_redirects=True)),
        ("http_client", openai.DefaultHttpxClient),
        ("http_async_client", lambda: httpx.AsyncClient(follow_redirects=True)),
    ],
    ids=["sdk-default", "httpx-follow-redirects", "openai-DefaultHttpxClient", "async"],
)
def test_azure_does_not_send_its_key_past_a_redirect(clean_env, endpoints, key, factory):
    # Azure's key travels in `api-key`, which httpx itself keeps on a cross-origin redirect;
    # openai >= 3.4.0 drops it, for its own client and a host's.
    clean_env.setenv("OPENAI_API_VERSION", "2025-03-01-preview")
    elsewhere = endpoints()
    configured = endpoints(redirect_to=elsewhere.url)
    is_async = key == "http_async_client"
    client = factory() if factory else None
    kwargs = {key: client} if key else {}
    try:
        chat = create_llm_client(
            "azure", "dep", base_url=configured.url, api_key=SENTINEL, max_retries=0, **kwargs
        ).get_llm()
        with pytest.raises(Exception):
            if is_async:
                asyncio.run(chat.ainvoke("hi"))
            else:
                chat.invoke("hi")
    finally:
        if is_async:
            asyncio.run(client.aclose())
        elif client is not None:
            client.close()
    assert [h.get("api-key") for _, h in configured.requests] == [SENTINEL]
    assert len(elsewhere.requests) == 1  # the redirect was followed ...
    assert "api-key" not in elsewhere.requests[0][1]  # ... without the key
    if client is not None:
        assert getattr(chat, "root_async_client" if is_async else "root_client")._client is client
