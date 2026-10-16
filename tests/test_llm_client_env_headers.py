"""SDK header and credential environment variables never reach a model request.

The SDKs read header variables, org/project ids and env tokens when a client is built;
yian's clients send exactly the credential and headers yian passes.
"""

import asyncio

import anthropic
import httpx
import pytest
from openai import Omit

from tests.fixtures.openai_responses import ResponsesServer
from tests.test_llm_client_blocks_anthropic import _anthropic_httpx, _Server
from tests.test_llm_client_http_clients import _GEMINI_REPLY
from yian.llm_clients import anthropic_client
from yian.llm_clients.connection import call_credentials, resolve_model_config
from yian.llm_clients.factory import create_llm_client
from yian.llm_clients.openai_client import NormalizedChatOpenAI
from yian.llm_clients.openai_subscription import SubscriptionAccess, create_subscription_model

ENV = {
    "OPENAI_CUSTOM_HEADERS": (
        "X-Env-Injected: from-env\n"
        "Authorization: Bearer env-custom-auth\n"
        "api-key: env-custom-api-key\n"
        "ChatGPT-Account-Id: env-account"
    ),
    "OPENAI_ORG_ID": "env-org",
    "OPENAI_ORGANIZATION": "env-org-2",
    "OPENAI_PROJECT_ID": "env-project",
    "AZURE_OPENAI_AD_TOKEN": "env-ad-token",
    "OPENAI_API_VERSION": "2025-04-01-preview",
    "ANTHROPIC_CUSTOM_HEADERS": (
        "X-Env-Injected: from-env\n"
        "X-Api-Key: env-custom-x-api-key\n"
        "Authorization: Bearer env-custom-anthropic-auth"
    ),
    "ANTHROPIC_AUTH_TOKEN": "env-anthropic-token",
    "OPENAI_ADMIN_KEY": "env-admin",
}

_UNSET = (
    "OPENAI_API_KEY", "AZURE_OPENAI_API_KEY", "ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY",
    "OPENAI_BASE_URL", "OPENAI_API_BASE", "ANTHROPIC_BASE_URL", "GOOGLE_GENAI_USE_VERTEXAI",
    "GOOGLE_GENAI_USE_ENTERPRISE", "GOOGLE_GEMINI_BASE_URL", "OPENAI_PROXY",
)

CHAT_REPLY = {
    "id": "c",
    "object": "chat.completion",
    "created": 0,
    "model": "m",
    "choices": [
        {"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
    ],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}


@pytest.fixture
def poisoned_env(tmp_path, monkeypatch):
    """Set every SDK header and credential variable to an ``env-`` value."""
    monkeypatch.setenv("YIAN_AUTH_FILE", str(tmp_path / "auth.json"))
    for name in _UNSET:
        monkeypatch.delenv(name, raising=False)
    for name, value in ENV.items():
        monkeypatch.setenv(name, value)
    return monkeypatch


def assert_clean(request):
    """Assert a request carries nothing taken from the environment."""
    for name, value in request.headers.multi_items():
        assert "env-" not in value.lower(), (name, value)
    for name in ("openai-organization", "openai-project", "x-env-injected"):
        assert name not in request.headers, name


class ChatServer:
    """Answer Chat Completions requests and record them."""

    def __init__(self):
        self.requests = []

    def __call__(self, request):
        self.requests.append(request)
        return httpx.Response(200, json=CHAT_REPLY)


def invoke_both(build, handler, package=httpx):
    """Build a model on mock transports and call `invoke` and `ainvoke` once each.

    Args:
        build (`Callable[[httpx.Client, httpx.AsyncClient], Any]`):
            Returns the chat model for the given sync and async HTTP clients.
        handler (`Callable[[httpx.Request], httpx.Response]`):
            The mock server.
        package (`module`, *optional*, defaults to `httpx`):
            The httpx package to build the clients from; the SDK's own one for Anthropic.
    """

    async def run():
        transport = package.MockTransport(handler)
        with package.Client(transport=transport) as client:
            async with package.AsyncClient(transport=transport) as async_client:
                model = build(client, async_client)
                model.invoke("hi")
                await model.ainvoke("hi")

    asyncio.run(run())


def _openai(api_key):
    def build(client, async_client):
        return create_llm_client(
            "openai", "gpt-5.5", api_key=api_key,
            http_client=client, http_async_client=async_client,
        ).get_llm()

    return build


def test_openai_responses_send_only_the_explicit_key(poisoned_env):
    server = ResponsesServer(tool_first=False)
    invoke_both(_openai("sk-explicit"), server)
    assert len(server.requests) == 2
    for request in server.requests:
        assert request.headers["authorization"] == "Bearer sk-explicit"
        assert_clean(request)


def test_openai_callable_key_sends_only_its_value(poisoned_env):
    # The shape `model_from_config` passes for a stored key.
    server = ResponsesServer(tool_first=False)
    invoke_both(_openai(lambda: "sk-explicit"), server)
    assert len(server.requests) == 2
    for request in server.requests:
        assert request.headers["authorization"] == "Bearer sk-explicit"
        assert_clean(request)


def test_openai_compatible_provider_sends_only_the_explicit_key(poisoned_env):
    server = ChatServer()
    invoke_both(
        lambda client, async_client: create_llm_client(
            "deepseek", "deepseek-chat", api_key="ds-explicit",
            http_client=client, http_async_client=async_client,
        ).get_llm(),
        server,
    )
    assert len(server.requests) == 2
    for request in server.requests:
        assert request.headers["authorization"] == "Bearer ds-explicit"
        assert_clean(request)


def test_subscription_sends_only_its_own_headers(poisoned_env):
    server = ResponsesServer(tool_first=False)
    invoke_both(
        lambda client, async_client: create_subscription_model(
            "gpt-5.5",
            access_source=lambda: SubscriptionAccess("synthetic-access", "synthetic-account"),
            http_client=client,
            http_async_client=async_client,
        ),
        server,
    )
    assert len(server.requests) == 2
    for request in server.requests:
        assert request.headers["authorization"] == "Bearer synthetic-access"
        assert request.headers["chatgpt-account-id"] == "synthetic-account"
        assert_clean(request)


def test_explicit_default_headers_survive(poisoned_env):
    poisoned_env.setenv("OPENAI_CUSTOM_HEADERS", ENV["OPENAI_CUSTOM_HEADERS"] + "\nx-host: env-host")
    server = ChatServer()
    with httpx.Client(transport=httpx.MockTransport(server)) as client:
        NormalizedChatOpenAI(
            model="m", api_key="k", base_url="https://compat.test/v1",
            default_headers={"X-Host": "h"}, http_client=client,
        ).invoke("hi")
    [request] = server.requests
    assert request.headers.get_list("x-host") == ["h"]
    assert_clean(request)


def test_openai_default_clients_drop_env_headers(poisoned_env):
    llm = create_llm_client("openai", "gpt-5.5", api_key="sk-explicit").get_llm()
    for client in (llm.root_client, llm.root_async_client):
        headers = client.default_headers
        for name in ("X-Env-Injected", "OpenAI-Organization", "OpenAI-Project"):
            assert isinstance(headers.get(name, Omit()), Omit), (name, headers.get(name))
        assert "api-key" not in headers
        # The SDK adds the key itself per request; any Authorization here is a custom one.
        assert "env-" not in str(headers.get("Authorization", ""))
    assert llm.openai_organization is None


def _azure(**kwargs):
    def build(client, async_client):
        return create_llm_client(
            "azure", "dep", base_url="https://res.openai.azure.com",
            http_client=client, http_async_client=async_client, **kwargs,
        ).get_llm()

    return build


def test_azure_sends_only_the_explicit_key(poisoned_env):
    server = ChatServer()
    invoke_both(_azure(api_key="az-explicit"), server)
    assert len(server.requests) == 2
    for request in server.requests:
        assert request.headers["api-key"] == "az-explicit"
        assert "authorization" not in request.headers
        assert request.headers["user-agent"] == "langchain-partner-python-azure-openai"
        assert_clean(request)


def test_azure_user_agent_from_env_does_not_win(poisoned_env):
    poisoned_env.setenv("OPENAI_CUSTOM_HEADERS", ENV["OPENAI_CUSTOM_HEADERS"] + "\nuser-agent: env-ua")
    server = ChatServer()
    invoke_both(_azure(api_key="az-explicit"), server)
    assert len(server.requests) == 2
    for request in server.requests:
        assert request.headers.get_list("user-agent") == ["langchain-partner-python-azure-openai"]
        assert_clean(request)


def test_azure_without_a_key_keeps_the_sdk_order(poisoned_env):
    # An environment connection: the SDK's own order sends AZURE_OPENAI_AD_TOKEN first.
    poisoned_env.setenv("OPENAI_API_KEY", "sk-openai")
    server = ChatServer()
    invoke_both(_azure(), server)
    assert len(server.requests) == 2
    for request in server.requests:
        assert request.headers["authorization"] == "Bearer env-ad-token"
        assert "api-key" not in request.headers
        for _, value in request.headers.multi_items():
            assert "sk-openai" not in value
        for name in ("openai-organization", "openai-project", "x-env-injected"):
            assert name not in request.headers, name


def _anthropic(**kwargs):
    def build(client, async_client):
        return create_llm_client(
            "anthropic", "claude-sonnet-4-5", api_key="sk-ant-explicit",
            http_client=client, http_async_client=async_client, **kwargs,
        ).get_llm()

    return build


def _assert_anthropic_key_only(requests):
    assert len(requests) == 2
    for request in requests:
        assert request.headers["x-api-key"] == "sk-ant-explicit"
        assert "authorization" not in request.headers
        assert "anthropic-version" in request.headers
        assert_clean(request)


def test_anthropic_sends_only_the_explicit_key(poisoned_env):
    server = _Server()
    invoke_both(_anthropic(), server, package=_anthropic_httpx())
    _assert_anthropic_key_only(server.requests)


def test_anthropic_with_a_base_url_sends_only_the_explicit_key(poisoned_env):
    # The package's clients don't follow redirects, as a host base_url requires.
    server = _Server()
    invoke_both(_anthropic(base_url="https://gateway.test"), server, package=_anthropic_httpx())
    _assert_anthropic_key_only(server.requests)
    assert all(request.url.host == "gateway.test" for request in server.requests)


# Without a base_url the SDK's own client is used; with one, yian rebuilds it through
# `with_options`, which reads the environment again.
@pytest.mark.parametrize("base_url", [None, "https://gateway.test"])
def test_anthropic_default_client_drops_env_headers(poisoned_env, base_url):
    chat = create_llm_client(
        "anthropic", "claude-sonnet-4-5", base_url=base_url, api_key="sk-ant-explicit"
    ).get_llm()
    for client in (chat._client, chat._async_client):
        headers = client.default_headers
        assert headers["X-Api-Key"] == "sk-ant-explicit"
        assert "X-Env-Injected" not in headers
        assert "Authorization" not in headers


def test_drop_environment_headers_clears_an_env_auth_token():
    # What a 0.x SDK holds after reading ANTHROPIC_AUTH_TOKEN beside a passed key.
    server = _Server()
    package = _anthropic_httpx()
    with package.Client(transport=package.MockTransport(server)) as http_client:
        client = anthropic.Anthropic(
            api_key="k", auth_token="env-t", http_client=http_client, max_retries=0
        )
        anthropic_client._own_headers(client, {})
        client.messages.create(
            model="m", max_tokens=1, messages=[{"role": "user", "content": "hi"}]
        )
    [request] = server.requests
    assert "authorization" not in request.headers
    assert request.headers["x-api-key"] == "k"


_GOOGLE_ENV = {
    "GOOGLE_GENAI_DEFAULT_HEADERS": '{"X-Env-Injected": "from-env"}',
    "GOOGLE_GENAI_ACCESS_TOKEN": "env-google-access-token",
    "GOOGLE_GENAI_API_KEY": "env-google-genai-key",
    "GOOGLE_API_KEY": "env-google-key",
    "GEMINI_API_KEY": "env-gemini-key",
}


def _google(client, async_client):
    return create_llm_client(
        "google", "gemini-2.5-flash", api_key="g-explicit",
        http_client=client, http_async_client=async_client,
    ).get_llm()


class GeminiServer:
    """Answer Gemini generateContent requests and record them."""

    def __init__(self):
        self.requests = []

    def __call__(self, request):
        self.requests.append(request)
        return httpx.Response(200, json=_GEMINI_REPLY)


def test_google_sends_only_the_explicit_key(poisoned_env):
    # google-genai sends none of these variables beside a passed key; this guards a later SDK.
    for name, value in _GOOGLE_ENV.items():
        poisoned_env.setenv(name, value)
    server = GeminiServer()
    invoke_both(_google, server)
    assert len(server.requests) == 2
    for request in server.requests:
        assert request.headers["x-goog-api-key"] == "g-explicit"
        assert_clean(request)


def _deepseek(client, async_client):
    return create_llm_client(
        "deepseek", "deepseek-chat", api_key="ds-explicit",
        http_client=client, http_async_client=async_client,
    ).get_llm()


_AZURE_URL = "https://res.openai.azure.com"
_TRACKED_CASES = {
    "openai": (
        _openai("sk-explicit"), lambda: ResponsesServer(tool_first=False), httpx,
        {"provider": "openai", "model": "gpt-5.5", "api_key": "sk-explicit"},
    ),
    "deepseek": (
        _deepseek, ChatServer, httpx,
        {"provider": "deepseek", "model": "deepseek-chat", "api_key": "ds-explicit"},
    ),
    "azure-key": (
        _azure(api_key="az-explicit"), ChatServer, httpx,
        {"provider": "azure", "model": "dep", "base_url": _AZURE_URL, "api_key": "az-explicit"},
    ),
    "azure-no-key": (
        _azure(), ChatServer, httpx,
        {"provider": "azure", "model": "dep", "base_url": _AZURE_URL},
    ),
    "anthropic": (
        _anthropic(), _Server, None,
        {"provider": "anthropic", "model": "claude-sonnet-4-5", "api_key": "sk-ant-explicit"},
    ),
    "google": (
        _google, GeminiServer, httpx,
        {"provider": "google", "model": "gemini-2.5-flash", "api_key": "g-explicit"},
    ),
}


@pytest.mark.parametrize("case", list(_TRACKED_CASES))
def test_wire_credentials_are_tracked(poisoned_env, case):
    # Every credential on the wire must be one call_credentials lists, so errors are scrubbed.
    for name, value in _GOOGLE_ENV.items():
        poisoned_env.setenv(name, value)
    build, make_server, package, config = _TRACKED_CASES[case]
    server = make_server()
    invoke_both(build, server, package=package or _anthropic_httpx())
    tracked = call_credentials(resolve_model_config(config))
    assert len(server.requests) == 2
    for request in server.requests:
        sent = [
            request.headers[name].removeprefix("Bearer ")
            for name in ("authorization", "api-key", "x-api-key", "x-goog-api-key")
            if name in request.headers
        ]
        assert sent
        for value in sent:
            assert value in tracked, (value, tracked)
