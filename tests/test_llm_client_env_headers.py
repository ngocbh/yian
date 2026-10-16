"""SDK header and credential environment variables never reach a model request.

The SDKs read header variables, org/project ids and env tokens when a client is built;
yian's clients send exactly the credential and headers yian passes.
"""

import asyncio

import httpx
import pytest
from openai import Omit

from tests.fixtures.openai_responses import ResponsesServer
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


def invoke_both(build, handler):
    """Build a model on mock transports and call `invoke` and `ainvoke` once each.

    Args:
        build (`Callable[[httpx.Client, httpx.AsyncClient], Any]`):
            Returns the chat model for the given sync and async HTTP clients.
        handler (`Callable[[httpx.Request], httpx.Response]`):
            The mock server.
    """

    async def run():
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as async_client:
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
