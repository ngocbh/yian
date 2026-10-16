"""Connection settings (organization, project, headers) reach the request and nothing else.

The values go into the SDK clients' custom headers, never into a langchain field, so
they stay out of `repr`, `model_dump` and `dumpd`; a model that sends them never follows
a redirect, which would re-send them to another host.
"""

import asyncio
import warnings

import httpx
import openai
import pytest
from langchain_core.load import dumpd

from tests.fixtures.openai_responses import ResponsesServer
from tests.test_llm_client_env_headers import (
    CHAT_REPLY,
    ChatServer,
    fresh_warned,  # noqa: F401  (autouse fixture)
    invoke_both,
    poisoned_env,  # noqa: F401  (fixture)
)
from yian.llm_clients.config import LLMConfig
from yian.llm_clients.factory import create_llm_client, model_from_config

_GATEWAY = "gw-secret-value"
_SETTINGS = {
    "OpenAI-Organization": "org-config",
    "OpenAI-Project": "proj-config",
    "X-Gateway": _GATEWAY,
}
_AZURE_URL = "https://res.openai.azure.com"
_DROP_WARNING = "yian sends only the headers"


def _openai(headers, **kwargs):
    def build(client, async_client):
        return create_llm_client(
            "openai", "gpt-5.5", api_key="sk-explicit", connection_headers=headers,
            http_client=client, http_async_client=async_client, **kwargs,
        ).get_llm()

    return build


def _assert_no_env_values(request):
    for name, value in request.headers.multi_items():
        assert "env-" not in value.lower(), (name, value)
    assert "x-env-injected" not in request.headers


def test_openai_sends_organization_project_and_headers(poisoned_env):
    server = ResponsesServer(tool_first=False)
    with pytest.warns(UserWarning, match=_DROP_WARNING) as record:
        invoke_both(_openai(_SETTINGS), server)
    assert len(server.requests) == 2
    for request in server.requests:
        assert request.headers["authorization"] == "Bearer sk-explicit"
        assert request.headers.get_list("openai-organization") == ["org-config"]
        assert request.headers.get_list("openai-project") == ["proj-config"]
        assert request.headers.get_list("x-gateway") == [_GATEWAY]
        _assert_no_env_values(request)
    [message] = {str(w.message) for w in record if _DROP_WARNING in str(w.message)}
    # OPENAI_ORG_ID and OPENAI_PROJECT_ID were overridden, not dropped.
    assert "OpenAI-Organization" not in message
    assert "OpenAI-Project" not in message
    for text in (
        "X-Env-Injected", "ChatGPT-Account-Id", "llm_config.organization", "project",
        "headers", "yian auth",
    ):
        assert text in message, (text, message)
    assert "env-" not in message


def test_openai_without_env_headers_and_with_settings_does_not_warn(poisoned_env):
    poisoned_env.delenv("OPENAI_CUSTOM_HEADERS")
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        invoke_both(_openai(_SETTINGS), ResponsesServer(tool_first=False))
    assert not [w for w in record if _DROP_WARNING in str(w.message)]


def test_openai_compatible_provider_sends_openai_project_as_a_header(poisoned_env):
    server = ChatServer()
    headers = {"OpenAI-Project": "proj-gateway", "X-Gateway": _GATEWAY}
    invoke_both(
        lambda client, async_client: create_llm_client(
            "deepseek", "deepseek-chat", api_key="ds-explicit", connection_headers=headers,
            http_client=client, http_async_client=async_client,
        ).get_llm(),
        server,
    )
    assert len(server.requests) == 2
    for request in server.requests:
        assert request.headers["authorization"] == "Bearer ds-explicit"
        assert request.headers.get_list("openai-project") == ["proj-gateway"]
        assert request.headers.get_list("x-gateway") == [_GATEWAY]
        _assert_no_env_values(request)


def _azure(headers, **kwargs):
    def build(client, async_client):
        return create_llm_client(
            "azure", "dep", base_url=_AZURE_URL, api_key="az-explicit",
            connection_headers=headers, http_client=client, http_async_client=async_client,
            **kwargs,
        ).get_llm()

    return build


def test_azure_sends_headers(poisoned_env):
    server = ChatServer()
    invoke_both(_azure({"X-Gateway": _GATEWAY}), server)
    assert len(server.requests) == 2
    for request in server.requests:
        assert request.headers["api-key"] == "az-explicit"
        assert "authorization" not in request.headers
        assert request.headers["user-agent"] == "langchain-partner-python-azure-openai"
        assert request.headers.get_list("x-gateway") == [_GATEWAY]
        _assert_no_env_values(request)


_BUILDS = {
    "openai": lambda: create_llm_client(
        "openai", "gpt-5.5", api_key="sk-explicit", connection_headers=_SETTINGS
    ).get_llm(),
    "deepseek": lambda: create_llm_client(
        "deepseek", "deepseek-chat", api_key="ds-explicit",
        connection_headers={"X-Gateway": _GATEWAY},
    ).get_llm(),
    "azure": lambda: create_llm_client(
        "azure", "dep", base_url=_AZURE_URL, api_key="az-explicit",
        connection_headers={"X-Gateway": _GATEWAY},
    ).get_llm(),
}


@pytest.mark.parametrize("provider", list(_BUILDS))
def test_header_values_stay_out_of_the_model_fields(poisoned_env, provider):
    model = _BUILDS[provider]()
    for text in (repr(model), str(model.model_dump()), str(dumpd(model))):
        assert _GATEWAY not in text
        assert "org-config" not in text


def _route(model, handler):
    """Send the model's SDK-built HTTP clients to `handler`, keeping their other settings."""
    transport = httpx.MockTransport(handler)
    for client in (model.root_client._client, model.root_async_client._client):
        client._transport = transport
        # httpx picks a proxy mount from the environment before `_transport`.
        client._mounts = {}


def _invoke_routed(model, handler):
    _route(model, handler)
    model.invoke("hi")
    asyncio.run(model.ainvoke("hi"))


class RedirectServer:
    """Redirect every request to another origin, which would answer it; record them."""

    def __init__(self):
        self.requests = []

    def __call__(self, request):
        self.requests.append(request)
        if request.url.host == "other.test":
            return httpx.Response(200, json=CHAT_REPLY)
        return httpx.Response(307, headers={"location": f"https://other.test{request.url.path}"})


_REDIRECT_BUILDS = {
    "openai": lambda: create_llm_client(
        "openai", "gpt-5.5", api_key="sk-explicit", connection_headers=_SETTINGS,
        max_retries=0,
    ).get_llm(),
    "azure": lambda: create_llm_client(
        "azure", "dep", base_url=_AZURE_URL, api_key="az-explicit",
        connection_headers={"X-Gateway": _GATEWAY}, max_retries=0,
    ).get_llm(),
}


@pytest.mark.parametrize("provider", list(_REDIRECT_BUILDS))
def test_a_model_with_headers_does_not_follow_redirects(poisoned_env, provider):
    model = _REDIRECT_BUILDS[provider]()
    server = RedirectServer()
    _route(model, server)
    with pytest.raises(openai.APIStatusError):
        model.invoke("hi")
    with pytest.raises(openai.APIStatusError):
        asyncio.run(model.ainvoke("hi"))
    assert len(server.requests) == 2
    for request in server.requests:
        assert request.url.host != "other.test"
        assert request.headers["x-gateway"] == _GATEWAY


def test_a_model_without_headers_keeps_the_shared_client(poisoned_env):
    def build(**kwargs):
        return create_llm_client("openai", "gpt-5.5", api_key="sk-explicit", **kwargs).get_llm()

    plain = build()
    shared = (plain.root_client._client, plain.root_async_client._client)
    assert all(client.follow_redirects for client in shared)
    with_headers = build(connection_headers={"X-Gateway": _GATEWAY})
    for client in (with_headers.root_client._client, with_headers.root_async_client._client):
        assert client.follow_redirects is False
        assert client not in shared
    again = build()
    assert (again.root_client._client, again.root_async_client._client) == shared
    assert all(client.follow_redirects for client in shared)
    assert "x-gateway" not in {name.lower() for name in again.root_client._custom_headers}


def test_a_model_with_headers_uses_openai_proxy(poisoned_env):
    poisoned_env.setenv("OPENAI_PROXY", "http://proxy.test:3128")
    model = create_llm_client(
        "openai", "gpt-5.5", api_key="sk-explicit", connection_headers={"X-Gateway": _GATEWAY}
    ).get_llm()
    url = httpx.URL("https://api.openai.com/v1/responses")
    for client in (model.root_client._client, model.root_async_client._client):
        assert client.follow_redirects is False
        transport = client._transport_for_url(url)
        assert transport is not client._transport
        assert transport._pool._proxy_url.host == b"proxy.test"


@pytest.mark.parametrize("build", [_openai, _azure], ids=["openai", "azure"])
@pytest.mark.parametrize("key", ["http_client", "http_async_client"])
def test_a_redirecting_host_client_is_refused_with_headers(poisoned_env, build, key):
    clients = {"http_client": httpx.Client(), "http_async_client": httpx.AsyncClient()}
    clients[key] = (httpx.Client if key == "http_client" else httpx.AsyncClient)(
        follow_redirects=True
    )
    with pytest.raises(
        ValueError,
        match=f"{key} follows redirects, which would send the API key or connection headers "
        "to another host; build it with follow_redirects=False",
    ):
        build({"X-Gateway": _GATEWAY})(clients["http_client"], clients["http_async_client"])


def test_a_redirecting_host_client_is_allowed_without_headers(poisoned_env):
    with httpx.Client(follow_redirects=True) as client:
        create_llm_client("openai", "gpt-5.5", api_key="sk-explicit", http_client=client).get_llm()


@pytest.mark.parametrize(
    ("provider", "headers", "message"),
    [
        ("openai", {"Authorization": "Bearer x"}, "header Authorization is refused"),
        ("deepseek", {"api-key": "x"}, "header api-key is refused"),
        ("azure", {"X-Gateway": "bad\r\nvalue"}, "header X-Gateway value must be"),
        ("openai", {"OpenAI-Organization": "org with space"}, "organization must be"),
        ("openai", {"OpenAI-Project": "proj-a", "openai-project": "proj-b"}, "more than once"),
    ],
)
def test_get_llm_checks_connection_headers(poisoned_env, provider, headers, message):
    extra = {"base_url": _AZURE_URL} if provider == "azure" else {}
    client = create_llm_client(
        provider, "m", api_key="k", connection_headers=headers, **extra
    )
    with pytest.raises(ValueError, match=message) as error:
        client.get_llm()
    for value in headers.values():
        assert value not in str(error.value)


@pytest.mark.parametrize("carrier", ["LLMConfig", "dict"])
def test_model_from_config_sends_the_settings(poisoned_env, carrier):
    fields = {
        "provider": "openai", "model": "gpt-5.5", "api_key": "sk-host",
        "organization": "org-config", "project": "proj-config",
        "headers": {"X-Gateway": _GATEWAY},
    }
    model = model_from_config(LLMConfig(**fields) if carrier == "LLMConfig" else fields)
    server = ResponsesServer(tool_first=False)
    _invoke_routed(model, server)
    assert len(server.requests) == 2
    for request in server.requests:
        assert request.headers["authorization"] == "Bearer sk-host"
        assert request.headers.get_list("openai-organization") == ["org-config"]
        assert request.headers.get_list("openai-project") == ["proj-config"]
        assert request.headers.get_list("x-gateway") == [_GATEWAY]
        _assert_no_env_values(request)


def test_model_from_config_without_settings_passes_no_connection_headers(poisoned_env):
    model = model_from_config(
        {"provider": "openai", "model": "gpt-5.5", "api_key": "sk-host"}
    )
    assert model.connection_headers == {}
    assert model.root_client._client.follow_redirects
