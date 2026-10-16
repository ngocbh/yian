"""The shared httpx2 client: no redirects, yian's User-Agent, no environment headers."""

import httpx2
import pytest

from yian.llm.transport import DEFAULT_TIMEOUT, USER_AGENT, build_client, check_client


def test_client_refuses_redirects():
    with build_client() as client:
        assert client.follow_redirects is False


def test_user_agent():
    assert USER_AGENT.startswith("yian/")
    with build_client() as client:
        assert client.headers["User-Agent"] == USER_AGENT


def test_default_timeout():
    with build_client() as client:
        assert client.timeout == DEFAULT_TIMEOUT
    with build_client(timeout=3.0) as client:
        assert client.timeout == httpx2.Timeout(3.0)


def test_redirect_response_is_not_followed():
    seen = []

    def handler(request):
        seen.append(str(request.url))
        return httpx2.Response(302, headers={"Location": "https://elsewhere.example/x"})

    with build_client(transport=httpx2.MockTransport(handler)) as client:
        response = client.get("https://api.example/v1")
    assert response.status_code == 302
    assert seen == ["https://api.example/v1"]


def test_sends_no_environment_headers(monkeypatch):
    monkeypatch.setenv("OPENAI_CUSTOM_HEADERS", "Authorization: Bearer env")
    monkeypatch.setenv("ANTHROPIC_CUSTOM_HEADERS", "x-env: 1")
    monkeypatch.setenv("OPENAI_ORG_ID", "org-env")
    # httpx2 ignores environment proxies when a transport is passed
    # (allow_env_proxies = trust_env and transport is None). The variables point at an
    # unused address so a regression in that rule cannot reach the network.
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9")
    seen = []

    def handler(request):
        seen.append(request)
        return httpx2.Response(200)

    with build_client(transport=httpx2.MockTransport(handler)) as client:
        client.get("https://api.example/v1", headers={"X-Caller": "yes"})

    (request,) = seen
    assert {name.lower() for name in request.headers} == {
        "host", "accept", "accept-encoding", "connection", "user-agent", "x-caller",
    }
    assert request.headers["user-agent"] == USER_AGENT
    assert request.headers["x-caller"] == "yes"


def test_check_client_rejects_redirecting_client():
    with httpx2.Client(follow_redirects=True) as client:
        with pytest.raises(ValueError, match="my_client"):
            check_client("my_client", client)


def test_check_client_accepts_built_client():
    with build_client() as client:
        check_client("client", client)
