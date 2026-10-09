"""Exercise subscription requests through the SDK without network or credentials."""

import asyncio

import httpx
import pytest
from langchain_core.messages import (
    ChatMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from openai import AuthenticationError

from yian.llm_clients.openai_subscription import (
    SubscriptionAccess,
    create_subscription_model,
)
from yian.nodes.agent.memory import decode_message, encode_message
from yian.nodes.agent.modes.utils import text_of
from tests.fixtures.openai_responses import ResponsesServer


@tool
def double(value: int) -> int:
    """Double an integer."""
    return value * 2


def _access():
    return SubscriptionAccess("synthetic-access", "synthetic-account")


def test_streamed_tools_and_encrypted_continuation_round_trip(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-be-used")
    server = ResponsesServer()
    with httpx.Client(transport=httpx.MockTransport(server)) as client:
        model = create_subscription_model(
            "gpt-5.5",
            access_source=_access,
            http_client=client,
        ).bind_tools([double])
        messages = [SystemMessage("Use the supplied tool."), HumanMessage("Double 3.")]
        reply = model.invoke(messages)
        assert reply.tool_calls[0]["args"] == {"value": 3}
        restored = decode_message(encode_message(reply, "exchange-1"))
        assert restored == reply
        final = model.invoke(
            [
                *messages,
                restored,
                ToolMessage(
                    str(double.invoke(restored.tool_calls[0]["args"])),
                    tool_call_id="call_test",
                ),
            ]
        )
        assert text_of(final) == "6"
    for request, body in zip(server.requests, server.bodies):
        assert str(request.url) == "https://chatgpt.com/backend-api/codex/responses"
        assert request.headers["authorization"] == "Bearer synthetic-access"
        assert request.headers["ChatGPT-Account-Id"] == "synthetic-account"
        assert request.headers["originator"] == "yian"
        assert body["stream"] is True and body["store"] is False
        assert body["instructions"] == "Use the supplied tool."
        assert "reasoning.encrypted_content" in body["include"]
        assert "previous_response_id" not in body
        assert all(
            item.get("role") not in ("system", "developer") for item in body["input"]
        )
    assert any(
        item.get("encrypted_content") == "synthetic-continuation"
        for item in server.bodies[1]["input"]
    )
    assert any(
        item.get("type") == "function_call_output" and item["output"] == "6"
        for item in server.bodies[1]["input"]
    )
    assert all(stream.closed for stream in server.streams)


@pytest.mark.parametrize("terminal", [None, "incomplete"])
def test_partial_response_cannot_return_tool_calls(terminal):
    server = ResponsesServer(terminal=terminal)
    with httpx.Client(transport=httpx.MockTransport(server)) as client:
        model = create_subscription_model(
            "gpt-5.5", access_source=_access, http_client=client
        )
        with pytest.raises(RuntimeError, match="complete"):
            model.bind_tools([double]).invoke("Double 3.")
    assert len(server.requests) == 1
    assert server.streams[0].closed


def test_access_source_is_resolved_each_request_without_storing_bearer():
    server, calls = ResponsesServer(), []

    def access():
        calls.append(None)
        return SubscriptionAccess(f"synthetic-{len(calls)}", "account", residency="us")

    with httpx.Client(transport=httpx.MockTransport(server)) as client:
        model = create_subscription_model(
            "gpt-5.5", access_source=access, http_client=client
        )
        assert calls == []
        model.invoke("First")
        model.invoke("Second")
        assert "synthetic-" not in repr(model)
    assert [r.headers["authorization"] for r in server.requests] == [
        "Bearer synthetic-1",
        "Bearer synthetic-2",
    ]
    assert server.requests[0].headers["x-openai-internal-codex-residency"] == "us"
    assert "synthetic-access" not in repr(_access())


def test_auth_failure_does_not_fall_back_to_api_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-be-used")
    server = ResponsesServer(http_status=401)
    with httpx.Client(transport=httpx.MockTransport(server)) as client:
        model = create_subscription_model(
            "gpt-5.5", access_source=_access, http_client=client, max_retries=0
        )
        with pytest.raises(AuthenticationError):
            model.invoke("Hi")
    assert len(server.requests) == 1
    assert server.requests[0].headers["authorization"] == "Bearer synthetic-access"


def test_missing_credentials_fail_before_http():
    def access():
        raise RuntimeError("login required")

    server = ResponsesServer()
    with httpx.Client(transport=httpx.MockTransport(server)) as client:
        model = create_subscription_model(
            "gpt-5.5", access_source=access, http_client=client
        )
        with pytest.raises(RuntimeError, match="login required"):
            model.invoke("Hi")
    assert server.requests == []


@pytest.mark.parametrize("stop_after", [1, 4])
def test_cooperative_stop_closes_stream_without_reply(stop_after):
    class Stopped(Exception):
        pass

    checks = []

    def check():
        checks.append(None)
        if len(checks) >= stop_after:
            raise Stopped()

    server = ResponsesServer()
    with httpx.Client(transport=httpx.MockTransport(server)) as client:
        model = create_subscription_model(
            "gpt-5.5",
            access_source=_access,
            http_client=client,
            check_stopped=check,
        )
        with pytest.raises(Stopped):
            model.invoke("Hi")
    assert all(stream.closed for stream in server.streams)
    if stop_after == 1:
        assert server.requests == []


def test_system_and_developer_text_blocks_become_instructions():
    server = ResponsesServer()
    with httpx.Client(transport=httpx.MockTransport(server)) as client:
        model = create_subscription_model(
            "gpt-5.5", access_source=_access, http_client=client
        )
        model.invoke(
            [
                SystemMessage([{"type": "text", "text": "First."}]),
                ChatMessage(role="developer", content="Second."),
                HumanMessage("Hi"),
            ]
        )
    assert server.bodies[0]["instructions"] == "First.\n\nSecond."


def test_nontext_system_content_is_rejected_before_http():
    server = ResponsesServer()
    with httpx.Client(transport=httpx.MockTransport(server)) as client:
        model = create_subscription_model(
            "gpt-5.5", access_source=_access, http_client=client
        )
        with pytest.raises(ValueError, match="instruction"):
            model.invoke(
                [
                    SystemMessage([{"type": "image_url", "image_url": "invalid"}]),
                    HumanMessage("Hi"),
                ]
            )
    assert server.requests == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"store": True},
        {"previous_response_id": "old-response"},
        {"max_output_tokens": 50},
        {"extra_body": {"store": True}},
        {"extra_body": {"previous_response_id": "old-response"}},
        {"extra_body": {"include": []}},
        {"extra_body": {"temperature": 0.7}},
    ],
)
def test_incompatible_request_options_fail_before_http(kwargs):
    server = ResponsesServer()
    with httpx.Client(transport=httpx.MockTransport(server)) as client:
        model = create_subscription_model(
            "gpt-5.5", access_source=_access, http_client=client
        )
        with pytest.raises(ValueError, match="subscription"):
            model.invoke("Hi", **kwargs)
    assert server.requests == []


@pytest.mark.parametrize("token", ["", "secret\nvalue", "secret value", None])
def test_invalid_access_value_is_rejected_without_echoing_secret(token):
    with pytest.raises(ValueError) as error:
        SubscriptionAccess(token, "account")
    assert "secret" not in str(error.value)


@pytest.mark.parametrize("terminal", ["completed", None, "incomplete"])
def test_async_uses_same_completion_guard(terminal):
    server = ResponsesServer(terminal=terminal)

    async def invoke():
        async with httpx.AsyncClient(transport=httpx.MockTransport(server)) as client:
            model = create_subscription_model(
                "gpt-5.5", access_source=_access, http_async_client=client
            )
            if terminal == "completed":
                reply = await model.ainvoke("Hi")
                assert reply.tool_calls[0]["args"] == {"value": 3}
            else:
                with pytest.raises(RuntimeError, match="complete"):
                    await model.ainvoke("Hi")

    asyncio.run(invoke())
    assert server.streams[0].closed
