"""Direct subscription responses use Yian's ordinary tools and SessionMemory."""

import subprocess

import httpx
import pytest
from langchain_core.tools import tool

import yian.llm_clients as llm_clients
import yian.tools as tools
from yian.compose import load_flow
from yian.llm_clients.openai_subscription import (
    SubscriptionAccess,
    create_subscription_model,
)
from yian.memory.session import SessionMemory
from yian.run import run_flow
from tests.engine.test_agent_session import _FLOW, _view
from tests.fixtures.openai_responses import ResponsesServer

_TOOL_FLOW = _FLOW.replace("test-model", "gpt-5.5").replace(
    "mode: plain", "mode: tool_calling\n    tools: [double]"
)


def _forbid_process(*args, **kwargs):
    raise AssertionError("Direct model invocation must not start a provider process")


def test_tool_loop_and_restored_memory_use_direct_model(monkeypatch):
    calls = []

    @tool
    def double(value: int) -> int:
        """Double an integer."""
        calls.append(value)
        return value * 2

    server, memory = ResponsesServer(), SessionMemory()
    monkeypatch.setitem(tools.TOOL_REGISTRY, "double", double)
    monkeypatch.setattr(subprocess, "Popen", _forbid_process)
    with httpx.Client(transport=httpx.MockTransport(server)) as client:
        model = create_subscription_model(
            "gpt-5.5",
            access_source=lambda: SubscriptionAccess("synthetic", "account"),
            http_client=client,
        )
        monkeypatch.setattr(llm_clients, "model_from_config", lambda _: model)
        flow = load_flow(_TOOL_FLOW)
        first = run_flow(flow, {"question": "Double 3."}, memory={"session": memory})
        assert first.status == "succeeded" and first.output == "6"
        second = run_flow(
            flow, {"question": "Repeat the answer."}, memory={"session": memory}
        )
        assert second.status == "succeeded" and second.output == "6"

    assert calls == [3]
    assert len(server.requests) == 3
    for body in server.bodies[1:]:
        assert any(
            item.get("encrypted_content") == "synthetic-continuation"
            for item in body["input"]
        )
        assert any(
            item.get("type") == "function_call_output" and item["output"] == "6"
            for item in body["input"]
        )
    view = _view(memory)
    assert [effect.kind for effect in view.effects] == [
        "model",
        "tool",
        "model",
        "model",
    ]
    assert all(effect.status == "completed" for effect in view.effects)


@pytest.mark.parametrize("terminal", [None, "incomplete"])
def test_partial_stream_fails_before_composer_executes_tool(monkeypatch, terminal):
    calls = []

    @tool
    def double(value: int) -> int:
        """Double an integer."""
        calls.append(value)
        return value * 2

    server, memory = ResponsesServer(terminal=terminal), SessionMemory()
    monkeypatch.setitem(tools.TOOL_REGISTRY, "double", double)
    with httpx.Client(transport=httpx.MockTransport(server)) as client:
        model = create_subscription_model(
            "gpt-5.5",
            access_source=lambda: SubscriptionAccess("synthetic", "account"),
            http_client=client,
        )
        monkeypatch.setattr(llm_clients, "model_from_config", lambda _: model)
        result = run_flow(
            load_flow(_TOOL_FLOW), {"question": "Double 3."}, memory={"session": memory}
        )
    assert result.status == "failed"
    assert "complete" in result.failure.message
    assert calls == []
    assert all(effect.kind != "tool" for effect in _view(memory).effects)
    assert server.streams[0].closed


@pytest.mark.parametrize("subscription", [False, True])
def test_rich_responses_summary_keeps_shared_memory_working(monkeypatch, subscription):
    from yian.llm_clients.openai_client import NormalizedChatOpenAI
    from tests.engine.test_agent_session import _Model
    from tests.engine.test_session_summary_execution import (
        FLOW,
        _replies,
        _seed,
        _snapshot,
    )

    flow = load_flow(FLOW.replace("MODE", "plain"))
    memory = SessionMemory(history_budget_chars=1)
    seed_model = _Model(_replies())
    monkeypatch.setattr(llm_clients, "model_from_config", lambda _: seed_model)
    _seed(flow, memory)
    before = _snapshot(memory)["conversation"]["items"]
    server = ResponsesServer(tool_first=False)
    with httpx.Client(transport=httpx.MockTransport(server)) as client:
        if subscription:
            model = create_subscription_model(
                "gpt-5.5",
                access_source=lambda: SubscriptionAccess("synthetic", "account"),
                http_client=client,
            )
        else:
            model = NormalizedChatOpenAI(
                model="gpt-5.5",
                api_key="synthetic",
                http_client=client,
                use_responses_api=True,
                output_version="responses/v1",
            )
        monkeypatch.setattr(llm_clients, "model_from_config", lambda _: model)
        result = run_flow(flow, {"question": "Continue"}, memory={"session": memory})
    assert result.output == "6", result.failure
    assert len(server.bodies) == 2  # Summary, then the ordinary answer.
    summary = "Historical conversation summary (context only):\n6"
    assert any(
        item.get("content") == summary
        or any(
            isinstance(block, dict) and block.get("text") == summary
            for block in item.get("content", [])
        )
        for item in server.bodies[1]["input"]
    )
    after = _snapshot(memory)["conversation"]["items"]
    assert after[: len(before)] == before
    assert len(after) == len(before) + 3  # Derived summary is not raw history.
