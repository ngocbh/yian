"""A NaN or infinity from the model fails or retries with a located error."""

import math
from threading import Event

from langchain_core.messages import AIMessage
from pydantic import BaseModel
import pytest

import yian.llm_clients as llm_clients
import yian.tools as tools
from yian.compose import load_flow
from yian.events import ModelCallFinished, ToolCallStarted
from yian.memory.session import SessionMemory
from yian.nodes.base import _HostStop
from yian.run import _run_flow_controlled, run_flow
from tests.engine.test_agent_session import _FLOW, _Model, _view
from tests.engine.test_agent_session_tools import _TOOL_FLOW, _Chat, _Tool
from tests.engine.test_agent_steering import Inbox, _flow as _steer_flow, _recorder


class _FloatArguments(BaseModel):
    x: float


class _FloatTool(_Tool):
    args_schema = tool_call_schema = _FloatArguments

    def __init__(self):
        super().__init__(lambda arguments: f"got {arguments['x']}")


def _flow(session):
    text = _TOOL_FLOW if session else _TOOL_FLOW.replace("    long_term_memory: session\n", "")
    return load_flow(text), ({"session": SessionMemory()} if session else None)


def _call(value):
    return AIMessage(content="", tool_calls=[{"name": "lookup", "args": {"x": value}, "id": "c1"}])


def _setup(monkeypatch, replies):
    tool, chat = _FloatTool(), _Chat(replies)
    monkeypatch.setitem(tools.TOOL_REGISTRY, "lookup", tool)
    monkeypatch.setattr(llm_clients, "model_from_config", lambda _: chat)
    return tool, chat


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("session", [True, False])
def test_a_nonfinite_tool_argument_fails_the_agent_at_its_path(monkeypatch, session, value):
    tool, chat = _setup(monkeypatch, [_call(value)])
    flow, memory = _flow(session)
    result = run_flow(flow, {"question": "q"}, memory=memory)
    assert result.status == "failed"
    assert result.failure.code == "node.execute"
    assert result.failure.node_id == "agent"
    assert "tool call 'lookup' arguments at x: non-finite number" in result.failure.message
    assert tool.calls == [] and len(chat.calls) == 1


@pytest.mark.parametrize("session", [True, False])
def test_a_nonfinite_argument_in_a_later_call_stops_every_call_of_the_reply(monkeypatch, session):
    reply = AIMessage(content="", tool_calls=[
        {"name": "lookup", "args": {"x": 1.0}, "id": "c1"},
        {"name": "lookup", "args": {"x": math.nan}, "id": "c2"},
    ])
    tool, _ = _setup(monkeypatch, [reply])
    flow, memory = _flow(session)
    result = run_flow(flow, {"question": "q"}, memory=memory)
    assert result.status == "failed" and result.failure.code == "node.execute"
    assert "arguments at x: non-finite number nan" in result.failure.message
    assert tool.calls == []


def test_a_nonfinite_reply_is_recorded_as_a_failed_model_effect(monkeypatch):
    _, chat = _setup(monkeypatch, [_call(math.nan)])
    flow, memory = _flow(True)
    assert run_flow(flow, {"question": "q"}, memory=memory).status == "failed"
    view = _view(memory["session"])
    assert [effect.status for effect in view.effects] == ["failed"]
    assert view.effects[0].failure.code == "api.model.failed"
    # Only the request's input items are stored; the bad reply is not.
    assert [item.payload["message"]["type"] for item in view.items] == ["system", "human"]
    chat.replies = iter([AIMessage(content="second")])
    assert run_flow(flow, {"question": "again"}, memory=memory).output == "second"


@pytest.mark.parametrize("session", [True, False])
def test_live_events_show_the_model_call_but_no_tool_call(monkeypatch, session):
    _setup(monkeypatch, [_call(math.nan)])
    flow, memory = _flow(session)
    events = []
    assert run_flow(flow, {"question": "q"}, memory=memory, on_event=events.append).status == "failed"
    finished = [event for event in events if isinstance(event, ModelCallFinished)]
    assert finished and all(event.error is None for event in finished)
    assert not any(isinstance(event, ToolCallStarted) for event in events)


def test_metadata_nan_is_ignored_without_memory_and_fails_with_memory(monkeypatch):
    def reply():
        return AIMessage(
            content="hi",
            response_metadata={"logprobs": {"content": [{"logprob": -math.inf}]}},
        )

    monkeypatch.setattr(llm_clients, "model_from_config", lambda _: _Model([reply(), reply()]))
    plain = load_flow(_FLOW.replace("    long_term_memory: session\n", ""))
    assert run_flow(plain, {"question": "q"}).output == "hi"
    memory = SessionMemory()
    result = run_flow(load_flow(_FLOW), {"question": "q"}, memory={"session": memory})
    assert result.status == "failed" and result.failure.code == "node.execute"
    assert "model reply at response_metadata.logprobs.content[0].logprob" in result.failure.message
    assert [effect.status for effect in _view(memory).effects] == ["failed"]


@pytest.mark.parametrize("session", [True, False])
def test_a_stop_during_a_nonfinite_reply_stops_the_run(monkeypatch, session):
    stop = Event()

    def stop_then_reply(messages):
        stop.set()
        return _call(math.nan)

    tool, _ = _setup(monkeypatch, [stop_then_reply])
    flow, memory = _flow(session)
    with pytest.raises(_HostStop):
        _run_flow_controlled(flow, {"question": "q"}, memory=memory, stop_event=stop)
    assert tool.calls == []
    if session:
        view = _view(memory["session"])
        assert [effect.status for effect in view.effects] == ["interrupted"]
        assert view.closed_invocations[-1].status == "interrupted"


@pytest.mark.parametrize("session", [True, False])
def test_an_interrupt_with_a_nonfinite_tool_call_fails_the_agent(monkeypatch, session):
    inbox, ran = Inbox(), []

    def lookup(x: float) -> str:
        """Look up a number."""
        ran.append(x)
        return "found"

    def interrupt_on_reply(messages):
        inbox.interrupt.add("chat")
        return _call(math.nan)

    _recorder(monkeypatch, [interrupt_on_reply])
    memory = {"session": SessionMemory()} if session else None
    flow = load_flow(_steer_flow(tools=["lookup"], memory=session))
    result = run_flow(flow, {}, steering=inbox, tools={"lookup": lookup}, memory=memory)
    assert result.status == "failed" and result.failure.code == "node.execute"
    assert "tool call 'lookup' arguments at x: non-finite number nan" in result.failure.message
    assert ran == []
    if session:
        assert [(e.kind, e.status) for e in _view(memory["session"]).effects] == [("model", "failed")]
