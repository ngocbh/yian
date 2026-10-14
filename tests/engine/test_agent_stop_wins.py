"""A stop pending when a model or tool call fails stops the run cleanly."""

from threading import Event

from langchain_core.messages import AIMessage
import pytest

import yian.llm_clients as llm_clients
import yian.tools as tools
from yian.memory.session import SessionMemory
from yian.nodes.base import _HostStop
from yian.run import _run_flow_controlled
from tests.engine.test_agent_nonfinite_replies import _flow, _setup
from tests.engine.test_agent_session import _Model, _view
from tests.engine.test_agent_session_structured import _typed_flow
from tests.engine.test_agent_session_tools import _Chat, _Tool, _call


@pytest.mark.parametrize("session", [True, False])
def test_a_stop_during_a_failing_model_call_stops_the_run(monkeypatch, session):
    """A provider error raised while a stop is pending records the call interrupted."""
    stop = Event()

    def stop_then_raise(messages):
        stop.set()
        raise RuntimeError("provider boom")

    _setup(monkeypatch, [stop_then_raise])
    flow, memory = _flow(session)
    with pytest.raises(_HostStop):
        _run_flow_controlled(flow, {"question": "q"}, memory=memory, stop_event=stop)
    if session:
        view = _view(memory["session"])
        assert [effect.status for effect in view.effects] == ["interrupted"]
        assert view.effects[0].failure.code == "api.model.interrupted"
        assert view.closed_invocations[-1].status == "interrupted"


@pytest.mark.parametrize("session", [True, False])
def test_a_stop_during_a_failing_tool_call_stops_the_run(monkeypatch, session):
    """A tool error raised while a stop is pending records the call interrupted."""
    stop = Event()

    def stop_then_raise(arguments):
        stop.set()
        raise RuntimeError("tool boom")

    monkeypatch.setitem(tools.TOOL_REGISTRY, "lookup", _Tool(stop_then_raise))
    monkeypatch.setattr(
        llm_clients, "model_from_config", lambda _: _Chat([_call(), AIMessage(content="done")])
    )
    flow, memory = _flow(session)
    with pytest.raises(_HostStop):
        _run_flow_controlled(flow, {"question": "q"}, memory=memory, stop_event=stop)
    if session:
        view = _view(memory["session"])
        assert [effect.status for effect in view.effects] == ["completed", "interrupted"]
        assert view.effects[1].failure.code == "api.tool.interrupted"
        assert view.closed_invocations[-1].status == "interrupted"


def test_a_stop_during_a_failing_structured_call_stops_the_run(monkeypatch):
    """Structured generation calls the same helper, so a pending stop wins there too."""
    stop = Event()

    def stop_then_raise(messages):
        stop.set()
        raise RuntimeError("provider boom")

    monkeypatch.setattr(llm_clients, "model_from_config", lambda _: _Model([stop_then_raise]))
    memory = SessionMemory()
    with pytest.raises(_HostStop):
        _run_flow_controlled(
            _typed_flow(native=False), {"question": "q"}, memory={"session": memory}, stop_event=stop
        )
    view = _view(memory)
    assert [effect.status for effect in view.effects] == ["interrupted"]
    assert view.closed_invocations[-1].status == "interrupted"
