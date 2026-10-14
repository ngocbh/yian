"""A stop pending when a model or tool call fails stops the run cleanly."""

from threading import Event

from langchain_core.messages import AIMessage, HumanMessage
import pytest

import yian.llm_clients as llm_clients
import yian.tools as tools
from yian.memory.session import SessionMemory
from yian.nodes.agent.memory import invoke_model, invoke_tool_effect
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
    with pytest.raises(_HostStop) as caught:
        _run_flow_controlled(flow, {"question": "q"}, memory=memory, stop_event=stop)
    if session:
        assert str(caught.value.__context__) == "provider boom"
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
    with pytest.raises(_HostStop) as caught:
        _run_flow_controlled(flow, {"question": "q"}, memory=memory, stop_event=stop)
    if session:
        assert str(caught.value.__context__) == "tool boom"
        view = _view(memory["session"])
        assert [effect.status for effect in view.effects] == ["completed", "interrupted"]
        assert view.effects[1].failure.code == "api.tool.interrupted"
        assert view.closed_invocations[-1].status == "interrupted"


def test_a_stop_during_a_failing_structured_call_stops_the_run(monkeypatch):
    """A stop pending when a structured model call fails records the call interrupted."""
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


class _StopOnObserve:
    """A turn that records the observation, then raises a stop that arrived meanwhile."""

    def __init__(self):
        self.observed = []

    def request(self, operation, payload, items=()):
        return 1

    def observe(self, ordinal, status, *, items=(), failure=None):
        self.observed.append(status)
        raise _HostStop()


class _BadStr:
    def __str__(self):
        raise RuntimeError("cannot represent")


def _fail():
    raise RuntimeError("boom")


@pytest.mark.parametrize(
    ("call", "observed"),
    [("model", ["failed"]), ("tool", ["failed"]), ("representation", ["completed"])],
)
def test_a_stop_raised_while_observing_a_failure_wins(call, observed):
    """A stop honored after a failed call is recorded still stops the run."""
    turn = _StopOnObserve()

    with pytest.raises(_HostStop):
        if call == "model":
            invoke_model(turn, _fail, new_messages=[HumanMessage("q")], check_stopped=lambda: None)
        else:
            invoke_tool_effect(
                turn, _fail if call == "tool" else _BadStr, name="lookup", arguments={},
                model_exchange_id="e", call_index=0, tool_call_id="t", check_stopped=lambda: None,
            )
    assert turn.observed == observed
