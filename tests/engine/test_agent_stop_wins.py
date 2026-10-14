"""A stop pending when a model or tool call fails stops the run cleanly."""

import math
from threading import Event
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage
import pytest

from yian.events import ModelCallStarted
import yian.llm_clients as llm_clients
import yian.tools as tools
from yian.memory import MemoryContext
from yian.memory.session import SessionMemory
from yian.nodes.agent.memory import invoke_model, invoke_tool_effect
from yian.nodes.agent.structured import generate_structured
from yian.nodes.base import _HostStop
from yian.run import _run_flow_controlled
from yian.typesys.values import Type, ValueKind
from tests.engine._session_fakes import harness, request
from tests.engine.test_agent_nonfinite_replies import _NATIVE, _flow, _setup
from tests.engine.test_agent_session import _Model, _view
from tests.engine.test_agent_session_structured import _typed_flow
from tests.engine.test_agent_session_tools import _Chat, _Tool, _call


def _contexts(error):
    """Return the messages of every exception in `error`'s `__context__` chain."""
    messages = []
    while error.__context__ is not None:
        error = error.__context__
        messages.append(str(error))
    return messages


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
        assert not getattr(caught.value, "__notes__", None)
        assert "provider boom" in _contexts(caught.value)
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
        assert not getattr(caught.value, "__notes__", None)
        assert "tool boom" in _contexts(caught.value)
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
    with pytest.raises(_HostStop) as caught:
        _run_flow_controlled(
            _typed_flow(native=False), {"question": "q"}, memory={"session": memory}, stop_event=stop
        )
    assert not getattr(caught.value, "__notes__", None)
    assert "provider boom" in _contexts(caught.value)
    view = _view(memory)
    assert [effect.status for effect in view.effects] == ["interrupted"]
    assert view.effects[0].failure.code == "api.model.interrupted"
    assert view.closed_invocations[-1].status == "interrupted"


class _StopOnObserve:
    """A turn that records the observation, then raises a stop that arrived meanwhile."""

    def __init__(self):
        self.observed = []
        self.stop = None

    def request(self, operation, payload, items=()):
        return 1

    def observe(self, ordinal, status, *, items=(), failure=None):
        self.observed.append(status)
        self.stop = _HostStop()
        raise self.stop


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
    """A stop raised by `observe` after recording the outcome is raised instead of a host fault."""
    turn = _StopOnObserve()

    with pytest.raises(_HostStop) as caught:
        if call == "model":
            invoke_model(turn, _fail, new_messages=[HumanMessage("q")], check_stopped=lambda: None)
        else:
            invoke_tool_effect(
                turn, _fail if call == "tool" else _BadStr, name="lookup", arguments={},
                model_exchange_id="e", call_index=0, tool_call_id="t", check_stopped=lambda: None,
            )
    assert turn.observed == observed
    assert caught.value is turn.stop
    primary = caught.value.__context__
    assert str(primary) == ("cannot represent" if call == "representation" else "boom")
    assert not any(
        note.startswith("Conversation observation failed")
        for note in getattr(primary, "__notes__", ())
    )


@pytest.mark.parametrize("call", ["model", "tool"])
@pytest.mark.parametrize("initiating", [True, False])
def test_a_stop_failing_a_call_yields_to_an_observed_stop_only_if_derived(call, initiating):
    """A stop `observe` raises replaces a derived stop failing the call, not an initiating one."""
    turn = _StopOnObserve()
    primary = _HostStop(primary=initiating)

    def stop():
        raise primary

    with pytest.raises(_HostStop) as caught:
        if call == "model":
            invoke_model(turn, stop, new_messages=[HumanMessage("q")], check_stopped=lambda: None)
        else:
            invoke_tool_effect(
                turn, stop, name="lookup", arguments={}, model_exchange_id="e", call_index=0,
                tool_call_id="t", check_stopped=lambda: None,
            )
    assert turn.observed == ["interrupted"]
    assert caught.value is (primary if initiating else turn.stop)


@pytest.mark.parametrize("session", [True, False])
def test_a_stop_wins_over_a_malformed_native_carrier(session):
    """A malformed NaN carrier, a host fault, returned while a stop is pending stops the call."""
    stop = Event()

    def check():
        if stop.is_set():
            raise _HostStop(primary=False)

    class Model:
        def with_structured_output(self, schema, *, include_raw=False):
            def invoke(messages):
                stop.set()
                raw = AIMessage(content="", tool_calls=[
                    {"name": schema.__name__, "args": {"value": math.nan}, "id": "f"}])
                return {"raw": raw, "parsed": None, "parsing_error": None, "extra": 1}
            return SimpleNamespace(invoke=invoke)

    def generate(memory_context):
        generate_structured(Model(), [], Type.scalar(ValueKind.NUMBER), llm_config=_NATIVE,
                            memory_context=memory_context, check_stopped=check)

    memory = SessionMemory()
    with pytest.raises(_HostStop):
        if session:
            with harness(memory) as opened:
                with opened.frame.open_turn(request(), check_stopped=check) as turn:
                    generate(MemoryContext(transcript=turn))
        else:
            generate(MemoryContext())
    if session:
        effects = _view(memory).effects
        assert [effect.status for effect in effects] == ["interrupted"]
        assert effects[0].failure.code == "api.model.interrupted"


@pytest.mark.parametrize("session", [True, False])
def test_a_host_callback_fault_escapes_a_pending_stop(monkeypatch, session):
    """An `on_event` sink that sets the stop and raises surfaces its error, as without memory."""
    stop = Event()

    def on_event(event):
        if isinstance(event, ModelCallStarted):
            stop.set()
            raise ValueError("sink boom")

    _setup(monkeypatch, [AIMessage(content="fine")])
    flow, memory = _flow(session)
    with pytest.raises(ValueError, match="sink boom"):
        _run_flow_controlled(
            flow, {"question": "q"}, memory=memory, stop_event=stop,
            on_event=on_event, num_workers=0,
        )
    if session:
        view = _view(memory["session"])
        assert [effect.status for effect in view.effects] == ["failed"]
        assert view.effects[0].failure.code == "api.model.failed"
