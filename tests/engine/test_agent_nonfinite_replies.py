"""A NaN or infinity from the model fails or retries with a located error."""

import math
from threading import Event
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, ValidationError
import pytest

import yian.llm_clients as llm_clients
import yian.tools as tools
from yian.compose import load_flow
from yian.events import ModelCallFinished, ToolCallStarted
from yian.memory import MemoryContext
from yian.memory.session import SessionMemory
from yian.nodes.agent.structured import _structured_validation_path, generate_structured
from yian.nodes.base import _CapabilityFault, _HostStop
from yian.run import _run_flow_controlled, run_flow
from yian.typesys.types import read_typedefs, type_for
from yian.typesys.values import Type, TypeCheckError, ValueKind
from tests.engine._session_fakes import harness, request
from tests.engine.test_agent_session import _FLOW, _Model, _view
from tests.engine.test_agent_session_tools import _TOOL_FLOW, _Chat, _Tool
from tests.engine.test_agent_steering import Inbox, _flow as _steer_flow, _recorder


_SESSION_LINE = "    long_term_memory: session\n"


class _FloatArguments(BaseModel):
    x: float


class _FloatTool(_Tool):
    args_schema = tool_call_schema = _FloatArguments

    def __init__(self):
        super().__init__(lambda arguments: f"got {arguments['x']}")


def _flow(session):
    text = _TOOL_FLOW if session else _TOOL_FLOW.replace(_SESSION_LINE, "")
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
    _setup(monkeypatch, [_call(math.nan), AIMessage(content="second")])
    flow, memory = _flow(True)
    assert run_flow(flow, {"question": "q"}, memory=memory).status == "failed"
    view = _view(memory["session"])
    assert [effect.status for effect in view.effects] == ["failed"]
    assert view.effects[0].failure.code == "api.model.failed"
    # Only the request's input items are stored; the bad reply is not.
    assert [item.payload["message"]["type"] for item in view.items] == ["system", "human"]
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
    plain = load_flow(_FLOW.replace(_SESSION_LINE, ""))
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


# A NaN or infinity in a structured reply is a schema failure, retried via `retries:`.

_NATIVE = {"provider": "openai", "model": "gpt-4o"}
_FALLBACK = {"provider": "vllm", "model": "no-structured-sentinel"}
_SCORE = Type(
    kind=ValueKind.OBJECT,
    fields={"score": Type.scalar(ValueKind.NUMBER)},
    required=frozenset({"score"}),
)


class _Carrier:
    """A native model that honours `include_raw`, putting each value in the raw tool call.

    Each value is sent as `{key: value}`; `metadata` is every raw reply's `response_metadata`.
    """

    def __init__(self, values, *, key="value", metadata=None):
        self.values = iter(values)
        self.key = key
        self.metadata = metadata or {}
        self.calls = []

    def with_structured_output(self, schema, *, include_raw=False):
        outer = self

        class Bound:
            def invoke(self, messages):
                outer.calls.append(list(messages))
                value = next(outer.values)
                args = {outer.key: value}
                if not include_raw:
                    return schema.model_validate(args)
                raw = AIMessage(
                    content="",
                    response_metadata=outer.metadata,
                    tool_calls=[{"name": schema.__name__, "args": args, "id": "format-id"}],
                )
                try:
                    parsed = schema.model_validate(args)
                except Exception as error:
                    return {"raw": raw, "parsed": None, "parsing_error": error}
                return {"raw": raw, "parsed": parsed, "parsing_error": None}

        return Bound()


def _texts(*replies):
    return _Model([AIMessage(content=reply) for reply in replies])


def _generate(model, typ, *, config=_FALLBACK, max_retries=2):
    return generate_structured(model, [HumanMessage(content="q")], typ,
                               max_retries=max_retries, llm_config=config)


def _correction(model, call=1):
    return model.calls[call][-1].content


def _structured_flow(output, *, native, session, retries=None):
    config = "{provider: openai, model: gpt-4o}" if native else "{provider: vllm, model: no-structured-sentinel}"
    text = (_FLOW.replace("{provider: openai, model: test-model}", config)
            .replace("    output: str", f"    output: {output}"))
    if not session:
        text = text.replace(_SESSION_LINE, "")
    if retries is not None:
        text = text.replace("    output:", f"    retries: {retries}\n    output:", 1)
    return load_flow(text), ({"session": SessionMemory()} if session else None)


@pytest.mark.parametrize("bad", ["NaN", "Infinity", "-Infinity", "1e999"])
def test_a_nonfinite_fallback_field_is_retried(bad):
    model = _texts(f'{{"score": {bad}}}', '{"score": 1.5}')
    assert _generate(model, _SCORE) == {"score": 1.5}
    assert len(model.calls) == 2
    assert "score" in _correction(model) and "finite" in _correction(model)


def test_an_exhausted_nonfinite_fallback_field_is_located_at_the_field():
    with pytest.raises(ValidationError) as caught:
        _generate(_texts('{"score": NaN}'), _SCORE, max_retries=0)
    assert _structured_validation_path(caught.value, _SCORE) == ("score",)


def test_a_bounded_float_rejects_nan_as_not_finite():
    with pytest.raises(ValidationError) as caught:
        _generate(_texts('{"value": NaN}'), type_for("float(0..10)", {}), max_retries=0)
    assert caught.value.errors()[0]["type"] == "finite_number"


@pytest.mark.parametrize("typ", [type_for("list(float)", {}), Type(kind=ValueKind.LIST_NUMBER)])
def test_a_nonfinite_list_item_is_retried(typ):
    model = _texts('{"items": [1.0, NaN]}', '{"items": [1.0, 2.0]}')
    assert _generate(model, typ) == [1.0, 2.0]
    assert "items.1" in _correction(model) and "finite" in _correction(model)


def test_a_nonfinite_union_member_field_is_retried():
    registry = read_typedefs({"A": {"kind": "Literal(a)", "x": "float"}, "B": {"kind": "Literal(b)"}})
    model = _texts('{"value": {"kind": "a", "x": NaN}}', '{"value": {"kind": "a", "x": 1.0}}')
    assert _generate(model, type_for("Union(A, B)", registry)) == {"kind": "a", "x": 1.0}
    assert len(model.calls) == 2 and "finite" in _correction(model)


def test_a_nonfinite_native_value_without_memory_is_retried():
    model = _Carrier([math.nan, 2.0])
    assert _generate(model, Type.scalar(ValueKind.NUMBER), config=_NATIVE) == 2.0
    assert len(model.calls) == 2


def test_a_nonfinite_freeform_value_is_retried_and_located():
    freeform = Type(kind=ValueKind.OBJECT)
    model = _texts('{"value": {"k": NaN}}', '{"value": {"k": 1}}')
    assert _generate(model, freeform) == {"k": 1}
    assert "structured output at k: non-finite number" in _correction(model)
    with pytest.raises(TypeCheckError) as caught:
        _generate(_texts('{"value": {"k": NaN}}'), freeform, max_retries=0)
    assert caught.value.path == ("k",)


def test_a_nonfinite_native_session_reply_is_retried_with_the_full_error(monkeypatch):
    model = _Carrier([math.nan, 2])
    monkeypatch.setattr(llm_clients, "model_from_config", lambda _: model)
    flow, memory = _structured_flow("float", native=True, session=True)
    assert run_flow(flow, {"question": "q"}, memory=memory).output == 2.0
    assert [effect.status for effect in _view(memory["session"]).effects] == ["failed", "completed"]
    assert "finite" in _correction(model)
    assert "provider error (" not in _correction(model)


@pytest.mark.parametrize("session", [True, False])
def test_an_exhausted_nonfinite_native_field_is_located_at_the_field(monkeypatch, session):
    monkeypatch.setattr(llm_clients, "model_from_config", lambda _: _Carrier([math.nan], key="score"))
    flow, memory = _structured_flow("{score: float}", native=True, session=session, retries=0)
    result = run_flow(flow, {"question": "q"}, memory=memory)
    assert result.status == "failed" and result.failure.code == "node.execute"
    assert result.failure.trace[-1].key.endswith("/s:path/s:score")


def test_nonfinite_session_metadata_fails_after_every_retry(monkeypatch):
    model = _Carrier([1.0, 2.0, 3.0, 4.0],
                     metadata={"logprobs": {"content": [{"logprob": -math.inf}]}})
    monkeypatch.setattr(llm_clients, "model_from_config", lambda _: model)
    flow, memory = _structured_flow("float", native=True, session=True)
    result = run_flow(flow, {"question": "q"}, memory=memory)
    assert result.status == "failed" and result.failure.code == "node.execute"
    assert "model reply at response_metadata.logprobs.content[0].logprob" in result.failure.message
    assert len(model.calls) == 3
    assert "provider error (" not in _correction(model)


def test_a_nonfinite_native_session_freeform_value_is_retried_with_the_full_error():
    model = _Carrier([{"k": math.nan}, {"k": 1}])
    memory = SessionMemory()
    with harness(memory) as opened:
        with opened.frame.open_turn(request(), check_stopped=lambda: None) as turn:
            value = generate_structured(model, [HumanMessage(content="q")], Type(kind=ValueKind.OBJECT),
                                        llm_config=_NATIVE, memory_context=MemoryContext(transcript=turn))
    assert value == {"k": 1}
    assert "model reply at tool_calls[0].args.value.k: non-finite number" in _correction(model)
    assert [effect.status for effect in _view(memory).effects] == ["failed", "completed"]


def test_a_malformed_native_carrier_holding_nan_stays_a_capability_fault():
    calls = []

    class Model:
        def with_structured_output(self, schema, *, include_raw):
            def invoke(messages):
                calls.append(messages)
                raw = AIMessage(content="", tool_calls=[
                    {"name": schema.__name__, "args": {"value": math.nan}, "id": "format-id"}])
                return {"raw": raw, "parsed": None, "parsing_error": None, "extra": 1}
            return SimpleNamespace(invoke=invoke)

    memory = SessionMemory()
    with pytest.raises(_CapabilityFault):
        with harness(memory) as opened:
            with opened.frame.open_turn(request(), check_stopped=lambda: None) as turn:
                generate_structured(Model(), [], Type.scalar(ValueKind.NUMBER), llm_config=_NATIVE,
                                    memory_context=MemoryContext(transcript=turn))
    assert len(calls) == 1
    assert [effect.status for effect in _view(memory).effects] == ["failed"]


def test_a_nonfinite_fallback_session_reply_is_stored_and_retried(monkeypatch):
    model = _texts('{"value": NaN}', '{"value": 2}')
    monkeypatch.setattr(llm_clients, "model_from_config", lambda _: model)
    flow, memory = _structured_flow("float", native=False, session=True)
    assert run_flow(flow, {"question": "q"}, memory=memory).output == 2.0
    assert [effect.status for effect in _view(memory["session"]).effects] == ["completed", "completed"]


def test_a_nonfinite_record_reply_retries_end_to_end_and_fails_at_the_field(monkeypatch):
    model = _texts('{"score": NaN}', '{"score": 0.5}')
    monkeypatch.setattr(llm_clients, "model_from_config", lambda _: model)
    flow, _ = _structured_flow("{score: float}", native=False, session=False)
    assert run_flow(flow, {"question": "q"}).output == {"score": 0.5}

    monkeypatch.setattr(llm_clients, "model_from_config", lambda _: _texts('{"score": NaN}'))
    flow, _ = _structured_flow("{score: float}", native=False, session=False, retries=0)
    result = run_flow(flow, {"question": "q"})
    assert result.status == "failed" and result.failure.code == "node.execute"
    assert result.failure.trace[-1].key.endswith("/s:path/s:score")
