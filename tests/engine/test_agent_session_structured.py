"""Structured session calls retain surfaced raw replies independently of parsing."""

from yian.memory import MemoryContext

from copy import deepcopy
from asyncio import CancelledError as AsyncCancelledError
from concurrent.futures import CancelledError as FutureCancelledError
import json
from threading import Event
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
import pytest

import yian.llm_clients as llm_clients
from yian.compose import load_flow
from yian.memory.session import SessionMemory
from yian.run import run_flow
from tests.engine.test_agent_session import _FLOW, _Model, _view
from tests.engine._session_fakes import harness, request


def _config(native):
    return {"provider": "openai", "model": "gpt-4o"} if native else {"provider": "vllm", "model": "no-structured-sentinel"}


class _Native:
    def __init__(self, values):
        self.values = iter(values)
        self.calls = []
        self.raw = []
        self.errors = []

    def with_structured_output(self, schema, *, include_raw=False):
        assert include_raw is True
        outer = self
        class Bound:
            def invoke(self, messages):
                outer.calls.append(deepcopy(messages))
                value = next(outer.values)
                raw = AIMessage(content="", additional_kwargs={"reasoning_content": "native reasoning"},
                                tool_calls=[{"name": schema.__name__, "args": {"value": value}, "id": "format-id"}])
                outer.raw.append(raw)
                try:
                    parsed = schema.model_validate({"value": value})
                except Exception as error:
                    outer.errors.append(error)
                    return {"raw": raw, "parsed": None, "parsing_error": error}
                return {"raw": raw, "parsed": parsed, "parsing_error": None}
        return Bound()


def _typed_flow(native=True):
    config = "{provider: openai, model: gpt-4o}" if native else "{provider: vllm, model: no-structured-sentinel}"
    return load_flow(_FLOW.replace("{provider: openai, model: test-model}", config)
                     .replace("    output: str", "    output: int"))


@pytest.mark.parametrize("native", [False, True])
def test_structured_runs_restore_actual_history_without_schema_tool_execution(monkeypatch, native):
    model = _Native([1, 2]) if native else _Model([AIMessage(content='{"value":1}'), AIMessage(content='{"value":2}')])
    monkeypatch.setattr(llm_clients, "model_from_config", lambda _: model)
    memory, flow = SessionMemory(), _typed_flow(native)
    assert run_flow(flow, {"question": "first"}, memory={"session": memory}).output == 1
    assert run_flow(flow, {"question": "second"}, memory={"session": memory}).output == 2
    assert any(message.content == "first" for message in model.calls[-1])
    if native:
        assert any(isinstance(message, HumanMessage) and "context only" in message.content for message in model.calls[-1])
        assert not any(getattr(message, "tool_calls", None) or isinstance(message, ToolMessage) for message in model.calls[-1])
    view = _view(memory)
    assert len(view.effects) == 2 and all(effect.status == "completed" for effect in view.effects)
    assert all(effect.request["purpose"] == ("structured_native" if native else "structured_fallback") for effect in view.effects)
    assert len(view.items) == (6 if native else 8)
    if native:
        assert view.items[2].payload["message"]["data"]["tool_calls"][0]["id"] == "format-id"


@pytest.mark.parametrize("native", [False, True])
def test_structured_retries_store_raw_replies_and_only_new_correction_deltas(monkeypatch, native):
    model = _Native(["bad", 2]) if native else _Model([AIMessage(content='{"value":"bad"}'), AIMessage(content='{"value":2}')])
    monkeypatch.setattr(llm_clients, "model_from_config", lambda _: model)
    memory = SessionMemory()
    result = run_flow(_typed_flow(native), {"question": "first"}, memory={"session": memory})
    assert result.output == 2 and len(model.calls) == 2
    assert not any(isinstance(message, AIMessage) for message in model.calls[1])
    assert "previous output was invalid" in model.calls[1][-1].content
    assert "valid integer" in model.calls[1][-1].content
    if native:
        assert str(model.errors[0]) in model.calls[1][-1].content
    view = _view(memory)
    assert len(view.effects) == 2 and all(effect.status == "completed" for effect in view.effects)
    assert len(view.items) == (5 if native else 6)
    assert [item.payload["message"]["type"] for item in view.items].count("ai") == 2
    assert sum("previous output was invalid" in str(item.payload["message"]["data"]["content"]) for item in view.items) == 1
    assert next(item.payload["message"]["data"]["content"] for item in view.items
                if "previous output was invalid" in str(item.payload["message"]["data"]["content"])) == model.calls[1][-1].content
    json.dumps([item.payload for item in view.items], allow_nan=False)


@pytest.mark.parametrize("native", [False, True])
def test_structured_exhaustion_retains_schema_error_identity_and_origin(native):
    from pydantic import ValidationError
    from yian.nodes.agent.structured import _structured_validation_path, generate_structured
    from yian.typesys.values import Type, ValueKind
    typ = Type.scalar(ValueKind.INTEGER)
    model = _Native(["bad"]) if native else _Model([AIMessage(content='{"value":"bad"}')])
    memory = SessionMemory()
    with pytest.raises(ValidationError) as caught:
        with harness(memory) as opened:
            with opened.frame.open_turn(request(), check_stopped=lambda: None) as turn:
                generate_structured(model, [HumanMessage(content="question")], typ,
                                    max_retries=0, llm_config=_config(native), memory_context=MemoryContext(transcript=turn))
    if native:
        assert caught.value is model.errors[-1]
    assert _structured_validation_path(caught.value, typ) == ()
    view = _view(memory)
    assert view.effects[-1].status == "completed" and view.closed_invocations[-1].status == "failed"
    assert view.items[-1].payload["message"]["type"] == "ai"


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("stage", ["request", "observe", "encode"])
def test_structured_memory_faults_do_not_retry_model(monkeypatch, native, stage):
    from yian.nodes.agent import memory as conversation
    from yian.nodes.agent.structured import generate_structured
    from yian.nodes.base import _CapabilityFault
    from yian.typesys.values import Type, ValueKind
    model = _Native([1, 2]) if native else _Model([AIMessage(content='{"value":1}'), AIMessage(content='{"value":2}')])
    error = ValueError("memory fault")
    def request(*args, **kwargs):
        if stage == "request":
            raise error
        return 1
    def observe(*args, **kwargs):
        if stage == "observe":
            raise error
    real_encode = conversation.encode_message
    def encode(message, exchange):
        if isinstance(message, AIMessage):
            raise error
        return real_encode(message, exchange)
    if stage == "encode":
        monkeypatch.setattr(conversation, "encode_message", encode)
    with pytest.raises(_CapabilityFault) as caught:
        generate_structured(model, [], Type.scalar(ValueKind.INTEGER), llm_config=_config(native),
                            memory_context=MemoryContext(transcript=SimpleNamespace(request=request, observe=observe)))
    assert caught.value.original is error
    assert len(model.calls) == (0 if stage == "request" else 1)


@pytest.mark.parametrize("shape", ["not_dict", "missing_raw", "invalid_raw", "non_json_raw", "missing_parsed", "invalid_error", "invalid_parsed"])
def test_native_malformed_carriers_fail_without_retry_and_keep_available_raw(shape):
    from yian.nodes.agent.structured import generate_structured
    from yian.nodes.base import _CapabilityFault
    from yian.typesys.values import Type, ValueKind
    calls = []
    class Model:
        def with_structured_output(self, schema, *, include_raw):
            def invoke(messages):
                calls.append(messages)
                response = {"raw": AIMessage(content="raw"), "parsed": schema.model_validate({"value": 1}), "parsing_error": None}
                if shape == "not_dict":
                    return []
                if shape == "missing_raw":
                    del response["raw"]
                elif shape == "invalid_raw":
                    response["raw"] = "not a message"
                elif shape == "non_json_raw":
                    response["raw"] = AIMessage(content="raw", response_metadata={"bad": object()})
                elif shape == "missing_parsed":
                    del response["parsed"]
                elif shape == "invalid_error":
                    response["parsing_error"] = "not an exception"
                elif shape == "invalid_parsed":
                    response["parsed"] = object()
                return response
            return SimpleNamespace(invoke=invoke)
    memory = SessionMemory()
    with pytest.raises(_CapabilityFault):
        with harness(memory) as opened:
            with opened.frame.open_turn(request(), check_stopped=lambda: None) as turn:
                generate_structured(Model(), [], Type.scalar(ValueKind.INTEGER),
                                    llm_config=_config(True), memory_context=MemoryContext(transcript=turn))
    assert len(calls) == 1
    available = shape in {"missing_parsed", "invalid_error", "invalid_parsed"}
    assert _view(memory).effects[-1].status == ("completed" if available else "pending")
    assert len(_view(memory).items) == int(available)


class _Carriers:
    """A native model whose bound runnable returns each queued carrier, built from the schema."""

    def __init__(self, carriers):
        self.carriers = iter(carriers)
        self.calls = []

    def with_structured_output(self, schema, *, include_raw):
        assert include_raw is True
        def invoke(messages):
            self.calls.append(deepcopy(messages))
            return next(self.carriers)(schema)
        return SimpleNamespace(invoke=invoke)


def _unparsed_carrier(raw=None):
    """Build a carrier for a reply with no call to the schema's tool."""
    return lambda schema: {"raw": raw or AIMessage(content="42"), "parsed": None, "parsing_error": None}


def _valid_carrier(schema):
    raw = AIMessage(content="", tool_calls=[{"name": schema.__name__, "args": {"value": 7}, "id": "format-id"}])
    return {"raw": raw, "parsed": schema.model_validate({"value": 7}), "parsing_error": None}


def _run_carriers(model, memory, max_retries=2):
    from yian.nodes.agent.structured import generate_structured
    from yian.typesys.values import Type, ValueKind
    with harness(memory) as opened:
        with opened.frame.open_turn(request(), check_stopped=lambda: None) as turn:
            return generate_structured(model, [HumanMessage(content="question")], Type.scalar(ValueKind.INTEGER),
                                       max_retries=max_retries, llm_config=_config(True),
                                       memory_context=MemoryContext(transcript=turn))


def test_native_carrier_without_a_parsed_value_is_corrected_and_retried():
    from yian.llm_clients.base_client import missing_tool_call
    from yian.nodes.agent.structured import type_to_schema
    from yian.typesys.values import Type, ValueKind
    model, memory = _Carriers([_unparsed_carrier(), _valid_carrier]), SessionMemory()
    assert _run_carriers(model, memory) == 7 and len(model.calls) == 2
    view = _view(memory)
    assert [item.payload["message"]["type"] for item in view.items][-3:] == ["ai", "human", "ai"]
    assert all(effect.status == "completed" for effect in view.effects)
    assert not any(isinstance(message, AIMessage) for message in model.calls[1])
    correction = model.calls[1][-1]
    assert isinstance(correction, HumanMessage)
    assert str(missing_tool_call(type_to_schema(Type.scalar(ValueKind.INTEGER)).__name__)) in correction.content
    assert "provider error (" not in correction.content


def test_native_carriers_without_a_parsed_value_fail_once_retries_are_spent():
    from langchain_core.exceptions import OutputParserException
    model = _Carriers([_unparsed_carrier(), _unparsed_carrier()])
    with pytest.raises(OutputParserException):
        _run_carriers(model, SessionMemory(), max_retries=1)
    assert len(model.calls) == 2


def test_native_carrier_without_a_parsed_value_and_a_nonfinite_raw_is_retried_unstored():
    nonfinite = AIMessage(content="42", response_metadata={"x": float("nan")})
    model, memory = _Carriers([_unparsed_carrier(nonfinite), _valid_carrier]), SessionMemory()
    assert _run_carriers(model, memory) == 7 and len(model.calls) == 2
    view = _view(memory)
    assert [item.payload["message"]["type"] for item in view.items].count("ai") == 1
    assert not any(item.payload["message"]["data"]["content"] == "42" for item in view.items)


@pytest.mark.parametrize("native,stage", [(native, stage) for native in (False, True)
                                          for stage in ("before", "after_reply", "after_bad_reply", "after_parse")]
                         + [(True, "after_bind")])
def test_structured_stop_boundaries_preserve_surfaced_raw(monkeypatch, native, stage):
    from yian.nodes.agent import structured
    from yian.nodes.base import _HostStop
    from yian.typesys.values import Type, ValueKind
    stopped, primary, memory = Event(), _HostStop("stop structured"), SessionMemory()
    def check():
        if stopped.is_set():
            raise primary
    value = "bad" if stage == "after_bad_reply" else 1
    model = _Native([value]) if native else _Model([AIMessage(content=json.dumps({"value": value}))])
    if native:
        bind = model.with_structured_output
        def wrapped_bind(*args, **kwargs):
            bound = bind(*args, **kwargs)
            invoke = bound.invoke
            def wrapped_invoke(messages):
                response = invoke(messages)
                if stage in {"after_reply", "after_bad_reply"}:
                    stopped.set()
                return response
            bound.invoke = wrapped_invoke
            if stage == "after_bind":
                stopped.set()
            return bound
        monkeypatch.setattr(model, "with_structured_output", wrapped_bind)
    else:
        invoke = model.invoke
        def wrapped_invoke(messages):
            response = invoke(messages)
            if stage in {"after_reply", "after_bad_reply"}:
                stopped.set()
            return response
        monkeypatch.setattr(model, "invoke", wrapped_invoke)
    unwrap = structured._unwrap
    def wrapped_unwrap(*args):
        value = unwrap(*args)
        if stage == "after_parse":
            stopped.set()
        return value
    monkeypatch.setattr(structured, "_unwrap", wrapped_unwrap)
    with pytest.raises(_HostStop) as caught:
        with harness(memory) as opened:
            with opened.frame.open_turn(request(), check_stopped=check) as turn:
                if stage == "before":
                    stopped.set()
                structured.generate_structured(model, [HumanMessage(content="question")], Type.scalar(ValueKind.INTEGER),
                                               llm_config=_config(native), check_stopped=check, memory_context=MemoryContext(transcript=turn))
    assert caught.value is primary
    surfaced = stage not in {"before", "after_bind"}
    assert len(model.calls) == int(surfaced)
    view = _view(memory)
    assert view.closed_invocations[-1].status == "interrupted"
    if surfaced:
        assert view.effects[-1].status == "completed" and view.items[-1].payload["message"]["type"] == "ai"
    else:
        assert view.effects == () and view.items == ()


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("error_type", [ValueError, FutureCancelledError])
def test_provider_error_and_failed_publication_bypass_retry_with_original_public_identity(monkeypatch, native, error_type):
    from yian.memory.session import _Turn
    primary, secondary = error_type("provider failed"), RuntimeError("cannot publish failure")
    model = _Model([primary, AIMessage(content='{"value":2}')])
    if native:
        monkeypatch.setattr(model, "with_structured_output", lambda schema, include_raw: model, raising=False)
    monkeypatch.setattr(llm_clients, "model_from_config", lambda _: model)
    original_observe = _Turn.observe
    def fail_publication(turn, ordinal, status, **kwargs):
        if status in {"failed", "interrupted"}:
            raise secondary
        return original_observe(turn, ordinal, status, **kwargs)
    monkeypatch.setattr(_Turn, "observe", fail_publication)
    memory = SessionMemory()
    with pytest.raises(error_type) as caught:
        run_flow(_typed_flow(native), {"question": "first"}, memory={"session": memory})
    assert caught.value is primary and len(model.calls) == 1
    if hasattr(primary, "add_note"):
        assert any("cannot publish failure" in note for note in primary.__notes__)
    view = _view(memory)
    assert len(view.effects) == 1 and view.effects[0].status == "pending"
    assert view.closed_invocations[-1].status == "failed"


@pytest.mark.parametrize("native", [False, True])
def test_tool_then_structured_purpose_distinguishes_schema_calls_from_real_grants(monkeypatch, native):
    import yian.tools as tools
    from tests.engine.test_agent_session_tools import _Chat, _Tool, _call
    tool, memory = _Tool(), SessionMemory()
    native_model = _Native([9, 10])
    replies = [_call(name="ScalarWrapper", call_id="format-id"), AIMessage(content="ready")]
    if not native:
        replies.append(AIMessage(content='{"value":9}'))
    replies.append(AIMessage(content="ready again"))
    if not native:
        replies.append(AIMessage(content='{"value":10}'))
    chat = _Chat(replies)
    if native:
        monkeypatch.setattr(chat, "with_structured_output", native_model.with_structured_output, raising=False)
    monkeypatch.setattr(llm_clients, "model_from_config", lambda _: chat)
    monkeypatch.setitem(tools.TOOL_REGISTRY, "ScalarWrapper", tool)
    flow = _typed_flow(native)
    flow.compiled.nodes["agent"].tools = ["ScalarWrapper"]
    assert run_flow(flow, {"question": "first"}, memory={"session": memory}).output == 9
    assert run_flow(flow, {"question": "second"}, memory={"session": memory}).output == 10
    assert tool.calls == [{"value": 1}]
    current = chat.calls[2 if native else 3]
    assert len([message for message in current if isinstance(message, ToolMessage)]) == 1
    if native:
        assert len([message for message in current if getattr(message, "tool_calls", None)]) == 1
        assert any(isinstance(message, HumanMessage) and "structured" in message.content for message in current)
    view = _view(memory)
    assert len(view.items) == (10 if native else 12)
    assert [effect.kind for effect in view.effects] == ["model", "tool", "model", "model", "model", "model"]


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("error_type", [AsyncCancelledError, FutureCancelledError])
def test_provider_cancellation_with_failed_publication_is_not_a_structured_retry(native, error_type):
    from yian.nodes.agent.structured import generate_structured
    from yian.nodes.base import _CapabilityFault
    from yian.typesys.values import Type, ValueKind
    primary = error_type("cancelled provider")
    model = _Model([primary, AIMessage(content='{"value":1}')])
    if native:
        model.with_structured_output = lambda schema, include_raw: model
    statuses = []
    def observe(ordinal, status, **kwargs):
        statuses.append(status)
        raise RuntimeError("cannot save interruption")
    turn = SimpleNamespace(request=lambda *args, **kwargs: 1, observe=observe)
    with pytest.raises(_CapabilityFault if error_type is FutureCancelledError else error_type) as caught:
        generate_structured(model, [], Type.scalar(ValueKind.INTEGER), llm_config=_config(native), memory_context=MemoryContext(transcript=turn))
    original = caught.value.original if error_type is FutureCancelledError else caught.value
    assert original is primary and len(model.calls) == 1 and statuses == ["interrupted"]


class _RetryProvider:
    """Fail binding or invocation before returning a real structured reply."""

    def __init__(self, error, *, stage="invoke", failures=1):
        self.error, self.stage, self.failures = error, stage, failures
        self.calls, self.bindings = [], []

    def _fail(self, stage):
        if stage == self.stage and self.failures:
            self.failures -= 1
            raise self.error

    def invoke(self, messages, *, schema=None, include_raw=False):
        self.calls.append(deepcopy(messages))
        self._fail("invoke")
        raw = AIMessage(content='{"value":2}')
        if schema is None:
            return raw
        parsed = schema.model_validate({"value": 2})
        return {"raw": raw, "parsed": parsed, "parsing_error": None} if include_raw else parsed

    def with_structured_output(self, schema, **kwargs):
        self.bindings.append(kwargs)
        self._fail("bind")
        return SimpleNamespace(invoke=lambda messages: self.invoke(messages, schema=schema, **kwargs))


@pytest.mark.parametrize("native,stage", [(False, "invoke"), (True, "invoke"), (True, "bind")])
@pytest.mark.parametrize("error_type", [ValueError, RuntimeError])
def test_session_provider_corrections_do_not_send_or_store_exception_secrets(monkeypatch, native, stage, error_type):
    model = _RetryProvider(error_type("api_key=secret-marker"), stage=stage)
    monkeypatch.setattr(llm_clients, "model_from_config", lambda _: model)
    memory = SessionMemory()
    assert run_flow(_typed_flow(native), {"question": "question"}, memory={"session": memory}).output == 2
    assert len(model.calls) == (1 if stage == "bind" else 2)
    correction = model.calls[-1][-1].content
    assert "secret-marker" not in correction and error_type.__name__ in correction
    view = _view(memory)
    stored = [item.payload["message"]["data"]["content"] for item in view.items
              if "previous output was invalid" in str(item.payload["message"]["data"]["content"])]
    assert stored == [correction]
    assert [effect.status for effect in view.effects] == (["completed"] if stage == "bind" else ["failed", "completed"])
    with memory.open_run() as opened:
        assert "secret-marker" not in json.dumps(opened.snapshot)


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("error_type", [ValueError, RuntimeError, FutureCancelledError])
def test_no_session_provider_correction_text_and_retry_behavior_are_unchanged(native, error_type):
    from yian.nodes.agent.structured import generate_structured
    from yian.typesys.values import Type, ValueKind
    primary = error_type("original provider diagnostic")
    model = _RetryProvider(primary)
    assert generate_structured(model, [], Type.scalar(ValueKind.INTEGER), llm_config=_config(native)) == 2
    assert len(model.calls) == 2
    assert "Your previous output was invalid: original provider diagnostic." in model.calls[-1][-1].content
    if native:
        assert model.bindings == [{}, {}]


@pytest.mark.parametrize("native", [False, True])
def test_session_future_cancellation_with_published_interruption_does_not_retry(native):
    from yian.nodes.agent.structured import generate_structured
    from yian.typesys.values import Type, ValueKind
    primary, memory = FutureCancelledError("provider cancelled"), SessionMemory()
    model = _RetryProvider(primary)
    with pytest.raises(FutureCancelledError) as caught:
        with harness(memory) as opened:
            with opened.frame.open_turn(request(), check_stopped=lambda: None) as turn:
                generate_structured(model, [HumanMessage(content="question")], Type.scalar(ValueKind.INTEGER),
                                    llm_config=_config(native), memory_context=MemoryContext(transcript=turn))
    assert caught.value is primary and len(model.calls) == 1
    view = _view(memory)
    assert [effect.status for effect in view.effects] == ["interrupted"]
    assert all(entry.status != "completed" for entry in view.closed_invocations)
    assert not any(item.payload["message"]["type"] == "ai" for item in view.items)


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("error_type", [ValueError, RuntimeError])
def test_sanitized_provider_retry_exhaustion_preserves_original_error(native, error_type):
    from yian.nodes.agent.structured import generate_structured
    from yian.typesys.values import Type, ValueKind
    primary, memory = error_type("api_key=secret-marker"), SessionMemory()
    model = _RetryProvider(primary, failures=2)
    with pytest.raises(error_type) as caught:
        with harness(memory) as opened:
            with opened.frame.open_turn(request(), check_stopped=lambda: None) as turn:
                generate_structured(model, [], Type.scalar(ValueKind.INTEGER), max_retries=1,
                                    llm_config=_config(native), memory_context=MemoryContext(transcript=turn))
    assert caught.value is primary and len(model.calls) == 2
    assert "secret-marker" not in model.calls[-1][-1].content
    with memory.open_run() as opened:
        assert "secret-marker" not in json.dumps(opened.snapshot)
