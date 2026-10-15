"""An int over 4300 digits from a model reply, tool call, or feedback fails with a located error."""

import json
import math
import re
import sys

from langchain_core.messages import AIMessage
import pytest

import yian.llm_clients as llm_clients
from yian.compose import load_flow
from yian.memory.session import SessionMemory
from yian.nodes.agent import AgentNode
from yian.run import run_flow
from yian.typesys.values import Type, TypeCheckError, ValueKind
from tests.engine.test_agent_backward import _BackwardChat, _context
from tests.engine.test_agent_nonfinite_replies import (
    _Carrier,
    _correction,
    _flow as _tool_flow,
    _generate,
    _structured_flow,
    _texts,
)
from tests.engine.test_agent_session import _FLOW, _Model
from tests.engine.test_agent_session_tools import _Chat, _Tool
from tests.engine.test_value_nesting_boundaries import _code_flow
import yian.tools as tools


BIG = 10**4999 + 12345
D = "1" * 5000
LIMIT_MSG = "integer with 5000 digits is larger than yian allows (4300 digits)"
PYTHON_MSG = (
    "Exceeds the limit (4300 digits) for integer string conversion: value has 5000 digits; "
    "use sys.set_int_max_str_digits() to increase the limit"
)
_SESSION_LINE = "    long_term_memory: session\n"
_RECORD = Type(
    kind=ValueKind.OBJECT,
    fields={"n": Type.scalar(ValueKind.INTEGER)},
    required=frozenset({"n"}),
)


def _assert_clean(message):
    assert "set_int_max_str_digits" not in message
    assert re.search(r"[0-9]{600}", message) is None


# Structured replies


# With memory the raw reply is checked before it is stored, as for a NaN.
@pytest.mark.parametrize("session, where", [
    (False, "structured output: "),
    (True, "model reply at tool_calls[0].args.value: "),
])
def test_a_long_int_in_a_provider_parsed_reply_is_retried_then_fails(monkeypatch, session, where):
    model = _Carrier([BIG, BIG])
    monkeypatch.setattr(llm_clients, "model_from_config", lambda _: model)
    flow, memory = _structured_flow("int", native=True, session=session, retries=1)
    result = run_flow(flow, {"question": "q"}, memory=memory)
    assert result.status == "failed" and result.failure.code == "node.execute"
    assert len(model.calls) == 2
    assert where + LIMIT_MSG in result.failure.message
    _assert_clean(result.failure.message)
    assert LIMIT_MSG in _correction(model)
    _assert_clean(_correction(model))


def test_a_long_int_in_a_provider_parsed_reply_retries_to_a_good_value(monkeypatch):
    model = _Carrier([BIG, 7])
    monkeypatch.setattr(llm_clients, "model_from_config", lambda _: model)
    flow, memory = _structured_flow("int", native=True, session=False, retries=1)
    assert run_flow(flow, {"question": "q"}, memory=memory).output == 7


@pytest.mark.parametrize(
    "typ, reply, where",
    [
        (_RECORD, f'{{"n": {D}}}', "structured output at n: "),
        (Type.scalar(ValueKind.INTEGER), f'{{"value": {D}}}', "structured output: "),
        (Type(kind=ValueKind.LIST_ANY), f'{{"items": [1, -{D}]}}', "structured output at [1]: "),
    ],
    ids=["record", "int", "list"],
)
def test_a_long_int_in_a_prompted_reply_is_retried_then_fails_located(typ, reply, where):
    model = _texts(reply, reply)
    with pytest.raises(TypeCheckError) as caught:
        _generate(model, typ, max_retries=1)
    assert str(caught.value) == where + LIMIT_MSG
    assert len(model.calls) == 2
    assert where + LIMIT_MSG in _correction(model)
    _assert_clean(_correction(model))


def test_a_long_int_in_a_prompted_reply_retries_to_a_good_value():
    model = _texts(f'{{"n": {D}}}', '{"n": 5}')
    assert _generate(model, _RECORD) == {"n": 5}
    assert "structured output at n: " + LIMIT_MSG in _correction(model)


@pytest.mark.parametrize("python_limit", [None, 0, 6000])
def test_a_prompted_long_int_fails_the_same_whatever_pythons_limit(python_limit):
    get_limit = getattr(sys, "get_int_max_str_digits", None)
    if python_limit is not None and get_limit is None:
        pytest.skip("this Python has no int/str digit limit")
    before = get_limit() if get_limit is not None else None
    try:
        if python_limit is not None:
            sys.set_int_max_str_digits(python_limit)
        with pytest.raises(TypeCheckError) as caught:
            _generate(_texts(f'{{"n": {D}}}'), _RECORD, max_retries=0)
    finally:
        if before is not None:
            sys.set_int_max_str_digits(before)
    assert str(caught.value) == "structured output at n: " + LIMIT_MSG


def test_a_long_int_prompted_record_reply_fails_the_run_at_the_field(monkeypatch):
    model = _texts(f'{{"n": {D}}}')
    monkeypatch.setattr(llm_clients, "model_from_config", lambda _: model)
    flow, _ = _structured_flow("{n: int}", native=False, session=False, retries=0)
    result = run_flow(flow, {"question": "q"})
    assert result.status == "failed" and result.failure.code == "node.execute"
    assert "structured output at n: " + LIMIT_MSG in result.failure.message
    _assert_clean(result.failure.message)


class _DigitLimitCarrier:
    """A native model whose parser fails with Python's digit-limit error.

    With `include_raw` it returns the error in the carrier; otherwise it raises it.
    """

    def __init__(self):
        self.calls = []

    def with_structured_output(self, schema, *, include_raw=False):
        outer = self

        class Bound:
            def invoke(self, messages):
                outer.calls.append(list(messages))
                error = ValueError(PYTHON_MSG)
                if not include_raw:
                    raise error
                raw = AIMessage(content="", tool_calls=[
                    {"name": schema.__name__, "args": {}, "id": "format-id"}])
                return {"raw": raw, "parsed": None, "parsing_error": error}

        return Bound()


@pytest.mark.parametrize("session", [False, True])
def test_a_parser_digit_limit_error_reads_as_yians_in_correction_and_failure(monkeypatch, session):
    model = _DigitLimitCarrier()
    monkeypatch.setattr(llm_clients, "model_from_config", lambda _: model)
    flow, memory = _structured_flow("int", native=True, session=session, retries=1)
    result = run_flow(flow, {"question": "q"}, memory=memory)
    assert result.status == "failed" and result.failure.code == "node.execute"
    assert len(model.calls) == 2
    assert LIMIT_MSG in _correction(model)
    _assert_clean(_correction(model))
    assert LIMIT_MSG in result.failure.message
    _assert_clean(result.failure.message)


# Tool-call arguments


def _setup_tool(monkeypatch, replies):
    tool, chat = _Tool(), _Chat(replies)
    monkeypatch.setitem(tools.TOOL_REGISTRY, "lookup", tool)
    monkeypatch.setattr(llm_clients, "model_from_config", lambda _: chat)
    return tool, chat


def _invalid_call(args):
    return AIMessage(content="", invalid_tool_calls=[{
        "name": "lookup", "args": args, "error": PYTHON_MSG, "id": "c1", "type": "invalid_tool_call",
    }])


@pytest.mark.parametrize("session", [False, True])
@pytest.mark.parametrize("reply", [
    AIMessage(content="", tool_calls=[{"name": "lookup", "args": {"value": BIG}, "id": "c1"}]),
    _invalid_call(f'{{"value": {D}}}'),
], ids=["valid", "invalid"])
def test_a_long_int_tool_argument_fails_before_the_tool_runs(monkeypatch, session, reply):
    tool, chat = _setup_tool(monkeypatch, [reply])
    flow, memory = _tool_flow(session)
    result = run_flow(flow, {"question": "q"}, memory=memory)
    assert result.status == "failed" and result.failure.code == "node.execute"
    assert result.failure.node_id == "agent"
    assert "tool call 'lookup' arguments at value: " + LIMIT_MSG in result.failure.message
    _assert_clean(result.failure.message)
    assert tool.calls == [] and len(chat.calls) == 1


def test_an_invalid_tool_call_with_unrelated_bad_json_keeps_todays_behaviour(monkeypatch):
    reply = AIMessage(content="", invalid_tool_calls=[{
        "name": "lookup", "args": '{"value": ', "error": "bad json", "id": "c1",
        "type": "invalid_tool_call",
    }])
    tool, _ = _setup_tool(monkeypatch, [reply])
    flow, memory = _tool_flow(False)
    result = run_flow(flow, {"question": "q"}, memory=memory)
    assert result.status == "succeeded" and result.output == ""
    assert tool.calls == []


@pytest.mark.parametrize("python_limit", [None, 0])
@pytest.mark.parametrize("session", [False, True])
def test_a_tool_returning_a_long_int_fails_naming_the_tool(monkeypatch, session, python_limit):
    get = getattr(sys, "get_int_max_str_digits", None)
    if python_limit is not None and get is None:
        pytest.skip("this Python has no int/str digit limit")
    tool, _ = _setup_long_result_tool(monkeypatch)
    flow, memory = _tool_flow(session)
    saved = get() if get is not None else None
    try:
        if python_limit is not None:
            sys.set_int_max_str_digits(python_limit)
        result = run_flow(flow, {"question": "q"}, memory=memory)
    finally:
        if saved is not None:
            sys.set_int_max_str_digits(saved)
    assert result.status == "failed" and result.failure.node_id == "agent"
    assert "tool 'lookup' result: " + LIMIT_MSG in result.failure.message
    _assert_clean(result.failure.message)
    assert len(tool.calls) == 1


def _setup_long_result_tool(monkeypatch):
    """A `lookup` tool returning a 5000-digit int, called once by the model."""
    tool, chat = _Tool(lambda _args: BIG), _Chat([
        AIMessage(content="", tool_calls=[{"name": "lookup", "args": {"value": 1}, "id": "c1"}]),
        AIMessage(content="done"),
    ])
    monkeypatch.setitem(tools.TOOL_REGISTRY, "lookup", tool)
    monkeypatch.setattr(llm_clients, "model_from_config", lambda _: chat)
    return tool, chat

# Failure messages


@pytest.mark.parametrize("text, expected", [
    (PYTHON_MSG, LIMIT_MSG),
    ("Exceeds the limit (4300 digits) for integer string conversion; "
     "use sys.set_int_max_str_digits() to increase the limit",
     "integer longer than yian allows (4300 digits)"),
], ids=["with-count", "without-count"])
def test_a_node_raising_pythons_digit_limit_error_fails_with_yians_message(monkeypatch, text, expected):
    monkeypatch.setattr(llm_clients, "model_from_config", lambda _: _Model([ValueError(text)]))
    flow = load_flow(_FLOW.replace(_SESSION_LINE, ""))
    result = run_flow(flow, {"question": "q"})
    assert result.status == "failed" and result.failure.code == "node.execute"
    assert expected in result.failure.message
    _assert_clean(result.failure.message)


# Backward replies and feedback


def test_a_long_int_in_a_backward_reply_fails_the_backward_located(monkeypatch):
    text = f'{{"input_feedback": {{}}, "self_feedback": ["ok", {D}], "variable_feedback": {{}}}}'
    chat = _BackwardChat([AIMessage(content=text)])
    monkeypatch.setattr(llm_clients, "model_from_config", lambda _config: chat)
    node = AgentNode("writer", prompt="p", llm_config={"provider": "test", "model": "recorded"})
    with pytest.raises(TypeCheckError) as caught:
        node.backward(_context(), ("Review.",))
    assert str(caught.value) == "backward response at self_feedback[1]: " + LIMIT_MSG
    _assert_clean(str(caught.value))


def _nested_message(levels):
    message = "x"
    for _ in range(levels):
        message = [message]
    return message


def _backward_reply(message, monkeypatch):
    text = json.dumps({"input_feedback": {"topic": [message]}, "self_feedback": [],
                       "variable_feedback": {}})
    monkeypatch.setattr(llm_clients, "model_from_config",
                        lambda _config: _BackwardChat([AIMessage(content=text)]))
    node = AgentNode("writer", prompt="p", llm_config={"provider": "test", "model": "recorded"})
    return node.backward(_context(), ("Review.",))


@pytest.mark.parametrize("levels", [99, 100])
def test_a_backward_reply_measures_nesting_from_its_feedback_messages(monkeypatch, levels):
    message = _nested_message(levels)
    assert _backward_reply(message, monkeypatch).input_feedback == {"topic": (message,)}


def test_a_backward_feedback_message_past_the_limit_still_fails(monkeypatch):
    with pytest.raises(TypeCheckError) as caught:
        _backward_reply(_nested_message(102), monkeypatch)
    assert str(caught.value).startswith("backward response at input_feedback.topic[0]")
    assert "deeper than 100 levels" in str(caught.value)


@pytest.mark.parametrize("feedback, where", [
    ({"note": BIG}, "feedback at note: "),
    ([BIG], "feedback at [0]: "),
], ids=["record", "list"])
def test_long_int_feedback_is_rejected_located_and_the_result_stays_usable(feedback, where):
    flow = load_flow(_code_flow('return {"note": "x"}'))
    result = run_flow(flow, {}, learnable=True)
    assert result.status == "succeeded"
    with pytest.raises(TypeCheckError) as caught:
        result.backward(feedback)
    assert str(caught.value) == where + LIMIT_MSG
    _assert_clean(str(caught.value))
    assert result.backward_outcome is None
    assert result.backward("good").status == "succeeded"


def test_nan_feedback_still_raises_a_type_error():
    flow = load_flow(_code_flow('return {"note": "x"}'))
    result = run_flow(flow, {}, learnable=True)
    with pytest.raises(TypeError):
        result.backward({"note": math.nan})
