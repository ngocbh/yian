"""AGENT in-loop observers: model and tool call brackets with token usage.

A scripted LangChain chat model stands in for the provider (no API keys). The
evaluator supplies `emit_stream`/`emit_event` only to nodes declaring
`needs_live_events`; without sinks the model is invoked exactly as before.
"""

import pytest
from langchain_core.messages import AIMessage
from pydantic import BaseModel

import agent_composer.llm_clients as llm_clients_mod
import agent_composer.tools as tools_mod
from agent_composer.events import (
    ModelCallFinished,
    ModelCallStarted,
    TokenUsage,
    ToolCallFinished,
    ToolCallStarted,
)
from agent_composer.nodes.agent import MODES, AgentNode
from agent_composer.nodes.base import Output
from agent_composer.nodes.contracts import ValueContract
from agent_composer.runtime.eval_node import EvaluationFailure, eval_node
from agent_composer.typesys.pool import VariablePool
from agent_composer.typesys.values import Type, ValueKind
from tests.engine._stream_fakes import ScriptedStreamChat, reply, usage

CONFIG = {"provider": "anthropic", "model": "claude-test"}


class _Args(BaseModel):
    topic: str


class _Tool:
    args_schema = tool_call_schema = _Args
    name = "lookup"

    def __init__(self, action=None):
        self.seen = []
        self.action = action or (lambda args: f"found:{args['topic']}")

    def invoke(self, args):
        self.seen.append(args)
        return self.action(args)


def _agent(monkeypatch, script, *, mode="tool_calling", tools=(), output=None):
    chat = ScriptedStreamChat(script=list(script), mode=[])
    monkeypatch.setattr(llm_clients_mod, "model_from_config", lambda _config: chat)
    node = AgentNode("writer", prompt="hi", tools=list(tools), llm_config=dict(CONFIG), mode=mode)
    if output is not None:
        node.output_contract = ValueContract(type=output)
    return node, chat


def _evaluate(node, *, stream=True, events=True):
    chunks, observed = [], []
    outcome = eval_node(
        node, None, VariablePool(),
        emit_stream=chunks.append if stream else None,
        emit_event=observed.append if events else None,
    ).outcome
    return outcome, chunks, observed


def test_agent_mode_context_receives_only_supplied_sinks(monkeypatch):
    seen = []

    def probe(ctx):
        seen.append((ctx.emit_stream is not None, ctx.emit_event is not None))
        return Output(value="ok")

    monkeypatch.setitem(MODES, "probe", probe)
    assert AgentNode.needs_live_events is True
    for stream, event in [(True, True), (False, True), (False, False)]:
        node = AgentNode("writer", prompt="hi", llm_config=dict(CONFIG), mode="probe")
        _evaluate(node, stream=stream, events=event)
    assert seen == [(True, True), (False, True), (False, False)]


def test_agent_without_sinks_invokes_exactly_as_before(monkeypatch):
    node, chat = _agent(monkeypatch, [reply("same answer")], mode="plain")
    outcome, chunks, observed = _evaluate(node, stream=False, events=False)
    assert outcome.value == "same answer"
    assert chat.mode == ["invoke"]
    assert chunks == [] and observed == []


def test_event_sink_alone_reports_usage_without_streaming(monkeypatch):
    tokens = usage(7, 2, cache_read=4, reasoning=1)
    node, chat = _agent(monkeypatch, [reply("ok", tokens=tokens)], mode="plain")
    outcome, chunks, observed = _evaluate(node, stream=False)
    assert outcome.value == "ok" and chat.mode == ["invoke"] and chunks == []
    assert observed[-1].usage == TokenUsage(
        7, 2, 9, cache_read_tokens=4, reasoning_tokens=1
    )


def test_provider_without_usage_reports_none(monkeypatch):
    node, _ = _agent(monkeypatch, [reply("ok")], mode="plain")
    _, _, observed = _evaluate(node)
    assert observed[-1].usage is None and observed[-1].error is None


def test_model_error_is_reported_by_type_then_fails_the_node(monkeypatch):
    node, _ = _agent(monkeypatch, [RuntimeError("key=sk-hidden")], mode="plain")
    outcome, chunks, observed = _evaluate(node)
    assert isinstance(outcome, EvaluationFailure)
    assert chunks == []
    assert observed[-1].error == "RuntimeError" and observed[-1].usage is None
    assert "sk-hidden" not in repr(observed)


def test_event_sink_failure_escapes_as_a_host_fault(monkeypatch):
    class HostBroken(Exception):
        pass

    def broken(_item):
        raise HostBroken("host")

    node, _ = _agent(monkeypatch, [reply("text")], mode="plain")
    with pytest.raises(HostBroken):
        eval_node(node, None, VariablePool(), emit_event=broken)


def test_history_summary_call_is_observed_through_the_wrapper():
    from agent_composer.memory.session import SessionMemory
    from agent_composer.nodes.agent import context
    from tests.engine._session_fakes import harness, request
    from tests.engine.test_agent_session import _Model
    from tests.engine.test_agent_session_summaries import _seed

    memory = SessionMemory(history_budget_chars=1, recent_turns=0)
    _seed(memory)
    model = _Model([AIMessage(content="summary text", usage_metadata=usage(3, 1))])
    wrapped = []

    def observe(invoke):
        reply_message = invoke()
        wrapped.append(reply_message.usage_metadata["total_tokens"])
        return reply_message

    with harness(memory) as opened:
        with opened.frame.open_turn(request("current"), check_stopped=lambda: None) as turn:
            messages = context.prepare_history(
                turn, model, check_stopped=lambda: None, observe_summary=observe
            )
    assert wrapped == [4]
    assert "summary text" in str(messages)


def test_history_summary_calls_publish_their_own_observers(monkeypatch):
    from agent_composer.compose import load_flow
    from agent_composer.memory.session import SessionMemory
    from agent_composer.run import run_flow
    from tests.engine.test_agent_session import _FLOW

    chat = ScriptedStreamChat(
        script=[reply("first"), reply("old facts", tokens=usage(8, 2)), reply("second")], mode=[]
    )
    monkeypatch.setattr(llm_clients_mod, "model_from_config", lambda _config: chat)
    memory = SessionMemory(history_budget_chars=1, recent_turns=0)
    flow = load_flow(_FLOW)
    run_flow(flow, {"question": "q1"}, memory={"session": memory})
    events = []
    assert run_flow(flow, {"question": "q2"}, memory={"session": memory},
                    on_event=events.append).output == "second"
    finished = [event for event in events if isinstance(event, ModelCallFinished)]
    assert [(event.purpose, event.usage) for event in finished] == [
        ("history_summary", TokenUsage(8, 2, 10)), ("chat", None),
    ]


def test_structured_fallback_reports_usage_and_streams_nothing(monkeypatch):
    monkeypatch.setattr(
        "agent_composer.nodes.agent.structured._supports_native_structured", lambda _c: False
    )
    node, chat = _agent(
        monkeypatch, [reply('{"value": 3}', tokens=usage(9, 1))], mode="plain",
        output=Type.scalar(ValueKind.INTEGER),
    )
    outcome, chunks, observed = _evaluate(node)
    assert outcome.value == 3 and chat.mode == ["invoke"] and chunks == []
    assert [type(event) for event in observed] == [ModelCallStarted, ModelCallFinished]
    assert {event.purpose for event in observed} == {"structured_fallback"}
    assert observed[-1].usage == TokenUsage(9, 1, 10)


def _native_script():
    """A text reply without the schema tool call, then bad arguments, then a valid call."""
    return [
        reply("no tool call", tokens=usage(5, 1)),
        reply(calls=[("ScalarWrapper", {"value": "four"}, "c1")], tokens=usage(6, 1)),
        reply(calls=[("ScalarWrapper", {"value": 4}, "c2")], tokens=usage(7, 2)),
    ]


@pytest.mark.parametrize("events", [False, True])
def test_structured_native_retries_the_same_with_or_without_an_event_sink(monkeypatch, events):
    monkeypatch.setattr(
        "agent_composer.nodes.agent.structured._supports_native_structured", lambda _c: True
    )
    node, chat = _agent(
        monkeypatch, _native_script(), mode="plain", output=Type.scalar(ValueKind.INTEGER)
    )
    outcome, chunks, observed = _evaluate(node, stream=False, events=events)
    assert outcome.value == 4
    assert chat.mode == ["invoke", "invoke", "invoke"] and chunks == []
    finished = [event for event in observed if isinstance(event, ModelCallFinished)]
    assert [(event.purpose, event.usage, event.error) for event in finished] == (
        [
            ("structured_native", TokenUsage(5, 1, 6), None),
            ("structured_native", TokenUsage(6, 1, 7), "ValidationError"),
            ("structured_native", TokenUsage(7, 2, 9), None),
        ]
        if events else []
    )


def test_structured_native_turn_reports_usage_and_retains_the_raw_reply(monkeypatch):
    from agent_composer.compose import load_flow
    from agent_composer.memory.session import SessionMemory
    from agent_composer.run import run_flow
    from tests.engine.test_agent_session import _FLOW, _view

    monkeypatch.setattr(
        "agent_composer.nodes.agent.structured._supports_native_structured", lambda _c: True
    )
    chat = ScriptedStreamChat(script=_native_script()[2:], mode=[])
    monkeypatch.setattr(llm_clients_mod, "model_from_config", lambda _config: chat)
    memory = SessionMemory()
    events = []
    result = run_flow(
        load_flow(_FLOW.replace("output: str", "output: int")), {"question": "q"},
        memory={"session": memory}, on_event=events.append,
    )
    assert result.output == 4 and chat.mode == ["invoke"]
    finished = [event for event in events if isinstance(event, ModelCallFinished)]
    assert [(event.purpose, event.usage) for event in finished] == [
        ("structured_native", TokenUsage(7, 2, 9)),
    ]
    view = _view(memory)
    assert [(effect.kind, effect.status) for effect in view.effects] == [("model", "completed")]
    replies = [item.payload["message"] for item in view.items if item.payload["message"]["type"] == "ai"]
    assert replies[0]["data"]["tool_calls"][0]["args"] == {"value": 4}


def test_tool_calling_agent_orders_model_and_tool_observers(monkeypatch):
    tool = _Tool()
    monkeypatch.setitem(tools_mod.TOOL_REGISTRY, "lookup", tool)
    node, chat = _agent(
        monkeypatch,
        [
            reply("let me check", calls=[("lookup", {"topic": "acme"}, "call-1")],
                  tokens=usage(10, 4)),
            reply("acme is fine", tokens=usage(20, 3)),
        ],
        tools=["lookup"],
    )
    outcome, chunks, observed = _evaluate(node, stream=False)
    assert outcome.value == "acme is fine"
    assert chat.mode == ["invoke", "invoke"] and chunks == []
    assert tool.seen == [{"topic": "acme"}]
    first, second = observed[0].exchange_id, observed[4].exchange_id
    assert first != second
    assert observed == [
        ModelCallStarted("writer", first, "chat", "anthropic", "claude-test"),
        ModelCallFinished("writer", first, "chat", "anthropic", "claude-test",
                          usage=TokenUsage(10, 4, 14)),
        ToolCallStarted("writer", first, 0, "call-1", "lookup", {"topic": "acme"}),
        ToolCallFinished("writer", first, 0, "call-1", "lookup", result="found:acme"),
        ModelCallStarted("writer", second, "chat", "anthropic", "claude-test"),
        ModelCallFinished("writer", second, "chat", "anthropic", "claude-test",
                          usage=TokenUsage(20, 3, 23)),
    ]


def test_tool_error_is_reported_by_type_then_fails_the_node(monkeypatch):
    def explode(_args):
        raise ValueError("secret detail")

    monkeypatch.setitem(tools_mod.TOOL_REGISTRY, "lookup", _Tool(explode))
    node, _ = _agent(
        monkeypatch, [reply(calls=[("lookup", {"topic": "x"}, "call-1")])], tools=["lookup"]
    )
    outcome, _, observed = _evaluate(node)
    assert isinstance(outcome, EvaluationFailure)
    assert observed[-1] == ToolCallFinished(
        "writer", observed[0].exchange_id, 0, "call-1", "lookup", error="ValueError"
    )


def test_two_tool_calls_in_one_reply_get_their_own_indexed_brackets(monkeypatch):
    tool = _Tool()
    monkeypatch.setitem(tools_mod.TOOL_REGISTRY, "lookup", tool)
    node, _ = _agent(
        monkeypatch,
        [
            reply(calls=[("lookup", {"topic": "a"}, "call-a"), ("lookup", {"topic": "b"}, "call-b")]),
            reply("both fine"),
        ],
        tools=["lookup"],
    )
    outcome, _, observed = _evaluate(node, stream=False)
    assert outcome.value == "both fine"
    assert tool.seen == [{"topic": "a"}, {"topic": "b"}]
    brackets = [event for event in observed if isinstance(event, (ToolCallStarted, ToolCallFinished))]
    assert [(type(event), event.call_index, event.tool_call_id) for event in brackets] == [
        (ToolCallStarted, 0, "call-a"), (ToolCallFinished, 0, "call-a"),
        (ToolCallStarted, 1, "call-b"), (ToolCallFinished, 1, "call-b"),
    ]
    assert {event.exchange_id for event in brackets} == {observed[0].exchange_id}
