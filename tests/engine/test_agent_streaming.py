"""AGENT reply streaming: visible chat text reaches `emit_stream` as it arrives.

A chat request streams only when the host supplied a stream sink. Its chunks share
the request's `exchange_id`. A completed request ends with one empty `final=True`
chunk; a failed request ends without one. Without a stream sink the model is
invoked exactly as before.
"""

import pytest
from langchain_core.messages import AIMessage

import yian.llm_clients as llm_clients_mod
import yian.tools as tools_mod
from yian.compose import load_flow
from yian.events import (
    ModelCallFinished,
    ModelCallStarted,
    StreamChunk,
    TokenUsage,
    ToolCallStarted,
)
from yian.memory.session import SessionMemory
from yian.nodes.agent import AgentNode
from yian.run import run_flow
from yian.runtime.eval_node import EvaluationFailure, eval_node
from yian.typesys.pool import VariablePool
from tests.engine._stream_fakes import Cut, ScriptedStreamChat, reply, usage
from tests.engine.test_agent_live_events import CONFIG, _agent, _evaluate, _Tool
from tests.engine.test_agent_session import _FLOW, _view


def test_stream_chunk_exchange_id_defaults_to_none():
    assert StreamChunk("n", "text", "hi").exchange_id is None
    assert StreamChunk("n", "text", "hi", exchange_id="x").exchange_id == "x"


def test_plain_agent_streams_text_chunks_then_one_final_chunk(monkeypatch):
    node, chat = _agent(
        monkeypatch, [reply("hello big world", tokens=usage(5, 3))]
    )
    outcome, chunks, observed = _evaluate(node)
    assert outcome.value == "hello big world"
    assert chat.mode == ["stream"]
    exchange = observed[0].exchange_id
    assert chunks == [
        StreamChunk("writer", "text", "hello", exchange_id=exchange),
        StreamChunk("writer", "text", " big", exchange_id=exchange),
        StreamChunk("writer", "text", " world", exchange_id=exchange),
        StreamChunk("writer", "text", "", final=True, exchange_id=exchange),
    ]
    assert observed == [
        ModelCallStarted("writer", exchange, "chat", "anthropic", "claude-test"),
        ModelCallFinished(
            "writer", exchange, "chat", "anthropic", "claude-test",
            usage=TokenUsage(5, 3, 8),
        ),
    ]


def test_stream_sink_alone_streams_without_observers(monkeypatch):
    node, chat = _agent(monkeypatch, [reply("just text")])
    outcome, chunks, observed = _evaluate(node, events=False)
    assert outcome.value == "just text" and chat.mode == ["stream"] and observed == []
    assert [(c.chunk, c.final) for c in chunks] == [("just", False), (" text", False), ("", True)]


def test_tool_calling_agent_streams_each_exchange_between_its_observers(monkeypatch):
    monkeypatch.setitem(tools_mod.TOOL_REGISTRY, "lookup", _Tool())
    node, chat = _agent(
        monkeypatch,
        [
            reply("let me check", calls=[("lookup", {"topic": "acme"}, "call-1")]),
            reply("acme is fine"),
        ],
        tools=["lookup"],
    )
    outcome, chunks, observed = _evaluate(node)
    assert outcome.value == "acme is fine"
    assert chat.mode == ["stream", "stream"]
    first, second = observed[0].exchange_id, observed[4].exchange_id
    assert observed[2] == ToolCallStarted("writer", first, 0, "call-1", "lookup", {"topic": "acme"})
    assert [(c.exchange_id, c.chunk, c.final) for c in chunks] == [
        (first, "let", False), (first, " me", False), (first, " check", False),
        (first, "", True),
        (second, "acme", False), (second, " is", False), (second, " fine", False),
        (second, "", True),
    ]


def test_two_streamed_tool_calls_aggregate_by_index(monkeypatch):
    tool = _Tool()
    monkeypatch.setitem(tools_mod.TOOL_REGISTRY, "lookup", tool)
    calls = [("lookup", {"topic": "a"}, "call-a"), ("lookup", {"topic": "b"}, "call-b")]
    node, chat = _agent(monkeypatch, [reply(calls=calls), reply("both fine")], tools=["lookup"])
    outcome, chunks, observed = _evaluate(node)
    assert outcome.value == "both fine" and chat.mode == ["stream", "stream"]
    assert tool.seen == [{"topic": "a"}, {"topic": "b"}]
    started = [e for e in observed if isinstance(e, ToolCallStarted)]
    assert [(e.call_index, e.tool_call_id, e.arguments) for e in started] == [
        (0, "call-a", {"topic": "a"}), (1, "call-b", {"topic": "b"}),
    ]
    assert [(c.chunk, c.final) for c in chunks] == [("", True), ("both", False), (" fine", False), ("", True)]


def test_stream_sink_failure_escapes_as_a_host_fault(monkeypatch):
    class HostBroken(Exception):
        pass

    def broken(_item):
        raise HostBroken("host")

    node, _ = _agent(monkeypatch, [reply("text")])
    with pytest.raises(HostBroken):
        eval_node(node, None, VariablePool(), emit_stream=broken)


def test_duck_typed_client_without_stream_emits_one_chunk(monkeypatch):
    class _InvokeOnly:
        def bind_tools(self, _tools):
            return self

        def invoke(self, _messages):
            return AIMessage(content="whole reply")

    monkeypatch.setattr(llm_clients_mod, "model_from_config", lambda _config: _InvokeOnly())
    node = AgentNode("writer", prompt="hi", llm_config=dict(CONFIG))
    outcome, chunks, _ = _evaluate(node, events=False)
    assert outcome.value == "whole reply"
    assert [(c.chunk, c.final) for c in chunks] == [("whole reply", False), ("", True)]


def test_streamed_tool_exchange_is_retained_and_restored_like_an_invoked_one(monkeypatch):
    tool_flow = _FLOW.replace("    kind: agent\n", "    kind: agent\n    tools: [lookup]\n")
    assert tool_flow != _FLOW
    flow = load_flow(tool_flow)
    monkeypatch.setitem(tools_mod.TOOL_REGISTRY, "lookup", _Tool())
    chat = ScriptedStreamChat(
        script=[reply(calls=[("lookup", {"topic": "a"}, "call-1")]), reply("one"), reply("two")],
        mode=[],
    )
    monkeypatch.setattr(llm_clients_mod, "model_from_config", lambda _config: chat)
    memory = SessionMemory()
    chunks = []
    first = run_flow(flow, {"question": "q1"}, memory={"session": memory}, on_stream=chunks.append)
    assert first.output == "one" and chat.mode == ["stream", "stream"]
    second = run_flow(flow, {"question": "q2"}, memory={"session": memory})
    assert second.output == "two" and chat.mode[-1] == "invoke"
    view = _view(memory)
    assert [effect.kind for effect in view.effects] == ["model", "tool", "model", "model"]
    assert all(effect.status == "completed" for effect in view.effects)
    replies = [item.payload["message"] for item in view.items if item.payload["message"]["type"] == "ai"]
    assert replies[0]["data"]["tool_calls"][0]["name"] == "lookup"
    assert [message["data"]["content"] for message in replies] == ["", "one", "two"]


# --- a request that fails mid-stream --------------------------------------- #


def test_mid_stream_failure_sends_no_final_chunk_and_fails_the_node(monkeypatch):
    node, chat = _agent(
        monkeypatch, [Cut("partial answer", ConnectionError("reset by sk-hidden"))]
    )
    outcome, chunks, observed = _evaluate(node)
    assert isinstance(outcome, EvaluationFailure) and chat.mode == ["stream"]
    exchange = observed[0].exchange_id
    assert [(c.exchange_id, c.chunk, c.final) for c in chunks] == [
        (exchange, "partial", False), (exchange, " answer", False),
    ]
    assert observed[-1] == ModelCallFinished(
        "writer", exchange, "chat", "anthropic", "claude-test", error="ConnectionError"
    )
    assert "sk-hidden" not in repr(observed)


def test_reply_that_fails_normalization_sends_no_final_chunk(monkeypatch):
    class _Unnormalizable(ScriptedStreamChat):
        def normalize_reply(self, _message):
            raise ValueError("malformed reply")

    chat = _Unnormalizable(script=[reply("hello world")], mode=[])
    monkeypatch.setattr(llm_clients_mod, "model_from_config", lambda _config: chat)
    node = AgentNode("writer", prompt="hi", llm_config=dict(CONFIG))
    outcome, chunks, observed = _evaluate(node)
    assert isinstance(outcome, EvaluationFailure)
    assert [(c.chunk, c.final) for c in chunks] == [("hello", False), (" world", False)]
    assert observed[-1].error == "ValueError"


def test_mid_stream_failure_in_a_session_turn_retains_a_failed_model_effect(monkeypatch):
    chat = ScriptedStreamChat(script=[Cut("partial", ConnectionError("reset"))], mode=[])
    monkeypatch.setattr(llm_clients_mod, "model_from_config", lambda _config: chat)
    memory = SessionMemory()
    chunks, events = [], []
    result = run_flow(
        load_flow(_FLOW), {"question": "q1"}, memory={"session": memory},
        on_stream=chunks.append, on_event=events.append,
    )
    assert result.status == "failed" and chat.mode == ["stream"]
    assert [(c.chunk, c.final) for c in chunks] == [("partial", False)]
    assert [e.error for e in events if isinstance(e, ModelCallFinished)] == ["ConnectionError"]
    view = _view(memory)
    assert [(effect.kind, effect.status) for effect in view.effects] == [("model", "failed")]
    assert not [item for item in view.items if item.payload["message"]["type"] == "ai"]
