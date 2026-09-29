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
