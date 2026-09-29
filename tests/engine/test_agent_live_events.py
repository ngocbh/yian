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
