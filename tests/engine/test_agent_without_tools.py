"""An AGENT without tools or skills makes a single model call; one with tools runs the loop.

Fake chat models stand in for the provider (no API keys).
"""

from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel

import yian.llm_clients as llm_clients_mod
import yian.tools as tools_mod
from yian.compose import load_flow
from yian.events import StreamChunk
from yian.llm_clients import LLMConfig
from yian.nodes.agent import AgentNode
from yian.run import run_flow
from yian.runtime.eval_node import eval_node
from yian.typesys.pool import VariablePool
from tests.engine._stream_fakes import ScriptedStreamChat, reply


def _flow(*lines: str) -> str:
    """One `chat` agent with no tools, plus the given body lines."""
    body = "".join(f"    {line}\n" for line in lines)
    return f"""\
id: single
name: single
llm_config: {{provider: anthropic, model: claude-test}}
nodes:
  chat:
    kind: agent
    prompt: Help the user.
{body}output: ${{chat.output}}
"""


class Recorder:
    """A duck-typed chat model that records each request and serves scripted replies."""

    def __init__(self, replies=(), structured=()):
        self.replies = list(replies)
        self.structured = list(structured)
        self.calls = []
        self.structured_calls = []
        self.bound = []

    def bind_tools(self, tools, **kwargs):
        self.bound.append(tools)
        return self

    def invoke(self, messages):
        self.calls.append(list(messages))
        return self.replies.pop(0)

    def with_structured_output(self, schema, **kwargs):
        outer = self

        class _Bound:
            def invoke(self, messages):
                outer.structured_calls.append(list(messages))
                return schema.model_validate(outer.structured.pop(0))

        return _Bound()


class Inbox:
    """A host inbox: steer messages queued per node id, plus interrupt flags."""

    def __init__(self, steers=None, *, interrupt=()):
        self.queued = {node_id: list(messages) for node_id, messages in (steers or {}).items()}
        self.interrupt = set(interrupt)

    def drain(self, node_id):
        return self.queued.pop(node_id, [])

    def interrupted(self, node_id):
        return node_id in self.interrupt


def _patch(monkeypatch, model):
    monkeypatch.setattr(llm_clients_mod, "model_from_config", lambda _config: model)
    return model


def test_text_agent_without_tools_makes_one_unbound_call(monkeypatch):
    model = _patch(monkeypatch, Recorder([AIMessage(content="hello")]))
    result = run_flow(load_flow(_flow("output: str")), {})
    assert result.output == "hello"
    assert len(model.calls) == 1
    assert model.bound == []


def test_structured_agent_without_tools_makes_one_structured_call(monkeypatch):
    model = _patch(monkeypatch, Recorder(structured=[{"name": "Ada", "score": 3}]))
    result = run_flow(load_flow(_flow("output: {name: str, score: int}")), {})
    assert result.output == {"name": "Ada", "score": 3}
    assert len(model.structured_calls) == 1
    assert model.calls == [] and model.bound == []


class _TopicToolArgs(BaseModel):
    """Schema for the fake value tool's one public argument."""

    topic: str


class _FakeTool:
    def __init__(self, fn):
        self._fn = fn
        self.seen = []
        self.args_schema = _TopicToolArgs
        self.tool_call_schema = _TopicToolArgs

    def invoke(self, args):
        self.seen.append(args)
        return self._fn(args)


def test_python_built_agent_with_tools_runs_the_tool_loop(monkeypatch):
    tool = _FakeTool(lambda args: f"px:{args['topic']}")
    monkeypatch.setitem(tools_mod.TOOL_REGISTRY, "value", tool)
    model = _patch(monkeypatch, Recorder([
        reply(calls=[("value", {"topic": "ACME"}, "1")]),
        reply("done"),
    ]))
    node = AgentNode("n", prompt="hi", tools=["value"], llm_config=LLMConfig())
    assert eval_node(node, None, VariablePool()).outcome.value == "done"
    assert tool.seen == [{"topic": "ACME"}]
    assert len(model.bound) == 1 and len(model.calls) == 2


def test_skills_and_steerable_need_no_mode():
    assert AgentNode("n", prompt="p", skills=["review"]).skills == ["review"]
    assert AgentNode("n", prompt="p", steerable=True).steerable is True


def test_steer_joins_the_single_request_after_the_prompt(monkeypatch):
    model = _patch(monkeypatch, Recorder([AIMessage(content="ok")]))
    flow = load_flow(_flow("steerable: true", "output: str"))
    result = run_flow(flow, {}, steering=Inbox({"chat": ["be brief"]}))
    assert result.output == "ok"
    assert len(model.calls) == 1
    assert model.calls[0][-2:] == [HumanMessage("Help the user."), HumanMessage("be brief")]


def test_interrupt_while_streaming_keeps_the_partial_text(monkeypatch):
    chat = _patch(monkeypatch, ScriptedStreamChat(script=[reply("hello big world")], mode=[]))
    inbox = Inbox()
    chunks = []

    def on_stream(chunk):
        chunks.append(chunk)
        if chunk.chunk == " big":
            inbox.interrupt.add("chat")

    result = run_flow(
        load_flow(_flow("steerable: true", "output: str")), {}, steering=inbox,
        on_stream=on_stream,
        num_workers=0,  # serial delivery, so the flag is set before the next chunk
    )
    assert result.status == "succeeded"
    assert result.output == "hello big"
    assert chat.mode == ["stream"]
    assert [(c.chunk, c.final) for c in chunks if isinstance(c, StreamChunk)] == [
        ("hello", False), (" big", False), ("", True),
    ]
