"""The AGENT node (mode + skills) with a fake langchain model.

No API keys: `model_from_config` is monkeypatched to return a scripted fake chat
model that yields canned `AIMessage`s; tool execution goes through `TOOL_REGISTRY`
(monkeypatched). Covers `plain` mode, `tool_calling` (plain answer, a tool
round-trip, strict grants, exception propagation, and the iteration cap).

(The AGENT-through-the-loader run path is covered once the Compose-loader agent
run lands; AGENT-mode compile-time validation moved with the v0 compiler.)
"""

import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel

import agent_composer.llm_clients as llm_clients_mod
import agent_composer.tools as tools_mod
from agent_composer.nodes.agent import AgentNode
from agent_composer.runtime.eval_node import EvaluationFailure, eval_node
from agent_composer.typesys.pool import VariablePool
from agent_composer.llm_clients import LLMConfig


class _FakeChat:
    """Returns queued AIMessages in order; records calls and bound tools."""

    def __init__(self, replies):
        self._replies = list(replies)
        self.calls = 0
        self.bound_tools = None

    def bind_tools(self, tools):
        self.bound_tools = tools
        return self

    def invoke(self, messages):
        self.calls += 1
        return self._replies.pop(0)


def _ai_tool_call(name, args, call_id="1"):
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )


def _patch_model(monkeypatch, chat):
    monkeypatch.setattr(llm_clients_mod, "model_from_config", lambda cfg: chat)


class _NoToolArgs(BaseModel):
    """Schema for fake tools that accept no public arguments."""


class _TopicToolArgs(BaseModel):
    """Schema for the fake value tool's one public argument."""

    topic: str


class _FakeTool:
    def __init__(self, fn, args_schema=_NoToolArgs):
        self._fn = fn
        self.seen = []
        self.args_schema = args_schema
        self.tool_call_schema = args_schema

    def invoke(self, args):
        self.seen.append(args)
        return self._fn(args)


def _node(tools=None, prompt="hi", mode="tool_calling"):
    return AgentNode("n", prompt=prompt, tools=tools or [], llm_config=LLMConfig(), mode=mode)


def _run_node(node, pool=None):
    """Evaluate one agent into its raw output or failure carrier."""
    return eval_node(node, None, pool or VariablePool())


def test_plain_mode_single_call_ignores_tools(monkeypatch):
    chat = _FakeChat([AIMessage(content="plain answer")])
    _patch_model(monkeypatch, chat)
    # tools listed but plain mode must not bind or loop — one call, no tool use
    assert _run_node(_node(tools=["value"], mode="plain")).value == "plain answer"
    assert chat.calls == 1
    assert chat.bound_tools is None


def test_tool_calling_returns_plain_text(monkeypatch):
    _patch_model(monkeypatch, _FakeChat([AIMessage(content="answer")]))
    assert _run_node(_node()).value == "answer"


def test_tool_calling_round_trip(monkeypatch):
    tool = _FakeTool(lambda a: f"px:{a['topic']}", _TopicToolArgs)
    monkeypatch.setitem(tools_mod.TOOL_REGISTRY, "value", tool)
    chat = _FakeChat([
        _ai_tool_call("value", {"topic": "ACME"}),
        AIMessage(content="done"),
    ])
    _patch_model(monkeypatch, chat)
    assert _run_node(_node(tools=["value"])).value == "done"
    assert tool.seen == [{"topic": "ACME"}]   # the tool actually ran
    assert chat.calls == 2                      # ask -> tool -> ask


def test_tool_calling_executes_model_calls_in_returned_order(monkeypatch):
    seen = []

    def first(value: int) -> str:
        """Record the first call."""
        seen.append(("first", value))
        return "one"

    def second(value: int) -> str:
        """Record the second call."""
        seen.append(("second", value))
        return "two"

    monkeypatch.setitem(
        tools_mod.TOOL_REGISTRY,
        "first",
        StructuredTool.from_function(first, name="first"),
    )
    monkeypatch.setitem(
        tools_mod.TOOL_REGISTRY,
        "second",
        StructuredTool.from_function(second, name="second"),
    )
    chat = _FakeChat(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "first", "args": {"value": 1}, "id": "1", "type": "tool_call"},
                    {"name": "second", "args": {"value": 2}, "id": "2", "type": "tool_call"},
                ],
            ),
            AIMessage(content="done"),
        ]
    )
    _patch_model(monkeypatch, chat)

    assert _run_node(_node(tools=["first", "second"])).value == "done"
    assert seen == [("first", 1), ("second", 2)]


def test_tool_calling_rejects_globally_registered_but_ungranted_tool(monkeypatch):
    seen = []

    def secret() -> str:
        """A process tool this agent was not granted."""
        seen.append("called")
        return "secret"

    monkeypatch.setitem(
        tools_mod.TOOL_REGISTRY,
        "secret",
        StructuredTool.from_function(secret, name="secret"),
    )
    chat = _FakeChat([_ai_tool_call("secret", {})])
    _patch_model(monkeypatch, chat)

    event = _run_node(_node(tools=[]))
    assert isinstance(event, EvaluationFailure)
    assert "not granted" in event.error
    assert seen == []


def test_tool_calling_rejects_model_injected_arguments(monkeypatch):
    seen = []

    def lookup(question: str) -> str:
        """Answer one public question."""
        seen.append(question)
        return question

    monkeypatch.setitem(
        tools_mod.TOOL_REGISTRY,
        "lookup",
        StructuredTool.from_function(lookup, name="lookup"),
    )
    chat = _FakeChat(
        [_ai_tool_call("lookup", {"question": "status?", "node_id": "injected"})]
    )
    _patch_model(monkeypatch, chat)

    event = _run_node(_node(tools=["lookup"]))
    assert isinstance(event, EvaluationFailure)
    assert "unsupported argument" in event.error
    assert seen == []


def test_tool_calling_propagates_tool_exception_as_node_failure(monkeypatch):
    def explode(topic: str) -> str:
        """Raise the failure the node boundary must report."""
        raise RuntimeError(f"cannot load {topic}")

    monkeypatch.setitem(
        tools_mod.TOOL_REGISTRY,
        "explode",
        StructuredTool.from_function(explode, name="explode"),
    )
    chat = _FakeChat([_ai_tool_call("explode", {"topic": "ACME"})])
    _patch_model(monkeypatch, chat)

    event = _run_node(_node(tools=["explode"]))
    assert isinstance(event, EvaluationFailure)
    assert event.error_type == "RuntimeError"
    assert "cannot load ACME" in event.error


def test_tool_calling_cap_trips_when_bounded(monkeypatch):
    # A POSITIVE env cap bounds the loop: always-asks-for-a-tool -> never answers -> trips
    # the cap after `cap` turns. The raised AgentLoopError becomes EvaluationFailure at
    # eval_node. (The DEFAULT is -1 / no cap — see the no-cap test below.)
    cap = 5
    monkeypatch.setitem(tools_mod.TOOL_REGISTRY, "noop", _FakeTool(lambda a: "ok"))
    looping = _FakeChat([_ai_tool_call("noop", {}, str(i)) for i in range(cap)])
    _patch_model(monkeypatch, looping)
    node = _node(tools=["noop"])
    node.env = {"max_tool_iterations": cap}
    ev = _run_node(node)
    assert isinstance(ev, EvaluationFailure)
    assert ev.error_type == "AgentLoopError"
    assert "tool-iteration cap" in ev.error


def test_default_no_cap_runs_unbounded(monkeypatch):
    from agent_composer.nodes.agent.modes.tool_calling import MAX_TOOL_ITERATIONS

    # The default cap is the -1 no-cap sentinel: an agent making many tool turns (well past the
    # old default of 100) then answering still finishes, no AgentLoopError.
    assert MAX_TOOL_ITERATIONS == -1
    monkeypatch.setitem(tools_mod.TOOL_REGISTRY, "noop", _FakeTool(lambda a: "ok"))
    replies = [_ai_tool_call("noop", {}, str(i)) for i in range(150)]
    replies.append(AIMessage(content="done"))
    chat = _FakeChat(replies)
    _patch_model(monkeypatch, chat)
    assert _run_node(_node(tools=["noop"])).value == "done"
    assert chat.calls == 151          # 150 tool turns + the final answer, all uncapped


def test_env_max_tool_iterations_overrides_cap(monkeypatch):
    # A per-node `env: {max_tool_iterations: 2}` bounds the loop: the agent trips the cap after
    # exactly 2 model turns (vs the -1 no-cap default). Proves the env value threads
    # node.env -> _max_tool_iterations -> AgentRunContext -> the mode loop.
    monkeypatch.setitem(tools_mod.TOOL_REGISTRY, "noop", _FakeTool(lambda a: "ok"))
    looping = _FakeChat([_ai_tool_call("noop", {}, str(i)) for i in range(2)])
    _patch_model(monkeypatch, looping)
    node = _node(tools=["noop"])
    node.env = {"max_tool_iterations": 2}
    ev = _run_node(node)
    assert isinstance(ev, EvaluationFailure)
    assert ev.error_type == "AgentLoopError"
    assert "(2)" in ev.error          # the cap in the message is the env override, not the default
    assert looping.calls == 2         # stopped at exactly the env bound


def test_env_max_tool_iterations_minus_one_accepted():
    # -1 is the explicit no-cap sentinel and passes validation (unlike 0 / other negatives).
    node = _node()
    node.env = {"max_tool_iterations": -1}
    assert node._max_tool_iterations() == -1


def test_env_max_tool_iterations_rejects_bad_value(monkeypatch):
    # A non-(positive-int-or-(-1)) env value is rejected as EvaluationFailure (ValueError at the
    # boundary), not silently coerced. bool is explicitly rejected (isinstance(True, int) is True).
    _patch_model(monkeypatch, _FakeChat([AIMessage(content="unused")]))
    node = _node()
    node.env = {"max_tool_iterations": 0}
    ev = _run_node(node)
    assert isinstance(ev, EvaluationFailure)
    assert "max_tool_iterations" in ev.error


def test_agent_run_has_no_scratch_kwarg():
    # AgentNode.run is a pure `run(self, inputs)` with no engine scratch capability.
    import inspect

    assert "scratch" not in inspect.signature(AgentNode.run).parameters
