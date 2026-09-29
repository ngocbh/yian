"""Per-run tools: `ToolSet` lookup, `ask_user` wrapping, and run isolation."""

from __future__ import annotations

import functools
from typing import Any

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import StructuredTool

import agent_composer.llm_clients as llm_clients
from agent_composer.compile.model import END_ID, START_ID
from agent_composer.nodes.agent import AgentNode
from agent_composer.nodes.base import Node, NodeKind, Output
from agent_composer.nodes.human_input import HumanInputNode
from agent_composer.nodes.tool.node import ToolNode
from agent_composer.runtime.eval_node import EvaluationFailure, eval_node
from agent_composer.tools import TOOL_REGISTRY, ToolSet, register_tool, registered_tools
from agent_composer.typesys.pool import VariablePool
from tests.engine._graph_builder import _engine, _graph


@pytest.fixture(autouse=True)
def _restore_tool_registry():
    """Keep process-global registrations isolated between tests."""
    original = dict(TOOL_REGISTRY)
    yield
    TOOL_REGISTRY.clear()
    TOOL_REGISTRY.update(original)


def _upper(text: str) -> str:
    """Return the text in upper case."""
    return text.upper()


def test_callable_entry_becomes_a_tool_named_by_its_key():
    tools = ToolSet({"shout": _upper})

    (tool,) = tools.resolve(["shout"])

    assert tool.name == "shout"
    assert tool.description == "Return the text in upper case."
    assert tools.invoke("shout", {"text": "hi"}) == "HI"


def test_per_run_entry_replaces_global_tool_for_that_run_only():
    @register_tool("run_tools_probe")
    def probe(text: str) -> str:
        """Global probe."""
        return f"global:{text}"

    replaced = ToolSet({"run_tools_probe": lambda text: f"run:{text}"})

    assert replaced.invoke("run_tools_probe", {"text": "a"}) == "run:a"
    assert ToolSet().invoke("run_tools_probe", {"text": "a"}) == "global:a"


@pytest.mark.parametrize("tools", [{"shout": _upper}, {}])
def test_a_mapping_hides_the_global_registry(tools: dict[str, Any]):
    @register_tool("run_tools_global_only")
    def global_only(text: str) -> str:
        """Global tool."""
        return text

    assert ToolSet().invoke("run_tools_global_only", {"text": "x"}) == "x"
    with pytest.raises(ValueError, match="unknown tool id 'run_tools_global_only'"):
        ToolSet(tools).resolve(["run_tools_global_only"])


def test_registered_tools_is_a_copy_a_run_can_extend():
    @register_tool("run_tools_snapshot")
    def snapshot_probe(text: str) -> str:
        """Global tool."""
        return f"global:{text}"

    registered_tools().pop("run_tools_snapshot")
    assert "run_tools_snapshot" in TOOL_REGISTRY

    tools = ToolSet({**registered_tools(), "shout": _upper})
    assert tools.invoke("run_tools_snapshot", {"text": "a"}) == "global:a"
    assert tools.invoke("shout", {"text": "a"}) == "A"


def test_public_invocation_rejects_unknown_arguments():
    with pytest.raises(ValueError, match="unsupported argument"):
        ToolSet({"shout": _upper}).invoke("shout", {"text": "a", "extra": 1})


def test_base_tool_entry_must_use_its_key_as_name():
    tool = StructuredTool.from_function(_upper, name="upper", description="Upper.")

    assert ToolSet({"upper": tool}).resolve(["upper"]) == [tool]
    with pytest.raises(ValueError, match="must match its key"):
        ToolSet({"shout": tool})


class _Shouter:
    """A callable object; its bound method is the supported form."""

    def __call__(self, text: str) -> str:
        return text.upper()

    def shout(self, text: str) -> str:
        """Return the text in upper case."""
        return text.upper()


def test_bound_method_entry_becomes_a_tool():
    tools = ToolSet({"shout": _Shouter().shout})

    assert tools.invoke("shout", {"text": "hi"}) == "HI"


@pytest.mark.parametrize(
    ("tools", "error"),
    [
        ([("shout", _upper)], "must be a mapping"),
        ({1: _upper}, "non-empty strings"),
        ({"": _upper}, "non-empty strings"),
        ({"group": [_upper]}, "got list"),
        ({"shout": functools.partial(_upper, text="hi")}, r"tools\['shout'\].*got partial"),
        ({"shout": _Shouter()}, r"tools\['shout'\].*got _Shouter"),
    ],
)
def test_invalid_tools_are_rejected(tools: Any, error: str):
    with pytest.raises(TypeError, match=error):
        ToolSet(tools)


def _recording_handler(seen: list[dict[str, Any]], answer: Any = "answer"):
    """Build one host handler that records the complete trusted call record."""

    def handler(**request: Any) -> Any:
        seen.append(request)
        return answer

    return handler


_TRUSTED = {
    "question": "Review the proposal",
    "questions": [{"id": "decision", "prompt": "Approve?"}],
    "answer_schema": [{"name": "decision", "type": "bool"}],
    "title": "Human review",
    "node_id": "approval",
}


def test_ask_user_entry_is_wrapped_so_the_model_sees_only_question():
    seen: list[dict[str, Any]] = []
    tools = ToolSet({"ask_user": _recording_handler(seen)})

    (tool,) = tools.resolve(["ask_user"])

    assert set(tool.tool_call_schema.model_fields) == {"question"}
    assert tools.invoke("ask_user", {"question": "Approve?"}) == "answer"
    assert seen == [
        {
            "question": "Approve?",
            "questions": None,
            "answer_schema": None,
            "title": None,
            "node_id": None,
        }
    ]


@pytest.mark.parametrize("key", ["questions", "answer_schema", "title", "node_id"])
def test_model_cannot_supply_trusted_ask_user_fields(key: str):
    seen: list[dict[str, Any]] = []
    tools = ToolSet({"ask_user": _recording_handler(seen)})

    with pytest.raises(ValueError, match="unsupported argument"):
        tools.invoke("ask_user", {"question": "Approve?", key: _TRUSTED[key]})
    assert seen == []


def test_trusted_ask_user_call_passes_all_fields_to_the_handler():
    seen: list[dict[str, Any]] = []
    tools = ToolSet({"ask_user": _recording_handler(seen, answer={"decision": True})})

    assert tools.invoke_trusted("ask_user", _TRUSTED) == {"decision": True}
    assert seen == [_TRUSTED]


@pytest.mark.parametrize(
    "entry",
    [
        StructuredTool.from_function(_upper, name="ask_user", description="Ask."),
        "not callable",
    ],
)
def test_ask_user_entry_must_be_a_plain_callable(entry: Any):
    with pytest.raises(TypeError, match="ask_user"):
        ToolSet({"ask_user": entry})


def test_ask_user_handler_must_accept_every_field():
    with pytest.raises(TypeError, match=r"def handler\(\*\*request\)"):
        ToolSet({"ask_user": lambda question: "yes"})


def _ping() -> str:
    """Return a fixed reply."""
    return "run-pong"


def test_eval_node_passes_the_run_toolset_to_a_tool_node():
    node = ToolNode("ping", tool_id="ping")

    evaluated = eval_node(node, None, VariablePool(), tools=ToolSet({"ping": _ping}))

    assert evaluated.outcome == Output("run-pong")


def test_only_nodes_that_declare_needs_tools_receive_the_toolset():
    seen: list[list[str]] = []

    class _CapsProbe(Node):
        kind = NodeKind.CODE

        def forward(self, inputs, **caps):
            seen.append(sorted(caps))
            return Output(None)

    class _ToolsProbe(_CapsProbe):
        needs_tools = True

    tools = ToolSet({"ping": _ping})
    eval_node(_CapsProbe("plain"), None, VariablePool(), tools=tools)
    eval_node(_ToolsProbe("opted_in"), None, VariablePool(), tools=tools)

    assert seen == [[], ["tools"]]


def test_eval_node_rejects_a_raw_tools_mapping():
    with pytest.raises(TypeError, match="tools must be a ToolSet or None"):
        eval_node(ToolNode("ping", tool_id="ping"), None, VariablePool(), tools={"ping": _ping})


def test_flow_engine_passes_its_toolset_to_every_tool_node():
    graph = _graph(
        [ToolNode("ping", tool_id="ping")],
        [(START_ID, "ping"), ("ping", END_ID)],
    )
    engine = _engine(graph, tools=ToolSet({"ping": _ping}))

    list(engine.run())

    assert engine.state.pool.get("ping") == "run-pong"


def test_flow_engine_rejects_a_raw_tools_mapping():
    graph = _graph(
        [ToolNode("ping", tool_id="ping")],
        [(START_ID, "ping"), ("ping", END_ID)],
    )
    with pytest.raises(TypeError, match="tools must be a ToolSet or None"):
        _engine(graph, tools={"ping": _ping})


def test_human_input_asks_the_run_ask_user_handler():
    seen: list[dict[str, Any]] = []
    tools = ToolSet({"ask_user": _recording_handler(seen, answer="yes")})
    node = HumanInputNode("approval", prompt="Ship ${plan}?", title="Gate")

    assert node.run({"plan": "v2"}, tools=tools) == Output("yes")
    assert seen == [
        {
            "question": "Ship v2?",
            "questions": None,
            "answer_schema": None,
            "title": "Gate",
            "node_id": "approval",
        }
    ]


def test_eval_node_gives_human_input_the_run_toolset():
    seen: list[dict[str, Any]] = []
    tools = ToolSet({"ask_user": _recording_handler(seen, answer="yes")})

    evaluated = eval_node(HumanInputNode("ask", prompt="Ready?"), None, VariablePool(), tools=tools)

    assert evaluated.outcome == Output("yes")
    assert [request["node_id"] for request in seen] == ["ask"]


class _ScriptedChat:
    """Return scripted replies and keep the tools bound for the model."""

    def __init__(self, replies: list[AIMessage]):
        self.replies = list(replies)
        self.calls: list[list[Any]] = []
        self.bound_tools: list[Any] = []

    def bind_tools(self, tools):
        self.bound_tools = list(tools)
        return self

    def invoke(self, messages):
        self.calls.append(list(messages))
        return self.replies.pop(0)


def _tool_call(name: str, args: dict[str, Any]) -> AIMessage:
    """Build one model reply that calls one tool."""
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": "call-1", "type": "tool_call"}],
    )


def test_agent_binds_and_calls_granted_per_run_tools(monkeypatch):
    chat = _ScriptedChat([_tool_call("shout", {"text": "hi"}), AIMessage(content="done")])
    monkeypatch.setattr(llm_clients, "model_from_config", lambda config: chat)
    seen: list[dict[str, Any]] = []
    tools = ToolSet({"shout": _upper, "ask_user": _recording_handler(seen)})
    node = AgentNode("agent", prompt="go", tools=["shout", "ask_user"], mode="tool_calling")

    evaluated = eval_node(node, None, VariablePool(), tools=tools)

    assert evaluated.outcome == Output("done")
    assert [tool.name for tool in chat.bound_tools] == ["shout", "ask_user"]
    assert set(chat.bound_tools[1].tool_call_schema.model_fields) == {"question"}
    result = chat.calls[1][-1]
    assert isinstance(result, ToolMessage) and result.content == "HI"
    assert seen == []


def test_agent_cannot_call_a_per_run_tool_the_flow_did_not_grant(monkeypatch):
    chat = _ScriptedChat([_tool_call("shout", {"text": "hi"})])
    monkeypatch.setattr(llm_clients, "model_from_config", lambda config: chat)
    node = AgentNode("agent", prompt="go", tools=["ask_user"], mode="tool_calling")
    tools = ToolSet({"shout": _upper, "ask_user": lambda **request: "yes"})

    evaluated = eval_node(node, None, VariablePool(), tools=tools)

    assert isinstance(evaluated.outcome, EvaluationFailure)
    assert "not granted" in evaluated.outcome.failure.message
