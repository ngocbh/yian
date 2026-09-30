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
from agent_composer.tools import registry
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
        ({"group": {"shout": _upper}}, r"tools\['group'\].*got dict"),
        ({"group": [lambda text: text]}, r"tools\['group'\]\[0\].*lambda"),
        ({"group": ["shout"]}, r"tools\['group'\]\[0\].*got str"),
        ({"shout": functools.partial(_upper, text="hi")}, r"tools\['shout'\].*got partial"),
        ({"shout": _Shouter()}, r"tools\['shout'\].*got _Shouter"),
    ],
)
def test_invalid_tools_are_rejected(tools: Any, error: str):
    with pytest.raises(TypeError, match=error):
        ToolSet(tools)


def _lower(text: str) -> str:
    """Return the text in lower case."""
    return text.lower()


def _named(name: str) -> StructuredTool:
    """Build one BaseTool that echoes its text under `name`."""
    return StructuredTool.from_function(lambda text: f"{name}:{text}", name=name, description=name)


def test_group_entry_holds_tools_named_by_function_or_tool_name():
    tools = ToolSet({"toolbox": [_upper, _named("echo")], "shout": _upper})

    assert tools.expand(["toolbox"]) == ["_upper", "echo"]
    assert [tool.name for tool in tools.resolve(["_upper", "echo"])] == ["_upper", "echo"]
    assert tools.invoke("echo", {"text": "a"}) == "echo:a"
    assert tools.invoke("_upper", {"text": "a"}) == "A"


def test_expand_replaces_each_group_with_its_tools_in_order():
    tools = ToolSet(
        {"search": _upper, "toolbox": [_named("a"), _named("b")], "empty": [], "work": (_lower,)}
    )

    assert tools.expand(["search", "toolbox", "empty", "work"]) == ["search", "a", "b", "_lower"]
    assert ToolSet().expand(["search"]) == ["search"]


def test_a_group_is_not_itself_a_tool():
    tools = ToolSet({"toolbox": [_named("a")]})

    with pytest.raises(ValueError, match="'toolbox' is a tool group"):
        tools.resolve(["toolbox"])
    with pytest.raises(ValueError, match="'toolbox' is a tool group"):
        tools.invoke("toolbox", {})


_SHARED = _named("x")


@pytest.mark.parametrize(
    ("tools", "error"),
    [
        ({"one": [_named("x")], "two": [_named("x")]}, "'x' is in both groups 'one' and 'two'"),
        ({"one": [_named("x"), _named("x")]}, "'x' appears twice in group 'one'"),
        ({"one": [_SHARED], "two": [_SHARED], "x": []}, "'x' in group 'one' clashes"),
        ({"x": _upper, "one": [_named("x")]}, "'x' in group 'one' clashes with tools\['x'\]"),
        ({"x": [], "one": [_named("x")]}, "'x' in group 'one' clashes with tools\['x'\]"),
        ({"one": [_named("ask_user")]}, "group 'one' has a tool named 'ask_user'"),
        ({"one": [_named("load_skill")]}, "group 'one' has a tool named 'load_skill'"),
    ],
)
def test_group_tool_names_must_be_unique_in_the_run(tools: dict[str, Any], error: str):
    with pytest.raises(ValueError, match=error):
        ToolSet(tools)


def test_the_same_tool_object_may_sit_in_overlapping_groups():
    web = _named("web")
    tools = ToolSet(
        {"toolbox": [web, _named("a"), web], "company": [_upper, web], "more": [_upper]}
    )

    assert tools.expand(["toolbox"]) == ["web", "a"]
    assert tools.expand(["company", "toolbox", "more"]) == ["_upper", "web", "a"]
    assert tools.resolve(["web"]) == [web]
    assert tools.invoke("_upper", {"text": "a"}) == "A"


def test_the_same_bound_method_may_sit_in_two_groups():
    shouter = _Shouter()
    tools = ToolSet({"one": [shouter.shout], "two": [shouter.shout]})

    assert tools.expand(["one", "two"]) == ["shout"]


def test_expand_keeps_the_first_of_each_repeated_name():
    tools = ToolSet({"box": [_named("a"), _named("b")], "search": _upper})

    assert tools.expand(["box", "search", "box", "a", "search"]) == ["a", "b", "search"]


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


def test_ask_user_entry_is_wrapped_so_the_model_sees_question_and_options():
    seen: list[dict[str, Any]] = []
    tools = ToolSet({"ask_user": _recording_handler(seen)})

    (tool,) = tools.resolve(["ask_user"])

    assert set(tool.tool_call_schema.model_fields) == {"question", "options"}
    assert tools.invoke("ask_user", {"question": "Approve?"}) == "answer"
    assert seen == [
        {
            "question": "Approve?",
            "questions": None,
            "answer_schema": None,
            "title": None,
            "node_id": None,
            "options": None,
        }
    ]


def test_model_options_reach_the_handler_normalized():
    seen: list[dict[str, Any]] = []
    tools = ToolSet({"ask_user": _recording_handler(seen)})

    tools.invoke("ask_user", {"question": "Which?", "options": ["A", "a", " B "]})

    assert seen[0]["options"] == ["A", "B"]


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
    assert seen == [{**_TRUSTED, "options": None}]


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
    with pytest.raises(TypeError, match=r"options.*def handler\(\*\*request\)"):
        ToolSet({"ask_user": lambda question: "yes"})


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, None),
        ("A, B", ["A, B"]),
        ('["A","B"]', ["A", "B"]),
        ('["A", 2]', ["A"]),
        ("[broken", ["[broken"]),
        ([1, "", " a ", "A"], ["a"]),
        ({"x": 1}, None),
        (5, None),
        ([str(i) for i in range(10)], ["0", "1", "2", "3", "4", "5"]),
        (["x" * 200], ["x" * 79 + "…"]),
        (["two\n  lines"], ["two lines"]),
        (["", "  "], None),
    ],
)
def test_ask_user_options_are_normalized_leniently(raw: Any, expected: Any):
    assert registry._normalize_options(raw) == expected


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
            "options": None,
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
    assert set(chat.bound_tools[1].tool_call_schema.model_fields) == {"question", "options"}
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


def _replies(*calls: tuple[str, dict[str, Any]]) -> list[AIMessage]:
    """Build one tool-call reply per call, then a final answer."""
    return [_tool_call(name, args) for name, args in calls] + [AIMessage(content="done")]


def test_agent_granted_three_groups_sees_the_union_of_their_tools(monkeypatch):
    chat = _ScriptedChat(_replies(("a", {"text": "1"}), ("c", {"text": "3"})))
    monkeypatch.setattr(llm_clients, "model_from_config", lambda config: chat)
    tools = ToolSet(
        {
            "toolbox": [_named("a")],
            "workspace": [_named("b"), _named("c")],
            "company": [_named("d")],
            "hidden": [_named("e")],
        }
    )
    node = AgentNode(
        "agent", prompt="go", tools=["toolbox", "workspace", "company"], mode="tool_calling"
    )

    evaluated = eval_node(node, None, VariablePool(), tools=tools)

    assert evaluated.outcome == Output("done")
    assert [tool.name for tool in chat.bound_tools] == ["a", "b", "c", "d"]
    assert [message.content for message in chat.calls[-1] if isinstance(message, ToolMessage)] == [
        "a:1",
        "c:3",
    ]


@pytest.mark.parametrize("name", ["e", "toolbox"])
def test_agent_grant_check_sees_expanded_group_names(monkeypatch, name: str):
    chat = _ScriptedChat([_tool_call(name, {"text": "x"})])
    monkeypatch.setattr(llm_clients, "model_from_config", lambda config: chat)
    tools = ToolSet({"toolbox": [_named("a")], "hidden": [_named("e")]})
    node = AgentNode("agent", prompt="go", tools=["toolbox"], mode="tool_calling")

    evaluated = eval_node(node, None, VariablePool(), tools=tools)

    assert isinstance(evaluated.outcome, EvaluationFailure)
    assert f"requested tool {name!r}, but it is not granted; granted tools: ['a']" in (
        evaluated.outcome.failure.message
    )


def test_agent_binds_a_tool_shared_by_overlapping_groups_once(monkeypatch):
    chat = _ScriptedChat(_replies(("web", {"text": "q"})))
    monkeypatch.setattr(llm_clients, "model_from_config", lambda config: chat)
    web = _named("web")
    tools = ToolSet({"toolbox": [web, _named("a")], "company": [web, _named("b")]})
    node = AgentNode(
        "agent", prompt="go", tools=["toolbox", "company", "toolbox"], mode="tool_calling"
    )

    evaluated = eval_node(node, None, VariablePool(), tools=tools)

    assert evaluated.outcome == Output("done")
    assert [tool.name for tool in chat.bound_tools] == ["web", "a", "b"]


def test_agent_with_groups_and_ask_user_asks_the_top_level_handler(monkeypatch):
    chat = _ScriptedChat(_replies(("ask_user", {"question": "Which?"}), ("a", {"text": "go"})))
    monkeypatch.setattr(llm_clients, "model_from_config", lambda config: chat)
    seen: list[dict[str, Any]] = []
    tools = ToolSet({"toolbox": [_named("a")], "ask_user": _recording_handler(seen, "B")})
    node = AgentNode("agent", prompt="go", tools=["toolbox", "ask_user"], mode="tool_calling")

    evaluated = eval_node(node, None, VariablePool(), tools=tools)

    assert evaluated.outcome == Output("done")
    assert [tool.name for tool in chat.bound_tools] == ["a", "ask_user"]
    assert [request["question"] for request in seen] == ["Which?"]
    results = [m.content for m in chat.calls[-1] if isinstance(m, ToolMessage)]
    assert results == ["B", "a:go"]
