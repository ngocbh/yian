"""AGENT keeps blocking human interaction inside one live model/tool loop."""

from __future__ import annotations

import inspect

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import StructuredTool
import pytest

import agent_composer.llm_clients as llm_clients
import agent_composer.tools as tools_module
from agent_composer.events import NodeFailed, NodeSucceeded
from agent_composer.nodes.agent import AgentNode
from agent_composer.runtime.engine import FlowEngine
from agent_composer.runtime.eval_node import eval_node
from agent_composer.tools import install_ask_user
from agent_composer.typesys.pool import VariablePool
from agent_composer.typesys.values import Type, ValueKind


class _ScriptedChat:
    """Return scripted replies and retain each message list seen by the model."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []
        self.bound_tools = []

    def bind_tools(self, tools):
        self.bound_tools = list(tools)
        return self

    def invoke(self, messages):
        self.calls.append(list(messages))
        return self.replies.pop(0)


def _tool_reply(name: str, args: dict, call_id: str) -> AIMessage:
    """Build one model tool call in LangChain's normalized shape."""
    return AIMessage(
        content="",
        tool_calls=[
            {"name": name, "args": args, "id": call_id, "type": "tool_call"}
        ],
    )


def _drive(node: AgentNode):
    """Return the terminal event for one direct engine-boundary evaluation."""
    return list(eval_node(node, None, VariablePool()))[-1]


def test_mode_resolves_model_factory_at_call_time(monkeypatch):
    node = AgentNode("agent", prompt="go", mode="plain")
    chat = _ScriptedChat([AIMessage(content="late-bound")])
    monkeypatch.setattr(llm_clients, "model_from_config", lambda config: chat)

    event = _drive(node)

    assert isinstance(event, NodeSucceeded)
    assert event.output == "late-bound"


def test_ask_user_answer_is_a_tool_message_in_the_same_loop(monkeypatch):
    chat = _ScriptedChat(
        [
            _tool_reply("ask_user", {"question": "Approve?"}, "question-1"),
            AIMessage(content="approved result"),
        ]
    )
    prompts = []

    def answer(**request):
        prompts.append(request)
        return "yes"

    node = AgentNode(
        "agent",
        prompt="go",
        tools=["ask_user"],
        mode="tool_calling",
    )
    monkeypatch.setattr(llm_clients, "model_from_config", lambda config: chat)
    with install_ask_user(answer):
        event = _drive(node)

    assert isinstance(event, NodeSucceeded)
    assert event.output == "approved result"
    assert [tool.name for tool in chat.bound_tools] == ["ask_user"]
    assert "Ask the user only through an explicitly granted tool" in chat.calls[0][0].content
    assert prompts == [
        {
            "question": "Approve?",
            "questions": None,
            "answer_schema": None,
            "title": None,
            "node_id": None,
        }
    ]
    answer_message = chat.calls[1][-1]
    assert isinstance(answer_message, ToolMessage)
    assert answer_message.content == "yes"
    assert answer_message.tool_call_id == "question-1"


def test_two_blocking_questions_keep_one_conversation(monkeypatch):
    chat = _ScriptedChat(
        [
            _tool_reply("ask_user", {"question": "First?"}, "question-1"),
            _tool_reply("ask_user", {"question": "Second?"}, "question-2"),
            AIMessage(content="complete"),
        ]
    )
    node = AgentNode(
        "agent",
        prompt="go",
        tools=["ask_user"],
        mode="tool_calling",
    )
    monkeypatch.setattr(llm_clients, "model_from_config", lambda config: chat)
    answers = iter(["one", "two"])
    questions = []

    def answer(**request):
        questions.append(request["question"])
        return next(answers)

    with install_ask_user(answer):
        event = _drive(node)

    assert isinstance(event, NodeSucceeded)
    assert event.output == "complete"
    assert questions == ["First?", "Second?"]
    tool_messages = [
        message for message in chat.calls[-1] if isinstance(message, ToolMessage)
    ]
    assert [(message.tool_call_id, message.content) for message in tool_messages] == [
        ("question-1", "one"),
        ("question-2", "two"),
    ]


def test_model_tool_calls_run_in_exact_returned_order(monkeypatch):
    seen = []

    def first(value: int) -> str:
        """Record the first tool call."""
        seen.append(("first", value))
        return "one"

    def second(value: int) -> str:
        """Record the second tool call."""
        seen.append(("second", value))
        return "two"

    monkeypatch.setitem(
        tools_module.TOOL_REGISTRY,
        "first",
        StructuredTool.from_function(first, name="first"),
    )
    monkeypatch.setitem(
        tools_module.TOOL_REGISTRY,
        "second",
        StructuredTool.from_function(second, name="second"),
    )
    chat = _ScriptedChat(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "second",
                        "args": {"value": 2},
                        "id": "2",
                        "type": "tool_call",
                    },
                    {
                        "name": "ask_user",
                        "args": {"question": "Between tools?"},
                        "id": "human",
                        "type": "tool_call",
                    },
                    {
                        "name": "first",
                        "args": {"value": 1},
                        "id": "1",
                        "type": "tool_call",
                    },
                ],
            ),
            AIMessage(content="done"),
        ]
    )
    monkeypatch.setattr(llm_clients, "model_from_config", lambda config: chat)

    def answer(**request):
        seen.append(("ask_user", request["question"]))
        return "human answer"

    with install_ask_user(answer):
        event = _drive(
            AgentNode(
                "agent",
                prompt="go",
                tools=["first", "ask_user", "second"],
                mode="tool_calling",
            )
        )

    assert isinstance(event, NodeSucceeded)
    assert seen == [
        ("second", 2),
        ("ask_user", "Between tools?"),
        ("first", 1),
    ]


def test_ask_user_requires_an_explicit_authored_grant(monkeypatch):
    chat = _ScriptedChat(
        [_tool_reply("ask_user", {"question": "Secret?"}, "question-1")]
    )
    monkeypatch.setattr(llm_clients, "model_from_config", lambda config: chat)
    handled = []

    with install_ask_user(lambda **request: handled.append(request) or "no"):
        event = _drive(AgentNode("agent", prompt="go", tools=[], mode="tool_calling"))

    assert isinstance(event, NodeFailed)
    assert "not granted" in event.error
    assert handled == []


def test_ask_user_exception_propagates_from_the_live_node(monkeypatch):
    failure = RuntimeError("terminal failed")
    chat = _ScriptedChat(
        [_tool_reply("ask_user", {"question": "Continue?"}, "question-1")]
    )
    monkeypatch.setattr(llm_clients, "model_from_config", lambda config: chat)

    def fail(**request):
        raise failure

    node = AgentNode(
        "agent",
        prompt="go",
        tools=["ask_user"],
        mode="tool_calling",
    )
    with install_ask_user(fail), pytest.raises(RuntimeError) as caught:
        node.run({})

    assert caught.value is failure


def test_structured_output_succeeds_after_blocking_ask(monkeypatch):
    class _StructuredChat(_ScriptedChat):
        def with_structured_output(self, schema):
            class _Bound:
                def invoke(self, messages):
                    return schema.model_validate({"name": "Ada", "score": 9})

            return _Bound()

    chat = _StructuredChat(
        [
            _tool_reply("ask_user", {"question": "Approve?"}, "question-1"),
            AIMessage(content="ready for structured output"),
        ]
    )
    monkeypatch.setattr(llm_clients, "model_from_config", lambda config: chat)
    node = AgentNode(
        "agent",
        prompt="go",
        tools=["ask_user"],
        mode="tool_calling",
        llm_config={"provider": "openai", "model": "gpt-4o"},
    )
    node.output_type = Type(
        kind=ValueKind.OBJECT,
        fields={
            "name": Type.scalar(ValueKind.STRING),
            "score": Type.scalar(ValueKind.INTEGER),
        },
        required=frozenset({"name", "score"}),
    )

    with install_ask_user(lambda **request: "yes"):
        event = _drive(node)

    assert isinstance(event, NodeSucceeded)
    assert event.output == {"name": "Ada", "score": 9}


def test_post_assert_fires_after_blocking_ask(monkeypatch):
    chat = _ScriptedChat(
        [
            _tool_reply("ask_user", {"question": "Approve?"}, "question-1"),
            AIMessage(content="BAD"),
        ]
    )
    monkeypatch.setattr(llm_clients, "model_from_config", lambda config: chat)
    node = AgentNode(
        "agent",
        prompt="go",
        tools=["ask_user"],
        mode="tool_calling",
    )
    node.post_asserts = ["output == 'GOOD'"]

    with install_ask_user(lambda **request: "yes"):
        event = _drive(node)

    assert isinstance(event, NodeFailed)
    assert event.error_type == "NodeAssertFailed"
    assert "post-assert" in event.error.lower()


def test_agent_has_no_continuation_or_engine_llm_cap_surface():
    import agent_composer.nodes.agent as agent_package
    from agent_composer.nodes.base import Node

    assert not hasattr(agent_package, "AgentEntry")
    assert not hasattr(agent_package, "Fresh")
    assert not hasattr(agent_package, "Resume")
    assert "entry" not in inspect.signature(AgentNode).parameters
    assert "controls" not in inspect.signature(AgentNode).parameters
    assert "llm" not in inspect.signature(eval_node).parameters
    assert "llm" not in inspect.signature(FlowEngine).parameters
    assert not hasattr(Node, "needs_llm")
    assert AgentNode.is_spawner is False
    assert not hasattr(Node, "grow_restamps_self")
