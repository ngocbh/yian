"""`tool_calling` mode emits the declared `output:` type on its final answer turn."""

import pytest

import yian.llm_clients as llm_clients_mod
import yian.tools as tools_mod
from yian.llm_clients import LLMConfig
from yian.nodes.agent import AgentNode
from yian.nodes.contracts import ValueContract
from yian.runtime.eval_node import eval_node
from yian.typesys.pool import VariablePool
from yian.typesys.values import Type, ValueKind

from langchain_core.messages import AIMessage
from pydantic import BaseModel


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


def _ai_tool_call(name, args, call_id="1"):
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )


class _StructuredFinalChat:
    """A tool-calling model: runs one tool turn, then its final (no-tool) answer is produced
    structurally via with_structured_output for the declared record type."""

    def __init__(self, schema_value):
        self._value = schema_value
        self._replies = [_ai_tool_call("value", {"topic": "ACME"}), AIMessage(content="done")]
        self.structured_called_with = None

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        return self._replies.pop(0)

    def with_structured_output(self, schema):
        self.structured_called_with = schema
        value = self._value

        class _Bound:
            def invoke(self, messages):
                return schema.model_validate(value)

        return _Bound()


class _DirectStructuredChat:
    """Native structured fake that rejects an ordinary discovery invocation."""

    def __init__(self, *schema_values):
        self._values = list(schema_values)
        self.structured_calls = 0

    def bind_tools(self, tools):
        raise AssertionError("a no-tools structured run must not bind tools")

    def invoke(self, messages):
        raise AssertionError("a no-tools structured run must not make a discovery call")

    def with_structured_output(self, schema):
        outer = self

        class _Bound:
            def invoke(self, messages):
                outer.structured_calls += 1
                return schema.model_validate(outer._values.pop(0))

        return _Bound()


class _DirectFallbackChat:
    """Fallback structured fake whose sole invoke must carry the schema prompt."""

    def __init__(self):
        self.calls = 0

    def bind_tools(self, tools):
        raise AssertionError("a no-tools structured run must not bind tools")

    def with_structured_output(self, schema):
        raise AssertionError("the no-structured sentinel must use the fallback")

    def invoke(self, messages):
        self.calls += 1
        assert "json" in messages[-1].content.lower()
        return AIMessage(content='{"value": 5}')


def _run_node(node, pool=None):
    return eval_node(node, None, pool or VariablePool()).outcome


def test_tool_calling_final_answer_is_structured(monkeypatch):
    tool = _FakeTool(lambda a: f"px:{a['topic']}")
    monkeypatch.setitem(tools_mod.TOOL_REGISTRY, "value", tool)
    chat = _StructuredFinalChat({"name": "ACME", "score": 9})
    monkeypatch.setattr(llm_clients_mod, "model_from_config", lambda cfg: chat)

    node = AgentNode("n", prompt="hi", tools=["value"], llm_config=LLMConfig(), mode="tool_calling")
    node.output_contract = ValueContract(
        Type(
            kind=ValueKind.OBJECT,
            fields={
                "name": Type.scalar(ValueKind.STRING),
                "score": Type.scalar(ValueKind.INTEGER),
            },
            required=frozenset({"name", "score"}),
        )
    )
    out = _run_node(node).value
    assert out == {"name": "ACME", "score": 9}  # a plain dict, structured final answer
    assert chat.structured_called_with is not None  # the structured emit turn ran
    assert tool.seen == [{"topic": "ACME"}]  # the mid-loop tool still ran


@pytest.mark.parametrize(
    ("output_type", "schema_value", "expected"),
    [
        (
            Type(
                kind=ValueKind.STRING,
                tags=frozenset({"positive", "neutral", "negative"}),
            ),
            {"value": "neutral"},
            "neutral",
        ),
        (Type.scalar(ValueKind.INTEGER), {"value": 7}, 7),
        (
            Type(
                kind=ValueKind.LIST_STRING,
                element=Type.scalar(ValueKind.STRING),
            ),
            {"items": ["alpha", "beta"]},
            ["alpha", "beta"],
        ),
        (
            Type(
                kind=ValueKind.OBJECT,
                fields={
                    "name": Type.scalar(ValueKind.STRING),
                    "score": Type.scalar(ValueKind.INTEGER),
                },
                required=frozenset({"name", "score"}),
            ),
            {"name": "ACME", "score": 9},
            {"name": "ACME", "score": 9},
        ),
    ],
    ids=["literal", "scalar", "list", "record"],
)
def test_no_tools_structured_output_skips_discovery_call(
    monkeypatch, output_type, schema_value, expected
):
    chat = _DirectStructuredChat(schema_value)
    monkeypatch.setattr(llm_clients_mod, "model_from_config", lambda cfg: chat)
    node = AgentNode(
        "n",
        prompt="produce a value",
        llm_config=LLMConfig(provider="anthropic", model="claude-sonnet-4-5"),
        mode="tool_calling",
    )
    node.output_contract = ValueContract(output_type)

    assert _run_node(node).value == expected
    assert chat.structured_calls == 1


def test_no_tools_structured_fallback_uses_one_schema_prompt(monkeypatch):
    chat = _DirectFallbackChat()
    monkeypatch.setattr(llm_clients_mod, "model_from_config", lambda cfg: chat)
    node = AgentNode(
        "n",
        prompt="produce a number",
        llm_config=LLMConfig(provider="vllm", model="no-structured-sentinel"),
        mode="tool_calling",
    )
    node.output_contract = ValueContract(Type.scalar(ValueKind.INTEGER))

    assert _run_node(node).value == 5
    assert chat.calls == 1


def test_no_tools_structured_retry_never_adds_a_discovery_call(monkeypatch):
    chat = _DirectStructuredChat({"value": "not-an-int"}, {"value": 11})
    monkeypatch.setattr(llm_clients_mod, "model_from_config", lambda cfg: chat)
    node = AgentNode(
        "n",
        prompt="produce a number",
        llm_config=LLMConfig(provider="anthropic", model="claude-sonnet-4-5"),
        mode="tool_calling",
    )
    node.output_contract = ValueContract(Type.scalar(ValueKind.INTEGER))

    assert _run_node(node).value == 11
    assert chat.structured_calls == 2


def test_tool_calling_literal_final_answer_is_structured(monkeypatch):
    chat = _StructuredFinalChat({"value": "neutral"})
    chat._replies = [AIMessage(content="The sentiment is neutral.")]
    monkeypatch.setattr(llm_clients_mod, "model_from_config", lambda cfg: chat)
    node = AgentNode(
        "n",
        prompt="classify",
        llm_config=LLMConfig(provider="ollama", model="qwen2.5:72b"),
        mode="tool_calling",
    )
    node.output_contract = ValueContract(
        Type(
            kind=ValueKind.STRING,
            tags=frozenset({"positive", "neutral", "negative"}),
        )
    )

    assert _run_node(node).value == "neutral"
    assert chat.structured_called_with is not None


def test_tool_calling_declared_bare_str_answer_unchanged(monkeypatch):
    chat = _StructuredFinalChat({})
    chat._replies = [AIMessage(content="plain answer")]
    monkeypatch.setattr(llm_clients_mod, "model_from_config", lambda cfg: chat)
    node = AgentNode("n", prompt="hi", llm_config=LLMConfig(), mode="tool_calling")
    node.output_contract = ValueContract(Type.scalar(ValueKind.STRING))
    # An unconstrained string remains text passthrough with no structured emit turn.
    assert _run_node(node).value == "plain answer"
    assert chat.structured_called_with is None
