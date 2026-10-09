"""Strict public tool invocation and trusted per-run ``ask_user`` host metadata."""

from __future__ import annotations

from typing import Any, Callable

import pytest
from langchain_core.tools import BaseTool, StructuredTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic.v1 import BaseModel as BaseModelV1

from yian.nodes.tool import ToolNode
import yian.tools as tools_package
from yian.tools import (
    TOOL_REGISTRY,
    ToolSet,
    invoke_tool,
    invoke_tool_trusted,
    register_tool,
)


class _LegacyToolArgs(BaseModelV1):
    """A valid LangChain Pydantic-v1 tool schema retained for compatibility."""

    value: int


class _JsonSchemaTool(BaseTool):
    """A valid LangChain tool using its supported JSON-schema argument form."""

    name: str = "json_schema_tool"
    description: str = "Increment one integer."
    args_schema: dict[str, Any] = {
        "type": "object",
        "properties": {"value": {"type": "integer"}},
        "required": ["value"],
        "additionalProperties": False,
    }

    def _run(self, value: int) -> int:
        return value + 1


@pytest.fixture(autouse=True)
def _restore_tool_registry():
    """Keep process-global registrations isolated between registry tests."""
    original = dict(TOOL_REGISTRY)
    yield
    TOOL_REGISTRY.clear()
    TOOL_REGISTRY.update(original)


def _recording_handler(
    seen: list[dict[str, Any]], answer: Any = "answer"
) -> Callable[..., Any]:
    """Build one host handler that records the complete trusted call record."""

    def handler(
        *,
        question: str,
        questions: list[dict] | None = None,
        answer_schema: list[dict] | None = None,
        title: str | None = None,
        node_id: str | None = None,
        options: list[str] | None = None,
    ) -> Any:
        seen.append(
            {
                "question": question,
                "questions": questions,
                "answer_schema": answer_schema,
                "title": title,
                "node_id": node_id,
                "options": options,
            }
        )
        return answer

    return handler


def test_ask_user_public_schema_contains_question_and_options():
    seen: list[dict[str, Any]] = []
    tools = ToolSet({"ask_user": _recording_handler(seen)})

    tool = tools.resolve(["ask_user"])[0]
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
    assert "ask_user" not in TOOL_REGISTRY


def test_ask_user_openai_schema_offers_optional_string_options():
    tool = ToolSet({"ask_user": _recording_handler([])}).resolve(["ask_user"])[0]

    schema = convert_to_openai_tool(tool)["function"]["parameters"]

    assert schema["required"] == ["question"]
    options = schema["properties"]["options"]
    assert options["anyOf"] == [{"type": "array", "items": {"type": "string"}}, {"type": "null"}]
    assert "user can always type a different answer" in options["description"]


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("questions", [{"id": "reason", "prompt": "Why?"}]),
        ("answer_schema", [{"name": "reason", "type": "str"}]),
        ("title", "Approval"),
        ("node_id", "gate"),
        ("unexpected", True),
    ],
)
def test_public_invocation_rejects_injected_and_unknown_keys(key, value):
    seen: list[dict[str, Any]] = []
    tools = ToolSet({"ask_user": _recording_handler(seen)})

    with pytest.raises(ValueError, match="unsupported argument"):
        tools.invoke("ask_user", {"question": "Approve?", key: value})

    assert seen == []


@pytest.mark.parametrize("invoke", [invoke_tool, invoke_tool_trusted])
def test_invocation_rejects_unknown_tool(invoke):
    with pytest.raises(ValueError, match="unknown tool"):
        invoke("not_registered", {})


def test_trusted_invocation_passes_structured_metadata():
    seen: list[dict[str, Any]] = []
    metadata = {
        "question": "Review the proposal",
        "questions": [{"id": "decision", "prompt": "Approve?"}],
        "answer_schema": [{"name": "decision", "type": "bool"}],
        "title": "Human review",
        "node_id": "approval",
    }

    tools = ToolSet({"ask_user": _recording_handler(seen, answer={"decision": True})})

    assert tools.invoke_trusted("ask_user", metadata) == {"decision": True}

    assert seen == [{**metadata, "options": None}]


def test_trusted_invocation_rejects_unknown_metadata():
    tools = ToolSet({"ask_user": _recording_handler([])})

    with pytest.raises(ValueError, match="unsupported argument"):
        tools.invoke_trusted(
            "ask_user",
            {"question": "Approve?", "not_metadata": "model supplied"},
        )


def test_invocation_supports_pydantic_v1_schema():
    def increment(value: int) -> int:
        """Increment one integer."""
        return value + 1

    TOOL_REGISTRY["pydantic_v1"] = StructuredTool.from_function(
        increment,
        name="pydantic_v1",
        args_schema=_LegacyToolArgs,
    )

    assert invoke_tool("pydantic_v1", {"value": 3}) == 4


def test_invocation_rejects_unvalidated_json_schema_tools():
    TOOL_REGISTRY["json_schema"] = _JsonSchemaTool()

    with pytest.raises(TypeError, match="Pydantic argument schema"):
        invoke_tool("json_schema", {"value": 2})


def test_tool_node_uses_public_invocation_path():
    @register_tool("tool_node_public_probe")
    def echo(question: str) -> str:
        """Return one public question."""
        return question

    node = ToolNode("probe", tool_id="tool_node_public_probe")
    with pytest.raises(ValueError, match="unsupported argument"):
        node.forward({"question": "Approve?", "node_id": "model-injected"})


def test_generic_registration_rejects_reserved_ask_user_id():
    with pytest.raises(ValueError, match="reserved; pass tools="):

        @register_tool("ask_user")
        def replacement(question: str) -> str:
            """A host must pass this reserved tool per run."""
            return question


@pytest.mark.parametrize("tools", [None, {}])
def test_ask_user_never_falls_back_to_the_global_registry(tools):
    def global_answer(question: str) -> str:
        """A stray global entry that no run may use as its handler."""
        return "global answer"

    TOOL_REGISTRY["ask_user"] = StructuredTool.from_function(global_answer, name="ask_user")

    with pytest.raises(ValueError, match="no ask_user handler for this run"):
        ToolSet(tools).invoke("ask_user", {"question": "Approve?"})


def test_install_ask_user_is_removed():
    assert not hasattr(tools_package, "install_ask_user")
    assert "install_ask_user" not in tools_package.__all__
