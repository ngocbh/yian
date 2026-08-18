"""Strict public tool invocation and trusted ``ask_user`` host metadata."""

from __future__ import annotations

from typing import Any, Callable

import pytest
from langchain_core.tools import BaseTool, StructuredTool
from pydantic.v1 import BaseModel as BaseModelV1

from agent_composer.nodes.tool import ToolNode
from agent_composer.tools import (
    TOOL_REGISTRY,
    install_ask_user,
    invoke_tool,
    invoke_tool_trusted,
    register_tool,
    resolve_tools,
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
    ) -> Any:
        seen.append(
            {
                "question": question,
                "questions": questions,
                "answer_schema": answer_schema,
                "title": title,
                "node_id": node_id,
            }
        )
        return answer

    return handler


def test_ask_user_public_schema_contains_only_question():
    seen: list[dict[str, Any]] = []

    with install_ask_user(_recording_handler(seen)):
        tool = resolve_tools(["ask_user"])[0]
        assert set(tool.tool_call_schema.model_fields) == {"question"}
        assert invoke_tool("ask_user", {"question": "Approve?"}) == "answer"

    assert seen == [
        {
            "question": "Approve?",
            "questions": None,
            "answer_schema": None,
            "title": None,
            "node_id": None,
        }
    ]
    assert "ask_user" not in TOOL_REGISTRY


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

    with install_ask_user(_recording_handler(seen)):
        with pytest.raises(ValueError, match="unsupported argument"):
            invoke_tool("ask_user", {"question": "Approve?", key: value})

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

    with install_ask_user(_recording_handler(seen, answer={"decision": True})):
        assert invoke_tool_trusted("ask_user", metadata) == {"decision": True}

    assert seen == [metadata]


def test_trusted_invocation_rejects_unknown_metadata():
    with install_ask_user(_recording_handler([])):
        with pytest.raises(ValueError, match="unsupported argument"):
            invoke_tool_trusted(
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
    with pytest.raises(ValueError, match="reserved"):

        @register_tool("ask_user")
        def replacement(question: str) -> str:
            """A host must install this reserved tool through its ownership API."""
            return question


def test_install_ask_user_rejects_duplicate_active_owner():
    first_seen: list[dict[str, Any]] = []
    second_seen: list[dict[str, Any]] = []

    with install_ask_user(_recording_handler(first_seen)):
        installed = TOOL_REGISTRY["ask_user"]
        with pytest.raises(ValueError, match="already has an active owner"):
            with install_ask_user(_recording_handler(second_seen)):
                pass
        assert TOOL_REGISTRY["ask_user"] is installed
        assert invoke_tool("ask_user", {"question": "First?"}) == "answer"

    assert [call["question"] for call in first_seen] == ["First?"]
    assert second_seen == []


def test_install_ask_user_allows_sequential_owners():
    first_seen: list[dict[str, Any]] = []
    second_seen: list[dict[str, Any]] = []

    with install_ask_user(_recording_handler(first_seen, answer="first")):
        assert invoke_tool("ask_user", {"question": "One?"}) == "first"
    with install_ask_user(_recording_handler(second_seen, answer="second")):
        assert invoke_tool("ask_user", {"question": "Two?"}) == "second"

    assert [call["question"] for call in first_seen] == ["One?"]
    assert [call["question"] for call in second_seen] == ["Two?"]


def test_install_cleanup_does_not_remove_replacement_registration():
    def replacement(question: str) -> str:
        """Replacement used to prove cleanup checks the installed tool identity."""
        return question

    replacement_tool = StructuredTool.from_function(
        replacement,
        name="replacement_ask_user",
    )

    with install_ask_user(_recording_handler([])):
        TOOL_REGISTRY["ask_user"] = replacement_tool

    assert TOOL_REGISTRY["ask_user"] is replacement_tool
