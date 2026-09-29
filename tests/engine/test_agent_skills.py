"""AGENT skill grants: validation, the system listing, and the load_skill tool."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from langchain_core.messages import AIMessage
import pytest

import agent_composer.llm_clients as llm_clients
import agent_composer.nodes.agent.modes.tool_calling as tool_calling_module
from agent_composer.memory.contracts import ToolCallInteraction, ToolResultInteraction
from agent_composer.nodes.agent import AgentNode
from agent_composer.nodes.agent.modes.common import DEFAULT_SYSTEM, MODES
from agent_composer.nodes.base import Output
from agent_composer.nodes.contracts import ValueContract
from agent_composer.runtime.eval_node import EvaluationFailure, eval_node
from agent_composer.skills import Skill, SkillError
from agent_composer.skills.tool import LOAD_SKILL, system_prompt_with_skills
from agent_composer.typesys import VariablePool
from agent_composer.typesys.values import Type, ValueKind
from tests.engine._fakes import FuncNode


class _ScriptedModel:
    """Return scripted replies and capture every request and bound tool name."""

    def __init__(self, replies: list[AIMessage]) -> None:
        self.replies = list(replies)
        self.calls: list[list] = []
        self.bound: list[str] = []

    def bind_tools(self, tools):
        self.bound = [tool.name for tool in tools]
        return self

    def invoke(self, messages):
        self.calls.append(list(messages))
        return self.replies.pop(0)


def _call(name: str, args: dict, call_id: str = "1") -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}])


@pytest.fixture
def skills(tmp_path: Path) -> dict[str, Skill]:
    directory = tmp_path / "review"
    (directory / "references").mkdir(parents=True)
    (directory / "references" / "style.md").write_text("Prefer small diffs.\n", encoding="utf-8")
    return {
        "review": Skill("review", "Review code changes.", "Read references/style.md first.", directory),
        "deploy": Skill("deploy", "Ship a release.", "Tag, then publish."),
        "other": Skill("other", "Unrelated.", "Not granted."),
    }


def _eval(node, **kwargs):
    return eval_node(node, SimpleNamespace(wiring={}), VariablePool(), **kwargs)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"skills": ["review"], "mode": "plain"}, "skills need mode 'tool_calling', not 'plain'"),
        ({"tools": [LOAD_SKILL]}, "reserved tool name"),
        ({"skills": ["review"], "tools": [LOAD_SKILL]}, "reserved tool name"),
        ({"skills": ["review", "review"]}, "duplicate skill grant"),
        ({"skills": ["Review"]}, "invalid skill name"),
        ({"skills": "some"}, "list of names or 'all'"),
    ],
)
def test_agent_rejects_invalid_skill_grants(kwargs: dict, message: str) -> None:
    kwargs.setdefault("mode", "tool_calling")
    with pytest.raises(SkillError, match=message):
        AgentNode("agent", prompt="p", **kwargs)


def test_agent_rejects_skills_in_any_other_registered_mode(monkeypatch) -> None:
    monkeypatch.setitem(MODES, "custom", MODES["tool_calling"])
    with pytest.raises(SkillError, match="not 'custom'"):
        AgentNode("agent", prompt="p", mode="custom", skills=["review"])


def test_agent_skill_grants_default_to_none() -> None:
    assert AgentNode("agent").skills == []
    assert AgentNode("agent", mode="tool_calling", skills="all").skills == "all"
