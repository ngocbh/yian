"""AGENT skill grants: validation, the system listing, and the load_skill tool."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from langchain_core.messages import AIMessage
import pytest

import yian.llm_clients as llm_clients
import yian.nodes.agent.modes.tool_calling as tool_calling_module
from yian.memory.contracts import MemoryContext, ToolCallInteraction, ToolResultInteraction
from yian.nodes.agent import AgentNode
from yian.nodes.agent.modes.common import DEFAULT_SYSTEM, MODES
from yian.nodes.base import Output
from yian.nodes.contracts import ValueContract
from yian.runtime.eval_node import EvaluationFailure, eval_node
from yian.skills import Skill, SkillError
from yian.skills.tool import LOAD_SKILL, system_prompt_with_skills
from yian.typesys import VariablePool
from yian.typesys.values import Type, ValueKind
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


def test_missing_grant_fails_the_node(monkeypatch, skills) -> None:
    monkeypatch.setattr(llm_clients, "model_from_config", lambda _config: _ScriptedModel([]))
    missing = _eval(AgentNode("agent", prompt="p", mode="tool_calling", skills=["absent"]), skills=skills)
    assert type(missing.outcome) is EvaluationFailure
    assert "not supplied by the host: ['absent']" in missing.outcome.failure.message


def test_grant_without_a_skills_mapping_fails_the_node(monkeypatch) -> None:
    monkeypatch.setattr(llm_clients, "model_from_config", lambda _config: _ScriptedModel([]))
    unsupplied = _eval(AgentNode("agent", prompt="p", mode="tool_calling", skills="all"))
    assert type(unsupplied.outcome) is EvaluationFailure
    assert "the host supplied no skills= mapping" in unsupplied.outcome.failure.message


def test_only_nodes_with_the_skills_trait_receive_skills(skills) -> None:
    assert _eval(FuncNode("code", lambda _inputs: "ok"), skills=skills).outcome == Output("ok")


def test_all_grants_every_supplied_skill(monkeypatch, skills) -> None:
    model = _ScriptedModel([AIMessage(content="ok")])
    monkeypatch.setattr(llm_clients, "model_from_config", lambda _config: model)
    node = AgentNode("agent", prompt="p", mode="tool_calling", skills="all")
    assert _eval(node, skills=skills).outcome == Output("ok")
    assert model.calls[0][0].content == system_prompt_with_skills(DEFAULT_SYSTEM, list(skills.values()))


def test_no_grant_or_no_supply_keeps_the_system_prompt_byte_for_byte(monkeypatch, skills) -> None:
    model = _ScriptedModel([AIMessage(content="a"), AIMessage(content="b")])
    monkeypatch.setattr(llm_clients, "model_from_config", lambda _config: model)
    _eval(AgentNode("plain-grant", prompt="p", mode="tool_calling"), skills=skills)
    _eval(AgentNode("all-grant", prompt="p", mode="tool_calling", skills="all"), skills={})
    assert [call[0].content for call in model.calls] == [DEFAULT_SYSTEM, DEFAULT_SYSTEM]


def test_memory_context_gets_the_system_prompt_without_the_listing(monkeypatch, skills) -> None:
    model = _ScriptedModel([AIMessage(content="ok")])
    monkeypatch.setattr(llm_clients, "model_from_config", lambda _config: model)
    seen: list[str] = []

    def open_context(system: str, prompt: str, compatibility: dict) -> MemoryContext:
        seen.append(system)
        return MemoryContext()

    node = AgentNode("agent", prompt="p", mode="tool_calling", skills=["review"])
    assert node.forward({}, open_memory_context=open_context, skills=skills) == Output("ok")
    assert seen == [DEFAULT_SYSTEM]
    assert model.calls[0][0].content == system_prompt_with_skills(DEFAULT_SYSTEM, [skills["review"]])


def test_granted_skills_are_listed_and_loaded_on_demand(monkeypatch, skills) -> None:
    model = _ScriptedModel(
        [
            _call(LOAD_SKILL, {"name": "review"}, "a"),
            _call(LOAD_SKILL, {"name": "review", "path": "references/style.md"}, "b"),
            AIMessage(content="reviewed"),
        ]
    )
    monkeypatch.setattr(llm_clients, "model_from_config", lambda _config: model)
    node = AgentNode("agent", prompt="Review it", mode="tool_calling", skills=["review", "deploy"])

    evaluated = _eval(node, skills=skills, capture_evidence=True)

    assert evaluated.outcome == Output("reviewed")
    expected_system = system_prompt_with_skills(DEFAULT_SYSTEM, [skills["deploy"], skills["review"]])
    assert model.calls[0][0].content == expected_system
    assert "- deploy: Ship a release.\n- review: Review code changes." in expected_system
    assert "other" not in expected_system
    assert model.bound == [LOAD_SKILL]
    assert model.calls[1][-1].content == "Read references/style.md first."
    assert model.calls[2][-1].content == "Prefer small diffs.\n"
    evidence = evaluated.evidence
    assert evidence.base_system_prompt == expected_system
    assert ToolCallInteraction(LOAD_SKILL, {"name": "review"}) in evidence.interactions
    assert ToolResultInteraction(LOAD_SKILL, "Prefer small diffs.\n") in evidence.interactions


def test_skills_with_structured_output_still_run_the_tool_loop(monkeypatch, skills) -> None:
    model = _ScriptedModel([_call(LOAD_SKILL, {"name": "review"}), AIMessage(content="seven")])
    monkeypatch.setattr(llm_clients, "model_from_config", lambda _config: model)
    structured: list[list] = []

    def fake_structured(_model, messages, _output_type, **_kwargs):
        structured.append([message.content for message in messages])
        return 7

    monkeypatch.setattr(tool_calling_module, "generate_structured", fake_structured)
    node = AgentNode("agent", prompt="p", mode="tool_calling", skills=["review"])
    node.output_contract = ValueContract(Type.scalar(ValueKind.INTEGER))

    assert _eval(node, skills=skills).outcome == Output(7)
    assert model.bound == [LOAD_SKILL]
    assert len(model.calls) == 2
    assert "Read references/style.md first." in structured[0]


def test_escape_attempt_fails_the_node(monkeypatch, skills) -> None:
    model = _ScriptedModel([_call(LOAD_SKILL, {"name": "review", "path": "../review/../../secret"})])
    monkeypatch.setattr(llm_clients, "model_from_config", lambda _config: model)
    escaped = _eval(AgentNode("agent", prompt="p", mode="tool_calling", skills=["review"]), skills=skills)
    assert type(escaped.outcome) is EvaluationFailure
    assert "relative path inside the skill directory" in escaped.outcome.failure.message


def test_load_skill_is_not_callable_without_a_grant(monkeypatch) -> None:
    model = _ScriptedModel([_call(LOAD_SKILL, {"name": "review"})])
    monkeypatch.setattr(llm_clients, "model_from_config", lambda _config: model)
    evaluated = _eval(AgentNode("agent", prompt="p", mode="tool_calling"))
    assert type(evaluated.outcome) is EvaluationFailure
    assert "requested tool 'load_skill', but it is not granted" in evaluated.outcome.failure.message
