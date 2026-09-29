"""run_flow/resume_flow `skills=`: early checks and durable resume with on-demand skills."""

from __future__ import annotations

from pathlib import Path

from langchain_core.messages import AIMessage
import pytest

import agent_composer.llm_clients as llm_clients
from agent_composer.compose import load_flow
from agent_composer.compose.skills import load_skills
from agent_composer.memory.session import SessionMemory
from agent_composer.nodes.agent.modes.common import DEFAULT_SYSTEM
from agent_composer.run import resume_flow, run_flow
from agent_composer.runtime.log import JsonlRunLog, NodeStartedRecord, NodeSucceededRecord
from agent_composer.skills import Skill
from agent_composer.skills.tool import LOAD_SKILL, system_prompt_with_skills
from tests.engine.test_agent_session import _view


class _SimulatedCrash(BaseException):
    """Process-loss marker that bypasses node-failure conversion."""


class _ScriptedModel:
    """Return scripted replies (or raise a crash) and capture each request."""

    def __init__(self, replies: list) -> None:
        self.replies = list(replies)
        self.calls: list[list] = []

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        self.calls.append(list(messages))
        reply = self.replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return reply


def _call(args: dict, call_id: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": LOAD_SKILL, "args": args, "id": call_id, "type": "tool_call"}])


FLOW = """\
id: skills-e2e
name: skills-e2e
input:
  change: str
nodes:
  review:
    kind: agent
    input:
      change: ${input.change}
    prompt: "Review ${change}"
    skills: [code-review, release-notes]
    output: str
output: ${review.output}
"""

SESSION_FLOW = FLOW.replace(
    "name: skills-e2e\n",
    "name: skills-e2e\nlong_term_memory: session\nllm_config: {provider: openai, model: test-model}\n",
)

ALL_FLOW = FLOW.replace("skills: [code-review, release-notes]", "skills: all")

NESTED = """\
id: nested
name: nested
defs:
  body:
    nodes:
      inner:
        kind: agent
        prompt: hi
        skills: [code-review]
        output: str
    output: ${inner.output}
nodes:
  outer:
    kind: call
    call: body
output: ${outer.output}
"""


def _write_skill(root: Path, name: str, description: str, body: str) -> Path:
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\nlicense: MIT\n---\n\n{body}", encoding="utf-8"
    )
    return directory


@pytest.fixture
def skill_root(tmp_path: Path) -> Path:
    root = tmp_path / "skills"
    review = _write_skill(root, "code-review", "Review a diff for bugs.", "Read references/checklist.md.\n")
    (review / "references").mkdir()
    (review / "references" / "checklist.md").write_text("1. Tests pass.\n", encoding="utf-8")
    _write_skill(root, "release-notes", "Write release notes.", "Group by area.\n")
    _write_skill(root, "unused", "Not granted.", "Never shown.\n")
    (tmp_path / "secret.txt").write_text("do not leak\n", encoding="utf-8")
    return root


def test_run_flow_delivers_skills_to_the_agent(monkeypatch, skill_root: Path) -> None:
    model = _ScriptedModel([AIMessage(content="ok")])
    monkeypatch.setattr(llm_clients, "model_from_config", lambda _config: model)
    skills = load_skills([skill_root])
    result = run_flow(load_flow(FLOW), {"change": "x"}, skills=skills)
    assert result.status == "succeeded", result.failure
    granted = [skills["code-review"], skills["release-notes"]]
    assert model.calls[0][0].content == system_prompt_with_skills(DEFAULT_SYSTEM, granted)


def test_load_skill_runs_through_a_session_conversation(monkeypatch, skill_root: Path) -> None:
    model = _ScriptedModel([_call({"name": "code-review"}, "a"), AIMessage(content="done")])
    monkeypatch.setattr(llm_clients, "model_from_config", lambda _config: model)
    memory = SessionMemory()
    result = run_flow(
        load_flow(SESSION_FLOW), {"change": "x"}, memory={"session": memory}, skills=load_skills([skill_root])
    )
    assert result.status == "succeeded", result.failure
    assert model.calls[1][-1].content == "Read references/checklist.md.\n"
    view = _view(memory)
    assert [effect.kind for effect in view.effects] == ["model", "tool", "model"]
    assert all(effect.status == "completed" for effect in view.effects)


def test_missing_skill_is_rejected_before_any_record(tmp_path: Path) -> None:
    log = tmp_path / "run.jsonl"
    with pytest.raises(ValueError, match=r"AGENT node 'review' grants skills not supplied to this run: \['release-notes'\]"):
        run_flow(load_flow(FLOW), {"change": "x"}, log=log, skills={"code-review": Skill("code-review", "d", "")})
    assert not log.exists()
    with pytest.raises(ValueError, match=r"'outer/inner' grants skills not supplied"):
        run_flow(load_flow(NESTED), {}, skills={})


@pytest.mark.parametrize("text", [FLOW, ALL_FLOW, NESTED], ids=["listed", "all", "nested"])
def test_a_granting_flow_needs_a_skills_mapping(tmp_path: Path, text: str) -> None:
    log = tmp_path / "run.jsonl"
    with pytest.raises(ValueError, match=r"grants skills, but no skills= mapping was passed"):
        run_flow(load_flow(text), {"change": "x"} if text != NESTED else {}, log=log)
    assert not log.exists()


def test_skills_all_with_an_empty_mapping_grants_none(monkeypatch) -> None:
    model = _ScriptedModel([AIMessage(content="ok")])
    monkeypatch.setattr(llm_clients, "model_from_config", lambda _config: model)
    assert run_flow(load_flow(ALL_FLOW), {"change": "x"}, skills={}).status == "succeeded"
    assert model.calls[0][0].content == DEFAULT_SYSTEM


def test_resume_of_a_skills_all_flow_needs_the_mapping(monkeypatch, tmp_path: Path, skill_root: Path) -> None:
    log = tmp_path / "run.jsonl"
    monkeypatch.setattr(llm_clients, "model_from_config", lambda _config: _ScriptedModel([_SimulatedCrash()]))
    with pytest.raises(_SimulatedCrash):
        run_flow(load_flow(ALL_FLOW), {"change": "x"}, log=log, skills=load_skills([skill_root]))
    before = log.read_bytes()
    with pytest.raises(ValueError, match="no skills= mapping was passed"):
        resume_flow(load_flow(ALL_FLOW), log=log)
    assert log.read_bytes() == before
    model = _ScriptedModel([AIMessage(content="ok")])
    monkeypatch.setattr(llm_clients, "model_from_config", lambda _config: model)
    assert resume_flow(load_flow(ALL_FLOW), log=log, skills=load_skills([skill_root])).status == "succeeded"
    assert "- unused: Not granted." in model.calls[0][0].content


def test_skills_mapping_must_be_keyed_by_skill_name() -> None:
    with pytest.raises(TypeError, match=r"skills\['alias'\] must be a Skill named 'alias'"):
        run_flow(load_flow(FLOW), {"change": "x"}, skills={"alias": Skill("code-review", "d", "")})


def test_end_to_end_listing_load_escape_and_resume(monkeypatch, tmp_path: Path, skill_root: Path) -> None:
    skills = load_skills([skill_root])
    log = tmp_path / "run.jsonl"
    first = _ScriptedModel(
        [
            _call({"name": "code-review"}, "a"),
            _call({"name": "code-review", "path": "references/checklist.md"}, "b"),
            _SimulatedCrash("process lost"),
        ]
    )
    monkeypatch.setattr(llm_clients, "model_from_config", lambda _config: first)
    with pytest.raises(_SimulatedCrash):
        run_flow(load_flow(FLOW), {"change": "diff-1"}, log=log, run_id="skills-e2e", skills=skills)

    system = first.calls[0][0].content
    assert "- code-review: Review a diff for bugs.\n- release-notes: Write release notes." in system
    assert "unused" not in system
    assert first.calls[1][-1].content == "Read references/checklist.md.\n"
    assert first.calls[2][-1].content == "1. Tests pass.\n"

    before = log.read_bytes()
    with pytest.raises(ValueError, match="no skills= mapping was passed"):
        resume_flow(load_flow(FLOW), log=log)
    assert log.read_bytes() == before

    second = _ScriptedModel(
        [
            _call({"name": "code-review"}, "a"),
            _call({"name": "code-review", "path": "references/checklist.md"}, "b"),
            AIMessage(content="looks good"),
        ]
    )
    monkeypatch.setattr(llm_clients, "model_from_config", lambda _config: second)
    resumed = resume_flow(load_flow(FLOW), log=log, skills=load_skills([skill_root]))

    assert resumed.status == "succeeded", resumed.failure
    assert resumed.output == "looks good"
    assert [message.content for message in second.calls[2]] == [
        message.content for message in first.calls[2]
    ]
    with JsonlRunLog.resume(log) as reopened:
        records = [envelope.record for envelope in reopened]
    assert sum(isinstance(r, NodeStartedRecord) and r.node_id == "review" for r in records) == 2
    assert sum(isinstance(r, NodeSucceededRecord) and r.node_id == "review" for r in records) == 1

    escape = _ScriptedModel([_call({"name": "code-review", "path": "../../secret.txt"}, "c")])
    monkeypatch.setattr(llm_clients, "model_from_config", lambda _config: escape)
    failed = run_flow(load_flow(FLOW), {"change": "diff-2"}, skills=skills)
    assert failed.status == "failed"
    assert "relative path inside the skill directory" in failed.failure.message
    assert "do not leak" not in failed.failure.message
