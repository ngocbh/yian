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
