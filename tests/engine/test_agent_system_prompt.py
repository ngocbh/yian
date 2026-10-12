"""An authored AGENT `system_prompt:` replaces DEFAULT_SYSTEM on every model path.

A capturing model stands in for the provider (no API keys). Skills and memory
sections still append after the effective system prompt, and the run log keeps
the rendered prompt the model saw for recovery and backward learning.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage
import pytest

import yian.llm_clients as llm_clients
from yian.compose import load_flow
from yian.compose.helpers.types import read_variables
from yian.memory import MemoryContext, MemorySection
from yian.nodes.agent import AgentNode
from yian.nodes.agent.memory import system_prompt_with_memory
from yian.nodes.agent.context import DEFAULT_SYSTEM
from yian.nodes.base import Output
from yian.nodes.contracts import Parameter, ValueContract
from yian.run import resume_flow, run_flow
from yian.runtime.eval_node import EvaluationFailure, eval_node
from yian.runtime.log import NodeSucceededRecord
from yian.skills import Skill
from yian.skills.tool import system_prompt_with_skills
from yian.tools import ToolSet
from yian.typesys import Type, ValueKind, VariablePool
from tests.engine.test_agent_memory_recall import _CaptureModel, _keys, _opener, _scope
from tests.engine.test_run_backward import _records

TEMPLATE = "You are ${name}, ${title}. Speak as ${name} in the first person."
RENDERED = "You are Ada, CTO. Speak as Ada in the first person."


def _node(*, structured: bool = False, **kwargs) -> AgentNode:
    """Build one AGENT whose system prompt and prompt read bound inputs."""
    kwargs.setdefault("system_prompt", TEMPLATE)
    node = AgentNode("agent", prompt="Hello from ${name}", **kwargs)
    node.params = [Parameter(name="name", required=True), Parameter(name="title", required=True)]
    if structured:
        node.output_contract = ValueContract(Type.scalar(ValueKind.INTEGER))
    return node


def _eval(node: AgentNode, **kwargs):
    """Evaluate `node` with `name=Ada` and `title=CTO` bound from the pool."""
    pool = VariablePool()
    pool.set("person", {"name": "Ada", "title": "CTO"})
    flow = SimpleNamespace(
        wiring={"agent": {"name": "${person.output.name}", "title": "${person.output.title}"}}
    )
    return eval_node(node, flow, pool, **kwargs)


def _model(monkeypatch) -> _CaptureModel:
    model = _CaptureModel()
    monkeypatch.setattr(llm_clients, "model_from_config", lambda _config: model)
    return model


@pytest.mark.parametrize(
    ("structured", "tools"),
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_every_path_sends_the_rendered_system_prompt(monkeypatch, structured, tools) -> None:
    model = _model(monkeypatch)
    monkeypatch.setattr(ToolSet, "resolve", lambda _self, _names: [object()])
    node = _node(structured=structured, tools=["noop"] if tools else None)

    evaluated = _eval(node, capture_evidence=True)

    assert evaluated.outcome == Output(7 if structured else "done")
    assert model.calls
    assert [call[0].content for call in model.calls] == [RENDERED] * len(model.calls)
    assert model.calls[0][1] == HumanMessage(content="Hello from Ada")
    assert evaluated.evidence.base_system_prompt == RENDERED


def test_system_prompt_reads_the_node_variables(monkeypatch) -> None:
    model = _model(monkeypatch)
    variables = read_variables({"tone": "str = warm"}, {})
    _eval(_node(system_prompt="${name}, be ${tone}.", variables=variables))
    assert model.calls[0][0].content == "Ada, be warm."


def test_memory_sections_append_after_the_authored_prompt(monkeypatch) -> None:
    model = _model(monkeypatch)
    node = _node()
    node.memory_scope = _scope()
    flow_key, agent_key = _keys()
    sections = (MemorySection(flow_key, "outer facts"), MemorySection(agent_key, "local facts"))
    recalled: list[str] = []

    def recall(_scope, context):
        recalled.append(context.base_system_prompt)
        return sections

    evaluated = _eval(node, capture_evidence=True, memory_context_opener=_opener(recall))

    assert evaluated.outcome == Output("done")
    assert recalled == [RENDERED]
    assert model.calls[0][0].content == system_prompt_with_memory(RENDERED, sections)
    assert model.calls[0][0].content.startswith(RENDERED + "\n\nMemory context")
    assert evaluated.evidence.base_system_prompt == RENDERED
    assert evaluated.evidence.memory_sections == sections


def test_skills_append_after_the_authored_prompt_and_recall_skips_them(monkeypatch) -> None:
    model = _model(monkeypatch)
    skill = Skill("review", "Review code changes.", "Read the diff.")
    seen: list[str] = []

    def open_context(system: str, prompt: str, compatibility: dict) -> MemoryContext:
        seen.append(system)
        return MemoryContext()

    node = AgentNode(
        "agent", prompt="p", system_prompt="You review code.", skills=["review"]
    )
    assert node.forward({}, open_memory_context=open_context, skills={"review": skill}) == Output("done")
    assert seen == ["You review code."]
    assert model.calls[0][0].content == system_prompt_with_skills("You review code.", [skill])


@pytest.mark.parametrize("tools", [False, True])
def test_omitted_system_prompt_keeps_default_system(monkeypatch, tools) -> None:
    model = _model(monkeypatch)
    monkeypatch.setattr(ToolSet, "resolve", lambda _self, _names: [object()])
    seen: list[str] = []

    def open_context(system: str, prompt: str, compatibility: dict) -> MemoryContext:
        seen.append(system)
        return MemoryContext()

    node = AgentNode("agent", prompt="p", tools=["noop"] if tools else None)
    assert node.forward({}, open_memory_context=open_context) == Output("done")
    assert seen == [DEFAULT_SYSTEM]
    assert model.calls[0][0].content == DEFAULT_SYSTEM


def test_a_null_system_prompt_value_fails_the_node_before_any_model_call(monkeypatch) -> None:
    model = _model(monkeypatch)
    node = AgentNode("agent", prompt="p", system_prompt="You are ${name}.")
    node.params = [Parameter(name="name", required=True)]
    pool = VariablePool()
    pool.set("person", None)

    evaluated = eval_node(node, SimpleNamespace(wiring={"agent": {"name": "${person.output}"}}), pool)

    assert type(evaluated.outcome) is EvaluationFailure
    assert model.calls == []


# --- through YAML, the run log, and recovery ------------------------------------ #

_CHAT = """\
id: chat
name: chat
llm_config: {provider: anthropic, model: claude-test}
input:
  name: str
  title: str
  message: str
nodes:
  reply:
    kind: agent
    input: {name: "${input.name}", title: "${input.title}", message: "${input.message}"}
    system_prompt: "SYSTEM"
    prompt: ${message}
    output: str
  after:
    kind: agent
    input: {draft: "${reply.output}"}
    prompt: "polish: ${draft}"
    output: str
output: ${after.output}
"""
_INPUTS = {"name": "Ada", "title": "CTO", "message": "How are we doing?"}


class _SimulatedCrash(BaseException):
    """Process loss that bypasses ordinary node-failure conversion."""


class _Scripted:
    """Record each request and serve scripted replies; a callable reply runs."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls: list[list] = []

    def bind_tools(self, _tools, **_kwargs):
        return self

    def invoke(self, messages):
        self.calls.append(list(messages))
        answer = self.replies.pop(0)
        return answer(messages) if callable(answer) else answer


def _scripted(monkeypatch, replies) -> _Scripted:
    model = _Scripted(replies)
    monkeypatch.setattr(llm_clients, "model_from_config", lambda _config: model)
    return model


def _crash(_messages):
    raise _SimulatedCrash("process lost")


def _chat_flow(template: str = TEMPLATE):
    return load_flow(_CHAT.replace('"SYSTEM"', json.dumps(template)))


def _reply_evidence(path):
    """Return the recorded evidence of the `reply` node's success."""
    (record,) = [
        record for record in _records(path)
        if type(record) is NodeSucceededRecord and record.node_id == "reply"
    ]
    return record.evidence


def test_run_flow_sends_the_rendered_persona(monkeypatch) -> None:
    model = _scripted(monkeypatch, [AIMessage(content="draft"), AIMessage(content="final")])
    result = run_flow(_chat_flow(), _INPUTS)
    assert result.status == "succeeded" and result.output == "final"
    assert model.calls[0][0].content == RENDERED
    assert model.calls[1][0].content == DEFAULT_SYSTEM


def test_resume_replays_a_finished_node_with_its_recorded_system_prompt(monkeypatch, tmp_path) -> None:
    path = tmp_path / "run.jsonl"
    flow = _chat_flow()
    _scripted(monkeypatch, [AIMessage(content="draft"), _crash])
    with pytest.raises(_SimulatedCrash):
        run_flow(flow, _INPUTS, log=path, learnable=True)

    resumed_model = _scripted(monkeypatch, [AIMessage(content="final")])
    resumed = resume_flow(flow, log=path)

    assert resumed.status == "succeeded" and resumed.output == "final"
    assert [call[-1].content for call in resumed_model.calls] == ["polish: draft"]
    assert _reply_evidence(path).base_system_prompt == RENDERED

    # Backward on the resumed run reads the system prompt recorded before the crash.
    backward = _scripted(monkeypatch, [
        AIMessage(content=json.dumps(
            {"input_feedback": {"draft": ["Too stiff."]}, "self_feedback": [], "variable_feedback": {}}
        )),
        AIMessage(content=json.dumps({"input_feedback": {}, "self_feedback": [], "variable_feedback": {}})),
    ])
    resumed.backward("Warmer, please.")
    material = json.loads(backward.calls[1][1].content)
    assert material["forward"]["node_id"] == "reply"
    assert material["forward"]["evaluation_evidence"]["base_system_prompt"] == RENDERED


def test_retry_after_a_mid_node_crash_renders_the_same_system_prompt(monkeypatch, tmp_path) -> None:
    path = tmp_path / "run.jsonl"
    flow = _chat_flow()
    first = _scripted(monkeypatch, [_crash])
    with pytest.raises(_SimulatedCrash):
        run_flow(flow, _INPUTS, log=path, learnable=True)
    assert first.calls[0][0].content == RENDERED

    retry = _scripted(monkeypatch, [AIMessage(content="draft"), AIMessage(content="final")])
    resumed = resume_flow(flow, log=path)

    assert resumed.status == "succeeded" and resumed.output == "final"
    assert retry.calls[0][0].content == RENDERED
    assert _reply_evidence(path).base_system_prompt == RENDERED


def test_resume_refuses_a_flow_whose_system_prompt_changed(monkeypatch, tmp_path) -> None:
    path = tmp_path / "run.jsonl"
    _scripted(monkeypatch, [AIMessage(content="draft"), _crash])
    with pytest.raises(_SimulatedCrash):
        run_flow(_chat_flow(), _INPUTS, log=path)

    _scripted(monkeypatch, [])
    with pytest.raises(ValueError, match="flow fingerprint mismatch"):
        resume_flow(_chat_flow("You are ${name}."), log=path)
