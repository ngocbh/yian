"""Structured AGENT output survives an `ask_user` pause/resume.

A `tool_calling` agent that declares a non-text `output:` AND pauses on `ask_user` must
still emit the declared type on its resumed final-answer turn — the continuation node must
carry the spawner's `output_type` (and `retries`), or the resumed answer is plain text and
the write boundary rejects it.
"""

from types import SimpleNamespace

from langchain_core.messages import AIMessage

import agent_composer.llm_clients as llm_clients_mod
from agent_composer.compile.expand import build_grow
from agent_composer.compile.model import END_ID, START_ID, CompiledFlow, Edge, FlowOutput
from agent_composer.run import resume_command
from agent_composer.events import RunPaused, RunResumed, RunSucceeded
from agent_composer.llm_clients import LLMConfig
from agent_composer.nodes.agent import AgentNode
from agent_composer.nodes.end import EndNode
from agent_composer.nodes.start import StartNode
from agent_composer.runtime.engine import FlowEngine
from agent_composer.typesys.values import Type, ValueKind
from agent_composer.suspension.pause import HumanInputRequired


def _ai_tool_call(name, args, call_id="1"):
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )


class _StructuredAskChat:
    def __init__(self):
        self._replies = [
            _ai_tool_call("ask_user", {"question": "Approve?"}, call_id="q1"),
            AIMessage(content="FINAL"),
        ]
        self.calls = 0
        self.structured_called = False

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        self.calls += 1
        return self._replies.pop(0)

    def with_structured_output(self, schema):
        self.structured_called = True

        class _Bound:
            def invoke(self, messages):
                return schema.model_validate({"name": "Ada", "score": 9})

        return _Bound()


def _record_type():
    return Type(
        kind=ValueKind.OBJECT,
        fields={
            "name": Type.scalar(ValueKind.STRING),
            "score": Type.scalar(ValueKind.INTEGER),
        },
        required=frozenset({"name", "score"}),
    )


def test_structured_output_survives_ask_user_resume(monkeypatch):
    chat = _StructuredAskChat()
    monkeypatch.setattr(llm_clients_mod, "model_from_config", lambda cfg: chat)

    node = AgentNode(
        "agent", prompt="go", controls=["ask_user"], llm_config=LLMConfig(), mode="tool_calling"
    )
    node.output_type = _record_type()
    graph = CompiledFlow.from_parts(
        {
            "agent": node,
            START_ID: StartNode(START_ID, input_decls=[]),
            END_ID: EndNode.record(END_ID, output_names=["output"]),
        },
        [
            Edge("e0", START_ID, "agent"),
            Edge("agent->__end__#0", "agent", END_ID, input_group="output"),
        ],
        outputs=[FlowOutput(name="output", from_="${agent.output}")],
        wiring={END_ID: {"output": "${agent.output}"}},
    )
    eng = FlowEngine(graph, grow_builder=build_grow)
    evs = list(eng.run())
    assert isinstance(evs[-1], RunPaused)
    reason = evs[-1].reasons[0]
    assert isinstance(reason, HumanInputRequired)

    cmd = resume_command(SimpleNamespace(compiled=graph), reason, "yes")
    evs2 = list(eng.resume(commands=[cmd]))
    assert isinstance(evs2[0], RunResumed)
    assert isinstance(evs2[-1], RunSucceeded)
    # the resumed final answer is the declared record, validated at the write boundary
    assert evs2[-1].output == {"name": "Ada", "score": 9}
    assert chat.structured_called  # the resumed final turn went through structured generation
