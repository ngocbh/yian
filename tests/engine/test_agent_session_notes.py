"""Explicit API backward feedback becomes conversation-local recalled notes."""

import json

from langchain_core.messages import AIMessage
import pytest

import yian.llm_clients as llm_clients
from yian import load_flow, run_flow
from yian.memory.session import SessionMemory
from tests.engine.test_agent_session import _Model

NOTE = "Check the measurement units before answering."
FLOW = """id: learned-session
name: learned_session
llm_config: {provider: openai, model: test-model}
input: {question: str}
nodes:
  answer:
    kind: agent
    mode: plain
    long_term_memory: session
    input: {question: "${input.question}"}
    prompt: ${question}
    output: str
output: ${answer.output}
"""


def _feedback():
    return AIMessage(
        content=json.dumps(
            {
                "input_feedback": {},
                "self_feedback": [NOTE],
                "variable_feedback": {},
            }
        )
    )


def _snapshot(memory):
    with memory.open_run() as opened:
        return opened.snapshot


def test_backward_reuses_self_feedback_without_an_extra_model_call(monkeypatch):
    source = FLOW
    flow = load_flow(source)
    model = _Model(
        [
            AIMessage(content="first answer"),
            _feedback(),
            AIMessage(content="next answer"),
        ]
    )
    monkeypatch.setattr(llm_clients, "model_from_config", lambda _: model)
    memory = SessionMemory()
    first = run_flow(
        flow, {"question": "first question"}, memory={"session": memory}, learnable=True
    )
    assert first.status == "succeeded"
    history = _snapshot(memory)["conversation"]

    outcome = first.backward("The answer used inconsistent units.")
    assert outcome.status == "succeeded"
    assert (
        len(model.calls) == 2
    )  # Existing forward + backward; no note-generation call.
    assert _snapshot(memory)["conversation"] == history

    second = run_flow(flow, {"question": "next question"}, memory={"session": memory})
    assert second.status == "succeeded" and second.output == "next answer"
    assert len(model.calls) == 3
    assert sum(str(message.content).count(NOTE) for message in model.calls[-1]) == 1
    assert len(_snapshot(memory)["notes"]) == 1


def test_backward_without_feedback_adds_no_notes_or_model_call(monkeypatch):
    model = _Model([AIMessage(content="answer")])
    monkeypatch.setattr(llm_clients, "model_from_config", lambda _: model)
    memory = SessionMemory()
    result = run_flow(
        load_flow(FLOW),
        {"question": "question"},
        memory={"session": memory},
        learnable=True,
    )
    history = _snapshot(memory)["conversation"]
    assert result.backward().status == "succeeded"
    assert len(model.calls) == 1
    assert _snapshot(memory)["conversation"] == history
    assert _snapshot(memory).get("notes", []) == []


def test_forward_without_learning_keeps_history_but_adds_no_notes(monkeypatch):
    model = _Model(
        [AIMessage(content="first answer"), AIMessage(content="next answer")]
    )
    monkeypatch.setattr(llm_clients, "model_from_config", lambda _: model)
    memory = SessionMemory()
    flow = load_flow(FLOW)
    for question in ("first question", "next question"):
        result = run_flow(flow, {"question": question}, memory={"session": memory})
        assert result.status == "succeeded" and result.learning is None
    assert len(model.calls) == 2
    assert "first answer" in [message.content for message in model.calls[-1]]
    assert _snapshot(memory).get("notes", []) == []


def test_learned_notes_do_not_cross_sibling_conversations(monkeypatch):
    flow = load_flow("""id: sibling-notes
name: sibling_notes
llm_config: {provider: openai, model: test-model}
input: {question: str}
nodes:
  left:
    kind: agent
    mode: plain
    long_term_memory: left_chat
    input: {question: "${input.question}"}
    prompt: "left: ${question}"
    output: str
  right:
    kind: agent
    mode: plain
    long_term_memory: right_chat
    input: {question: "${input.question}"}
    prompt: "right: ${question}"
    output: str
output:
  left: ${left.output}
  right: ${right.output}
""")
    model = _Model(
        [
            AIMessage(content="answer"),
            AIMessage(content="answer"),
            _feedback(),
            AIMessage(content="answer"),
            AIMessage(content="answer"),
        ]
    )
    monkeypatch.setattr(llm_clients, "model_from_config", lambda _: model)
    left, right = SessionMemory(), SessionMemory()
    memory = {"left_chat": left, "right_chat": right}
    first = run_flow(flow, {"question": "first question"}, memory=memory, learnable=True)
    assert first.status == "succeeded"
    assert first.backward({"left": "Check units.", "right": None}).status == "succeeded"
    assert len(model.calls) == 3
    assert (
        run_flow(flow, {"question": "next question"}, memory=memory).status
        == "succeeded"
    )
    assert len(model.calls) == 5
    for messages in model.calls[3:]:
        leaf = str(messages[-1].content).split(":", 1)[0]
        assert leaf in {"left", "right"}
        assert any(NOTE in str(message.content) for message in messages) == (
            leaf == "left"
        )
    assert len(_snapshot(left)["notes"]) == 1
    assert _snapshot(right).get("notes", []) == []


def test_admitted_failed_forward_can_learn_without_rewriting_history(monkeypatch):
    model = _Model(
        [
            RuntimeError("provider could not answer"),
            _feedback(),
            AIMessage(content="recovered"),
        ]
    )
    monkeypatch.setattr(llm_clients, "model_from_config", lambda _: model)
    memory = SessionMemory()
    flow = load_flow(FLOW)
    first = run_flow(
        flow, {"question": "first question"}, memory={"session": memory}, learnable=True
    )
    assert first.status == "failed"
    history = _snapshot(memory)["conversation"]
    assert first.backward("Check the input units.").status == "succeeded"
    assert _snapshot(memory)["conversation"] == history
    second = run_flow(flow, {"question": "next question"}, memory={"session": memory})
    assert second.status == "succeeded"
    assert len(model.calls) == 3
    assert any(NOTE in str(message.content) for message in model.calls[-1])
