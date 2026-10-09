"""API summaries change active context without replacing retained history."""

from copy import deepcopy
import json

from langchain_core.messages import AIMessage
import pytest

import yian.llm_clients as llm_clients
from yian import load_flow, run_flow
from yian.memory.session import SessionMemory
from yian.run import resume_flow
from tests.engine.test_agent_session import _Model

FLOW = """id: summarized-session
name: summarized_session
llm_config: {provider: openai, model: old-model}
input: {question: str}
nodes:
  answer:
    kind: agent
    mode: MODE
    long_term_memory: session
    input: {question: "${input.question}"}
    prompt: ${question}
    output: str
output: ${answer.output}
"""
SUMMARY = "Retained summary of the earliest conversation. " * 3
QUESTIONS = [f"prior question {index}: " + "x" * 30 for index in range(5)]


def _snapshot(memory):
    with memory.open_run() as opened:
        return opened.snapshot


def _replies():
    return [AIMessage(content=f"prior answer {index}") for index in range(5)]


def _seed(flow, memory):
    for index, question in enumerate(QUESTIONS):
        result = run_flow(flow, {"question": question}, memory={"session": memory})
        assert result.output == f"prior answer {index}"
        assert result.learning is None


@pytest.mark.parametrize("mode", ["plain", "tool_calling"])
def test_soft_budget_keeps_four_prior_turns_and_all_raw_messages(monkeypatch, mode):
    flow = load_flow(FLOW.replace("MODE", mode))
    memory = SessionMemory(history_budget_chars=1, recent_turns=4)
    model = _Model(
        [*_replies(), AIMessage(content=SUMMARY), AIMessage(content="current answer")]
    )
    monkeypatch.setattr(llm_clients, "model_from_config", lambda _: model)
    _seed(flow, memory)
    assert len(model.calls) == 5  # No eligible older turn yet, despite the budget.
    before = deepcopy(_snapshot(memory)["conversation"]["items"])

    result = run_flow(
        flow, {"question": "current question"}, memory={"session": memory}
    )

    assert result.output == "current answer" and result.learning is None
    assert len(model.calls) == 7  # One oversized summary, without a retry, then answer.
    summary_input = "\n".join(str(message.content) for message in model.calls[5])
    assert QUESTIONS[0] in summary_input
    assert "prior answer 0" in summary_input
    assert all(question not in summary_input for question in QUESTIONS[1:])
    assert "current question" not in summary_input
    answer_context = [message.content for message in model.calls[6]]
    assert any(SUMMARY in str(content) for content in answer_context)
    assert QUESTIONS[0] not in answer_context
    assert "prior answer 0" not in answer_context
    for index in range(1, 5):
        assert QUESTIONS[index] in answer_context
        assert f"prior answer {index}" in answer_context
    assert answer_context[-1] == "current question"
    snapshot = _snapshot(memory)
    items = snapshot["conversation"]["items"]
    assert items[: len(before)] == before
    assert len(items) == 18
    assert SUMMARY not in json.dumps(items)
    assert snapshot["notes"] == []


@pytest.mark.parametrize("mode", ["plain", "tool_calling"])
def test_summary_failure_is_a_node_failure_without_answer_or_history_deletion(
    monkeypatch, mode
):
    flow = load_flow(FLOW.replace("MODE", mode))
    memory = SessionMemory(history_budget_chars=1)
    error = RuntimeError("summary request failed")
    model = _Model([*_replies(), error])
    monkeypatch.setattr(llm_clients, "model_from_config", lambda _: model)
    _seed(flow, memory)
    before = _snapshot(memory)["conversation"]

    failed = run_flow(
        flow, {"question": "current question"}, memory={"session": memory}
    )

    assert failed.failure.code == "node.execute"
    assert len(model.calls) == 6
    after = _snapshot(memory)["conversation"]
    assert after["items"] == before["items"]
    assert after["effects"] == before["effects"]
    assert after["closed_invocations"][-1]["status"] == "failed"


def test_new_summary_uses_current_model_without_a_second_model_factory(monkeypatch):
    flow = load_flow(FLOW.replace("MODE", "plain"))
    memory = SessionMemory(history_budget_chars=1)
    old = _Model(_replies())
    current = _Model([AIMessage(content=SUMMARY), AIMessage(content="new answer")])
    factories = []

    def factory(config):
        factories.append(deepcopy(config))
        return current if config["model"] == "new-model" else old

    monkeypatch.setattr(llm_clients, "model_from_config", factory)
    _seed(flow, memory)
    changed = load_flow(FLOW.replace("MODE", "plain").replace("old-model", "new-model"))
    result = run_flow(changed, {"question": "new question"}, memory={"session": memory})

    assert result.output == "new answer"
    assert len(factories) == 6 and factories[-1]["model"] == "new-model"
    assert len(old.calls) == 5 and len(current.calls) == 2
    assert QUESTIONS[0] in "\n".join(
        str(message.content) for message in current.calls[0]
    )


def test_interrupted_summary_can_resume_against_existing_memory(monkeypatch, tmp_path):
    flow = load_flow(FLOW.replace("MODE", "plain"))
    memory = SessionMemory(history_budget_chars=1)
    stopped = KeyboardInterrupt("interrupted summary")
    model = _Model(
        [
            *_replies(),
            stopped,
            AIMessage(content=SUMMARY),
            AIMessage(content="recovered"),
        ]
    )
    monkeypatch.setattr(llm_clients, "model_from_config", lambda _: model)
    _seed(flow, memory)
    before = _snapshot(memory)["conversation"]["items"]
    log = tmp_path / "interrupted.jsonl"
    with pytest.raises(KeyboardInterrupt) as caught:
        run_flow(
            flow, {"question": "recover this"}, memory={"session": memory}, log=log
        )
    assert caught.value is stopped
    assert _snapshot(memory)["conversation"]["items"] == before

    resumed = resume_flow(flow, log=log, memory={"session": memory})

    assert resumed.output == "recovered" and len(model.calls) == 8
    assert _snapshot(memory)["conversation"]["items"][: len(before)] == before
    assert model.calls[-1][-1].content == "recover this"
