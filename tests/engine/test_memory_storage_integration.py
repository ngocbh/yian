"""Durable backends retain memory through the public run and recovery boundary."""

from copy import deepcopy
import json
import sqlite3
import subprocess
import sys

from langchain_core.messages import AIMessage
import pytest

import agent_composer.llm_clients as llm_clients
from agent_composer.compose import load_flow
from agent_composer.memory.contracts import MemoryOperationError
from agent_composer.memory.episodic import EpisodicMemory
from agent_composer.memory.reflective import ReflectiveMemory
import agent_composer.memory.reflective as reflective
from agent_composer.memory.session import SessionMemory
from agent_composer.run import resume_flow, run_flow

_MODEL_CONFIG = {"provider": "openai", "model": "storage-test-model"}
_CODE_FLOW = """id: durable-learning
name: durable_learning
long_term_memory: POLICY
nodes:
  work:
    kind: code
    code: return 7
    output: int
output: ${work.output}
"""
_AGENT_FLOW = """id: durable-session
name: durable_session
llm_config: {provider: openai, model: storage-test-model}
input:
  question: str
nodes:
  agent:
    kind: agent
    mode: plain
    long_term_memory: session
    prompt: ${question}
    input:
      question: ${input.question}
    output: str
output: ${agent.output}
"""


def _backend(policy, storage):
    if policy == "reflective":
        return ReflectiveMemory(llm_config=_MODEL_CONFIG, storage=storage)
    return {"episodic": EpisodicMemory, "session": SessionMemory}[policy](
        storage=storage
    )


def _snapshot(memory):
    with memory.open_run() as opened:
        return opened.snapshot


class _ChatModel:
    def __init__(self):
        self.calls = []

    def invoke(self, messages):
        self.calls.append(deepcopy(messages))
        return AIMessage(content=f"reply {len(self.calls)}")


class _ReflectionModel:
    def with_structured_output(self, schema):
        self.schema = schema
        return self

    def invoke(self, messages):
        return self.schema.model_validate(
            {
                "actions": [
                    {
                        "action": "add",
                        "lesson_id": "checked-output",
                        "text": "Verify the returned output.",
                    }
                ]
            }
        )


@pytest.mark.parametrize("policy", ["episodic", "reflective", "session"])
def test_process_local_backends_never_open_sqlite(policy, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("process-local memory opened SQLite")

    monkeypatch.setattr(sqlite3, "connect", forbidden)
    with _backend(policy, None) as memory:
        _snapshot(memory)


@pytest.mark.parametrize("policy", ["episodic", "reflective", "session"])
def test_snapshot_identity_survives_a_real_process_restart(policy, tmp_path):
    path = tmp_path / "private-memory.sqlite"
    child = """
import json
import sys
from agent_composer.memory.episodic import EpisodicMemory
from agent_composer.memory.reflective import ReflectiveMemory
from agent_composer.memory.session import SessionMemory
policy, path = sys.argv[1:]
factory = {'episodic': EpisodicMemory, 'reflective': ReflectiveMemory,
           'session': SessionMemory}[policy]
options = {'storage': path}
if policy == 'reflective':
    options['llm_config'] = {'provider': 'openai', 'model': 'storage-test-model'}
with factory(**options) as memory:
    with memory.open_run() as opened:
        print(json.dumps(opened.snapshot))
"""
    completed = subprocess.run(
        [sys.executable, "-c", child, policy, str(path)],
        capture_output=True,
        text=True,
        timeout=20,
        check=True,
    )
    recorded = json.loads(completed.stdout)
    with _backend(policy, path) as reopened:
        assert _snapshot(reopened) == recorded
    assert str(path) not in json.dumps(recorded)


@pytest.mark.parametrize("policy", ["episodic", "reflective", "session"])
def test_idle_run_does_not_keep_storage_owned_after_backend_close(policy, tmp_path):
    path = tmp_path / "memory.sqlite"
    with _backend(policy, path) as memory:
        opened = memory.open_run()
        recorded = opened.snapshot
        memory.close()
        with pytest.raises(MemoryOperationError):
            _ = opened.snapshot
        with _backend(policy, path) as reopened:
            assert _snapshot(reopened) == recorded
        opened.close()


@pytest.mark.parametrize("policy", ["episodic", "reflective"])
def test_explicit_learning_remains_saved_after_reopen(policy, tmp_path, monkeypatch):
    monkeypatch.setattr(reflective, "model_from_config", lambda _: _ReflectionModel())
    path = tmp_path / "memory.sqlite"
    log = tmp_path / "run.jsonl"
    flow = load_flow(_CODE_FLOW.replace("POLICY", policy))
    with _backend(policy, path) as memory:
        result = run_flow(flow, {}, memory={policy: memory}, learnable=True, log=log)
        assert result.status == "succeeded" and result.output == 7
        assert _snapshot(memory)["frames"] == []
        result.backward()
        assert result.learning.status == "succeeded"
        recorded = _snapshot(memory)
        assert recorded["frames"]
        field = "episodes" if policy == "episodic" else "lessons"
        assert recorded["frames"][0][field]
    prefix = log.read_bytes()
    with _backend(policy, path) as reopened:
        assert _snapshot(reopened) == recorded
        inspected = resume_flow(flow, log=log, memory={policy: reopened})
        assert inspected.output == 7
        assert _snapshot(reopened) == recorded
    assert log.read_bytes() == prefix
    assert str(path).encode() not in prefix


def test_api_session_history_is_used_after_reopening_storage(tmp_path, monkeypatch):
    model = _ChatModel()
    monkeypatch.setattr(llm_clients, "model_from_config", lambda _: model)
    path = tmp_path / "private-session.sqlite"
    flow = load_flow(_AGENT_FLOW)
    with SessionMemory(storage=path) as memory:
        first = run_flow(flow, {"question": "first"}, memory={"session": memory})
        assert first.output == "reply 1" and first.learning is None
        recorded = _snapshot(memory)
    with SessionMemory(storage=path) as reopened:
        assert _snapshot(reopened) == recorded
        second = run_flow(flow, {"question": "second"}, memory={"session": reopened})
        assert second.output == "reply 2" and second.learning is None
        assert [message.content for message in model.calls[1][1:]] == [
            "first",
            "reply 1",
            "second",
        ]
        assert len(_snapshot(reopened)["conversations"][0]["items"]) == 6


def test_api_session_learned_notes_survive_storage_reopen(tmp_path, monkeypatch):
    from tests.engine.test_agent_session import _Model

    note = "Check the measurement units before answering."
    model = _Model(
        [
            AIMessage(content="first answer"),
            AIMessage(
                content=json.dumps(
                    {
                        "input_feedback": {},
                        "self_feedback": [note],
                        "variable_feedback": {},
                    }
                )
            ),
            AIMessage(content="second answer"),
        ]
    )
    monkeypatch.setattr(llm_clients, "model_from_config", lambda _: model)
    path = tmp_path / "session.sqlite"
    flow = load_flow(_AGENT_FLOW)
    with SessionMemory(storage=path) as memory:
        first = run_flow(
            flow, {"question": "first"}, memory={"session": memory}, learnable=True
        )
        raw = deepcopy(_snapshot(memory)["conversations"])
        first.backward("Remember to check the units.")
        assert first.learning.status == "succeeded"
        recorded = _snapshot(memory)
        assert recorded["conversations"] == raw
        assert len(recorded["notes"]) == 1
    with SessionMemory(storage=path) as reopened:
        assert _snapshot(reopened) == recorded
        second = run_flow(flow, {"question": "second"}, memory={"session": reopened})
        assert second.output == "second answer"
        assert note in model.calls[-1][0].content
        assert len(model.calls) == 3


def test_resume_reopens_latest_saved_history_instead_of_the_old_run_snapshot(
    tmp_path, monkeypatch
):
    from agent_composer.nodes.agent import AgentNode
    from agent_composer.nodes.base import _HostStop
    from tests.engine.test_agent_session import _Model

    stops = []
    original_ctx = AgentNode._ctx

    def capture_ctx(node, *args, **kwargs):
        context = original_ctx(node, *args, **kwargs)
        stops.append(context.stop_event)
        return context

    def finish_after_stop(messages):
        stops[-1].set()
        return AIMessage(content="reply saved before cancellation")

    model = _Model([finish_after_stop, AIMessage(content="retry complete")])
    monkeypatch.setattr(AgentNode, "_ctx", capture_ctx)
    monkeypatch.setattr(llm_clients, "model_from_config", lambda _: model)
    flow = load_flow(_AGENT_FLOW)
    path = tmp_path / "memory.sqlite"
    log = tmp_path / "run.jsonl"
    with SessionMemory(storage=path) as memory:
        with pytest.raises(_HostStop):
            run_flow(
                flow, {"question": "question"}, memory={"session": memory}, log=log
            )
        saved = _snapshot(memory)
        assert saved["conversations"][0]["effects"][0]["status"] == "completed"
    prefix = log.read_bytes()
    with SessionMemory(storage=path) as reopened:
        assert _snapshot(reopened) == saved
        resumed = resume_flow(flow, log=log, memory={"session": reopened})
        assert resumed.output == "retry complete"
        context = [message.content for message in model.calls[-1]]
        assert "reply saved before cancellation" in context
        assert context.count("question") == 2
        effects = _snapshot(reopened)["conversations"][0]["effects"]
        assert len(effects) == 2
        assert effects[0]["invocation"] != effects[1]["invocation"]
    assert log.read_bytes().startswith(prefix)
    assert str(path).encode() not in log.read_bytes()


def test_api_summary_cache_survives_reopen_under_current_budget(tmp_path, monkeypatch):
    from tests.engine.test_agent_session import _Model

    summary = "Saved historical summary"
    model = _Model(
        [AIMessage(content=f"answer {index}") for index in range(5)]
        + [
            AIMessage(content=summary),
            AIMessage(content="answer 5"),
            AIMessage(content="answer 6"),
        ]
    )
    monkeypatch.setattr(llm_clients, "model_from_config", lambda _: model)
    path = tmp_path / "session.sqlite"
    flow = load_flow(_AGENT_FLOW)
    with SessionMemory(storage=path, history_budget_chars=1) as memory:
        for index in range(6):
            result = run_flow(
                flow, {"question": f"question {index}"}, memory={"session": memory}
            )
            assert result.output == f"answer {index}"
        assert len(model.calls) == 7
        raw = _snapshot(memory)["conversations"][0]["items"]
        assert summary not in json.dumps(raw)
    with SessionMemory(storage=path, history_budget_chars=100_000) as reopened:
        result = run_flow(
            flow, {"question": "question 6"}, memory={"session": reopened}
        )
        assert result.output == "answer 6" and len(model.calls) == 8
        context = [message.content for message in model.calls[-1]]
        assert any(summary in content for content in context)
        assert "question 0" not in context
        assert all(f"question {index}" in context for index in range(1, 7))
        assert _snapshot(reopened)["conversations"][0]["items"][: len(raw)] == raw


def test_more_recent_turns_on_reopen_restores_raw_context(tmp_path, monkeypatch):
    from tests.engine.test_agent_session import _Model

    summary = "Previously summarized first turn"
    model = _Model(
        [
            AIMessage(content="answer 0"),
            AIMessage(content=summary),
            AIMessage(content="answer 1"),
            AIMessage(content="answer 2"),
        ]
    )
    monkeypatch.setattr(llm_clients, "model_from_config", lambda _: model)
    path = tmp_path / "session.sqlite"
    flow = load_flow(_AGENT_FLOW)
    with SessionMemory(storage=path, history_budget_chars=1, recent_turns=0) as memory:
        for index in range(2):
            result = run_flow(
                flow, {"question": f"question {index}"}, memory={"session": memory}
            )
            assert result.output == f"answer {index}"
        assert len(model.calls) == 3
    with SessionMemory(
        storage=path, history_budget_chars=100_000, recent_turns=4
    ) as reopened:
        result = run_flow(
            flow, {"question": "question 2"}, memory={"session": reopened}
        )
        assert result.output == "answer 2" and len(model.calls) == 4
        assert [message.content for message in model.calls[-1][1:]] == [
            "question 0",
            "answer 0",
            "question 1",
            "answer 1",
            "question 2",
        ]
