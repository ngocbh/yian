"""Public fresh/resume orchestration for host-injected memory policies."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import importlib

import pytest

from agent_composer.compose import load_flow
from agent_composer.events import NodeExpanded, NodeStarted, RunStarted
from agent_composer.memory import (
    ConversationItem,
    DeclSite,
    MemoryOperationError,
    MemorySetupError,
    MemorySpec,
    MemoryStack,
)
from agent_composer.memory.session import SessionMemory
from agent_composer.nodes.agent.node import AgentNode
from agent_composer.nodes.base import Output
from agent_composer.run import resume_flow, run_flow
from agent_composer.runtime.log import (
    JsonlRunLog,
    MemoryCommitRequestedRecord,
    MemoryCommitSucceededRecord,
    BackwardFinishedRecord,
    BackwardStartedRecord,
    RunResumedRecord,
    RunStartedRecord,
    decode_record,
    encode_record,
)

from tests.engine._memory_fakes import _Memory


_FLOW = """id: memory-run
name: memory_run
long_term_memory: test
nodes:
  work:
    kind: code
    code: return 7
    output: int
output: ${work.output}
"""

_FAILED_FLOW = """id: memory-run-failed
name: memory_run_failed
long_term_memory: test
nodes:
  work:
    kind: code
    code: |
      if True:
          raise RuntimeError("boom")
      return 0
    output: int
output: ${work.output}
"""

_MAP_FLOW = """id: memory-map
name: memory_map
long_term_memory: test
input:
  values: list[int]
defs:
  increment:
    input:
      value: int
    nodes: {}
    output: ${input.value + 1}
nodes:
  each:
    kind: map
    call: increment
    over: ${input.values}
    input:
      value: ${item}
output: ${each.output}
"""

_AGENT_FLOW = """id: memory-run
name: memory_run
llm_config: {provider: openai, model: fake-model}
nodes:
  work:
    kind: agent
    mode: plain
    long_term_memory: test
    prompt: hi
    output: int
output: ${work.output}
"""


def _records(path: Path):
    """Return the canonical durable records without retaining the file lock."""
    with JsonlRunLog.resume(path) as log:
        return tuple(envelope.record for envelope in log)


@pytest.mark.parametrize("resume", [False, True])
@pytest.mark.parametrize("num_workers", [0, 2])
@pytest.mark.parametrize("learnable", [False, True])
def test_memory_context_opener_uses_same_runtime_on_fresh_and_resume(
    tmp_path, monkeypatch, resume, num_workers, learnable,
):
    path = tmp_path / "conversation.jsonl"
    memory = SessionMemory()
    entered = []
    engines = []
    boundary = importlib.import_module("agent_composer.run")
    original_engine = boundary.FlowEngine

    def engine(*args, **kwargs):
        assert "memory_recall" not in kwargs
        assert kwargs["memory_context_opener"].__name__ == "open_context"
        engines.append(kwargs["memory_context_opener"].__self__)
        return original_engine(*args, **kwargs)

    def forward(node, inputs, *, open_memory_context, **_caps):
        turn = open_memory_context("system", "prompt", {"format": "test-v1"}).transcript
        entered.append(turn.view.key.consumer_static_path)
        ordinal = turn.request("model", {"input": "test"})
        turn.observe(ordinal, "completed", items=(ConversationItem("reply", "test-v1", {"value": 7}),))
        return Output(7)

    monkeypatch.setattr(AgentNode, "forward", forward)
    monkeypatch.setattr(boundary, "FlowEngine", engine)
    flow = load_flow(_AGENT_FLOW)
    if resume:
        def interrupt(event):
            if isinstance(event, NodeStarted) and event.node_id == "work":
                raise RuntimeError("interrupt before body")
        with pytest.raises(RuntimeError, match="interrupt before body"):
            run_flow(flow, {}, log=path, memory={"test": memory},
                     num_workers=num_workers, on_event=interrupt, learnable=learnable)
        assert entered == []
        before = path.read_bytes()
        result = resume_flow(flow, log=path, memory={"test": memory})
        assert path.read_bytes().startswith(before)
    else:
        result = run_flow(flow, {}, log=path, memory={"test": memory},
                          num_workers=num_workers, learnable=learnable)
    assert result.output == 7
    assert entered == [("memory-run", "work")]
    assert len(engines) == (2 if resume else 1)
    assert all(runtime._closed for runtime in engines)
    assert not any(type(record).__name__.startswith("Conversation") for record in _records(path))
    with memory.open_run() as opened:
        histories = opened.snapshot["conversations"]
    assert len(histories) == 1
    assert histories[0]["closed_invocations"][-1]["status"] == "completed"
    assert histories[0]["items"][0]["payload"] == {"value": 7}


@pytest.mark.parametrize("num_workers", [0, 2])
def test_conversation_completes_before_dispatcher_output_contract_acceptance(
    tmp_path, monkeypatch, num_workers,
):
    memory = SessionMemory()
    def forward(node, inputs, *, open_memory_context, **_caps):
        open_memory_context("system", "prompt", {}).transcript
        return Output("not an integer")
    monkeypatch.setattr(AgentNode, "forward", forward)
    result = run_flow(load_flow(_AGENT_FLOW), {}, log=tmp_path / "rejected.jsonl",
                      memory={"test": memory}, num_workers=num_workers)
    assert result.status == "failed"
    assert result.failure.code == "contract.output"
    with memory.open_run() as opened:
        history = opened.snapshot["conversations"][0]
    assert history["closed_invocations"][-1]["status"] == "completed"


def test_authored_policy_is_dormant_when_memory_is_none(tmp_path: Path) -> None:
    path = tmp_path / "dormant.jsonl"

    result = run_flow(load_flow(_FLOW), {}, log=path)

    assert result.status == "succeeded"
    assert result.output == 7
    assert result.learning is None
    start = _records(path)[0]
    assert isinstance(start, RunStartedRecord)
    assert start.memory_layout is None
    assert start.memory_snapshots is None
    assert start.learnable is False


def test_learnable_true_retains_without_enabling_an_authored_policy(tmp_path: Path) -> None:
    path = tmp_path / "retained.jsonl"
    result = run_flow(load_flow(_FLOW), {}, log=path, learnable=True)
    assert result.learning is None and result.backward_outcome is None
    assert _records(path)[0].memory_layout is None
    assert result.backward().learning is None


def test_recall_only_records_snapshot_without_evidence_or_learning(
    tmp_path: Path,
) -> None:
    path = tmp_path / "recall-only.jsonl"
    memory = _Memory()

    result = run_flow(
        load_flow(_FLOW),
        {},
        log=path,
        memory={"test": memory},
    )

    assert result.status == "succeeded"
    assert result.learning is None
    assert len(memory.opened_specs) == 1
    assert memory.learn_contexts == []
    assert memory.commit_calls == []
    assert memory.close_count == 1
    records = _records(path)
    start = records[0]
    assert isinstance(start, RunStartedRecord)
    assert start.memory_layout is not None
    assert len(start.memory_layout) == 1
    assert start.memory_snapshots is not None
    assert start.learnable is False
    assert not any(
        isinstance(record, BackwardStartedRecord)
        for record in records
    )


def test_learning_success_is_separate_from_the_primary_result(tmp_path: Path) -> None:
    path = tmp_path / "learn.jsonl"
    memory = _Memory()

    result = run_flow(
        load_flow(_FLOW),
        {},
        log=path,
        memory={"test": memory},
        learnable=True,
    )

    assert result.status == "succeeded"
    assert result.output == 7
    assert result.learning is None
    result.backward()
    assert result.learning is not None
    assert result.learning.status == "succeeded"
    assert len(memory.learn_contexts) == 1
    assert len(memory.learn_contexts[0].episodes) == 3
    assert memory.learn_contexts[0].feedback == ()
    assert len(memory.commit_calls) == 1
    assert memory.commit_calls[0][1] == ()
    records = _records(path)
    assert isinstance(records[-3], MemoryCommitRequestedRecord)
    assert isinstance(records[-2], MemoryCommitSucceededRecord)
    assert isinstance(records[-1], BackwardFinishedRecord)


def test_learning_failure_does_not_replace_primary_success(tmp_path: Path) -> None:
    path = tmp_path / "learn-failed.jsonl"
    memory = _Memory(
        learn_outcomes=[MemoryOperationError("learner", "declined", "Cannot learn.", True)],
    )

    result = run_flow(
        load_flow(_FLOW),
        {},
        log=path,
        memory={"test": memory},
        learnable=True,
    )

    assert result.status == "succeeded"
    assert result.output == 7
    assert result.failure is None
    outcome = result.backward()
    assert outcome.status == "failed" and outcome.failure.stage == "learn"
    assert result.learning is not None
    assert result.learning.status == "failed"
    assert len(result.learning.failures) == 1
    assert result.learning.failures[0].operation == "learn"
    assert result.learning.failures[0].failure.message == "Cannot learn."
    assert any(
        isinstance(record, BackwardFinishedRecord)
        for record in _records(path)
    )


def test_primary_failure_still_learns_without_changing_its_failure(tmp_path: Path) -> None:
    path = tmp_path / "primary-failed.jsonl"
    memory = _Memory()

    result = run_flow(
        load_flow(_FAILED_FLOW),
        {},
        log=path,
        memory={"test": memory},
        learnable=True,
    )

    assert result.status == "failed"
    assert result.failure is not None
    assert result.failure.code == "node.execute"
    result.backward("Review the failure.")
    assert result.learning is not None
    assert result.learning.status == "succeeded"
    assert len(memory.learn_contexts) == 1
    assert len(memory.commit_calls) == 1


def test_precommit_failure_requires_an_explicit_whole_pass_retry(
    tmp_path: Path,
) -> None:
    path = tmp_path / "retry.jsonl"
    retryable = MemoryOperationError(
        "learn",
        "test.busy",
        "try later",
        True,
    )
    memory = _Memory(learn_outcomes=[retryable, []])
    flow = load_flow(_FLOW)

    first = run_flow(
        flow,
        {},
        log=path,
        memory={"test": memory},
        learnable=True,
    )
    first.backward()
    primary_prefix = len(_records(path))

    assert first.status == "succeeded"
    assert first.learning is not None
    assert first.learning.status == "failed"
    resumed = resume_flow(flow, log=path, memory={"test": memory})
    assert resumed.learning == first.learning
    assert len(_records(path)) == primary_prefix and len(memory.learn_contexts) == 1
    resumed.backward()

    assert resumed.status == "succeeded"
    assert resumed.output == 7
    assert resumed.events == []
    assert resumed.learning is not None
    assert resumed.learning.status == "succeeded"
    assert len(memory.learn_contexts) == 2
    suffix = _records(path)[primary_prefix:]
    assert not any(isinstance(record, RunResumedRecord) for record in suffix)

    settled_count = len(_records(path))
    silent = resume_flow(
        flow,
        log=path,
        memory={"test": object()},  # ignored: primary and learning are settled
    )
    assert silent.learning is not None
    assert silent.learning.status == "succeeded"
    assert len(_records(path)) == settled_count


def test_explicit_retry_uses_static_layout_after_map_growth(
    tmp_path: Path,
) -> None:
    path = tmp_path / "map-learning.jsonl"
    retryable = MemoryOperationError("learn", "test.busy", "later", True)
    memory = _Memory(learn_outcomes=[retryable, []])
    flow = load_flow(_MAP_FLOW)

    first = run_flow(
        flow,
        {"values": [1, 4]},
        log=path,
        memory={"test": memory},
        learnable=True,
    )
    first.backward()
    resumed = resume_flow(flow, log=path, memory={"test": memory})

    assert first.status == "succeeded"
    assert first.output == [2, 5]
    assert first.learning is not None
    assert first.learning.status == "failed"
    assert resumed.status == "succeeded"
    assert resumed.output == [2, 5]
    assert resumed.events == []
    assert resumed.learning is not None
    assert resumed.learning == first.learning
    resumed.backward()
    assert resumed.learning.status == "succeeded"


def test_unfinished_primary_resume_uses_static_layout_after_map_growth(
    tmp_path: Path,
) -> None:
    path = tmp_path / "map-primary.jsonl"
    memory = _Memory()
    flow = load_flow(_MAP_FLOW)

    def stop_after_growth(event: object) -> None:
        if isinstance(event, NodeExpanded):
            raise RuntimeError("host stop after map growth")

    with pytest.raises(RuntimeError, match="host stop after map growth"):
        run_flow(
            flow,
            {"values": [1, 4]},
            log=path,
            memory={"test": memory},
            on_event=stop_after_growth,
        )

    resumed = resume_flow(flow, log=path, memory={"test": memory})

    assert resumed.status == "succeeded"
    assert resumed.output == [2, 5]


def test_unfinished_primary_requires_recorded_memory_mapping_on_resume(
    tmp_path: Path,
) -> None:
    path = tmp_path / "unfinished.jsonl"
    memory = _Memory()

    def stop_after_start(event: object) -> None:
        if isinstance(event, RunStarted):
            raise RuntimeError("host stop")

    with pytest.raises(RuntimeError, match="host stop"):
        run_flow(
            load_flow(_FLOW),
            {},
            log=path,
            memory={"test": memory},
            on_event=stop_after_start,
        )

    with pytest.raises(MemorySetupError, match="requires"):
        resume_flow(load_flow(_FLOW), log=path)

    resumed = resume_flow(
        load_flow(_FLOW),
        log=path,
        memory={"test": memory},
    )
    assert resumed.status == "succeeded"
    assert resumed.output == 7
    assert memory.open_snapshots == [None, {"namespace": "test"}]


def test_run_started_records_the_resolved_memory_table_and_stacks(tmp_path: Path) -> None:
    path = tmp_path / "table.jsonl"

    result = run_flow(load_flow(_FLOW), {}, log=path, memory={"test": _Memory()})

    assert result.status == "succeeded"
    start = _records(path)[0]
    assert start.memories == (
        MemorySpec("test", "test", {}, (DeclSite((), ("root",), None),), False),
    )
    assert ("work",) in {stack.locator for stack in start.memory_stacks}
    assert all(stack.memories == ("test",) for stack in start.memory_stacks)


@pytest.mark.parametrize("tampered", ["stacks", "table"])
def test_replay_rejects_a_recorded_key_table_or_stacks_that_differ_from_the_flow(
    tmp_path: Path, tampered: str
) -> None:
    path = tmp_path / "tampered.jsonl"
    memory = _Memory()

    def stop_after_start(event: object) -> None:
        if isinstance(event, RunStarted):
            raise RuntimeError("host stop")

    with pytest.raises(RuntimeError, match="host stop"):
        run_flow(load_flow(_FLOW), {}, log=path, memory={"test": memory}, on_event=stop_after_start)

    lines = path.read_bytes().splitlines()
    envelope = decode_record(lines[0])
    record = envelope.record
    if tampered == "stacks":
        emptied = tuple(MemoryStack(stack.locator, ()) for stack in record.memory_stacks)
        record = replace(record, memory_stacks=emptied)
    else:
        spec, = record.memories
        changed = MemorySpec(spec.key, spec.type, {"recall_limit": 1}, spec.owners, False)
        record = replace(record, memories=(changed,))
    lines[0] = encode_record(replace(envelope, record=record))
    path.write_bytes(b"\n".join(lines) + b"\n")

    with pytest.raises(ValueError, match="memory table or stacks"):
        resume_flow(load_flow(_FLOW), log=path, memory={"test": memory})
