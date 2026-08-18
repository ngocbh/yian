"""Public fresh/resume orchestration for host-injected memory policies."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from agent_composer.compose import load_flow
from agent_composer.events import NodeExpanded, RunStarted
from agent_composer.memory import (
    LearnContext,
    MemoryFrameSpec,
    MemoryMutation,
    MemoryOperationError,
    MemorySetupError,
    RecallContext,
)
from agent_composer.run import resume_flow, run_flow
from agent_composer.runtime.log import (
    JsonlRunLog,
    MemoryCommitRequestedRecord,
    MemoryCommitSucceededRecord,
    MemoryFrameLearningFailedRecord,
    MemoryFrameLearningStartedRecord,
    RunResumedRecord,
    RunStartedRecord,
)


_FLOW = """id: memory-run
name: memory_run
memory_policy: test
nodes:
  work:
    kind: code
    code: return 7
    output: int
output: ${work.output}
"""

_FAILED_FLOW = """id: memory-run-failed
name: memory_run_failed
memory_policy: test
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
memory_policy: test
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


class _Frame:
    """One fake frame backed by process-local test memory state."""

    def __init__(self, memory: "_Memory", spec: MemoryFrameSpec) -> None:
        self.memory = memory
        self.spec = spec

    def recall(self, context: RecallContext | None = None) -> str:
        self.memory.recall_contexts.append(context)
        return self.memory.recall_text

    def learn(self, context: LearnContext) -> list[MemoryMutation]:
        self.memory.learn_contexts.append(context)
        outcome = (
            self.memory.learn_outcomes.pop(0)
            if self.memory.learn_outcomes
            else []
        )
        if isinstance(outcome, BaseException):
            raise outcome
        if callable(outcome):
            return outcome(context)
        return outcome


class _Session:
    """One immutable-snapshot fake session."""

    def __init__(self, memory: "_Memory", snapshot: dict[str, Any]) -> None:
        self.memory = memory
        self._snapshot = copy.deepcopy(snapshot)
        self.frames: dict[object, _Frame] = {}
        self.closed = False

    @property
    def snapshot(self) -> dict[str, Any]:
        return copy.deepcopy(self._snapshot)

    def open_frame(self, spec: MemoryFrameSpec) -> _Frame:
        frame = _Frame(self.memory, spec)
        self.frames[spec.ref] = frame
        self.memory.opened_specs.append(spec)
        return frame

    def commit(
        self,
        commit_id: str,
        mutations: tuple[MemoryMutation, ...],
    ) -> None:
        self.memory.commit_calls.append((commit_id, mutations))
        outcome = (
            self.memory.commit_outcomes.pop(0)
            if self.memory.commit_outcomes
            else None
        )
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            self.memory.close_count += 1


class _Memory:
    """Observable structural Memory implementation for public-boundary tests."""

    def __init__(
        self,
        *,
        recall_text: str = "",
        learn_outcomes: list[object] | None = None,
        commit_outcomes: list[object] | None = None,
    ) -> None:
        self.recall_text = recall_text
        self.learn_outcomes = list(learn_outcomes or [])
        self.commit_outcomes = list(commit_outcomes or [])
        self.open_snapshots: list[dict[str, Any] | None] = []
        self.opened_specs: list[MemoryFrameSpec] = []
        self.recall_contexts: list[RecallContext | None] = []
        self.learn_contexts: list[LearnContext] = []
        self.commit_calls: list[tuple[str, tuple[MemoryMutation, ...]]] = []
        self.close_count = 0
        self.reset_count = 0

    def open_run(self, *, snapshot=None) -> _Session:
        restored = None if snapshot is None else copy.deepcopy(snapshot)
        self.open_snapshots.append(restored)
        effective = {"namespace": "test"} if restored is None else restored
        return _Session(self, effective)

    def reset(self) -> None:
        self.reset_count += 1


def _records(path: Path):
    """Return the canonical durable records without retaining the file lock."""
    with JsonlRunLog.resume(path) as log:
        return tuple(envelope.record for envelope in log)


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
    assert start.learn is False


def test_learn_true_requires_memory_before_creating_a_log(tmp_path: Path) -> None:
    path = tmp_path / "invalid.jsonl"

    with pytest.raises(MemorySetupError, match="requires"):
        run_flow(load_flow(_FLOW), {}, log=path, learn=True)

    assert not path.exists()


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
    assert start.learn is False
    assert not any(
        isinstance(record, MemoryFrameLearningStartedRecord)
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
        learn=True,
    )

    assert result.status == "succeeded"
    assert result.output == 7
    assert result.learning is not None
    assert result.learning.status == "succeeded"
    assert len(memory.learn_contexts) == 1
    assert memory.learn_contexts[0].episodes == ()
    assert memory.learn_contexts[0].feedback == ()
    assert len(memory.commit_calls) == 1
    assert memory.commit_calls[0][1] == ()
    records = _records(path)
    assert isinstance(records[-2], MemoryCommitRequestedRecord)
    assert isinstance(records[-1], MemoryCommitSucceededRecord)


def test_learning_failure_does_not_replace_primary_success(tmp_path: Path) -> None:
    path = tmp_path / "learn-failed.jsonl"
    memory = _Memory(
        learn_outcomes=[RuntimeError("credential=must-not-leak")],
    )

    result = run_flow(
        load_flow(_FLOW),
        {},
        log=path,
        memory={"test": memory},
        learn=True,
    )

    assert result.status == "succeeded"
    assert result.output == 7
    assert result.failure is None
    assert result.learning is not None
    assert result.learning.status == "failed"
    assert len(result.learning.failures) == 1
    assert result.learning.failures[0].operation == "learn"
    assert "must-not-leak" not in result.learning.failures[0].failure.message
    assert any(
        isinstance(record, MemoryFrameLearningFailedRecord)
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
        learn=True,
    )

    assert result.status == "failed"
    assert result.failure is not None
    assert result.failure.code == "node.execute"
    assert result.learning is not None
    assert result.learning.status == "succeeded"
    assert len(memory.learn_contexts) == 1
    assert len(memory.commit_calls) == 1


def test_retryable_learning_resumes_without_rerunning_settled_primary(
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
        learn=True,
    )
    primary_prefix = len(_records(path))

    assert first.status == "succeeded"
    assert first.learning is not None
    assert first.learning.status == "incomplete"
    resumed = resume_flow(flow, log=path, memory={"test": memory})

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


def test_learning_only_resume_uses_static_layout_after_map_growth(
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
        learn=True,
    )
    resumed = resume_flow(flow, log=path, memory={"test": memory})

    assert first.status == "succeeded"
    assert first.output == [2, 5]
    assert first.learning is not None
    assert first.learning.status == "incomplete"
    assert resumed.status == "succeeded"
    assert resumed.output == [2, 5]
    assert resumed.events == []
    assert resumed.learning is not None
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
