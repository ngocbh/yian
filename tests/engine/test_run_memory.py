"""Public fresh/resume orchestration for host-injected memory policies."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_composer.compose import load_flow
from agent_composer.events import NodeExpanded, RunStarted
from agent_composer.memory import (
    MemoryOperationError,
    MemorySetupError,
)
from agent_composer.run import resume_flow, run_flow
from agent_composer.runtime.log import (
    JsonlRunLog,
    MemoryCommitRequestedRecord,
    MemoryCommitSucceededRecord,
    BackwardFinishedRecord,
    BackwardStartedRecord,
    RunResumedRecord,
    RunStartedRecord,
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


def test_learn_true_retains_without_enabling_an_authored_policy(tmp_path: Path) -> None:
    path = tmp_path / "retained.jsonl"
    result = run_flow(load_flow(_FLOW), {}, log=path, learn=True)
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
    assert start.learn is False
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
        learn=True,
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
        learn=True,
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
        learn=True,
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
        learn=True,
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
        learn=True,
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
