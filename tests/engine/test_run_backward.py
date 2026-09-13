"""Explicit public backward attempts over retained forward run authority."""

import pytest

from agent_composer.compose import load_flow
from agent_composer.memory import MemoryOperationError
from agent_composer.run import resume_flow, run_flow
from agent_composer.runtime.log import (
    BackwardFinishedRecord, BackwardReadyRecord, BackwardStartedRecord, JsonlRunLog,
    MemoryCommitRequestedRecord, MemoryCommitSucceededRecord,
)
from tests.engine._memory_fakes import _Memory


_IDENTITY = """id: identity
name: identity
input:
  text: str
nodes: {}
output: ${input.text}
"""
_WITH_MEMORY = _IDENTITY.replace("name: identity", "name: identity\nlong_term_memory: test")


def _records(path):
    with JsonlRunLog.resume(path) as log:
        return tuple(envelope.record for envelope in log)


@pytest.mark.parametrize("durable", [False, True])
def test_backward_without_memory_is_explicit_and_single_success(tmp_path, durable):
    result = run_flow(load_flow(_IDENTITY), {"text": "hello"}, learn=True,
                      log=tmp_path / "run.jsonl" if durable else None)
    assert result.backward_outcome is None
    assert result.learning is None
    before = result.input, result.status, result.output, result.failure, list(result.events)
    outcome = result.backward(feedback="Too long.")
    assert outcome.status == "succeeded"
    assert outcome.feedback.input_feedback == {"text": ("Too long.",)}
    assert result.backward_outcome == outcome and result.learning is None
    assert before == (result.input, result.status, result.output, result.failure, result.events)
    with pytest.raises(ValueError):
        result.backward(feedback="Again.")
    assert result.backward_outcome == outcome


def test_forward_returns_before_any_learning_or_commit(tmp_path):
    path = tmp_path / "run.jsonl"
    memory = _Memory()
    result = run_flow(load_flow(_WITH_MEMORY), {"text": "hello"}, learn=True,
                      log=path, memory={"test": memory})
    assert memory.learn_contexts == memory.commit_calls == []
    assert memory.close_count == 1
    assert result.learning is result.backward_outcome is None
    forward = _records(path)
    assert not any(isinstance(record, BackwardStartedRecord) for record in forward)
    result.backward("Review.")
    assert len(memory.learn_contexts) == len(memory.commit_calls) == 1
    assert len(memory.learn_contexts[0].episodes) == 2
    assert result.learning.status == "succeeded"
    assert _records(path)[:len(forward)] == forward
    assert tuple(type(record) for record in _records(path)[len(forward):]) == (
        BackwardStartedRecord, BackwardReadyRecord, MemoryCommitRequestedRecord,
        MemoryCommitSucceededRecord, BackwardFinishedRecord,
    )


def test_invalid_feedback_does_not_change_latest_outcome(tmp_path):
    path = tmp_path / "retry.jsonl"
    memory = _Memory(learn_outcomes=[MemoryOperationError("learner", "declined", "Try again.", True), []])
    result = run_flow(load_flow(_WITH_MEMORY), {"text": "hello"}, learn=True,
                      log=path, memory={"test": memory})
    previous = result.backward("Review.")
    assert previous.status == "failed" and previous.feedback.input_feedback == {"text": ("Review.",)}
    prefix = _records(path)
    with pytest.raises(ValueError):
        result.backward([None])
    assert result.backward_outcome == previous and result.learning == previous.learning
    assert _records(path) == prefix
    assert len(memory.learn_contexts) == 1 and memory.commit_calls == []
    assert result.backward("Corrected.").status == "succeeded"
    assert [record.attempt_id for record in _records(path) if isinstance(record, BackwardStartedRecord)] == [1, 2]


@pytest.mark.parametrize("previous_failure", [False, True])
def test_invalid_feedback_on_stale_handle_preserves_its_observation(tmp_path, previous_failure):
    path = tmp_path / "stale.jsonl"
    flow = load_flow(_WITH_MEMORY if previous_failure else _IDENTITY)
    backend = _Memory(learn_outcomes=[MemoryOperationError("learner", "declined", "Try again.", True), []])
    memory = {"test": backend} if previous_failure else None
    first = run_flow(flow, {"text": "hello"}, learn=True, log=path, memory=memory)
    if previous_failure:
        assert first.backward("First review.").status == "failed"
    previous, previous_learning = first.backward_outcome, first.learning
    second = resume_flow(flow, log=path, memory=memory)
    assert second.backward("New review.").status == "succeeded"
    prefix = path.read_bytes()
    learned, committed = len(backend.learn_contexts), len(backend.commit_calls)

    with pytest.raises(ValueError):
        first.backward([None])

    assert first.backward_outcome is previous
    assert first.learning is previous_learning
    assert path.read_bytes() == prefix
    assert (len(backend.learn_contexts), len(backend.commit_calls)) == (learned, committed)


def test_stale_handle_publishes_its_new_accepted_prefix_on_host_failure(tmp_path):
    path = tmp_path / "accepted.jsonl"
    flow = load_flow(_WITH_MEMORY)
    failure = MemoryError("learner allocation failed")
    backend = _Memory(learn_outcomes=[
        MemoryOperationError("learner", "declined", "Try again.", True), failure,
    ])
    memory = {"test": backend}
    first = run_flow(flow, {"text": "hello"}, learn=True, log=path, memory=memory)
    second = resume_flow(flow, log=path, memory=memory)
    assert second.backward("First review.").status == "failed"
    assert first.backward_outcome is first.learning is None

    with pytest.raises(MemoryError) as caught:
        first.backward("New review.")

    assert caught.value is failure
    assert first.backward_outcome.failure.code == "backward.interrupted"
    assert first.learning.status == "incomplete"
    last = _records(path)[-1]
    assert type(last) is BackwardStartedRecord and last.attempt_id == 2
    inspected = resume_flow(flow, log=path)
    assert first.backward_outcome == inspected.backward_outcome
    assert first.learning == inspected.learning


@pytest.mark.parametrize("other_finishes", [False, True])
def test_resumed_interrupted_observation_keeps_identity_on_invalid_feedback(tmp_path, other_finishes):
    path = tmp_path / "interrupted.jsonl"
    flow = load_flow(_WITH_MEMORY)
    backend = _Memory(learn_outcomes=[MemoryError("interrupted learner"), []])
    memory = {"test": backend}
    original = run_flow(flow, {"text": "hello"}, learn=True, log=path, memory=memory)
    with pytest.raises(MemoryError):
        original.backward("First review.")
    resumed = resume_flow(flow, log=path, memory=memory)
    previous, previous_learning = resumed.backward_outcome, resumed.learning
    assert previous.failure.code == "backward.interrupted"
    assert previous_learning.status == "incomplete"
    if other_finishes:
        other = resume_flow(flow, log=path, memory=memory)
        assert other.backward("New review.").status == "succeeded"
    prefix = path.read_bytes()
    learned, committed = len(backend.learn_contexts), len(backend.commit_calls)

    with pytest.raises(ValueError):
        resumed.backward([None])

    assert resumed.backward_outcome is previous
    assert resumed.learning is previous_learning
    assert path.read_bytes() == prefix
    assert (len(backend.learn_contexts), len(backend.commit_calls)) == (learned, committed)


def test_nonretained_result_rejects_backward():
    result = run_flow(load_flow(_IDENTITY), {"text": "hello"})
    with pytest.raises(ValueError, match="learn=True"):
        result.backward("Review.")
    assert result.learning is result.backward_outcome is None


def test_failed_forward_and_successful_backward_have_separate_statuses():
    flow = load_flow("""id: failure
name: failure
nodes:
  work:
    kind: code
    code: |
      if True:
          raise RuntimeError('forward failed')
      return 0
output: ${work.output}
""")
    result = run_flow(flow, {}, learn=True)
    assert result.status == "failed"
    failure, events = result.failure, list(result.events)
    assert result.backward("Review failure.").status == "succeeded"
    assert result.status == "failed" and result.failure == failure
    assert result.output is None and result.events == events


def test_terminal_resume_only_reconstructs_backward_outcome(tmp_path):
    path = tmp_path / "run.jsonl"
    flow = load_flow(_WITH_MEMORY)
    memory = _Memory()
    result = run_flow(flow, {"text": "hello"}, learn=True, log=path, memory={"test": memory})
    before = _records(path)
    inspected = resume_flow(flow, log=path)
    assert inspected.backward_outcome is inspected.learning is None
    assert _records(path) == before and memory.open_snapshots == [None]
    outcome = result.backward("Review.")
    before = _records(path)
    inspected = resume_flow(flow, log=path, memory={"test": object()})
    assert inspected.backward_outcome == outcome and inspected.learning == outcome.learning
    assert inspected.events == [] and inspected.output == "hello"
    assert _records(path) == before and memory.open_snapshots == [None, {"namespace": "test"}]


@pytest.mark.parametrize("prefix", ["started", "ready", "requested", "confirmed"])
def test_terminal_resume_inspects_every_unfinished_backward_prefix(tmp_path, prefix):
    from agent_composer.run import _replay_run_log
    from agent_composer.runtime.backward import compute_backward
    from agent_composer.runtime.contracts import FrameProposal
    from agent_composer.runtime.learning import memory_commit_id
    from agent_composer.runtime.state import commit_prepared, prepare_record

    path = tmp_path / "prefix.jsonl"
    flow = load_flow(_WITH_MEMORY)
    memory = _Memory()
    run_flow(flow, {"text": "hello"}, learn=True, log=path, memory={"test": memory})
    with JsonlRunLog.resume(path) as log:
        state = _replay_run_log(flow, log)
        root, attributed = compute_backward(state.trace, "Review.")
        commit_prepared(log, state, prepare_record(state, BackwardStartedRecord(1, "Review.")))
        if prefix != "started":
            proposals = tuple(FrameProposal(spec.ref, ()) for spec in state.memory_layout)
            commit_prepared(log, state, prepare_record(state, BackwardReadyRecord(1, root, attributed, proposals)))
        if prefix in ("requested", "confirmed"):
            identity = memory_commit_id(state.run_id, "test", ())
            commit_prepared(log, state, prepare_record(state, MemoryCommitRequestedRecord("test", identity, (), attempt_id=1)))
            if prefix == "confirmed":
                commit_prepared(log, state, prepare_record(state, MemoryCommitSucceededRecord("test", identity, attempt_id=1)))
        expected = state.backward_state.outcome
    before = path.read_bytes()
    inspected = resume_flow(flow, log=path)
    assert inspected.backward_outcome == expected
    assert inspected.backward_outcome.failure.code == "backward.interrupted"
    assert inspected.learning.status == "incomplete"
    assert inspected.learning.policies[0].status == {
        "started": "unrequested", "ready": "unrequested", "requested": "unknown", "confirmed": "succeeded",
    }[prefix]
    assert inspected.events == [] and path.read_bytes() == before
    assert memory.open_snapshots == [None] and memory.learn_contexts == memory.commit_calls == []
