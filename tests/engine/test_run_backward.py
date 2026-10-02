"""Explicit public backward attempts over retained forward run authority."""

from asyncio import CancelledError as AsyncCancelledError
from concurrent.futures import CancelledError as FutureCancelledError, ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import pytest

from agent_composer.compose import load_flow
from agent_composer.events import RunStarted, RunSucceeded
from agent_composer.memory import MemoryOperationError, MemorySetupError
from agent_composer.run import resume_flow, run_flow
from agent_composer.runtime.log import (
    BackwardFinishedRecord, BackwardReadyRecord, BackwardStartedRecord, JsonlRunLog,
    MemoryCommitRequestedRecord, MemoryCommitSucceededRecord, MemoryRunLog,
)
from tests.engine._memory_fakes import _Memory, _Session


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
def test_backward_without_memory_is_explicit_and_repeatable(tmp_path, durable):
    result = run_flow(load_flow(_IDENTITY), {"text": "hello"}, learnable=True,
                      log=tmp_path / "run.jsonl" if durable else None)
    assert result.backward_outcome is None
    assert result.learning is None
    before = result.input, result.status, result.output, result.failure, list(result.events)
    outcome = result.backward(feedback="Too long.")
    assert outcome.status == "succeeded"
    assert outcome.feedback.input_feedback == {"text": ("Too long.",)}
    assert result.backward_outcome == outcome and result.learning is None
    assert before == (result.input, result.status, result.output, result.failure, result.events)
    again = result.backward(feedback="Again.")
    assert again.status == "succeeded"
    assert again.feedback.input_feedback == {"text": ("Again.",)}
    assert result.backward_outcome == again and result.learning is None
    if durable:
        starts = [r.attempt_id for r in _records(tmp_path / "run.jsonl") if type(r) is BackwardStartedRecord]
        assert starts == [1, 2]


def test_forward_returns_before_any_learning_or_commit(tmp_path):
    path = tmp_path / "run.jsonl"
    memory = _Memory()
    result = run_flow(load_flow(_WITH_MEMORY), {"text": "hello"}, learnable=True,
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


def test_same_handle_rejects_nested_backward_before_a_second_commit():
    rejected = []

    def learn(context):
        try:
            result.backward("Nested review.")
        except RuntimeError as exc:
            rejected.append(exc)
        return []

    backend = _Memory(learn_outcomes=[learn])
    result = run_flow(load_flow(_WITH_MEMORY), {"text": "hello"}, learnable=True,
                      memory={"test": backend})

    assert result.backward("Outer review.").status == "succeeded"

    assert len(backend.commit_calls) == 1
    assert len(backend.learn_contexts) == 1
    assert len(rejected) == 1 and "in progress" in str(rejected[0])
    assert backend.close_count == 2
    assert result.backward("Later review.").status == "succeeded"
    assert len(backend.commit_calls) == 2 and backend.close_count == 3


@pytest.mark.parametrize("durable", [False, True])
def test_same_handle_rejects_concurrent_backward_without_waiting(tmp_path, durable):
    entered, release = threading.Event(), threading.Event()

    def learn(context):
        entered.set()
        assert release.wait(timeout=5)
        return []

    backend = _Memory(learn_outcomes=[learn])
    result = run_flow(load_flow(_WITH_MEMORY), {"text": "hello"}, learnable=True,
                      log=tmp_path / "run.jsonl" if durable else None, memory={"test": backend})
    with ThreadPoolExecutor(max_workers=1) as workers:
        pending = workers.submit(result.backward, "First review.")
        try:
            assert entered.wait(timeout=5)
            with pytest.raises(RuntimeError, match="in progress"):
                result.backward("Concurrent review.")
            assert len(backend.learn_contexts) == 1 and not backend.commit_calls
        finally:
            release.set()
        assert pending.result(timeout=5).status == "succeeded"
    assert len(backend.commit_calls) == 1 and backend.close_count == 2


@pytest.mark.parametrize("acquisition", ["fresh_event", "fresh_after", "resume_event", "resume_after"])
def test_delayed_backward_anchors_the_log_at_acquisition(tmp_path, monkeypatch, acquisition):
    acquired, elsewhere = tmp_path / "acquired", tmp_path / "elsewhere"
    acquired.mkdir()
    elsewhere.mkdir()
    monkeypatch.chdir(acquired)
    flow = load_flow(_IDENTITY)

    def move_after_forward(event):
        if isinstance(event, RunSucceeded):
            monkeypatch.chdir(elsewhere)

    if acquisition == "resume_event":
        interrupted = RuntimeError("interrupt before forward evaluation")

        def stop_after_start(event):
            if isinstance(event, RunStarted):
                raise interrupted

        with pytest.raises(RuntimeError) as caught:
            run_flow(flow, {"text": "hello"}, learnable=True, log="run.jsonl", on_event=stop_after_start)
        assert caught.value is interrupted
        result = resume_flow(flow, log="run.jsonl", on_event=move_after_forward)
    elif acquisition == "resume_after":
        run_flow(flow, {"text": "hello"}, learnable=True, log="run.jsonl")
        result = resume_flow(flow, log="run.jsonl")
    else:
        result = run_flow(flow, {"text": "hello"}, learnable=True, log="run.jsonl",
                          on_event=move_after_forward if acquisition == "fresh_event" else None)
    monkeypatch.chdir(elsewhere)

    assert result.log == Path("run.jsonl")
    assert result.backward("Review.").feedback.input_feedback == {"text": ("Review.",)}
    assert type(_records(acquired / "run.jsonl")[-1]) is BackwardFinishedRecord
    assert not (elsewhere / "run.jsonl").exists()


@pytest.mark.parametrize("opening", ["create", "resume"])
def test_acquisition_uses_the_captured_locator_if_opener_changes_cwd(tmp_path, monkeypatch, opening):
    acquired, elsewhere = tmp_path / "acquired", tmp_path / "elsewhere"
    acquired.mkdir()
    elsewhere.mkdir()
    monkeypatch.chdir(acquired)
    flow = load_flow(_IDENTITY)
    other_prefix = None
    if opening == "resume":
        run_flow(flow, {"text": "hello"}, learnable=True, log="run.jsonl")
        run_flow(flow, {"text": "other run"}, learnable=True, log=elsewhere / "run.jsonl")
        other_prefix = (elsewhere / "run.jsonl").read_bytes()
    original_open = getattr(JsonlRunLog, opening)

    def change_before_open(cls, path, *args):
        monkeypatch.chdir(elsewhere)
        return original_open(path, *args)

    monkeypatch.setattr(JsonlRunLog, opening, classmethod(change_before_open))
    result = (run_flow(flow, {"text": "hello"}, learnable=True, log="run.jsonl")
              if opening == "create" else resume_flow(flow, log="run.jsonl"))

    assert result.output == "hello" and result.log == Path("run.jsonl")
    assert result.backward("Review.").feedback.input_feedback == {"text": ("Review.",)}
    assert type(_records(acquired / "run.jsonl")[-1]) is BackwardFinishedRecord
    if other_prefix is None:
        assert not (elsewhere / "run.jsonl").exists()
    else:
        assert (elsewhere / "run.jsonl").read_bytes() == other_prefix


@pytest.mark.parametrize("phase", ["learner", "commit", "finished"])
@pytest.mark.parametrize("durable", [False, True])
def test_prefix_export_failure_preserves_primary_and_prevents_stale_retry(tmp_path, monkeypatch, phase, durable):
    primary = MemoryError("learner failed") if phase == "learner" else RuntimeError("backend failed")
    export_error = OSError("prefix export failed")
    close_error = OSError("log close failed")
    backend = _Memory(learn_outcomes=[primary] if phase == "learner" else [],
                      commit_outcomes=[primary] if phase == "commit" else [])
    flow = load_flow(_WITH_MEMORY)
    path = tmp_path / "export.jsonl" if durable else None
    result = run_flow(flow, {"text": "hello"}, learnable=True, log=path, memory={"test": backend})
    log_type = JsonlRunLog if durable else MemoryRunLog
    original_iter, original_close = log_type.__iter__, log_type.close
    closed = []

    def export(log):
        if any(type(envelope.record) is BackwardStartedRecord for envelope in log._records):
            raise export_error
        return original_iter(log)

    def close(log):
        original_close(log)
        closed.append(log)
        raise close_error

    monkeypatch.setattr(log_type, "__iter__", export)
    monkeypatch.setattr(log_type, "close", close)
    with pytest.raises(BaseException) as caught:
        result.backward("Review.")

    assert caught.value is (export_error if phase == "finished" else primary)
    assert len(closed) == 1 and closed[0]._closed
    assert backend.close_count == 2
    assert result.backward_outcome is not None
    assert result.backward_outcome.status == ("succeeded" if phase == "finished" else "failed")
    if phase == "commit":
        assert result.learning.policies[0].status == "unknown"
    learned, committed = len(backend.learn_contexts), len(backend.commit_calls)
    monkeypatch.setattr(log_type, "__iter__", original_iter)
    monkeypatch.setattr(log_type, "close", original_close)

    with pytest.raises(RuntimeError, match="unavailable"):
        result.backward("Do not retry stale history.")

    assert (len(backend.learn_contexts), len(backend.commit_calls)) == (learned, committed)
    if durable:
        assert resume_flow(flow, log=path).backward_outcome == result.backward_outcome


@pytest.mark.parametrize("record_type, disposition", [
    (BackwardStartedRecord, "unrequested"),
    (MemoryCommitRequestedRecord, "unknown"),
    (MemoryCommitSucceededRecord, "succeeded"),
    (BackwardFinishedRecord, "finished"),
])
@pytest.mark.parametrize("durable", [False, True])
def test_accepted_append_error_refreshes_the_actual_prefix(tmp_path, monkeypatch, record_type, disposition, durable):
    backend = _Memory()
    path = tmp_path / "accepted.jsonl" if durable else None
    flow = load_flow(_WITH_MEMORY)
    result = run_flow(flow, {"text": "hello"}, learnable=True, log=path, memory={"test": backend})
    log_type = JsonlRunLog if durable else MemoryRunLog
    original_append = log_type.append
    failure = OSError("append accepted before returning")
    fail_once = True

    def append(log, record):
        nonlocal fail_once
        envelope = original_append(log, record)
        if fail_once and type(record) is record_type:
            fail_once = False
            raise failure
        return envelope

    monkeypatch.setattr(log_type, "append", append)
    with pytest.raises(OSError) as caught:
        result.backward("Review.")

    assert caught.value is failure
    assert result.backward_outcome is not None
    if disposition == "finished":
        assert result.backward_outcome.status == result.learning.status == "succeeded"
    else:
        assert result.backward_outcome.failure.code == "backward.interrupted"
        assert result.learning.status == "incomplete"
        assert result.learning.policies[0].status == disposition
    if durable:
        inspected = resume_flow(flow, log=path)
        assert result.backward_outcome == inspected.backward_outcome
    if disposition == "unrequested":
        assert backend.learn_contexts == backend.commit_calls == []
        assert result.backward("Fresh whole pass.").status == "succeeded"
        assert len(backend.learn_contexts) == len(backend.commit_calls) == 1
    else:
        learned, committed = len(backend.learn_contexts), len(backend.commit_calls)
        assert result.backward("Later whole pass.").status == "succeeded"
        assert (len(backend.learn_contexts), len(backend.commit_calls)) == (learned + 1, committed + 1)


def test_each_explicit_attempt_restores_fresh_sessions_from_recorded_snapshots():
    backend = _Memory(learn_outcomes=[MemoryOperationError("learner", "declined", "Try again.", True), []])
    replacement = _Memory()
    mapping = {"test": backend}
    result = run_flow(load_flow(_WITH_MEMORY), {"text": "hello"}, learnable=True, memory=mapping)
    assert len(backend.sessions) == 1 and backend.sessions[0].closed
    backend.sessions[0]._snapshot["namespace"] = "changed after forward"
    mapping["test"] = replacement

    assert result.backward("First review.").status == "failed"
    assert result.backward("Fresh review.").status == "succeeded"

    assert backend.open_snapshots == [None, {"namespace": "test"}, {"namespace": "test"}]
    assert len({id(session) for session in backend.sessions}) == 3
    assert all(session.closed for session in backend.sessions)
    assert len(backend.learn_contexts) == 2 and len(backend.commit_calls) == 1
    assert replacement.sessions == []


def test_terminal_resume_retains_mapping_references_without_opening_them(tmp_path):
    path = tmp_path / "terminal.jsonl"
    flow = load_flow(_WITH_MEMORY)
    original = _Memory()
    run_flow(flow, {"text": "hello"}, learnable=True, log=path, memory={"test": original})
    missing = resume_flow(flow, log=path)
    prefix = path.read_bytes()
    with pytest.raises(MemorySetupError):
        missing.backward("Review.")
    assert path.read_bytes() == prefix and missing.backward_outcome is None
    backend, replacement = _Memory(), _Memory()
    mapping = {"test": backend}
    inspected = resume_flow(flow, log=path, memory=mapping)
    assert backend.sessions == [] and path.read_bytes() == prefix
    mapping["test"] = replacement

    assert inspected.backward("Review.").status == "succeeded"

    assert backend.open_snapshots == [{"namespace": "test"}]
    assert len(backend.sessions) == 1 and backend.sessions[0].closed
    assert replacement.sessions == [] and original.learn_contexts == []


def test_failed_snapshot_restoration_keeps_interrupted_prefix_and_observation(tmp_path, monkeypatch):
    path = tmp_path / "restore.jsonl"
    backend = _Memory(learn_outcomes=[MemoryError("interrupted learner"), []])
    result = run_flow(load_flow(_WITH_MEMORY), {"text": "hello"}, learnable=True,
                      log=path, memory={"test": backend})
    with pytest.raises(MemoryError):
        result.backward("First review.")
    previous, learning, prefix = result.backward_outcome, result.learning, path.read_bytes()
    original_open = backend.open_run

    def mismatched_open(*, snapshot=None):
        session = original_open(snapshot=snapshot)
        session._snapshot["namespace"] = "wrong snapshot"
        return session

    monkeypatch.setattr(backend, "open_run", mismatched_open)
    with pytest.raises(MemorySetupError):
        result.backward("Review after restoration.")

    assert result.backward_outcome is previous and result.learning is learning
    assert path.read_bytes() == prefix and backend.sessions[-1].closed
    assert len(backend.learn_contexts) == 1 and backend.commit_calls == []
    monkeypatch.setattr(backend, "open_run", original_open)
    assert result.backward("Fresh review.").status == "succeeded"
    assert [record.attempt_id for record in _records(path) if type(record) is BackwardStartedRecord] == [1, 2]


@pytest.mark.parametrize("durable", [False, True])
def test_delayed_backward_uses_private_program_and_canonical_values(tmp_path, durable):
    flow = load_flow(_IDENTITY.replace("text: str", "text: object"))
    supplied = {"text": {"items": [1, 2]}}
    result = run_flow(flow, supplied, learnable=True, log=tmp_path / "private.jsonl" if durable else None)
    supplied["text"]["items"].append(3)
    assert result.input == {"text": {"items": [1, 2]}}
    result.input["text"]["items"].append(4)
    result.output["items"].append(5)
    result.run_id = "changed public identity"
    result.log = tmp_path / "changed-public-path.jsonl"
    flow.compiled.wiring[flow.compiled.end_id].clear()
    flow.compiled.nodes.clear()

    outcome = result.backward({"items": ["Review first.", None]})

    assert outcome.status == "succeeded"
    assert outcome.feedback.input_feedback == {"text": ({"items": ["Review first.", None]},)}
    assert result.output == {"items": [1, 2, 5]}
    assert result.input == {"text": {"items": [1, 2, 4]}}
    assert not result.log.exists()


def test_distinct_durable_handles_keep_file_lock_exclusion(tmp_path):
    path = tmp_path / "locked.jsonl"
    rejected = []

    def learn(context):
        try:
            second.backward("Nested on another handle.")
        except BlockingIOError as exc:
            rejected.append(exc)
        return []

    flow = load_flow(_WITH_MEMORY)
    backend = _Memory(learn_outcomes=[learn])
    first = run_flow(flow, {"text": "hello"}, learnable=True, log=path, memory={"test": backend})
    second = resume_flow(flow, log=path, memory={"test": backend})

    assert first.backward("Review.").status == "succeeded"

    assert len(rejected) == len(backend.learn_contexts) == len(backend.commit_calls) == 1
    assert second.backward_outcome is None
    assert resume_flow(flow, log=path).backward_outcome == first.backward_outcome


@pytest.mark.parametrize("phase", ["export", "close"])
def test_handle_guard_covers_prefix_export_and_log_cleanup(monkeypatch, phase):
    backend = _Memory()
    result = run_flow(load_flow(_WITH_MEMORY), {"text": "hello"}, learnable=True, memory={"test": backend})
    method = "__iter__" if phase == "export" else "close"
    original = getattr(MemoryRunLog, method)
    rejected = []

    def guarded(log):
        if any(type(envelope.record) is BackwardStartedRecord for envelope in log._records):
            try:
                result.backward("Nested during cleanup.")
            except RuntimeError as exc:
                rejected.append(exc)
        return original(log)

    monkeypatch.setattr(MemoryRunLog, method, guarded)
    assert result.backward("Review.").status == "succeeded"
    assert len(rejected) == 1 and "in progress" in str(rejected[0])
    assert len(backend.learn_contexts) == len(backend.commit_calls) == 1


@pytest.mark.parametrize("resource", ["session", "log"])
@pytest.mark.parametrize("learner_fails", [False, True])
def test_cleanup_error_preserves_finished_authority_and_releases_guard(monkeypatch, resource, learner_fails):
    backend = _Memory(learn_outcomes=[MemoryOperationError("learner", "declined", "Try again.", True), []]
                      if learner_fails else [])
    result = run_flow(load_flow(_WITH_MEMORY), {"text": "hello"}, learnable=True, memory={"test": backend})
    owner = _Session if resource == "session" else MemoryRunLog
    original_close = owner.close
    failure = OSError("cleanup failed")

    def close(resource):
        original_close(resource)
        raise failure

    monkeypatch.setattr(owner, "close", close)
    with pytest.raises(OSError) as caught:
        result.backward("Review.")
    assert caught.value is failure
    assert result.backward_outcome.status == ("failed" if learner_fails else "succeeded")
    assert all(session.closed for session in backend.sessions)
    monkeypatch.setattr(owner, "close", original_close)

    assert result.backward("Fresh review.").status == "succeeded"
    assert len(backend.learn_contexts) == 2
    assert len(backend.commit_calls) == (1 if learner_fails else 2)


@pytest.mark.parametrize("failure", [AsyncCancelledError("cancelled"), FutureCancelledError("cancelled")])
def test_cancelled_backward_retains_started_and_releases_resources(failure):
    backend = _Memory(learn_outcomes=[failure, []])
    result = run_flow(load_flow(_WITH_MEMORY), {"text": "hello"}, learnable=True, memory={"test": backend})

    with pytest.raises(type(failure)) as caught:
        result.backward("Interrupted review.")

    assert caught.value is failure
    assert result.backward_outcome.failure.code == "backward.interrupted"
    assert result.backward_outcome.feedback is None
    assert all(session.closed for session in backend.sessions)
    assert backend.commit_calls == []
    assert result.backward("Fresh whole pass.").status == "succeeded"
    assert len(backend.learn_contexts) == 2 and len(backend.commit_calls) == 1


def test_invalid_feedback_does_not_change_latest_outcome(tmp_path):
    path = tmp_path / "retry.jsonl"
    memory = _Memory(learn_outcomes=[MemoryOperationError("learner", "declined", "Try again.", True), []])
    result = run_flow(load_flow(_WITH_MEMORY), {"text": "hello"}, learnable=True,
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
    first = run_flow(flow, {"text": "hello"}, learnable=True, log=path, memory=memory)
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
    first = run_flow(flow, {"text": "hello"}, learnable=True, log=path, memory=memory)
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
    original = run_flow(flow, {"text": "hello"}, learnable=True, log=path, memory=memory)
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
    with pytest.raises(ValueError, match="learnable=True"):
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
    result = run_flow(flow, {}, learnable=True)
    assert result.status == "failed"
    failure, events = result.failure, list(result.events)
    assert result.backward("Review failure.").status == "succeeded"
    assert result.status == "failed" and result.failure == failure
    assert result.output is None and result.events == events


def test_terminal_resume_only_reconstructs_backward_outcome(tmp_path):
    path = tmp_path / "run.jsonl"
    flow = load_flow(_WITH_MEMORY)
    memory = _Memory()
    result = run_flow(flow, {"text": "hello"}, learnable=True, log=path, memory={"test": memory})
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
    from agent_composer.runtime.contracts import MemoryProposal
    from agent_composer.runtime.learning import memory_commit_id
    from agent_composer.runtime.state import commit_prepared, prepare_record

    path = tmp_path / "prefix.jsonl"
    flow = load_flow(_WITH_MEMORY)
    memory = _Memory()
    run_flow(flow, {"text": "hello"}, learnable=True, log=path, memory={"test": memory})
    with JsonlRunLog.resume(path) as log:
        state = _replay_run_log(flow, log)
        root, attributed = compute_backward(state.trace, "Review.")
        commit_prepared(log, state, prepare_record(state, BackwardStartedRecord(1, "Review.")))
        if prefix != "started":
            proposals = tuple(MemoryProposal(spec.key, ()) for spec in state.memories)
            commit_prepared(log, state, prepare_record(state, BackwardReadyRecord(1, root, attributed, proposals)))
        if prefix in ("requested", "confirmed"):
            identity = memory_commit_id(state.run_id, "test", (), 1)
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


_KILLED_ATTEMPT = """
import sys, time
from pathlib import Path
from agent_composer.compose import load_flow
from agent_composer.run import resume_flow
from tests.engine._memory_fakes import _Memory, _Session

log, source, marker, phase = sys.argv[1:]

def block(*args):
    Path(marker).write_text(phase)
    time.sleep(120)

backend = _Memory(learn_outcomes=[lambda context: block()] if phase == "learn" else [])
if phase == "commit":
    _Session.commit = block
resume_flow(load_flow(Path(source).read_text()), log=Path(log), memory={"test": backend}).backward("Second review.")
"""


@pytest.mark.parametrize("phase", ["learn", "commit"])
def test_killed_attempt_does_not_block_a_later_attempt(tmp_path, phase):
    root = Path(__file__).resolve().parents[2]
    path, source, marker = tmp_path / "run.jsonl", tmp_path / "flow.yaml", tmp_path / "blocked"
    source.write_text(_WITH_MEMORY)
    flow = load_flow(_WITH_MEMORY)
    first = _Memory()
    assert run_flow(flow, {"text": "hello"}, learnable=True, log=path,
                    memory={"test": first}).backward("First review.").status == "succeeded"

    env = {**os.environ, "PYTHONPATH": os.pathsep.join((str(root / "src"), str(root)))}
    child = subprocess.Popen([sys.executable, "-c", _KILLED_ATTEMPT, str(path), str(source), str(marker), phase],
                             cwd=root, env=env)
    try:
        deadline = time.monotonic() + 60
        while not marker.exists():
            assert child.poll() is None, "child exited before blocking"
            assert time.monotonic() < deadline, "child never reached the blocking call"
            time.sleep(0.05)
    finally:
        child.kill()
        child.wait()

    third = _Memory()
    resumed = resume_flow(flow, log=path, memory={"test": third})
    projected = resumed.backward_outcome
    assert projected.status == "failed" and projected.failure.code == "backward.interrupted"
    outcome = resumed.backward("Third review.")
    assert outcome.status == "succeeded" and resumed.backward_outcome == outcome
    records = _records(path)
    assert [r.attempt_id for r in records if type(r) is BackwardStartedRecord] == [1, 2, 3]
    finished = {r.attempt_id: r.outcome for r in records if type(r) is BackwardFinishedRecord}
    assert finished[1].status == finished[3].status == "succeeded"
    assert finished[2] == projected
    assert finished[2].learning.policies[0].status == ("unknown" if phase == "commit" else "unrequested")
    requests = [r for r in records if type(r) is MemoryCommitRequestedRecord]
    assert [r.attempt_id for r in requests] == ([1, 2, 3] if phase == "commit" else [1, 3])
    assert len({r.commit_id for r in requests}) == len(requests)
    assert [commit_id for commit_id, _ in third.commit_calls] == [requests[-1].commit_id]
