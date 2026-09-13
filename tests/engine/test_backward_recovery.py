"""Public backward failures, durable attempt prefixes, and whole-pass retries."""

from asyncio import CancelledError
from concurrent.futures import CancelledError as FutureCancelledError
from copy import deepcopy
import json
from textwrap import indent

from langchain_core.messages import AIMessage
import pytest

import agent_composer.llm_clients as llm_clients
from agent_composer.compose import load_flow
from agent_composer.memory import MemoryMutation, MemoryOperationError
from agent_composer.nodes.base import BackwardResult
from agent_composer.run import resume_flow, run_flow
from agent_composer.runtime.log import (
    BackwardFinishedRecord, BackwardReadyRecord, BackwardStartedRecord,
    JsonlRunLog, MemoryCommitFailedRecord, MemoryCommitRequestedRecord,
    MemoryCommitSucceededRecord,
)
from tests.engine._memory_fakes import _Memory
from tests.engine.test_agent_backward import _BackwardChat, _reply
from tests.engine.test_run_log import _FileProxy
from tests.engine.test_run_memory import _records


_CHAIN = """id: backward-recovery
name: backward_recovery
long_term_memory: test
llm_config: {provider: ollama, model: local-test}
input:
  topic: str
nodes:
  first:
    kind: agent
    mode: plain
    input:
      topic: ${input.topic}
    prompt: 'Draft ${topic}'
  second:
    kind: agent
    mode: plain
    long_term_memory: test
    input:
      draft: ${first.output}
    prompt: 'Review ${draft}'
output: ${second.output}
"""


def _chain(tmp_path, monkeypatch, *, workers=0, backend=None):
    """Run the same real two-AGENT public flow before installing backward replies."""
    flow = load_flow(_CHAIN)
    backend = _Memory() if backend is None else backend
    chat = _BackwardChat([AIMessage(content="draft"), AIMessage(content="answer")])
    monkeypatch.setattr(llm_clients, "model_from_config", lambda config: chat)
    path = tmp_path / "chain.jsonl"
    result = run_flow(flow, {"topic": "reefs"}, log=path, learnable=True,
                      num_workers=workers, memory={"test": backend})
    assert result.status == "succeeded" and result.output == "answer"
    assert chat.calls == 2 and backend.learn_contexts == backend.commit_calls == []
    return flow, path, backend, result, chat


@pytest.mark.parametrize("failed,value,output_type,feedback", [
    (False, {"items": [1, 2]}, "object", {"items": [None]}),
    (False, [[1], [2, 3]], "list[list[int]]", [None, [None]]),
    (False, [[1], [2, 3]], "list[list[int]]", [None, [None, None, None]]),
    (False, {"known": 1}, "object", {"missing": None}),
    (False, [1, 2], "list[int]", {"0": "Review."}),
    (True, 0, "int", {}),
    (True, 0, "int", []),
])
def test_invalid_root_shapes_preserve_the_public_forward_result_and_unopened_attempt(
    tmp_path, monkeypatch, failed, value, output_type, feedback,
):
    code = "if True:\n    raise RuntimeError('forward stopped')\nreturn 0" if failed else f"return {value!r}"
    flow = load_flow(f"""id: invalid-root-feedback
name: invalid_root_feedback
long_term_memory: test
nodes:
  work:
    kind: code
    code: |
{indent(code, '      ')}
    output: {output_type}
output: ${{work.output}}
""")
    backend = _Memory()
    path = tmp_path / "invalid.jsonl"
    result = run_flow(flow, {}, log=path, learnable=True, memory={"test": backend})
    assert result.status == ("failed" if failed else "succeeded")
    forward = deepcopy((result.input, result.status, result.output, result.failure, result.events))
    previous = (result.backward_outcome, result.learning)
    before = path.read_bytes()
    counters = (len(backend.open_snapshots), backend.close_count, len(backend.learn_contexts), len(backend.commit_calls))
    monkeypatch.setattr(llm_clients, "model_from_config", lambda config: pytest.fail("invalid feedback invoked model"))

    with pytest.raises((TypeError, ValueError)):
        result.backward(feedback)

    assert (result.input, result.status, result.output, result.failure, result.events) == forward
    assert result.backward_outcome is previous[0] and result.learning is previous[1]
    assert path.read_bytes() == before
    assert (len(backend.open_snapshots), backend.close_count, len(backend.learn_contexts), len(backend.commit_calls)) == counters
    assert not any(type(record) is BackwardStartedRecord for record in _records(path))
    inspected = resume_flow(flow, log=path, memory={"test": backend})
    assert inspected.events == [] and inspected.backward_outcome is None
    assert (inspected.input, inspected.status, inspected.output, inspected.failure) == forward[:4]
    assert path.read_bytes() == before
    assert (len(backend.open_snapshots), backend.close_count, len(backend.learn_contexts), len(backend.commit_calls)) == counters


@pytest.mark.parametrize("workers", [0, 2])
@pytest.mark.parametrize("invalid_result", [False, True])
def test_real_agent_failure_discards_partial_reverse_results_and_retries_every_callback(
    tmp_path, monkeypatch, workers, invalid_result,
):
    flow, path, backend, result, forward_chat = _chain(tmp_path, monkeypatch, workers=workers)
    forward = deepcopy((result.input, result.status, result.output, result.failure, result.events))
    forward_bytes = path.read_bytes()
    count = len(_records(path))
    bad = _reply(input_feedback={"topic": ["valid sibling"], "foreign": ["reject"]}) if invalid_result else AIMessage(content="not JSON")
    reverse = _BackwardChat([_reply(input_feedback={"draft": ["temporary"]}), bad])
    monkeypatch.setattr(llm_clients, "model_from_config", lambda config: reverse)

    outcome = result.backward("First review.")

    assert outcome.status == "failed" and outcome.failure.stage == "backward"
    assert outcome.failure.code == ("backward.result" if invalid_result else "backward.execute")
    assert outcome.failure.node_id == "first"
    assert outcome.feedback is None and outcome.learning.status == "incomplete"
    assert [policy.status for policy in outcome.learning.policies] == ["unrequested"]
    accepted = _records(path)[count:]
    assert [type(record) for record in accepted] == [BackwardStartedRecord, BackwardFinishedRecord]
    assert accepted[-1].attributed_results == ()
    assert path.read_bytes().startswith(forward_bytes)
    assert reverse.calls == 2 and forward_chat.calls == 2
    assert [json.loads(messages[1].content)["forward"]["node_id"] for messages in reverse.messages] == ["second", "first"]
    assert backend.learn_contexts == backend.commit_calls == [] and backend.close_count == 2
    assert (result.input, result.status, result.output, result.failure, result.events) == forward
    before = path.read_bytes()
    inspected = resume_flow(flow, log=path, memory={"test": backend})
    assert inspected.events == [] and inspected.backward_outcome == outcome
    assert (inspected.input, inspected.status, inspected.output, inspected.failure) == forward[:4]
    assert path.read_bytes() == before and reverse.calls == 2 and len(backend.open_snapshots) == 2

    retry = _BackwardChat([_reply(input_feedback={"draft": ["fresh draft"]}),
                           _reply(input_feedback={"topic": ["fresh topic"]})])
    monkeypatch.setattr(llm_clients, "model_from_config", lambda config: retry)
    completed = inspected.backward("Second review.")
    assert completed.status == "succeeded" and completed.feedback == BackwardResult({"topic": ("fresh topic",)})
    assert retry.calls == 2 and reverse.calls == 2 and forward_chat.calls == 2
    assert json.loads(retry.messages[0][1].content)["feedback"] == ["Second review."]
    assert [record.attempt_id for record in _records(path) if type(record) is BackwardStartedRecord] == [1, 2]
    assert len(backend.learn_contexts) == 2 and len(backend.commit_calls) == 1
    request, = [record for record in _records(path) if type(record) is MemoryCommitRequestedRecord]
    assert request.mutations == ()
    assert backend.commit_calls == [(request.commit_id, request.mutations)]
    assert backend.open_snapshots == [None, {"namespace": "test"}, {"namespace": "test"}]
    assert backend.close_count == 3 and all(session.closed for session in backend.sessions)
    assert (result.input, result.status, result.output, result.failure, result.events) == forward
    assert (inspected.input, inspected.status, inspected.output, inspected.failure, inspected.events) == (*forward[:4], [])


@pytest.mark.parametrize("last_kind,disposition", [
    ("backward_ready", "unrequested"),
    ("memory_commit_requested", "unknown"),
    ("memory_commit_succeeded", "succeeded"),
])
def test_terminal_resume_inspects_actual_public_attempt_prefixes_without_reopening_memory(
    tmp_path, monkeypatch, last_kind, disposition,
):
    flow, path, backend, result, forward_chat = _chain(tmp_path, monkeypatch)
    forward = deepcopy((result.input, result.status, result.output, result.failure, result.events))

    def propose(context):
        return [MemoryMutation("own/" + "/".join(context.frame_ref.static_path), context.frame_ref, {"retain": True})]

    backend.learn_outcomes = [propose, propose]
    reverse = _BackwardChat([_reply(input_feedback={"draft": ["draft review"]}),
                             _reply(input_feedback={"topic": ["topic review"]})])
    monkeypatch.setattr(llm_clients, "model_from_config", lambda config: reverse)
    completed = result.backward("Review.")
    assert completed.status == "succeeded"
    request, = [record for record in _records(path) if type(record) is MemoryCommitRequestedRecord]
    assert len(request.mutations) == 2
    assert backend.commit_calls == [(request.commit_id, request.mutations)]
    complete_bytes = path.read_bytes()
    lines = complete_bytes.splitlines(keepends=True)
    boundary = next(index for index, line in enumerate(lines) if json.loads(line)["kind"] == last_kind)
    prefix_path = tmp_path / f"{last_kind}.jsonl"
    prefix_bytes = b"".join(lines[:boundary + 1])
    prefix_path.write_bytes(prefix_bytes)
    counters = (len(backend.open_snapshots), backend.close_count, len(backend.learn_contexts), len(backend.commit_calls))
    monkeypatch.setattr(llm_clients, "model_from_config", lambda config: pytest.fail("terminal inspection invoked model"))

    inspected = resume_flow(flow, log=prefix_path, memory={"test": object()})

    outcome = inspected.backward_outcome
    assert outcome.status == "failed" and outcome.failure.code == "backward.interrupted"
    assert outcome.feedback == completed.feedback == BackwardResult({"topic": ("topic review",)})
    assert outcome.learning.status == "incomplete"
    policy, = outcome.learning.policies
    assert policy.policy_name == "test" and policy.status == disposition and policy.failure is None
    assert (policy.commit_id is None) is (disposition == "unrequested")
    assert inspected.events == []
    assert (inspected.input, inspected.status, inspected.output, inspected.failure) == forward[:4]
    assert (result.input, result.status, result.output, result.failure, result.events) == forward
    assert prefix_path.read_bytes() == prefix_bytes and path.read_bytes() == complete_bytes
    assert (len(backend.open_snapshots), backend.close_count, len(backend.learn_contexts), len(backend.commit_calls)) == counters
    assert reverse.calls == forward_chat.calls == 2
    ready, = [record for record in _records(prefix_path) if type(record) is BackwardReadyRecord]
    assert ready.root_result == completed.feedback and len(ready.attributed_results) == 4
    assert len(ready.frame_proposals) == 2 and all(len(proposal.mutations) == 1 for proposal in ready.frame_proposals)


@pytest.mark.parametrize("failure_kind", ["learner", "duplicate_proposal"])
def test_late_learning_failure_keeps_completed_feedback_but_retry_rebuilds_every_proposal(tmp_path, monkeypatch, failure_kind):
    flow, path, backend, result, forward_chat = _chain(tmp_path, monkeypatch)
    forward = deepcopy((result.input, result.status, result.output, result.failure, result.events))
    before = path.read_bytes()
    count = len(_records(path))

    def duplicate(context):
        return [MemoryMutation("duplicate", context.frame_ref, {"attempt": 1})]

    backend.learn_outcomes = [duplicate, duplicate if failure_kind == "duplicate_proposal" else
                              MemoryOperationError("learner", "declined", "Later frame declined.", False)]
    reverse = _BackwardChat([_reply(input_feedback={"draft": ["old draft"]}),
                             _reply(input_feedback={"topic": ["old topic"]})])
    monkeypatch.setattr(llm_clients, "model_from_config", lambda config: reverse)
    outcome = result.backward("First review.")
    assert outcome.status == "failed" and outcome.failure.stage == "learn"
    assert outcome.failure.code == ("declined" if failure_kind == "learner" else "memory.invalid_result")
    assert outcome.feedback == BackwardResult({"topic": ("old topic",)})
    assert outcome.learning.status == "failed" and outcome.learning.policies[0].status == "unrequested"
    accepted = _records(path)[count:]
    assert [type(record) for record in accepted] == [BackwardStartedRecord, BackwardFinishedRecord]
    assert len(accepted[-1].attributed_results) == 4
    assert path.read_bytes().startswith(before)
    assert reverse.calls == 2 and len(backend.learn_contexts) == 2 and backend.commit_calls == []
    assert outcome.failure.frame_ref == backend.learn_contexts[1].frame_ref
    assert backend.close_count == 2
    assert (result.input, result.status, result.output, result.failure, result.events) == forward
    before = path.read_bytes()
    inspected = resume_flow(flow, log=path, memory={"test": backend})
    assert inspected.events == [] and inspected.backward_outcome == outcome
    assert (inspected.input, inspected.status, inspected.output, inspected.failure) == forward[:4]
    assert path.read_bytes() == before and reverse.calls == 2 and len(backend.open_snapshots) == 2

    def fresh(context):
        return [MemoryMutation("fresh/" + "/".join(context.frame_ref.static_path), context.frame_ref, {"attempt": 2})]

    backend.learn_outcomes = [fresh, fresh]
    retry = _BackwardChat([_reply(input_feedback={"draft": ["fresh draft"]}),
                           _reply(input_feedback={"topic": ["fresh topic"]})])
    monkeypatch.setattr(llm_clients, "model_from_config", lambda config: retry)
    completed = inspected.backward("Second review.")
    assert completed.status == "succeeded" and completed.feedback == BackwardResult({"topic": ("fresh topic",)})
    assert retry.calls == reverse.calls == forward_chat.calls == 2
    assert len(backend.learn_contexts) == 4 and len(backend.commit_calls) == 1
    for index in (0, 1):
        original, repeated = backend.learn_contexts[index], backend.learn_contexts[index + 2]
        assert original.frame_ref == repeated.frame_ref and original.episodes == repeated.episodes
        assert original.feedback != repeated.feedback
    committed = backend.commit_calls[0][1]
    assert len(committed) == 2 and all(mutation.mutation_id.startswith("fresh/") for mutation in committed)
    assert all(mutation.payload == {"attempt": 2} for mutation in committed)
    request, = [record for record in _records(path) if type(record) is MemoryCommitRequestedRecord]
    assert len(request.mutations) == 2 and request.attempt_id == 2
    assert backend.commit_calls == [(request.commit_id, request.mutations)]
    assert backend.open_snapshots == [None, {"namespace": "test"}, {"namespace": "test"}]
    assert backend.close_count == 3 and all(session.closed for session in backend.sessions)
    assert (result.input, result.status, result.output, result.failure, result.events) == forward
    assert (inspected.input, inspected.status, inspected.output, inspected.failure, inspected.events) == (*forward[:4], [])


def test_unmaterializable_owner_context_is_a_complete_known_failure_on_each_explicit_retry(tmp_path, monkeypatch):
    flow = load_flow("""id: deep-owner-context
name: deep_owner_context
long_term_memory: test
input:
  payload: object
nodes: {}
output: ${input.payload}
""")
    value = {"leaf": "retained"}
    for _ in range(140):
        value = {"next": value}
    backend = _Memory()
    path = tmp_path / "deep.jsonl"
    result = run_flow(flow, {"payload": value}, log=path, learnable=True, memory={"test": backend})
    assert result.status == "succeeded" and result.output == value
    forward = deepcopy((result.input, result.status, result.output, result.failure, result.events))
    monkeypatch.setattr(llm_clients, "model_from_config", lambda config: pytest.fail("structural context failure invoked model"))
    before = path.read_bytes()
    first = result.backward("First review.")
    assert first.failure.stage == "learn" and first.failure.code == "memory.invalid_context"
    assert first.feedback == BackwardResult({"payload": ("First review.",)})
    assert first.learning.status == "failed" and first.learning.policies[0].status == "unrequested"
    assert path.read_bytes().startswith(before)
    assert (result.input, result.status, result.output, result.failure, result.events) == forward
    before = path.read_bytes()
    inspected = resume_flow(flow, log=path, memory={"test": backend})
    assert inspected.events == [] and inspected.backward_outcome == first
    assert (inspected.input, inspected.status, inspected.output, inspected.failure) == forward[:4]
    assert path.read_bytes() == before and len(backend.open_snapshots) == 2
    second = inspected.backward("Second review.")
    assert second.failure.code == first.failure.code and second.failure.frame_ref == first.failure.frame_ref
    assert second.feedback == BackwardResult({"payload": ("Second review.",)})
    assert backend.learn_contexts == backend.commit_calls == []
    records = _records(path)
    assert [record.attempt_id for record in records if type(record) is BackwardStartedRecord] == [1, 2]
    finished = [record for record in records if type(record) is BackwardFinishedRecord]
    assert len(finished) == 2 and all(len(record.attributed_results) == 2 for record in finished)
    assert not any(type(record) in (BackwardReadyRecord, MemoryCommitRequestedRecord) for record in records)
    assert backend.open_snapshots == [None, {"namespace": "test"}, {"namespace": "test"}]
    assert backend.close_count == 3 and all(session.closed for session in backend.sessions)
    assert (result.input, result.status, result.output, result.failure, result.events) == forward
    assert (inspected.input, inspected.status, inspected.output, inspected.failure, inspected.events) == (*forward[:4], [])


@pytest.mark.parametrize("error_type", [CancelledError, FutureCancelledError])
def test_reverse_cancellation_retains_started_only_and_resume_recomputes_the_entire_pass(tmp_path, monkeypatch, error_type):
    flow, path, backend, result, forward_chat = _chain(tmp_path, monkeypatch)
    forward = deepcopy((result.input, result.status, result.output, result.failure, result.events))
    before = path.read_bytes()
    count = len(_records(path))
    cancellation = error_type("stop reverse work")
    reverse = _BackwardChat([_reply(input_feedback={"draft": ["temporary"]}), cancellation])
    monkeypatch.setattr(llm_clients, "model_from_config", lambda config: reverse)
    with pytest.raises(error_type) as caught:
        result.backward("First review.")
    assert caught.value is cancellation
    assert [type(record) for record in _records(path)[count:]] == [BackwardStartedRecord]
    assert path.read_bytes().startswith(before)
    outcome = result.backward_outcome
    assert outcome.failure.code == "backward.interrupted" and outcome.feedback is None
    assert outcome.learning.status == "incomplete" and outcome.learning.policies[0].status == "unrequested"
    assert reverse.calls == 2 and forward_chat.calls == 2
    assert backend.learn_contexts == backend.commit_calls == [] and backend.close_count == 2
    assert (result.input, result.status, result.output, result.failure, result.events) == forward

    before = path.read_bytes()
    inspected = resume_flow(flow, log=path, memory={"test": backend})
    assert inspected.events == [] and inspected.backward_outcome == outcome
    assert (inspected.input, inspected.status, inspected.output, inspected.failure) == forward[:4]
    assert path.read_bytes() == before and reverse.calls == 2 and len(backend.open_snapshots) == 2

    retry = _BackwardChat([_reply(input_feedback={"draft": ["fresh draft"]}),
                           _reply(input_feedback={"topic": ["fresh topic"]})])
    monkeypatch.setattr(llm_clients, "model_from_config", lambda config: retry)
    assert inspected.backward("Second review.").feedback == BackwardResult({"topic": ("fresh topic",)})
    accepted = _records(path)[count:]
    assert [type(record) for record in accepted[:3]] == [BackwardStartedRecord, BackwardFinishedRecord, BackwardStartedRecord]
    assert accepted[1].outcome.feedback is None and accepted[1].attributed_results == ()
    assert accepted[2].attempt_id == 2
    assert retry.calls == 2 and reverse.calls == 2 and forward_chat.calls == 2
    assert len(backend.learn_contexts) == 2 and len(backend.commit_calls) == 1
    request, = [record for record in _records(path) if type(record) is MemoryCommitRequestedRecord]
    assert request.mutations == ()
    assert backend.commit_calls == [(request.commit_id, request.mutations)]
    assert backend.open_snapshots == [None, {"namespace": "test"}, {"namespace": "test"}]
    assert backend.close_count == 3 and all(session.closed for session in backend.sessions)
    assert (result.input, result.status, result.output, result.failure, result.events) == forward
    assert (inspected.input, inspected.status, inspected.output, inspected.failure, inspected.events) == (*forward[:4], [])


@pytest.mark.parametrize("phase,rejected_type", [
    ("started", BackwardStartedRecord),
    ("ready", BackwardReadyRecord),
    ("request", MemoryCommitRequestedRecord),
    ("partial_request", MemoryCommitRequestedRecord),
    ("success_observation", MemoryCommitSucceededRecord),
    ("failure_observation", MemoryCommitFailedRecord),
    ("finished", BackwardFinishedRecord),
    ("failed_finished", BackwardFinishedRecord),
])
def test_public_append_rejection_keeps_exact_authority_and_only_pre_request_retries(tmp_path, monkeypatch, phase, rejected_type):
    backend = _Memory(commit_outcomes=[MemoryOperationError("store", "declined", "Store declined.", False)]
                      if phase in ("failure_observation", "failed_finished") else [])
    flow, path, backend, result, forward_chat = _chain(tmp_path, monkeypatch, backend=backend)
    forward = deepcopy((result.input, result.status, result.output, result.failure, result.events))
    forward_bytes = path.read_bytes()
    count = len(_records(path))
    generation = 1

    def propose(context):
        return [MemoryMutation(f"pass-{generation}/" + "/".join(context.frame_ref.static_path),
                               context.frame_ref, {"generation": generation})]

    backend.learn_outcomes = [propose] * 4
    reverse = _BackwardChat([_reply(input_feedback={"draft": ["old draft"]}),
                             _reply(input_feedback={"topic": ["old topic"]})])
    monkeypatch.setattr(llm_clients, "model_from_config", lambda config: reverse)
    append = JsonlRunLog.append
    error = OSError("rejected append")
    rejected_prefixes = []
    partial_writes = []

    def reject(log, record):
        if type(record) is not rejected_type:
            return append(log, record)
        rejected_prefixes.append(path.read_bytes())
        if phase == "partial_request":
            wrapped = log._file

            def partial_write(data):
                partial = data[:len(data) // 2]
                wrapped.write(partial)
                wrapped.flush()
                assert path.stat().st_size > len(rejected_prefixes[-1])
                partial_writes.append(partial)
                raise error

            log._file = _FileProxy(wrapped, on_write=partial_write)
            try:
                return append(log, record)
            finally:
                log._file = wrapped
        raise error

    monkeypatch.setattr(JsonlRunLog, "append", reject)
    with pytest.raises(OSError) as caught:
        result.backward("First review.")
    assert caught.value is error
    assert rejected_prefixes == [path.read_bytes()]
    assert bool(partial_writes) is (phase == "partial_request")
    assert path.read_bytes().startswith(forward_bytes)
    expected = [] if phase == "started" else [BackwardStartedRecord]
    if phase not in ("started", "ready"):
        expected.append(BackwardReadyRecord)
    if phase in ("success_observation", "failure_observation", "finished", "failed_finished"):
        expected.append(MemoryCommitRequestedRecord)
    if phase in ("finished", "failed_finished"):
        expected.append(MemoryCommitSucceededRecord if phase == "finished" else MemoryCommitFailedRecord)
    accepted = _records(path)[count:]
    assert [type(record) for record in accepted] == expected
    expected_callbacks = 0 if phase == "started" else 2
    expected_commits = int(phase in ("success_observation", "failure_observation", "finished", "failed_finished"))
    assert reverse.calls == expected_callbacks and forward_chat.calls == 2
    assert len(backend.learn_contexts) == expected_callbacks and len(backend.commit_calls) == expected_commits
    if expected_commits:
        request, = [record for record in accepted if type(record) is MemoryCommitRequestedRecord]
        assert len(request.mutations) == 2
        assert backend.commit_calls == [(request.commit_id, request.mutations)]
    assert backend.close_count == 2 and all(session.closed for session in backend.sessions)
    assert (result.input, result.status, result.output, result.failure, result.events) == forward
    outcome = result.backward_outcome
    if phase == "started":
        assert outcome is None and result.learning is None
    else:
        assert outcome.failure.code == "backward.interrupted"
        assert outcome.feedback == (None if phase == "ready" else BackwardResult({"topic": ("old topic",)}))
        assert outcome.learning.status == ("failed" if phase == "failed_finished" else "incomplete")
        disposition = {"ready": "unrequested", "request": "unrequested", "partial_request": "unrequested",
                       "success_observation": "unknown", "failure_observation": "unknown",
                       "finished": "succeeded", "failed_finished": "failed"}[phase]
        assert outcome.learning.policies[0].status == disposition

    monkeypatch.setattr(JsonlRunLog, "append", append)
    before = path.read_bytes()
    inspected = resume_flow(flow, log=path, memory={"test": backend})
    assert inspected.events == [] and inspected.backward_outcome == outcome
    assert (inspected.input, inspected.status, inspected.output, inspected.failure) == forward[:4]
    assert path.read_bytes() == before and reverse.calls == expected_callbacks and len(backend.open_snapshots) == 2
    if expected_commits:
        previous = inspected.backward_outcome
        with pytest.raises(ValueError, match="accepted prefix"):
            inspected.backward("Forbidden retry.")
        assert inspected.backward_outcome is previous
        assert path.read_bytes() == before and reverse.calls == expected_callbacks
        assert len(backend.open_snapshots) == backend.close_count == 2
        assert len(backend.commit_calls) == 1 and len(backend.learn_contexts) == 2
        assert (inspected.input, inspected.status, inspected.output, inspected.failure, inspected.events) == (*forward[:4], [])
        return

    generation = 2
    retry = _BackwardChat([_reply(input_feedback={"draft": ["fresh draft"]}),
                           _reply(input_feedback={"topic": ["fresh topic"]})])
    monkeypatch.setattr(llm_clients, "model_from_config", lambda config: retry)
    completed = inspected.backward("Second review.")
    assert completed.status == "succeeded" and completed.feedback == BackwardResult({"topic": ("fresh topic",)})
    assert retry.calls == 2 and reverse.calls == expected_callbacks and forward_chat.calls == 2
    assert len(backend.learn_contexts) == expected_callbacks + 2 and len(backend.commit_calls) == 1
    records = _records(path)
    request, = [record for record in records if type(record) is MemoryCommitRequestedRecord]
    assert len(request.mutations) == 2
    assert backend.commit_calls == [(request.commit_id, request.mutations)]
    assert all(mutation.mutation_id.startswith("pass-2/") and mutation.payload == {"generation": 2}
               for mutation in backend.commit_calls[0][1])
    assert [record.attempt_id for record in records if type(record) is BackwardStartedRecord] == ([1] if phase == "started" else [1, 2])
    if phase != "started":
        prior, = [record for record in records if type(record) is BackwardFinishedRecord and record.attempt_id == 1]
        assert prior.outcome.failure.code == "backward.interrupted"
        assert prior.outcome.feedback == outcome.feedback
    assert backend.open_snapshots == [None, {"namespace": "test"}, {"namespace": "test"}]
    assert backend.close_count == 3 and all(session.closed for session in backend.sessions)
    assert (result.input, result.status, result.output, result.failure, result.events) == forward
    assert (inspected.input, inspected.status, inspected.output, inspected.failure, inspected.events) == (*forward[:4], [])


def test_three_policy_failure_preserves_ordered_dispositions_and_closes_public_retry(tmp_path, monkeypatch):
    source = _CHAIN.replace("long_term_memory: test", "long_term_memory: A", 1)
    source = source.replace("  first:\n    kind: agent", "  first:\n    kind: agent\n    long_term_memory: B")
    flow = load_flow(source.replace("    long_term_memory: test", "    long_term_memory: C"))
    backends = {name: _Memory() for name in ("A", "B", "C")}
    backends["B"].commit_outcomes = [MemoryOperationError("store", "backward.interrupted", "Store declined.", True)]
    forward_chat = _BackwardChat([AIMessage(content="draft"), AIMessage(content="answer")])
    monkeypatch.setattr(llm_clients, "model_from_config", lambda config: forward_chat)
    path = tmp_path / "policies.jsonl"
    result = run_flow(flow, {"topic": "reefs"}, log=path, learnable=True, memory=backends)
    assert result.status == "succeeded" and result.output == "answer"
    forward = deepcopy((result.input, result.status, result.output, result.failure, result.events))
    forward_bytes = path.read_bytes()
    count = len(_records(path))
    learned = []

    def propose(policy_name, context):
        assert all(backend.commit_calls == [] for backend in backends.values())
        learned.append(policy_name)
        return [MemoryMutation("same-id", context.frame_ref, {"policy": policy_name})]

    for name, backend in backends.items():
        backend.learn_outcomes = [lambda context, name=name: propose(name, context)]
    reverse = _BackwardChat([_reply(input_feedback={"draft": ["draft review"]}),
                             _reply(input_feedback={"topic": ["topic review"]})])
    monkeypatch.setattr(llm_clients, "model_from_config", lambda config: reverse)
    outcome = result.backward("Review.")

    assert learned == ["A", "B", "C"]
    assert outcome.status == "failed" and outcome.failure.stage == "commit"
    assert outcome.failure.code == "backward.interrupted" and outcome.failure.policy_name == "B"
    assert outcome.feedback == BackwardResult({"topic": ("topic review",)})
    assert outcome.learning.status == "failed"
    assert [(policy.policy_name, policy.status) for policy in outcome.learning.policies] == [
        ("A", "succeeded"), ("B", "failed"), ("C", "unrequested"),
    ]
    assert outcome.learning.policies[1].failure == outcome.failure.cause
    assert outcome.failure.cause.retryable is True
    accepted = _records(path)[count:]
    assert [type(record) for record in accepted] == [
        BackwardStartedRecord, BackwardReadyRecord, MemoryCommitRequestedRecord,
        MemoryCommitSucceededRecord, MemoryCommitRequestedRecord, MemoryCommitFailedRecord,
        BackwardFinishedRecord,
    ]
    requests = [record for record in accepted if type(record) is MemoryCommitRequestedRecord]
    assert [record.policy_name for record in requests] == ["A", "B"]
    for request in requests:
        assert len(request.mutations) == 1
        assert backends[request.policy_name].commit_calls == [(request.commit_id, request.mutations)]
    assert [proposal.frame_ref for proposal in accepted[1].frame_proposals] == [
        backend.learn_contexts[0].frame_ref for backend in backends.values()
    ]
    assert [proposal.mutations[0].payload for proposal in accepted[1].frame_proposals] == [
        {"policy": name} for name in backends
    ]
    assert [len(backend.commit_calls) for backend in backends.values()] == [1, 1, 0]
    assert path.read_bytes().startswith(forward_bytes)
    assert reverse.calls == forward_chat.calls == 2
    assert all(len(backend.learn_contexts) == 1 and backend.close_count == 2 for backend in backends.values())
    assert all(session.closed for backend in backends.values() for session in backend.sessions)
    assert (result.input, result.status, result.output, result.failure, result.events) == forward
    before = path.read_bytes()
    counters = [(len(backend.open_snapshots), backend.close_count, len(backend.learn_contexts), len(backend.commit_calls))
                for backend in backends.values()]
    monkeypatch.setattr(llm_clients, "model_from_config", lambda config: pytest.fail("closed retry invoked model"))

    inspected = resume_flow(flow, log=path, memory=backends)
    assert inspected.events == [] and inspected.backward_outcome == outcome
    assert (inspected.input, inspected.status, inspected.output, inspected.failure) == forward[:4]
    previous = (inspected.backward_outcome, inspected.learning)
    with pytest.raises(ValueError, match="accepted prefix"):
        inspected.backward("Forbidden retry.")
    assert inspected.backward_outcome is previous[0] and inspected.learning is previous[1]
    assert path.read_bytes() == before and reverse.calls == forward_chat.calls == 2
    assert [(len(backend.open_snapshots), backend.close_count, len(backend.learn_contexts), len(backend.commit_calls))
            for backend in backends.values()] == counters
    assert (result.input, result.status, result.output, result.failure, result.events) == forward
    assert (inspected.input, inspected.status, inspected.output, inspected.failure, inspected.events) == (*forward[:4], [])


@pytest.mark.parametrize("corruption", ["root_projection", "requested_batch"])
def test_public_replay_rejects_complete_record_corruption_before_any_new_effects(tmp_path, monkeypatch, corruption):
    flow, path, backend, result, forward_chat = _chain(tmp_path, monkeypatch)
    forward = deepcopy((result.input, result.status, result.output, result.failure, result.events))

    def propose(context):
        return [MemoryMutation("own/" + "/".join(context.frame_ref.static_path), context.frame_ref, {"retain": True})]

    backend.learn_outcomes = [propose, propose]
    reverse = _BackwardChat([_reply(input_feedback={"draft": ["draft review"]}),
                             _reply(input_feedback={"topic": ["topic review"]})])
    monkeypatch.setattr(llm_clients, "model_from_config", lambda config: reverse)
    assert result.backward("Review.").status == "succeeded"
    request, = [record for record in _records(path) if type(record) is MemoryCommitRequestedRecord]
    assert len(request.mutations) == 2
    assert backend.commit_calls == [(request.commit_id, request.mutations)]
    previous = (result.backward_outcome, result.learning)
    counters = (len(backend.open_snapshots), backend.close_count, len(backend.learn_contexts), len(backend.commit_calls))
    lines = path.read_bytes().splitlines(keepends=True)
    kind = "backward_ready" if corruption == "root_projection" else "memory_commit_requested"
    index = next(index for index, line in enumerate(lines) if json.loads(line)["kind"] == kind)
    record = json.loads(lines[index])
    if corruption == "root_projection":
        record["payload"]["root_result"]["input_feedback"]["topic"] = ["forged projection"]
    else:
        record["payload"]["mutations"][0]["payload"] = {"forged": True}
    lines[index] = (json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
    corrupted = b"".join(lines)
    path.write_bytes(corrupted)
    monkeypatch.setattr(llm_clients, "model_from_config", lambda config: pytest.fail("corrupt replay invoked model"))

    message = "root result must match" if corruption == "root_projection" else "batch and identity must match"
    with pytest.raises(ValueError, match=message):
        result.backward("Review after tampering.")
    assert result.backward_outcome is previous[0] and result.learning is previous[1]
    with pytest.raises(ValueError, match=message):
        resume_flow(flow, log=path, memory={"test": backend})

    assert path.read_bytes() == corrupted
    assert (len(backend.open_snapshots), backend.close_count, len(backend.learn_contexts), len(backend.commit_calls)) == counters
    assert reverse.calls == forward_chat.calls == 2
    assert (result.input, result.status, result.output, result.failure, result.events) == forward
