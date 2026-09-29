"""Explicit backward proposals and ordered, append-authorized memory effects."""

from asyncio import CancelledError as AsyncCancelledError
from concurrent.futures import CancelledError as FutureCancelledError
import copy

import pytest

from agent_composer.compile.model import START_ID, FlowOutput
from agent_composer.memory import (
    LearnContext, MemoryFrameRef, MemoryFrameSpec, MemoryMutation, MemoryOwner,
    MemoryOperationError, MemoryScope, MemorySnapshotEnvelope,
)
from agent_composer.nodes.base import BackwardResult, _CapabilityFault
import agent_composer.runtime.backward as backward_module
from agent_composer.runtime.contracts import FrameProposal
from agent_composer.runtime.log import (
    BackwardFinishedRecord, BackwardReadyRecord, BackwardStartedRecord,
    MemoryCommitFailedRecord, MemoryCommitRequestedRecord, MemoryCommitSucceededRecord,
)
from agent_composer.runtime.memory_runtime import MemoryRuntime
from agent_composer.runtime.state import commit_prepared, prepare_record
from tests.engine._backward_fakes import RecordingBackwardNode
from tests.engine._graph_builder import _engine, _graph
from tests.engine._memory_fakes import _Memory, _Session


def _layout(*policies):
    return tuple(MemoryFrameSpec(
        MemoryFrameRef("root", ("root", f"frame{index}"), MemoryOwner.AGENT), policy, {},
    ) for index, policy in enumerate(policies))


def _open(layout, memory):
    return MemoryRuntime.open(
        layout=layout, origins={spec.ref: None for spec in layout}, memory=memory,
        snapshots=tuple(MemorySnapshotEnvelope(name, {}) for name in sorted({
            spec.policy_name for spec in layout
        })),
    )


def test_proposal_phase_visits_every_frame_without_committing():
    layout = _layout("A", "A", "B")
    contexts = tuple(LearnContext("run", spec.ref, ()) for spec in layout)
    order = []

    def propose(context):
        order.append(context.frame_ref)
        return [MemoryMutation("same-id", context.frame_ref, {"notes": ["one"]})]

    memory = {"A": _Memory(learn_outcomes=[propose, []]), "B": _Memory(learn_outcomes=[propose])}
    runtime = _open(layout, memory)
    try:
        proposals = runtime.propose_learning(contexts)
    finally:
        runtime.close()
    assert tuple(proposal.frame_ref for proposal in proposals) == tuple(spec.ref for spec in layout)
    assert all(type(proposal) is FrameProposal for proposal in proposals)
    assert [len(proposal.mutations) for proposal in proposals] == [1, 0, 1]
    assert order == [layout[0].ref, layout[2].ref]
    assert memory["A"].learn_contexts == list(contexts[:2])
    assert memory["B"].learn_contexts == list(contexts[2:])
    assert all(not backend.commit_calls and backend.close_count == 1 for backend in memory.values())


@pytest.mark.parametrize("invalid", ["missing", "reordered", "duplicate", "foreign", "run", "list"])
def test_proposal_contexts_validate_completely_before_any_learner(invalid):
    layout = _layout("A", "A")
    contexts = tuple(LearnContext("run", spec.ref, ()) for spec in layout)
    if invalid == "missing":
        contexts = contexts[:1]
    elif invalid == "reordered":
        contexts = contexts[::-1]
    elif invalid == "duplicate":
        contexts = (contexts[0], contexts[0])
    elif invalid == "foreign":
        contexts = (contexts[0], LearnContext("run", _layout("B", "B", "B")[2].ref, ()))
    elif invalid == "run":
        contexts = (contexts[0], LearnContext("other", layout[1].ref, ()))
    else:
        contexts = list(contexts)
    memory = _Memory()
    runtime = _open(layout, {"A": memory})
    try:
        with pytest.raises((TypeError, ValueError)):
            runtime.propose_learning(contexts)
    finally:
        runtime.close()
    assert not memory.learn_contexts and not memory.commit_calls


def _terminal(policies=("A",), *, value="reefs", callback=None, learnable=True):
    """Build one real retained CODE occurrence plus complete enabled frame inventory."""
    node = RecordingBackwardNode(
        "work", lambda inputs: inputs["topic"],
        callback or (lambda _context, feedback: BackwardResult({"topic": feedback})),
        reads={"topic": "${input.topic}"},
    )
    flow = _graph([node], [(START_ID, "work")], outputs=[FlowOutput("answer", "${work.output}")])
    layout = tuple(MemoryFrameSpec(
        MemoryFrameRef("root", ("root",) if index == 0 else ("root", f"unused{index}"), MemoryOwner.FLOW),
        policy, {},
    ) for index, policy in enumerate(policies))
    if layout:
        for current in flow.nodes.values():
            current.memory_scope = MemoryScope("root", ("root", current.id), (layout[0].ref,))
    engine = _engine(flow, {"topic": value}, learnable=learnable, memory_layout=layout or None,
                     memory_snapshots=tuple(MemorySnapshotEnvelope(name, {}) for name in sorted(set(policies)))
                     if layout else None)
    list(engine.run())
    return engine, node


def _records(engine):
    return tuple(envelope.record for envelope in engine.log.records())


def _drive(engine, memory, feedback="review"):
    return backward_module.drive_backward(
        engine.state, engine.log, feedback,
        open_memory=lambda: _open(engine.state.memory_layout or (), memory),
    )


def test_backward_finishes_all_callbacks_before_ready_and_policy_requests(monkeypatch):
    engine, node = _terminal(("A", "A", "B"))
    layout = engine.state.memory_layout
    visited = []

    def learn(context):
        assert len(node.calls) == 1
        assert type(_records(engine)[-1]) is BackwardStartedRecord
        assert not any(backend.commit_calls for backend in memory.values())
        visited.append(context.frame_ref)
        return [MemoryMutation(str(len(visited)), context.frame_ref, {"notes": ["one"]})]

    memory = {"A": _Memory(learn_outcomes=[learn, learn]), "B": _Memory(learn_outcomes=[learn])}
    original_commit = _Session.commit

    def commit(session, commit_id, mutations):
        assert visited == [spec.ref for spec in layout]
        assert type(_records(engine)[-1]) is MemoryCommitRequestedRecord
        assert _records(engine)[-1].attempt_id == 1
        assert engine.state.backward_state.ready is not None
        return original_commit(session, commit_id, mutations)

    monkeypatch.setattr(_Session, "commit", commit)
    forward_pool = copy.deepcopy(engine.state.pool.store)
    result = _drive(engine, memory)
    assert result.status == "succeeded" and result.feedback.input_feedback == {"topic": ("review",)}
    assert [policy.status for policy in result.learning.policies] == ["succeeded", "succeeded"]
    assert [len(memory[name].commit_calls[0][1]) for name in ("A", "B")] == [2, 1]
    assert [len(context.episodes) for context in memory["A"].learn_contexts] == [3, 0]
    assert memory["B"].learn_contexts[0].episodes == ()
    assert all(backend.open_snapshots == [{}] and backend.close_count == 1 for backend in memory.values())
    assert engine.state.pool.store == forward_pool
    assert engine.state.trace.next_sequence == engine.state.next_sequence == engine.log.next_sequence
    assert type(_records(engine)[-1]) is BackwardFinishedRecord


@pytest.mark.parametrize("failure_kind", ["backend", "foreign", "duplicate", "tuple", "forged"])
def test_late_proposal_failure_discards_all_staged_mutations(failure_kind):
    engine, node = _terminal(("A", "A", "B"))
    layout = engine.state.memory_layout
    prior = MemoryMutation("same", layout[0].ref, {"lesson": "first"})

    def fail(context):
        if failure_kind == "backend":
            raise MemoryOperationError("provider", "backward.interrupted", "Backend declined.", True)
        if failure_kind == "foreign":
            return [MemoryMutation("foreign", layout[0].ref, {})]
        if failure_kind == "duplicate":
            return [MemoryMutation("same", context.frame_ref, {})]
        if failure_kind == "tuple":
            return ()
        mutation = MemoryMutation("valid", context.frame_ref, {})
        object.__setattr__(mutation, "mutation_id", True)
        return [mutation]

    memory = {"A": _Memory(learn_outcomes=[[prior], fail]), "B": _Memory()}
    outcome = _drive(engine, memory)
    assert outcome.status == "failed" and outcome.failure.stage == "learn"
    assert outcome.failure.frame_ref == layout[1].ref and outcome.failure.policy_name == "A"
    assert outcome.failure.code == ("backward.interrupted" if failure_kind == "backend" else "memory.invalid_result")
    assert outcome.feedback.input_feedback == {"topic": ("review",)}
    assert outcome.learning.status == "failed"
    assert [policy.status for policy in outcome.learning.policies] == ["unrequested", "unrequested"]
    assert len(engine.state.backward_state.finished.attributed_results) == len(engine.state.trace.occurrences)
    assert engine.state.backward_state.ready is None and engine.state.backward_state.can_start
    assert len(node.calls) == 1 and len(memory["A"].learn_contexts) == 2
    assert not memory["B"].learn_contexts
    assert all(not backend.commit_calls and backend.close_count == 1 for backend in memory.values())


@pytest.mark.parametrize("invalid_result", [False, True])
def test_policy_failure_preserves_prior_success_and_stops_later_policies(invalid_result):
    engine, node = _terminal(("A", "B", "C"))
    failed = object() if invalid_result else MemoryOperationError(
        "storage", "backward.interrupted", "Store refused.", True,
    )
    memory = {"A": _Memory(), "B": _Memory(commit_outcomes=[failed]), "C": _Memory()}
    outcome = _drive(engine, memory)
    assert outcome.status == "failed" and outcome.failure.stage == "commit"
    assert outcome.failure.policy_name == "B" and outcome.failure.frame_ref is None
    assert outcome.failure.code == ("memory.invalid_result" if invalid_result else "backward.interrupted")
    assert outcome.learning.status == "failed"
    assert [policy.status for policy in outcome.learning.policies] == ["succeeded", "failed", "unrequested"]
    assert [len(memory[name].commit_calls) for name in ("A", "B", "C")] == [1, 1, 0]
    assert all(len(backend.learn_contexts) == 1 and backend.close_count == 1 for backend in memory.values())
    assert len(node.calls) == 1 and engine.state.backward_state.can_start
    assert engine.state.backward_state.finished.attributed_results == ()
    assert len(engine.state.backward_state.ready.attributed_results) == 3


def test_empty_feedback_still_learns_empty_frames_and_commits_empty_batches():
    engine, node = _terminal(("A", "A", "B"))
    memory = {"A": _Memory(), "B": _Memory()}
    outcome = _drive(engine, memory, None)
    assert not node.calls and outcome.status == "succeeded"
    assert outcome.feedback == BackwardResult()
    assert [len(memory[name].learn_contexts) for name in ("A", "B")] == [2, 1]
    assert all(not context.feedback for backend in memory.values() for context in backend.learn_contexts)
    assert all(len(backend.commit_calls) == 1 and backend.commit_calls[0][1] == () for backend in memory.values())


def test_no_policy_backward_still_records_ready_and_finished_without_opening_resources():
    engine, node = _terminal(())
    before = len(_records(engine))
    outcome = backward_module.drive_backward(
        engine.state, engine.log, "review", open_memory=lambda: pytest.fail("no enabled policy"),
    )
    assert outcome.status == "succeeded" and outcome.learning is None
    assert outcome.feedback.input_feedback == {"topic": ("review",)} and len(node.calls) == 1
    assert [type(record) for record in _records(engine)[before:]] == [
        BackwardStartedRecord, BackwardReadyRecord, BackwardFinishedRecord,
    ]
    with pytest.raises(ValueError, match="cannot start"):
        backward_module.drive_backward(engine.state, engine.log, "again")
    assert len(node.calls) == 1


def test_context_depth_failure_keeps_completed_backward_without_calling_learners():
    value = "leaf"
    for _ in range(140):
        value = [value]
    engine, node = _terminal(value=value)
    memory = _Memory()
    outcome = _drive(engine, {"A": memory})
    assert outcome.failure.stage == "learn" and outcome.failure.code == "memory.invalid_context"
    assert outcome.failure.message == "memory learn context could not be constructed"
    assert outcome.failure.frame_ref == engine.state.memory_layout[0].ref
    assert outcome.feedback.input_feedback == {"topic": ("review",)}
    assert len(engine.state.backward_state.finished.attributed_results) == 3
    assert len(node.calls) == 1 and not memory.learn_contexts and not memory.commit_calls
    assert memory.close_count == 1 and engine.state.backward_state.can_start


@pytest.mark.parametrize("invalid_result", [False, True])
def test_computation_failure_finishes_without_partial_feedback_or_learning(invalid_result):
    def callback(_context, _feedback):
        if invalid_result:
            return BackwardResult({"undeclared": ("no",)})
        raise ValueError("Callback failed.")

    engine, node = _terminal(callback=callback)
    memory = _Memory()
    outcome = _drive(engine, {"A": memory})
    assert outcome.failure.code == ("backward.result" if invalid_result else "backward.execute")
    assert outcome.feedback is None and engine.state.backward_state.finished.attributed_results == ()
    assert outcome.learning.status == "incomplete" and outcome.learning.policies[0].status == "unrequested"
    assert len(node.calls) == 1 and not memory.learn_contexts and not memory.commit_calls
    assert memory.close_count == 1


_APPEND_PHASES = (
    (BackwardStartedRecord, None), (BackwardReadyRecord, None),
    (MemoryCommitRequestedRecord, None), (MemoryCommitSucceededRecord, None),
    (MemoryCommitFailedRecord, "commit"), (BackwardFinishedRecord, None),
    (BackwardFinishedRecord, "learn"), (BackwardFinishedRecord, "commit"),
)


@pytest.mark.parametrize("record_type,known_failure", _APPEND_PHASES)
def test_append_fault_preserves_exact_prefix_and_closes_resources(record_type, known_failure, monkeypatch):
    engine, node = _terminal(("A", "B"))
    typed_error = MemoryOperationError("backend", "declined", "Declined.", False)
    memory = {"A": _Memory(learn_outcomes=[typed_error] if known_failure == "learn" else [],
                           commit_outcomes=[typed_error] if known_failure == "commit" else []), "B": _Memory()}
    original_append = engine.log.append
    error = OSError("append unavailable")
    rejected = []
    forward_result, forward_pool = engine.state.result, copy.deepcopy(engine.state.pool.store)

    def append(record):
        if type(record) is record_type:
            rejected.append(record)
            raise error
        return original_append(record)

    monkeypatch.setattr(engine.log, "append", append)
    with pytest.raises(OSError) as caught:
        _drive(engine, memory)
    assert caught.value is error and len(rejected) == 1
    assert engine.state.result is forward_result and engine.state.pool.store == forward_pool
    assert engine.state.next_sequence == engine.state.trace.next_sequence == engine.log.next_sequence
    assert engine.state.backward_state.finished is None
    assert all(backend.close_count == 1 for backend in memory.values())
    current = engine.state.backward_state
    if record_type is BackwardStartedRecord:
        assert current.started is None and not node.calls and not memory["A"].learn_contexts
    else:
        assert current.started is not None and len(node.calls) == 1
    if record_type in (BackwardStartedRecord, BackwardReadyRecord) or known_failure == "learn":
        assert current.ready is None and not memory["A"].commit_calls
    elif record_type is MemoryCommitRequestedRecord:
        assert current.ready is not None and not current.ever_requested and not memory["A"].commit_calls
    elif record_type in (MemoryCommitSucceededRecord, MemoryCommitFailedRecord):
        assert current.policy_outcomes[0].status == "unknown" and len(memory["A"].commit_calls) == 1
        assert not memory["B"].commit_calls
    else:
        assert current.outcome.failure.code == "backward.interrupted"
        assert current.policy_outcomes[0].status == ("failed" if known_failure == "commit" else "succeeded")


@pytest.mark.parametrize("failure_record", [BackwardReadyRecord, MemoryCommitRequestedRecord])
def test_manual_precommit_retry_reopens_snapshots_and_recomputes_every_frame(failure_record, monkeypatch):
    engine, node = _terminal(("A", "A"))
    visits = []

    def learn(context):
        visits.append(context.frame_ref)
        return [MemoryMutation(str(len(visits)), context.frame_ref, {})]

    memory = _Memory(learn_outcomes=[learn] * 4)
    append = engine.log.append

    def fail(record):
        if type(record) is failure_record:
            raise OSError("interrupted before request")
        return append(record)

    monkeypatch.setattr(engine.log, "append", fail)
    with pytest.raises(OSError):
        _drive(engine, {"A": memory}, "first")
    old = engine.state.backward_state
    assert old.finished is None and not old.ever_requested
    monkeypatch.setattr(engine.log, "append", append)
    before = len(_records(engine))
    with pytest.raises(ValueError):
        _drive(engine, {"A": memory}, [])
    assert engine.state.backward_state is old and len(_records(engine)) == before
    assert memory.open_snapshots == [{}] and len(node.calls) == 1

    outcome = _drive(engine, {"A": memory}, "second")
    assert outcome.status == "succeeded" and outcome.feedback.input_feedback == {"topic": ("second",)}
    assert memory.open_snapshots == [{}, {}] and memory.close_count == 2
    assert visits == list(engine.state.backward_state.frame_order) * 2 and len(node.calls) == 2
    assert [mutation.mutation_id for mutation in memory.commit_calls[0][1]] == ["3", "4"]
    assert len(engine.state.backward_state.attempts) == 2
    first = engine.state.backward_state.attempts[0].finished.outcome
    assert first.failure.code == "backward.interrupted"
    assert first.feedback == (old.ready.root_result if old.ready is not None else None)
    assert [type(record) for record in _records(engine)[before:before + 2]] == [
        BackwardFinishedRecord, BackwardStartedRecord,
    ]
    assert engine.state.backward_state.started.attempt_id == 2


def test_known_precommit_failure_allows_fresh_complete_pass_without_prior_proposals():
    engine, node = _terminal(("A", "A"))

    def proposal(name):
        return lambda context: [MemoryMutation(name, context.frame_ref, {})]

    memory = _Memory(learn_outcomes=[
        proposal("old"), MemoryOperationError("learn", "rejected", "Rejected.", False),
        proposal("new-first"), proposal("new-second"),
    ])
    assert _drive(engine, {"A": memory}).status == "failed"
    assert engine.state.backward_state.can_start
    assert _drive(engine, {"A": memory}).status == "succeeded"
    assert len(node.calls) == 2 and len(memory.learn_contexts) == 4
    assert memory.open_snapshots == [{}, {}] and memory.close_count == 2
    assert [mutation.mutation_id for mutation in memory.commit_calls[0][1]] == ["new-first", "new-second"]


@pytest.mark.parametrize("problem", ["layout", "snapshots", "opener"])
def test_failed_resource_setup_does_not_close_an_interrupted_attempt(problem):
    engine, node = _terminal()
    commit_prepared(engine.log, engine.state, prepare_record(engine.state, BackwardStartedRecord(1, "first")))
    old, sequence = engine.state.backward_state, engine.state.next_sequence
    memory = _Memory()

    def open_memory():
        if problem == "opener":
            raise OSError("cannot reopen")
        layout = _layout("A") if problem == "layout" else engine.state.memory_layout
        return MemoryRuntime.open(
            layout=layout, origins={spec.ref: None for spec in layout}, memory={"A": memory},
            snapshots=(MemorySnapshotEnvelope("A", {"changed": True} if problem == "snapshots" else {}),),
        )

    with pytest.raises((OSError, ValueError)):
        backward_module.drive_backward(engine.state, engine.log, "second", open_memory=open_memory)
    assert engine.state.backward_state is old and engine.state.next_sequence == sequence
    assert not node.calls and not memory.learn_contexts and not memory.commit_calls
    assert memory.close_count == (0 if problem == "opener" else 1)


@pytest.mark.parametrize("operation", ["learn", "commit"])
@pytest.mark.parametrize("error_type", [MemoryError, RuntimeError, AsyncCancelledError, FutureCancelledError,
                                       KeyboardInterrupt, SystemExit, RecursionError])
def test_backend_host_or_unexpected_error_escapes_with_exact_prefix(operation, error_type):
    engine, node = _terminal(("A", "B"))
    error = error_type("host stopped")
    memory = {"A": _Memory(learn_outcomes=[error] if operation == "learn" else [],
                           commit_outcomes=[error] if operation == "commit" else []), "B": _Memory()}
    with pytest.raises(error_type) as caught:
        _drive(engine, memory)
    assert caught.value is error and len(node.calls) == 1
    backward = engine.state.backward_state
    assert backward.finished is None
    assert backward.policy_outcomes[0].status == ("unrequested" if operation == "learn" else "unknown")
    assert not memory["B"].commit_calls
    assert all(backend.close_count == 1 for backend in memory.values())
    if operation == "commit":
        before = len(_records(engine))
        with pytest.raises(ValueError, match="cannot start"):
            _drive(engine, memory)
        assert len(_records(engine)) == before and all(backend.open_snapshots == [{}] for backend in memory.values())


@pytest.mark.parametrize("operation", ["learn", "commit"])
@pytest.mark.parametrize("chain", ["cause", "context"])
@pytest.mark.parametrize("error_type", [MemoryError, AsyncCancelledError, FutureCancelledError, KeyboardInterrupt, SystemExit])
def test_typed_backend_wrappers_preserve_original_host_error(operation, chain, error_type):
    engine, _ = _terminal()
    original = error_type("original host stop")
    wrapper = MemoryOperationError("provider", "provider.failed", "Wrapped failure.", True)
    setattr(wrapper, f"__{chain}__", original)
    memory = _Memory(learn_outcomes=[wrapper] if operation == "learn" else [],
                     commit_outcomes=[wrapper] if operation == "commit" else [])
    with pytest.raises(error_type) as caught:
        _drive(engine, {"A": memory})
    assert caught.value is original
    assert engine.state.backward_state.finished is None and memory.close_count == 1
    assert engine.state.backward_state.policy_outcomes[0].status == ("unrequested" if operation == "learn" else "unknown")


def test_host_capability_identity_and_cleanup_error_preserve_primary_exception(monkeypatch):
    engine, _ = _terminal(("A", "B"))
    original = RuntimeError("capability failed")
    memory = {"A": _Memory(learn_outcomes=[_CapabilityFault(original)]), "B": _Memory()}
    close = _Session.close

    def fail_close(session):
        close(session)
        raise OSError("close failed")

    monkeypatch.setattr(_Session, "close", fail_close)
    with pytest.raises(RuntimeError) as caught:
        _drive(engine, memory)
    assert caught.value is original and all(backend.close_count == 1 for backend in memory.values())
    assert engine.state.backward_state.finished is None and not engine.state.backward_state.ever_requested


def test_cleanup_error_after_finished_preserves_success_and_admits_a_later_attempt(monkeypatch):
    engine, _ = _terminal()
    memory = _Memory()
    close = _Session.close

    def fail_close(session):
        close(session)
        raise OSError("close failed")

    monkeypatch.setattr(_Session, "close", fail_close)
    with pytest.raises(OSError, match="close failed"):
        _drive(engine, {"A": memory})
    assert engine.state.backward_state.finished.outcome.status == "succeeded"
    assert engine.state.backward_state.can_start and memory.close_count == 1


def test_backward_and_memory_replay_calls_no_callbacks_or_materialization(monkeypatch):
    from agent_composer.compile.expand import build_grow
    import agent_composer.runtime.learning as learning_module
    from agent_composer.runtime.state import RunState, _replay_records

    engine, _ = _terminal(("A", "B"))
    original = copy.deepcopy(engine.state.flow)
    memory = {"A": _Memory(), "B": _Memory(commit_outcomes=[
        MemoryOperationError("commit", "not.available", "Unavailable.", False),
    ])}
    _drive(engine, memory)
    for node in original.nodes.values():
        node.forward = lambda *_: pytest.fail("replayed forward")
        node.backward = lambda *_: pytest.fail("replayed backward")
        node.capture_backward_context = lambda *_: pytest.fail("replayed capture")
    monkeypatch.setattr(learning_module, "_materialize_context", lambda *_: pytest.fail("replayed context"))
    monkeypatch.setattr(MemoryRuntime, "propose_learning", lambda *_: pytest.fail("replayed learner"))
    monkeypatch.setattr(MemoryRuntime, "commit_learning", lambda *_: pytest.fail("replayed commit"))
    replayed = _replay_records(RunState(original), tuple(engine.log), grow_builder=build_grow)
    assert replayed.backward_state == engine.state.backward_state
    assert replayed.trace.occurrences == engine.state.trace.occurrences
    assert replayed.next_sequence == replayed.trace.next_sequence == engine.state.next_sequence


def test_prepare_failure_is_not_sanitized_as_a_learner_failure(monkeypatch):
    engine, _ = _terminal()
    memory = _Memory()
    original_prepare = backward_module.prepare_record
    error = MemoryOperationError("host", "prepare.failed", "Prepare failed.", False)

    def fail(state, record):
        if type(record) is BackwardReadyRecord:
            raise error
        return original_prepare(state, record)

    monkeypatch.setattr(backward_module, "prepare_record", fail)
    with pytest.raises(MemoryOperationError) as caught:
        _drive(engine, {"A": memory})
    assert caught.value is error and engine.state.backward_state.ready is None
    assert engine.state.backward_state.finished is None and not memory.commit_calls
    assert memory.close_count == 1


def test_learner_mutation_of_its_arguments_and_returned_values_cannot_change_authority():
    engine, _ = _terminal(("A", "A"))
    original_ref = engine.state.memory_layout[0].ref
    returned = MemoryMutation("first", original_ref, {"items": ["saved"]})

    def learn_first(context):
        object.__setattr__(context.frame_ref, "static_path", ("wrong",))
        return [returned]

    def learn_second(context):
        object.__setattr__(returned, "mutation_id", "changed-later")
        return []

    memory = _Memory(learn_outcomes=[learn_first, learn_second])
    outcome = _drive(engine, {"A": memory})
    assert outcome.status == "succeeded" and original_ref.static_path == ("root",)
    assert memory.commit_calls[0][1][0].mutation_id == "first"
    assert memory.commit_calls[0][1][0].payload == {"items": ["saved"]}


def test_opened_memory_runtime_subclass_is_closed_by_the_driver():
    class SpecializedRuntime(MemoryRuntime):
        pass

    engine, _ = _terminal()
    memory = _Memory()
    layout = engine.state.memory_layout
    runtime = SpecializedRuntime.open(
        layout=layout, origins={spec.ref: None for spec in layout}, memory={"A": memory},
        snapshots=engine.state.memory_snapshots,
    )
    outcome = backward_module.drive_backward(engine.state, engine.log, "review", open_memory=lambda: runtime)
    assert outcome.status == "succeeded" and memory.close_count == 1


@pytest.mark.parametrize("problem", ["learn_disabled", "missing_opener", "invalid_feedback"])
def test_submission_gates_reject_before_opening_or_appending(problem):
    engine, node = _terminal(learnable=problem != "learn_disabled")
    memory = _Memory()
    records = _records(engine)
    with pytest.raises(ValueError):
        backward_module.drive_backward(
            engine.state, engine.log, [] if problem == "invalid_feedback" else "review",
            open_memory=None if problem == "missing_opener" else lambda: _open(engine.state.memory_layout, {"A": memory}),
        )
    assert _records(engine) == records and not node.calls and not memory.open_snapshots


def test_wrapped_host_error_during_partial_setup_closes_prior_sessions(monkeypatch):
    engine, _ = _terminal(("A", "B"))
    memory = {"A": _Memory(), "B": _Memory()}
    original = MemoryError("cannot allocate")
    wrapped = MemoryOperationError("open", "wrapped", "Open failed.", False)
    wrapped.__cause__ = original
    monkeypatch.setattr(memory["B"], "open_run", lambda **_: (_ for _ in ()).throw(wrapped))
    records = _records(engine)
    with pytest.raises(MemoryError) as caught:
        _drive(engine, memory)
    assert caught.value is original and _records(engine) == records
    assert memory["A"].close_count == 1 and memory["B"].close_count == 0


@pytest.mark.parametrize("with_primary", [False, True])
@pytest.mark.parametrize("error_type", [MemoryError, AsyncCancelledError, FutureCancelledError,
                                       KeyboardInterrupt, SystemExit])
def test_cleanup_unwraps_host_failure_after_closing_all_sessions(with_primary, error_type, monkeypatch):
    engine, _ = _terminal(("A", "B", "C"))
    primary = RuntimeError("learner failed unexpectedly")
    memory = {"A": _Memory(learn_outcomes=[primary] if with_primary else []), "B": _Memory(), "C": _Memory()}
    original = error_type("host failed during close")
    wrapper = MemoryOperationError("close", "wrapped", "Close failed.", False)
    wrapper.__cause__ = original
    closed = []
    close = _Session.close

    def fail_close(session):
        close(session)
        name = next(name for name, backend in memory.items() if backend is session.memory)
        closed.append(name)
        if name == "B":
            raise wrapper

    monkeypatch.setattr(_Session, "close", fail_close)
    with pytest.raises(RuntimeError if with_primary else error_type) as caught:
        _drive(engine, memory)
    assert caught.value is (primary if with_primary else original)
    assert closed == ["C", "B", "A"] and all(backend.close_count == 1 for backend in memory.values())
    if with_primary:
        assert engine.state.backward_state.finished is None
        assert not engine.state.backward_state.ever_requested
    else:
        assert engine.state.backward_state.finished.outcome.status == "succeeded"
        assert engine.state.backward_state.can_start


@pytest.mark.parametrize("operation", ["learn", "commit"])
@pytest.mark.parametrize("chain", ["direct", "cause", "context"])
@pytest.mark.parametrize("depth", [2, 5])
def test_nested_capability_faults_never_escape_private_wrappers(operation, chain, depth):
    engine, _ = _terminal()
    original = RuntimeError("original capability failure")
    nested = original
    for _ in range(depth):
        nested = _CapabilityFault(nested)
    if chain == "direct":
        wrapper = nested
    else:
        wrapper = MemoryOperationError("backend", "wrapped", "Known wrapper.", False)
        setattr(wrapper, f"__{chain}__", nested)
    memory = _Memory(learn_outcomes=[wrapper] if operation == "learn" else [],
                     commit_outcomes=[wrapper] if operation == "commit" else [])
    with pytest.raises(RuntimeError) as caught:
        _drive(engine, {"A": memory})
    assert caught.value is original and memory.close_count == 1
    assert engine.state.backward_state.finished is None
    assert engine.state.backward_state.policy_outcomes[0].status == ("unrequested" if operation == "learn" else "unknown")


@pytest.mark.parametrize("chain", ["cause", "context"])
def test_capability_originals_are_checked_for_wrapped_host_controls(chain):
    engine, _ = _terminal()
    original = MemoryError("original allocation failure")
    wrapped = MemoryOperationError("backend", "wrapped", "Backend wrapper.", False)
    setattr(wrapped, f"__{chain}__", original)
    memory = _Memory(learn_outcomes=[_CapabilityFault(_CapabilityFault(wrapped))])
    with pytest.raises(MemoryError) as caught:
        _drive(engine, {"A": memory})
    assert caught.value is original and memory.close_count == 1
    assert engine.state.backward_state.finished is None


def test_cyclic_capability_fault_is_a_host_error_without_exposing_private_carriers():
    engine, _ = _terminal()
    malformed = _CapabilityFault(RuntimeError("unreachable"))
    malformed.original = malformed
    memory = _Memory(learn_outcomes=[malformed])
    with pytest.raises(RuntimeError, match="cyclic capability fault"):
        _drive(engine, {"A": memory})
    assert engine.state.backward_state.finished is None and memory.close_count == 1


@pytest.mark.parametrize("target", ["mutation", "frame_ref", "inner_payload", "public_payload"])
@pytest.mark.parametrize("completion", ["success", "known_failure", "host_failure"])
def test_commit_batch_is_detached_from_every_accepted_authority(target, completion, monkeypatch):
    engine, _ = _terminal()
    original_frame = copy.deepcopy(engine.state.memory_layout[0].ref)
    known = MemoryOperationError("commit", "refused", "Commit refused.", False)
    host = MemoryError("commit interrupted")
    memory = _Memory(
        learn_outcomes=[lambda context: [MemoryMutation("proposal", context.frame_ref, {"notes": ["saved"]})]],
        commit_outcomes=[known] if completion == "known_failure" else [host] if completion == "host_failure" else [],
    )
    commit = _Session.commit

    def mutate(session, commit_id, mutations):
        accepted = engine.state.backward_state.current.requests[-1]
        recorded = _records(engine)[-1]
        assert type(recorded) is MemoryCommitRequestedRecord and accepted == recorded
        mutation = mutations[0]
        if target == "mutation":
            object.__setattr__(mutation, "mutation_id", "backend-changed")
        elif target == "frame_ref":
            object.__setattr__(mutation.frame_ref, "static_path", ("root", "foreign"))
        elif target == "inner_payload":
            object.__setattr__(mutation._payload.entries[0][1].values[0], "value", "backend-changed")
        else:
            mutation.payload["notes"].append("backend-copy")
        assert accepted == recorded
        return commit(session, commit_id, mutations)

    monkeypatch.setattr(_Session, "commit", mutate)
    if completion == "host_failure":
        with pytest.raises(MemoryError) as caught:
            _drive(engine, {"A": memory})
        assert caught.value is host and engine.state.backward_state.finished is None
    else:
        outcome = _drive(engine, {"A": memory})
        assert outcome.status == ("succeeded" if completion == "success" else "failed")
    backward = engine.state.backward_state
    request = backward.current.requests[0]
    assert request == next(record for record in _records(engine) if type(record) is MemoryCommitRequestedRecord)
    assert backward.ready == next(record for record in _records(engine) if type(record) is BackwardReadyRecord)
    assert request.mutations[0].mutation_id == "proposal"
    assert request.mutations[0].frame_ref == original_frame == engine.state.memory_layout[0].ref
    assert request.mutations[0].payload == {"notes": ["saved"]}
    assert backward.policy_outcomes[0].status == {
        "success": "succeeded", "known_failure": "failed", "host_failure": "unknown",
    }[completion]
    assert engine.state.trace.next_sequence == engine.state.next_sequence == engine.log.next_sequence
    assert len(memory.commit_calls) == 1 and memory.close_count == 1
