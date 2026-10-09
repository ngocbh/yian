"""Reverse computation and explicit, record-authorized backward attempts.

The fold consumes recorded routes and node-owned backward hooks. It does not
interpret kinds, evaluate expressions, drive child flows, or write logs/memory.
Small durable attribution references live in runtime.contracts below this module;
materialized interpretation data and ordering keys remain transient. The shared
learning projection owns formatting without performing this callback fold.
The internal attempt driver surrounds that pure computation with accepted records
and injected memory resources. It never resumes proposals or invokes public run
lifecycle operations.
"""

from __future__ import annotations

import traceback
from typing import Callable

from yian.expr.lineage import FeedbackValue
from yian.memory import LearningFailure, MemoryOperationFailure
from yian.nodes.base import (
    BackwardContext, BackwardResult,
    _copy_backward_value, validate_backward_result,
)
from yian.runtime.contracts import (
    AttributedBackwardResult, BackwardFailure, BackwardOutcome,
)
from yian.runtime.trace import RetainedTrace
from yian.runtime.learning import (
    _contextual_transfers, _materialize_context, _project_root_backward, _seed_backward,
    _prefix_learning, _proposal_batches, OwnerContextError, memory_commit_id, project_owner_contexts,
)
from yian.runtime.log import (
    BackwardFinishedRecord, BackwardReadyRecord, BackwardStartedRecord,
    MemoryCommitFailedRecord, MemoryCommitRequestedRecord, MemoryCommitSucceededRecord, RunLog, RunRecord,
)
from yian.runtime.memory_runtime import MemoryRuntime, _LearningProposalError, _raise_host_error
from yian.runtime.state import RunState, commit_prepared, prepare_record


class _BackwardComputeError(Exception):
    """Attribute a callback or result-contract failure without changing forward state."""

    def __init__(self, failure: BackwardFailure) -> None:
        super().__init__(failure.message)
        self.failure = failure


def _failure(context: BackwardContext, code: str, exc: Exception) -> _BackwardComputeError:
    return _BackwardComputeError(BackwardFailure(
        stage="backward", code=code, message=str(exc), scope=context.scope,
        run_id=context.run_id, node_id=context.node_id, node_started_seq=context.node_started_seq,
        cause_type=type(exc).__name__, traceback="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
    ))


def compute_backward(
    trace: RetainedTrace, root_feedback: FeedbackValue,
) -> tuple[BackwardResult, tuple[AttributedBackwardResult, ...]]:
    """Project root feedback through one accepted-outcome reverse walk.

    Args:
        trace (`RetainedTrace`): A retained terminal execution with private
            backward callbacks.
        root_feedback (`FeedbackValue`): One ordinary text, record, positional
            list, or None value selecting the actual terminal output. Failed
            terminals accept text or None.

    Returns:
        `tuple[BackwardResult, tuple[AttributedBackwardResult, ...]]`: Root input
            feedback and results in accepted-outcome order, including unreached
            occurrences whose callbacks were not invoked.

    Raises:
        `TypeError`: For malformed feedback or trace carriers.
        `ValueError`: For an unfinished trace, invalid feedback shape, or broken
            retained route/source authority.
        `_BackwardComputeError`: For an attributed callback or result-contract error.
            Host capability faults and cancellation propagate unchanged.
    """
    pending, direct_root = _seed_backward(trace, root_feedback)
    occurrences = trace.occurrences
    results = {item.source: AttributedBackwardResult(item.source, (), (), BackwardResult(), False)
               for item in occurrences}
    root_inputs: dict[str, tuple[FeedbackValue, ...]] = {}
    for occurrence in reversed(occurrences):
        messages = sorted(pending.pop(occurrence.source, ()), key=lambda item: item.order)
        if not messages:
            continue
        context = trace.backward_context(
            occurrence.source,
            downstream_context=tuple(_materialize_context(trace, message, i) for i, message in enumerate(messages)),
        )
        received = tuple(item.feedback for item in messages)
        feedback = tuple(_copy_backward_value(message, "feedback") for message in received)
        handler = trace.backward_handlers[occurrence.source]
        try:
            result = handler(context, feedback)
        except Exception as exc:
            _raise_host_error(exc)
            raise _failure(context, "backward.execute", exc) from exc
        try:
            validate_backward_result(result, context)
        except (TypeError, ValueError) as exc:
            raise _failure(context, "backward.result", exc) from exc
        results[occurrence.source] = AttributedBackwardResult(
            occurrence.source, received, tuple(item.attribution for item in messages), result, True,
        )
        input_feedback = result.input_feedback
        if occurrence.source == trace.root_start_source:
            root_inputs = input_feedback
        for source, transfer in _contextual_transfers(trace, occurrence, context, messages, result):
            pending.setdefault(source, []).append(transfer)
    root = _project_root_backward(trace, pending, root_inputs, direct_root)
    return root, tuple(results[item.source] for item in occurrences)


def _accept_backward(state: RunState, log: RunLog, record: RunRecord) -> None:
    """Append and reduce one exact observer-free backward transition."""
    prepared = prepare_record(state, record)
    if commit_prepared(log, state, prepared):
        raise RuntimeError("backward records cannot emit observers")


def _finish_backward(
    state: RunState, log: RunLog, outcome: BackwardOutcome,
    attributed: tuple[AttributedBackwardResult, ...] = (),
) -> BackwardOutcome:
    """Return only the outcome of an accepted Finished record."""
    _accept_backward(state, log, BackwardFinishedRecord(
        state.backward_state.started.attempt_id, outcome, attributed,
    ))
    return state.backward_state.finished.outcome


def _drive_opened_backward(state: RunState, log: RunLog, memory: MemoryRuntime | None) -> BackwardOutcome:
    """Compute once, stage every memory, then request and observe each memory key in order."""
    started = state.backward_state.started
    try:
        root, attributed = compute_backward(state.trace, started.feedback)
    except _BackwardComputeError as exc:
        _raise_host_error(exc)
        return _finish_backward(state, log, BackwardOutcome(
            "failed", failure=exc.failure, learning=_prefix_learning(state.backward_state),
        ))
    try:
        contexts = project_owner_contexts(
            state.trace, state.memories or (), state.memory_stacks or (),
            started.feedback, root, attributed,
        )
        proposals = memory.propose_learning(contexts) if memory is not None else ()
    except (OwnerContextError, _LearningProposalError) as exc:
        _raise_host_error(exc)
        if type(exc) is OwnerContextError:
            _raise_host_error(exc.original)
            cause = MemoryOperationFailure(
                "learn", "memory.invalid_context", "memory learn context could not be constructed", False,
            )
        else:
            cause = exc.failure
        key = exc.memory
        failure = BackwardFailure("learn", cause.code, cause.message, memory=key, cause=cause)
        learning = _prefix_learning(state.backward_state, learner_failure=LearningFailure(
            key, "learn", cause,
        ))
        return _finish_backward(state, log, BackwardOutcome("failed", root, failure, learning), attributed)

    _accept_backward(state, log, BackwardReadyRecord(started.attempt_id, root, attributed, proposals))
    backward = state.backward_state
    for key, batch in zip(backward.memory_order, _proposal_batches(backward, backward.ready)):
        commit_id = memory_commit_id(state.run_id, key, batch, backward.commit_attempt_key)
        _accept_backward(state, log, MemoryCommitRequestedRecord(
            key, commit_id, batch, attempt_id=started.attempt_id,
        ))
        cause = memory.commit_learning(state.backward_state.current.requests[-1])
        if cause is not None:
            _accept_backward(state, log, MemoryCommitFailedRecord(
                key, commit_id, cause, attempt_id=started.attempt_id,
            ))
            failure = BackwardFailure("commit", cause.code, cause.message, memory=key, cause=cause)
            return _finish_backward(state, log, BackwardOutcome(
                "failed", root, failure, _prefix_learning(state.backward_state),
            ))
        _accept_backward(state, log, MemoryCommitSucceededRecord(
            key, commit_id, attempt_id=started.attempt_id,
        ))
    return _finish_backward(state, log, BackwardOutcome(
        "succeeded", root, learning=_prefix_learning(state.backward_state, succeeded=True),
    ))


def drive_backward(
    state: RunState, log: RunLog, root_feedback: FeedbackValue,
    *, open_memory: Callable[[], MemoryRuntime] | None = None,
) -> BackwardOutcome:
    """Drive one new explicit whole backward attempt against an accepted terminal run.

    Every call starts a new attempt, including after a successful, failed, or
    committing earlier attempt.

    Args:
        state (`RunState`): Verified retained forward and backward record authority.
        log (`RunLog`): The open append authority for that state.
        root_feedback (`FeedbackValue`): Ordinary feedback on the recorded terminal.
        open_memory (`Callable[[], MemoryRuntime] | None`): Fresh resource opener
            using the recorded memory table and snapshots; required only for
            enabled memories. This driver closes the returned runtime on every exit.

    Returns:
        `BackwardOutcome`: The exact accepted Finished disposition. An unfinished
            earlier attempt, including one with unconfirmed commit requests, is
            first closed as interrupted; all callbacks and proposals are recomputed.

    Raises:
        `TypeError`, `ValueError`: For invalid feedback, terminal state, or resources.
        `RuntimeError`: For mismatched log/state authority. Host, setup, cancellation,
            unexpected backend, and append/reducer faults escape with their accepted
            prefix intact. Cleanup never manufactures a completion record.
    """
    if type(state) is not RunState or not isinstance(log, RunLog):
        raise TypeError("backward driving requires a RunState and RunLog")
    started = BackwardStartedRecord(len(state.backward_state.attempts) + 1, root_feedback)
    _seed_backward(state.trace, started.feedback)
    backward = state.backward_state
    interrupted = backward.started is not None and backward.finished is None
    first_record = (BackwardFinishedRecord(backward.started.attempt_id, backward.outcome)
                    if interrupted else started)
    prepared = prepare_record(state, first_record)
    memory = None
    primary_error = None
    try:
        if backward.memory_order:
            if not callable(open_memory):
                raise ValueError("enabled backward memories require a fresh memory runtime")
            opened = open_memory()
            if not isinstance(opened, MemoryRuntime):
                raise TypeError("backward memory opener must return a MemoryRuntime")
            memory = opened
            if (memory._closed or memory.memories != state.memories
                    or memory.snapshots != state.memory_snapshots):
                raise ValueError("opened memory runtime does not match the recorded memory table and snapshots")
        if commit_prepared(log, state, prepared):
            raise RuntimeError("backward records cannot emit observers")
        if interrupted:
            _accept_backward(state, log, started)
        return _drive_opened_backward(state, log, memory)
    except BaseException as exc:
        primary_error = exc
        try:
            _raise_host_error(exc)
        except BaseException as original:
            primary_error = original
            raise
        raise
    finally:
        if memory is not None:
            try:
                memory._close_preserving(primary_error)
            except BaseException as exc:
                _raise_host_error(exc)
                raise


__all__ = ["compute_backward", "drive_backward"]
