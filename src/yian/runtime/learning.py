"""Pure backward transfers and accepted backward/memory reducer projections.

This module shares exact route-transfer validation with the backward callback
driver and owns immutable attempt state. It projects owner-local contexts, the
only episode projection, for every enabled memory key from complete accepted
history and backward results.
Contracts, records, and retained traces sit below it; state and callback drivers
consume its pure projections. It invokes no callbacks or backend operations.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
import json
from typing import Any

from yian.diagnostics import Failure
from yian.expr.lineage import FeedbackValue, RouteContext, SourceRef, has_feedback, route_feedback, validate_feedback
from yian.memory import (
    DeclSite,
    LearnContext,
    LearningFailure,
    LearningResult,
    MemoryEpisode,
    MemoryEpisodeStatus,
    MemoryMutation,
    MemoryOperationFailure,
    MemorySpec,
    MemoryStack,
    PolicyCommitOutcome,
)
from yian.nodes.base import BackwardContext, BackwardResult, _copy_backward_value, validate_backward_result
from yian.runtime.contracts import AttributedBackwardResult, BackwardAttribution, BackwardFailure, BackwardOutcome
from yian.runtime.log import (
    BackwardStartedRecord,
    BackwardReadyRecord,
    BackwardFinishedRecord,
    MemoryCommitRequestedRecord,
    MemoryCommitSucceededRecord,
    MemoryCommitFailedRecord,
    NodeExpandedRecord,
    NodeFailedRecord,
    NodeRoutedRecord,
    NodeSucceededRecord,
    _copy_attributed_results,
    _copy_backward_result,
    _execution_scope_payload,
    _revalidate_run_record,
    _route_payload,
    canonical_seed,
)
from yian.runtime.trace import RetainedTrace, TraceOccurrence, TraceTerminal

_ROOT_FLOW_SITE = DeclSite((), ("root",), None)


@dataclass(frozen=True, slots=True)
class _BackwardMessage:
    """Transient ordered transfer, with contexts materialized only by consumers."""

    feedback: FeedbackValue
    attribution: BackwardAttribution
    order: tuple[int, ...]
    context: tuple[RouteContext, ...] = ()


def _source_data(source: SourceRef | None) -> dict[str, Any] | None:
    """Project a reference without materializing its accepted value."""
    if source is None:
        return None
    return {"run_id": source.run_id, "execution_seq": source.execution_seq, "path": list(source.path)}


def _route_context_data(trace: RetainedTrace, contexts: tuple[RouteContext, ...]) -> list[dict[str, Any]]:
    """Resolve accepted values while preserving unavailable temporary snapshots."""
    details = []
    for context in contexts:
        sources = [{"source": _source_data(source), "value": trace.lookup_source(source)}
                   for source in context.sources]
        result = context.result
        if context.result_json is None and context.result_type is None and len(sources) == 1:
            result = sources[0]["value"]
        details.append({
            "expression": context.expression, "result": result, "sources": sources,
            "role": context.role,
            "operand": list(context.operand) if type(context.operand) is tuple else context.operand,
            "positions": [list(position) for position in context.positions],
            "span": list(context.span) if context.span is not None else None,
            "result_type": context.result_type,
            "snapshot": _route_payload(context.snapshot) if context.snapshot is not None else None,
        })
    return details


def _materialize_context(trace: RetainedTrace, message: _BackwardMessage, index: int) -> dict[str, Any]:
    """Resolve only the accepted values needed by one transient callback view."""
    attribution = message.attribution
    return {"message_index": index, "sender": _source_data(attribution.sender),
            "input_name": attribution.input_name, "sender_message_index": attribution.message_index,
            "contribution_index": attribution.contribution_index, "parents": list(attribution.parents),
            "contexts": _route_context_data(trace, message.context)}


def _output_context(
    source: SourceRef, context: BackwardContext, name: str, has_output: bool,
) -> tuple[RouteContext, ...]:
    """Reference an accepted structural output without rebuilding or copying it."""
    if not has_output:
        return ()
    path = context.node_data.get("input_output_paths", {}).get(name)
    if path is None:
        return ()
    value = context.output
    for part in path:
        value = value[part]
    positions = ((path[0],),) if path and type(path[0]) is int else ()
    return (RouteContext("", None, (source,), "output", tuple(path), positions),)


def _inherited_context(
    occurrence: TraceOccurrence, context: BackwardContext, messages: list[_BackwardMessage],
    parents: tuple[int, ...], name: str, has_output: bool,
) -> tuple[RouteContext, ...]:
    """Preserve ordered parent contexts before adding actual structural output."""
    inherited = tuple(detail for index in parents for detail in messages[index].context)
    return (*inherited, *_output_context(occurrence.source, context, name, has_output))


def _contextual_transfers(
    trace: RetainedTrace, occurrence: TraceOccurrence, context: BackwardContext,
    messages: list[_BackwardMessage], result: BackwardResult,
) -> tuple[tuple[SourceRef, _BackwardMessage], ...]:
    """Attach transient inherited/output context after the pure transfer step."""
    transfers = []
    for source, transfer in _transfer_backward(trace, occurrence, context, messages, result):
        attribution = transfer.attribution
        inherited = _inherited_context(occurrence, context, messages, attribution.parents,
                                       attribution.input_name, occurrence.source in trace.sources)
        transfers.append((source, replace(transfer, context=(*inherited, *transfer.context))))
    return tuple(transfers)


def _seed_backward(
    trace: RetainedTrace, feedback: FeedbackValue,
) -> tuple[dict[SourceRef, list[_BackwardMessage]], BackwardResult]:
    """Validate the whole terminal message before deciding whether to seed work."""
    if type(trace) is not RetainedTrace or type(trace.terminal) is not TraceTerminal:
        raise ValueError("backward requires a retained terminal trace")
    terminal = trace.terminal
    TraceTerminal.__post_init__(terminal)
    validate_feedback(feedback, terminal.output)
    feedback = _copy_backward_value(feedback, "root_feedback")
    if not has_feedback(feedback):
        return {}, BackwardResult()
    if terminal.source is None:
        root = BackwardResult(self_feedback=(feedback,)) if not terminal.succeeded else BackwardResult()
        return {}, root
    return {terminal.source: [_BackwardMessage(
        feedback, BackwardAttribution(None, None, 0, None, ()), (0,),
    )]}, BackwardResult()


def _structural_parents(
    context: BackwardContext, messages: list[_BackwardMessage], result: BackwardResult,
    has_output: bool,
) -> dict[str, tuple[int, ...]] | None:
    """Verify metadata-prescribed exact transfer before assigning incoming ordinals."""
    paths = context.node_data.get("input_output_paths")
    inputs = context.inputs
    if paths is None or inputs is None or (not has_output and context.failure is None):
        return None
    expected = {}
    parents = {}
    for name in context.input_names:
        if name not in paths or name not in inputs:
            continue
        if has_output:
            value = context.output
            for part in paths[name]:
                value = value[part]
        selected = []
        indices = []
        for index, message in enumerate(messages):
            feedback = message.feedback
            for part in paths[name]:
                if feedback is None or type(feedback) is str:
                    break
                feedback = feedback.get(part) if type(feedback) is dict else feedback[part]
            if has_feedback(feedback):
                selected.append(feedback)
                indices.append(index)
        if selected:
            expected[name] = tuple(selected)
            parents[name] = tuple(indices)
    return parents if result.input_feedback == expected else None


def _transfer_backward(
    trace: RetainedTrace, occurrence: TraceOccurrence, context: BackwardContext,
    messages: list[_BackwardMessage], result: BackwardResult,
) -> tuple[tuple[SourceRef, _BackwardMessage], ...]:
    """Derive exact transfers from accepted routes without invoking or formatting."""
    structural = _structural_parents(context, messages, result, occurrence.source in trace.sources)
    routes = dict(zip(occurrence.input_route.names, occurrence.input_route.children))
    transfers = []
    for input_index, name in enumerate(context.input_names):
        for message_index, feedback in enumerate(result.input_feedback.get(name, ())):
            parents = ((structural[name][message_index],) if structural is not None
                       else tuple(range(len(messages))))
            routed = route_feedback(routes[name], feedback, trace.lookup_source,
                                    recorded_value=context.inputs[name])
            for contribution_index, contribution in enumerate(routed.contributions):
                attribution = BackwardAttribution(occurrence.source, name, message_index,
                                                  contribution_index, parents)
                order = (*messages[parents[0]].order, input_index, message_index, contribution_index)
                transfers.append((contribution.source, _BackwardMessage(
                    contribution.feedback, attribution, order, contribution.context,
                )))
    return tuple(transfers)


def _project_root_backward(
    trace: RetainedTrace, pending: dict[SourceRef, list[_BackwardMessage]],
    root_inputs: dict[str, tuple[FeedbackValue, ...]], direct_root: BackwardResult,
) -> BackwardResult:
    """Project validated START and actual source-zero fields without duplication."""
    source_zero = SourceRef(trace.run_id, 0)
    messages = sorted(pending.pop(source_zero, ()), key=lambda item: item.order)
    if messages:
        initial = trace.lookup_source(source_zero)
        for message in messages:
            if message.attribution.sender == trace.root_start_source and trace.root_start_source is not None:
                continue
            for name in trace.root_input_names:
                if name not in initial:
                    continue
                selected = message.feedback if type(message.feedback) is str else message.feedback.get(name)
                if has_feedback(selected):
                    root_inputs[name] = (*root_inputs.get(name, ()), selected)
    if pending:
        raise ValueError("feedback targets an unavailable or later accepted occurrence")
    return BackwardResult(root_inputs, direct_root.self_feedback)


def validate_attributed_backward(
    trace: RetainedTrace, submitted_feedback: FeedbackValue,
    root_result: BackwardResult, attributed_results: tuple[AttributedBackwardResult, ...],
) -> None:
    """Validate a complete reverse result against exact accepted transfer streams.

    Args:
        trace (`RetainedTrace`): Accepted terminal forward authority.
        submitted_feedback (`FeedbackValue`): Original complete root message.
        root_result (`BackwardResult`): Submitted root projection.
        attributed_results (`tuple[AttributedBackwardResult, ...]`): Every accepted
            occurrence in outcome order, including unreached empty results.

    Raises:
        `TypeError`: For malformed carriers, channels, or message containers.
        `ValueError`: For inventory, transfer, local contract, or root disagreement.
    """
    pending, direct_root = _seed_backward(trace, submitted_feedback)
    root_result = _copy_backward_result(root_result)
    attributed_results = _copy_attributed_results(attributed_results)
    occurrences = trace.occurrences
    if tuple(item.source for item in attributed_results) != tuple(item.source for item in occurrences):
        raise ValueError("backward results must match the complete accepted occurrence order")
    root_inputs = {}
    for occurrence, item in reversed(tuple(zip(occurrences, attributed_results))):
        messages = sorted(pending.pop(occurrence.source, ()), key=lambda message: message.order)
        if (item.feedback != tuple(message.feedback for message in messages)
                or item.attribution != tuple(message.attribution for message in messages)
                or item.invoked is not bool(messages)):
            raise ValueError("backward received feedback and attribution must match exact transfers")
        context = trace.backward_context(occurrence.source)
        validate_backward_result(item.result, context)
        if not messages:
            continue
        if occurrence.source == trace.root_start_source:
            root_inputs = item.result.input_feedback
        for source, transfer in _transfer_backward(trace, occurrence, context, messages, item.result):
            pending.setdefault(source, []).append(transfer)
    if root_result != _project_root_backward(trace, pending, root_inputs, direct_root):
        raise ValueError("backward root result must match the accepted START and source-zero projection")


@dataclass(frozen=True, slots=True)
class _BackwardAttempt:
    """Accepted computation and commit facts for one explicit attempt."""

    started: BackwardStartedRecord
    ready: BackwardReadyRecord | None = None
    finished: BackwardFinishedRecord | None = None
    requests: tuple[MemoryCommitRequestedRecord, ...] = ()
    observations: tuple[MemoryCommitSucceededRecord | MemoryCommitFailedRecord, ...] = ()

    def __post_init__(self) -> None:
        for record, carrier in ((self.started, BackwardStartedRecord),
                                (self.ready, BackwardReadyRecord), (self.finished, BackwardFinishedRecord)):
            if record is None and carrier is not BackwardStartedRecord:
                continue
            if type(record) is not carrier:
                raise TypeError("backward attempt requires exact accepted records")
            _revalidate_run_record(record)
            if record.attempt_id != self.started.attempt_id:
                raise ValueError("backward attempt record identities disagree")
        for records, carriers in ((self.requests, (MemoryCommitRequestedRecord,)),
                                  (self.observations, (MemoryCommitSucceededRecord, MemoryCommitFailedRecord))):
            if type(records) is not tuple:
                raise TypeError("backward commit records must be immutable tuples")
            for record in records:
                if type(record) not in carriers:
                    raise TypeError("backward commit requires an exact accepted record")
                _revalidate_run_record(record)
                if record.attempt_id != self.started.attempt_id:
                    raise ValueError("backward commit attempt identities disagree")


def _interrupted_failure() -> BackwardFailure:
    return BackwardFailure("backward", "backward.interrupted", "backward attempt interrupted before completion")


def _policy_outcomes(memory_order: tuple[str, ...], attempt: _BackwardAttempt) -> tuple[PolicyCommitOutcome, ...]:
    requests = {record.memory: record for record in attempt.requests}
    observations = {record.memory: record for record in attempt.observations}
    outcomes = []
    for name in memory_order:
        request, observation = requests.get(name), observations.get(name)
        if request is None:
            outcomes.append(PolicyCommitOutcome(name, "unrequested"))
        elif observation is None:
            outcomes.append(PolicyCommitOutcome(name, "unknown", request.commit_id))
        elif type(observation) is MemoryCommitSucceededRecord:
            outcomes.append(PolicyCommitOutcome(name, "succeeded", request.commit_id))
        else:
            outcomes.append(PolicyCommitOutcome(name, "failed", request.commit_id, observation.failure))
    return tuple(outcomes)


def _prefix_learning(
    backward: RuntimeBackwardState, *, succeeded: bool = False, learner_failure: LearningFailure | None = None,
) -> LearningResult | None:
    if not backward.memory_order:
        return None
    policies = backward.policy_outcomes
    failures = tuple(LearningFailure(item.memory, "commit", item.failure)
                     for item in policies if item.status == "failed")
    if learner_failure is not None:
        failures = (learner_failure, *failures)
    status = "failed" if failures else "succeeded" if succeeded else "incomplete"
    return LearningResult(status, failures, policies)


@dataclass(frozen=True, slots=True)
class RuntimeBackwardState:
    """Immutable accepted backward attempt history and per-memory dispositions.

    ``memory_order`` shares the RunStarted authority root. Completed proposals
    remain inspectable; only new records can advance an attempt. No prefix
    view resumes work or invokes a callback.

    Attributes:
        memory_order (`tuple[str, ...]`): Recorded memory keys in table order.
        attempts (`tuple[_BackwardAttempt, ...]`): Accepted immutable attempt
            history; a later start never inherits earlier proposals.
    """

    memory_order: tuple[str, ...] = ()
    attempts: tuple[_BackwardAttempt, ...] = ()

    def __post_init__(self) -> None:
        """Reject a malformed key order and impossible accepted-history shapes."""
        if type(self.memory_order) is not tuple or any(
            type(key) is not str or not key for key in self.memory_order
        ):
            raise TypeError("backward memory order must be a tuple of non-empty strings")
        if self.memory_order != tuple(sorted(set(self.memory_order))):
            raise ValueError("backward memory order must be sorted and unique")
        if type(self.attempts) is not tuple:
            raise TypeError("backward history must be an immutable tuple")
        for index, attempt in enumerate(self.attempts):
            if type(attempt) is not _BackwardAttempt:
                raise TypeError("backward history requires exact attempt states")
            _BackwardAttempt.__post_init__(attempt)
            if attempt.started.attempt_id != index + 1:
                raise ValueError("backward attempt IDs must follow accepted start order")
            if index < len(self.attempts) - 1 and attempt.finished is None:
                raise ValueError("an unfinished attempt cannot precede another start")
            if tuple(record.memory for record in attempt.requests) != self.memory_order[:len(attempt.requests)]:
                raise ValueError("backward requests must follow recorded memory order")
            if len(attempt.observations) > len(attempt.requests) or len(attempt.requests) > len(attempt.observations) + 1:
                raise ValueError("backward requests and observations must alternate")
            for result_index, observation in enumerate(attempt.observations):
                request = attempt.requests[result_index]
                if observation.memory != request.memory or observation.commit_id != request.commit_id:
                    raise ValueError("backward observation must match its accepted request")
                if type(observation) is MemoryCommitFailedRecord and result_index < len(attempt.requests) - 1:
                    raise ValueError("a failed commit cannot be followed by another request")
            if attempt.requests and attempt.ready is None:
                raise ValueError("backward requests require accepted Ready proposals")

    def _from_validated(self, attempts: tuple[_BackwardAttempt, ...]) -> RuntimeBackwardState:
        """Replace history whose one changed attempt the reducer already checked."""
        # Rechecking every earlier attempt on each record would make the Nth
        # attempt cost O(N) and a replayed log O(N^2).
        value = object.__new__(type(self))
        object.__setattr__(value, "memory_order", self.memory_order)
        object.__setattr__(value, "attempts", attempts)
        return value

    @property
    def current(self) -> _BackwardAttempt | None:
        """Return the most recently accepted attempt, including a closed one."""
        return self.attempts[-1] if self.attempts else None

    @property
    def started(self) -> BackwardStartedRecord | None:
        """Return the current attempt's exact accepted submission."""
        return self.current.started if self.current is not None else None

    @property
    def ready(self) -> BackwardReadyRecord | None:
        """Return completed results/proposals without treating them as resumable."""
        return self.current.ready if self.current is not None else None

    @property
    def finished(self) -> BackwardFinishedRecord | None:
        """Return the accepted closing record, if this attempt was closed."""
        return self.current.finished if self.current is not None else None

    @property
    def ever_requested(self) -> bool:
        """Report any accepted commit request, independent of later observations."""
        return any(attempt.requests for attempt in self.attempts)

    @property
    def commit_attempt_key(self) -> int | None:
        """Return the attempt ID folded into the current attempt's commit IDs."""
        return None if self.current is None else self.current.started.attempt_id

    @property
    def policy_outcomes(self) -> tuple[PolicyCommitOutcome, ...]:
        """Project each configured memory from accepted requests and observations."""
        if self.current is None:
            return tuple(PolicyCommitOutcome(name, "unrequested") for name in self.memory_order)
        return _policy_outcomes(self.memory_order, self.current)

    @property
    def can_start(self) -> bool:
        """Report whether a new attempt may start; terminal/learn gates remain state-owned.

        Any finished attempt, whatever its outcome or commit requests, admits a
        later start. Only an unfinished attempt blocks one.
        """
        return self.current is None or self.finished is not None

    @property
    def outcome(self) -> BackwardOutcome | None:
        """Inspect the accepted Finished or derive interruption without appending."""
        if self.current is None:
            return None
        if self.finished is not None:
            return self.finished.outcome
        return BackwardOutcome("failed", self.ready.root_result if self.ready else None,
                               _interrupted_failure(), _prefix_learning(self))


def _proposal_batches(
    backward: RuntimeBackwardState, ready: BackwardReadyRecord,
) -> tuple[tuple[MemoryMutation, ...], ...]:
    """Return each recorded memory's batch in table order, including empty ones."""
    if tuple(proposal.memory for proposal in ready.proposals) != backward.memory_order:
        raise ValueError("memory proposals must cover the complete recorded memory table in order")
    return tuple(proposal.mutations for proposal in ready.proposals)


def _validate_backward_finished(
    trace: RetainedTrace, backward: RuntimeBackwardState, record: BackwardFinishedRecord,
) -> None:
    """Require completed feedback and failure dispositions from accepted authority."""
    current, outcome = backward.current, record.outcome
    ready = current.ready
    if ready is not None:
        if outcome.feedback != ready.root_result:
            raise ValueError("Finished feedback must match accepted Ready")
        if record.attributed_results and record.attributed_results != ready.attributed_results:
            raise ValueError("Finished attribution must match accepted Ready")
    if outcome.status == "succeeded":
        if ready is None or any(policy.status != "succeeded" for policy in backward.policy_outcomes):
            raise ValueError("successful Finished requires Ready and every memory confirmation")
        if outcome.learning != _prefix_learning(backward, succeeded=True):
            raise ValueError("successful Finished learning must match accepted memory confirmations")
        return
    failure = outcome.failure
    if failure == _interrupted_failure():
        if outcome != backward.outcome or (ready is None and record.attributed_results):
            raise ValueError("interrupted Finished must exactly project the accepted prefix")
        return
    if ready is not None:
        failed = tuple(policy for policy in backward.policy_outcomes if policy.status == "failed")
        if len(failed) != 1:
            raise ValueError("failed Finished after Ready requires a recorded commit failure")
        policy = failed[0]
        expected = BackwardFailure("commit", policy.failure.code, policy.failure.message,
                                   memory=policy.memory, cause=policy.failure)
        if failure != expected or outcome.learning != _prefix_learning(backward):
            raise ValueError("commit Finished must match the exact accepted failure and memory states")
        return
    if failure.stage == "learn":
        if type(failure.cause) is not MemoryOperationFailure or failure.memory not in backward.memory_order:
            raise ValueError("learner failure requires one recorded memory key and memory-operation cause")
        expected = BackwardFailure("learn", failure.cause.code, failure.cause.message,
                                   memory=failure.memory, cause=failure.cause)
        if failure != expected or outcome.feedback is None:
            raise ValueError("learner Finished must retain the exact attributed failure and completed feedback")
        validate_attributed_backward(trace, current.started.feedback, outcome.feedback, record.attributed_results)
        known = LearningFailure(failure.memory, "learn", failure.cause)
        if outcome.learning != _prefix_learning(backward, learner_failure=known):
            raise ValueError("learner Finished learning must contain its exact known failure")
        return
    if failure.stage != "backward" or outcome.feedback is not None or record.attributed_results:
        raise ValueError("incomplete backward computation cannot carry completed or partial results")
    _seed_backward(trace, current.started.feedback)
    if not has_feedback(current.started.feedback) or failure.code not in ("backward.execute", "backward.result"):
        raise ValueError("backward computation failure requires effective feedback and a driver failure code")
    if (failure.run_id != trace.run_id or failure.node_started_seq is None
            or failure.memory is not None):
        raise ValueError("backward computation failure requires accepted occurrence attribution")
    source = SourceRef(trace.run_id, failure.node_started_seq)
    try:
        trace.occurrence(source)
    except KeyError:
        raise ValueError("backward computation failure requires an accepted occurrence") from None
    context = trace.backward_context(source)
    if failure.node_id != context.node_id or failure.scope != context.scope:
        raise ValueError("backward computation failure must match its recorded invocation")
    if outcome.learning != _prefix_learning(backward):
        raise ValueError("failed computation learning must retain incomplete unrequested memories")


def reduce_backward_record(
    trace: RetainedTrace, backward: RuntimeBackwardState,
    record: BackwardStartedRecord | BackwardReadyRecord | BackwardFinishedRecord
            | MemoryCommitRequestedRecord | MemoryCommitSucceededRecord | MemoryCommitFailedRecord,
) -> RuntimeBackwardState:
    """Validate and project one accepted backward record without effects.

    Args:
        trace (`RetainedTrace`): Recorded terminal forward authority.
        backward (`RuntimeBackwardState`): Immutable accepted attempt prefix.
        record (`RunRecord`): A backward lifecycle or attributed commit record.

    Returns:
        `RuntimeBackwardState`: The validated next immutable attempt projection.

    Raises:
        `TypeError`: For malformed exact carriers or result channels.
        `ValueError`: For attempt, transfer, proposal, or disposition disagreement.
    """
    if type(backward) is not RuntimeBackwardState:
        raise TypeError("backward reduction requires RuntimeBackwardState")
    if type(record) is BackwardStartedRecord:
        _seed_backward(trace, record.feedback)
        if not backward.can_start:
            raise ValueError("backward attempt cannot start from the accepted prefix")
        if record.attempt_id != len(backward.attempts) + 1:
            raise ValueError("backward attempt_id must be the next accepted start")
        return backward._from_validated((*backward.attempts, _BackwardAttempt(record)))
    current = backward.current
    if current is None or current.finished is not None or record.attempt_id != current.started.attempt_id:
        raise ValueError("backward record requires its matching unfinished attempt")
    if type(record) is BackwardReadyRecord:
        if current.ready is not None:
            raise ValueError("backward attempt already has a Ready")
        validate_attributed_backward(trace, current.started.feedback, record.root_result, record.attributed_results)
        _proposal_batches(backward, record)
        next_attempt = replace(current, ready=record)
    elif type(record) is BackwardFinishedRecord:
        _validate_backward_finished(trace, backward, record)
        next_attempt = replace(current, finished=record)
    elif type(record) is MemoryCommitRequestedRecord:
        if current.ready is None or any(policy.status in ("unknown", "failed") for policy in backward.policy_outcomes):
            raise ValueError("backward commit request requires Ready and no pending or failed memory")
        index = len(current.requests)
        if index >= len(backward.memory_order) or record.memory != backward.memory_order[index]:
            raise ValueError("backward request must name the next recorded memory")
        batch = _proposal_batches(backward, current.ready)[index]
        expected_id = memory_commit_id(trace.run_id, record.memory, batch, current.started.attempt_id)
        if record.mutations != batch or record.commit_id != expected_id:
            raise ValueError("backward commit batch and identity must match the ordered Ready proposals")
        next_attempt = replace(current, requests=(*current.requests, record))
    elif type(record) in (MemoryCommitSucceededRecord, MemoryCommitFailedRecord):
        if len(current.requests) != len(current.observations) + 1:
            raise ValueError("backward commit result requires a matching pending request")
        request = current.requests[-1]
        if record.memory != request.memory or record.commit_id != request.commit_id:
            raise ValueError("backward commit result must match its pending memory and identity")
        next_attempt = replace(current, observations=(*current.observations, record))
    else:
        raise ValueError("backward commit requires a completed memory proposal")
    return backward._from_validated((*backward.attempts[:-1], next_attempt))


def failure_json(failure: Failure) -> dict[str, object]:
    """Project one canonical durable Failure into source-neutral JSON."""
    if type(failure) is not Failure:
        raise TypeError("failure projection requires a Failure")
    Failure(
        code=failure.code,
        message=failure.message,
        node_id=failure.node_id,
        trace=failure.trace,
        cause_type=failure.cause_type,
        traceback=failure.traceback,
    )
    return {
        "code": failure.code,
        "message": failure.message,
        "node_id": failure.node_id,
        "trace": [{"key": origin.key} for origin in failure.trace],
        "cause_type": failure.cause_type,
        "traceback": failure.traceback,
    }


def _episode_outcome(
    record: NodeSucceededRecord | NodeRoutedRecord | NodeExpandedRecord | NodeFailedRecord,
) -> tuple[MemoryEpisodeStatus, Any, dict[str, object] | None]:
    """Project the actual accepted outcome without assuming model evidence."""
    if type(record) is NodeSucceededRecord:
        if type(record.value) is not dict or "value" not in record.value:
            raise ValueError("successful record has no canonical typed value")
        output = canonical_seed(record.value["value"])[1]
        status = MemoryEpisodeStatus.SUCCEEDED
        failure = None
    elif type(record) is NodeRoutedRecord:
        if record.terminal_failure is None:
            output = {"kind": "route", "handle": record.handle}
            status = MemoryEpisodeStatus.SUCCEEDED
            failure = None
        else:
            output = None
            status = MemoryEpisodeStatus.FAILED
            failure = failure_json(record.terminal_failure)
    elif type(record) is NodeExpandedRecord:
        output = {
            "kind": "grow",
            "seed": canonical_seed(record.seed)[1],
            "prune_ids": list(record.prune_ids),
        }
        status = MemoryEpisodeStatus.SUCCEEDED
        failure = None
    elif type(record) is NodeFailedRecord:
        output = None
        status = MemoryEpisodeStatus.FAILED
        failure = failure_json(record.failure)
    else:
        raise TypeError("unsupported graph outcome for episode projection")

    return status, output, failure


class OwnerContextError(Exception):
    """A validated owner's material cannot fit the existing memory JSON boundary."""

    def __init__(self, memory: str, original: Exception) -> None:
        """Retain the receiving memory key and original context-construction error."""
        if type(memory) is not str or not memory:
            raise TypeError("owner context failure requires a memory key string")
        self.memory = memory
        self.original = original
        super().__init__(str(original))


def _owner_feedback(
    trace: RetainedTrace, occurrence: TraceOccurrence, context: BackwardContext,
    messages: list[_BackwardMessage], result: BackwardResult,
) -> list[dict[str, Any]]:
    """Project every local channel, including input messages stopped at literals."""
    entries = []
    for index, message in enumerate(messages):
        entries.append({"channel": "output", "target_name": None, "value": message.feedback,
                        "contribution_context": _materialize_context(trace, message, index)})
    has_output = occurrence.source in trace.sources
    structural = _structural_parents(context, messages, result, has_output)
    routes = dict(zip(occurrence.input_route.names, occurrence.input_route.children))
    for name in context.input_names:
        for index, feedback in enumerate(result.input_feedback.get(name, ())):
            parents = (structural[name][index],) if structural is not None else tuple(range(len(messages)))
            inherited = _inherited_context(occurrence, context, messages, parents, name, has_output)
            routed = route_feedback(routes[name], feedback, trace.lookup_source, recorded_value=context.inputs[name])
            entries.append({
                "channel": "input", "target_name": name, "value": feedback,
                "contribution_context": {
                    "message_index": index, "parents": list(parents),
                    "contexts": _route_context_data(trace, inherited),
                    "contributions": [
                        {"source": _source_data(contribution.source), "value": contribution.feedback,
                         "contribution_index": contribution_index,
                         "contexts": _route_context_data(trace, contribution.context)}
                        for contribution_index, contribution in enumerate(routed.contributions)
                    ],
                },
            })
            for stop_index, stopped in enumerate(routed.stopped):
                entries.append({
                    "channel": "stopped", "target_name": name, "value": stopped.feedback,
                    "contribution_context": {
                        "message_index": index, "stop_index": stop_index, "parents": list(parents),
                        "contexts": _route_context_data(trace, (*inherited, *stopped.context)),
                    },
                })
    entries.extend({"channel": "self", "target_name": None, "value": feedback,
                    "message_index": index, "contribution_context": {}}
                   for index, feedback in enumerate(result.self_feedback))
    variables = result.variable_feedback
    for name in context.variable_names:
        if name not in variables:
            continue
        entries.extend({"channel": "variable", "target_name": name, "value": feedback,
                        "message_index": index, "contribution_context": {}}
                       for index, feedback in enumerate(variables[name]))
    return entries


def _root_owner_feedback(result: BackwardResult) -> list[dict[str, Any]]:
    """Keep source-free/root-input fallback channels without duplicating START."""
    entries = [
        {"channel": "input", "target_name": name, "value": feedback,
         "contribution_context": {"message_index": index}}
        for name, batch in result.input_feedback.items() for index, feedback in enumerate(batch)
    ]
    entries.extend({"channel": "self", "target_name": None, "value": feedback,
                    "message_index": index, "contribution_context": {}}
                   for index, feedback in enumerate(result.self_feedback))
    entries.extend({"channel": "variable", "target_name": name, "value": feedback,
                    "message_index": index, "contribution_context": {}}
                   for name, batch in result.variable_feedback.items() for index, feedback in enumerate(batch))
    return [{"execution_seq": None, "owner_static_path": None, "occurrence_path": None,
             "execution_scope": None, "root": True, **entry} for entry in entries]


def project_owner_contexts(
    trace: RetainedTrace,
    memories: tuple[MemorySpec, ...],
    stacks: tuple[MemoryStack, ...],
    submitted_feedback: FeedbackValue,
    root_result: BackwardResult,
    attributed_results: tuple[AttributedBackwardResult, ...],
) -> tuple[LearnContext, ...]:
    """Build every enabled memory's context from a validated complete backward result.

    A memory receives an occurrence when the occurrence's recorded scope names
    its key. The source-free root channel goes to every memory owned by the
    root flow.

    Args:
        trace (`RetainedTrace`): Accepted terminal executions and historical scopes.
        memories (`tuple[MemorySpec, ...]`): Complete recorded key table,
            including keys without executed nodes.
        stacks (`tuple[MemoryStack, ...]`): Recorded per-target key stacks.
        submitted_feedback (`FeedbackValue`): Original ordinary root message.
        root_result (`BackwardResult`): Completed root projection.
        attributed_results (`tuple[AttributedBackwardResult, ...]`): Complete
            accepted occurrence inventory and validated local results.

    Returns:
        `tuple[LearnContext, ...]`: Detached episode/feedback batches in table order.
            Feedback preserves accepted execution coordinates. Memory scope selects
            receivers and episode ownership without relabeling that feedback.

    Raises:
        `TypeError` or `ValueError`: For an invalid table, stacks, trace, or
            backward authority.
        `KeyError`: For missing accepted source authority during materialization.
        `OwnerContextError`: When validated material cannot satisfy a receiving
            memory's existing JSON contract. No callback or backend is invoked.
    """
    if type(memories) is not tuple or any(type(spec) is not MemorySpec for spec in memories):
        raise TypeError("owner contexts require the complete recorded memory table")
    if type(stacks) is not tuple or any(type(stack) is not MemoryStack for stack in stacks):
        raise TypeError("owner contexts require the recorded memory stacks")
    keys = tuple(spec.key for spec in memories)
    if keys != tuple(sorted(set(keys))):
        raise ValueError("owner contexts require a memory table sorted by unique key")
    known = frozenset(keys)
    recorded = {stack.locator: stack.memories for stack in stacks}
    if len(recorded) != len(stacks):
        raise ValueError("recorded memory stacks must have unique locators")
    validate_attributed_backward(trace, submitted_feedback, root_result, attributed_results)
    occurrences = trace.occurrences
    scopes = {}
    for occurrence in occurrences:
        scope = occurrence.memory_scope
        if scope is None:
            continue
        if type(scope.memories) is not tuple:
            raise TypeError("historical memory scope memories must be a tuple")
        if recorded.get(scope.static_path) != scope.memories:
            raise ValueError("historical memory scope disagrees with its recorded memory stack")
        if not known.issuperset(scope.memories):
            raise ValueError("historical memory scope names a key absent from the recorded table")
        if scope.memories:
            scopes[occurrence.source] = scope

    pending, direct_root = _seed_backward(trace, submitted_feedback)
    feedback_by_source = {}
    for occurrence, item in reversed(tuple(zip(occurrences, attributed_results))):
        messages = sorted(pending.pop(occurrence.source, ()), key=lambda message: message.order)
        context = trace.backward_context(occurrence.source)
        if occurrence.source in scopes:
            feedback_by_source[occurrence.source] = _owner_feedback(trace, occurrence, context, messages, item.result)
        for source, transfer in _contextual_transfers(trace, occurrence, context, messages, item.result):
            pending.setdefault(source, []).append(transfer)
    fallback = _root_owner_feedback(_project_root_backward(trace, pending, {}, direct_root))

    contexts = []
    for spec in memories:
        episodes, feedback = [], []
        for occurrence in occurrences:
            scope = scopes.get(occurrence.source)
            if scope is None or spec.key not in scope.memories:
                continue
            record = occurrence.record
            capture, evidence = record.forward_capture, record.evidence
            status, output, failure = _episode_outcome(record)
            execution_scope = _execution_scope_payload(capture.scope)
            coordinates = {
                "execution_seq": occurrence.source.execution_seq,
                "owner_static_path": None if execution_scope is None else execution_scope["static_path"],
                "occurrence_path": None if execution_scope is None else execution_scope["occurrence_path"],
                "execution_scope": execution_scope,
            }
            feedback.extend({**coordinates, **entry} for entry in feedback_by_source[occurrence.source])
            try:
                episodes.append(MemoryEpisode(
                    run_id=trace.run_id,
                    occurrence_path=scope.occurrence_path,
                    owner_static_path=scope.static_path,
                    status=status, inputs=capture.bound_inputs, output=output, failure=failure,
                    interactions=() if evidence is None else evidence.interactions,
                    base_system_prompt=None if evidence is None else evidence.base_system_prompt,
                    prompt=None if evidence is None else evidence.prompt,
                    memory_sections=() if evidence is None else evidence.memory_sections,
                    execution_seq=occurrence.source.execution_seq,
                ))
            except (TypeError, ValueError) as exc:
                raise OwnerContextError(spec.key, exc) from exc
        if _ROOT_FLOW_SITE in spec.owners:
            feedback.extend(fallback)
        try:
            contexts.append(LearnContext(trace.run_id, spec.key, tuple(episodes), tuple(feedback)))
        except (TypeError, ValueError) as exc:
            raise OwnerContextError(spec.key, exc) from exc
    return tuple(contexts)


def memory_commit_id(
    run_id: str,
    memory: str,
    mutations: tuple[MemoryMutation, ...],
    attempt_id: int,
) -> str:
    """Return the sole deterministic ID for one memory's mutation batch in one attempt.

    The SHA-256 input is UTF-8 compact JSON with sorted object keys and no NaN:
    ``[run_id, attempt_id, memory, [[mutation_id, payload], ...]]``. Batch
    order is semantic. The result is ``ac-memory-v3:<hex digest>``.
    """
    if type(attempt_id) is not int or attempt_id < 1:
        raise ValueError("attempt_id must be a positive integer")
    if type(run_id) is not str or not run_id:
        raise ValueError("run_id must be a non-empty string")
    if type(memory) is not str or not memory:
        raise ValueError("memory must be a non-empty string")
    if type(mutations) is not tuple:
        raise TypeError("mutations must be a tuple")
    batch: list[list[object]] = []
    for mutation in mutations:
        if type(mutation) is not MemoryMutation:
            raise TypeError("mutations items must be MemoryMutation values")
        MemoryMutation(mutation.mutation_id, mutation.payload)
        batch.append([mutation.mutation_id, mutation.payload])
    encoded = json.dumps(
        [run_id, attempt_id, memory, batch],
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"ac-memory-v3:{sha256(encoded).hexdigest()}"


__all__ = [
    "RuntimeBackwardState",
    "OwnerContextError",
    "failure_json",
    "memory_commit_id",
    "project_owner_contexts",
    "reduce_backward_record",
    "validate_attributed_backward",
]
