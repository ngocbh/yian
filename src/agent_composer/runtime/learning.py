"""Pure backward transfers and accepted backward/memory reducer projections.

This module shares exact route-transfer validation with the backward callback
driver and owns immutable attempt state and pure episode/layout inventories.
It projects owner-local contexts for every enabled frame from complete accepted
history and backward results alongside forward episode projections.
Contracts, records, and retained traces sit below it; state and callback drivers
consume its pure projections. It invokes no callbacks or backend operations.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass, replace
from hashlib import sha256
import json
from typing import Any

from agent_composer.diagnostics import Failure
from agent_composer.expr.lineage import FeedbackValue, RouteContext, SourceRef, has_feedback, route_feedback, validate_feedback
from agent_composer.memory import (
    EvaluationEvidence,
    LearnContext,
    LearningFailure,
    LearningResult,
    MemoryEpisode,
    MemoryEpisodeStatus,
    MemoryFrameRef,
    MemoryFrameSpec,
    MemoryMutation,
    MemoryOccurrenceRef,
    MemoryOperationFailure,
    MemoryOwner,
    PolicyCommitOutcome,
)
from agent_composer.nodes.base import BackwardContext, BackwardResult, _copy_backward_value, validate_backward_result
from agent_composer.runtime.contracts import AttributedBackwardResult, BackwardAttribution, BackwardFailure, BackwardOutcome
from agent_composer.runtime.log import (
    ConversationAdmittedRecord,
    ConversationEffectRequestedRecord,
    ConversationEffectObservedRecord,
    ConversationClosedRecord,
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
from agent_composer.runtime.trace import RetainedTrace, TraceOccurrence, TraceTerminal


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


def _policy_outcomes(policy_order: tuple[str, ...], attempt: _BackwardAttempt) -> tuple[PolicyCommitOutcome, ...]:
    requests = {record.policy_name: record for record in attempt.requests}
    observations = {record.policy_name: record for record in attempt.observations}
    outcomes = []
    for name in policy_order:
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
    if not backward.policy_order:
        return None
    policies = backward.policy_outcomes
    failures = tuple(LearningFailure(item.policy_name, "commit", item.failure)
                     for item in policies if item.status == "failed")
    if learner_failure is not None:
        failures = (learner_failure, *failures)
    status = "failed" if failures else "succeeded" if succeeded else "incomplete"
    return LearningResult(status, failures, policies)


@dataclass(frozen=True, slots=True)
class RuntimeBackwardState:
    """Immutable accepted backward attempt history and policy dispositions.

    Layout inventories share the RunStarted authority roots. Completed proposals
    remain inspectable; only new records can advance an attempt. No prefix view
    resumes work or invokes a callback.

    Attributes:
        layout (`tuple[MemoryFrameSpec, ...] | None`): Recorded enabled frames.
        frame_order (`tuple[MemoryFrameRef, ...]`): Exact recorded owning refs.
        policy_order (`tuple[str, ...]`): Canonical configured policy order.
        policy_frame_indices (`tuple[tuple[int, ...], ...]`): Frame positions for
            each policy, preserving layout order during batch aggregation.
        attempts (`tuple[_BackwardAttempt, ...]`): Accepted immutable attempt
            history; a later start never inherits earlier proposals.
    """

    layout: tuple[MemoryFrameSpec, ...] | None = None
    frame_order: tuple[MemoryFrameRef, ...] = ()
    policy_order: tuple[str, ...] = ()
    policy_frame_indices: tuple[tuple[int, ...], ...] = ()
    attempts: tuple[_BackwardAttempt, ...] = ()

    def __post_init__(self) -> None:
        """Reject malformed inventories and impossible accepted-history shapes."""
        if self.layout is not None:
            _validate_layout(self.layout)
        if type(self.frame_order) is not tuple or any(type(ref) is not MemoryFrameRef for ref in self.frame_order):
            raise TypeError("backward frame order must be a tuple of exact MemoryFrameRef values")
        for ref in self.frame_order:
            MemoryFrameRef(ref.root_flow_id, ref.static_path, ref.owner)
        if type(self.policy_order) is not tuple or any(type(name) is not str for name in self.policy_order):
            raise TypeError("backward policy order must be a tuple of strings")
        if type(self.policy_frame_indices) is not tuple or any(
            type(indices) is not tuple or any(type(index) is not int for index in indices)
            for indices in self.policy_frame_indices
        ):
            raise TypeError("backward policy frame indices must be tuples of exact integers")
        expected_frames = tuple(spec.ref for spec in (self.layout or ()))
        expected_policies = tuple(sorted({spec.policy_name for spec in (self.layout or ())}))
        expected_indices = tuple(tuple(i for i, spec in enumerate(self.layout or ()) if spec.policy_name == name)
                                 for name in expected_policies)
        if (self.frame_order != expected_frames or self.policy_order != expected_policies
                or self.policy_frame_indices != expected_indices):
            raise ValueError("backward inventories must match the recorded layout")
        if type(self.attempts) is not tuple:
            raise TypeError("backward history must be an immutable tuple")
        for index, attempt in enumerate(self.attempts):
            if type(attempt) is not _BackwardAttempt:
                raise TypeError("backward history requires exact attempt states")
            _BackwardAttempt.__post_init__(attempt)
            if attempt.started.attempt_id != index + 1:
                raise ValueError("backward attempt IDs must follow accepted start order")
            if index < len(self.attempts) - 1 and (attempt.finished is None or attempt.requests
                    or attempt.finished.outcome.status == "succeeded"):
                raise ValueError("a closed or unfinished attempt cannot precede another start")
            if tuple(record.policy_name for record in attempt.requests) != self.policy_order[:len(attempt.requests)]:
                raise ValueError("backward requests must follow recorded policy order")
            if len(attempt.observations) > len(attempt.requests) or len(attempt.requests) > len(attempt.observations) + 1:
                raise ValueError("backward requests and observations must alternate")
            for result_index, observation in enumerate(attempt.observations):
                request = attempt.requests[result_index]
                if observation.policy_name != request.policy_name or observation.commit_id != request.commit_id:
                    raise ValueError("backward observation must match its accepted request")
                if type(observation) is MemoryCommitFailedRecord and result_index < len(attempt.requests) - 1:
                    raise ValueError("a failed commit cannot be followed by another request")
            if attempt.requests and attempt.ready is None:
                raise ValueError("backward requests require accepted Ready proposals")

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
    def policy_outcomes(self) -> tuple[PolicyCommitOutcome, ...]:
        """Project each configured policy from accepted requests and observations."""
        if self.current is None:
            return tuple(PolicyCommitOutcome(name, "unrequested") for name in self.policy_order)
        return _policy_outcomes(self.policy_order, self.current)

    @property
    def can_start(self) -> bool:
        """Report history eligibility; terminal/learn gates remain state-owned."""
        return (not self.ever_requested and (self.current is None or self.finished is not None)
                and not any(attempt.finished is not None and attempt.finished.outcome.status == "succeeded"
                            for attempt in self.attempts))

    @property
    def outcome(self) -> BackwardOutcome | None:
        """Inspect the accepted Finished or derive interruption without appending."""
        if self.current is None:
            return None
        if self.finished is not None:
            return self.finished.outcome
        return BackwardOutcome("failed", self.ready.root_result if self.ready else None,
                               _interrupted_failure(), _prefix_learning(self))


def _proposal_batches(backward: RuntimeBackwardState, ready: BackwardReadyRecord) -> tuple[tuple[MemoryMutation, ...], ...]:
    """Aggregate every recorded frame in layout order, including empty proposals."""
    if tuple(proposal.frame_ref for proposal in ready.frame_proposals) != backward.frame_order:
        raise ValueError("frame proposals must cover the complete recorded layout in order")
    batches = tuple(tuple(mutation for index in indices for mutation in ready.frame_proposals[index].mutations)
                    for indices in backward.policy_frame_indices)
    for batch in batches:
        identities = tuple(mutation.mutation_id for mutation in batch)
        if len(set(identities)) != len(identities):
            raise ValueError("backward mutation IDs must be unique within each policy")
    return batches


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
            raise ValueError("successful Finished requires Ready and every policy confirmation")
        if outcome.learning != _prefix_learning(backward, succeeded=True):
            raise ValueError("successful Finished learning must match accepted policy confirmations")
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
                                   policy_name=policy.policy_name, cause=policy.failure)
        if failure != expected or outcome.learning != _prefix_learning(backward):
            raise ValueError("commit Finished must match the exact accepted failure and policy states")
        return
    if failure.stage == "learn":
        if type(failure.cause) is not MemoryOperationFailure or failure.frame_ref not in backward.frame_order:
            raise ValueError("learner failure requires one exact enabled frame and memory-operation cause")
        spec = backward.layout[backward.frame_order.index(failure.frame_ref)]
        expected = BackwardFailure("learn", failure.cause.code, failure.cause.message,
                                   policy_name=spec.policy_name, frame_ref=spec.ref, cause=failure.cause)
        if failure != expected or outcome.feedback is None:
            raise ValueError("learner Finished must retain the exact attributed failure and completed feedback")
        validate_attributed_backward(trace, current.started.feedback, outcome.feedback, record.attributed_results)
        known = LearningFailure(spec.policy_name, "learn", failure.cause, spec.ref)
        if outcome.learning != _prefix_learning(backward, learner_failure=known):
            raise ValueError("learner Finished learning must contain its exact known failure")
        return
    if failure.stage != "backward" or outcome.feedback is not None or record.attributed_results:
        raise ValueError("incomplete backward computation cannot carry completed or partial results")
    _seed_backward(trace, current.started.feedback)
    if not has_feedback(current.started.feedback) or failure.code not in ("backward.execute", "backward.result"):
        raise ValueError("backward computation failure requires effective feedback and a driver failure code")
    if (failure.run_id != trace.run_id or failure.node_started_seq is None
            or failure.policy_name is not None or failure.frame_ref is not None):
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
        raise ValueError("failed computation learning must retain incomplete unrequested policies")


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
        return replace(backward, attempts=(*backward.attempts, _BackwardAttempt(record)))
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
            raise ValueError("backward commit request requires Ready and no pending or failed policy")
        index = len(current.requests)
        if index >= len(backward.policy_order) or record.policy_name != backward.policy_order[index]:
            raise ValueError("backward request must name the next enabled policy")
        batch = _proposal_batches(backward, current.ready)[index]
        if record.mutations != batch or record.commit_id != memory_commit_id(trace.run_id, record.policy_name, batch):
            raise ValueError("backward commit batch and identity must match the ordered Ready proposals")
        next_attempt = replace(current, requests=(*current.requests, record))
    elif type(record) in (MemoryCommitSucceededRecord, MemoryCommitFailedRecord):
        if len(current.requests) != len(current.observations) + 1:
            raise ValueError("backward commit result requires a matching pending request")
        request = current.requests[-1]
        if record.policy_name != request.policy_name or record.commit_id != request.commit_id:
            raise ValueError("backward commit result must match its pending policy and identity")
        next_attempt = replace(current, observations=(*current.observations, record))
    else:
        raise ValueError("backward commit requires a completed policy proposal")
    return replace(backward, attempts=(*backward.attempts[:-1], next_attempt))


@dataclass(frozen=True, slots=True)
class _EpisodeLog:
    """Persistent append log with canonical power-of-two chronological chunks.

    Level zero holds one episode, level one holds two, and so on.  Occupied
    levels are read from highest to lowest to recover record order.  Appending
    merges only the binary-counter carry path, so retaining a long MAP/LOOP
    history does not copy or revalidate every earlier episode per outcome.
    """

    levels: tuple[tuple[MemoryEpisode, ...] | None, ...] = ()

    def __post_init__(self) -> None:
        """Validate a directly constructed log once at its public boundary."""
        if type(self.levels) is not tuple:
            raise TypeError("episode log levels must be a tuple")
        if self.levels and self.levels[-1] is None:
            raise ValueError("episode log cannot have trailing empty levels")
        for index, chunk in enumerate(self.levels):
            if chunk is None:
                continue
            if type(chunk) is not tuple or len(chunk) != 1 << index:
                raise ValueError("episode log chunks must match their binary level")
            for episode in chunk:
                _revalidate_episode(episode)

    @classmethod
    def _from_validated(
        cls,
        levels: tuple[tuple[MemoryEpisode, ...] | None, ...],
    ) -> _EpisodeLog:
        """Build from an existing valid log and one already-validated carry."""
        value = object.__new__(cls)
        object.__setattr__(value, "levels", levels)
        return value

    def append(self, episode: MemoryEpisode) -> _EpisodeLog:
        """Return a log with one validated episode appended in record order."""
        _revalidate_episode(episode)
        levels = list(self.levels)
        carry = (episode,)
        index = 0
        while index < len(levels) and levels[index] is not None:
            prior = levels[index]
            if prior is None:  # narrowed by the loop condition
                raise AssertionError("occupied episode level disappeared")
            carry = prior + carry
            levels[index] = None
            index += 1
        if index == len(levels):
            levels.append(carry)
        else:
            levels[index] = carry
        return self._from_validated(tuple(levels))

    def to_tuple(self) -> tuple[MemoryEpisode, ...]:
        """Materialize chronological episodes only at the learning boundary."""
        return tuple(
            episode
            for chunk in reversed(self.levels)
            if chunk is not None
            for episode in chunk
        )


def frame_key(ref: MemoryFrameRef) -> tuple[str, tuple[str, ...], str]:
    """Return the canonical structural ordering key for one frame."""
    if type(ref) is not MemoryFrameRef:
        raise TypeError("frame key requires a MemoryFrameRef")
    return (ref.root_flow_id, ref.static_path, ref.owner.value)


def occurrence_key(
    occurrence: MemoryOccurrenceRef,
) -> tuple[
    tuple[str, tuple[str, ...], str],
    tuple[tuple[str, tuple[str, ...], int], ...],
]:
    """Return the canonical structural ordering key for one occurrence."""
    if type(occurrence) is not MemoryOccurrenceRef:
        raise TypeError("occurrence key requires a MemoryOccurrenceRef")
    return (
        frame_key(occurrence.frame_ref),
        tuple(
            (segment.kind.value, segment.static_path, segment.index)
            for segment in occurrence.occurrence_path
        ),
    )


@dataclass(frozen=True, slots=True)
class RuntimeLearningState:
    """Pure immutable episode and layout inventory without attempt progress."""

    frame_order: tuple[MemoryFrameRef, ...] = ()
    policy_order: tuple[str, ...] = ()
    policy_frame_indices: tuple[tuple[int, ...], ...] = ()
    episodes: tuple[_EpisodeLog, ...] = ()
    episode_run_id: str | None = None

    def __post_init__(self) -> None:
        """Require canonical inventories and revalidate all nested values."""
        if type(self.frame_order) is not tuple:
            raise TypeError("learning frame_order must be a tuple")
        for ref in self.frame_order:
            if type(ref) is not MemoryFrameRef:
                raise TypeError("learning frame_order items must be MemoryFrameRef")
            MemoryFrameRef(ref.root_flow_id, ref.static_path, ref.owner)
        keys = tuple(frame_key(ref) for ref in self.frame_order)
        if keys != tuple(sorted(set(keys))):
            raise ValueError("learning frame_order must be sorted and unique")
        if type(self.policy_order) is not tuple or any(
            type(name) is not str or not name for name in self.policy_order
        ):
            raise TypeError(
                "learning policy_order must be a tuple of non-empty strings"
            )
        if self.policy_order != tuple(sorted(set(self.policy_order))):
            raise ValueError("learning policy_order must be sorted and unique")
        if type(self.policy_frame_indices) is not tuple:
            raise TypeError("policy frame indices must be a tuple")
        if type(self.episodes) is not tuple:
            raise TypeError("learning episodes must be a tuple")
        if len(self.policy_frame_indices) != len(self.policy_order):
            raise ValueError("policy frame indices must align with policy_order")
        if len(self.episodes) != len(self.frame_order):
            raise ValueError("learning episodes must align with frame_order")
        flattened_indices: list[int] = []
        for indices in self.policy_frame_indices:
            if type(indices) is not tuple or any(
                type(index) is not int for index in indices
            ):
                raise TypeError("policy frame indices must contain integer tuples")
            if indices != tuple(sorted(set(indices))):
                raise ValueError("policy frame indices must be sorted and unique")
            flattened_indices.extend(indices)
        if sorted(flattened_indices) != list(range(len(self.frame_order))):
            raise ValueError("policy frame indices must partition frame_order")
        if self.episode_run_id is not None and (
            type(self.episode_run_id) is not str or not self.episode_run_id
        ):
            raise ValueError("episode_run_id must be a non-empty string or None")
        episode_run_ids: set[str] = set()
        for index, episode_log in enumerate(self.episodes):
            if type(episode_log) is not _EpisodeLog:
                raise TypeError("frame episodes must be persistent episode logs")
            _EpisodeLog(episode_log.levels)
            ref = self.frame_order[index]
            for episode in episode_log.to_tuple():
                episode_run_ids.add(episode.run_id)
                if episode.occurrence.frame_ref != ref:
                    raise ValueError("episode must align with its frame")
        if len(episode_run_ids) > 1:
            raise ValueError("all learning episodes must share one run_id")
        expected_episode_run_id = next(iter(episode_run_ids), None)
        if self.episode_run_id != expected_episode_run_id:
            raise ValueError("episode_run_id must match retained episodes")

    @classmethod
    def for_layout(
        cls,
        layout: tuple[MemoryFrameSpec, ...] | None,
    ) -> RuntimeLearningState:
        """Return pristine learning state for one canonical recorded layout."""
        if layout is None:
            return cls()
        _validate_layout(layout)
        frame_order = tuple(spec.ref for spec in layout)
        policy_order = tuple(sorted({spec.policy_name for spec in layout}))
        policy_buckets = {name: [] for name in policy_order}
        for index, spec in enumerate(layout):
            policy_buckets[spec.policy_name].append(index)
        policy_frame_indices = tuple(
            tuple(policy_buckets[name]) for name in policy_order
        )
        return cls(
            frame_order=frame_order,
            policy_order=policy_order,
            policy_frame_indices=policy_frame_indices,
            episodes=tuple(_EpisodeLog() for _ in frame_order),
        )

    def validate_layout(self, layout: tuple[MemoryFrameSpec, ...] | None) -> None:
        """Require inventories to remain aligned with authoritative layout."""
        if layout is None:
            expected_frames: tuple[MemoryFrameRef, ...] = ()
            expected_policies: tuple[str, ...] = ()
        else:
            if type(layout) is not tuple:
                raise TypeError("memory_layout must be a tuple when enabled")
            if any(type(spec) is not MemoryFrameSpec for spec in layout):
                raise TypeError(
                    "memory_layout items must be MemoryFrameSpec values"
                )
            expected_frames = tuple(spec.ref for spec in layout)
            expected_policies = tuple(
                sorted({spec.policy_name for spec in layout})
            )
        if self.frame_order != expected_frames:
            raise ValueError("learning frame inventory disagrees with memory_layout")
        if self.policy_order != expected_policies:
            raise ValueError("learning policy inventory disagrees with memory_layout")
        expected_buckets = {name: [] for name in expected_policies}
        if layout is not None:
            for index, spec in enumerate(layout):
                expected_buckets[spec.policy_name].append(index)
        expected_indices = tuple(
            tuple(expected_buckets[name]) for name in expected_policies
        )
        if self.policy_frame_indices != expected_indices:
            raise ValueError("learning policy frames disagree with memory_layout")

    def episodes_for(self, ref: MemoryFrameRef) -> tuple[MemoryEpisode, ...]:
        """Return accepted episodes for one exact recorded frame."""
        return self.episodes[self._frame_index(ref)].to_tuple()

    def occurrence_inventory(
        self,
        ref: MemoryFrameRef,
    ) -> tuple[MemoryOccurrenceRef, ...]:
        """Return the sorted unique structural occurrences accepted for a frame."""
        return self._occurrence_inventory_at(self._frame_index(ref))

    def applicable_frames(
        self,
        root_flow_id: str,
        static_path: tuple[str, ...],
    ) -> tuple[MemoryFrameRef, ...]:
        """Return exact lexical FLOW ancestors followed by a local AGENT frame."""
        if type(root_flow_id) is not str or not root_flow_id:
            raise ValueError(
                "applicable frame lookup requires a non-empty root_flow_id"
            )
        if (
            type(static_path) is not tuple
            or not static_path
            or any(type(part) is not str or not part for part in static_path)
        ):
            raise ValueError("applicable frame lookup requires a non-empty static_path")
        if static_path[0] != root_flow_id:
            raise ValueError("applicable frame path must begin with root_flow_id")

        applicable: list[MemoryFrameRef] = []
        for length in range(1, len(static_path)):
            ref = self._frame_for_key(
                (root_flow_id, static_path[:length], MemoryOwner.FLOW.value)
            )
            if ref is not None:
                applicable.append(ref)
        local_agent = self._frame_for_key(
            (root_flow_id, static_path, MemoryOwner.AGENT.value)
        )
        if local_agent is not None:
            applicable.append(local_agent)
        return tuple(applicable)

    def learn_context(self, run_id: str, ref: MemoryFrameRef) -> LearnContext:
        """Return a fresh closed context with ordered episodes and no feedback."""
        if type(run_id) is not str or not run_id:
            raise ValueError("learning context requires a non-empty run_id")
        if self.episode_run_id is not None and self.episode_run_id != run_id:
            raise ValueError("learning context run_id disagrees with its episodes")
        episodes = self.episodes_for(ref)
        return LearnContext(
            run_id=run_id,
            frame_ref=ref,
            episodes=episodes,
            feedback=(),
        )

    def append_episodes(
        self,
        additions: tuple[tuple[MemoryFrameRef, MemoryEpisode], ...],
    ) -> RuntimeLearningState:
        """Append one accepted outcome's projections in durable record order."""
        if type(additions) is not tuple:
            raise TypeError("episode additions must be a tuple")
        if not additions:
            return self
        updated = list(self.episodes)
        addition_run_ids: set[str] = set()
        for ref, episode in additions:
            index = self._frame_index(ref)
            appended = updated[index].append(episode)
            if episode.occurrence.frame_ref != ref:
                raise ValueError("episode addition frame does not match occurrence")
            addition_run_ids.add(episode.run_id)
            updated[index] = appended
        if len(addition_run_ids) != 1:
            raise ValueError("one accepted outcome must carry one episode run_id")
        addition_run_id = next(iter(addition_run_ids))
        if (
            self.episode_run_id is not None
            and addition_run_id != self.episode_run_id
        ):
            raise ValueError("episode addition run_id disagrees with retained history")
        return self._from_validated(
            episodes=tuple(updated),
            episode_run_id=addition_run_id,
        )

    def _occurrence_inventory_at(
        self,
        index: int,
    ) -> tuple[MemoryOccurrenceRef, ...]:
        """Return one frame's structural inventory from its persistent log."""
        occurrences = {
            episode.occurrence for episode in self.episodes[index].to_tuple()
        }
        return tuple(sorted(occurrences, key=occurrence_key))

    def _from_validated(
        self,
        *,
        episodes: tuple[_EpisodeLog, ...] | None = None,
        episode_run_id: str | None = None,
    ) -> RuntimeLearningState:
        """Replace validated aligned roots without rescanning frozen history."""
        value = object.__new__(type(self))
        object.__setattr__(value, "frame_order", self.frame_order)
        object.__setattr__(value, "policy_order", self.policy_order)
        object.__setattr__(
            value,
            "policy_frame_indices",
            self.policy_frame_indices,
        )
        object.__setattr__(
            value,
            "episodes",
            self.episodes if episodes is None else episodes,
        )
        object.__setattr__(
            value,
            "episode_run_id",
            self.episode_run_id if episode_run_id is None else episode_run_id,
        )
        return value

    def _frame_index(self, ref: MemoryFrameRef) -> int:
        """Resolve one exact frame or reject an unrecorded identity."""
        if type(ref) is not MemoryFrameRef:
            raise TypeError("frame lookup requires a MemoryFrameRef")
        key = frame_key(ref)
        index = bisect_left(self.frame_order, key, key=frame_key)
        if index >= len(self.frame_order) or self.frame_order[index] != ref:
            raise ValueError("memory frame is absent from recorded layout")
        return index

    def _frame_for_key(
        self,
        key: tuple[str, tuple[str, ...], str],
    ) -> MemoryFrameRef | None:
        """Return the exact sorted frame at ``key`` without scanning inventory."""
        index = bisect_left(self.frame_order, key, key=frame_key)
        if index >= len(self.frame_order):
            return None
        ref = self.frame_order[index]
        return ref if frame_key(ref) == key else None


def _validate_layout(layout: tuple[MemoryFrameSpec, ...]) -> None:
    """Revalidate exact layout carriers and canonical structural order."""
    if type(layout) is not tuple:
        raise TypeError("memory_layout must be a tuple when enabled")
    for spec in layout:
        if type(spec) is not MemoryFrameSpec:
            raise TypeError("memory_layout items must be MemoryFrameSpec values")
        MemoryFrameSpec(spec.ref, spec.policy_name, spec.config)
    keys = tuple(frame_key(spec.ref) for spec in layout)
    if keys != tuple(sorted(set(keys))):
        raise ValueError("memory_layout frame refs must be sorted and unique")
    if len({spec.ref.root_flow_id for spec in layout}) > 1:
        raise ValueError("memory_layout frame refs must share one root_flow_id")


def _revalidate_episode(episode: MemoryEpisode) -> None:
    """Replay every exact public invariant of one immutable episode."""
    if type(episode) is not MemoryEpisode:
        raise TypeError("episodes must contain MemoryEpisode values")
    MemoryEpisode(
        run_id=episode.run_id,
        occurrence=episode.occurrence,
        owner_static_path=episode.owner_static_path,
        status=episode.status,
        inputs=episode.inputs,
        output=episode.output,
        failure=episode.failure,
        interactions=episode.interactions,
        base_system_prompt=episode.base_system_prompt,
        prompt=episode.prompt,
        memory_sections=episode.memory_sections,
        execution_seq=episode.execution_seq,
    )


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


def project_outcome_episodes(
    run_id: str,
    record: (
        NodeSucceededRecord
        | NodeRoutedRecord
        | NodeExpandedRecord
        | NodeFailedRecord
        | ConversationAdmittedRecord
        | ConversationEffectRequestedRecord
        | ConversationEffectObservedRecord
        | ConversationClosedRecord
    ),
    *,
    execution_seq: int | None = None,
) -> tuple[tuple[MemoryFrameRef, MemoryEpisode], ...]:
    """Project one canonical accepted graph outcome into every scoped frame."""
    if type(run_id) is not str or not run_id:
        raise ValueError("episode projection requires a non-empty run_id")
    if type(record) in (ConversationAdmittedRecord, ConversationEffectRequestedRecord,
                        ConversationEffectObservedRecord, ConversationClosedRecord):
        return ()
    evidence = record.evidence
    if evidence is None:
        return ()
    if type(evidence) is not EvaluationEvidence:
        raise TypeError("episode projection evidence must be EvaluationEvidence")
    if evidence.scope is None:
        return ()

    status, output, failure = _episode_outcome(record)

    additions: list[tuple[MemoryFrameRef, MemoryEpisode]] = []
    for frame_ref in evidence.scope.frame_stack:
        occurrence = MemoryOccurrenceRef(
            frame_ref,
            evidence.scope.occurrence_path,
        )
        additions.append(
            (
                frame_ref,
                MemoryEpisode(
                    run_id=run_id,
                    occurrence=occurrence,
                    owner_static_path=evidence.scope.static_path,
                    status=status,
                    inputs=evidence.bound_inputs,
                    output=output,
                    failure=failure,
                    interactions=evidence.interactions,
                    base_system_prompt=evidence.base_system_prompt,
                    prompt=evidence.prompt,
                    memory_sections=evidence.memory_sections,
                    execution_seq=execution_seq,
                ),
            )
        )
    return tuple(additions)


class OwnerContextError(Exception):
    """A validated owner's material cannot fit the existing memory JSON boundary."""

    def __init__(self, frame_ref: MemoryFrameRef, original: Exception) -> None:
        """Retain the exact receiving frame and original context-construction error."""
        if type(frame_ref) is not MemoryFrameRef:
            raise TypeError("owner context failure requires an exact MemoryFrameRef")
        self.frame_ref = MemoryFrameRef(frame_ref.root_flow_id, frame_ref.static_path, frame_ref.owner)
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
    trace: RetainedTrace, layout: tuple[MemoryFrameSpec, ...], submitted_feedback: FeedbackValue,
    root_result: BackwardResult, attributed_results: tuple[AttributedBackwardResult, ...],
) -> tuple[LearnContext, ...]:
    """Build every enabled owner's context from a validated complete backward result.

    Args:
        trace (`RetainedTrace`): Accepted terminal executions and historical scopes.
        layout (`tuple[MemoryFrameSpec, ...]`): Complete recorded enabled layout,
            including frames without executed nodes.
        submitted_feedback (`FeedbackValue`): Original ordinary root message.
        root_result (`BackwardResult`): Completed root projection.
        attributed_results (`tuple[AttributedBackwardResult, ...]`): Complete
            accepted occurrence inventory and validated local results.

    Returns:
        `tuple[LearnContext, ...]`: Detached episode/feedback batches in layout order.
            Feedback preserves accepted execution coordinates. Memory scope selects
            receivers and episode ownership without relabeling that feedback.

    Raises:
        `TypeError` or `ValueError`: For invalid layout, trace, or backward authority.
        `KeyError`: For missing accepted source authority during materialization.
        `OwnerContextError`: When validated material cannot satisfy a receiving
            frame's existing memory JSON contract. No callback or backend is invoked.
    """
    if type(layout) is not tuple:
        raise TypeError("owner contexts require the complete recorded layout tuple")
    inventory = RuntimeLearningState.for_layout(layout)
    validate_attributed_backward(trace, submitted_feedback, root_result, attributed_results)
    occurrences = trace.occurrences
    scopes = {}
    for occurrence in occurrences:
        scope = occurrence.memory_scope
        if scope is not None:
            if scope.frame_stack != inventory.applicable_frames(scope.root_flow_id, scope.static_path):
                raise ValueError("historical memory scope disagrees with the complete recorded layout")
            if scope.frame_stack:
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
    for spec in layout:
        episodes, feedback = [], []
        for occurrence in occurrences:
            scope = scopes.get(occurrence.source)
            if scope is None or spec.ref not in scope.frame_stack:
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
                    occurrence=MemoryOccurrenceRef(spec.ref, scope.occurrence_path),
                    owner_static_path=scope.static_path,
                    status=status, inputs=capture.bound_inputs, output=output, failure=failure,
                    interactions=() if evidence is None else evidence.interactions,
                    base_system_prompt=None if evidence is None else evidence.base_system_prompt,
                    prompt=None if evidence is None else evidence.prompt,
                    memory_sections=() if evidence is None else evidence.memory_sections,
                    execution_seq=occurrence.source.execution_seq,
                ))
            except (TypeError, ValueError) as exc:
                raise OwnerContextError(spec.ref, exc) from exc
        if spec.ref.owner is MemoryOwner.FLOW and spec.ref.static_path == (spec.ref.root_flow_id,):
            feedback.extend(fallback)
        try:
            contexts.append(LearnContext(trace.run_id, spec.ref, tuple(episodes), tuple(feedback)))
        except (TypeError, ValueError) as exc:
            raise OwnerContextError(spec.ref, exc) from exc
    return tuple(contexts)


def memory_commit_id(
    run_id: str,
    policy_name: str,
    mutations: tuple[MemoryMutation, ...],
) -> str:
    """Return the sole deterministic ID for a policy's canonical mutation batch.

    The SHA-256 input is UTF-8 compact JSON with sorted object keys and no NaN:
    ``[run_id, policy_name, canonical_batch]``.  Batch order is semantic and each
    item contains ``mutation_id``, exact frame-ref fields, and the JSON payload.
    """
    if type(run_id) is not str or not run_id:
        raise ValueError("run_id must be a non-empty string")
    if type(policy_name) is not str or not policy_name:
        raise ValueError("policy_name must be a non-empty string")
    if type(mutations) is not tuple:
        raise TypeError("mutations must be a tuple")
    batch: list[dict[str, object]] = []
    for mutation in mutations:
        if type(mutation) is not MemoryMutation:
            raise TypeError("mutations items must be MemoryMutation values")
        MemoryMutation(
            mutation.mutation_id,
            mutation.frame_ref,
            mutation.payload,
        )
        batch.append(
            {
                "mutation_id": mutation.mutation_id,
                "frame_ref": {
                    "root_flow_id": mutation.frame_ref.root_flow_id,
                    "static_path": list(mutation.frame_ref.static_path),
                    "owner": mutation.frame_ref.owner.value,
                },
                "payload": mutation.payload,
            }
        )
    encoded = json.dumps(
        [run_id, policy_name, batch],
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"ac-memory-v1:{sha256(encoded).hexdigest()}"


__all__ = [
    "RuntimeBackwardState",
    "RuntimeLearningState",
    "OwnerContextError",
    "failure_json",
    "frame_key",
    "memory_commit_id",
    "occurrence_key",
    "project_outcome_episodes",
    "project_owner_contexts",
    "reduce_backward_record",
    "validate_attributed_backward",
]
