"""Detached backward proposal and record contracts without reducer semantics."""

from dataclasses import FrozenInstanceError
import json

import pytest

from agent_composer.memory import MemoryFrameRef, MemoryMutation, MemoryOwner


def _frame(name="agent"):
    return MemoryFrameRef("root", ("root", name), MemoryOwner.AGENT)


def test_frame_proposal_owns_detached_mutations_in_supplied_order():
    from agent_composer.runtime.contracts import FrameProposal

    ref = _frame()
    payload = {"values": [1]}
    mutations = (MemoryMutation("second", ref, payload), MemoryMutation("first", ref, {}))
    proposal = FrameProposal(ref, mutations)
    payload["values"].append(2)
    proposal.mutations[0].payload["values"].clear()
    assert [mutation.mutation_id for mutation in proposal.mutations] == ["second", "first"]
    assert proposal.mutations[0].payload == {"values": [1]}
    assert proposal.mutations[0] is not mutations[0]
    assert proposal.frame_ref == ref
    with pytest.raises(FrozenInstanceError):
        proposal.mutations = ()
    with pytest.raises(TypeError):
        FrameProposal(ref, list(mutations))
    with pytest.raises(ValueError, match="frame_ref"):
        FrameProposal(ref, (MemoryMutation("wrong", _frame("other"), {}),))
    with pytest.raises(ValueError, match="unique"):
        FrameProposal(ref, (mutations[0], mutations[0]))
    assert FrameProposal(ref, ()).mutations == ()


def _result():
    from agent_composer.nodes.base import BackwardResult

    return BackwardResult({"topic": ([None, {"uri": "Review."}],)},
                          self_feedback=("Behavior.",), variable_feedback={"tone": ("Concise.",)})


def _attributed():
    from agent_composer.expr.lineage import SourceRef
    from agent_composer.runtime.contracts import AttributedBackwardResult, BackwardAttribution

    return (AttributedBackwardResult(SourceRef("run", 1), ("Review.",),
                                     (BackwardAttribution(None, None, 0, None, ()),), _result(), True),)


def test_backward_record_constructors_detach_values_without_reducer_inference():
    from agent_composer.runtime.contracts import BackwardOutcome, FrameProposal
    from agent_composer.runtime.log import BackwardStartedRecord, BackwardReadyRecord, BackwardFinishedRecord

    feedback = {"parts": [None, "Review."]}
    started = BackwardStartedRecord(1, feedback)
    feedback["parts"].clear()
    started.feedback["parts"].clear()
    assert started.feedback == {"parts": [None, "Review."]}
    proposal = FrameProposal(_frame(), (MemoryMutation("m1", _frame(), {"notes": ["one"]}),))
    ready = BackwardReadyRecord(1, _result(), _attributed(), (proposal,))
    ready.root_result.input_feedback["topic"][0].clear()
    ready.attributed_results[0].feedback[0].upper()
    assert ready.root_result == _result()
    assert ready.frame_proposals[0] is not proposal
    outcome = BackwardOutcome("succeeded", _result())
    finished = BackwardFinishedRecord(1, outcome)
    assert finished.attributed_results == ()
    assert finished.outcome == outcome and finished.outcome is not outcome
    for record in (started, ready, finished):
        with pytest.raises(FrozenInstanceError):
            record.attempt_id = 2
    for invalid in (0, -1, True, "1", 1.0):
        with pytest.raises((TypeError, ValueError)):
            BackwardStartedRecord(invalid, None)
    with pytest.raises(TypeError):
        BackwardReadyRecord(1, _result(), list(_attributed()), ())
    with pytest.raises(TypeError):
        BackwardReadyRecord(1, _result(), (), [proposal])


def test_commit_attempt_ids_are_optional_keyword_only_positive_integers():
    from agent_composer.memory import MemoryOperationFailure
    from agent_composer.runtime.log import MemoryCommitRequestedRecord, MemoryCommitSucceededRecord, MemoryCommitFailedRecord

    failure = MemoryOperationFailure("backend", "failed", "Failed.", True)
    for carrier, args in (
        (MemoryCommitRequestedRecord, ("history", "commit-1", ())),
        (MemoryCommitSucceededRecord, ("history", "commit-1")),
        (MemoryCommitFailedRecord, ("history", "commit-1", failure)),
    ):
        assert carrier(*args).attempt_id is None
        assert carrier(*args, attempt_id=2).attempt_id == 2
        with pytest.raises(TypeError):
            carrier(*args, 2)
        for invalid in (0, -1, True, "2", 2.0):
            with pytest.raises((TypeError, ValueError)):
                carrier(*args, attempt_id=invalid)


def _backward_records():
    from agent_composer.memory import LearningResult, PolicyCommitOutcome
    from agent_composer.runtime.contracts import BackwardOutcome, FrameProposal
    from agent_composer.runtime.log import BackwardStartedRecord, BackwardReadyRecord, BackwardFinishedRecord

    proposal = FrameProposal(_frame(), (MemoryMutation("m1", _frame(), {"notes": ["one"]}),))
    outcome = BackwardOutcome("succeeded", _result(), learning=LearningResult(
        "succeeded", policies=(PolicyCommitOutcome("history", "succeeded", "commit-1"),),
    ))
    return [BackwardStartedRecord(1, [None, {"uri": "Review."}]),
            BackwardReadyRecord(1, _result(), _attributed(), (proposal,)),
            BackwardFinishedRecord(1, outcome, _attributed())]


def _round_trip(record):
    from agent_composer.runtime.log import SCHEMA_VERSION, RecordEnvelope, decode_record, encode_record

    envelope = RecordEnvelope(SCHEMA_VERSION, "run", 0, record)
    encoded = encode_record(envelope)
    decoded = decode_record(encoded)
    assert decoded == envelope
    assert encode_record(decoded) == encoded
    return decoded.record


@pytest.mark.parametrize("index", range(3))
def test_backward_record_wire_round_trip_preserves_all_channels(index):
    record = _backward_records()[index]
    assert _round_trip(record) == record


def test_commit_wire_preserves_nullable_and_supplied_attempt_ids():
    from agent_composer.memory import MemoryOperationFailure
    from agent_composer.runtime.log import MemoryCommitRequestedRecord, MemoryCommitSucceededRecord, MemoryCommitFailedRecord

    failure = MemoryOperationFailure("backend", "failed", "Failed.", False)
    for attempt_id in (None, 3):
        for record in (
            MemoryCommitRequestedRecord("history", "commit-1", (), attempt_id=attempt_id),
            MemoryCommitSucceededRecord("history", "commit-1", attempt_id=attempt_id),
            MemoryCommitFailedRecord("history", "commit-1", failure, attempt_id=attempt_id),
        ):
            assert _round_trip(record).attempt_id == attempt_id


def _failed_record(cause_kind="memory", *, completed=True, learning_status="failed"):
    from agent_composer.diagnostics import Failure, OriginRef
    from agent_composer.memory import LearningFailure, LearningResult, MemoryOperationFailure, PolicyCommitOutcome
    from agent_composer.nodes.contracts import ExecutionPosition, ExecutionScope
    from agent_composer.runtime.contracts import BackwardFailure, BackwardOutcome
    from agent_composer.runtime.log import BackwardFinishedRecord

    cause = MemoryOperationFailure("backend-stage", "backend.failed", "Retry later.", True)
    causes = {"none": None, "memory": cause,
              "forward": Failure("node.execute", "Failed.", "agent", (OriginRef("agent.prompt"),),
                                 "ValueError", "forward traceback")}
    failure = BackwardFailure(
        "learn", "memory.learn", "Could not propose.",
        ExecutionScope(("each", "agent"), ("each",),
                       (ExecutionPosition("map_item", ("each",), 2),)),
        "run", "each/2/agent", 7, "history", _frame(), causes[cause_kind],
        "MemoryOperationError", "backward traceback",
    )
    learning = LearningResult(learning_status, (
        LearningFailure("history", "learn", cause, _frame()),
        LearningFailure("store", "commit", cause),
    ), (
        PolicyCommitOutcome("later", "unrequested"),
        PolicyCommitOutcome("pending", "unknown", "commit-unknown"),
        PolicyCommitOutcome("confirmed", "succeeded", "commit-ok"),
        PolicyCommitOutcome("store", "failed", "commit-failed", cause),
    ))
    return BackwardFinishedRecord(
        2, BackwardOutcome("failed", _result() if completed else None, failure, learning),
        _attributed() if completed else (),
    )


@pytest.mark.parametrize("cause_kind", ["none", "forward", "memory"])
@pytest.mark.parametrize("completed", [False, True])
@pytest.mark.parametrize("learning_status", ["failed", "incomplete"])
def test_failed_backward_wire_keeps_causes_completed_feedback_and_policy_dispositions(
    cause_kind, completed, learning_status,
):
    record = _failed_record(cause_kind, completed=completed, learning_status=learning_status)
    decoded = _round_trip(record)
    assert (decoded.outcome.feedback is not None) is completed
    assert bool(decoded.attributed_results) is completed
    assert decoded.outcome.failure.scope.occurrence_path[0].index == 2
    assert tuple(p.status for p in decoded.outcome.learning.policies) == (
        "unrequested", "unknown", "succeeded", "failed",
    )
    assert decoded.outcome.learning.failures[0].failure.retryable is True


def test_wire_preserves_transfer_coordinates_unreached_inventory_and_empty_defaults():
    from agent_composer.expr.lineage import SourceRef
    from agent_composer.nodes.base import BackwardResult
    from agent_composer.runtime.contracts import AttributedBackwardResult, BackwardAttribution, BackwardOutcome
    from agent_composer.runtime.log import BackwardReadyRecord, BackwardFinishedRecord, BackwardStartedRecord

    transfer = BackwardAttribution(SourceRef("run", 9), "topic", 3, 2, (0, 2))
    attributed = (
        AttributedBackwardResult(SourceRef("run", 7), ("",), (transfer,), _result(), True),
        AttributedBackwardResult(SourceRef("run", 5), (), (), BackwardResult(), False),
    )
    ready = _round_trip(BackwardReadyRecord(1, BackwardResult(), attributed, ()))
    assert ready.attributed_results == attributed
    assert ready.attributed_results[0].attribution[0] == transfer
    assert _round_trip(BackwardFinishedRecord(1, BackwardOutcome("succeeded"))).attributed_results == ()
    for feedback in (None, "", [], {}, [None, {"empty": None}]):
        assert _round_trip(BackwardStartedRecord(1, feedback)).feedback == feedback


def test_backward_record_feedback_keeps_the_existing_wider_depth_boundary():
    from agent_composer.expr.lineage import SourceRef
    from agent_composer.nodes.base import BackwardResult
    from agent_composer.runtime.contracts import AttributedBackwardResult, BackwardAttribution, BackwardOutcome
    from agent_composer.runtime.log import BackwardReadyRecord, BackwardFinishedRecord, BackwardStartedRecord

    feedback = "Review."
    for _ in range(140):
        feedback = [feedback]
    result = BackwardResult({"deep": (feedback,)})
    attributed = (AttributedBackwardResult(SourceRef("run", 1), (feedback,),
                    (BackwardAttribution(None, None, 0, None, ()),), result, True),)
    for record in (
        BackwardStartedRecord(1, feedback),
        BackwardReadyRecord(1, result, attributed, ()),
        BackwardFinishedRecord(1, BackwardOutcome("succeeded", result), attributed),
    ):
        assert _round_trip(record) == record


def test_backward_records_isolate_payload_views_before_and_after_log_round_trip():
    from agent_composer.runtime.log import MemoryRunLog

    ready = _backward_records()[1]
    log = MemoryRunLog("run")
    accepted = log.append(ready)
    ready.root_result.input_feedback["topic"][0][1]["uri"] = "changed"
    ready.frame_proposals[0].mutations[0].payload["notes"].clear()
    accepted.record.attributed_results[0].result.variable_feedback.clear()
    accepted.payload["frame_proposals"][0]["mutations"][0]["payload"]["notes"].clear()
    replayed = tuple(log)[0].record
    replayed.root_result.input_feedback["topic"][0].clear()
    replayed.frame_proposals[0].mutations[0].payload["notes"].append("changed")
    assert tuple(log)[0].record == _backward_records()[1]


def test_frame_and_mutation_proposal_order_survives_wire_round_trip():
    from agent_composer.runtime.contracts import FrameProposal
    from agent_composer.runtime.log import BackwardReadyRecord

    proposals = tuple(
        FrameProposal(_frame(name), tuple(
            MemoryMutation(mutation_id, _frame(name), {"step": mutation_id})
            for mutation_id in ("second", "first")
        )) for name in ("later", "earlier")
    )
    decoded = _round_trip(BackwardReadyRecord(1, _result(), (), proposals))
    assert tuple(p.frame_ref for p in decoded.frame_proposals) == tuple(p.frame_ref for p in proposals)
    assert tuple(m.mutation_id for m in decoded.frame_proposals[0].mutations) == ("second", "first")


def _wire(record):
    from agent_composer.runtime.log import RecordEnvelope, SCHEMA_VERSION, encode_record

    return json.loads(encode_record(RecordEnvelope(SCHEMA_VERSION, "run", 0, record)))


def _at_path(value, path):
    for key in path:
        value = value[key]
    return value


@pytest.mark.parametrize("record_kind, path", [
    ("started", ()), ("ready", ()), ("finished", ()),
    ("ready", ("root_result",)),
    ("ready", ("attributed_results", 0)),
    ("ready", ("attributed_results", 0, "source")),
    ("ready", ("attributed_results", 0, "attribution", 0)),
    ("ready", ("attributed_results", 0, "result")),
    ("ready", ("frame_proposals", 0)),
    ("ready", ("frame_proposals", 0, "frame_ref")),
    ("ready", ("frame_proposals", 0, "mutations", 0)),
    ("ready", ("frame_proposals", 0, "mutations", 0, "frame_ref")),
    ("finished", ("outcome",)),
    ("finished", ("outcome", "feedback")),
    ("finished", ("outcome", "failure")),
    ("finished", ("outcome", "failure", "scope")),
    ("finished", ("outcome", "failure", "scope", "occurrence_path", 0)),
    ("finished", ("outcome", "failure", "frame_ref")),
    ("finished", ("outcome", "failure", "cause")),
    ("finished", ("outcome", "learning")),
    ("finished", ("outcome", "learning", "failures", 0)),
    ("finished", ("outcome", "learning", "failures", 0, "failure")),
    ("finished", ("outcome", "learning", "policies", 0)),
    ("finished", ("outcome", "learning", "policies", 3, "failure")),
])
@pytest.mark.parametrize("change", ["extra", "missing"])
def test_backward_wire_nested_records_have_closed_required_fields(record_kind, path, change):
    from agent_composer.runtime.log import decode_record

    records = {"started": _backward_records()[0], "ready": _backward_records()[1],
               "finished": _failed_record()}
    wire = _wire(records[record_kind])
    target = _at_path(wire["payload"], path)
    if change == "extra":
        target["unexpected"] = None
    else:
        target.pop(next(iter(target)))
    with pytest.raises((TypeError, ValueError)):
        decode_record(json.dumps(wire).encode())


@pytest.mark.parametrize("record_kind, path, invalid", [
    ("ready", ("attempt_id",), True),
    ("ready", ("root_result", "input_feedback", "topic", 0), 1),
    ("ready", ("root_result", "self_feedback"), [None]),
    ("ready", ("root_result", "variable_feedback", "tone"), [{}]),
    ("ready", ("attributed_results", 0, "source", "execution_seq"), True),
    ("ready", ("attributed_results", 0, "source", "path"), [False]),
    ("ready", ("attributed_results", 0, "attribution", 0, "parents"), [0]),
    ("ready", ("attributed_results", 0, "attribution", 0, "message_index"), 1),
    ("ready", ("attributed_results", 0, "feedback"), []),
    ("ready", ("attributed_results", 0, "invoked"), 1),
    ("ready", ("frame_proposals", 0, "mutations", 0, "frame_ref", "owner"), "other"),
    ("finished", ("outcome", "failure", "scope", "occurrence_path", 0, "index"), True),
    ("finished", ("outcome", "failure", "node_started_seq"), 0),
    ("finished", ("outcome", "failure", "cause", "retryable"), 1),
    ("finished", ("outcome", "learning", "failures", 0, "frame_ref"), None),
    ("finished", ("outcome", "learning", "policies", 1, "commit_id"), None),
    ("finished", ("outcome", "learning", "policies", 3, "failure"), None),
])
def test_backward_wire_rejects_malformed_nested_values(record_kind, path, invalid):
    from agent_composer.runtime.log import decode_record

    record = _backward_records()[1] if record_kind == "ready" else _failed_record()
    wire = _wire(record)
    _at_path(wire["payload"], path[:-1])[path[-1]] = invalid
    with pytest.raises((TypeError, ValueError)):
        decode_record(json.dumps(wire).encode())


def test_backward_records_reject_subclasses_of_exact_nested_carriers():
    from agent_composer.nodes.base import BackwardResult
    from agent_composer.runtime.contracts import AttributedBackwardResult, BackwardOutcome, FrameProposal
    from agent_composer.runtime.log import BackwardReadyRecord, BackwardFinishedRecord

    class ResultSubclass(BackwardResult):
        pass

    class AttributionSubclass(AttributedBackwardResult):
        pass

    class ProposalSubclass(FrameProposal):
        pass

    class OutcomeSubclass(BackwardOutcome):
        pass

    class FrameSubclass(MemoryFrameRef):
        pass

    class MutationSubclass(MemoryMutation):
        pass

    attributed = _attributed()[0]
    factories = (
        lambda: BackwardReadyRecord(1, ResultSubclass(), (), ()),
        lambda: BackwardReadyRecord(1, _result(), (AttributionSubclass(
            attributed.source, attributed.feedback, attributed.attribution, attributed.result, True,
        ),), ()),
        lambda: BackwardReadyRecord(1, _result(), (), (ProposalSubclass(_frame(), ()),)),
        lambda: BackwardFinishedRecord(1, OutcomeSubclass("succeeded")),
        lambda: FrameProposal(FrameSubclass("root", ("root", "agent"), MemoryOwner.AGENT), ()),
        lambda: FrameProposal(_frame(), (MutationSubclass("m1", _frame(), {}),)),
    )
    for factory in factories:
        with pytest.raises(TypeError):
            factory()


@pytest.mark.parametrize("record_kind, path, field, invalid", [
    ("ready", (), "attempt_id", True),
    ("ready", (), "attributed_results", []),
    ("ready", (), "frame_proposals", []),
    ("ready", ("root_result",), "_input_feedback", {"topic": ["Review."]}),
    ("ready", ("root_result",), "self_feedback", ["Review."]),
    ("ready", ("attributed_results", 0), "_feedback", ["Review."]),
    ("ready", ("attributed_results", 0), "invoked", 1),
    ("ready", ("attributed_results", 0, "source"), "path", []),
    ("ready", ("attributed_results", 0, "attribution", 0), "parents", []),
    ("ready", ("frame_proposals", 0), "mutations", []),
    ("ready", ("frame_proposals", 0, "mutations", 0), "mutation_id", ""),
    ("ready", ("frame_proposals", 0, "frame_ref"), "static_path", ["root", "agent"]),
    ("finished", ("outcome",), "status", "other"),
    ("finished", ("outcome", "failure"), "node_started_seq", True),
    ("finished", ("outcome", "failure", "scope"), "occurrence_path", []),
    ("finished", ("outcome", "failure", "cause"), "retryable", 1),
    ("finished", ("outcome", "learning"), "failures", []),
    ("finished", ("outcome", "learning"), "policies", []),
    ("finished", ("outcome", "learning", "failures", 0), "operation", "other"),
    ("finished", ("outcome", "learning", "policies", 0), "commit_id", "unexpected"),
])
def test_backward_codec_revalidates_nested_carriers_forged_after_construction(record_kind, path, field, invalid):
    from agent_composer.runtime.log import RecordEnvelope, SCHEMA_VERSION, encode_record

    record = _backward_records()[1] if record_kind == "ready" else _failed_record()
    target = record
    for key in path:
        target = target[key] if type(key) is int else getattr(target, key)
    object.__setattr__(target, field, invalid)
    with pytest.raises((TypeError, ValueError)):
        encode_record(RecordEnvelope(SCHEMA_VERSION, "run", 0, record))


@pytest.mark.parametrize("carrier", ["ready_root", "ready_occurrence", "finished"])
def test_backward_codec_rejects_forged_dictionary_subclasses_before_copying(carrier):
    from agent_composer.runtime.log import RecordEnvelope, SCHEMA_VERSION, encode_record

    class DictionarySubclass(dict):
        pass

    record = _failed_record() if carrier == "finished" else _backward_records()[1]
    result = (record.outcome.feedback if carrier == "finished" else record.root_result
              if carrier == "ready_root" else record.attributed_results[0].result)
    object.__setattr__(result, "_variable_feedback", DictionarySubclass(tone=("Review.",)))
    with pytest.raises(TypeError, match="exact dictionaries"):
        encode_record(RecordEnvelope(SCHEMA_VERSION, "run", 0, record))


@pytest.mark.parametrize("feedback", [1, True, [0], {"field": ()}, {1: "Review."}])
def test_backward_started_rejects_nonordinary_feedback_before_recording(feedback):
    from agent_composer.runtime.log import BackwardStartedRecord

    with pytest.raises((TypeError, ValueError)):
        BackwardStartedRecord(1, feedback)


def test_forward_cause_wire_rejects_hybrid_and_extra_origin_fields():
    from agent_composer.runtime.log import decode_record

    wire = _wire(_failed_record("forward"))
    cause = wire["payload"]["outcome"]["failure"]["cause"]
    cause["stage"] = "backend"
    with pytest.raises(ValueError):
        decode_record(json.dumps(wire).encode())
    del cause["stage"]
    cause["trace"][0]["file"] = "/private/source.yaml"
    with pytest.raises(ValueError):
        decode_record(json.dumps(wire).encode())


@pytest.mark.parametrize("node_id", [None, "", "agent"])
def test_backward_failure_cause_round_trips_optional_execution_attribution(node_id):
    from agent_composer.diagnostics import Failure, OriginRef
    from agent_composer.runtime.contracts import BackwardFailure, BackwardOutcome
    from agent_composer.runtime.log import BackwardFinishedRecord

    cause = Failure("graph.terminal_unreachable", "No executing producer.", node_id,
                    (OriginRef("root.boundary"),))
    record = BackwardFinishedRecord(1, BackwardOutcome(
        "failed", failure=BackwardFailure("backward", "backward.execute", "Could not finish.", cause=cause),
    ))
    decoded = _round_trip(record)
    assert decoded.outcome.failure.cause == cause
    assert type(decoded.outcome.failure.cause) is Failure
    assert decoded.outcome.failure.cause.trace == (OriginRef("root.boundary"),)


@pytest.mark.parametrize("invalid", [1, True, [], {}])
def test_backward_failure_cause_wire_rejects_nonstring_execution_attribution(invalid):
    from agent_composer.runtime.log import decode_record

    wire = _wire(_failed_record("forward"))
    wire["payload"]["outcome"]["failure"]["cause"]["node_id"] = invalid
    with pytest.raises(ValueError):
        decode_record(json.dumps(wire).encode())


@pytest.mark.parametrize("record_kind", ["failed", "routed"])
@pytest.mark.parametrize("node_id", [None, ""])
def test_node_failure_wire_keeps_required_execution_attribution(record_kind, node_id):
    from agent_composer.diagnostics import Failure
    from agent_composer.runtime.log import NodeFailedRecord, NodeRoutedRecord, decode_record

    failure = Failure("graph.terminal_unreachable", "Terminal unreachable.", "agent")
    record = (NodeFailedRecord(failure) if record_kind == "failed" else
              NodeRoutedRecord("agent", "other", terminal_failure=failure))
    wire = _wire(record)
    field = "failure" if record_kind == "failed" else "terminal_failure"
    wire["payload"][field]["node_id"] = node_id
    with pytest.raises(ValueError):
        decode_record(json.dumps(wire).encode())
