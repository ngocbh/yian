"""Detached backward proposal and record contracts without reducer semantics."""

from dataclasses import FrozenInstanceError
import json

import pytest

from yian.memory import MemoryMutation


@pytest.mark.parametrize("failed,index", [
    (False, 7), (False, 8), (False, 9), (False, 10), (False, 11),
    (True, 10), (True, 11),
], ids=["started", "ready", "requested", "succeeded", "finished", "failed", "failed-finished"])
def test_historical_backward_prepares_recorded_schema_bytes(failed, index):
    import yian.run as run_module
    from yian.compose import load_flow
    from yian.runtime.log import decode_record
    from yian.runtime.state import _HistoricalLogCursor, commit_prepared, prepare_record
    from tests.engine.test_backward_recovery import (
        _RECORDED_BACKWARD, _RECORDED_BACKWARD_FAILED, _RECORDED_BACKWARD_FLOW,
    )
    from tests.engine.test_run_recovery_prefixes import _ReadOnlyLog

    wires = (_RECORDED_BACKWARD_FAILED if failed else _RECORDED_BACKWARD)[:index + 1]
    state = run_module._replay_run_log(
        load_flow(_RECORDED_BACKWARD_FLOW),
        _ReadOnlyLog(tuple(decode_record(wire) for wire in wires[:-1])),
    )
    before = state.backward_state
    prepared = prepare_record(state, decode_record(wires[-1]).record)
    assert prepared._plan.envelope_bytes == wires[-1]
    assert state.backward_state is before
    cursor = _HistoricalLogCursor(wires)
    for wire in wires[:-1]:
        cursor.append(decode_record(wire).record)
    commit_prepared(cursor, state, prepared)
    assert state.next_sequence == index + 1
    assert state.backward_state is not before


def test_memory_proposal_owns_detached_mutations_in_supplied_order():
    from yian.runtime.contracts import MemoryProposal

    payload = {"values": [1]}
    mutations = (MemoryMutation("second", payload), MemoryMutation("first", {}))
    proposal = MemoryProposal("agent", mutations)
    payload["values"].append(2)
    proposal.mutations[0].payload["values"].clear()
    assert [mutation.mutation_id for mutation in proposal.mutations] == ["second", "first"]
    assert proposal.mutations[0].payload == {"values": [1]}
    assert proposal.mutations[0] is not mutations[0]
    assert proposal.memory == "agent"
    with pytest.raises(FrozenInstanceError):
        proposal.mutations = ()
    with pytest.raises(TypeError):
        MemoryProposal("agent", list(mutations))
    with pytest.raises(ValueError, match="unique"):
        MemoryProposal("agent", (mutations[0], mutations[0]))
    assert MemoryProposal("agent", ()).mutations == ()


def _result():
    from yian.nodes.base import BackwardResult

    return BackwardResult({"topic": ([None, {"uri": "Review."}],)},
                          self_feedback=("Behavior.",), variable_feedback={"tone": ("Concise.",)})


def _attributed():
    from yian.expr.lineage import SourceRef
    from yian.runtime.contracts import AttributedBackwardResult, BackwardAttribution

    return (AttributedBackwardResult(SourceRef("run", 1), ("Review.",),
                                     (BackwardAttribution(None, None, 0, None, ()),), _result(), True),)


def test_backward_record_constructors_detach_values_without_reducer_inference():
    from yian.runtime.contracts import BackwardOutcome, MemoryProposal
    from yian.runtime.log import BackwardStartedRecord, BackwardReadyRecord, BackwardFinishedRecord

    feedback = {"parts": [None, "Review."]}
    started = BackwardStartedRecord(1, feedback)
    feedback["parts"].clear()
    started.feedback["parts"].clear()
    assert started.feedback == {"parts": [None, "Review."]}
    proposal = MemoryProposal("agent", (MemoryMutation("m1", {"notes": ["one"]}),))
    ready = BackwardReadyRecord(1, _result(), _attributed(), (proposal,))
    ready.root_result.input_feedback["topic"][0].clear()
    ready.attributed_results[0].feedback[0].upper()
    assert ready.root_result == _result()
    assert ready.proposals[0] is not proposal
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


def test_commit_attempt_ids_are_required_keyword_only_positive_integers():
    from yian.memory import MemoryOperationFailure
    from yian.runtime.log import MemoryCommitRequestedRecord, MemoryCommitSucceededRecord, MemoryCommitFailedRecord

    failure = MemoryOperationFailure("backend", "failed", "Failed.", True)
    for carrier, args in (
        (MemoryCommitRequestedRecord, ("history", "commit-1", ())),
        (MemoryCommitSucceededRecord, ("history", "commit-1")),
        (MemoryCommitFailedRecord, ("history", "commit-1", failure)),
    ):
        with pytest.raises(TypeError):
            carrier(*args)
        assert carrier(*args, attempt_id=2).attempt_id == 2
        with pytest.raises(TypeError):
            carrier(*args, 2)
        for invalid in (None, 0, -1, True, "2", 2.0):
            with pytest.raises((TypeError, ValueError)):
                carrier(*args, attempt_id=invalid)


def _backward_records():
    from yian.memory import LearningResult, PolicyCommitOutcome
    from yian.runtime.contracts import BackwardOutcome, MemoryProposal
    from yian.runtime.log import BackwardStartedRecord, BackwardReadyRecord, BackwardFinishedRecord

    proposal = MemoryProposal("agent", (MemoryMutation("m1", {"notes": ["one"]}),))
    outcome = BackwardOutcome("succeeded", _result(), learning=LearningResult(
        "succeeded", policies=(PolicyCommitOutcome("history", "succeeded", "commit-1"),),
    ))
    return [BackwardStartedRecord(1, [None, {"uri": "Review."}]),
            BackwardReadyRecord(1, _result(), _attributed(), (proposal,)),
            BackwardFinishedRecord(1, outcome, _attributed())]


def _round_trip(record):
    from yian.runtime.log import SCHEMA_VERSION, RecordEnvelope, decode_record, encode_record

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


def test_commit_wire_preserves_supplied_attempt_ids():
    from yian.memory import MemoryOperationFailure
    from yian.runtime.log import MemoryCommitRequestedRecord, MemoryCommitSucceededRecord, MemoryCommitFailedRecord

    failure = MemoryOperationFailure("backend", "failed", "Failed.", False)
    for attempt_id in (1, 3):
        for record in (
            MemoryCommitRequestedRecord("history", "commit-1", (), attempt_id=attempt_id),
            MemoryCommitSucceededRecord("history", "commit-1", attempt_id=attempt_id),
            MemoryCommitFailedRecord("history", "commit-1", failure, attempt_id=attempt_id),
        ):
            assert _round_trip(record).attempt_id == attempt_id


def _failed_record(cause_kind="memory", *, completed=True, learning_status="failed"):
    from yian.diagnostics import Failure, OriginRef
    from yian.memory import LearningFailure, LearningResult, MemoryOperationFailure, PolicyCommitOutcome
    from yian.nodes.contracts import ExecutionPosition, ExecutionScope
    from yian.runtime.contracts import BackwardFailure, BackwardOutcome
    from yian.runtime.log import BackwardFinishedRecord

    cause = MemoryOperationFailure("backend-stage", "backend.failed", "Retry later.", True)
    causes = {"none": None, "memory": cause,
              "forward": Failure("node.execute", "Failed.", "agent", (OriginRef("agent.prompt"),),
                                 "ValueError", "forward traceback")}
    failure = BackwardFailure(
        "learn", "memory.learn", "Could not propose.",
        ExecutionScope(("each", "agent"), ("each",),
                       (ExecutionPosition("map_item", ("each",), 2),)),
        "run", "each/2/agent", 7, "history", causes[cause_kind],
        "MemoryOperationError", "backward traceback",
    )
    learning = LearningResult(learning_status, (
        LearningFailure("history", "learn", cause),
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
    from yian.expr.lineage import SourceRef
    from yian.nodes.base import BackwardResult
    from yian.runtime.contracts import AttributedBackwardResult, BackwardAttribution, BackwardOutcome
    from yian.runtime.log import BackwardReadyRecord, BackwardFinishedRecord, BackwardStartedRecord

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
    from yian.expr.lineage import SourceRef
    from yian.nodes.base import BackwardResult
    from yian.runtime.contracts import AttributedBackwardResult, BackwardAttribution, BackwardOutcome
    from yian.runtime.log import BackwardReadyRecord, BackwardFinishedRecord, BackwardStartedRecord

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
    from yian.runtime.log import MemoryRunLog

    ready = _backward_records()[1]
    log = MemoryRunLog("run")
    accepted = log.append(ready)
    ready.root_result.input_feedback["topic"][0][1]["uri"] = "changed"
    ready.proposals[0].mutations[0].payload["notes"].clear()
    accepted.record.attributed_results[0].result.variable_feedback.clear()
    accepted.payload["proposals"][0]["mutations"][0]["payload"]["notes"].clear()
    replayed = tuple(log)[0].record
    replayed.root_result.input_feedback["topic"][0].clear()
    replayed.proposals[0].mutations[0].payload["notes"].append("changed")
    assert tuple(log)[0].record == _backward_records()[1]


def test_memory_and_mutation_proposal_order_survives_wire_round_trip():
    from yian.runtime.contracts import MemoryProposal
    from yian.runtime.log import BackwardReadyRecord

    proposals = tuple(
        MemoryProposal(name, tuple(
            MemoryMutation(mutation_id, {"step": mutation_id})
            for mutation_id in ("second", "first")
        )) for name in ("later", "earlier")
    )
    decoded = _round_trip(BackwardReadyRecord(1, _result(), (), proposals))
    assert tuple(p.memory for p in decoded.proposals) == ("later", "earlier")
    assert tuple(m.mutation_id for m in decoded.proposals[0].mutations) == ("second", "first")


def _wire(record):
    from yian.runtime.log import RecordEnvelope, SCHEMA_VERSION, encode_record

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
    ("ready", ("proposals", 0)),
    ("ready", ("proposals", 0, "mutations", 0)),
    ("finished", ("outcome",)),
    ("finished", ("outcome", "feedback")),
    ("finished", ("outcome", "failure")),
    ("finished", ("outcome", "failure", "scope")),
    ("finished", ("outcome", "failure", "scope", "occurrence_path", 0)),
    ("finished", ("outcome", "failure", "cause")),
    ("finished", ("outcome", "learning")),
    ("finished", ("outcome", "learning", "failures", 0)),
    ("finished", ("outcome", "learning", "failures", 0, "failure")),
    ("finished", ("outcome", "learning", "policies", 0)),
    ("finished", ("outcome", "learning", "policies", 3, "failure")),
])
@pytest.mark.parametrize("change", ["extra", "missing"])
def test_backward_wire_nested_records_have_closed_required_fields(record_kind, path, change):
    from yian.runtime.log import decode_record

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
    ("ready", ("proposals", 0, "memory"), ""),
    ("finished", ("outcome", "failure", "scope", "occurrence_path", 0, "index"), True),
    ("finished", ("outcome", "failure", "node_started_seq"), 0),
    ("finished", ("outcome", "failure", "cause", "retryable"), 1),
    ("finished", ("outcome", "learning", "failures", 0, "memory"), ""),
    ("finished", ("outcome", "learning", "policies", 1, "commit_id"), None),
    ("finished", ("outcome", "learning", "policies", 3, "failure"), None),
])
def test_backward_wire_rejects_malformed_nested_values(record_kind, path, invalid):
    from yian.runtime.log import decode_record

    record = _backward_records()[1] if record_kind == "ready" else _failed_record()
    wire = _wire(record)
    _at_path(wire["payload"], path[:-1])[path[-1]] = invalid
    with pytest.raises((TypeError, ValueError)):
        decode_record(json.dumps(wire).encode())


def test_backward_records_reject_subclasses_of_exact_nested_carriers():
    from yian.nodes.base import BackwardResult
    from yian.runtime.contracts import AttributedBackwardResult, BackwardOutcome, MemoryProposal
    from yian.runtime.log import BackwardReadyRecord, BackwardFinishedRecord

    class ResultSubclass(BackwardResult):
        pass

    class AttributionSubclass(AttributedBackwardResult):
        pass

    class ProposalSubclass(MemoryProposal):
        pass

    class OutcomeSubclass(BackwardOutcome):
        pass

    class MutationSubclass(MemoryMutation):
        pass

    attributed = _attributed()[0]
    factories = (
        lambda: BackwardReadyRecord(1, ResultSubclass(), (), ()),
        lambda: BackwardReadyRecord(1, _result(), (AttributionSubclass(
            attributed.source, attributed.feedback, attributed.attribution, attributed.result, True,
        ),), ()),
        lambda: BackwardReadyRecord(1, _result(), (), (ProposalSubclass("agent", ()),)),
        lambda: BackwardFinishedRecord(1, OutcomeSubclass("succeeded")),
        lambda: MemoryProposal("agent", (MutationSubclass("m1", {}),)),
    )
    for factory in factories:
        with pytest.raises(TypeError):
            factory()


@pytest.mark.parametrize("record_kind, path, field, invalid", [
    ("ready", (), "attempt_id", True),
    ("ready", (), "attributed_results", []),
    ("ready", (), "proposals", []),
    ("ready", ("root_result",), "_input_feedback", {"topic": ["Review."]}),
    ("ready", ("root_result",), "self_feedback", ["Review."]),
    ("ready", ("attributed_results", 0), "_feedback", ["Review."]),
    ("ready", ("attributed_results", 0), "invoked", 1),
    ("ready", ("attributed_results", 0, "source"), "path", []),
    ("ready", ("attributed_results", 0, "attribution", 0), "parents", []),
    ("ready", ("proposals", 0), "mutations", []),
    ("ready", ("proposals", 0, "mutations", 0), "mutation_id", ""),
    ("ready", ("proposals", 0), "memory", ""),
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
    from yian.runtime.log import RecordEnvelope, SCHEMA_VERSION, encode_record

    record = _backward_records()[1] if record_kind == "ready" else _failed_record()
    target = record
    for key in path:
        target = target[key] if type(key) is int else getattr(target, key)
    object.__setattr__(target, field, invalid)
    with pytest.raises((TypeError, ValueError)):
        encode_record(RecordEnvelope(SCHEMA_VERSION, "run", 0, record))


@pytest.mark.parametrize("carrier", ["ready_root", "ready_occurrence", "finished"])
def test_backward_codec_rejects_forged_dictionary_subclasses_before_copying(carrier):
    from yian.runtime.log import RecordEnvelope, SCHEMA_VERSION, encode_record

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
    from yian.runtime.log import BackwardStartedRecord

    with pytest.raises((TypeError, ValueError)):
        BackwardStartedRecord(1, feedback)


def test_forward_cause_wire_rejects_hybrid_and_extra_origin_fields():
    from yian.runtime.log import decode_record

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
    from yian.diagnostics import Failure, OriginRef
    from yian.runtime.contracts import BackwardFailure, BackwardOutcome
    from yian.runtime.log import BackwardFinishedRecord

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
    from yian.runtime.log import decode_record

    wire = _wire(_failed_record("forward"))
    wire["payload"]["outcome"]["failure"]["cause"]["node_id"] = invalid
    with pytest.raises(ValueError):
        decode_record(json.dumps(wire).encode())


@pytest.mark.parametrize("record_kind", ["failed", "routed"])
@pytest.mark.parametrize("node_id", [None, ""])
def test_node_failure_wire_keeps_required_execution_attribution(record_kind, node_id):
    from yian.diagnostics import Failure
    from yian.runtime.log import NodeFailedRecord, NodeRoutedRecord, decode_record

    failure = Failure("graph.terminal_unreachable", "Terminal unreachable.", "agent")
    record = (NodeFailedRecord(failure) if record_kind == "failed" else
              NodeRoutedRecord("agent", "other", terminal_failure=failure))
    wire = _wire(record)
    field = "failure" if record_kind == "failed" else "terminal_failure"
    wire["payload"][field]["node_id"] = node_id
    with pytest.raises(ValueError):
        decode_record(json.dumps(wire).encode())


def _retained_backward(*, keys=None, feedback="Review.", topic="reefs"):
    from yian.compile.model import START_ID, FlowOutput
    from yian.memory import DeclSite, MemorySnapshotEnvelope, MemorySpec
    from yian.nodes.base import BackwardResult
    from yian.runtime.backward import compute_backward
    from tests.engine._backward_fakes import RecordingBackwardNode
    from tests.engine._graph_builder import _engine, _graph

    work = RecordingBackwardNode(
        "work", lambda inputs: inputs["x"], lambda _context, messages: BackwardResult({"x": messages}),
        reads={"x": "${input.topic}"},
    )
    flow = _graph([work], [(START_ID, "work")], outputs=[FlowOutput("answer", "${work.output}")])
    root = DeclSite((), ("root",), None)
    engine = _engine(
        flow, {"topic": topic}, learnable=True,
        memories=None if keys is None else tuple(MemorySpec(k, k, {}, (root,), False) for k in keys),
        memory_stacks=None if keys is None else (),
        memory_snapshots=None if keys is None else tuple(MemorySnapshotEnvelope(k, {}) for k in keys),
    )
    list(engine.run())
    return engine, compute_backward(engine.state.trace, feedback)


def test_attributed_validation_reuses_recorded_transfers_without_callbacks_or_materialization(monkeypatch):
    import yian.runtime.learning as learning
    from yian.runtime.learning import validate_attributed_backward

    engine, (root, attributed) = _retained_backward()
    monkeypatch.setattr(learning, "_materialize_context", lambda *_: pytest.fail("materialized context"))
    monkeypatch.setattr(learning, "_output_context", lambda *_: pytest.fail("formatted output context"))
    monkeypatch.setattr(type(engine.state.trace), "backward_handlers",
                        property(lambda _: pytest.fail("read backward callbacks")))
    assert validate_attributed_backward(engine.state.trace, "Review.", root, attributed) is None
    assert root.input_feedback == {"topic": ("Review.",)}


def test_backward_attempt_records_prepare_append_then_reduce_without_forward_effects():
    from yian.runtime.contracts import BackwardOutcome
    from yian.runtime.log import BackwardStartedRecord, BackwardReadyRecord, BackwardFinishedRecord, encode_pool
    from yian.runtime.state import commit_prepared, prepare_record

    engine, (root, attributed) = _retained_backward()
    state, log = engine.state, engine.log
    forward = (state.result, encode_pool(state.pool), dict(state.node_state), dict(state.edge_state),
               tuple(state.ready), tuple(state.in_flight), state.trace.occurrences)
    initial = state.backward_state
    records = (BackwardStartedRecord(1, "Review."), BackwardReadyRecord(1, root, attributed, ()),
               BackwardFinishedRecord(1, BackwardOutcome("succeeded", root)))
    for record in records:
        prior = state.backward_state
        sequence = state.next_sequence
        prepared = prepare_record(state, record)
        assert state.backward_state is prior
        assert log.next_sequence == sequence
        assert commit_prepared(log, state, prepared) == ()
        assert state.next_sequence == state.trace.next_sequence == log.next_sequence == sequence + 1
    assert state.backward_state is not initial
    assert state.backward_state.started == records[0]
    assert state.backward_state.ready == records[1]
    assert state.backward_state.finished == records[2]
    assert state.backward_state.outcome == records[2].outcome
    assert state.backward_state.can_start is True
    assert forward == (state.result, encode_pool(state.pool), dict(state.node_state), dict(state.edge_state),
                       tuple(state.ready), tuple(state.in_flight), state.trace.occurrences)


def _accept_backward(engine, record):
    from yian.runtime.state import commit_prepared, prepare_record

    return commit_prepared(engine.log, engine.state, prepare_record(engine.state, record))


def _proposal_table():
    from yian.runtime.contracts import MemoryProposal

    proposals = (
        MemoryProposal("alpha", (
            MemoryMutation("second", {"text": "A"}),
            MemoryMutation("first", {"text": "B"}),
        )),
        MemoryProposal("beta", ()),
    )
    return ("alpha", "beta"), proposals


def test_backward_policy_batches_and_unfinished_prefixes_follow_accepted_layout_order():
    from yian.memory import LearningResult
    from yian.runtime.contracts import BackwardOutcome
    from yian.runtime.learning import memory_commit_id
    from yian.runtime.log import (BackwardStartedRecord, BackwardReadyRecord, BackwardFinishedRecord,
        MemoryCommitRequestedRecord, MemoryCommitSucceededRecord)

    keys, proposals = _proposal_table()
    engine, (root, attributed) = _retained_backward(keys=keys)
    _accept_backward(engine, BackwardStartedRecord(1, "Review."))
    assert engine.state.backward_state.outcome.feedback is None
    assert engine.state.backward_state.outcome.learning.status == "incomplete"
    assert tuple(p.status for p in engine.state.backward_state.policy_outcomes) == ("unrequested", "unrequested")
    _accept_backward(engine, BackwardReadyRecord(1, root, attributed, proposals))
    assert engine.state.backward_state.outcome.feedback == root
    for policy, batch in (("alpha", proposals[0].mutations), ("beta", ())):
        commit_id = memory_commit_id(engine.state.run_id, policy, batch, 1)
        _accept_backward(engine, MemoryCommitRequestedRecord(policy, commit_id, batch, attempt_id=1))
        assert engine.state.backward_state.ever_requested
        assert next(p for p in engine.state.backward_state.policy_outcomes if p.memory == policy).status == "unknown"
        assert engine.state.backward_state.can_start is False
        _accept_backward(engine, MemoryCommitSucceededRecord(policy, commit_id, attempt_id=1))
    interrupted = engine.state.backward_state.outcome
    assert interrupted.status == "failed"
    assert interrupted.failure.code == "backward.interrupted"
    assert interrupted.learning.status == "incomplete"
    policies = engine.state.backward_state.policy_outcomes
    assert tuple(p.status for p in policies) == ("succeeded", "succeeded")
    outcome = BackwardOutcome("succeeded", root, learning=LearningResult("succeeded", (), policies))
    _accept_backward(engine, BackwardFinishedRecord(1, outcome))
    assert engine.state.backward_state.outcome == outcome
    assert engine.state.backward_state.current.requests[0].mutations == proposals[0].mutations
    assert engine.state.backward_state.current.requests[1].mutations == ()


@pytest.mark.parametrize("ready", [False, True])
def test_interrupted_attempt_requires_explicit_finish_before_a_fresh_start(ready):
    from yian.runtime.log import BackwardStartedRecord, BackwardReadyRecord, BackwardFinishedRecord
    from yian.runtime.state import prepare_record

    engine, (root, attributed) = _retained_backward()
    _accept_backward(engine, BackwardStartedRecord(1, "Review."))
    if ready:
        _accept_backward(engine, BackwardReadyRecord(1, root, attributed, ()))
    prior, sequence = engine.state.backward_state, engine.log.next_sequence
    for invalid in (BackwardStartedRecord(2, {}), BackwardStartedRecord(2, "Review.")):
        with pytest.raises(ValueError):
            prepare_record(engine.state, invalid)
        assert engine.state.backward_state is prior
        assert engine.log.next_sequence == sequence
    _accept_backward(engine, BackwardFinishedRecord(1, prior.outcome))
    assert engine.state.backward_state.can_start is True
    _accept_backward(engine, BackwardStartedRecord(2, "New feedback."))
    assert len(engine.state.backward_state.attempts) == 2
    assert engine.state.backward_state.ready is None
    assert engine.state.backward_state.outcome.feedback is None


@pytest.mark.parametrize("change", ["missing", "duplicate", "reordered", "foreign", "feedback", "input_coordinate",
    "message_coordinate", "contribution_coordinate", "parent_coordinate", "unreached", "unknown_input",
    "unknown_variable", "short_list", "root"])
def test_ready_rejects_inventory_local_contract_transfer_and_root_disagreement_before_append(change):
    from yian.expr.lineage import SourceRef
    from yian.nodes.base import BackwardResult
    from yian.runtime.contracts import AttributedBackwardResult, BackwardAttribution
    from yian.runtime.log import BackwardStartedRecord, BackwardReadyRecord
    from yian.runtime.state import prepare_record

    engine, (root, attributed) = _retained_backward(topic=[1, 2])
    _accept_backward(engine, BackwardStartedRecord(1, "Review."))
    submitted = list(attributed)
    work_index = next(i for i, occurrence in enumerate(engine.state.trace.occurrences) if occurrence.node_id == "work")
    item = submitted[work_index]
    source, feedback, attribution, result, invoked = item.source, item.feedback, item.attribution, item.result, item.invoked
    if change == "missing":
        submitted.pop()
    elif change == "duplicate":
        submitted[-1] = submitted[0]
    elif change == "reordered":
        submitted.reverse()
    elif change == "root":
        root = BackwardResult({"topic": ("different",)})
    else:
        if change == "foreign":
            source = SourceRef("foreign-run", source.execution_seq)
        elif change == "feedback":
            feedback = ("different",)
        elif change == "unreached":
            feedback, attribution, result, invoked = (), (), BackwardResult(), False
        elif change == "unknown_input":
            result = BackwardResult({"foreign": ("Review.",)})
        elif change == "unknown_variable":
            result = BackwardResult(variable_feedback={"foreign": ("Review.",)})
        elif change == "short_list":
            result = BackwardResult({"x": ([None],)})
        else:
            original = attribution[0]
            attribution = (BackwardAttribution(
                original.sender, "foreign" if change == "input_coordinate" else original.input_name,
                1 if change == "message_coordinate" else original.message_index,
                1 if change == "contribution_coordinate" else original.contribution_index,
                (0, 1) if change == "parent_coordinate" else original.parents,
            ),)
        submitted[work_index] = AttributedBackwardResult(source, feedback, attribution, result, invoked)
    prior, sequence = engine.state.backward_state, engine.log.next_sequence
    with pytest.raises((TypeError, ValueError)):
        prepare_record(engine.state, BackwardReadyRecord(1, root, tuple(submitted), ()))
    assert engine.state.backward_state is prior
    assert engine.log.next_sequence == engine.state.trace.next_sequence == sequence


@pytest.mark.parametrize("change", ["missing", "reordered", "duplicate", "foreign"])
def test_ready_proposals_require_the_complete_table_in_order(change):
    from yian.runtime.contracts import MemoryProposal
    from yian.runtime.log import BackwardStartedRecord, BackwardReadyRecord
    from yian.runtime.state import prepare_record

    keys, proposals = _proposal_table()
    engine, (root, attributed) = _retained_backward(keys=keys)
    _accept_backward(engine, BackwardStartedRecord(1, "Review."))
    submitted = list(proposals)
    if change == "missing":
        submitted.pop()
    elif change == "reordered":
        submitted.reverse()
    elif change == "duplicate":
        submitted[-1] = submitted[0]
    elif change == "foreign":
        submitted[-1] = MemoryProposal("foreign", ())
    with pytest.raises(ValueError):
        prepare_record(engine.state, BackwardReadyRecord(1, root, attributed, tuple(submitted)))
    assert engine.state.backward_state.ready is None


def test_mutation_ids_may_repeat_across_different_recorded_memories():
    from yian.runtime.contracts import MemoryProposal
    from yian.runtime.log import BackwardStartedRecord, BackwardReadyRecord

    keys, proposals = _proposal_table()
    proposals = (proposals[0], MemoryProposal("beta", (MemoryMutation("second", {}),)))
    engine, (root, attributed) = _retained_backward(keys=keys)
    _accept_backward(engine, BackwardStartedRecord(1, "Review."))
    _accept_backward(engine, BackwardReadyRecord(1, root, attributed, proposals))
    assert engine.state.backward_state.ready.proposals == proposals


@pytest.mark.parametrize("retryable", [False, True])
@pytest.mark.parametrize("code", ["learn.failed", "backward.interrupted"])
def test_learner_failed_finished_keeps_completed_feedback_and_all_unrequested_policies(retryable, code):
    from yian.memory import LearningFailure, LearningResult, MemoryOperationFailure
    from yian.runtime.contracts import BackwardFailure, BackwardOutcome
    from yian.runtime.log import BackwardStartedRecord, BackwardFinishedRecord

    keys, _ = _proposal_table()
    engine, (root, attributed) = _retained_backward(keys=keys)
    _accept_backward(engine, BackwardStartedRecord(1, "Review."))
    cause = MemoryOperationFailure("provider-stage", code, "No proposal.", retryable)
    failure = BackwardFailure("learn", cause.code, cause.message, memory="alpha", cause=cause)
    learning = LearningResult("failed", (LearningFailure("alpha", "learn", cause),),
                              engine.state.backward_state.policy_outcomes)
    finished = BackwardFinishedRecord(1, BackwardOutcome("failed", root, failure, learning), attributed)
    _accept_backward(engine, finished)
    assert engine.state.backward_state.outcome == finished.outcome
    assert engine.state.backward_state.finished.attributed_results == attributed
    assert engine.state.backward_state.ready is None
    assert engine.state.backward_state.can_start
    _accept_backward(engine, BackwardStartedRecord(2, "Different."))
    assert engine.state.backward_state.ready is None and engine.state.backward_state.outcome.feedback is None


@pytest.mark.parametrize("retryable", [False, True])
@pytest.mark.parametrize("code", ["commit.failed", "backward.interrupted"])
def test_known_commit_failure_stops_later_requests_and_admits_a_later_attempt(retryable, code):
    from yian.memory import LearningFailure, LearningResult, MemoryOperationFailure
    from yian.runtime.contracts import BackwardFailure, BackwardOutcome
    from yian.runtime.learning import memory_commit_id
    from yian.runtime.log import (BackwardStartedRecord, BackwardReadyRecord, BackwardFinishedRecord,
        MemoryCommitRequestedRecord, MemoryCommitFailedRecord)
    from yian.runtime.state import prepare_record

    keys, proposals = _proposal_table()
    engine, (root, attributed) = _retained_backward(keys=keys)
    _accept_backward(engine, BackwardStartedRecord(1, "Review."))
    _accept_backward(engine, BackwardReadyRecord(1, root, attributed, proposals))
    batch = proposals[0].mutations
    commit_id = memory_commit_id(engine.state.run_id, "alpha", batch, 1)
    _accept_backward(engine, MemoryCommitRequestedRecord("alpha", commit_id, batch, attempt_id=1))
    cause = MemoryOperationFailure("backend-stage", code, "Rejected.", retryable)
    _accept_backward(engine, MemoryCommitFailedRecord("alpha", commit_id, cause, attempt_id=1))
    assert engine.state.backward_state.outcome.learning.status == "failed"
    assert tuple(p.status for p in engine.state.backward_state.policy_outcomes) == ("failed", "unrequested")
    with pytest.raises(ValueError):
        prepare_record(engine.state, MemoryCommitRequestedRecord("beta", memory_commit_id(engine.state.run_id, "beta", (), 1), (), attempt_id=1))
    learning = LearningResult("failed", (LearningFailure("alpha", "commit", cause),), engine.state.backward_state.policy_outcomes)
    failure = BackwardFailure("commit", cause.code, cause.message, memory="alpha", cause=cause)
    _accept_backward(engine, BackwardFinishedRecord(1, BackwardOutcome("failed", root, failure, learning)))
    assert engine.state.backward_state.ever_requested and engine.state.backward_state.can_start
    _accept_backward(engine, BackwardStartedRecord(2, "Different."))
    assert engine.state.backward_state.current.started.attempt_id == 2


@pytest.mark.parametrize("closing", ["succeeded", "commit_failed", "interrupted_after_request"])
def test_any_finished_attempt_admits_a_later_start_with_fresh_proposals(closing):
    from yian.memory import LearningFailure, LearningResult, MemoryOperationFailure
    from yian.runtime.backward import compute_backward
    from yian.runtime.contracts import BackwardFailure, BackwardOutcome
    from yian.runtime.learning import memory_commit_id
    from yian.runtime.log import (BackwardStartedRecord, BackwardReadyRecord, BackwardFinishedRecord,
        MemoryCommitRequestedRecord, MemoryCommitSucceededRecord, MemoryCommitFailedRecord)
    from yian.runtime.state import prepare_record

    keys, proposals = _proposal_table()
    engine, (root, attributed) = _retained_backward(keys=keys)
    batch = proposals[0].mutations
    _accept_backward(engine, BackwardStartedRecord(1, "Review."))
    _accept_backward(engine, BackwardReadyRecord(1, root, attributed, proposals))
    commit_id = memory_commit_id(engine.state.run_id, "alpha", batch, 1)
    _accept_backward(engine, MemoryCommitRequestedRecord("alpha", commit_id, batch, attempt_id=1))
    if closing == "succeeded":
        _accept_backward(engine, MemoryCommitSucceededRecord("alpha", commit_id, attempt_id=1))
        beta_id = memory_commit_id(engine.state.run_id, "beta", (), 1)
        _accept_backward(engine, MemoryCommitRequestedRecord("beta", beta_id, (), attempt_id=1))
        _accept_backward(engine, MemoryCommitSucceededRecord("beta", beta_id, attempt_id=1))
        policies = engine.state.backward_state.policy_outcomes
        outcome = BackwardOutcome("succeeded", root, learning=LearningResult("succeeded", (), policies))
    elif closing == "commit_failed":
        cause = MemoryOperationFailure("backend-stage", "commit.failed", "Rejected.", True)
        _accept_backward(engine, MemoryCommitFailedRecord("alpha", commit_id, cause, attempt_id=1))
        learning = LearningResult("failed", (LearningFailure("alpha", "commit", cause),),
                                  engine.state.backward_state.policy_outcomes)
        failure = BackwardFailure("commit", cause.code, cause.message, memory="alpha", cause=cause)
        outcome = BackwardOutcome("failed", root, failure, learning)
    else:
        outcome = engine.state.backward_state.outcome
        assert outcome.failure.code == "backward.interrupted"
    assert not engine.state.backward_state.can_start
    with pytest.raises(ValueError):
        prepare_record(engine.state, BackwardStartedRecord(2, "Later review."))
    _accept_backward(engine, BackwardFinishedRecord(1, outcome))
    assert engine.state.backward_state.can_start
    _accept_backward(engine, BackwardStartedRecord(2, "Later review."))
    backward = engine.state.backward_state
    assert [attempt.started.attempt_id for attempt in backward.attempts] == [1, 2]
    assert backward.attempts[0].finished.outcome == outcome
    assert backward.ready is None and backward.current.requests == ()
    assert [policy.status for policy in backward.policy_outcomes] == ["unrequested", "unrequested"]
    later_root, later_attributed = compute_backward(engine.state.trace, "Later review.")
    _accept_backward(engine, BackwardReadyRecord(2, later_root, later_attributed, proposals))
    assert engine.state.backward_state.ready.root_result == later_root


def test_every_committing_attempt_uses_its_own_attempt_in_the_commit_id():
    from yian.memory import LearningResult
    from yian.runtime.backward import compute_backward
    from yian.runtime.contracts import BackwardOutcome
    from yian.runtime.learning import memory_commit_id
    from yian.runtime.log import (BackwardStartedRecord, BackwardReadyRecord, BackwardFinishedRecord,
        MemoryCommitRequestedRecord, MemoryCommitSucceededRecord)
    from yian.runtime.state import prepare_record

    keys, proposals = _proposal_table()
    engine, _ = _retained_backward(keys=keys)
    run_id = engine.state.run_id
    batches = (("alpha", proposals[0].mutations), ("beta", ()))
    for attempt in (1, 2):
        feedback = f"Review {attempt}."
        root, attributed = compute_backward(engine.state.trace, feedback)
        _accept_backward(engine, BackwardStartedRecord(attempt, feedback))
        _accept_backward(engine, BackwardReadyRecord(attempt, root, attributed, proposals))
        assert engine.state.backward_state.commit_attempt_key == attempt
        for policy, batch in batches:
            if attempt == 2:
                earlier = memory_commit_id(run_id, policy, batch, 1)
                with pytest.raises(ValueError, match="identity"):
                    prepare_record(engine.state, MemoryCommitRequestedRecord(policy, earlier, batch, attempt_id=2))
            commit_id = memory_commit_id(run_id, policy, batch, attempt)
            _accept_backward(engine, MemoryCommitRequestedRecord(policy, commit_id, batch, attempt_id=attempt))
            _accept_backward(engine, MemoryCommitSucceededRecord(policy, commit_id, attempt_id=attempt))
        policies = engine.state.backward_state.policy_outcomes
        _accept_backward(engine, BackwardFinishedRecord(attempt, BackwardOutcome(
            "succeeded", root, learning=LearningResult("succeeded", (), policies),
        )))
    ids = [request.commit_id for attempt in engine.state.backward_state.attempts for request in attempt.requests]
    assert len(set(ids)) == 4
    assert [commit_id.split(":")[0] for commit_id in ids] == ["ac-memory-v3"] * 4


def test_first_committing_attempt_uses_its_own_attempt_id_after_precommit_attempts():
    from yian.runtime.learning import memory_commit_id
    from yian.runtime.log import (BackwardStartedRecord, BackwardReadyRecord, BackwardFinishedRecord,
        MemoryCommitRequestedRecord)

    keys, proposals = _proposal_table()
    engine, (root, attributed) = _retained_backward(keys=keys)
    _accept_backward(engine, BackwardStartedRecord(1, "Review."))
    _accept_backward(engine, BackwardFinishedRecord(1, engine.state.backward_state.outcome))
    _accept_backward(engine, BackwardStartedRecord(2, "Review."))
    _accept_backward(engine, BackwardReadyRecord(2, root, attributed, proposals))
    assert engine.state.backward_state.commit_attempt_key == 2
    batch = proposals[0].mutations
    commit_id = memory_commit_id(engine.state.run_id, "alpha", batch, 2)
    _accept_backward(engine, MemoryCommitRequestedRecord("alpha", commit_id, batch, attempt_id=2))
    assert engine.state.backward_state.current.requests[0].commit_id.startswith("ac-memory-v3:")


@pytest.mark.parametrize("gate", ["pristine", "nonterminal", "learn_false", "missing_terminal"])
def test_backward_started_requires_initialized_learned_terminal_authority(gate):
    from yian.compile.model import START_ID, END_ID
    from yian.runtime.log import BackwardStartedRecord
    from yian.runtime.state import RunState, prepare_record
    from tests.engine._graph_builder import _engine, _graph

    engine = _engine(_graph([], [(START_ID, END_ID)]), learnable=gate != "learn_false")
    if gate in ("learn_false", "missing_terminal"):
        list(engine.run())
    if gate == "missing_terminal":
        engine.state.trace.terminal = None
    state = RunState(_graph([], [(START_ID, END_ID)])) if gate == "pristine" else engine.state
    prior, sequence = state.backward_state, engine.log.next_sequence
    with pytest.raises(ValueError):
        prepare_record(state, BackwardStartedRecord(1, "Review."))
    assert state.backward_state is prior and engine.log.next_sequence == sequence


def test_started_feedback_shape_and_attempt_identity_validate_before_acceptance():
    from yian.runtime.log import BackwardStartedRecord, BackwardReadyRecord, BackwardFinishedRecord
    from yian.runtime.state import prepare_record

    engine, (root, attributed) = _retained_backward(topic=[1, 2], feedback=[None, None])
    for record in (BackwardStartedRecord(1, [None]), BackwardStartedRecord(2, [None, None])):
        with pytest.raises(ValueError):
            prepare_record(engine.state, record)
    _accept_backward(engine, BackwardStartedRecord(1, [None, None]))
    _accept_backward(engine, BackwardReadyRecord(1, root, attributed, ()))
    with pytest.raises(ValueError):
        prepare_record(engine.state, BackwardReadyRecord(1, root, attributed, ()))
    _accept_backward(engine, BackwardFinishedRecord(1, engine.state.backward_state.outcome))
    with pytest.raises(ValueError):
        prepare_record(engine.state, BackwardStartedRecord(1, ""))
    _accept_backward(engine, BackwardStartedRecord(2, ""))
    assert engine.state.backward_state.started.feedback == ""


@pytest.mark.parametrize("change", ["policy", "batch_order", "batch_content", "identity", "attempt"])
def test_backward_commit_request_rejects_wrong_order_batch_identity_and_attempt(change):
    from yian.runtime.learning import memory_commit_id
    from yian.runtime.log import BackwardStartedRecord, BackwardReadyRecord, MemoryCommitRequestedRecord
    from yian.runtime.state import prepare_record

    keys, proposals = _proposal_table()
    engine, (root, attributed) = _retained_backward(keys=keys)
    _accept_backward(engine, BackwardStartedRecord(1, "Review."))
    _accept_backward(engine, BackwardReadyRecord(1, root, attributed, proposals))
    batch = proposals[0].mutations
    policy, attempt = "alpha", 1
    if change == "policy":
        policy, batch = "beta", ()
    elif change == "batch_order":
        batch = tuple(reversed(batch))
    elif change == "batch_content":
        batch = (MemoryMutation("second", {"changed": True}), *batch[1:])
    elif change == "attempt":
        attempt = 2
    commit_id = "commit-1" if change == "identity" else memory_commit_id(engine.state.run_id, policy, batch, 1)
    with pytest.raises(ValueError):
        prepare_record(engine.state, MemoryCommitRequestedRecord(policy, commit_id, batch, attempt_id=attempt))
    assert not engine.state.backward_state.ever_requested


@pytest.mark.parametrize("change", ["no_request", "policy", "identity", "attempt", "duplicate", "pending_request"])
def test_backward_commit_observation_requires_the_current_pending_request(change):
    from yian.runtime.learning import memory_commit_id
    from yian.runtime.log import BackwardStartedRecord, BackwardReadyRecord, MemoryCommitRequestedRecord, MemoryCommitSucceededRecord
    from yian.runtime.state import prepare_record

    keys, proposals = _proposal_table()
    engine, (root, attributed) = _retained_backward(keys=keys)
    _accept_backward(engine, BackwardStartedRecord(1, "Review."))
    _accept_backward(engine, BackwardReadyRecord(1, root, attributed, proposals))
    batch = proposals[0].mutations
    commit_id = memory_commit_id(engine.state.run_id, "alpha", batch, 1)
    request = MemoryCommitRequestedRecord("alpha", commit_id, batch, attempt_id=1)
    if change != "no_request":
        _accept_backward(engine, request)
    if change == "duplicate":
        _accept_backward(engine, MemoryCommitSucceededRecord("alpha", commit_id, attempt_id=1))
    record = request if change == "pending_request" else MemoryCommitSucceededRecord(
        "beta" if change == "policy" else "alpha", "other" if change == "identity" else commit_id,
        attempt_id=2 if change == "attempt" else 1,
    )
    with pytest.raises(ValueError):
        prepare_record(engine.state, record)


@pytest.mark.parametrize("phase", ["started", "ready", "request", "finished"])
def test_backward_append_failure_does_not_install_attempt_or_irreversible_request(phase, monkeypatch):
    from yian.runtime.contracts import BackwardOutcome
    from yian.runtime.learning import memory_commit_id
    from yian.runtime.log import BackwardStartedRecord, BackwardReadyRecord, BackwardFinishedRecord, MemoryCommitRequestedRecord
    from yian.runtime.state import prepare_record, commit_prepared

    keys, proposals = _proposal_table() if phase == "request" else (None, ())
    engine, (root, attributed) = _retained_backward(keys=keys)
    if phase != "started":
        _accept_backward(engine, BackwardStartedRecord(1, "Review."))
    if phase in ("request", "finished"):
        _accept_backward(engine, BackwardReadyRecord(1, root, attributed, proposals))
    if phase == "started":
        record = BackwardStartedRecord(1, "Review.")
    elif phase == "ready":
        record = BackwardReadyRecord(1, root, attributed, ())
    elif phase == "request":
        batch = proposals[0].mutations
        record = MemoryCommitRequestedRecord("alpha", memory_commit_id(engine.state.run_id, "alpha", batch, 1), batch, attempt_id=1)
    else:
        record = BackwardFinishedRecord(1, BackwardOutcome("succeeded", root))
    prepared = prepare_record(engine.state, record)
    prior, sequence, result = engine.state.backward_state, engine.state.next_sequence, engine.state.result
    monkeypatch.setattr(engine.log, "append", lambda _: (_ for _ in ()).throw(OSError("append failed")))
    with pytest.raises(OSError, match="append failed"):
        commit_prepared(engine.log, engine.state, prepared)
    assert engine.state.backward_state is prior and not prior.ever_requested
    assert engine.state.result is result
    assert engine.state.next_sequence == engine.state.trace.next_sequence == engine.log.next_sequence == sequence


@pytest.mark.parametrize("change", ["stale", "next_state", "trace", "missing_trace", "payload", "origin"])
def test_backward_prepared_plans_reject_stale_forged_or_mutated_authority_before_append(change):
    from dataclasses import replace
    from yian.runtime.log import BackwardStartedRecord
    from yian.runtime.state import prepare_record, commit_prepared

    engine, _ = _retained_backward()
    prepared = prepare_record(engine.state, BackwardStartedRecord(1, "Review."))
    if change == "stale":
        _accept_backward(engine, BackwardStartedRecord(1, "Review."))
    elif change == "next_state":
        prepared = replace(prepared, _plan=replace(prepared._plan, next_backward_state=engine.state.backward_state))
    elif change == "trace":
        prepared = replace(prepared, _plan=replace(prepared._plan, trace_delta=replace(prepared._plan.trace_delta, resumed=True)))
    elif change == "missing_trace":
        prepared = replace(prepared, _plan=replace(prepared._plan, trace_delta=None))
    elif change == "payload":
        object.__setattr__(prepared.record, "_feedback", "changed")
    else:
        other, _ = _retained_backward()
        prepared = replace(prepared, _plan=replace(prepared._plan, originating_state=other.state))
    prior, sequence = engine.state.backward_state, engine.log.next_sequence
    with pytest.raises((TypeError, ValueError)):
        commit_prepared(engine.log, engine.state, prepared)
    assert engine.state.backward_state is prior
    assert engine.log.next_sequence == sequence


@pytest.mark.parametrize("field", ["sequence", "resumed", "prune_ids", "connections", "boundaries",
    "initialized", "started", "occurrence", "writer", "consumed_connection", "boundary_update", "root_start_source",
    "root_writer", "causal_route_source", "terminal", "handler"])
def test_postterminal_trace_delta_rejects_equal_but_noncanonical_fields_before_append(field):
    from dataclasses import replace
    from yian.runtime.log import BackwardStartedRecord
    from yian.runtime.state import prepare_record, commit_prepared

    engine, _ = _retained_backward()
    state, log = engine.state, engine.log
    prepared = prepare_record(state, BackwardStartedRecord(1, "Review."))

    class EmptyPrune(tuple):
        def __iter__(self):
            return iter((state.flow.start_id,))

    class AbsentAlias:
        def __eq__(self, other):
            return other is None

        def __iter__(self):
            if field == "consumed_connection":
                pytest.fail("iterated a noncanonical connection before validation")
            return iter(())

    if field == "sequence":
        value = float(prepared._plan.trace_delta.sequence)
    elif field == "resumed":
        value = 0
    elif field in ("prune_ids", "connections", "boundaries"):
        value = EmptyPrune()
    elif field == "handler":
        value = lambda *_: pytest.fail("forged trace handler")
    else:
        value = AbsentAlias()
    delta = replace(prepared._plan.trace_delta, **{field: value})
    prepared = replace(prepared, _plan=replace(prepared._plan, trace_delta=delta))
    trace = state.trace
    writers, sources = dict(trace.current_writers), dict(trace.sources)
    learning, backward, sequence = state.learning_state, state.backward_state, state.next_sequence
    with pytest.raises((TypeError, ValueError)):
        commit_prepared(log, state, prepared)
    assert state.learning_state is learning and state.backward_state is backward
    assert state.trace is trace and type(trace.next_sequence) is int
    assert dict(trace.current_writers) == writers and dict(trace.sources) == sources
    assert state.next_sequence == trace.next_sequence == log.next_sequence == sequence


def test_postterminal_trace_delta_validation_covers_the_closed_field_inventory():
    from dataclasses import fields
    from yian.runtime.trace import TraceDelta

    assert {field.name for field in fields(TraceDelta)} == {
        "originating_trace", "sequence", "initialized", "started", "resumed", "occurrence", "handler",
        "writer", "prune_ids", "connections", "consumed_connection", "boundaries", "boundary_update",
        "root_start_source", "root_writer", "causal_route_source", "terminal",
    }


def test_failed_backward_computation_has_no_completed_feedback_and_keeps_unrequested_learning():
    from yian.runtime.contracts import BackwardFailure, BackwardOutcome
    from yian.runtime.log import BackwardStartedRecord, BackwardFinishedRecord

    keys, _ = _proposal_table()
    engine, _ = _retained_backward(keys=keys)
    _accept_backward(engine, BackwardStartedRecord(1, "Review."))
    occurrence = next(item for item in engine.state.trace.occurrences if item.node_id == "work")
    context = engine.state.trace.backward_context(occurrence.source)
    failure = BackwardFailure("backward", "backward.execute", "Callback failed.", context.scope,
                              context.run_id, context.node_id, context.node_started_seq,
                              cause_type="RuntimeError", traceback="debug trace")
    outcome = BackwardOutcome("failed", failure=failure, learning=engine.state.backward_state.outcome.learning)
    _accept_backward(engine, BackwardFinishedRecord(1, outcome))
    assert engine.state.backward_state.outcome == outcome
    assert engine.state.backward_state.finished.attributed_results == ()
    assert outcome.feedback is None and outcome.learning.status == "incomplete"
    assert tuple(p.status for p in outcome.learning.policies) == ("unrequested", "unrequested")
    assert engine.state.backward_state.can_start


@pytest.mark.parametrize("change", ["source", "node", "scope", "run", "feedback", "partial_inventory"])
def test_failed_computation_finished_rejects_foreign_context_or_fabricated_results(change):
    from yian.nodes.contracts import ExecutionScope
    from yian.runtime.contracts import BackwardFailure, BackwardOutcome
    from yian.runtime.log import BackwardStartedRecord, BackwardFinishedRecord
    from yian.runtime.state import prepare_record

    engine, (root, attributed) = _retained_backward()
    _accept_backward(engine, BackwardStartedRecord(1, "Review."))
    occurrence = next(item for item in engine.state.trace.occurrences if item.node_id == "work")
    context = engine.state.trace.backward_context(occurrence.source)
    failure = BackwardFailure(
        "backward", "backward.execute", "Callback failed.",
        ExecutionScope(("foreign",)) if change == "scope" else context.scope,
        "foreign" if change == "run" else context.run_id,
        "foreign" if change == "node" else context.node_id,
        9999 if change == "source" else context.node_started_seq,
    )
    record = BackwardFinishedRecord(1, BackwardOutcome("failed", root if change == "feedback" else None, failure),
                                    attributed[:1] if change == "partial_inventory" else ())
    with pytest.raises(ValueError):
        prepare_record(engine.state, record)


@pytest.mark.parametrize("feedback,code", [(None, "backward.execute"),
    ([None, None], "backward.result"), ("Review.", "backward.unknown")])
def test_computation_failure_requires_effective_feedback_and_a_driver_failure_code(feedback, code):
    from yian.runtime.contracts import BackwardFailure, BackwardOutcome
    from yian.runtime.log import BackwardStartedRecord, BackwardFinishedRecord
    from yian.runtime.state import prepare_record

    engine, _ = _retained_backward(topic=[1, 2], feedback=feedback)
    _accept_backward(engine, BackwardStartedRecord(1, feedback))
    source = engine.state.trace.occurrences[-1].source
    context = engine.state.trace.backward_context(source)
    failure = BackwardFailure("backward", code, "Callback failed.", context.scope,
                              context.run_id, context.node_id, context.node_started_seq)
    with pytest.raises(ValueError):
        prepare_record(engine.state, BackwardFinishedRecord(1, BackwardOutcome("failed", failure=failure)))


@pytest.mark.parametrize("phase", ["before_ready", "before_request", "pending", "one_confirmed"])
def test_successful_finished_requires_ready_and_every_enabled_policy_confirmation(phase):
    from yian.memory import LearningResult
    from yian.runtime.contracts import BackwardOutcome
    from yian.runtime.learning import memory_commit_id
    from yian.runtime.log import (BackwardStartedRecord, BackwardReadyRecord, BackwardFinishedRecord,
        MemoryCommitRequestedRecord, MemoryCommitSucceededRecord)
    from yian.runtime.state import prepare_record

    keys, proposals = _proposal_table()
    engine, (root, attributed) = _retained_backward(keys=keys)
    _accept_backward(engine, BackwardStartedRecord(1, "Review."))
    if phase != "before_ready":
        _accept_backward(engine, BackwardReadyRecord(1, root, attributed, proposals))
    if phase in ("pending", "one_confirmed"):
        batch = proposals[0].mutations
        identity = memory_commit_id(engine.state.run_id, "alpha", batch, 1)
        _accept_backward(engine, MemoryCommitRequestedRecord("alpha", identity, batch, attempt_id=1))
        if phase == "one_confirmed":
            _accept_backward(engine, MemoryCommitSucceededRecord("alpha", identity, attempt_id=1))
    with pytest.raises(ValueError):
        prepare_record(engine.state, BackwardFinishedRecord(1, BackwardOutcome("succeeded", root,
                                                                             learning=LearningResult("succeeded"))))


@pytest.mark.parametrize("change", ["root", "attribution", "missing_memory", "memory_order", "commit_id", "attempt"])
def test_finished_must_preserve_ready_results_and_exact_confirmed_policy_inventory(change):
    from dataclasses import replace
    from yian.memory import LearningResult
    from yian.nodes.base import BackwardResult
    from yian.runtime.contracts import BackwardOutcome
    from yian.runtime.learning import memory_commit_id
    from yian.runtime.log import (BackwardStartedRecord, BackwardReadyRecord, BackwardFinishedRecord,
        MemoryCommitRequestedRecord, MemoryCommitSucceededRecord)
    from yian.runtime.state import prepare_record

    keys, proposals = _proposal_table()
    engine, (root, attributed) = _retained_backward(keys=keys)
    _accept_backward(engine, BackwardStartedRecord(1, "Review."))
    _accept_backward(engine, BackwardReadyRecord(1, root, attributed, proposals))
    for policy, batch in (("alpha", proposals[0].mutations), ("beta", ())):
        identity = memory_commit_id(engine.state.run_id, policy, batch, 1)
        _accept_backward(engine, MemoryCommitRequestedRecord(policy, identity, batch, attempt_id=1))
        _accept_backward(engine, MemoryCommitSucceededRecord(policy, identity, attempt_id=1))
    policies = engine.state.backward_state.policy_outcomes
    if change == "root":
        root = BackwardResult({"topic": ("Different.",)})
    elif change == "missing_memory":
        policies = policies[:1]
    elif change == "memory_order":
        policies = tuple(reversed(policies))
    elif change == "commit_id":
        policies = (replace(policies[0], commit_id="other"), *policies[1:])
    record = BackwardFinishedRecord(2 if change == "attempt" else 1,
        BackwardOutcome("succeeded", root, learning=LearningResult("succeeded", (), policies)),
        tuple(reversed(attributed)) if change == "attribution" else ())
    with pytest.raises(ValueError):
        prepare_record(engine.state, record)


def test_explicit_interruption_keeps_unknown_request_and_ready_attribution_without_resuming():
    from yian.runtime.learning import memory_commit_id
    from yian.runtime.log import (BackwardStartedRecord, BackwardReadyRecord, BackwardFinishedRecord,
        MemoryCommitRequestedRecord)

    keys, proposals = _proposal_table()
    engine, (root, attributed) = _retained_backward(keys=keys)
    _accept_backward(engine, BackwardStartedRecord(1, "Review."))
    _accept_backward(engine, BackwardReadyRecord(1, root, attributed, proposals))
    batch = proposals[0].mutations
    identity = memory_commit_id(engine.state.run_id, "alpha", batch, 1)
    _accept_backward(engine, MemoryCommitRequestedRecord("alpha", identity, batch, attempt_id=1))
    projected = engine.state.backward_state.outcome
    _accept_backward(engine, BackwardFinishedRecord(1, projected, attributed))
    assert engine.state.backward_state.outcome == projected
    assert projected.feedback == root and projected.learning.status == "incomplete"
    assert tuple(item.status for item in projected.learning.policies) == ("unknown", "unrequested")
    _accept_backward(engine, BackwardStartedRecord(2, "Review again."))
    assert engine.state.backward_state.attempts[0].finished.outcome == projected
    assert engine.state.backward_state.ready is None


@pytest.mark.parametrize("change", ["message", "feedback", "learning", "attribution"])
def test_interrupted_finished_must_exactly_project_the_prefix(change):
    from dataclasses import replace
    from yian.memory import LearningResult
    from yian.nodes.base import BackwardResult
    from yian.runtime.log import BackwardStartedRecord, BackwardReadyRecord, BackwardFinishedRecord
    from yian.runtime.state import prepare_record

    keys, proposals = _proposal_table()
    engine, (root, attributed) = _retained_backward(keys=keys)
    _accept_backward(engine, BackwardStartedRecord(1, "Review."))
    _accept_backward(engine, BackwardReadyRecord(1, root, attributed, proposals))
    outcome = engine.state.backward_state.outcome
    if change == "message":
        outcome = replace(outcome, failure=replace(outcome.failure, message="different"))
    elif change == "feedback":
        outcome = replace(outcome, feedback=BackwardResult())
    elif change == "learning":
        outcome = replace(outcome, learning=LearningResult("incomplete"))
    with pytest.raises(ValueError):
        prepare_record(engine.state, BackwardFinishedRecord(1, outcome, attributed[:1] if change == "attribution" else ()))


def test_disabled_memory_ready_and_finished_reject_fabricated_memory_material():
    from yian.memory import LearningResult
    from yian.runtime.contracts import BackwardOutcome, MemoryProposal
    from yian.runtime.log import BackwardStartedRecord, BackwardReadyRecord, BackwardFinishedRecord
    from yian.runtime.state import prepare_record

    engine, (root, attributed) = _retained_backward()
    _accept_backward(engine, BackwardStartedRecord(1, "Review."))
    with pytest.raises(ValueError):
        prepare_record(engine.state, BackwardReadyRecord(1, root, attributed, (MemoryProposal("fake", ()),)))
    _accept_backward(engine, BackwardReadyRecord(1, root, attributed, ()))
    with pytest.raises(ValueError):
        prepare_record(engine.state, BackwardFinishedRecord(1, BackwardOutcome("succeeded", root,
                                                                             learning=LearningResult("succeeded"))))


@pytest.mark.parametrize("change", ["list_order", "string_subclass"])
def test_backward_state_revalidates_exact_memory_order(change):
    from dataclasses import replace

    class Name(str):
        pass

    keys, _ = _proposal_table()
    engine, _ = _retained_backward(keys=keys)
    changes = ({"memory_order": ["alpha", "beta"]} if change == "list_order" else
               {"memory_order": (Name("alpha"), "beta")})
    with pytest.raises(TypeError, match="backward memory order"):
        replace(engine.state.backward_state, **changes)
    with pytest.raises(ValueError, match="sorted and unique"):
        replace(engine.state.backward_state, memory_order=("beta", "alpha"))


def test_backward_replay_matches_every_accepted_prefix_without_callbacks_or_context_formatting(monkeypatch):
    from copy import deepcopy
    from yian.compile.expand import build_grow
    from yian.memory import LearningResult
    from yian.runtime.contracts import BackwardOutcome
    from yian.runtime.learning import memory_commit_id
    from yian.runtime.log import (BackwardStartedRecord, BackwardReadyRecord, BackwardFinishedRecord,
        MemoryCommitRequestedRecord, MemoryCommitSucceededRecord)
    from yian.runtime.state import RunState, _replay_records

    keys, proposals = _proposal_table()
    engine, (root, attributed) = _retained_backward(keys=keys)
    original = deepcopy(engine.state.flow)
    for node in original.nodes.values():
        node.forward = lambda *_: pytest.fail("replayed forward")
        node.backward = lambda *_: pytest.fail("replayed backward callback")
        node.capture_backward_context = lambda *_: pytest.fail("recaptured backward context")
    import yian.runtime.learning as learning

    monkeypatch.setattr(learning, "_materialize_context", lambda *_: pytest.fail("materialized context"))
    monkeypatch.setattr(learning, "_output_context", lambda *_: pytest.fail("formatted output context"))
    prefixes = [(len(tuple(engine.log)), engine.state.backward_state)]
    records = [BackwardStartedRecord(1, "Review."), BackwardReadyRecord(1, root, attributed, proposals)]
    for policy, batch in (("alpha", proposals[0].mutations), ("beta", ())):
        commit_id = memory_commit_id(engine.state.run_id, policy, batch, 1)
        records.extend((MemoryCommitRequestedRecord(policy, commit_id, batch, attempt_id=1),
                        MemoryCommitSucceededRecord(policy, commit_id, attempt_id=1)))
    for record in records:
        _accept_backward(engine, record)
        prefixes.append((len(tuple(engine.log)), engine.state.backward_state))
    outcome = BackwardOutcome("succeeded", root, learning=LearningResult("succeeded", (), engine.state.backward_state.policy_outcomes))
    _accept_backward(engine, BackwardFinishedRecord(1, outcome))
    prefixes.append((len(tuple(engine.log)), engine.state.backward_state))
    accepted = tuple(engine.log)
    for length, expected in prefixes:
        replay = _replay_records(RunState(deepcopy(original)), accepted[:length], grow_builder=build_grow)
        assert replay.backward_state == expected
        assert replay.backward_state.outcome == expected.outcome
        assert replay.trace.next_sequence == replay.next_sequence == length
        assert replay.result == engine.state.result
