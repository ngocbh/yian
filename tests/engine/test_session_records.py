"""Strict conversation checkpoint records and their versioned run-log boundary."""

from dataclasses import FrozenInstanceError, replace
from copy import deepcopy
import json

import pytest

from agent_composer.runtime import log as run_log
from agent_composer.memory import contracts as c
from agent_composer.memory.conversation import advance_conversation
from agent_composer.runtime.log import (
    ConversationAdmittedRecord,
    ConversationClosedRecord,
    ConversationEffectObservedRecord,
    ConversationEffectRequestedRecord,
)
from tests.engine._session_fakes import admission, item, request

_RECORD_TYPES = (
    ConversationAdmittedRecord,
    ConversationEffectRequestedRecord,
    ConversationEffectObservedRecord,
    ConversationClosedRecord,
)
_KINDS = (
    "conversation_admitted",
    "conversation_effect_requested",
    "conversation_effect_observed",
    "conversation_closed",
)


def _conversation_checkpoints():
    """Build four legacy checkpoint arms through the strict lower-level fold."""
    first = admission(request().invocation)
    view = advance_conversation(first.baseline, first)
    requested = c.ConversationEffectRequested(
        replace(first.stamp, expected_revision=view.revision, ordinal=1),
        "model",
        {"prompt": "hello"},
        (),
    )
    view = advance_conversation(view, requested)
    observed = c.ConversationEffectObserved(
        replace(first.stamp, expected_revision=view.revision, ordinal=2),
        1,
        "completed",
        (item("reply", "hello"),),
        None,
        None,
    )
    view = advance_conversation(view, observed)
    closed = c.ConversationClosed(
        replace(first.stamp, expected_revision=view.revision, ordinal=3), "completed"
    )
    advance_conversation(view, closed)
    return first, requested, observed, closed


@pytest.fixture
def checkpoints():
    return _conversation_checkpoints()


@pytest.mark.parametrize("index", range(4), ids=_KINDS)
def test_conversation_record_round_trip(checkpoints, index):
    record_type = _RECORD_TYPES[index]
    record = record_type(checkpoints[index])
    envelope = run_log.RecordEnvelope(8, "a", index + 2, record)
    encoded = run_log.encode_record(envelope)

    assert envelope.kind == _KINDS[index]
    assert run_log.decode_record(encoded) == envelope
    assert run_log.encode_record(run_log.decode_record(encoded)) == encoded
    assert record.checkpoint == checkpoints[index]
    assert record.checkpoint is not checkpoints[index]
    assert record.checkpoint.stamp is not checkpoints[index].stamp
    with pytest.raises(FrozenInstanceError):
        record.checkpoint = checkpoints[index]
    payload = envelope.payload
    payload["checkpoint"]["stamp"]["policy_name"] = "changed"
    assert run_log.encode_record(envelope) == encoded


@pytest.mark.parametrize("storage", ["jsonl", "memory"])
def test_logs_append_and_reopen_detached_conversation_records(
    tmp_path, checkpoints, storage
):
    records = tuple(
        cls(checkpoint) for cls, checkpoint in zip(_RECORD_TYPES, checkpoints)
    )
    path = tmp_path / "conversations.jsonl"
    opened = (
        run_log.JsonlRunLog.create(path, "a")
        if storage == "jsonl"
        else run_log.MemoryRunLog("a")
    )
    with opened:
        accepted = tuple(opened.append(record) for record in records)
        assert tuple(opened) == accepted
        assert all(envelope.schema == 8 for envelope in accepted)
        assert [envelope.seq for envelope in accepted] == list(range(4))
        for record, envelope in zip(records, accepted):
            assert envelope.record is not record
            assert envelope.record.checkpoint is not record.checkpoint
    reopened = (
        run_log.JsonlRunLog.resume(path)
        if storage == "jsonl"
        else run_log.MemoryRunLog._from_records(accepted)
    )
    with reopened:
        assert tuple(reopened) == accepted
        assert reopened.next_sequence == 4


@pytest.mark.parametrize("index", range(4), ids=_KINDS)
def test_conversation_records_require_schema_eight(checkpoints, index):
    record = _RECORD_TYPES[index](checkpoints[index])
    envelope = run_log.RecordEnvelope(8, "a", index + 2, record)
    wire = json.loads(run_log.encode_record(envelope))
    wire["schema"] = 7

    with pytest.raises(ValueError, match="schema"):
        run_log.RecordEnvelope(7, "a", index + 2, record)
    with pytest.raises(ValueError, match="schema"):
        run_log.decode_record(json.dumps(wire))
    object.__setattr__(envelope, "schema", 7)
    with pytest.raises(ValueError, match="schema"):
        run_log.encode_record(envelope)


@pytest.mark.parametrize("index", range(4), ids=_KINDS)
def test_conversation_record_rejects_wrong_checkpoint_arm(checkpoints, index):
    record_type = _RECORD_TYPES[index]
    wrong = checkpoints[(index + 1) % 4]
    with pytest.raises(TypeError, match="checkpoint"):
        record_type(wrong)
    record = record_type(checkpoints[index])
    envelope = run_log.RecordEnvelope(8, "a", index + 2, record)
    object.__setattr__(record, "checkpoint", wrong)
    with pytest.raises(TypeError, match="checkpoint"):
        run_log.encode_record(envelope)
    wire = {
        "schema": 8,
        "run_id": "a",
        "seq": index + 2,
        "kind": _KINDS[index],
        "payload": run_log.RecordEnvelope(
            8, "a", index + 2, _RECORD_TYPES[(index + 1) % 4](wrong)
        ).payload,
    }
    with pytest.raises(TypeError, match="checkpoint"):
        run_log.decode_record(json.dumps(wire))


@pytest.mark.parametrize("index", range(4), ids=_KINDS)
@pytest.mark.parametrize(
    "corruption",
    ["extra", "nested_extra", "wrong_type", "bool_ordinal", "foreign_key", "wrong_arm"],
)
def test_conversation_record_rejects_malformed_wire(checkpoints, index, corruption):
    record = _RECORD_TYPES[index](checkpoints[index])
    wire = json.loads(
        run_log.encode_record(run_log.RecordEnvelope(8, "a", index + 2, record))
    )
    checkpoint = wire["payload"]["checkpoint"]
    if corruption == "extra":
        wire["payload"]["extra"] = None
    elif corruption == "nested_extra":
        checkpoint["stamp"]["extra"] = None
    elif corruption == "wrong_type":
        wire["payload"]["checkpoint"] = []
    elif corruption == "bool_ordinal":
        checkpoint["stamp"]["ordinal"] = True
    elif corruption == "foreign_key":
        checkpoint["stamp"]["key"]["consumer_static_path"] = ["foreign", "agent"]
    else:
        checkpoint["type"] = "closed" if index != 3 else "admitted"
    with pytest.raises((TypeError, ValueError)):
        run_log.decode_record(json.dumps(wire))


def test_conversation_record_rejects_duplicate_fields_and_nonfinite(checkpoints):
    record = run_log.ConversationEffectRequestedRecord(checkpoints[1])
    encoded = run_log.encode_record(run_log.RecordEnvelope(8, "a", 3, record))
    with pytest.raises(ValueError, match="duplicate"):
        run_log.decode_record(
            encoded.replace(b'"ordinal":1', b'"ordinal":1,"ordinal":1')
        )
    wire = json.loads(encoded)
    wire["payload"]["checkpoint"]["payload"]["invalid"] = float("inf")
    with pytest.raises(ValueError):
        run_log.decode_record(json.dumps(wire))


@pytest.mark.parametrize("index", range(4), ids=_KINDS)
def test_conversation_record_revalidates_forged_nested_carriers(checkpoints, index):
    record_type = _RECORD_TYPES[index]
    checkpoint = checkpoints[index]
    object.__setattr__(checkpoint.stamp, "ordinal", True)
    with pytest.raises((TypeError, ValueError)):
        record_type(checkpoint)
    forged = object.__new__(record_type)
    object.__setattr__(forged, "checkpoint", checkpoint)
    envelope = run_log.RecordEnvelope(8, "a", index + 2, forged)
    with pytest.raises((TypeError, ValueError)):
        run_log.encode_record(envelope)


@pytest.mark.parametrize("checkpoint", [None, {}, (), object()])
def test_conversation_record_requires_exact_checkpoint_carrier(checkpoint):
    with pytest.raises(TypeError, match="checkpoint"):
        run_log.ConversationAdmittedRecord(checkpoint)


@pytest.mark.parametrize(
    "value", [object(), (), {"unordered"}, float("nan"), float("inf")]
)
def test_conversation_request_record_rejects_forged_non_json_payload(
    checkpoints, value
):
    checkpoint = deepcopy(checkpoints[1])
    object.__setattr__(checkpoint._payload, "entries", (("invalid", value),))
    with pytest.raises((TypeError, ValueError)):
        ConversationEffectRequestedRecord(checkpoint)
    record = object.__new__(ConversationEffectRequestedRecord)
    object.__setattr__(record, "checkpoint", checkpoint)
    envelope = run_log.RecordEnvelope(8, "a", 3, record)
    with pytest.raises((TypeError, ValueError)):
        run_log.encode_record(envelope)


@pytest.mark.parametrize("index", range(4), ids=_KINDS)
def test_conversation_record_rejects_checkpoint_subclasses(checkpoints, index):
    checkpoint = checkpoints[index]

    class OtherCheckpoint(type(checkpoint)):
        pass

    other = object.__new__(OtherCheckpoint)
    for name in type(checkpoint).__slots__:
        object.__setattr__(other, name, getattr(checkpoint, name))
    with pytest.raises(TypeError, match="checkpoint"):
        _RECORD_TYPES[index](other)


@pytest.mark.parametrize("index", range(4), ids=_KINDS)
def test_conversation_record_rejects_noncanonical_path_container(checkpoints, index):
    checkpoint = checkpoints[index]
    object.__setattr__(checkpoint.stamp.key, "consumer_static_path", ["root", "agent"])
    with pytest.raises((TypeError, ValueError)):
        _RECORD_TYPES[index](checkpoint)
    record = object.__new__(_RECORD_TYPES[index])
    object.__setattr__(record, "checkpoint", checkpoint)
    with pytest.raises((TypeError, ValueError)):
        run_log.encode_record(run_log.RecordEnvelope(8, "a", index + 2, record))
