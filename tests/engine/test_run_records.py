"""Contract tests for authoritative run records and their canonical wire codec."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass
from datetime import date, datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterator

import pytest

from agent_composer.events import SourceRange, SourceSpan
from agent_composer.runtime.log import (
    SCHEMA_VERSION,
    NodeExpandedRecord,
    NodeFailedRecord,
    NodeRoutedRecord,
    NodeStartedRecord,
    NodeSucceededRecord,
    RecordEnvelope,
    RunLog,
    RunResumedRecord,
    RunStartedRecord,
    canonical_seed,
    canonical_typed,
    decode_pool,
    decode_record,
    encode_pool,
    encode_record,
)
from agent_composer.typesys import FileRef, ValueKind, VariablePool


_FINGERPRINT = "a" * 64
_TOPOLOGY_DIGEST = "b" * 64


def _records():
    """Return one representative value for every closed run-record variant."""
    pool = VariablePool(start_id="start")
    pool.set("start", {"name": "Ada"})
    return [
        RunStartedRecord(
            flow_fingerprint=_FINGERPRINT,
            initial_pool=encode_pool(pool),
            start_wiring_override={"name": "Ada"},
            effective_llm_by_static_path={
                "agent": {"provider": "openai", "model": "gpt-test"}
            },
            num_workers=2,
        ),
        RunResumedRecord(retry_node_ids=("slow", "later"), num_workers=4),
        NodeStartedRecord(node_id="agent"),
        NodeSucceededRecord(
            node_id="agent",
            commit_target="agent",
            value={"kind": "str", "value": "done"},
        ),
        NodeRoutedRecord(node_id="choice", handle="approved"),
        NodeExpandedRecord(
            spawner_id="each",
            seed=[{"item": 1}, {"item": 2}],
            prune_ids=("each#old/end", "each#old/start"),
            topology_digest=_TOPOLOGY_DIGEST,
        ),
        NodeFailedRecord(
            node_id="broken",
            error="boom",
            error_type="ValueError",
            locator=SourceSpan(node="broken", kind="field", key="code"),
            range=SourceRange(start_line=8, start_col=3, end_line=8, end_col=12),
            traceback="Traceback (most recent call last): ...",
        ),
    ]


@pytest.mark.parametrize("record", _records(), ids=lambda record: type(record).__name__)
def test_every_record_round_trips_through_a_validated_envelope(record):
    envelope = RecordEnvelope(
        schema=SCHEMA_VERSION,
        run_id="run-123",
        seq=0,
        record=record,
    )

    encoded = encode_record(envelope)
    decoded = decode_record(encoded)

    assert decoded == envelope
    assert encode_record(decoded) == encoded
    assert decoded.kind
    assert isinstance(decoded.payload, dict)


def test_record_encoding_is_canonical_newline_free_and_byte_stable():
    envelope = RecordEnvelope(
        schema=SCHEMA_VERSION,
        run_id="run-canonical",
        seq=0,
        record=NodeStartedRecord(node_id="work"),
    )

    encoded = encode_record(envelope)

    assert encoded == (
        b'{"kind":"node_started","payload":{"node_id":"work"},'
        b'"run_id":"run-canonical","schema":1,"seq":0}'
    )
    assert not encoded.endswith(b"\n")
    assert encode_record(decode_record(encoded)) == encoded


def test_decode_maps_legacy_input_declaration_locator_to_generic_input():
    wire = {
        "schema": 1,
        "run_id": "run-legacy-input-locator",
        "seq": 0,
        "kind": "node_failed",
        "payload": {
            "node_id": "__start__",
            "error": "bad input",
            "error_type": "TypeCheckError",
            "locator": {
                "node": "__start__",
                "kind": "input_decl",
                "key": "window",
            },
            "range": None,
            "traceback": None,
        },
    }

    decoded = decode_record(_json_bytes(wire))

    assert decoded.record.locator == SourceSpan(
        node="__start__",
        kind="input",
        key="window",
    )
    assert b'"kind":"input"' in encode_record(decoded)


def test_records_and_envelopes_are_frozen():
    record = NodeStartedRecord(node_id="work")
    envelope = RecordEnvelope(schema=1, run_id="run", seq=0, record=record)

    with pytest.raises(FrozenInstanceError):
        record.node_id = "changed"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        envelope.seq = 1  # type: ignore[misc]


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


@pytest.mark.parametrize(
    "wire",
    [
        {
            "schema": 2,
            "run_id": "run",
            "seq": 0,
            "kind": "node_started",
            "payload": {"node_id": "work"},
        },
        {
            "schema": True,
            "run_id": "run",
            "seq": 0,
            "kind": "node_started",
            "payload": {"node_id": "work"},
        },
        {
            "schema": 1.0,
            "run_id": "run",
            "seq": 0,
            "kind": "node_started",
            "payload": {"node_id": "work"},
        },
        {
            "schema": 1,
            "run_id": "",
            "seq": 0,
            "kind": "node_started",
            "payload": {"node_id": "work"},
        },
        {
            "schema": 1,
            "run_id": "run",
            "seq": -1,
            "kind": "node_started",
            "payload": {"node_id": "work"},
        },
        {
            "schema": 1,
            "run_id": "run",
            "seq": True,
            "kind": "node_started",
            "payload": {"node_id": "work"},
        },
        {
            "schema": 1,
            "run_id": "run",
            "seq": 0,
            "kind": "not_a_record",
            "payload": {},
        },
        {
            "schema": 1,
            "run_id": "run",
            "seq": 0,
            "kind": "node_started",
            "payload": {"node_id": "work"},
            "unexpected": True,
        },
        {
            "schema": 1,
            "run_id": "run",
            "seq": 0,
            "kind": "node_started",
        },
    ],
)
def test_decode_rejects_invalid_envelopes(wire):
    with pytest.raises(ValueError):
        decode_record(_json_bytes(wire))


@pytest.mark.parametrize(
    "wire",
    [
        {
            "schema": 1,
            "run_id": "run",
            "seq": 0,
            "kind": "node_started",
            "payload": {"node_id": "work", "unexpected": True},
        },
        {
            "schema": 1,
            "run_id": "run",
            "seq": 0,
            "kind": "node_started",
            "payload": {},
        },
        {
            "schema": 1,
            "run_id": "run",
            "seq": 0,
            "kind": "node_started",
            "payload": {"node_id": 7},
        },
        {
            "schema": 1,
            "run_id": "run",
            "seq": 0,
            "kind": "node_succeeded",
            "payload": {
                "node_id": "work",
                "commit_target": "work",
                "value": {"kind": "int", "value": "7"},
            },
        },
        {
            "schema": 1,
            "run_id": "run",
            "seq": 0,
            "kind": "node_failed",
            "payload": {
                "node_id": "work",
                "error": "boom",
                "error_type": "ValueError",
                "locator": {
                    "node": "work",
                    "kind": "not_a_locator_kind",
                    "key": "code",
                },
                "range": None,
                "traceback": None,
            },
        },
    ],
)
def test_decode_rejects_unknown_missing_and_wrong_payload_fields(wire):
    with pytest.raises(ValueError):
        decode_record(_json_bytes(wire))


def test_decode_rejects_duplicate_keys_at_any_object_depth():
    duplicate_payload_key = (
        b'{"schema":1,"run_id":"run","seq":0,"kind":"node_started",'
        b'"payload":{"node_id":"first","node_id":"second"}}'
    )

    with pytest.raises(ValueError, match="duplicate JSON object key"):
        decode_record(duplicate_payload_key)


def test_encode_validates_programmatically_constructed_envelopes():
    with pytest.raises(ValueError):
        encode_record(
            RecordEnvelope(
                schema=1,
                run_id="run",
                seq=0,
                record=NodeStartedRecord(node_id=3),  # type: ignore[arg-type]
            )
        )


@pytest.mark.parametrize(
    "record",
    [
        RunResumedRecord(retry_node_ids="ab", num_workers=1),  # type: ignore[arg-type]
        RunResumedRecord(retry_node_ids=["a", "b"], num_workers=1),  # type: ignore[arg-type]
        RunResumedRecord(retry_node_ids=("a", 2), num_workers=1),  # type: ignore[arg-type]
        RunResumedRecord(retry_node_ids=("",), num_workers=1),
        RunResumedRecord(retry_node_ids=("a", "a"), num_workers=1),
        NodeExpandedRecord(
            spawner_id="grow",
            seed={},
            prune_ids="ab",  # type: ignore[arg-type]
            topology_digest=_TOPOLOGY_DIGEST,
        ),
        NodeExpandedRecord(
            spawner_id="grow",
            seed={},
            prune_ids=["a", "b"],  # type: ignore[arg-type]
            topology_digest=_TOPOLOGY_DIGEST,
        ),
        NodeExpandedRecord(
            spawner_id="grow",
            seed={},
            prune_ids=("a", 2),  # type: ignore[arg-type]
            topology_digest=_TOPOLOGY_DIGEST,
        ),
        NodeExpandedRecord(
            spawner_id="grow",
            seed={},
            prune_ids=("",),
            topology_digest=_TOPOLOGY_DIGEST,
        ),
        NodeExpandedRecord(
            spawner_id="grow",
            seed={},
            prune_ids=("a", "a"),
            topology_digest=_TOPOLOGY_DIGEST,
        ),
        NodeExpandedRecord(
            spawner_id="grow",
            seed={},
            prune_ids=("b", "a"),
            topology_digest=_TOPOLOGY_DIGEST,
        ),
    ],
)
def test_encode_rejects_invalid_id_tuple_carriers(record):
    with pytest.raises(ValueError):
        encode_record(RecordEnvelope(1, "run", 0, record))


def test_canonical_json_rejects_lone_unicode_surrogates_as_value_errors():
    surrogate = "\ud800"

    with pytest.raises(ValueError, match="Unicode surrogate"):
        canonical_seed({"nested": [surrogate]})
    with pytest.raises(ValueError, match="Unicode surrogate"):
        canonical_seed({surrogate: "nested key"})
    with pytest.raises(ValueError, match="Unicode surrogate"):
        encode_record(
            RecordEnvelope(1, surrogate, 0, NodeStartedRecord(node_id="work"))
        )
    with pytest.raises(ValueError, match="Unicode surrogate"):
        decode_record(
            '{"schema":1,"run_id":"run","seq":0,"kind":"node_started",'
            '"payload":{"node_id":"\\ud800"}}'
        )


def test_canonical_data_rejects_excessive_nesting_as_a_value_error():
    value: Any = "leaf"
    for _ in range(1_200):
        value = [value]

    with pytest.raises(ValueError, match="nesting exceeds"):
        canonical_seed(value)
    with pytest.raises(ValueError, match="nesting exceeds"):
        canonical_typed(value, None)


def test_typed_output_canonicalizes_tuple_date_and_file_ref():
    value = {
        "items": (1, 2),
        "day": date(2026, 8, 13),
        "at": datetime(2026, 8, 13, 9, 30, tzinfo=timezone.utc),
        "file": FileRef(uri="s3://bucket/report", mime="text/plain", name="report.txt"),
    }

    wire, decoded = canonical_typed(value, ValueKind.OBJECT)

    assert wire == {
        "kind": "object",
        "value": {
            "at": "2026-08-13T09:30:00Z",
            "day": "2026-08-13",
            "file": {
                "mime": "text/plain",
                "name": "report.txt",
                "uri": "s3://bucket/report",
            },
            "items": [1, 2],
        },
    }
    assert decoded.to_object() == wire["value"]
    assert canonical_typed(decoded, ValueKind.OBJECT)[0] == wire


def test_seed_canonicalizes_tuple_date_and_file_ref():
    seed = {
        "carried": ({"day": date(2026, 8, 13)}, 3),
        "file": FileRef(uri="memory://answer"),
    }

    wire, decoded = canonical_seed(seed)

    assert wire == {
        "carried": [{"day": "2026-08-13"}, 3],
        "file": {"mime": None, "name": None, "uri": "memory://answer"},
    }
    assert decoded == wire
    assert canonical_seed(decoded)[0] == wire


def test_canonical_helpers_reject_unordered_containers():
    @dataclass
    class Box:
        values: set[str]

    with pytest.raises(ValueError, match="unordered container"):
        canonical_seed({"values": {"alpha", "beta"}})
    with pytest.raises(ValueError, match="unordered container"):
        canonical_seed(Box({"alpha", "beta"}))
    with pytest.raises(ValueError, match="unordered container"):
        canonical_typed({"values": {"alpha", "beta"}}, ValueKind.OBJECT)


def test_canonical_seed_rejects_keys_before_json_stringification():
    with pytest.raises(ValueError, match="keys must be strings"):
        canonical_seed({True: "would become true"})


def test_canonical_helpers_reject_cyclic_node_data():
    cyclic: list[Any] = []
    cyclic.append(cyclic)

    with pytest.raises(ValueError, match="cyclic container"):
        canonical_seed(cyclic)
    with pytest.raises(ValueError, match="cyclic container"):
        canonical_typed(cyclic, None)


def test_pool_codec_is_canonical_and_lossless_for_typed_values():
    pool = VariablePool(start_id="start")
    pool.set("start", {"items": (1, 2), "day": date(2026, 8, 13)})
    pool.set("file", FileRef(uri="s3://bucket/input", name="input.txt"))
    pool.add_system("run_id", "run-pool")

    wire = encode_pool(pool)
    decoded = decode_pool(wire)

    assert wire["store"]["start"]["value"] == {
        "day": "2026-08-13",
        "items": [1, 2],
    }
    assert decoded.get("start") == wire["store"]["start"]["value"]
    assert decoded.get("file") == FileRef(
        uri="s3://bucket/input", mime=None, name="input.txt"
    )
    assert encode_pool(decoded) == wire


@pytest.mark.parametrize(
    "wire",
    [
        {"store": {}, "system": {}, "start_id": "start", "extra": True},
        {
            "store": {"node": {"kind": "int", "value": "1"}},
            "system": {},
            "start_id": "start",
        },
        {
            "store": {"node": {"kind": "int", "value": 1, "extra": True}},
            "system": {},
            "start_id": "start",
        },
    ],
)
def test_pool_decoder_rejects_unknown_fields_and_wrong_types(wire):
    with pytest.raises(ValueError):
        decode_pool(wire)


class _ProtocolLog:
    """Minimal structural implementation used to pin the `RunLog` protocol."""

    def __init__(self) -> None:
        self.run_id = "run-protocol"
        self.path: Path | None = None
        self.next_sequence = 0
        self._records: list[RecordEnvelope] = []
        self._closed = False
        self.close_count = 0

    def __iter__(self) -> Iterator[RecordEnvelope]:
        return iter(self._records)

    def append(self, record) -> RecordEnvelope:
        envelope = RecordEnvelope(1, self.run_id, self.next_sequence, record)
        self._records.append(envelope)
        self.next_sequence += 1
        return envelope

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self.close_count += 1

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def test_run_log_protocol_pins_iteration_append_close_and_context_management():
    log = _ProtocolLog()

    assert isinstance(log, RunLog)
    with log as entered:
        assert entered is log
        envelope = log.append(NodeStartedRecord(node_id="work"))
        assert envelope.seq == 0
        assert tuple(log) == (envelope,)
        assert log.next_sequence == 1

    log.close()
    assert log.close_count == 1
