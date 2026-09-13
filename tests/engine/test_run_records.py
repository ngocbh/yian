"""Contract tests for authoritative run records and their canonical wire codec."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass
from datetime import date, datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterator, get_args

import pytest

from agent_composer.diagnostics import Failure, OriginRef
from agent_composer.runtime.log import (
    SCHEMA_VERSION,
    NodeExpandedRecord,
    NodeFailedRecord,
    NodeRoutedRecord,
    NodeStartedRecord,
    NodeSucceededRecord,
    RecordEnvelope,
    RunLog,
    RunRecord,
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
    from agent_composer.memory import MemoryFrameRef, MemoryMutation, MemoryOperationFailure, MemoryOwner
    from agent_composer.runtime.log import (
        MemoryCommitFailedRecord, MemoryCommitRequestedRecord, MemoryCommitSucceededRecord,
    )
    from tests.engine.test_backward_records import _backward_records

    pool = VariablePool(start_id="start")
    pool.set("start", {"name": "Ada"})
    frame = MemoryFrameRef("root", ("root", "agent"), MemoryOwner.AGENT)
    mutations = (MemoryMutation("m1", frame, {"text": "Remember."}),)
    failure = MemoryOperationFailure("backend", "backend.failed", "Failed.", True)
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
            failure=Failure(
                code="node.execute",
                message="boom",
                node_id="broken",
                trace=(OriginRef("root/nodes/broken/code"),),
                cause_type="ValueError",
                traceback="Traceback (most recent call last): ...",
            ),
        ),
        MemoryCommitRequestedRecord("history", "commit-1", mutations, attempt_id=1),
        MemoryCommitSucceededRecord("history", "commit-1", attempt_id=1),
        MemoryCommitFailedRecord("history", "commit-1", failure, attempt_id=1),
        *_backward_records(),
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


def test_record_wire_examples_cover_the_closed_record_inventory():
    assert {type(record) for record in _records()} == set(get_args(RunRecord))


@pytest.mark.parametrize("kind,extra", [
    ("memory_frame_learning_started", {"occurrences": []}),
    ("memory_frame_learned", {"mutations": []}),
    ("memory_frame_learning_failed", {"failure": {
        "stage": "learn", "code": "declined", "message": "Declined.", "retryable": True,
    }}),
])
def test_removed_frame_learning_wire_kinds_are_rejected(kind, extra):
    wire = {"schema": SCHEMA_VERSION, "run_id": "run", "seq": 0, "kind": kind,
            "payload": {"policy_name": "history", "frame_ref": {
                "root_flow_id": "root", "static_path": ["root"], "owner": "flow",
            }, **extra}}
    with pytest.raises(ValueError, match="kind"):
        decode_record(json.dumps(wire).encode())


@pytest.mark.parametrize("kind,payload", [
    ("memory_commit_requested", {"mutations": []}),
    ("memory_commit_succeeded", {}),
    ("memory_commit_failed", {"failure": {
        "stage": "commit", "code": "declined", "message": "Declined.", "retryable": True,
    }}),
])
def test_memory_commit_wire_requires_a_positive_attempt(kind, payload):
    wire = {"schema": SCHEMA_VERSION, "run_id": "run", "seq": 0, "kind": kind,
            "payload": {"policy_name": "history", "commit_id": "commit-1", "attempt_id": None, **payload}}
    with pytest.raises(ValueError):
        decode_record(json.dumps(wire).encode())


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
        b'"run_id":"run-canonical","schema":6,"seq":0}'
    )
    assert not encoded.endswith(b"\n")
    assert encode_record(decode_record(encoded)) == encoded


def test_node_failure_encoding_nests_one_canonical_failure():
    envelope = RecordEnvelope(
        schema=SCHEMA_VERSION,
        run_id="run-failure",
        seq=7,
        record=NodeFailedRecord(
            failure=Failure(
                code="contract.input",
                message="bad input",
                node_id="__start__",
                trace=(OriginRef("root/input/window"), OriginRef("root/type/window")),
                cause_type="TypeCheckError",
                traceback=None,
            ),
        ),
    )

    encoded = encode_record(envelope)

    assert json.loads(encoded) == {
        "schema": 6,
        "run_id": "run-failure",
        "seq": 7,
        "kind": "node_failed",
        "payload": {
            "evidence": None, "forward_capture": None,
            "failure": {
                "code": "contract.input",
                "message": "bad input",
                "node_id": "__start__",
                "trace": [
                    {"key": "root/input/window"},
                    {"key": "root/type/window"},
                ],
                "cause_type": "TypeCheckError",
                "traceback": None,
            },
        },
    }
    assert decode_record(encoded) == envelope


def test_route_encoding_persists_only_its_optional_terminal_failure():
    failure = Failure(
        code="graph.terminal_unreachable",
        message="terminal output 'result' skipped",
        node_id="choice",
        trace=(OriginRef("root/nodes/choice"),),
    )
    causal = RecordEnvelope(
        schema=SCHEMA_VERSION,
        run_id="run-route",
        seq=3,
        record=NodeRoutedRecord(
            node_id="choice",
            handle="default",
            terminal_failure=failure,
        ),
    )
    noncausal = RecordEnvelope(
        schema=SCHEMA_VERSION,
        run_id="run-route",
        seq=4,
        record=NodeRoutedRecord(node_id="other", handle="yes"),
    )

    causal_wire = json.loads(encode_record(causal))
    noncausal_wire = json.loads(encode_record(noncausal))

    assert causal_wire["payload"] == {
        "evidence": None, "forward_capture": None,
        "node_id": "choice",
        "handle": "default",
        "terminal_failure": {
            "code": "graph.terminal_unreachable",
            "message": "terminal output 'result' skipped",
            "node_id": "choice",
            "trace": [{"key": "root/nodes/choice"}],
            "cause_type": None,
            "traceback": None,
        },
    }
    assert noncausal_wire["payload"] == {
        "evidence": None, "forward_capture": None,
        "node_id": "other",
        "handle": "yes",
        "terminal_failure": None,
    }
    assert decode_record(encode_record(causal)) == causal
    assert decode_record(encode_record(noncausal)) == noncausal


def test_records_and_envelopes_are_frozen():
    record = NodeStartedRecord(node_id="work")
    envelope = RecordEnvelope(
        schema=SCHEMA_VERSION, run_id="run", seq=0, record=record
    )

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
            "schema": 1,
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
            "schema": 6.0,
            "run_id": "run",
            "seq": 0,
            "kind": "node_started",
            "payload": {"node_id": "work"},
        },
        {
            "schema": 6,
            "run_id": "",
            "seq": 0,
            "kind": "node_started",
            "payload": {"node_id": "work"},
        },
        {
            "schema": 6,
            "run_id": "run",
            "seq": -1,
            "kind": "node_started",
            "payload": {"node_id": "work"},
        },
        {
            "schema": 6,
            "run_id": "run",
            "seq": True,
            "kind": "node_started",
            "payload": {"node_id": "work"},
        },
        {
            "schema": 6,
            "run_id": "run",
            "seq": 0,
            "kind": "not_a_record",
            "payload": {},
        },
        {
            "schema": 6,
            "run_id": "run",
            "seq": 0,
            "kind": "node_started",
            "payload": {"node_id": "work"},
            "unexpected": True,
        },
        {
            "schema": 6,
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
            "schema": 6,
            "run_id": "run",
            "seq": 0,
            "kind": "node_started",
            "payload": {"node_id": "work", "unexpected": True},
        },
        {
            "schema": 6,
            "run_id": "run",
            "seq": 0,
            "kind": "node_started",
            "payload": {},
        },
        {
            "schema": 6,
            "run_id": "run",
            "seq": 0,
            "kind": "node_started",
            "payload": {"node_id": 7},
        },
        {
            "schema": 6,
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
            "schema": 6,
            "run_id": "run",
            "seq": 0,
            "kind": "node_failed",
            "payload": {
                "failure": {
                    "code": "node.execute",
                    "message": "boom",
                    "node_id": "work",
                    "trace": [],
                    "cause_type": "ValueError",
                    "traceback": None,
                },
                "locator": None,
            },
        },
    ],
)
def test_decode_rejects_unknown_missing_and_wrong_payload_fields(wire):
    with pytest.raises(ValueError):
        decode_record(_json_bytes(wire))


def _node_failure_wire() -> dict[str, Any]:
    """Return one independently mutable valid schema-6 node-failure object."""
    return {
        "schema": 6,
        "run_id": "run",
        "seq": 0,
        "kind": "node_failed",
        "payload": {
            "evidence": None, "forward_capture": None,
            "failure": {
                "code": "node.execute",
                "message": "boom",
                "node_id": "work",
                "trace": [{"key": "root/nodes/work/code"}],
                "cause_type": "ValueError",
                "traceback": "trace",
            },
        },
    }


@pytest.mark.parametrize(
    ("case", "mutate"),
    [
        ("missing-failure", lambda payload: payload.pop("failure")),
        ("extra-payload", lambda payload: payload.update(extra=True)),
        (
            "missing-code",
            lambda payload: payload["failure"].pop("code"),
        ),
        (
            "missing-message",
            lambda payload: payload["failure"].pop("message"),
        ),
        (
            "missing-node-id",
            lambda payload: payload["failure"].pop("node_id"),
        ),
        (
            "missing-trace",
            lambda payload: payload["failure"].pop("trace"),
        ),
        (
            "missing-cause-type",
            lambda payload: payload["failure"].pop("cause_type"),
        ),
        (
            "missing-traceback",
            lambda payload: payload["failure"].pop("traceback"),
        ),
        (
            "unknown-code",
            lambda payload: payload["failure"].update(code="node.unknown"),
        ),
        (
            "wrong-message",
            lambda payload: payload["failure"].update(message=7),
        ),
        (
            "null-node-id",
            lambda payload: payload["failure"].update(node_id=None),
        ),
        (
            "empty-node-id",
            lambda payload: payload["failure"].update(node_id=""),
        ),
        (
            "wrong-trace-container",
            lambda payload: payload["failure"].update(trace="origin"),
        ),
        (
            "missing-origin-key",
            lambda payload: payload["failure"].update(trace=[{}]),
        ),
        (
            "empty-origin-key",
            lambda payload: payload["failure"].update(trace=[{"key": ""}]),
        ),
        (
            "wrong-origin-key",
            lambda payload: payload["failure"].update(trace=[{"key": 7}]),
        ),
        (
            "extra-origin-field",
            lambda payload: payload["failure"].update(
                trace=[{"key": "origin", "site": "legacy"}]
            ),
        ),
        (
            "extra-failure-field",
            lambda payload: payload["failure"].update(error="legacy"),
        ),
        (
            "wrong-cause-type",
            lambda payload: payload["failure"].update(cause_type=7),
        ),
        (
            "wrong-traceback",
            lambda payload: payload["failure"].update(traceback=[]),
        ),
    ],
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_decode_rejects_malformed_nested_failures(case, mutate):
    wire = _node_failure_wire()
    mutate(wire["payload"])

    with pytest.raises(ValueError):
        decode_record(_json_bytes(wire))


@pytest.mark.parametrize("field", ["locator", "range"])
def test_schema_6_node_failure_rejects_legacy_location_fields(field):
    """The schema-6 failure payload has one exact authority: `failure`."""
    wire = _node_failure_wire()
    wire["payload"][field] = None

    with pytest.raises(ValueError):
        decode_record(_json_bytes(wire))


@pytest.mark.parametrize("schema", [1, 2, 3, 4, 5])
def test_decode_rejects_old_schemas_even_with_a_nested_failure(schema):
    wire = _node_failure_wire()
    wire["schema"] = schema

    with pytest.raises(ValueError):
        decode_record(_json_bytes(wire))


def test_decode_rejects_flattened_failure_aliases():
    wire = _node_failure_wire()
    wire["payload"] = {
        "node_id": "work",
        "error": "boom",
        "error_type": "ValueError",
        "locator": None,
        "range": None,
        "traceback": "trace",
    }

    with pytest.raises(ValueError):
        decode_record(_json_bytes(wire))


def test_decode_requires_explicit_route_terminal_failure_field():
    wire = {
        "schema": 6,
        "run_id": "run",
        "seq": 0,
        "kind": "node_routed",
        "payload": {"node_id": "choice", "handle": "default", "evidence": None},
    }

    with pytest.raises(ValueError):
        decode_record(_json_bytes(wire))


def test_decode_rejects_extra_nested_route_terminal_failure_field():
    wire = {
        "schema": 6,
        "run_id": "run",
        "seq": 0,
        "kind": "node_routed",
        "payload": {
            "evidence": None, "forward_capture": None,
            "node_id": "choice",
            "handle": "default",
            "terminal_failure": {
                "code": "graph.terminal_unreachable",
                "message": "terminal output 'result' skipped",
                "node_id": "choice",
                "trace": [],
                "cause_type": None,
                "traceback": None,
                "extra": True,
            },
        },
    }

    with pytest.raises(ValueError):
        decode_record(_json_bytes(wire))


def test_decode_rejects_duplicate_keys_at_any_object_depth():
    duplicate_payload_key = (
        b'{"schema":6,"run_id":"run","seq":0,"kind":"node_started",'
        b'"payload":{"node_id":"first","node_id":"second"}}'
    )

    with pytest.raises(ValueError, match="duplicate JSON object key"):
        decode_record(duplicate_payload_key)

    duplicate_origin_key = (
        b'{"schema":6,"run_id":"run","seq":0,"kind":"node_failed",'
        b'"payload":{"failure":{"code":"node.execute","message":"boom",'
        b'"node_id":"work","trace":[{"key":"one","key":"two"}],'
        b'"cause_type":null,"traceback":null}}}'
    )

    with pytest.raises(ValueError, match="duplicate JSON object key"):
        decode_record(duplicate_origin_key)


def test_encode_validates_programmatically_constructed_envelopes():
    with pytest.raises(ValueError):
        encode_record(
            RecordEnvelope(
                schema=SCHEMA_VERSION,
                run_id="run",
                seq=0,
                record=NodeStartedRecord(node_id=3),  # type: ignore[arg-type]
            )
        )

    with pytest.raises(TypeError, match="failure must be a Failure"):
        encode_record(
            RecordEnvelope(
                schema=SCHEMA_VERSION,
                run_id="run",
                seq=0,
                record=NodeFailedRecord(failure=object()),  # type: ignore[arg-type]
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
        encode_record(RecordEnvelope(SCHEMA_VERSION, "run", 0, record))


def test_canonical_json_rejects_lone_unicode_surrogates_as_value_errors():
    surrogate = "\ud800"

    with pytest.raises(ValueError, match="Unicode surrogate"):
        canonical_seed({"nested": [surrogate]})
    with pytest.raises(ValueError, match="Unicode surrogate"):
        canonical_seed({surrogate: "nested key"})
    with pytest.raises(ValueError, match="Unicode surrogate"):
        encode_record(
            RecordEnvelope(
                SCHEMA_VERSION, surrogate, 0, NodeStartedRecord(node_id="work")
            )
        )
    with pytest.raises(ValueError, match="Unicode surrogate"):
        decode_record(
            '{"schema":6,"run_id":"run","seq":0,"kind":"node_started",'
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


def test_decode_record_rejects_excessive_nesting_as_a_value_error():
    nested = '{"nested":' * 1_500 + "null" + "}" * 1_500
    wire = (
        '{"schema":6,"run_id":"run","seq":0,"kind":"node_started",'
        f'"payload":{{"node_id":"work","extra":{nested}}}}}'
    )

    with pytest.raises(ValueError, match="nesting exceeds"):
        decode_record(wire)


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


def test_strict_seed_keeps_supported_values_and_independent_canonical_copies():
    value = {"files": (FileRef(uri="memory://answer"),), "count": 4, "ratio": 0.5}
    wire, decoded = canonical_seed(value, reject_nonfinite=True)
    assert wire == canonical_seed(value)[0] == decoded
    assert type(decoded["count"]) is int
    decoded["files"][0]["uri"] = "changed"
    assert wire["files"][0]["uri"] == "memory://answer"
    assert value["files"][0].uri == "memory://answer"
    with pytest.raises(TypeError, match="reject_nonfinite"):
        canonical_seed(value, reject_nonfinite=1)


@pytest.mark.parametrize("number", [float("inf"), float("-inf"), float("nan")])
@pytest.mark.parametrize("container", ["raw", "dict", "tuple", "model", "dataclass", "typed"])
def test_strict_seed_rejects_nonfinite_before_json_mode_normalization(number, container):
    from pydantic import BaseModel
    from agent_composer.typesys.values import ObjectValue

    class Model(BaseModel):
        value: float

    @dataclass
    class Box:
        value: float

    value = {
        "raw": number,
        "dict": {"nested": [number]},
        "tuple": (number,),
        "model": Model(value=number),
        "dataclass": Box(number),
        "typed": ObjectValue(value={"nested": number}),
    }[container]
    with pytest.raises(ValueError, match="non-finite"):
        canonical_seed(value, reject_nonfinite=True)
    if container == "raw":
        assert canonical_seed(value) == (None, None)


@pytest.mark.parametrize("number", [float("inf"), float("-inf"), float("nan")])
@pytest.mark.parametrize("nested", [False, True])
def test_strict_seed_rejects_nonfinite_pydantic_extras(number, nested):
    from pydantic import BaseModel, ConfigDict, RootModel

    class Extras(BaseModel):
        model_config = ConfigDict(extra="allow")
        declared: int = 1

    value = Extras(lost=number)
    if nested:
        value = RootModel[Extras](Extras(child={"items": (value,)}))
    with pytest.raises(ValueError, match="non-finite"):
        canonical_seed(value, reject_nonfinite=True)
    expected = {"declared": 1, "lost": None}
    if nested:
        expected = {"declared": 1, "child": {"items": [expected]}}
    assert canonical_seed(value) == (expected, expected)


def test_strict_seed_preserves_finite_pydantic_extras_and_file_values():
    from pydantic import BaseModel, ConfigDict, RootModel

    class Extras(BaseModel):
        model_config = ConfigDict(extra="allow")

    value = RootModel[Extras](Extras(child=Extras(score=0.5, file=FileRef(uri="memory://answer"))))
    expected = {"child": {"score": 0.5, "file": {"uri": "memory://answer", "mime": None, "name": None}}}
    wire, decoded = canonical_seed(value, reject_nonfinite=True)
    assert wire == decoded == expected
    decoded["child"]["score"] = 1
    assert wire == expected
    assert value.root.child.score == 0.5


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
        envelope = RecordEnvelope(
            SCHEMA_VERSION, self.run_id, self.next_sequence, record
        )
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
