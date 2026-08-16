"""Closed run-record vocabulary and canonical JSON codecs.

The dispatcher is the sole producer of these durable records.  Records contain
only transition data; their enclosing :class:`RecordEnvelope` owns run identity,
schema version, and sequence.  This module deliberately knows nothing about the
scheduler or reducer so the same validated carriers can back memory and JSONL
logs without creating an authority cycle.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, is_dataclass
import fcntl
import json
import math
import os
from pathlib import Path
from types import TracebackType
from typing import (
    Any,
    BinaryIO,
    Final,
    Iterator,
    Literal,
    Protocol,
    TypeAlias,
    runtime_checkable,
)

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator

from agent_composer.diagnostics import Failure, FailureCode, OriginRef
from agent_composer.events import SourceRange, SourceSpan
from agent_composer.typesys.pool import VariablePool
from agent_composer.typesys.values import (
    ANY_VALUE_ADAPTER,
    Type,
    TypedValue,
    ValueKind,
    build_value,
    build_value_as,
)


SCHEMA_VERSION: Final = 2
"""The one run-record wire schema understood by this release."""

_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_ANY_ADAPTER = TypeAdapter(Any)


@dataclass(frozen=True, slots=True)
class RunStartedRecord:
    """Initialize a run from its complete canonical inputs and resolved configuration.

    Attributes:
        flow_fingerprint (`str`):
            Lowercase SHA-256 identity of the authored flow.
        initial_pool (`dict[str, Any]`):
            Canonical wire representation of the full initial variable pool.
        start_wiring_override (`dict[str, Any]`):
            Per-run wiring override for the synthesized START node.
        effective_llm_by_static_path (`dict[str, dict[str, Any]]`):
            Complete resolved agent configuration keyed by stable static node path.
        num_workers (`int`):
            Initial dispatcher worker count; zero selects serial execution.
    """

    flow_fingerprint: str
    initial_pool: dict[str, Any]
    start_wiring_override: dict[str, Any]
    effective_llm_by_static_path: dict[str, dict[str, Any]]
    num_workers: int


@dataclass(frozen=True, slots=True)
class RunResumedRecord:
    """Start a recovery segment with ordered unfinished work and a worker count."""

    retry_node_ids: tuple[str, ...]
    num_workers: int


@dataclass(frozen=True, slots=True)
class NodeStartedRecord:
    """Move the exact ready-head node into the ordered in-flight set."""

    node_id: str


@dataclass(frozen=True, slots=True)
class NodeSucceededRecord:
    """Commit one canonical typed value under an already-resolved target."""

    node_id: str
    commit_target: str
    value: dict[str, Any]


@dataclass(frozen=True, slots=True)
class NodeRoutedRecord:
    """Select one named outgoing route for an executing node."""

    node_id: str
    handle: str


@dataclass(frozen=True, slots=True)
class NodeExpandedRecord:
    """Splice one deterministic expansion from its canonical seed and digest."""

    spawner_id: str
    seed: Any
    prune_ids: tuple[str, ...]
    topology_digest: str


@dataclass(frozen=True, slots=True)
class NodeFailedRecord:
    """Make a run terminal-failed with the captured node diagnostic."""

    failure: Failure
    locator: SourceSpan | None = None
    range: SourceRange | None = None


RunRecord: TypeAlias = (
    RunStartedRecord
    | RunResumedRecord
    | NodeStartedRecord
    | NodeSucceededRecord
    | NodeRoutedRecord
    | NodeExpandedRecord
    | NodeFailedRecord
)
"""The exhaustive durable transition vocabulary for one run."""

_RUN_RECORD_TYPES = (
    RunStartedRecord,
    RunResumedRecord,
    NodeStartedRecord,
    NodeSucceededRecord,
    NodeRoutedRecord,
    NodeExpandedRecord,
    NodeFailedRecord,
)


@dataclass(frozen=True, slots=True)
class RecordEnvelope:
    """Bind one validated run record to its schema, run ID, and zero-based sequence.

    The in-memory envelope retains the typed record.  Its wire codec derives the
    stable snake-case ``kind`` and strict ``payload`` fields, ensuring log readers
    never expose an unvalidated payload as runtime authority.
    """

    schema: int
    run_id: str
    seq: int
    record: RunRecord

    def __post_init__(self) -> None:
        """Reject invalid envelope identity before it can reach a log implementation."""
        if type(self.schema) is not int or self.schema != SCHEMA_VERSION:
            raise ValueError(f"unsupported run-record schema: {self.schema!r}")
        if type(self.run_id) is not str or not self.run_id:
            raise ValueError("run-record envelope run_id must be a non-empty string")
        if type(self.seq) is not int or self.seq < 0:
            raise ValueError("run-record envelope seq must be a non-negative integer")
        if not isinstance(self.record, _RUN_RECORD_TYPES):
            raise TypeError(f"unsupported run record: {type(self.record).__name__}")

    @property
    def kind(self) -> str:
        """Return the stable snake-case wire discriminator for ``record``."""
        return _kind_for_record(self.record)

    @property
    def payload(self) -> dict[str, Any]:
        """Return a validated canonical JSON payload for ``record``."""
        return _payload_for_record(self.record)


@runtime_checkable
class RunLog(Protocol):
    """Ordered append-only owner of validated record envelopes for one run.

    Implementations must make ``close`` idempotent.  An engine borrows a log and
    never closes it; the run boundary or direct caller owns the context manager.
    Because frozen record dataclasses may contain mutable JSON values, appenders
    must retain a canonical encode/decode copy rather than the caller's object.
    """

    @property
    def run_id(self) -> str:
        """The sole non-empty run identity accepted by this log."""
        ...

    @property
    def path(self) -> Path | None:
        """The durable JSONL path, or ``None`` for a process-local log."""
        ...

    @property
    def next_sequence(self) -> int:
        """The zero-based sequence that the next successful append will receive."""
        ...

    def __iter__(self) -> Iterator[RecordEnvelope]:
        """Iterate validated envelopes in contiguous sequence order."""
        ...

    def append(self, record: RunRecord) -> RecordEnvelope:
        """Validate and copy ``record``, append it under ``next_sequence``, and return it."""
        ...

    def close(self) -> None:
        """Release owned resources; repeated calls have no effect."""
        ...

    def __enter__(self) -> "RunLog":
        """Return this open log to a context-manager caller."""
        ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the log when its ownership context exits."""
        ...


class MemoryRunLog:
    """Process-local ordered authority for one run's canonical records.

    The log owns a codec-round-tripped copy of every appended record.  Closing
    prevents later appends but deliberately keeps the cached prefix iterable so
    callers can inspect a completed result after leaving an ownership context.
    """

    def __init__(self, run_id: str) -> None:
        """Create an empty zero-based log for one non-empty run identity."""
        self._run_id = _validated_run_id(run_id)
        self._records: list[RecordEnvelope] = []
        self._next_sequence = 0
        self._closed = False

    @property
    def run_id(self) -> str:
        """Return the sole run identity accepted by this log."""
        return self._run_id

    @property
    def path(self) -> None:
        """Return ``None`` because this log has no durable backing file."""
        return None

    @property
    def next_sequence(self) -> int:
        """Return the sequence assigned by the next successful append."""
        return self._next_sequence

    def __iter__(self) -> Iterator[RecordEnvelope]:
        """Iterate independent copies of a stable retained-prefix snapshot."""
        retained = tuple(self._records)
        return (_copy_envelope(envelope) for envelope in retained)

    def records(self) -> tuple[RecordEnvelope, ...]:
        """Return independent copies in an immutable outer carrier."""
        return tuple(self)

    def append(self, record: RunRecord) -> RecordEnvelope:
        """Validate, canonically copy, and retain one record in sequence order."""
        self._ensure_open()
        retained = _canonical_envelope(record, self._run_id, self._next_sequence)
        self._records.append(retained)
        self._next_sequence += 1
        return _copy_envelope(retained)

    def close(self) -> None:
        """Close the append boundary; repeated calls have no effect."""
        self._closed = True

    def __enter__(self) -> "MemoryRunLog":
        """Return this open log to its ownership context."""
        self._ensure_open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the append boundary on every context exit."""
        self.close()

    def _ensure_open(self) -> None:
        """Reject writes and new ownership contexts after close."""
        if self._closed:
            raise RuntimeError("run log is closed")


class JsonlRunLog:
    """Exclusively locked, append-and-fsync JSONL authority for one run.

    Instances are opened only through :meth:`create` or :meth:`resume`.  The
    process holds a nonblocking exclusive advisory lock for the full instance
    lifetime, while an in-memory validated prefix makes iteration independent of
    the backing file after close.
    """

    def __init__(
        self,
        path: Path,
        run_id: str,
        file: BinaryIO,
        records: list[RecordEnvelope],
        prefix_size: int,
    ) -> None:
        """Adopt an already-locked file and its validated complete prefix."""
        self._path = path
        self._run_id = run_id
        self._file = file
        self._records = records
        self._next_sequence = len(records)
        self._prefix_size = prefix_size
        self._closed = False

    @classmethod
    def create(cls, path: os.PathLike[str] | str, run_id: str) -> "JsonlRunLog":
        """Exclusively create and lock a new owner-only JSONL file.

        The ``O_EXCL`` open never replaces an existing path.  All failure paths
        after opening release the advisory lock and descriptor before propagating
        the host exception.
        """
        canonical_run_id = _validated_run_id(run_id)
        canonical_path = Path(path)
        fd: int | None = None
        file: BinaryIO | None = None
        locked = False
        try:
            fd = os.open(
                canonical_path,
                os.O_RDWR | os.O_CREAT | os.O_EXCL,
                0o600,
            )
            os.fchmod(fd, 0o600)
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            locked = True
            file = os.fdopen(fd, "r+b", buffering=0)
            fd = None  # the binary file now owns this descriptor
            return cls(canonical_path, canonical_run_id, file, [], 0)
        except BaseException:
            if file is not None:
                _release_file(file)
            elif fd is not None:
                _release_fd(fd, locked=locked)
            raise

    @classmethod
    def resume(cls, path: os.PathLike[str] | str) -> "JsonlRunLog":
        """Lock, repair, and validate one existing complete JSONL prefix.

        Lock acquisition happens before any file read.  Bytes after the final
        newline are never records—even when those bytes form valid JSON—and are
        durably truncated before complete lines are decoded.
        """
        canonical_path = Path(path)
        fd: int | None = None
        file: BinaryIO | None = None
        locked = False
        try:
            fd = os.open(canonical_path, os.O_RDWR)
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            locked = True
            file = os.fdopen(fd, "r+b", buffering=0)
            fd = None  # the binary file now owns this descriptor

            data = file.read()
            complete_end = data.rfind(b"\n") + 1
            if complete_end != len(data):
                descriptor = file.fileno()
                os.ftruncate(descriptor, complete_end)
                os.lseek(descriptor, complete_end, os.SEEK_SET)
                file.flush()
                os.fsync(descriptor)
                data = data[:complete_end]
            else:
                os.lseek(file.fileno(), complete_end, os.SEEK_SET)

            records = _decode_complete_prefix(data)
            if not records:
                raise ValueError("cannot resume an empty run log")
            return cls(
                canonical_path,
                records[0].run_id,
                file,
                records,
                complete_end,
            )
        except BaseException:
            if file is not None:
                _release_file(file)
            elif fd is not None:
                _release_fd(fd, locked=locked)
            raise

    @property
    def run_id(self) -> str:
        """Return the sole run identity validated across this file."""
        return self._run_id

    @property
    def path(self) -> Path:
        """Return the durable JSONL path."""
        return self._path

    @property
    def next_sequence(self) -> int:
        """Return the sequence assigned by the next successful append."""
        return self._next_sequence

    def __iter__(self) -> Iterator[RecordEnvelope]:
        """Iterate independent copies of a stable validated-prefix snapshot."""
        retained = tuple(self._records)
        return (_copy_envelope(envelope) for envelope in retained)

    def records(self) -> tuple[RecordEnvelope, ...]:
        """Return independent copies in an immutable outer carrier."""
        return tuple(self)

    def append(self, record: RunRecord) -> RecordEnvelope:
        """Append one complete line and retain it only after its fsync succeeds."""
        self._ensure_open()
        retained = _canonical_envelope(record, self._run_id, self._next_sequence)
        line = encode_record(retained) + b"\n"
        descriptor = self._file.fileno()
        prior_prefix = self._prefix_size
        os.lseek(descriptor, prior_prefix, os.SEEK_SET)

        try:
            written = self._file.write(line)
            if written != len(line):
                raise OSError(
                    f"short run-log write: wrote {written!r} of {len(line)} bytes"
                )
            self._file.flush()
            os.fsync(descriptor)
        except BaseException as exc:
            rollback_errors = _rollback_append(descriptor, prior_prefix)
            if rollback_errors and hasattr(exc, "add_note"):
                exc.add_note(
                    "run-log rollback also failed: "
                    + "; ".join(str(error) for error in rollback_errors)
                )
            raise

        self._records.append(retained)
        self._next_sequence += 1
        self._prefix_size += len(line)
        return _copy_envelope(retained)

    def close(self) -> None:
        """Release the lock and file exactly once while retaining cached records."""
        if self._closed:
            return
        self._closed = True
        _release_file(self._file)

    def __enter__(self) -> "JsonlRunLog":
        """Return this open log to its ownership context."""
        self._ensure_open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Release the file and lock on every context exit."""
        self.close()

    def _ensure_open(self) -> None:
        """Reject writes and new ownership contexts after close."""
        if self._closed:
            raise RuntimeError("run log is closed")


class _StrictModel(BaseModel):
    """Base for exact, non-coercing envelope and payload validation."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class _EnvelopeWire(_StrictModel):
    """Strict top-level JSON object shared by every record kind."""

    schema_version: Literal[SCHEMA_VERSION] = Field(alias="schema")
    run_id: str = Field(min_length=1)
    seq: int = Field(ge=0)
    kind: str = Field(min_length=1)
    payload: dict[str, Any]

    @field_validator("schema_version", mode="before")
    @classmethod
    def _exact_integer_schema(cls, value: Any) -> Any:
        """Reject bools and integral floats before the schema literal can coerce them."""
        if type(value) is not int:
            raise ValueError("run-record schema must be an integer")
        return value


class _RunStartedPayload(_StrictModel):
    """Strict wire payload for :class:`RunStartedRecord`."""

    flow_fingerprint: str = Field(pattern=_SHA256_PATTERN)
    initial_pool: dict[str, Any]
    start_wiring_override: dict[str, Any]
    effective_llm_by_static_path: dict[str, dict[str, Any]]
    num_workers: int = Field(ge=0)


class _RunResumedPayload(_StrictModel):
    """Strict wire payload for :class:`RunResumedRecord`."""

    retry_node_ids: list[str]
    num_workers: int = Field(ge=0)

    @field_validator("retry_node_ids")
    @classmethod
    def _unique_nonempty_ids(cls, value: list[str]) -> list[str]:
        """Require an ordered set of non-empty unfinished node IDs."""
        if any(not node_id for node_id in value):
            raise ValueError("retry node IDs must be non-empty")
        if len(set(value)) != len(value):
            raise ValueError("retry node IDs must be unique")
        return value


class _NodeStartedPayload(_StrictModel):
    """Strict wire payload for :class:`NodeStartedRecord`."""

    node_id: str = Field(min_length=1)


class _NodeSucceededPayload(_StrictModel):
    """Strict wire payload for :class:`NodeSucceededRecord`."""

    node_id: str = Field(min_length=1)
    commit_target: str = Field(min_length=1)
    value: dict[str, Any]


class _NodeRoutedPayload(_StrictModel):
    """Strict wire payload for :class:`NodeRoutedRecord`."""

    node_id: str = Field(min_length=1)
    handle: str = Field(min_length=1)


class _NodeExpandedPayload(_StrictModel):
    """Strict wire payload for :class:`NodeExpandedRecord`."""

    spawner_id: str = Field(min_length=1)
    seed: Any
    prune_ids: list[str]
    topology_digest: str = Field(pattern=_SHA256_PATTERN)

    @field_validator("prune_ids")
    @classmethod
    def _sorted_unique_ids(cls, value: list[str]) -> list[str]:
        """Require the canonical sorted set representation used by expansion records."""
        if any(not node_id for node_id in value):
            raise ValueError("prune IDs must be non-empty")
        if value != sorted(set(value)):
            raise ValueError("prune IDs must be sorted and unique")
        return value


class _SourceSpanPayload(_StrictModel):
    """Strict JSON form of a symbolic source locator."""

    node: str | None
    kind: Literal["input", "assert", "field"]
    key: str


class _SourceRangePayload(_StrictModel):
    """Strict JSON form of a concrete 1-based source range."""

    start_line: int = Field(ge=1)
    start_col: int = Field(ge=1)
    end_line: int = Field(ge=1)
    end_col: int = Field(ge=1)


class _OriginRefPayload(_StrictModel):
    """Strict single-key JSON form of one opaque authored origin."""

    key: str = Field(min_length=1)


class _FailurePayload(_StrictModel):
    """Strict nested JSON form of one canonical semantic failure."""

    code: FailureCode
    message: str
    node_id: str = Field(min_length=1)
    trace: list[_OriginRefPayload]
    cause_type: str | None
    traceback: str | None


class _NodeFailedPayload(_StrictModel):
    """Strict wire payload for :class:`NodeFailedRecord`."""

    failure: _FailurePayload
    locator: _SourceSpanPayload | None
    range: _SourceRangePayload | None


_KIND_TO_MODEL: dict[str, type[_StrictModel]] = {
    "run_started": _RunStartedPayload,
    "run_resumed": _RunResumedPayload,
    "node_started": _NodeStartedPayload,
    "node_succeeded": _NodeSucceededPayload,
    "node_routed": _NodeRoutedPayload,
    "node_expanded": _NodeExpandedPayload,
    "node_failed": _NodeFailedPayload,
}

_TYPE_TO_KIND: dict[type[Any], str] = {
    RunStartedRecord: "run_started",
    RunResumedRecord: "run_resumed",
    NodeStartedRecord: "node_started",
    NodeSucceededRecord: "node_succeeded",
    NodeRoutedRecord: "node_routed",
    NodeExpandedRecord: "node_expanded",
    NodeFailedRecord: "node_failed",
}


def canonical_typed(
    value: Any,
    declared: ValueKind | Type | None,
) -> tuple[dict[str, Any], TypedValue]:
    """Canonicalize a node value through its durable typed JSON representation.

    Args:
        value (`Any`):
            Live node output, or an existing `TypedValue`, to cross the durable boundary.
        declared (`ValueKind | Type | None`):
            Target output type.  ``None`` infers the natural storage type.

    Returns:
        `tuple[dict[str, Any], TypedValue]`:
            The canonical typed wire object and the value decoded from that exact object.

    Raises:
        `TypeCheckError`:
            The live value does not satisfy ``declared`` or has no supported inferred type.
        `ValueError`:
            The typed value contains data that cannot be represented as canonical JSON.
    """
    try:
        _reject_unordered_containers(value)
        typed = build_value(value) if declared is None else build_value_as(declared, value)
        dumped = ANY_VALUE_ADAPTER.dump_python(typed, mode="json", warnings="error")
        wire = _copy_json(dumped)
        if not isinstance(wire, dict):  # defensive: every AnyValue model is an object
            raise ValueError("typed value did not encode to a JSON object")
        decoded = _decode_typed(wire)
        return wire, decoded
    except RecursionError as exc:
        raise ValueError("canonical data nesting exceeds the supported depth") from exc


def canonical_seed(value: Any) -> tuple[Any, Any]:
    """Canonicalize an opaque growth seed through Pydantic's JSON mode.

    Args:
        value (`Any`):
            Node-produced builder input.  Tuples, dates, datetimes, and Pydantic
            models such as `FileRef` are converted to their JSON representations.

    Returns:
        `tuple[Any, Any]`:
            Independent canonical JSON wire and decoded values.  Live expansion
            building must consume the decoded second value.

    Raises:
        `ValueError`:
            The seed contains an unsupported or non-finite JSON value.
    """
    try:
        _reject_unordered_containers(value)
        dumped = _ANY_ADAPTER.dump_python(value, mode="json", warnings="error")
        wire = _copy_json(dumped)
        return wire, _copy_json(wire)
    except RecursionError as exc:
        raise ValueError("canonical data nesting exceeds the supported depth") from exc


def encode_pool(pool: VariablePool) -> dict[str, Any]:
    """Encode a variable pool to its canonical strict JSON object.

    Args:
        pool (`VariablePool`):
            Complete typed store, system namespace, and enclosing START node identity.

    Returns:
        `dict[str, Any]`:
            Canonical JSON wire suitable for a `RunStartedRecord` payload.

    Raises:
        `ValueError`:
            A nested typed value cannot cross the JSON boundary.
    """
    if not isinstance(pool, VariablePool):
        raise TypeError("encode_pool expects a VariablePool")
    _reject_unordered_containers(pool)
    dumped = pool.model_dump(mode="json", warnings="error")
    wire = _copy_json(dumped)
    if not isinstance(wire, dict):  # defensive: VariablePool is a JSON object
        raise ValueError("variable pool did not encode to a JSON object")
    decode_pool(wire)
    return wire


def decode_pool(wire: dict[str, Any]) -> VariablePool:
    """Decode and strictly validate a canonical variable-pool wire object.

    Args:
        wire (`dict[str, Any]`):
            JSON object with exactly ``store``, ``system``, and ``start_id`` fields.

    Returns:
        `VariablePool`:
            A new pool whose typed values were decoded from the supplied wire.

    Raises:
        `ValueError`:
            The object has missing/extra fields, invalid typed values, or non-JSON data.
    """
    copied = _copy_json(wire)
    if not isinstance(copied, dict):
        raise ValueError("variable-pool wire must be a JSON object")
    _require_fields(copied, {"store", "system", "start_id"}, "variable pool")
    if not isinstance(copied["store"], dict) or not isinstance(copied["system"], dict):
        raise ValueError("variable-pool store and system must be JSON objects")
    for namespace in ("store", "system"):
        for key, typed_wire in copied[namespace].items():
            if not isinstance(key, str) or not isinstance(typed_wire, dict):
                raise ValueError(f"variable-pool {namespace} entries must be typed objects")
            _decode_typed(typed_wire)
    return VariablePool.model_validate(copied, strict=True)


def encode_record(envelope: RecordEnvelope) -> bytes:
    """Encode one validated envelope as canonical newline-free UTF-8 JSON.

    Args:
        envelope (`RecordEnvelope`):
            Typed record plus schema, single run identity, and zero-based sequence.

    Returns:
        `bytes`:
            Compact canonical JSON.  A JSONL implementation adds the terminating newline.

    Raises:
        `ValueError`:
            An envelope or record field violates the strict wire contract.
        `TypeError`:
            ``envelope`` or its record has an unsupported runtime type.
    """
    if not isinstance(envelope, RecordEnvelope):
        raise TypeError("encode_record expects a RecordEnvelope")
    raw = _copy_json(
        {
            "schema": envelope.schema,
            "run_id": envelope.run_id,
            "seq": envelope.seq,
            "kind": envelope.kind,
            "payload": envelope.payload,
        }
    )
    wire = _EnvelopeWire.model_validate(raw).model_dump(mode="json", by_alias=True)
    return _json_bytes(wire)


def decode_record(data: bytes | str) -> RecordEnvelope:
    """Decode canonical JSON into a validated typed record envelope.

    Args:
        data (`bytes | str`):
            One complete JSON object without a required trailing newline.

    Returns:
        `RecordEnvelope`:
            Validated schema/run/sequence identity and a closed `RunRecord` variant.

    Raises:
        `ValueError`:
            JSON, envelope identity, record kind, or payload fields are invalid.
        `TypeError`:
            ``data`` is neither bytes nor text.
    """
    if not isinstance(data, (bytes, str)):
        raise TypeError("decode_record expects bytes or str")
    try:
        raw = json.loads(
            data,
            object_pairs_hook=_object_without_duplicates,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("invalid run-record JSON") from exc
    raw = _copy_json(raw)
    envelope = _EnvelopeWire.model_validate(raw)
    model_type = _KIND_TO_MODEL.get(envelope.kind)
    if model_type is None:
        raise ValueError(f"unknown run-record kind: {envelope.kind!r}")
    payload = model_type.model_validate(envelope.payload)
    record = _record_from_payload(envelope.kind, payload)
    return RecordEnvelope(
        schema=envelope.schema_version,
        run_id=envelope.run_id,
        seq=envelope.seq,
        record=record,
    )

def _kind_for_record(record: RunRecord) -> str:
    """Map one closed record variant to its stable wire discriminator."""
    try:
        return _TYPE_TO_KIND[type(record)]
    except KeyError as exc:
        raise TypeError(f"unsupported run record: {type(record).__name__}") from exc


def _payload_for_record(record: RunRecord) -> dict[str, Any]:
    """Build and validate the canonical JSON payload for one record variant."""
    kind = _kind_for_record(record)
    if isinstance(record, RunStartedRecord):
        raw = {
            "flow_fingerprint": record.flow_fingerprint,
            "initial_pool": encode_pool(decode_pool(record.initial_pool)),
            "start_wiring_override": canonical_seed(record.start_wiring_override)[0],
            "effective_llm_by_static_path": canonical_seed(
                record.effective_llm_by_static_path
            )[0],
            "num_workers": record.num_workers,
        }
    elif isinstance(record, RunResumedRecord):
        raw = {
            "retry_node_ids": _validated_id_tuple(
                record.retry_node_ids, "retry_node_ids"
            ),
            "num_workers": record.num_workers,
        }
    elif isinstance(record, NodeStartedRecord):
        raw = asdict(record)
    elif isinstance(record, NodeSucceededRecord):
        decoded = _decode_typed(record.value)
        canonical = ANY_VALUE_ADAPTER.dump_python(decoded, mode="json", warnings="error")
        raw = {
            "node_id": record.node_id,
            "commit_target": record.commit_target,
            "value": _copy_json(canonical),
        }
    elif isinstance(record, NodeRoutedRecord):
        raw = asdict(record)
    elif isinstance(record, NodeExpandedRecord):
        raw = {
            "spawner_id": record.spawner_id,
            "seed": canonical_seed(record.seed)[0],
            "prune_ids": _validated_id_tuple(
                record.prune_ids, "prune_ids", require_sorted=True
            ),
            "topology_digest": record.topology_digest,
        }
    elif isinstance(record, NodeFailedRecord):
        failure = record.failure
        if type(failure) is not Failure:
            raise TypeError("NodeFailedRecord failure must be a Failure")
        raw = {
            "failure": {
                "code": failure.code,
                "message": failure.message,
                "node_id": failure.node_id,
                "trace": [{"key": origin.key} for origin in failure.trace],
                "cause_type": failure.cause_type,
                "traceback": failure.traceback,
            },
            "locator": asdict(record.locator) if record.locator is not None else None,
            "range": asdict(record.range) if record.range is not None else None,
        }
    else:  # protected by _kind_for_record; retained for exhaustive local reasoning
        raise TypeError(f"unsupported run record: {type(record).__name__}")

    payload = _KIND_TO_MODEL[kind].model_validate(raw).model_dump(mode="json")
    copied = _copy_json(payload)
    if not isinstance(copied, dict):  # defensive: every payload model is an object
        raise ValueError("run-record payload did not encode to a JSON object")
    return copied


def _record_from_payload(kind: str, payload: _StrictModel) -> RunRecord:
    """Construct the closed dataclass variant matching a validated payload model."""
    values = payload.model_dump(mode="python")
    if kind == "run_started":
        initial_pool = encode_pool(decode_pool(values["initial_pool"]))
        start_wiring_override = canonical_seed(values["start_wiring_override"])[0]
        effective = canonical_seed(values["effective_llm_by_static_path"])[0]
        return RunStartedRecord(
            flow_fingerprint=values["flow_fingerprint"],
            initial_pool=initial_pool,
            start_wiring_override=start_wiring_override,
            effective_llm_by_static_path=effective,
            num_workers=values["num_workers"],
        )
    if kind == "run_resumed":
        return RunResumedRecord(
            retry_node_ids=tuple(values["retry_node_ids"]),
            num_workers=values["num_workers"],
        )
    if kind == "node_started":
        return NodeStartedRecord(node_id=values["node_id"])
    if kind == "node_succeeded":
        decoded = _decode_typed(values["value"])
        canonical = ANY_VALUE_ADAPTER.dump_python(decoded, mode="json", warnings="error")
        return NodeSucceededRecord(
            node_id=values["node_id"],
            commit_target=values["commit_target"],
            value=_copy_json(canonical),
        )
    if kind == "node_routed":
        return NodeRoutedRecord(node_id=values["node_id"], handle=values["handle"])
    if kind == "node_expanded":
        seed = canonical_seed(values["seed"])[0]
        return NodeExpandedRecord(
            spawner_id=values["spawner_id"],
            seed=seed,
            prune_ids=tuple(values["prune_ids"]),
            topology_digest=values["topology_digest"],
        )
    if kind == "node_failed":
        failure_payload = values["failure"]
        locator_payload = values["locator"]
        range_payload = values["range"]
        return NodeFailedRecord(
            failure=Failure(
                code=failure_payload["code"],
                message=failure_payload["message"],
                node_id=failure_payload["node_id"],
                trace=tuple(
                    OriginRef(origin_payload["key"])
                    for origin_payload in failure_payload["trace"]
                ),
                cause_type=failure_payload["cause_type"],
                traceback=failure_payload["traceback"],
            ),
            locator=(
                SourceSpan(**locator_payload)
                if locator_payload is not None
                else None
            ),
            range=(
                SourceRange(**range_payload)
                if range_payload is not None
                else None
            ),
        )
    raise ValueError(f"unknown run-record kind: {kind!r}")


def _decode_typed(wire: dict[str, Any]) -> TypedValue:
    """Strictly decode one typed-value wire object without allowing extra fields."""
    copied = _copy_json(wire)
    if not isinstance(copied, dict):
        raise ValueError("typed-value wire must be a JSON object")
    _require_fields(copied, {"kind", "value"}, "typed value")
    if copied["kind"] == ValueKind.FILE.value:
        file_wire = copied["value"]
        if not isinstance(file_wire, dict):
            raise ValueError("file typed value must contain a JSON object")
        _require_fields(file_wire, {"uri", "mime", "name"}, "file reference")
    return ANY_VALUE_ADAPTER.validate_python(copied, strict=True)


def _copy_json(value: Any) -> Any:
    """Return a key-sorted deep copy while rejecting values outside strict JSON."""
    if value is None or type(value) in (bool, int):
        return value
    if type(value) is str:
        return _validated_json_string(value)
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("non-finite numbers are not valid canonical JSON")
        return value
    if isinstance(value, list):
        return [_copy_json(item) for item in value]
    if isinstance(value, dict):
        if any(type(key) is not str for key in value):
            raise ValueError("canonical JSON object keys must be strings")
        for key in value:
            _validated_json_string(key)
        return {key: _copy_json(value[key]) for key in sorted(value)}
    raise ValueError(f"{type(value).__name__} is not a canonical JSON value")


def _reject_unordered_containers(
    value: Any,
    *,
    _active: set[int] | None = None,
) -> None:
    """Reject unordered, cyclic, or non-string-keyed canonical input containers."""
    if type(value) is str:
        _validated_json_string(value)
        return
    if isinstance(value, (set, frozenset)):
        raise ValueError("unordered container cannot be canonicalized")
    is_container = (
        isinstance(value, (TypedValue, BaseModel, dict, list, tuple))
        or (is_dataclass(value) and not isinstance(value, type))
    )
    if not is_container:
        return

    active = set() if _active is None else _active
    identity = id(value)
    if identity in active:
        raise ValueError("cyclic container cannot be canonicalized")
    active.add(identity)
    try:
        if isinstance(value, TypedValue):
            _reject_unordered_containers(value.to_object(), _active=active)
            return
        if isinstance(value, BaseModel):
            for field_name in type(value).model_fields:
                _reject_unordered_containers(
                    getattr(value, field_name),
                    _active=active,
                )
            return
        if is_dataclass(value) and not isinstance(value, type):
            for field in fields(value):
                _reject_unordered_containers(
                    getattr(value, field.name),
                    _active=active,
                )
            return
        if isinstance(value, dict):
            if any(type(key) is not str for key in value):
                raise ValueError("canonical JSON object keys must be strings")
            for key, item in value.items():
                _reject_unordered_containers(key, _active=active)
                _reject_unordered_containers(item, _active=active)
            return
        for item in value:
            _reject_unordered_containers(item, _active=active)
    finally:
        active.remove(identity)


def _validated_json_string(value: str) -> str:
    """Return a string only when every code point has a valid UTF-8 representation."""
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError("Unicode surrogate is not valid canonical JSON") from exc
    return value


def _require_fields(value: dict[str, Any], expected: set[str], label: str) -> None:
    """Reject missing or extra object fields with one precise contract error."""
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"invalid {label} fields; missing={missing}, extra={extra}")


def _validated_id_tuple(
    value: Any,
    label: str,
    *,
    require_sorted: bool = False,
) -> list[str]:
    """Validate an immutable ordered ID carrier before converting it to JSON."""
    if type(value) is not tuple:
        raise ValueError(f"{label} must be a tuple of strings")
    if any(type(node_id) is not str or not node_id for node_id in value):
        raise ValueError(f"{label} must contain only non-empty strings")
    if len(set(value)) != len(value):
        raise ValueError(f"{label} must contain unique IDs")
    if require_sorted and list(value) != sorted(value):
        raise ValueError(f"{label} must be sorted")
    return list(value)


def _json_bytes(value: Any) -> bytes:
    """Serialize strict JSON with stable key order and no trailing newline."""
    canonical = _copy_json(value)
    try:
        return json.dumps(
            canonical,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except UnicodeEncodeError as exc:  # defensive if a future JSON carrier bypasses _copy_json
        raise ValueError("Unicode surrogate is not valid canonical JSON") from exc


def _reject_json_constant(value: str) -> None:
    """Reject the non-standard NaN and Infinity tokens accepted by ``json.loads``."""
    raise ValueError(f"invalid JSON numeric constant: {value}")


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Build one decoded JSON object while rejecting ambiguous duplicate members."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _validated_run_id(run_id: Any) -> str:
    """Return one exact non-empty string suitable for envelope identity."""
    if type(run_id) is not str or not run_id:
        raise ValueError("run log run_id must be a non-empty string")
    _validated_json_string(run_id)
    return run_id


def _canonical_envelope(
    record: RunRecord,
    run_id: str,
    sequence: int,
) -> RecordEnvelope:
    """Return a codec-round-tripped envelope independent of caller-owned values."""
    envelope = RecordEnvelope(
        schema=SCHEMA_VERSION,
        run_id=run_id,
        seq=sequence,
        record=record,
    )
    return decode_record(encode_record(envelope))


def _copy_envelope(envelope: RecordEnvelope) -> RecordEnvelope:
    """Return an independent canonical copy safe to expose outside log authority."""
    return decode_record(encode_record(envelope))


def _decode_complete_prefix(data: bytes) -> list[RecordEnvelope]:
    """Validate every newline-terminated record and its shared identity."""
    if not data:
        return []
    lines = data.split(b"\n")
    if lines[-1] != b"":
        raise ValueError("run-log prefix is not newline terminated")

    records: list[RecordEnvelope] = []
    run_id: str | None = None
    for expected_sequence, line in enumerate(lines[:-1]):
        envelope = decode_record(line)
        if envelope.seq != expected_sequence:
            raise ValueError(
                "run-log sequence mismatch: "
                f"expected {expected_sequence}, found {envelope.seq}"
            )
        if run_id is None:
            run_id = envelope.run_id
        elif envelope.run_id != run_id:
            raise ValueError(
                f"run-log run_id mismatch: expected {run_id!r}, "
                f"found {envelope.run_id!r}"
            )
        records.append(envelope)
    return records


def _rollback_append(descriptor: int, prefix_size: int) -> list[BaseException]:
    """Best-effort truncate, reposition, and sync after a failed append attempt."""
    errors: list[BaseException] = []
    for operation in (
        lambda: os.ftruncate(descriptor, prefix_size),
        lambda: os.lseek(descriptor, prefix_size, os.SEEK_SET),
        lambda: os.fsync(descriptor),
    ):
        try:
            operation()
        except BaseException as exc:
            errors.append(exc)
    return errors


def _release_file(file: BinaryIO) -> None:
    """Unlock and close an adopted file, closing even when explicit unlock fails."""
    try:
        fcntl.flock(file.fileno(), fcntl.LOCK_UN)
    finally:
        file.close()


def _release_fd(descriptor: int, *, locked: bool) -> None:
    """Release a not-yet-adopted descriptor on an opener failure path."""
    try:
        if locked:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


__all__ = [
    "SCHEMA_VERSION",
    "JsonlRunLog",
    "MemoryRunLog",
    "NodeExpandedRecord",
    "NodeFailedRecord",
    "NodeRoutedRecord",
    "NodeStartedRecord",
    "NodeSucceededRecord",
    "RecordEnvelope",
    "RunLog",
    "RunRecord",
    "RunResumedRecord",
    "RunStartedRecord",
    "canonical_seed",
    "canonical_typed",
    "decode_pool",
    "decode_record",
    "encode_pool",
    "encode_record",
]
