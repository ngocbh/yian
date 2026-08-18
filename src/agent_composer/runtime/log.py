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
    Annotated,
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
from agent_composer.memory import (
    EvaluationEvidence,
    MemoryBoundary,
    MemoryBoundaryKind,
    MemoryFrameRef,
    MemoryFrameSpec,
    MemoryMutation,
    MemoryOccurrenceKind,
    MemoryOccurrenceRef,
    MemoryOccurrenceSegment,
    MemoryOperationFailure,
    MemoryOwner,
    MemoryScope,
    MemorySection,
    MemorySnapshotEnvelope,
    ModelTextInteraction,
    ModelValueInteraction,
    ToolCallInteraction,
    ToolResultInteraction,
    VisibleInteraction,
)
from agent_composer.typesys.pool import VariablePool
from agent_composer.typesys.values import (
    ANY_VALUE_ADAPTER,
    Type,
    TypedValue,
    ValueKind,
    build_value,
    build_value_as,
)


SCHEMA_VERSION: Final = 4
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
    memory_layout: tuple[MemoryFrameSpec, ...] | None = None
    memory_snapshots: tuple[MemorySnapshotEnvelope, ...] | None = None
    learn: bool = False

    def __post_init__(self) -> None:
        """Validate the enabled-memory shape and its canonical policy inventory."""
        _validate_run_memory_configuration(
            self.memory_layout,
            self.memory_snapshots,
            self.learn,
        )


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
    evidence: EvaluationEvidence | None = None

    def __post_init__(self) -> None:
        """Require exact source-neutral evidence when it is present."""
        _validate_optional_evidence(self.evidence)


@dataclass(frozen=True, slots=True)
class NodeRoutedRecord:
    """Select one route and retain any terminal failure it first proves."""

    node_id: str
    handle: str
    terminal_failure: Failure | None = None
    evidence: EvaluationEvidence | None = None

    def __post_init__(self) -> None:
        """Require exact source-neutral evidence when it is present."""
        _validate_optional_evidence(self.evidence)


@dataclass(frozen=True, slots=True)
class NodeExpandedRecord:
    """Splice one deterministic expansion from its canonical seed and digest."""

    spawner_id: str
    seed: Any
    prune_ids: tuple[str, ...]
    topology_digest: str
    evidence: EvaluationEvidence | None = None

    def __post_init__(self) -> None:
        """Require exact source-neutral evidence when it is present."""
        _validate_optional_evidence(self.evidence)


@dataclass(frozen=True, slots=True)
class NodeFailedRecord:
    """Make a run terminal-failed with the captured semantic failure."""

    failure: Failure
    evidence: EvaluationEvidence | None = None

    def __post_init__(self) -> None:
        """Require exact source-neutral evidence when it is present."""
        _validate_optional_evidence(self.evidence)


@dataclass(frozen=True, slots=True)
class MemoryFrameLearningStartedRecord:
    """Start learning for one frame from its canonical occurrence set."""

    policy_name: str
    frame_ref: MemoryFrameRef
    occurrences: tuple[MemoryOccurrenceRef, ...]

    def __post_init__(self) -> None:
        """Validate policy, ownership, uniqueness, and canonical occurrence order."""
        _require_nonempty_string(self.policy_name, "policy_name")
        if type(self.frame_ref) is not MemoryFrameRef:
            raise TypeError("frame_ref must be a MemoryFrameRef")
        MemoryFrameRef(
            self.frame_ref.root_flow_id,
            self.frame_ref.static_path,
            self.frame_ref.owner,
        )
        if type(self.occurrences) is not tuple:
            raise TypeError("occurrences must be a tuple")
        for occurrence in self.occurrences:
            if type(occurrence) is not MemoryOccurrenceRef:
                raise TypeError("occurrences items must be MemoryOccurrenceRef values")
            MemoryOccurrenceRef(
                occurrence.frame_ref,
                occurrence.occurrence_path,
            )
            if occurrence.frame_ref != self.frame_ref:
                raise ValueError("occurrence frame_ref must match the learning frame")
        keys = tuple(_occurrence_key(item) for item in self.occurrences)
        if keys != tuple(sorted(set(keys))):
            raise ValueError("occurrences must be sorted and unique")


@dataclass(frozen=True, slots=True)
class MemoryFrameLearnedRecord:
    """Retain one frame's canonical proposed mutation batch."""

    policy_name: str
    frame_ref: MemoryFrameRef
    mutations: tuple[MemoryMutation, ...]

    def __post_init__(self) -> None:
        """Validate frame ownership and canonical unique mutation identities."""
        _require_nonempty_string(self.policy_name, "policy_name")
        if type(self.frame_ref) is not MemoryFrameRef:
            raise TypeError("frame_ref must be a MemoryFrameRef")
        MemoryFrameRef(
            self.frame_ref.root_flow_id,
            self.frame_ref.static_path,
            self.frame_ref.owner,
        )
        _validate_mutations(self.mutations, frame_ref=self.frame_ref)


@dataclass(frozen=True, slots=True)
class MemoryFrameLearningFailedRecord:
    """Retain one sanitized frame-local learning failure."""

    policy_name: str
    frame_ref: MemoryFrameRef
    failure: MemoryOperationFailure

    def __post_init__(self) -> None:
        """Validate exact policy, frame, and failure carriers."""
        _require_nonempty_string(self.policy_name, "policy_name")
        if type(self.frame_ref) is not MemoryFrameRef:
            raise TypeError("frame_ref must be a MemoryFrameRef")
        MemoryFrameRef(
            self.frame_ref.root_flow_id,
            self.frame_ref.static_path,
            self.frame_ref.owner,
        )
        if type(self.failure) is not MemoryOperationFailure:
            raise TypeError("failure must be a MemoryOperationFailure")
        MemoryOperationFailure(
            self.failure.stage,
            self.failure.code,
            self.failure.message,
            self.failure.retryable,
        )


@dataclass(frozen=True, slots=True)
class MemoryCommitRequestedRecord:
    """Request one policy's atomic idempotent mutation commit."""

    policy_name: str
    commit_id: str
    mutations: tuple[MemoryMutation, ...]

    def __post_init__(self) -> None:
        """Validate exact identity and canonical unique mutations."""
        _require_nonempty_string(self.policy_name, "policy_name")
        _require_nonempty_string(self.commit_id, "commit_id")
        _validate_mutations(self.mutations)


@dataclass(frozen=True, slots=True)
class MemoryCommitSucceededRecord:
    """Confirm one policy commit completed successfully."""

    policy_name: str
    commit_id: str

    def __post_init__(self) -> None:
        """Validate the exact policy and commit identities."""
        _require_nonempty_string(self.policy_name, "policy_name")
        _require_nonempty_string(self.commit_id, "commit_id")


@dataclass(frozen=True, slots=True)
class MemoryCommitFailedRecord:
    """Retain one sanitized policy commit failure."""

    policy_name: str
    commit_id: str
    failure: MemoryOperationFailure

    def __post_init__(self) -> None:
        """Validate exact identities and failure carrier."""
        _require_nonempty_string(self.policy_name, "policy_name")
        _require_nonempty_string(self.commit_id, "commit_id")
        if type(self.failure) is not MemoryOperationFailure:
            raise TypeError("failure must be a MemoryOperationFailure")
        MemoryOperationFailure(
            self.failure.stage,
            self.failure.code,
            self.failure.message,
            self.failure.retryable,
        )


RunRecord: TypeAlias = (
    RunStartedRecord
    | RunResumedRecord
    | NodeStartedRecord
    | NodeSucceededRecord
    | NodeRoutedRecord
    | NodeExpandedRecord
    | NodeFailedRecord
    | MemoryFrameLearningStartedRecord
    | MemoryFrameLearnedRecord
    | MemoryFrameLearningFailedRecord
    | MemoryCommitRequestedRecord
    | MemoryCommitSucceededRecord
    | MemoryCommitFailedRecord
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
    MemoryFrameLearningStartedRecord,
    MemoryFrameLearnedRecord,
    MemoryFrameLearningFailedRecord,
    MemoryCommitRequestedRecord,
    MemoryCommitSucceededRecord,
    MemoryCommitFailedRecord,
)


def _require_nonempty_string(value: object, name: str) -> str:
    """Return one exact non-empty record identity string."""
    if type(value) is not str:
        raise TypeError(f"{name} must be a string")
    if not value:
        raise ValueError(f"{name} must be non-empty")
    return value


def _frame_key(ref: MemoryFrameRef) -> tuple[str, tuple[str, ...], str]:
    """Return the canonical sortable identity of one memory frame."""
    return (ref.root_flow_id, ref.static_path, ref.owner.value)


def _occurrence_key(
    occurrence: MemoryOccurrenceRef,
) -> tuple[
    tuple[str, tuple[str, ...], str],
    tuple[tuple[str, tuple[str, ...], int], ...],
]:
    """Return the canonical sortable identity of one dynamic occurrence."""
    return (
        _frame_key(occurrence.frame_ref),
        tuple(
            (segment.kind.value, segment.static_path, segment.index)
            for segment in occurrence.occurrence_path
        ),
    )


def _validate_optional_evidence(evidence: EvaluationEvidence | None) -> None:
    """Reject forged or unsupported node-evidence carriers."""
    if evidence is not None and type(evidence) is not EvaluationEvidence:
        raise TypeError("evidence must be an EvaluationEvidence or None")
    if evidence is not None:
        EvaluationEvidence(
            evidence.scope,
            evidence.bound_inputs,
            evidence.base_system_prompt,
            evidence.prompt,
            evidence.memory_sections,
            evidence.interactions,
        )


def _validate_run_memory_configuration(
    layout: tuple[MemoryFrameSpec, ...] | None,
    snapshots: tuple[MemorySnapshotEnvelope, ...] | None,
    learn: bool,
) -> None:
    """Validate one RunStarted memory configuration and canonical ordering."""
    if type(learn) is not bool:
        raise TypeError("learn must be a boolean")
    if (layout is None) != (snapshots is None):
        raise ValueError(
            "memory_layout and memory_snapshots must both be enabled or disabled"
        )
    if layout is None:
        if learn:
            raise ValueError("learn requires enabled memory")
        return
    if type(layout) is not tuple or type(snapshots) is not tuple:
        raise TypeError(
            "memory_layout and memory_snapshots must be tuples when enabled"
        )
    for spec in layout:
        if type(spec) is not MemoryFrameSpec:
            raise TypeError("memory_layout items must be MemoryFrameSpec values")
        MemoryFrameSpec(spec.ref, spec.policy_name, spec.config)
    layout_keys = tuple(_frame_key(spec.ref) for spec in layout)
    if layout_keys != tuple(sorted(set(layout_keys))):
        raise ValueError("memory_layout frame refs must be sorted and unique")
    layout_roots = {spec.ref.root_flow_id for spec in layout}
    if len(layout_roots) > 1:
        raise ValueError("memory_layout frame refs must share one root_flow_id")

    for snapshot in snapshots:
        if type(snapshot) is not MemorySnapshotEnvelope:
            raise TypeError(
                "memory_snapshots items must be MemorySnapshotEnvelope values"
            )
        MemorySnapshotEnvelope(snapshot.policy_name, snapshot.snapshot)
    snapshot_names = tuple(snapshot.policy_name for snapshot in snapshots)
    if snapshot_names != tuple(sorted(set(snapshot_names))):
        raise ValueError("memory_snapshots policy names must be sorted and unique")
    referenced_names = {spec.policy_name for spec in layout}
    if set(snapshot_names) != referenced_names:
        raise ValueError(
            "memory_snapshots policy names must equal memory_layout policy names"
        )


def _validate_mutations(
    mutations: tuple[MemoryMutation, ...],
    *,
    frame_ref: MemoryFrameRef | None = None,
) -> None:
    """Validate carriers, ownership, unique IDs, and semantic proposal order."""
    if type(mutations) is not tuple:
        raise TypeError("mutations must be a tuple")
    for mutation in mutations:
        if type(mutation) is not MemoryMutation:
            raise TypeError("mutations items must be MemoryMutation values")
        MemoryMutation(mutation.mutation_id, mutation.frame_ref, mutation.payload)
        if frame_ref is not None and mutation.frame_ref != frame_ref:
            raise ValueError("mutation frame_ref must match the learning frame")
    ids = tuple(mutation.mutation_id for mutation in mutations)
    if len(ids) != len(set(ids)):
        raise ValueError("mutation IDs must be unique")


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
    memory_layout: list[_MemoryFrameSpecPayload] | None
    memory_snapshots: list[_MemorySnapshotEnvelopePayload] | None
    learn: bool


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
    evidence: _EvaluationEvidencePayload | None


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


class _MemoryFrameRefPayload(_StrictModel):
    """Strict wire form of one stable memory-frame identity."""

    root_flow_id: str = Field(min_length=1)
    static_path: list[str] = Field(min_length=1)
    owner: Literal["flow", "agent"]

    @field_validator("static_path")
    @classmethod
    def _nonempty_path(cls, value: list[str]) -> list[str]:
        """Reject empty path components."""
        if any(not part for part in value):
            raise ValueError("memory frame path components must be non-empty")
        return value


class _MemoryOccurrenceSegmentPayload(_StrictModel):
    """Strict wire form of one dynamic occurrence segment."""

    kind: Literal["map_item", "loop_iteration"]
    static_path: list[str] = Field(min_length=1)
    index: int = Field(ge=0)

    @field_validator("static_path")
    @classmethod
    def _nonempty_path(cls, value: list[str]) -> list[str]:
        """Reject empty path components."""
        if any(not part for part in value):
            raise ValueError("occurrence path components must be non-empty")
        return value


class _MemoryOccurrenceRefPayload(_StrictModel):
    """Strict wire form of one dynamic frame occurrence."""

    frame_ref: _MemoryFrameRefPayload
    occurrence_path: list[_MemoryOccurrenceSegmentPayload]


class _MemoryBoundaryPayload(_StrictModel):
    """Strict wire form of one flow-owned memory boundary."""

    kind: Literal["start", "end"]
    frame_ref: _MemoryFrameRefPayload


class _MemoryScopePayload(_StrictModel):
    """Strict wire form of one compiled node's memory scope."""

    root_flow_id: str = Field(min_length=1)
    static_path: list[str] = Field(min_length=1)
    frame_stack: list[_MemoryFrameRefPayload]
    occurrence_path: list[_MemoryOccurrenceSegmentPayload]
    boundary: _MemoryBoundaryPayload | None

    @field_validator("static_path")
    @classmethod
    def _nonempty_path(cls, value: list[str]) -> list[str]:
        """Reject empty path components."""
        if any(not part for part in value):
            raise ValueError("memory scope path components must be non-empty")
        return value


class _MemoryFrameSpecPayload(_StrictModel):
    """Strict origin-free wire form of one effective frame specification."""

    ref: _MemoryFrameRefPayload
    policy_name: str = Field(min_length=1)
    config: dict[str, Any]


class _MemorySnapshotEnvelopePayload(_StrictModel):
    """Strict wire form of one named policy snapshot."""

    policy_name: str = Field(min_length=1)
    snapshot: dict[str, Any]


class _MemorySectionPayload(_StrictModel):
    """Strict wire form of one separate recalled-memory section."""

    frame_ref: _MemoryFrameRefPayload
    text: str


class _ModelTextInteractionPayload(_StrictModel):
    """Strict wire form of visible model text."""

    kind: Literal["model_text"]
    text: str


class _ModelValueInteractionPayload(_StrictModel):
    """Strict wire form of a visible structured model value."""

    kind: Literal["model_value"]
    value: Any


class _ToolCallInteractionPayload(_StrictModel):
    """Strict wire form of one visible tool call."""

    kind: Literal["tool_call"]
    name: str = Field(min_length=1)
    arguments: dict[str, Any]


class _ToolResultInteractionPayload(_StrictModel):
    """Strict wire form of one visible tool result."""

    kind: Literal["tool_result"]
    name: str = Field(min_length=1)
    text: str


_VisibleInteractionPayload = Annotated[
    _ModelTextInteractionPayload
    | _ModelValueInteractionPayload
    | _ToolCallInteractionPayload
    | _ToolResultInteractionPayload,
    Field(discriminator="kind"),
]


class _EvaluationEvidencePayload(_StrictModel):
    """Strict wire form of one source-neutral accepted evaluation trace."""

    scope: _MemoryScopePayload
    bound_inputs: dict[str, Any] | None
    base_system_prompt: str | None
    prompt: str | None
    memory_sections: list[_MemorySectionPayload]
    interactions: list[_VisibleInteractionPayload]


class _MemoryMutationPayload(_StrictModel):
    """Strict wire form of one frame-owned memory mutation."""

    mutation_id: str = Field(min_length=1)
    frame_ref: _MemoryFrameRefPayload
    payload: Any


class _MemoryOperationFailurePayload(_StrictModel):
    """Strict wire form of one sanitized memory-operation failure."""

    stage: str = Field(min_length=1)
    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    retryable: bool


class _NodeRoutedPayload(_StrictModel):
    """Strict wire payload for :class:`NodeRoutedRecord`."""

    node_id: str = Field(min_length=1)
    handle: str = Field(min_length=1)
    terminal_failure: _FailurePayload | None
    evidence: _EvaluationEvidencePayload | None


class _NodeExpandedPayload(_StrictModel):
    """Strict wire payload for :class:`NodeExpandedRecord`."""

    spawner_id: str = Field(min_length=1)
    seed: Any
    prune_ids: list[str]
    topology_digest: str = Field(pattern=_SHA256_PATTERN)
    evidence: _EvaluationEvidencePayload | None

    @field_validator("prune_ids")
    @classmethod
    def _sorted_unique_ids(cls, value: list[str]) -> list[str]:
        """Require the canonical sorted set representation used by expansion records."""
        if any(not node_id for node_id in value):
            raise ValueError("prune IDs must be non-empty")
        if value != sorted(set(value)):
            raise ValueError("prune IDs must be sorted and unique")
        return value


class _NodeFailedPayload(_StrictModel):
    """Strict wire payload for :class:`NodeFailedRecord`."""

    failure: _FailurePayload
    evidence: _EvaluationEvidencePayload | None


class _MemoryFrameLearningStartedPayload(_StrictModel):
    """Strict wire payload for frame-learning start."""

    policy_name: str = Field(min_length=1)
    frame_ref: _MemoryFrameRefPayload
    occurrences: list[_MemoryOccurrenceRefPayload]


class _MemoryFrameLearnedPayload(_StrictModel):
    """Strict wire payload for frame-local proposed mutations."""

    policy_name: str = Field(min_length=1)
    frame_ref: _MemoryFrameRefPayload
    mutations: list[_MemoryMutationPayload]


class _MemoryFrameLearningFailedPayload(_StrictModel):
    """Strict wire payload for one frame-learning failure."""

    policy_name: str = Field(min_length=1)
    frame_ref: _MemoryFrameRefPayload
    failure: _MemoryOperationFailurePayload


class _MemoryCommitRequestedPayload(_StrictModel):
    """Strict wire payload for one atomic policy commit request."""

    policy_name: str = Field(min_length=1)
    commit_id: str = Field(min_length=1)
    mutations: list[_MemoryMutationPayload]


class _MemoryCommitSucceededPayload(_StrictModel):
    """Strict wire payload for one successful policy commit."""

    policy_name: str = Field(min_length=1)
    commit_id: str = Field(min_length=1)


class _MemoryCommitFailedPayload(_StrictModel):
    """Strict wire payload for one failed policy commit."""

    policy_name: str = Field(min_length=1)
    commit_id: str = Field(min_length=1)
    failure: _MemoryOperationFailurePayload


_KIND_TO_MODEL: dict[str, type[_StrictModel]] = {
    "run_started": _RunStartedPayload,
    "run_resumed": _RunResumedPayload,
    "node_started": _NodeStartedPayload,
    "node_succeeded": _NodeSucceededPayload,
    "node_routed": _NodeRoutedPayload,
    "node_expanded": _NodeExpandedPayload,
    "node_failed": _NodeFailedPayload,
    "memory_frame_learning_started": _MemoryFrameLearningStartedPayload,
    "memory_frame_learned": _MemoryFrameLearnedPayload,
    "memory_frame_learning_failed": _MemoryFrameLearningFailedPayload,
    "memory_commit_requested": _MemoryCommitRequestedPayload,
    "memory_commit_succeeded": _MemoryCommitSucceededPayload,
    "memory_commit_failed": _MemoryCommitFailedPayload,
}

_TYPE_TO_KIND: dict[type[Any], str] = {
    RunStartedRecord: "run_started",
    RunResumedRecord: "run_resumed",
    NodeStartedRecord: "node_started",
    NodeSucceededRecord: "node_succeeded",
    NodeRoutedRecord: "node_routed",
    NodeExpandedRecord: "node_expanded",
    NodeFailedRecord: "node_failed",
    MemoryFrameLearningStartedRecord: "memory_frame_learning_started",
    MemoryFrameLearnedRecord: "memory_frame_learned",
    MemoryFrameLearningFailedRecord: "memory_frame_learning_failed",
    MemoryCommitRequestedRecord: "memory_commit_requested",
    MemoryCommitSucceededRecord: "memory_commit_succeeded",
    MemoryCommitFailedRecord: "memory_commit_failed",
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
        raw = _copy_json(raw)
    except RecursionError as exc:
        raise ValueError("run-record JSON nesting exceeds the supported depth") from exc
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("invalid run-record JSON") from exc
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


def _failure_payload(failure: Failure, *, carrier: str) -> dict[str, Any]:
    """Return the one explicit wire form shared by failure-bearing records."""
    if type(failure) is not Failure:
        raise TypeError(f"{carrier} failure must be a Failure")
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


def _failure_from_payload(payload: dict[str, Any]) -> Failure:
    """Rebuild one canonical Failure from an already validated wire mapping."""
    return Failure(
        code=payload["code"],
        message=payload["message"],
        node_id=payload["node_id"],
        trace=tuple(OriginRef(origin["key"]) for origin in payload["trace"]),
        cause_type=payload["cause_type"],
        traceback=payload["traceback"],
    )


def _frame_ref_payload(ref: MemoryFrameRef) -> dict[str, Any]:
    """Encode one exact stable memory-frame identity."""
    if type(ref) is not MemoryFrameRef:
        raise TypeError("memory frame ref must be a MemoryFrameRef")
    MemoryFrameRef(ref.root_flow_id, ref.static_path, ref.owner)
    return {
        "root_flow_id": ref.root_flow_id,
        "static_path": list(ref.static_path),
        "owner": ref.owner.value,
    }


def _frame_ref_from_payload(payload: dict[str, Any]) -> MemoryFrameRef:
    """Decode one already-validated memory-frame identity."""
    return MemoryFrameRef(
        root_flow_id=payload["root_flow_id"],
        static_path=tuple(payload["static_path"]),
        owner=MemoryOwner(payload["owner"]),
    )


def _occurrence_segment_payload(segment: MemoryOccurrenceSegment) -> dict[str, Any]:
    """Encode one exact dynamic occurrence segment."""
    if type(segment) is not MemoryOccurrenceSegment:
        raise TypeError("occurrence segment must be a MemoryOccurrenceSegment")
    MemoryOccurrenceSegment(segment.kind, segment.static_path, segment.index)
    return {
        "kind": segment.kind.value,
        "static_path": list(segment.static_path),
        "index": segment.index,
    }


def _occurrence_segment_from_payload(
    payload: dict[str, Any],
) -> MemoryOccurrenceSegment:
    """Decode one already-validated dynamic occurrence segment."""
    return MemoryOccurrenceSegment(
        kind=MemoryOccurrenceKind(payload["kind"]),
        static_path=tuple(payload["static_path"]),
        index=payload["index"],
    )


def _occurrence_ref_payload(occurrence: MemoryOccurrenceRef) -> dict[str, Any]:
    """Encode one exact dynamic frame occurrence."""
    if type(occurrence) is not MemoryOccurrenceRef:
        raise TypeError("occurrence must be a MemoryOccurrenceRef")
    MemoryOccurrenceRef(occurrence.frame_ref, occurrence.occurrence_path)
    return {
        "frame_ref": _frame_ref_payload(occurrence.frame_ref),
        "occurrence_path": [
            _occurrence_segment_payload(segment)
            for segment in occurrence.occurrence_path
        ],
    }


def _occurrence_ref_from_payload(payload: dict[str, Any]) -> MemoryOccurrenceRef:
    """Decode one already-validated dynamic frame occurrence."""
    return MemoryOccurrenceRef(
        frame_ref=_frame_ref_from_payload(payload["frame_ref"]),
        occurrence_path=tuple(
            _occurrence_segment_from_payload(segment)
            for segment in payload["occurrence_path"]
        ),
    )


def _scope_payload(scope: MemoryScope) -> dict[str, Any]:
    """Encode one exact compiled memory scope."""
    if type(scope) is not MemoryScope:
        raise TypeError("evidence scope must be a MemoryScope")
    MemoryScope(
        scope.root_flow_id,
        scope.static_path,
        scope.frame_stack,
        scope.occurrence_path,
        scope.boundary,
    )
    boundary = scope.boundary
    return {
        "root_flow_id": scope.root_flow_id,
        "static_path": list(scope.static_path),
        "frame_stack": [_frame_ref_payload(ref) for ref in scope.frame_stack],
        "occurrence_path": [
            _occurrence_segment_payload(segment)
            for segment in scope.occurrence_path
        ],
        "boundary": (
            None
            if boundary is None
            else {
                "kind": boundary.kind.value,
                "frame_ref": _frame_ref_payload(boundary.frame_ref),
            }
        ),
    }


def _scope_from_payload(payload: dict[str, Any]) -> MemoryScope:
    """Decode one already-validated compiled memory scope."""
    boundary_payload = payload["boundary"]
    boundary = (
        None
        if boundary_payload is None
        else MemoryBoundary(
            MemoryBoundaryKind(boundary_payload["kind"]),
            _frame_ref_from_payload(boundary_payload["frame_ref"]),
        )
    )
    return MemoryScope(
        root_flow_id=payload["root_flow_id"],
        static_path=tuple(payload["static_path"]),
        frame_stack=tuple(
            _frame_ref_from_payload(ref) for ref in payload["frame_stack"]
        ),
        occurrence_path=tuple(
            _occurrence_segment_from_payload(segment)
            for segment in payload["occurrence_path"]
        ),
        boundary=boundary,
    )


def _frame_spec_payload(spec: MemoryFrameSpec) -> dict[str, Any]:
    """Encode one exact origin-free effective memory-frame specification."""
    if type(spec) is not MemoryFrameSpec:
        raise TypeError("memory layout item must be a MemoryFrameSpec")
    MemoryFrameSpec(spec.ref, spec.policy_name, spec.config)
    return {
        "ref": _frame_ref_payload(spec.ref),
        "policy_name": spec.policy_name,
        "config": spec.config,
    }


def _frame_spec_from_payload(payload: dict[str, Any]) -> MemoryFrameSpec:
    """Decode one already-validated effective memory-frame specification."""
    return MemoryFrameSpec(
        ref=_frame_ref_from_payload(payload["ref"]),
        policy_name=payload["policy_name"],
        config=_copy_json(payload["config"]),
    )


def _snapshot_payload(snapshot: MemorySnapshotEnvelope) -> dict[str, Any]:
    """Encode one exact named memory snapshot envelope."""
    if type(snapshot) is not MemorySnapshotEnvelope:
        raise TypeError("memory snapshot must be a MemorySnapshotEnvelope")
    MemorySnapshotEnvelope(snapshot.policy_name, snapshot.snapshot)
    return {
        "policy_name": snapshot.policy_name,
        "snapshot": snapshot.snapshot,
    }


def _snapshot_from_payload(payload: dict[str, Any]) -> MemorySnapshotEnvelope:
    """Decode one already-validated named memory snapshot envelope."""
    return MemorySnapshotEnvelope(
        policy_name=payload["policy_name"],
        snapshot=_copy_json(payload["snapshot"]),
    )


def _memory_section_payload(section: MemorySection) -> dict[str, Any]:
    """Encode one exact separate recalled-memory section."""
    if type(section) is not MemorySection:
        raise TypeError("memory section must be a MemorySection")
    MemorySection(section.frame_ref, section.text)
    return {
        "frame_ref": _frame_ref_payload(section.frame_ref),
        "text": section.text,
    }


def _memory_section_from_payload(payload: dict[str, Any]) -> MemorySection:
    """Decode one already-validated recalled-memory section."""
    return MemorySection(
        frame_ref=_frame_ref_from_payload(payload["frame_ref"]),
        text=payload["text"],
    )


def _interaction_payload(interaction: VisibleInteraction) -> dict[str, Any]:
    """Encode one exact member of the closed visible-interaction vocabulary."""
    if type(interaction) is ModelTextInteraction:
        ModelTextInteraction(interaction.text)
        return {"kind": "model_text", "text": interaction.text}
    if type(interaction) is ModelValueInteraction:
        ModelValueInteraction(interaction.value)
        return {"kind": "model_value", "value": interaction.value}
    if type(interaction) is ToolCallInteraction:
        ToolCallInteraction(interaction.name, interaction.arguments)
        return {
            "kind": "tool_call",
            "name": interaction.name,
            "arguments": interaction.arguments,
        }
    if type(interaction) is ToolResultInteraction:
        ToolResultInteraction(interaction.name, interaction.text)
        return {
            "kind": "tool_result",
            "name": interaction.name,
            "text": interaction.text,
        }
    raise TypeError("unsupported visible interaction")


def _interaction_from_payload(payload: dict[str, Any]) -> VisibleInteraction:
    """Decode one already-validated closed visible interaction."""
    kind = payload["kind"]
    if kind == "model_text":
        return ModelTextInteraction(payload["text"])
    if kind == "model_value":
        return ModelValueInteraction(_copy_json(payload["value"]))
    if kind == "tool_call":
        return ToolCallInteraction(payload["name"], _copy_json(payload["arguments"]))
    if kind == "tool_result":
        return ToolResultInteraction(payload["name"], payload["text"])
    raise ValueError(f"unknown visible interaction kind: {kind!r}")


def _evidence_payload(evidence: EvaluationEvidence | None) -> dict[str, Any] | None:
    """Encode optional exact evaluation evidence."""
    if evidence is None:
        return None
    if type(evidence) is not EvaluationEvidence:
        raise TypeError("evidence must be an EvaluationEvidence or None")
    EvaluationEvidence(
        evidence.scope,
        evidence.bound_inputs,
        evidence.base_system_prompt,
        evidence.prompt,
        evidence.memory_sections,
        evidence.interactions,
    )
    return {
        "scope": _scope_payload(evidence.scope),
        "bound_inputs": evidence.bound_inputs,
        "base_system_prompt": evidence.base_system_prompt,
        "prompt": evidence.prompt,
        "memory_sections": [
            _memory_section_payload(section) for section in evidence.memory_sections
        ],
        "interactions": [
            _interaction_payload(interaction) for interaction in evidence.interactions
        ],
    }


def _evidence_from_payload(payload: dict[str, Any] | None) -> EvaluationEvidence | None:
    """Decode optional already-validated evaluation evidence."""
    if payload is None:
        return None
    return EvaluationEvidence(
        scope=_scope_from_payload(payload["scope"]),
        bound_inputs=(
            None
            if payload["bound_inputs"] is None
            else _copy_json(payload["bound_inputs"])
        ),
        base_system_prompt=payload["base_system_prompt"],
        prompt=payload["prompt"],
        memory_sections=tuple(
            _memory_section_from_payload(section)
            for section in payload["memory_sections"]
        ),
        interactions=tuple(
            _interaction_from_payload(interaction)
            for interaction in payload["interactions"]
        ),
    )


def _mutation_payload(mutation: MemoryMutation) -> dict[str, Any]:
    """Encode one exact frame-owned memory mutation."""
    if type(mutation) is not MemoryMutation:
        raise TypeError("mutation must be a MemoryMutation")
    MemoryMutation(mutation.mutation_id, mutation.frame_ref, mutation.payload)
    return {
        "mutation_id": mutation.mutation_id,
        "frame_ref": _frame_ref_payload(mutation.frame_ref),
        "payload": mutation.payload,
    }


def _mutation_from_payload(payload: dict[str, Any]) -> MemoryMutation:
    """Decode one already-validated frame-owned memory mutation."""
    return MemoryMutation(
        mutation_id=payload["mutation_id"],
        frame_ref=_frame_ref_from_payload(payload["frame_ref"]),
        payload=_copy_json(payload["payload"]),
    )


def _memory_failure_payload(failure: MemoryOperationFailure) -> dict[str, Any]:
    """Encode one exact sanitized memory-operation failure."""
    if type(failure) is not MemoryOperationFailure:
        raise TypeError("failure must be a MemoryOperationFailure")
    MemoryOperationFailure(
        failure.stage,
        failure.code,
        failure.message,
        failure.retryable,
    )
    return {
        "stage": failure.stage,
        "code": failure.code,
        "message": failure.message,
        "retryable": failure.retryable,
    }


def _memory_failure_from_payload(payload: dict[str, Any]) -> MemoryOperationFailure:
    """Decode one already-validated sanitized memory-operation failure."""
    return MemoryOperationFailure(
        stage=payload["stage"],
        code=payload["code"],
        message=payload["message"],
        retryable=payload["retryable"],
    )


def _revalidate_run_record(record: RunRecord) -> None:
    """Replay exact outer and nested invariants before any record is encoded."""
    if type(record) is RunStartedRecord:
        RunStartedRecord(
            record.flow_fingerprint,
            record.initial_pool,
            record.start_wiring_override,
            record.effective_llm_by_static_path,
            record.num_workers,
            record.memory_layout,
            record.memory_snapshots,
            record.learn,
        )
        return
    if type(record) is RunResumedRecord:
        RunResumedRecord(record.retry_node_ids, record.num_workers)
        return
    if type(record) is NodeStartedRecord:
        NodeStartedRecord(record.node_id)
        return
    if type(record) is NodeSucceededRecord:
        NodeSucceededRecord(
            record.node_id,
            record.commit_target,
            record.value,
            record.evidence,
        )
        return
    if type(record) is NodeRoutedRecord:
        NodeRoutedRecord(
            record.node_id,
            record.handle,
            record.terminal_failure,
            record.evidence,
        )
        if record.terminal_failure is not None:
            _failure_payload(record.terminal_failure, carrier="NodeRoutedRecord")
        return
    if type(record) is NodeExpandedRecord:
        NodeExpandedRecord(
            record.spawner_id,
            record.seed,
            record.prune_ids,
            record.topology_digest,
            record.evidence,
        )
        return
    if type(record) is NodeFailedRecord:
        NodeFailedRecord(record.failure, record.evidence)
        _failure_payload(record.failure, carrier="NodeFailedRecord")
        return
    if type(record) is MemoryFrameLearningStartedRecord:
        MemoryFrameLearningStartedRecord(
            record.policy_name,
            record.frame_ref,
            record.occurrences,
        )
        return
    if type(record) is MemoryFrameLearnedRecord:
        MemoryFrameLearnedRecord(
            record.policy_name,
            record.frame_ref,
            record.mutations,
        )
        return
    if type(record) is MemoryFrameLearningFailedRecord:
        MemoryFrameLearningFailedRecord(
            record.policy_name,
            record.frame_ref,
            record.failure,
        )
        return
    if type(record) is MemoryCommitRequestedRecord:
        MemoryCommitRequestedRecord(
            record.policy_name,
            record.commit_id,
            record.mutations,
        )
        return
    if type(record) is MemoryCommitSucceededRecord:
        MemoryCommitSucceededRecord(record.policy_name, record.commit_id)
        return
    if type(record) is MemoryCommitFailedRecord:
        MemoryCommitFailedRecord(
            record.policy_name,
            record.commit_id,
            record.failure,
        )
        return
    raise TypeError(f"unsupported run record: {type(record).__name__}")


def _payload_for_record(record: RunRecord) -> dict[str, Any]:
    """Build and validate the canonical JSON payload for one record variant."""
    _revalidate_run_record(record)
    kind = _kind_for_record(record)
    if isinstance(record, RunStartedRecord):
        _validate_run_memory_configuration(
            record.memory_layout,
            record.memory_snapshots,
            record.learn,
        )
        raw = {
            "flow_fingerprint": record.flow_fingerprint,
            "initial_pool": encode_pool(decode_pool(record.initial_pool)),
            "start_wiring_override": canonical_seed(record.start_wiring_override)[0],
            "effective_llm_by_static_path": canonical_seed(
                record.effective_llm_by_static_path
            )[0],
            "num_workers": record.num_workers,
            "memory_layout": (
                None
                if record.memory_layout is None
                else [_frame_spec_payload(spec) for spec in record.memory_layout]
            ),
            "memory_snapshots": (
                None
                if record.memory_snapshots is None
                else [
                    _snapshot_payload(snapshot)
                    for snapshot in record.memory_snapshots
                ]
            ),
            "learn": record.learn,
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
            "evidence": _evidence_payload(record.evidence),
        }
    elif isinstance(record, NodeRoutedRecord):
        raw = {
            "node_id": record.node_id,
            "handle": record.handle,
            "terminal_failure": (
                _failure_payload(
                    record.terminal_failure,
                    carrier="NodeRoutedRecord",
                )
                if record.terminal_failure is not None
                else None
            ),
            "evidence": _evidence_payload(record.evidence),
        }
    elif isinstance(record, NodeExpandedRecord):
        raw = {
            "spawner_id": record.spawner_id,
            "seed": canonical_seed(record.seed)[0],
            "prune_ids": _validated_id_tuple(
                record.prune_ids, "prune_ids", require_sorted=True
            ),
            "topology_digest": record.topology_digest,
            "evidence": _evidence_payload(record.evidence),
        }
    elif isinstance(record, NodeFailedRecord):
        raw = {
            "failure": _failure_payload(
                record.failure,
                carrier="NodeFailedRecord",
            ),
            "evidence": _evidence_payload(record.evidence),
        }
    elif isinstance(record, MemoryFrameLearningStartedRecord):
        raw = {
            "policy_name": record.policy_name,
            "frame_ref": _frame_ref_payload(record.frame_ref),
            "occurrences": [
                _occurrence_ref_payload(occurrence)
                for occurrence in record.occurrences
            ],
        }
    elif isinstance(record, MemoryFrameLearnedRecord):
        raw = {
            "policy_name": record.policy_name,
            "frame_ref": _frame_ref_payload(record.frame_ref),
            "mutations": [
                _mutation_payload(mutation) for mutation in record.mutations
            ],
        }
    elif isinstance(record, MemoryFrameLearningFailedRecord):
        raw = {
            "policy_name": record.policy_name,
            "frame_ref": _frame_ref_payload(record.frame_ref),
            "failure": _memory_failure_payload(record.failure),
        }
    elif isinstance(record, MemoryCommitRequestedRecord):
        raw = {
            "policy_name": record.policy_name,
            "commit_id": record.commit_id,
            "mutations": [
                _mutation_payload(mutation) for mutation in record.mutations
            ],
        }
    elif isinstance(record, MemoryCommitSucceededRecord):
        raw = {
            "policy_name": record.policy_name,
            "commit_id": record.commit_id,
        }
    elif isinstance(record, MemoryCommitFailedRecord):
        raw = {
            "policy_name": record.policy_name,
            "commit_id": record.commit_id,
            "failure": _memory_failure_payload(record.failure),
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
            memory_layout=(
                None
                if values["memory_layout"] is None
                else tuple(
                    _frame_spec_from_payload(spec)
                    for spec in values["memory_layout"]
                )
            ),
            memory_snapshots=(
                None
                if values["memory_snapshots"] is None
                else tuple(
                    _snapshot_from_payload(snapshot)
                    for snapshot in values["memory_snapshots"]
                )
            ),
            learn=values["learn"],
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
            evidence=_evidence_from_payload(values["evidence"]),
        )
    if kind == "node_routed":
        terminal_payload = values["terminal_failure"]
        return NodeRoutedRecord(
            node_id=values["node_id"],
            handle=values["handle"],
            terminal_failure=(
                _failure_from_payload(terminal_payload)
                if terminal_payload is not None
                else None
            ),
            evidence=_evidence_from_payload(values["evidence"]),
        )
    if kind == "node_expanded":
        seed = canonical_seed(values["seed"])[0]
        return NodeExpandedRecord(
            spawner_id=values["spawner_id"],
            seed=seed,
            prune_ids=tuple(values["prune_ids"]),
            topology_digest=values["topology_digest"],
            evidence=_evidence_from_payload(values["evidence"]),
        )
    if kind == "node_failed":
        failure_payload = values["failure"]
        return NodeFailedRecord(
            failure=_failure_from_payload(failure_payload),
            evidence=_evidence_from_payload(values["evidence"]),
        )
    if kind == "memory_frame_learning_started":
        return MemoryFrameLearningStartedRecord(
            policy_name=values["policy_name"],
            frame_ref=_frame_ref_from_payload(values["frame_ref"]),
            occurrences=tuple(
                _occurrence_ref_from_payload(occurrence)
                for occurrence in values["occurrences"]
            ),
        )
    if kind == "memory_frame_learned":
        return MemoryFrameLearnedRecord(
            policy_name=values["policy_name"],
            frame_ref=_frame_ref_from_payload(values["frame_ref"]),
            mutations=tuple(
                _mutation_from_payload(mutation)
                for mutation in values["mutations"]
            ),
        )
    if kind == "memory_frame_learning_failed":
        return MemoryFrameLearningFailedRecord(
            policy_name=values["policy_name"],
            frame_ref=_frame_ref_from_payload(values["frame_ref"]),
            failure=_memory_failure_from_payload(values["failure"]),
        )
    if kind == "memory_commit_requested":
        return MemoryCommitRequestedRecord(
            policy_name=values["policy_name"],
            commit_id=values["commit_id"],
            mutations=tuple(
                _mutation_from_payload(mutation)
                for mutation in values["mutations"]
            ),
        )
    if kind == "memory_commit_succeeded":
        return MemoryCommitSucceededRecord(
            policy_name=values["policy_name"],
            commit_id=values["commit_id"],
        )
    if kind == "memory_commit_failed":
        return MemoryCommitFailedRecord(
            policy_name=values["policy_name"],
            commit_id=values["commit_id"],
            failure=_memory_failure_from_payload(values["failure"]),
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
    "MemoryCommitFailedRecord",
    "MemoryCommitRequestedRecord",
    "MemoryCommitSucceededRecord",
    "MemoryFrameLearnedRecord",
    "MemoryFrameLearningFailedRecord",
    "MemoryFrameLearningStartedRecord",
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
