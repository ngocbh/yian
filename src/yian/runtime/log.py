"""Closed run-record vocabulary and canonical JSON codecs.

The dispatcher is the sole producer of these durable records.  Records contain
only transition data; their enclosing :class:`RecordEnvelope` owns run identity,
schema version, and sequence.  This module deliberately knows nothing about the
scheduler or reducer so the same validated carriers can back memory and JSONL
logs without creating an authority cycle.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass, replace
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

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, field_validator

import yian.json_codec as json_codec
from yian.diagnostics import Failure, FailureCode, OriginRef
from yian.memory import (
    DeclSite,
    EvaluationEvidence,
    LearningFailure,
    LearningResult,
    MemoryMutation,
    MemoryOccurrenceKind,
    MemoryOccurrenceSegment,
    MemoryOperationFailure,
    MemoryScope,
    MemorySection,
    MemorySnapshotEnvelope,
    MemorySpec,
    MemoryStack,
    ModelTextInteraction,
    ModelValueInteraction,
    PolicyCommitOutcome,
    ToolCallInteraction,
    ToolResultInteraction,
    VisibleInteraction,
)
from yian.memory.contracts import _validate_mutations
from yian.runtime.contracts import (
    AttributedBackwardResult,
    BackwardAttribution,
    BackwardFailure,
    BackwardOutcome,
    ForwardCapture,
    MemoryProposal,
    _validate_route,
    _validate_source,
)
from yian.expr.lineage import FeedbackValue, SourceRef, ValueRoute, validate_feedback
from yian.nodes.base import BackwardResult, _copy_backward_value
from yian.nodes.contracts import ExecutionPosition, ExecutionScope
from yian.typesys.pool import VariablePool
from yian.typesys.values import (
    ANY_VALUE_ADAPTER,
    Type,
    TypedValue,
    ValueKind,
    build_value,
    build_value_as,
    reject_non_json,
    reject_nonfinite,
)


SCHEMA_VERSION: Final = 9
"""The only run-record schema that this version writes and reads."""

_RETIRED_SCHEMAS: Final = frozenset({7, 8})
"""Earlier schemas whose logs are rejected with an explicit message."""

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
        memory_snapshots (`tuple[MemorySnapshotEnvelope, ...] | None`):
            Canonical recorded snapshots, one per memory key; `None` when memory
            is disabled.
        learnable (`bool`):
            Whether the run retains learning evidence.
        memories (`tuple[MemorySpec, ...] | None`):
            Sorted memory key table; `None` when memory is disabled.
        memory_stacks (`tuple[MemoryStack, ...] | None`):
            Each scoped target's ordered key stack, sorted by locator.
    """

    flow_fingerprint: str
    initial_pool: dict[str, Any]
    start_wiring_override: dict[str, Any]
    effective_llm_by_static_path: dict[str, dict[str, Any]]
    num_workers: int
    memory_snapshots: tuple[MemorySnapshotEnvelope, ...] | None = None
    learnable: bool = False
    memories: tuple[MemorySpec, ...] | None = None
    memory_stacks: tuple[MemoryStack, ...] | None = None

    def __post_init__(self) -> None:
        """Validate the enabled-memory shape and its canonical key table."""
        _validate_run_memory_configuration(
            self.memories, self.memory_stacks, self.memory_snapshots, self.learnable,
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
    forward_capture: ForwardCapture | None = None

    def __post_init__(self) -> None:
        """Validate optional visible evidence and canonical forward capture."""
        _validate_optional_evidence(self.evidence)
        _validate_optional_capture(self.forward_capture)


@dataclass(frozen=True, slots=True)
class NodeRoutedRecord:
    """Select one route and retain any terminal failure it first proves."""

    node_id: str
    handle: str
    terminal_failure: Failure | None = None
    evidence: EvaluationEvidence | None = None
    forward_capture: ForwardCapture | None = None

    def __post_init__(self) -> None:
        """Validate optional visible evidence and canonical forward capture."""
        _validate_optional_evidence(self.evidence)
        _validate_optional_capture(self.forward_capture)


@dataclass(frozen=True, slots=True)
class NodeExpandedRecord:
    """Splice one deterministic expansion from its canonical seed and digest."""

    spawner_id: str
    seed: Any
    prune_ids: tuple[str, ...]
    topology_digest: str
    evidence: EvaluationEvidence | None = None
    forward_capture: ForwardCapture | None = None

    def __post_init__(self) -> None:
        """Validate optional visible evidence and canonical forward capture."""
        _validate_optional_evidence(self.evidence)
        _validate_optional_capture(self.forward_capture)


@dataclass(frozen=True, slots=True)
class NodeFailedRecord:
    """Make a run terminal-failed with the captured semantic failure."""

    failure: Failure
    evidence: EvaluationEvidence | None = None
    forward_capture: ForwardCapture | None = None

    def __post_init__(self) -> None:
        """Validate optional visible evidence and canonical forward capture."""
        _validate_optional_evidence(self.evidence)
        _validate_optional_capture(self.forward_capture)


def _validate_attempt_id(attempt_id: int) -> None:
    """Require an exact positive accepted-attempt identity."""
    if type(attempt_id) is not int:
        raise TypeError("attempt_id must be an integer")
    if attempt_id <= 0:
        raise ValueError("attempt_id must be positive")


def _copy_backward_result(result: BackwardResult) -> BackwardResult:
    """Revalidate the exact three-channel carrier before copying its public values."""
    if type(result) is not BackwardResult:
        raise TypeError("result must be an exact BackwardResult")
    if type(result._input_feedback) is not dict or type(result._variable_feedback) is not dict:
        raise TypeError("backward result channels must be exact dictionaries")
    return BackwardResult(result.input_feedback, result.self_feedback, result.variable_feedback)


def _copy_attributed_results(
    results: tuple[AttributedBackwardResult, ...],
) -> tuple[AttributedBackwardResult, ...]:
    """Detach exact occurrence results without inferring whole-trace membership."""
    if type(results) is not tuple:
        raise TypeError("attributed_results must be a tuple")
    copied = []
    for item in results:
        if type(item) is not AttributedBackwardResult:
            raise TypeError("attributed_results require exact AttributedBackwardResult values")
        if type(item._feedback) is not tuple:
            raise TypeError("attributed feedback must be a tuple")
        copied.append(
            AttributedBackwardResult(
                item.source, item.feedback, item.attribution,
                _copy_backward_result(item.result), item.invoked,
            )
        )
    return tuple(copied)


@dataclass(frozen=True, slots=True, init=False)
class BackwardStartedRecord:
    """Start one backward attempt with detached ordinary root feedback."""

    attempt_id: int
    _feedback: FeedbackValue = field(repr=False)

    def __init__(self, attempt_id: int, feedback: FeedbackValue) -> None:
        """Validate an attempt identity and snapshot the supplied message.

        Args:
            attempt_id (`int`): Exact positive identity within this run.
            feedback (`FeedbackValue`): One ordinary root message; output-shape
                agreement is checked by the reducer.

        Raises:
            `TypeError`: For an unsupported identity or message type.
            `ValueError`: For a nonpositive identity or invalid nested message.
        """
        _validate_attempt_id(attempt_id)
        validate_feedback(feedback, feedback)
        object.__setattr__(self, "attempt_id", attempt_id)
        object.__setattr__(self, "_feedback", _copy_backward_value(feedback, "feedback"))

    @property
    def feedback(self) -> FeedbackValue:
        """Return a detached copy of the attempt's original root message."""
        return _copy_backward_value(self._feedback, "feedback")


@dataclass(frozen=True, slots=True)
class BackwardReadyRecord:
    """Retain a computed root result, occurrence inventory, and memory proposals.

    Attributes:
        attempt_id (`int`): Exact positive identity within this run.
        root_result (`BackwardResult`): Completed three-channel root feedback.
        attributed_results (`tuple[AttributedBackwardResult, ...]`): Ordered
            accepted occurrence results, including unreached occurrences.
        proposals (`tuple[MemoryProposal, ...]`): Ordered detached proposals,
            one per recorded memory key. Table membership and attempt agreement
            belong to reduction.
    """

    attempt_id: int
    root_result: BackwardResult
    attributed_results: tuple[AttributedBackwardResult, ...]
    proposals: tuple[MemoryProposal, ...]

    def __post_init__(self) -> None:
        """Detach exact carriers; ownership and attempt agreement belong to reduction."""
        _validate_attempt_id(self.attempt_id)
        object.__setattr__(self, "root_result", _copy_backward_result(self.root_result))
        object.__setattr__(self, "attributed_results", _copy_attributed_results(self.attributed_results))
        if type(self.proposals) is not tuple:
            raise TypeError("proposals must be a tuple")
        if any(type(proposal) is not MemoryProposal for proposal in self.proposals):
            raise TypeError("proposals require exact MemoryProposal values")
        object.__setattr__(self, "proposals", tuple(replace(p) for p in self.proposals))


@dataclass(frozen=True, slots=True)
class BackwardFinishedRecord:
    """Settle one backward attempt with its outcome and completed inventory.

    Attributes:
        attempt_id (`int`): Exact positive identity within this run.
        outcome (`BackwardOutcome`): Backward and learning dispositions, keeping
            completed root feedback when a later operation fails.
        attributed_results (`tuple[AttributedBackwardResult, ...]`): Defaults to
            an empty tuple. A supplied completed inventory must agree with the
            attempt's accepted computation; the reducer checks that agreement.
    """

    attempt_id: int
    outcome: BackwardOutcome
    attributed_results: tuple[AttributedBackwardResult, ...] = ()

    def __post_init__(self) -> None:
        """Detach exact outcome data without reconstructing missing computation."""
        _validate_attempt_id(self.attempt_id)
        if type(self.outcome) is not BackwardOutcome:
            raise TypeError("outcome must be an exact BackwardOutcome")
        if self.outcome.feedback is not None:
            _copy_backward_result(self.outcome.feedback)
        object.__setattr__(self, "outcome", replace(self.outcome))
        object.__setattr__(self, "attributed_results", _copy_attributed_results(self.attributed_results))


@dataclass(frozen=True, slots=True)
class MemoryCommitRequestedRecord:
    """Request one memory's atomic idempotent mutation commit.

    The required keyword-only `attempt_id` identifies the accepted backward attempt.
    """

    memory: str
    commit_id: str
    mutations: tuple[MemoryMutation, ...]
    attempt_id: int = field(kw_only=True)

    def __post_init__(self) -> None:
        """Validate exact identity and canonical unique mutations."""
        _validate_attempt_id(self.attempt_id)
        _require_nonempty_string(self.memory, "memory")
        _require_nonempty_string(self.commit_id, "commit_id")
        _validate_mutations(self.mutations)


@dataclass(frozen=True, slots=True)
class MemoryCommitSucceededRecord:
    """Confirm one memory commit completed successfully.

    The required positive keyword-only `attempt_id` identifies the accepted
    backward attempt whose memory request this confirmation settles.
    """

    memory: str
    commit_id: str
    attempt_id: int = field(kw_only=True)

    def __post_init__(self) -> None:
        """Validate the exact memory and commit identities."""
        _validate_attempt_id(self.attempt_id)
        _require_nonempty_string(self.memory, "memory")
        _require_nonempty_string(self.commit_id, "commit_id")


@dataclass(frozen=True, slots=True)
class MemoryCommitFailedRecord:
    """Retain one observed memory commit failure.

    The required positive keyword-only `attempt_id` identifies the accepted
    backward attempt whose memory request this failure settles.
    """

    memory: str
    commit_id: str
    failure: MemoryOperationFailure
    attempt_id: int = field(kw_only=True)

    def __post_init__(self) -> None:
        """Validate exact identities and failure carrier."""
        _validate_attempt_id(self.attempt_id)
        _require_nonempty_string(self.memory, "memory")
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
    | MemoryCommitRequestedRecord
    | MemoryCommitSucceededRecord
    | MemoryCommitFailedRecord
    | BackwardStartedRecord
    | BackwardReadyRecord
    | BackwardFinishedRecord
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
    MemoryCommitRequestedRecord,
    MemoryCommitSucceededRecord,
    MemoryCommitFailedRecord,
    BackwardStartedRecord,
    BackwardReadyRecord,
    BackwardFinishedRecord,
)


def _require_nonempty_string(value: object, name: str) -> str:
    """Return one exact non-empty record identity string."""
    if type(value) is not str:
        raise TypeError(f"{name} must be a string")
    if not value:
        raise ValueError(f"{name} must be non-empty")
    return value


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
    memories: tuple[MemorySpec, ...] | None,
    stacks: tuple[MemoryStack, ...] | None,
    snapshots: tuple[MemorySnapshotEnvelope, ...] | None,
    learnable: bool,
) -> None:
    """Validate one RunStarted memory table, its stacks and its snapshots."""
    if type(learnable) is not bool:
        raise TypeError("learnable must be a boolean")
    if len({memories is None, stacks is None, snapshots is None}) != 1:
        raise ValueError(
            "memories, memory_stacks and memory_snapshots must all be recorded or all be absent"
        )
    if memories is None:
        return
    if type(memories) is not tuple or type(stacks) is not tuple or type(snapshots) is not tuple:
        raise TypeError(
            "memories, memory_stacks and memory_snapshots must be tuples when recorded"
        )
    for spec in memories:
        if type(spec) is not MemorySpec:
            raise TypeError("memories items must be MemorySpec values")
        MemorySpec(spec.key, spec.type, spec.config, spec.owners, spec.conversation_eligible)
    keys = tuple(spec.key for spec in memories)
    if keys != tuple(sorted(set(keys))):
        raise ValueError("memories keys must be sorted and unique")
    for snapshot in snapshots:
        if type(snapshot) is not MemorySnapshotEnvelope:
            raise TypeError("memory_snapshots items must be MemorySnapshotEnvelope values")
        MemorySnapshotEnvelope(snapshot.memory, snapshot.snapshot)
    snapshot_keys = tuple(snapshot.memory for snapshot in snapshots)
    if snapshot_keys != tuple(sorted(set(snapshot_keys))):
        raise ValueError("memory_snapshots memory keys must be sorted and unique")
    if snapshot_keys != keys:
        raise ValueError("memory_snapshots memory keys must equal memories keys")
    for stack in stacks:
        if type(stack) is not MemoryStack:
            raise TypeError("memory_stacks items must be MemoryStack values")
        MemoryStack(stack.locator, stack.memories)
    locators = tuple(stack.locator for stack in stacks)
    if locators != tuple(sorted(set(locators))):
        raise ValueError("memory_stacks locators must be sorted and unique")
    unknown = sorted({key for stack in stacks for key in stack.memories} - set(keys))
    if unknown:
        raise ValueError(f"memory_stacks name unknown memory keys: {unknown}")


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
        _require_schema_kind(self.schema, _kind_for_record(self.record))
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

    @classmethod
    def _from_records(cls, records: tuple[RecordEnvelope, ...]) -> MemoryRunLog:
        """Reopen a detached canonical prefix without renumbering accepted records."""
        if type(records) is not tuple or not records:
            raise ValueError("reopening requires a nonempty record tuple")
        if any(type(envelope) is not RecordEnvelope for envelope in records):
            raise TypeError("reopening requires exact RecordEnvelope values")
        canonical = tuple(_copy_envelope(envelope) for envelope in records)
        run_id = canonical[0].run_id
        for sequence, envelope in enumerate(canonical):
            if envelope.run_id != run_id or envelope.seq != sequence:
                raise ValueError("reopened prefix must have one run and contiguous sequences")
        log = cls(run_id)
        log._records = list(canonical)
        log._next_sequence = len(canonical)
        return log

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
            _lock_exclusive(fd, canonical_path)
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
        newline are never records—even when those bytes form valid JSON.  They
        are durably truncated only after the complete lines decode as a
        non-empty run log, so a rejected file is left unchanged.
        """
        canonical_path = Path(path)
        fd: int | None = None
        file: BinaryIO | None = None
        locked = False
        try:
            fd = os.open(canonical_path, os.O_RDWR)
            _lock_exclusive(fd, canonical_path)
            locked = True
            file = os.fdopen(fd, "r+b", buffering=0)
            fd = None  # the binary file now owns this descriptor

            data = file.read()
            complete_end = data.rfind(b"\n") + 1
            records = _decode_complete_prefix(data[:complete_end], canonical_path)
            if not records:
                raise ValueError(f"cannot resume an empty run log: {canonical_path}")

            descriptor = file.fileno()
            if complete_end != len(data):
                os.ftruncate(descriptor, complete_end)
                os.lseek(descriptor, complete_end, os.SEEK_SET)
                file.flush()
                os.fsync(descriptor)
            else:
                os.lseek(descriptor, complete_end, os.SEEK_SET)
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

    schema_version: Literal[9] = Field(alias="schema")
    run_id: str = Field(min_length=1)
    seq: int = Field(ge=0)
    kind: str = Field(min_length=1)
    payload: dict[str, Any]

    @field_validator("schema_version", mode="before")
    @classmethod
    def _exact_integer_schema(cls, value: Any) -> Any:
        """Reject bools, integral floats, and retired schemas before the literal check."""
        if type(value) is not int:
            raise ValueError("run-record schema must be an integer")
        _reject_retired_schema(value)
        return value


class _RunStartedPayload(_StrictModel):
    """Strict wire payload for :class:`RunStartedRecord`."""

    flow_fingerprint: str = Field(pattern=_SHA256_PATTERN)
    initial_pool: dict[str, Any]
    start_wiring_override: dict[str, Any]
    effective_llm_by_static_path: dict[str, dict[str, Any]]
    num_workers: int = Field(ge=0)
    memory_snapshots: list[_MemorySnapshotEnvelopePayload] | None
    learnable: bool
    memories: list[_MemorySpecPayload] | None
    memory_stacks: list[_MemoryStackPayload] | None


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


class _SourceRefPayload(_StrictModel):
    """Strict source identity and value-relative projection."""

    run_id: str = Field(min_length=1)
    execution_seq: int = Field(ge=0)
    path: list[str | int]


class _ValueRoutePayload(_StrictModel):
    """Strict recursive route shape; variant invariants validate on construction."""

    kind: Literal["literal", "source", "selected", "list", "record", "concat",
                  "repeat", "scalar", "interpolation", "opaque", "project"]
    source: _SourceRefPayload | None
    value_json: str | None
    children: list[_ValueRoutePayload]
    expression: str
    names: list[str]
    sizes: list[int]
    count: int
    content_index: int
    spans: list[list[int]]
    path: list[str | int]
    result_type: str | None
    snapshot: _ValueRoutePayload | None


class _ExecutionPositionPayload(_StrictModel):
    """Strict instantiated execution position."""

    kind: Literal["call", "map_item", "loop_body", "loop_driver"]
    static_path: list[str]
    index: int | None


class _ExecutionScopePayload(_StrictModel):
    """Strict memory-independent authored execution scope."""

    static_path: list[str]
    flow_path: list[str]
    occurrence_path: list[_ExecutionPositionPayload]
    boundary: Literal["start", "end"] | None


class _ItemCapturePayload(_StrictModel):
    """One explicitly indexed MAP seed binding."""

    index: int = Field(ge=0)
    capture: _ForwardCapturePayload


class _ForwardCapturePayload(_StrictModel):
    """Strict canonical inputs and binding metadata on an accepted record."""

    bound_inputs: dict[str, Any] | None
    route: _ValueRoutePayload
    evaluated_keys: list[str]
    items: list[_ItemCapturePayload]
    scope: _ExecutionScopePayload | None
    node_data: dict[str, Any] | None
    input_names: list[str]
    variable_names: list[str]

    @field_validator("evaluated_keys")
    @classmethod
    def _ordered_keys(cls, value: list[str]) -> list[str]:
        """Require the sorted set representation of evaluated binding keys."""
        if value != sorted(set(value)):
            raise ValueError("evaluated_keys must be sorted and unique")
        return value


class _NodeSucceededPayload(_StrictModel):
    """Strict wire payload for :class:`NodeSucceededRecord`."""

    node_id: str = Field(min_length=1)
    commit_target: str = Field(min_length=1)
    value: dict[str, Any]
    evidence: _EvaluationEvidencePayload | None
    forward_capture: _ForwardCapturePayload | None


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


class _MemoryScopePayload(_StrictModel):
    """Strict wire form of one compiled node's memory scope."""

    static_path: list[str] = Field(min_length=1)
    memories: list[str]
    occurrence_path: list[_MemoryOccurrenceSegmentPayload]

    @field_validator("static_path", "memories")
    @classmethod
    def _nonempty_parts(cls, value: list[str]) -> list[str]:
        """Reject empty path components and empty memory keys."""
        if any(not part for part in value):
            raise ValueError("memory scope components must be non-empty")
        return value


class _MemorySnapshotEnvelopePayload(_StrictModel):
    """Strict wire form of one memory snapshot."""

    memory: str = Field(min_length=1)
    snapshot: dict[str, Any]


class _DeclSitePayload(_StrictModel):
    """Strict wire form of one memory declaration site."""

    scope: list[str]
    owner: list[str] = Field(min_length=1)
    node: str | None


class _MemorySpecPayload(_StrictModel):
    """Strict wire form of one memory key-table entry."""

    key: str = Field(min_length=1)
    type: str = Field(min_length=1)
    config: dict[str, Any]
    owners: list[_DeclSitePayload] = Field(min_length=1)
    conversation_eligible: bool


class _MemoryStackPayload(_StrictModel):
    """Strict wire form of one target's memory key stack."""

    locator: list[str] = Field(min_length=1)
    memories: list[str]


class _MemorySectionPayload(_StrictModel):
    """Strict wire form of one memory's recalled section."""

    memory: str = Field(min_length=1)
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

    scope: _MemoryScopePayload | None
    bound_inputs: dict[str, Any] | None
    base_system_prompt: str | None
    prompt: str | None
    memory_sections: list[_MemorySectionPayload]
    interactions: list[_VisibleInteractionPayload]


class _MemoryMutationPayload(_StrictModel):
    """Strict wire form of one memory mutation."""

    mutation_id: str = Field(min_length=1)
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
    forward_capture: _ForwardCapturePayload | None


class _NodeExpandedPayload(_StrictModel):
    """Strict wire payload for :class:`NodeExpandedRecord`."""

    spawner_id: str = Field(min_length=1)
    seed: Any
    prune_ids: list[str]
    topology_digest: str = Field(pattern=_SHA256_PATTERN)
    evidence: _EvaluationEvidencePayload | None
    forward_capture: _ForwardCapturePayload | None

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
    forward_capture: _ForwardCapturePayload | None


class _MemoryCommitRequestedPayload(_StrictModel):
    """Strict wire payload for one atomic memory commit request."""

    memory: str = Field(min_length=1)
    commit_id: str = Field(min_length=1)
    attempt_id: int = Field(gt=0)
    mutations: list[_MemoryMutationPayload]


class _MemoryCommitSucceededPayload(_StrictModel):
    """Strict wire payload for one successful memory commit."""

    memory: str = Field(min_length=1)
    commit_id: str = Field(min_length=1)
    attempt_id: int = Field(gt=0)


class _MemoryCommitFailedPayload(_StrictModel):
    """Strict wire payload for one failed memory commit."""

    memory: str = Field(min_length=1)
    commit_id: str = Field(min_length=1)
    attempt_id: int = Field(gt=0)
    failure: _MemoryOperationFailurePayload


class _BackwardResultPayload(_StrictModel):
    """Ordinary input messages and the two independent textual channels."""

    input_feedback: dict[str, list[Any]]
    self_feedback: list[str]
    variable_feedback: dict[str, list[str]]


class _BackwardAttributionPayload(_StrictModel):
    """Reference-only ancestry coordinates for one received message."""

    sender: _SourceRefPayload | None
    input_name: str | None
    message_index: int = Field(ge=0)
    contribution_index: int | None
    parents: list[int]


class _AttributedBackwardResultPayload(_StrictModel):
    """One accepted occurrence's feedback, ancestry, and callback result."""

    source: _SourceRefPayload
    feedback: list[Any]
    attribution: list[_BackwardAttributionPayload]
    result: _BackwardResultPayload
    invoked: bool


class _MemoryProposalPayload(_StrictModel):
    """One memory's detached ordered mutation batch."""

    memory: str = Field(min_length=1)
    mutations: list[_MemoryMutationPayload]


class _LearningFailurePayload(_StrictModel):
    """One memory's learn or commit failure attribution."""

    memory: str = Field(min_length=1)
    operation: Literal["learn", "commit"]
    failure: _MemoryOperationFailurePayload


class _PolicyCommitOutcomePayload(_StrictModel):
    """One memory's commit disposition without inferred attempt completion."""

    memory: str = Field(min_length=1)
    status: Literal["unrequested", "unknown", "succeeded", "failed"]
    commit_id: str | None
    failure: _MemoryOperationFailurePayload | None


class _LearningResultPayload(_StrictModel):
    """Ordered failures and per-memory commit dispositions."""

    status: Literal["succeeded", "failed", "incomplete"]
    failures: list[_LearningFailurePayload]
    policies: list[_PolicyCommitOutcomePayload]


class _BackwardFailureCausePayload(_FailurePayload):
    """Strict semantic cause with the diagnostic carrier's optional node identity."""

    node_id: str | None


class _BackwardFailurePayload(_StrictModel):
    """A backward-operation diagnostic and its optional lower-layer cause."""

    stage: str = Field(min_length=1)
    code: str = Field(min_length=1)
    message: str
    scope: _ExecutionScopePayload | None
    run_id: str | None
    node_id: str | None
    node_started_seq: int | None
    memory: str | None
    cause: _BackwardFailureCausePayload | _MemoryOperationFailurePayload | None
    cause_type: str | None
    traceback: str | None


class _BackwardOutcomePayload(_StrictModel):
    """The settled backward disposition, separate from forward status."""

    status: Literal["succeeded", "failed"]
    feedback: _BackwardResultPayload | None
    failure: _BackwardFailurePayload | None
    learning: _LearningResultPayload | None


class _BackwardStartedPayload(_StrictModel):
    """One explicit attempt identity and original ordinary root feedback."""

    attempt_id: int = Field(gt=0)
    feedback: Any


class _BackwardReadyPayload(_StrictModel):
    """Completed compute results and ordered memory proposals."""

    attempt_id: int = Field(gt=0)
    root_result: _BackwardResultPayload
    attributed_results: list[_AttributedBackwardResultPayload]
    proposals: list[_MemoryProposalPayload]


class _BackwardFinishedPayload(_StrictModel):
    """An attempt outcome and optional completed attribution inventory."""

    attempt_id: int = Field(gt=0)
    outcome: _BackwardOutcomePayload
    attributed_results: list[_AttributedBackwardResultPayload]


_KIND_TO_MODEL: dict[str, type[_StrictModel]] = {
    "run_started": _RunStartedPayload,
    "run_resumed": _RunResumedPayload,
    "node_started": _NodeStartedPayload,
    "node_succeeded": _NodeSucceededPayload,
    "node_routed": _NodeRoutedPayload,
    "node_expanded": _NodeExpandedPayload,
    "node_failed": _NodeFailedPayload,
    "memory_commit_requested": _MemoryCommitRequestedPayload,
    "memory_commit_succeeded": _MemoryCommitSucceededPayload,
    "memory_commit_failed": _MemoryCommitFailedPayload,
    "backward_started": _BackwardStartedPayload,
    "backward_ready": _BackwardReadyPayload,
    "backward_finished": _BackwardFinishedPayload,
}

_TYPE_TO_KIND: dict[type[Any], str] = {
    RunStartedRecord: "run_started",
    RunResumedRecord: "run_resumed",
    NodeStartedRecord: "node_started",
    NodeSucceededRecord: "node_succeeded",
    NodeRoutedRecord: "node_routed",
    NodeExpandedRecord: "node_expanded",
    NodeFailedRecord: "node_failed",
    MemoryCommitRequestedRecord: "memory_commit_requested",
    MemoryCommitSucceededRecord: "memory_commit_succeeded",
    MemoryCommitFailedRecord: "memory_commit_failed",
    BackwardStartedRecord: "backward_started",
    BackwardReadyRecord: "backward_ready",
    BackwardFinishedRecord: "backward_finished",
}

_RECORD_KINDS = frozenset({
    "run_started", "run_resumed", "node_started", "node_succeeded", "node_routed",
    "node_expanded", "node_failed", "memory_commit_requested",
    "memory_commit_succeeded", "memory_commit_failed", "backward_started",
    "backward_ready", "backward_finished",
})


def _reject_retired_schema(schema: int) -> None:
    """Name a retired schema instead of reporting it as an unknown version."""
    if schema in _RETIRED_SCHEMAS:
        raise ValueError(
            f"run log schema {schema} predates schema {SCHEMA_VERSION} and is not supported"
        )


def _require_schema_kind(schema: int, kind: str) -> None:
    """Reject retired or unsupported schemas and kinds outside the closed sum."""
    if type(schema) is not int:
        raise ValueError(f"unsupported run-record schema: {schema!r}")
    _reject_retired_schema(schema)
    if schema != SCHEMA_VERSION:
        raise ValueError(f"unsupported run-record schema: {schema!r}")
    if kind not in _RECORD_KINDS:
        raise ValueError(f"run-record kind {kind!r} is unsupported in schema {schema}")


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
            The live value does not satisfy ``declared``, contains a NaN, infinity, or
            complex number, or, with no ``declared`` type, is not made of JSON values
            and `FileRef`s; ``path`` locates the rejected value. Without ``declared``
            the message reads ``output[ at <path>]: <reason>``.
        `ValueError`:
            The typed value contains data that cannot be represented as canonical JSON.
    """
    try:
        reject_nonfinite(value, "output")
        if declared is None:
            # A built value comes from decoded wire data, which is always JSON or files.
            if not isinstance(value, TypedValue):
                reject_non_json(value, "output", files=True)
            typed = build_value(value)
        else:
            typed = build_value_as(declared, value)
        _reject_unordered_containers(typed)
        dumped = ANY_VALUE_ADAPTER.dump_python(typed, mode="json", warnings="error")
        wire = _copy_json(dumped)
        if not isinstance(wire, dict):  # defensive: every AnyValue model is an object
            raise ValueError("typed value did not encode to a JSON object")
        decoded = _decode_typed(wire)
        return wire, decoded
    except RecursionError as exc:
        raise ValueError("canonical data nesting exceeds the supported depth") from exc


def canonical_seed(value: Any) -> tuple[Any, Any]:
    """Canonicalize a supported node value through Pydantic's JSON mode.

    Args:
        value (`Any`):
            Builder input or invocation data. Tuples, dates, datetimes, and
            Pydantic models such as `FileRef` become their JSON representations.

    Returns:
        `tuple[Any, Any]`:
            Independent canonical JSON wire and decoded values.  Live expansion
            building must consume the decoded second value.

    Raises:
        `TypeCheckError`:
            The value contains a NaN, infinity, or complex number; checked before
            JSON mode can turn a NaN or infinity into ``null``. ``path`` locates the
            rejected number.
        `ValueError`:
            The value is not canonicalizable.
    """
    try:
        _reject_unordered_containers(value)
        reject_nonfinite(value, "value")
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
        `TypeCheckError`:
            A stored or system value contains a NaN, infinity, or complex number.
        `ValueError`:
            A nested typed value cannot cross the JSON boundary.
    """
    if not isinstance(pool, VariablePool):
        raise TypeError("encode_pool expects a VariablePool")
    _reject_unordered_containers(pool)
    reject_nonfinite(pool, "variable pool")
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
    _require_schema_kind(envelope.schema, envelope.kind)
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
        raw = json_codec.loads(
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
    _require_schema_kind(envelope.schema_version, envelope.kind)
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


def _scope_payload(scope: MemoryScope) -> dict[str, Any]:
    """Encode one exact compiled memory scope."""
    if type(scope) is not MemoryScope:
        raise TypeError("evidence scope must be a MemoryScope")
    MemoryScope(scope.static_path, scope.memories, scope.occurrence_path)
    return {
        "static_path": list(scope.static_path),
        "memories": list(scope.memories),
        "occurrence_path": [
            _occurrence_segment_payload(segment)
            for segment in scope.occurrence_path
        ],
    }


def _scope_from_payload(payload: dict[str, Any]) -> MemoryScope:
    """Decode one already-validated compiled memory scope."""
    return MemoryScope(
        static_path=tuple(payload["static_path"]),
        memories=tuple(payload["memories"]),
        occurrence_path=tuple(
            _occurrence_segment_from_payload(segment)
            for segment in payload["occurrence_path"]
        ),
    )


def _snapshot_payload(snapshot: MemorySnapshotEnvelope) -> dict[str, Any]:
    """Encode one exact named memory snapshot envelope."""
    if type(snapshot) is not MemorySnapshotEnvelope:
        raise TypeError("memory snapshot must be a MemorySnapshotEnvelope")
    MemorySnapshotEnvelope(snapshot.memory, snapshot.snapshot)
    return {
        "memory": snapshot.memory,
        "snapshot": snapshot.snapshot,
    }


def _snapshot_from_payload(payload: dict[str, Any]) -> MemorySnapshotEnvelope:
    """Decode one already-validated named memory snapshot envelope."""
    return MemorySnapshotEnvelope(
        memory=payload["memory"],
        snapshot=_copy_json(payload["snapshot"]),
    )


def _memory_spec_payload(spec: MemorySpec) -> dict[str, Any]:
    """Encode one exact memory key-table entry."""
    if type(spec) is not MemorySpec:
        raise TypeError("memory table item must be a MemorySpec")
    return {
        "key": spec.key,
        "type": spec.type,
        "config": spec.config,
        "owners": [
            {"scope": list(site.scope), "owner": list(site.owner), "node": site.node}
            for site in spec.owners
        ],
        "conversation_eligible": spec.conversation_eligible,
    }


def _memory_spec_from_payload(payload: dict[str, Any]) -> MemorySpec:
    """Decode one already-validated memory key-table entry."""
    return MemorySpec(
        payload["key"],
        payload["type"],
        _copy_json(payload["config"]),
        tuple(
            DeclSite(tuple(site["scope"]), tuple(site["owner"]), site["node"])
            for site in payload["owners"]
        ),
        payload["conversation_eligible"],
    )


def _memory_stack_payload(stack: MemoryStack) -> dict[str, Any]:
    """Encode one exact target key stack."""
    if type(stack) is not MemoryStack:
        raise TypeError("memory stack item must be a MemoryStack")
    return {"locator": list(stack.locator), "memories": list(stack.memories)}


def _memory_stack_from_payload(payload: dict[str, Any]) -> MemoryStack:
    """Decode one already-validated target key stack."""
    return MemoryStack(tuple(payload["locator"]), tuple(payload["memories"]))


def _memory_section_payload(section: MemorySection) -> dict[str, Any]:
    """Encode one exact recalled-memory section."""
    if type(section) is not MemorySection:
        raise TypeError("memory section must be a MemorySection")
    MemorySection(section.memory, section.text)
    return {"memory": section.memory, "text": section.text}


def _memory_section_from_payload(payload: dict[str, Any]) -> MemorySection:
    """Decode one already-validated recalled-memory section."""
    return MemorySection(memory=payload["memory"], text=payload["text"])


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


def _validate_optional_capture(capture: ForwardCapture | None) -> None:
    """Revalidate frozen capture carriers before they cross durable authority."""
    if capture is None:
        return
    if type(capture) is not ForwardCapture:
        raise TypeError("forward_capture must be a ForwardCapture or None")
    ForwardCapture(capture.bound_inputs, capture.route, capture.evaluated_keys,
                   capture.items, capture.scope, capture.node_data,
                   input_names=capture.input_names, variable_names=capture.variable_names)


def _source_payload(source: SourceRef | None) -> dict[str, Any] | None:
    """Encode a revalidated source reference without copying its produced value."""
    if source is None:
        return None
    _validate_source(source)
    return {"run_id": source.run_id, "execution_seq": source.execution_seq, "path": list(source.path)}


def _source_from_payload(payload: dict[str, Any] | None) -> SourceRef | None:
    """Decode an already strictly shaped source reference."""
    if payload is None:
        return None
    source = SourceRef(payload["run_id"], payload["execution_seq"], tuple(payload["path"]))
    _validate_source(source)
    return source


def _execution_scope_payload(scope: ExecutionScope | None) -> dict[str, Any] | None:
    """Encode the shared memory-independent execution scope."""
    if scope is None:
        return None
    if type(scope) is not ExecutionScope:
        raise TypeError("scope must be an exact ExecutionScope")
    ExecutionScope.__post_init__(scope)
    return {
        "static_path": list(scope.static_path), "flow_path": list(scope.flow_path),
        "occurrence_path": [{"kind": position.kind, "static_path": list(position.static_path),
                             "index": position.index} for position in scope.occurrence_path],
        "boundary": scope.boundary,
    }


def _execution_scope_from_payload(payload: dict[str, Any] | None) -> ExecutionScope | None:
    """Decode a strictly shaped authored and occurrence scope."""
    if payload is None:
        return None
    return ExecutionScope(
        tuple(payload["static_path"]), tuple(payload["flow_path"]),
        tuple(ExecutionPosition(position["kind"], tuple(position["static_path"]), position["index"])
              for position in payload["occurrence_path"]), payload["boundary"],
    )


def _route_payload(route: ValueRoute) -> dict[str, Any]:
    """Encode an exact immutable route without accepting normalized forgeries."""
    _validate_route(route)
    return {
        "kind": route.kind,
        "source": _source_payload(route.source),
        "value_json": route.value_json,
        "children": [_route_payload(child) for child in route.children],
        "expression": route.expression,
        "names": list(route.names),
        "sizes": list(route.sizes),
        "count": route.count,
        "content_index": route.content_index,
        "spans": [list(span) for span in route.spans],
        "path": list(route.path),
        "result_type": route.result_type,
        "snapshot": None if route.snapshot is None else _route_payload(route.snapshot),
    }


def _route_from_payload(payload: dict[str, Any]) -> ValueRoute:
    """Decode a strictly shaped route, then check its exact variant invariants."""
    source = payload["source"]
    route = ValueRoute(
        kind=payload["kind"],
        source=_source_from_payload(source),
        value_json=payload["value_json"],
        children=tuple(_route_from_payload(child) for child in payload["children"]),
        expression=payload["expression"], names=tuple(payload["names"]),
        sizes=tuple(payload["sizes"]), count=payload["count"],
        content_index=payload["content_index"],
        spans=tuple(tuple(span) for span in payload["spans"]),
        path=tuple(payload["path"]), result_type=payload["result_type"],
        snapshot=None if payload["snapshot"] is None else _route_from_payload(payload["snapshot"]),
    )
    _validate_route(route)
    return route


def _capture_payload(capture: ForwardCapture | None) -> dict[str, Any] | None:
    """Encode canonical binding data without duplicating accepted outcomes."""
    if capture is None:
        return None
    _validate_optional_capture(capture)
    scope = capture.scope
    return {
        "bound_inputs": capture.bound_inputs,
        "route": _route_payload(capture.route),
        "evaluated_keys": sorted(capture.evaluated_keys),
        "items": [{"index": index, "capture": _capture_payload(item)} for index, item in capture.items],
        "scope": _execution_scope_payload(scope),
        "node_data": capture.node_data,
        "input_names": list(capture.input_names),
        "variable_names": list(capture.variable_names),
    }


def _capture_from_payload(payload: dict[str, Any] | None) -> ForwardCapture | None:
    """Decode validated capture data and preserve unavailable-input markers."""
    if payload is None:
        return None
    scope = payload["scope"]
    return ForwardCapture(
        payload["bound_inputs"], _route_from_payload(payload["route"]),
        frozenset(payload["evaluated_keys"]),
        tuple((item["index"], _capture_from_payload(item["capture"])) for item in payload["items"]),
        _execution_scope_from_payload(scope),
        payload["node_data"],
        input_names=tuple(payload["input_names"]),
        variable_names=tuple(payload["variable_names"]),
    )


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
        "scope": None if evidence.scope is None else _scope_payload(evidence.scope),
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
        scope=None if payload["scope"] is None else _scope_from_payload(payload["scope"]),
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
    """Encode one exact memory mutation."""
    if type(mutation) is not MemoryMutation:
        raise TypeError("mutation must be a MemoryMutation")
    MemoryMutation(mutation.mutation_id, mutation.payload)
    return {"mutation_id": mutation.mutation_id, "payload": mutation.payload}


def _mutation_from_payload(payload: dict[str, Any]) -> MemoryMutation:
    """Decode one already-validated memory mutation."""
    return MemoryMutation(
        mutation_id=payload["mutation_id"],
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


def _backward_result_payload(result: BackwardResult) -> dict[str, Any]:
    """Encode the three explicit channels, preserving independent batch order."""
    checked = _copy_backward_result(result)
    return {
        "input_feedback": {name: list(batch) for name, batch in checked.input_feedback.items()},
        "self_feedback": list(checked.self_feedback),
        "variable_feedback": {name: list(batch) for name, batch in checked.variable_feedback.items()},
    }


def _backward_result_from_payload(payload: dict[str, Any]) -> BackwardResult:
    """Decode ordinary messages without the narrower memory JSON gate."""
    return BackwardResult(
        {name: tuple(batch) for name, batch in payload["input_feedback"].items()},
        tuple(payload["self_feedback"]),
        {name: tuple(batch) for name, batch in payload["variable_feedback"].items()},
    )


def _attribution_payload(item: BackwardAttribution) -> dict[str, Any]:
    """Encode exact transfer coordinates without materialized downstream context."""
    if type(item) is not BackwardAttribution:
        raise TypeError("attribution must be an exact BackwardAttribution")
    BackwardAttribution.__post_init__(item)
    return {
        "sender": _source_payload(item.sender), "input_name": item.input_name,
        "message_index": item.message_index, "contribution_index": item.contribution_index,
        "parents": list(item.parents),
    }


def _attribution_from_payload(payload: dict[str, Any]) -> BackwardAttribution:
    """Decode local ancestry coordinates; trace agreement belongs to reduction."""
    return BackwardAttribution(
        _source_from_payload(payload["sender"]), payload["input_name"],
        payload["message_index"], payload["contribution_index"], tuple(payload["parents"]),
    )


def _attributed_result_payload(item: AttributedBackwardResult) -> dict[str, Any]:
    """Encode one detached occurrence result and its reference-only attribution."""
    checked = _copy_attributed_results((item,))[0]
    return {
        "source": _source_payload(checked.source), "feedback": list(checked.feedback),
        "attribution": [_attribution_payload(a) for a in checked.attribution],
        "result": _backward_result_payload(checked.result), "invoked": checked.invoked,
    }


def _attributed_result_from_payload(payload: dict[str, Any]) -> AttributedBackwardResult:
    """Decode one result without reconstructing an invocation or route."""
    return AttributedBackwardResult(
        _source_from_payload(payload["source"]), tuple(payload["feedback"]),
        tuple(_attribution_from_payload(a) for a in payload["attribution"]),
        _backward_result_from_payload(payload["result"]), payload["invoked"],
    )


def _memory_proposal_payload(proposal: MemoryProposal) -> dict[str, Any]:
    """Encode one revalidated memory proposal using the mutation codec."""
    if type(proposal) is not MemoryProposal:
        raise TypeError("proposal must be an exact MemoryProposal")
    checked = replace(proposal)
    return {
        "memory": checked.memory,
        "mutations": [_mutation_payload(m) for m in checked.mutations],
    }


def _memory_proposal_from_payload(payload: dict[str, Any]) -> MemoryProposal:
    """Decode and locally validate one memory's mutation order."""
    return MemoryProposal(
        payload["memory"],
        tuple(_mutation_from_payload(m) for m in payload["mutations"]),
    )


def _learning_result_payload(result: LearningResult) -> dict[str, Any]:
    """Encode independent failure attribution and ordered per-memory dispositions."""
    if type(result) is not LearningResult:
        raise TypeError("learning must be an exact LearningResult")
    checked = replace(result)
    return {
        "status": checked.status,
        "failures": [
            {
                "memory": item.memory,
                "operation": item.operation,
                "failure": _memory_failure_payload(item.failure),
            }
            for item in checked.failures
        ],
        "policies": [
            {
                "memory": item.memory,
                "status": item.status,
                "commit_id": item.commit_id,
                "failure": None if item.failure is None else _memory_failure_payload(item.failure),
            }
            for item in checked.policies
        ],
    }


def _learning_result_from_payload(payload: dict[str, Any]) -> LearningResult:
    """Decode leaf dispositions without inferring whole-attempt completion."""
    return LearningResult(
        payload["status"],
        tuple(
            LearningFailure(
                item["memory"], item["operation"],
                _memory_failure_from_payload(item["failure"]),
            )
            for item in payload["failures"]
        ),
        tuple(
            PolicyCommitOutcome(
                item["memory"], item["status"], item["commit_id"],
                None if item["failure"] is None else _memory_failure_from_payload(item["failure"]),
            )
            for item in payload["policies"]
        ),
    )


def _backward_failure_payload(failure: BackwardFailure) -> dict[str, Any]:
    """Encode one attributed diagnostic with an explicit closed lower cause."""
    if type(failure) is not BackwardFailure:
        raise TypeError("failure must be an exact BackwardFailure")
    checked = replace(failure)
    cause = checked.cause
    cause_payload = None
    if type(cause) is Failure:
        cause_payload = _failure_payload(cause, carrier="BackwardFailure")
    elif cause is not None:
        cause_payload = _memory_failure_payload(cause)
    return {
        "stage": checked.stage, "code": checked.code, "message": checked.message,
        "scope": _execution_scope_payload(checked.scope), "run_id": checked.run_id,
        "node_id": checked.node_id, "node_started_seq": checked.node_started_seq,
        "memory": checked.memory,
        "cause": cause_payload,
        "cause_type": checked.cause_type, "traceback": checked.traceback,
    }


def _backward_failure_from_payload(payload: dict[str, Any]) -> BackwardFailure:
    """Decode the two disjoint strict cause payload shapes without inference."""
    cause = payload["cause"]
    decoded_cause = None
    if cause is not None:
        decoded_cause = (
            _memory_failure_from_payload(cause) if "stage" in cause else _failure_from_payload(cause)
        )
    return BackwardFailure(
        stage=payload["stage"], code=payload["code"], message=payload["message"],
        scope=_execution_scope_from_payload(payload["scope"]), run_id=payload["run_id"],
        node_id=payload["node_id"], node_started_seq=payload["node_started_seq"],
        memory=payload["memory"],
        cause=decoded_cause,
        cause_type=payload["cause_type"], traceback=payload["traceback"],
    )


def _backward_outcome_payload(outcome: BackwardOutcome) -> dict[str, Any]:
    """Encode the independent backward outcome and optional learning result."""
    if type(outcome) is not BackwardOutcome:
        raise TypeError("outcome must be an exact BackwardOutcome")
    checked = replace(outcome)
    return {
        "status": checked.status,
        "feedback": None if checked.feedback is None else _backward_result_payload(checked.feedback),
        "failure": None if checked.failure is None else _backward_failure_payload(checked.failure),
        "learning": None if checked.learning is None else _learning_result_payload(checked.learning),
    }


def _backward_outcome_from_payload(payload: dict[str, Any]) -> BackwardOutcome:
    """Decode recorded outcome data without fabricating missing computation."""
    return BackwardOutcome(
        payload["status"],
        None if payload["feedback"] is None else _backward_result_from_payload(payload["feedback"]),
        None if payload["failure"] is None else _backward_failure_from_payload(payload["failure"]),
        None if payload["learning"] is None else _learning_result_from_payload(payload["learning"]),
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
            record.memory_snapshots,
            record.learnable,
            record.memories,
            record.memory_stacks,
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
            record.forward_capture,
        )
        return
    if type(record) is NodeRoutedRecord:
        NodeRoutedRecord(
            record.node_id,
            record.handle,
            record.terminal_failure,
            record.evidence,
            record.forward_capture,
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
            record.forward_capture,
        )
        return
    if type(record) is NodeFailedRecord:
        NodeFailedRecord(record.failure, record.evidence, record.forward_capture)
        _failure_payload(record.failure, carrier="NodeFailedRecord")
        return
    if type(record) is MemoryCommitRequestedRecord:
        MemoryCommitRequestedRecord(
            record.memory,
            record.commit_id,
            record.mutations,
            attempt_id=record.attempt_id,
        )
        return
    if type(record) is MemoryCommitSucceededRecord:
        MemoryCommitSucceededRecord(record.memory, record.commit_id, attempt_id=record.attempt_id)
        return
    if type(record) is MemoryCommitFailedRecord:
        MemoryCommitFailedRecord(
            record.memory,
            record.commit_id,
            record.failure,
            attempt_id=record.attempt_id,
        )
        return
    if type(record) is BackwardStartedRecord:
        BackwardStartedRecord(record.attempt_id, record.feedback)
        return
    if type(record) is BackwardReadyRecord:
        BackwardReadyRecord(record.attempt_id, record.root_result, record.attributed_results, record.proposals)
        return
    if type(record) is BackwardFinishedRecord:
        BackwardFinishedRecord(record.attempt_id, record.outcome, record.attributed_results)
        return
    raise TypeError(f"unsupported run record: {type(record).__name__}")


def _payload_for_record(record: RunRecord) -> dict[str, Any]:
    """Build and validate the canonical JSON payload for one record variant."""
    _revalidate_run_record(record)
    kind = _kind_for_record(record)
    if isinstance(record, RunStartedRecord):
        _validate_run_memory_configuration(
            record.memories, record.memory_stacks, record.memory_snapshots, record.learnable,
        )
        raw = {
            "flow_fingerprint": record.flow_fingerprint,
            "initial_pool": encode_pool(decode_pool(record.initial_pool)),
            "start_wiring_override": canonical_seed(record.start_wiring_override)[0],
            "effective_llm_by_static_path": canonical_seed(
                record.effective_llm_by_static_path
            )[0],
            "num_workers": record.num_workers,
            "memory_snapshots": (
                None
                if record.memory_snapshots is None
                else [
                    _snapshot_payload(snapshot)
                    for snapshot in record.memory_snapshots
                ]
            ),
            "learnable": record.learnable,
            "memories": (
                None
                if record.memories is None
                else [_memory_spec_payload(spec) for spec in record.memories]
            ),
            "memory_stacks": (
                None
                if record.memory_stacks is None
                else [_memory_stack_payload(stack) for stack in record.memory_stacks]
            ),
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
            "forward_capture": _capture_payload(record.forward_capture),
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
            "forward_capture": _capture_payload(record.forward_capture),
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
            "forward_capture": _capture_payload(record.forward_capture),
        }
    elif isinstance(record, NodeFailedRecord):
        raw = {
            "failure": _failure_payload(
                record.failure,
                carrier="NodeFailedRecord",
            ),
            "evidence": _evidence_payload(record.evidence),
            "forward_capture": _capture_payload(record.forward_capture),
        }
    elif isinstance(record, MemoryCommitRequestedRecord):
        raw = {
            "memory": record.memory,
            "commit_id": record.commit_id,
            "attempt_id": record.attempt_id,
            "mutations": [
                _mutation_payload(mutation) for mutation in record.mutations
            ],
        }
    elif isinstance(record, MemoryCommitSucceededRecord):
        raw = {
            "memory": record.memory,
            "commit_id": record.commit_id,
            "attempt_id": record.attempt_id,
        }
    elif isinstance(record, MemoryCommitFailedRecord):
        raw = {
            "memory": record.memory,
            "commit_id": record.commit_id,
            "attempt_id": record.attempt_id,
            "failure": _memory_failure_payload(record.failure),
        }
    elif isinstance(record, BackwardStartedRecord):
        raw = {"attempt_id": record.attempt_id, "feedback": record.feedback}
    elif isinstance(record, BackwardReadyRecord):
        raw = {
            "attempt_id": record.attempt_id,
            "root_result": _backward_result_payload(record.root_result),
            "attributed_results": [_attributed_result_payload(item) for item in record.attributed_results],
            "proposals": [_memory_proposal_payload(proposal) for proposal in record.proposals],
        }
    elif isinstance(record, BackwardFinishedRecord):
        raw = {
            "attempt_id": record.attempt_id,
            "outcome": _backward_outcome_payload(record.outcome),
            "attributed_results": [_attributed_result_payload(item) for item in record.attributed_results],
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
            memory_snapshots=(
                None
                if values["memory_snapshots"] is None
                else tuple(
                    _snapshot_from_payload(snapshot)
                    for snapshot in values["memory_snapshots"]
                )
            ),
            learnable=values["learnable"],
            memories=(
                None
                if values["memories"] is None
                else tuple(_memory_spec_from_payload(spec) for spec in values["memories"])
            ),
            memory_stacks=(
                None
                if values["memory_stacks"] is None
                else tuple(
                    _memory_stack_from_payload(stack) for stack in values["memory_stacks"]
                )
            ),
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
            forward_capture=_capture_from_payload(values["forward_capture"]),
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
            forward_capture=_capture_from_payload(values["forward_capture"]),
        )
    if kind == "node_expanded":
        seed = canonical_seed(values["seed"])[0]
        return NodeExpandedRecord(
            spawner_id=values["spawner_id"],
            seed=seed,
            prune_ids=tuple(values["prune_ids"]),
            topology_digest=values["topology_digest"],
            evidence=_evidence_from_payload(values["evidence"]),
            forward_capture=_capture_from_payload(values["forward_capture"]),
        )
    if kind == "node_failed":
        failure_payload = values["failure"]
        return NodeFailedRecord(
            failure=_failure_from_payload(failure_payload),
            evidence=_evidence_from_payload(values["evidence"]),
            forward_capture=_capture_from_payload(values["forward_capture"]),
        )
    if kind == "memory_commit_requested":
        return MemoryCommitRequestedRecord(
            memory=values["memory"],
            commit_id=values["commit_id"],
            attempt_id=values["attempt_id"],
            mutations=tuple(
                _mutation_from_payload(mutation)
                for mutation in values["mutations"]
            ),
        )
    if kind == "memory_commit_succeeded":
        return MemoryCommitSucceededRecord(
            memory=values["memory"],
            commit_id=values["commit_id"],
            attempt_id=values["attempt_id"],
        )
    if kind == "memory_commit_failed":
        return MemoryCommitFailedRecord(
            memory=values["memory"],
            commit_id=values["commit_id"],
            attempt_id=values["attempt_id"],
            failure=_memory_failure_from_payload(values["failure"]),
        )
    if kind == "backward_started":
        return BackwardStartedRecord(values["attempt_id"], values["feedback"])
    if kind == "backward_ready":
        return BackwardReadyRecord(
            values["attempt_id"], _backward_result_from_payload(values["root_result"]),
            tuple(_attributed_result_from_payload(item) for item in values["attributed_results"]),
            tuple(_memory_proposal_from_payload(item) for item in values["proposals"]),
        )
    if kind == "backward_finished":
        return BackwardFinishedRecord(
            values["attempt_id"], _backward_outcome_from_payload(values["outcome"]),
            tuple(_attributed_result_from_payload(item) for item in values["attributed_results"]),
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


def decode_typed_json(data: bytes | str) -> TypedValue:
    """Strictly decode a typed value from JSON that may hold ints of any size.

    Pydantic's JSON parser rejects ints past Python's digit limit as invalid
    JSON; those documents are read through `json_codec` and validated as Python
    data, which keeps object key order.

    Args:
        data (`bytes | str`):
            One typed-value JSON document, as written by `ANY_VALUE_ADAPTER.dump_json`.

    Returns:
        `TypedValue`:
            The strictly validated typed value.

    Raises:
        `ValidationError`:
            The document is not valid JSON or not a valid typed value.
    """
    try:
        # Parsing and validating in one pass avoids building Python objects
        # first; only documents with long ints need the slower path below.
        return ANY_VALUE_ADAPTER.validate_json(data, strict=True)
    except ValidationError as exc:
        if any(error["type"] != "json_invalid" for error in exc.errors()):
            raise
        try:
            parsed = json_codec.loads(data)
        except (ValueError, RecursionError):
            raise exc from None
    return ANY_VALUE_ADAPTER.validate_python(parsed, strict=True)


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
                _reject_unordered_containers(getattr(value, field_name), _active=active)
            _reject_unordered_containers(value.model_extra, _active=active)
            return
        if is_dataclass(value) and not isinstance(value, type):
            for field in fields(value):
                _reject_unordered_containers(getattr(value, field.name), _active=active)
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
        return json_codec.dumps(
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


def _decode_complete_prefix(data: bytes, path: Path) -> list[RecordEnvelope]:
    """Validate every newline-terminated record and its shared identity.

    Errors name ``path``, the 1-based line, and the expected seq so a damaged
    log can be located without bisecting it.
    """
    if not data:
        return []
    lines = data.split(b"\n")
    if lines[-1] != b"":
        raise ValueError(f"run log {path} is not newline terminated")

    records: list[RecordEnvelope] = []
    run_id: str | None = None
    for expected_sequence, line in enumerate(lines[:-1]):
        where = f"run log {path} line {expected_sequence + 1} (seq {expected_sequence})"
        try:
            envelope = decode_record(line)
        except ValueError as exc:
            raise ValueError(f"{where}: {exc}") from exc
        if envelope.seq != expected_sequence:
            raise ValueError(
                f"{where}: run-log sequence mismatch: "
                f"expected {expected_sequence}, found {envelope.seq}"
            )
        if run_id is None:
            run_id = envelope.run_id
        elif envelope.run_id != run_id:
            raise ValueError(
                f"{where}: run-log run_id mismatch: expected {run_id!r}, "
                f"found {envelope.run_id!r}"
            )
        records.append(envelope)
    return records


def _lock_exclusive(descriptor: int, path: Path) -> None:
    """Take the nonblocking exclusive lock, naming ``path`` when another holder has it.

    The raised ``BlockingIOError`` keeps the original ``errno`` so callers that
    match on ``EAGAIN``/``EWOULDBLOCK`` still work.
    """
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        error = BlockingIOError(
            f"run log {path} is locked by another writer (is the run still active?)"
        )
        error.errno = exc.errno
        raise error from exc


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
    "BackwardStartedRecord",
    "BackwardReadyRecord",
    "BackwardFinishedRecord",
    "JsonlRunLog",
    "MemoryRunLog",
    "MemoryCommitFailedRecord",
    "MemoryCommitRequestedRecord",
    "MemoryCommitSucceededRecord",
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
