"""Run-scoped orchestration for host-injected memory implementations.

The compile layer supplies the recorded memory table, one ``MemorySpec`` per
key, and each key's optional authored origin. This module validates the host
registry, opens exactly one session per key, captures the public recovery
snapshots, and admits ordered memory context for the evaluator. It does not
resolve YAML, inspect node kinds, or construct a concrete memory backend. The
explicit backward path stages one proposal per key and performs accepted
commit requests separately; the attempt driver owns their records and ordering.
"""

from __future__ import annotations

from asyncio import CancelledError as AsyncCancelledError
from collections.abc import Iterator, Mapping
from concurrent.futures import CancelledError as FutureCancelledError
from contextlib import contextmanager, nullcontext
from typing import Any

from yian.diagnostics import OriginRef
from yian.memory import (
    CheckStopped,
    ConversationRequest,
    LearnContext,
    Memory,
    MemoryContent,
    MemoryContext,
    MemoryContextRequest,
    MemoryOperationError,
    MemoryOperationFailure,
    MemoryRecallError,
    MemoryScope,
    MemorySection,
    MemorySession,
    MemorySetupError,
    MemorySnapshotEnvelope,
    MemorySpec,
)
from yian.memory.contracts import _validate_mutations
from yian.nodes.base import _CapabilityFault, _HostStop
from yian.runtime.contracts import MemoryProposal
from yian.runtime.log import MemoryCommitRequestedRecord


def _require_name(value: object, name: str) -> str:
    """Return one exact non-empty UTF-8 registry identity."""
    if type(value) is not str:
        raise TypeError(f"{name} must be a string")
    if not value:
        raise ValueError(f"{name} must be non-empty")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{name} must be valid UTF-8") from exc
    return value


def _revalidate_origin(value: OriginRef | None) -> OriginRef | None:
    """Replay the source-neutral origin contract after possible mutation."""
    if value is None:
        return None
    if not isinstance(value, OriginRef):
        raise TypeError("memory origins must be OriginRef values or None")
    OriginRef(value.key)
    return value


def _validate_memories(memories: tuple[MemorySpec, ...]) -> tuple[str, ...]:
    """Require exact revalidated specs in sorted unique key order; return the keys."""
    if type(memories) is not tuple or any(type(spec) is not MemorySpec for spec in memories):
        raise TypeError("memories must be a tuple of MemorySpec values")
    for spec in memories:
        MemorySpec(spec.key, spec.type, spec.config, spec.owners, spec.conversation_eligible)
    keys = tuple(spec.key for spec in memories)
    if keys != tuple(sorted(set(keys))):
        raise ValueError("memories must be sorted by key and unique")
    return keys


def _validate_origins(
    keys: tuple[str, ...],
    origins: dict[str, OriginRef | None],
) -> dict[str, OriginRef | None]:
    """Return a detached exact origin map with one entry per memory key."""
    if type(origins) is not dict:
        raise TypeError("memory origins must be a dictionary")
    if any(type(key) is not str for key in origins):
        raise TypeError("memory origin keys must be strings")
    if set(origins) != set(keys):
        raise ValueError("memory origins must exactly match the memory keys")
    return {key: _revalidate_origin(origins[key]) for key in keys}


def _validate_registry(memory: Mapping[str, Memory]) -> dict[str, Memory]:
    """Return a detached exact registry and reject cross-name object aliases."""
    if not isinstance(memory, Mapping):
        raise TypeError("memory must be a mapping of memory keys to backends, or None")
    result: dict[str, Memory] = {}
    identities: dict[int, str] = {}
    for raw_name, implementation in memory.items():
        name = _require_name(raw_name, "memory name")
        if not isinstance(implementation, Memory):
            raise TypeError(f"memory {name!r} must implement the Memory protocol")
        identity = id(implementation)
        prior = identities.get(identity)
        if prior is not None:
            raise ValueError(
                "one memory object cannot be registered under multiple names: "
                f"{prior!r} and {name!r}"
            )
        identities[identity] = name
        result[name] = implementation
    return result


def _validate_snapshots(
    snapshots: tuple[MemorySnapshotEnvelope, ...] | None,
    keys: tuple[str, ...],
) -> dict[str, dict[str, Any] | None]:
    """Return the per-key restore payload, or fresh ``None`` payloads."""
    if snapshots is None:
        return {name: None for name in keys}
    if type(snapshots) is not tuple:
        raise TypeError("memory snapshots must be a tuple or None")
    restored: dict[str, dict[str, Any] | None] = {}
    supplied: list[str] = []
    for envelope in snapshots:
        if type(envelope) is not MemorySnapshotEnvelope:
            raise TypeError(
                "memory snapshots must contain MemorySnapshotEnvelope values"
            )
        canonical = MemorySnapshotEnvelope(
            envelope.memory,
            envelope.snapshot,
        )
        if canonical.memory in restored:
            raise ValueError("memory snapshot keys must be unique")
        supplied.append(canonical.memory)
        restored[canonical.memory] = canonical.snapshot
    if tuple(supplied) != keys:
        raise ValueError(
            "memory snapshots must exactly match the memory keys in canonical order"
        )
    return restored


def _validate_session(
    value: object,
    key: str,
    origin: OriginRef | None,
) -> MemorySession:
    """Require the narrow callable session surface without reading properties."""
    for attribute in ("recall", "learn", "commit", "close"):
        if not callable(getattr(value, attribute, None)):
            raise MemorySetupError(
                f"memory {key!r} returned an invalid run session",
                origin=origin,
            )
    return value  # type: ignore[return-value]


class _LearningProposalError(Exception):
    """One known per-key proposal failure before any commit request."""

    def __init__(self, memory: str, failure: MemoryOperationFailure) -> None:
        """Keep the owning memory key and sanitized failure at the proposal boundary."""
        self.memory = memory
        self.failure = failure
        super().__init__(failure.message)


def _raise_host_error(error: BaseException) -> None:
    """Preserve host controls even when a backend wrapped them as ordinary errors."""
    pending = [(error, False)]
    seen: set[int] = set()
    saw_capability = False
    while pending:
        current, raise_original = pending.pop()
        if raise_original:
            raise current.with_traceback(current.__traceback__) from None
        if id(current) in seen:
            continue
        seen.add(id(current))
        if not isinstance(current, Exception) or isinstance(current, (
            _HostStop, AsyncCancelledError, FutureCancelledError, MemoryError, RecursionError,
        )):
            raise current.with_traceback(current.__traceback__) from None
        if current.__context__ is not None:
            pending.append((current.__context__, False))
        if current.__cause__ is not None:
            pending.append((current.__cause__, False))
        if isinstance(current, _CapabilityFault):
            saw_capability = True
            original = current.original
            if not isinstance(original, BaseException):
                raise TypeError("capability fault requires an original exception")
            # Inspect its chain before raising the original; nested controls may
            # have been wrapped by the injected capability itself.
            if not isinstance(original, _CapabilityFault):
                pending.append((original, True))
            pending.append((original, False))
    if saw_capability:
        raise RuntimeError("cyclic capability fault has no original exception")


class MemoryRuntime:
    """One opened set of per-key sessions shared by execution and learning."""

    def __init__(
        self,
        *,
        memories: tuple[MemorySpec, ...],
        origins: dict[str, OriginRef | None],
        sessions: dict[str, MemorySession],
        snapshots: tuple[MemorySnapshotEnvelope, ...],
    ) -> None:
        self._memories = memories
        self._origins = origins
        self._sessions = sessions
        self._snapshots = snapshots
        self._eligible = frozenset(
            spec.key for spec in memories if spec.conversation_eligible
        )
        self._closed = False

    @property
    def memories(self) -> tuple[MemorySpec, ...]:
        """Return the immutable recorded memory table for this run."""
        return self._memories

    @property
    def snapshots(self) -> tuple[MemorySnapshotEnvelope, ...]:
        """Return the immutable public snapshots captured after opening."""
        return self._snapshots

    @classmethod
    def open(
        cls,
        *,
        memories: tuple[MemorySpec, ...],
        origins: dict[str, OriginRef | None],
        memory: Mapping[str, Memory],
        snapshots: tuple[MemorySnapshotEnvelope, ...] | None = None,
    ) -> "MemoryRuntime":
        """Validate and open one session per recorded memory key.

        ``snapshots=None`` opens the implementations at their current state
        for a fresh run. A tuple restores the exact public snapshots recorded
        by ``RunStarted`` during recovery. An empty table opens nothing.
        """
        keys = _validate_memories(memories)
        canonical_origins = _validate_origins(keys, origins)
        registry = _validate_registry(memory)
        restore = _validate_snapshots(snapshots, keys)
        for key in keys:
            if key not in registry:
                raise MemorySetupError(
                    f"memory {key!r} was not supplied by the host",
                    origin=canonical_origins[key],
                )

        opened_sessions: dict[str, MemorySession] = {}
        try:
            for key in keys:
                try:
                    session = registry[key].open_run(snapshot=restore[key])
                    if callable(getattr(session, "close", None)):
                        opened_sessions[key] = session  # type: ignore[assignment]
                    opened_sessions[key] = _validate_session(
                        session, key, canonical_origins[key],
                    )
                except MemoryOperationError as exc:
                    raise MemorySetupError(
                        f"memory {key!r} could not open its run session: {exc.message}",
                        origin=canonical_origins[key],
                    ) from exc

            captured: list[MemorySnapshotEnvelope] = []
            for key in keys:
                try:
                    captured.append(
                        MemorySnapshotEnvelope(key, opened_sessions[key].snapshot)
                    )
                except MemoryOperationError as exc:
                    raise MemorySetupError(
                        f"memory {key!r} could not export its run snapshot: {exc.message}",
                        origin=canonical_origins[key],
                    ) from exc
                except (AttributeError, TypeError, ValueError, RecursionError) as exc:
                    raise MemorySetupError(
                        f"memory {key!r} exported an invalid run snapshot",
                        origin=canonical_origins[key],
                    ) from exc
            if snapshots is not None and tuple(captured) != snapshots:
                differing = next(
                    (
                        expected.memory
                        for expected, actual in zip(snapshots, captured)
                        if expected != actual
                    ),
                    keys[0] if keys else None,
                )
                raise MemorySetupError(
                    "restored memory session did not reproduce its recorded snapshot",
                    origin=None if differing is None else canonical_origins[differing],
                )
        except BaseException as primary:
            _close_sessions(opened_sessions, primary)
            raise

        return cls(
            memories=memories,
            origins=canonical_origins,
            sessions=opened_sessions,
            snapshots=tuple(captured),
        )

    @contextmanager
    def open_context(
        self,
        scope: MemoryScope,
        request: MemoryContextRequest,
        *,
        check_stopped: CheckStopped,
    ) -> Iterator[MemoryContext]:
        """Recall each key in the scope and admit its one eligible transcript.

        Sections keep the scope's outermost-first key order. Only a
        conversation-eligible key may contribute a transcript, and at most
        one key may do so; a losing source is never opened. A live
        transcript requires the accepted invocation, while text-only
        contexts do not. The admitted backend resource lasts until this
        context exits.

        Args:
            scope (`MemoryScope`): Exact compiled consumer scope over opened keys.
            request (`MemoryContextRequest`): Matching recall context, optional
                accepted invocation, and detached compatibility descriptor.
            check_stopped (`CheckStopped`): Checked after recall, before any
                transcript acquisition; also passed to the selected source.

        Returns:
            `AbstractContextManager[MemoryContext]`: Notes and the optional live
                transcript, valid until the context exits.

        Raises:
            `TypeError`: A carrier or callback has the wrong type.
            `ValueError`: Paths disagree or a transcript lacks an invocation.
            `RuntimeError`: The runtime is closed or a key is not opened.
            `MemoryRecallError`: A memory rejects recall, returns invalid
                content, returns a transcript without being
                conversation-eligible, or returns a second transcript.
            `Exception`: An unexpected recall or transcript backend error.

        Example:
            `with runtime.open_context(scope, request, check_stopped=check) as context:`
            admits all visible memory for one invocation.
        """
        if self._closed:
            raise RuntimeError("memory runtime is closed")
        if type(scope) is not MemoryScope:
            raise TypeError("memory context scope must be a MemoryScope")
        scope = MemoryScope(scope.static_path, scope.memories, scope.occurrence_path)
        if type(request) is not MemoryContextRequest:
            raise TypeError("memory request must be a MemoryContextRequest")
        request = MemoryContextRequest(
            request.recall, request.invocation, request.compatibility,
        )
        if not callable(check_stopped):
            raise TypeError("check_stopped must be callable")
        if request.recall.consumer_static_path != scope.static_path:
            raise ValueError("memory context request does not match the consumer scope")
        if not set(scope.memories) <= self._sessions.keys():
            raise RuntimeError("memory scope names a memory outside the opened table")
        sections: list[MemorySection] = []
        selected = None
        for key in scope.memories:
            session = self._sessions[key]
            origin = self._origins[key]
            try:
                content = session.recall(request.recall)
            except MemoryOperationError as exc:
                raise MemoryRecallError(
                    f"memory {key!r} recall failed: {exc.message}",
                    origin=origin,
                ) from exc
            try:
                if type(content) is MemoryContent:
                    content = MemoryContent(content.text, content.transcript)
                else:
                    content = MemoryContent(content)
                section = MemorySection(key, content.text)
            except (TypeError, ValueError, RecursionError) as exc:
                raise MemoryRecallError(
                    f"memory {key!r} returned invalid recall content",
                    origin=origin,
                ) from exc
            if section.text:
                sections.append(section)
            if content.transcript is not None:
                if key not in self._eligible:
                    raise MemoryRecallError(
                        f"memory {key!r} returned a transcript but is not "
                        "conversation-eligible; only a memory with one AGENT "
                        "owner may hold a conversation",
                        origin=origin,
                    )
                if selected is not None:
                    raise MemoryRecallError(
                        f"memory {key!r} returned a second transcript for "
                        f"{'/'.join(scope.static_path)!r}; one AGENT may hold "
                        "at most one conversation",
                        origin=origin,
                    )
                selected = content.transcript
        check_stopped()
        if selected is None:
            manager = nullcontext(None)
        else:
            if request.invocation is None:
                raise ValueError("transcript context requires an accepted invocation")
            transcript_request = ConversationRequest(
                scope.static_path, request.invocation, request.compatibility,
            )
            manager = selected.open_turn(transcript_request, check_stopped=check_stopped)
        with manager as transcript:
            yield MemoryContext(tuple(sections), transcript)

    def propose_learning(
        self, contexts: tuple[LearnContext, ...],
    ) -> tuple[MemoryProposal, ...]:
        """Stage every key's proposal without writing records or memory.

        Args:
            contexts (`tuple[LearnContext, ...]`): One detached context per
                recorded key, in table order, including empty batches.

        Returns:
            `tuple[MemoryProposal, ...]`: Validated mutations in table order.

        Raises:
            `TypeError`, `ValueError`: For invalid context coverage or carriers.
            `RuntimeError`: If this runtime is closed.
            `_LearningProposalError`: For a typed backend error or invalid proposal.
                Host controls and unexpected backend exceptions propagate unchanged.
        """
        if self._closed:
            raise RuntimeError("memory runtime is closed")
        if type(contexts) is not tuple or any(
            type(context) is not LearnContext for context in contexts
        ):
            raise TypeError("learning contexts must be a tuple of exact LearnContext values")
        # Each context is re-validated; frozen feedback exactly as a fresh context builds it is
        # immutable, so it is shared, and any other feedback is thawed and refrozen. Every other
        # carrier is a frozen slotted dataclass over tuples, so a deep copy would add nothing
        # but recursion deep enough to overflow on a value at the nesting limit.
        copied = tuple(LearnContext._revalidated(context) for context in contexts)
        if tuple(context.memory for context in copied) != tuple(
            spec.key for spec in self._memories
        ):
            raise ValueError(
                "learning contexts must cover the complete recorded memory table in order"
            )
        if len({context.run_id for context in copied}) > 1:
            raise ValueError("learning contexts must belong to one run")

        proposals = []
        for context in copied:
            key = context.memory
            try:
                raw = self._sessions[key].learn(context)
            except MemoryOperationError as exc:
                _raise_host_error(exc)
                raise _LearningProposalError(key, _operation_failure("learn", exc)) from exc
            try:
                if type(raw) is not list:
                    raise TypeError("memory learn() must return an exact list")
                proposal = MemoryProposal(key, tuple(raw))
            except (AttributeError, TypeError, ValueError) as exc:
                _raise_host_error(exc)
                raise _LearningProposalError(key, _invalid_result_failure("learn")) from exc
            proposals.append(proposal)
        return tuple(proposals)

    def commit_learning(self, request: MemoryCommitRequestedRecord) -> MemoryOperationFailure | None:
        """Perform one already accepted backward request without writing its observation.

        Args:
            request (`MemoryCommitRequestedRecord`): The accepted attempt-aware
                request, with its exact memory key, stable ID, and ordered batch.

        Returns:
            `MemoryOperationFailure | None`: A known backend/return-contract failure,
                or None on success. Unexpected exceptions and host controls escape.

        Raises:
            `TypeError`, `ValueError`: For a malformed or foreign request.
            `RuntimeError`: If this runtime is closed.
        """
        if self._closed:
            raise RuntimeError("memory runtime is closed")
        if type(request) is not MemoryCommitRequestedRecord:
            raise TypeError("backward commit requires an exact MemoryCommitRequestedRecord")
        request = MemoryCommitRequestedRecord(
            request.memory, request.commit_id, request.mutations, attempt_id=request.attempt_id,
        )
        if request.memory not in self._sessions:
            raise ValueError("backward commit requires an enabled memory and attempt")
        mutations = _validate_mutations(request.mutations)
        try:
            result = self._sessions[request.memory].commit(request.commit_id, mutations)
        except MemoryOperationError as exc:
            _raise_host_error(exc)
            return _operation_failure("commit", exc)
        return None if result is None else _invalid_result_failure("commit")

    def close(self) -> None:
        """Close every opened session in reverse key order exactly once."""
        self._close_preserving(None)

    def _close_preserving(self, primary_error: BaseException | None) -> None:
        """Close resources without replacing an initiating backward/host failure."""
        if self._closed:
            return
        self._closed = True
        _close_sessions(self._sessions, primary_error)


def _close_sessions(
    sessions: dict[str, MemorySession],
    primary_error: BaseException | None,
) -> None:
    """Close sessions in reverse order while preserving an initiating failure."""
    first_error: BaseException | None = None
    for key in reversed(tuple(sessions)):
        try:
            sessions[key].close()
        except BaseException as exc:
            target = primary_error if primary_error is not None else first_error
            if target is None:
                first_error = exc
                continue
            try:
                target.add_note(
                    f"memory {key!r} session close also raised "
                    f"{type(exc).__name__}: {exc}"
                )
            except BaseException:
                pass
    if primary_error is None and first_error is not None:
        raise first_error


def _operation_failure(
    operation: str,
    error: MemoryOperationError,
) -> MemoryOperationFailure:
    """Copy one typed backend failure or sanitize a forged carrier."""
    try:
        return MemoryOperationFailure(
            error.stage,
            error.code,
            error.message,
            error.retryable,
        )
    except (AttributeError, TypeError, ValueError, RecursionError):
        return _unexpected_operation_failure(operation)


def _unexpected_operation_failure(operation: str) -> MemoryOperationFailure:
    """Return one stable final failure without leaking an unexpected exception."""
    return MemoryOperationFailure(
        operation,
        "memory.unexpected",
        f"memory {operation} raised an unexpected error",
        False,
    )


def _invalid_result_failure(operation: str) -> MemoryOperationFailure:
    """Return one stable final failure for a malformed memory return value."""
    return MemoryOperationFailure(
        operation,
        "memory.invalid_result",
        f"memory {operation} returned an invalid result",
        False,
    )


__all__ = [
    "MemoryRuntime",
]
