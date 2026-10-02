"""Run-scoped orchestration for host-injected memory implementations.

The compile layer supplies one canonical tuple of source-neutral frame specs and
their optional authored origins.  This module validates the host registry, opens
exactly one session per referenced policy, opens every static frame, captures the
public recovery snapshots, and admits ordered memory context for the evaluator. It does
not resolve YAML, inspect node kinds, or construct a concrete memory backend.
The explicit backward path stages complete frame proposals and performs accepted
commit requests separately; the attempt driver owns their records and ordering.
"""

from __future__ import annotations

from asyncio import CancelledError as AsyncCancelledError
from collections.abc import Iterator
from concurrent.futures import CancelledError as FutureCancelledError
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from agent_composer.diagnostics import OriginRef
from agent_composer.memory import (
    CheckStopped,
    ConversationRequest,
    LearnContext,
    Memory,
    MemoryContent,
    MemoryContext,
    MemoryContextRequest,
    MemoryFrame,
    MemoryFrameRef,
    MemoryFrameSpec,
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
from agent_composer.memory.contracts import _validate_mutations
from agent_composer.nodes.base import _CapabilityFault, _HostStop
from agent_composer.runtime.contracts import FrameProposal
from agent_composer.runtime.log import MemoryCommitRequestedRecord


def _frame_key(ref: MemoryFrameRef) -> tuple[str, tuple[str, ...], str]:
    """Return the canonical structural order of one memory frame."""
    return (ref.root_flow_id, ref.static_path, ref.owner.value)


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
        raise TypeError("memory frame origins must be OriginRef values or None")
    OriginRef(value.key)
    return value


def _validate_layout(layout: tuple[MemoryFrameSpec, ...]) -> None:
    """Require canonical unique frame order and exact effective specs."""
    if type(layout) is not tuple:
        raise TypeError("memory layout must be a tuple")
    for spec in layout:
        if type(spec) is not MemoryFrameSpec:
            raise TypeError("memory layout items must be MemoryFrameSpec values")
        MemoryFrameSpec(spec.ref, spec.policy_name, spec.config)
    keys = tuple(_frame_key(spec.ref) for spec in layout)
    if keys != tuple(sorted(set(keys))):
        raise ValueError("memory layout frame refs must be sorted and unique")
    if len({spec.ref.root_flow_id for spec in layout}) > 1:
        raise ValueError("memory layout frame refs must share one root_flow_id")


def _validate_origins(
    layout: tuple[MemoryFrameSpec, ...],
    origins: dict[MemoryFrameRef, OriginRef | None],
) -> dict[MemoryFrameRef, OriginRef | None]:
    """Return a detached exact origin map aligned with ``layout``."""
    if type(origins) is not dict:
        raise TypeError("memory frame origins must be a dictionary")
    expected = {spec.ref for spec in layout}
    if set(origins) != expected:
        raise ValueError("memory frame origins must exactly match the layout")
    result: dict[MemoryFrameRef, OriginRef | None] = {}
    for ref, origin in origins.items():
        if type(ref) is not MemoryFrameRef:
            raise TypeError("memory frame origin keys must be MemoryFrameRef values")
        MemoryFrameRef(ref.root_flow_id, ref.static_path, ref.owner)
        result[ref] = _revalidate_origin(origin)
    return result


def _validate_registry(memory: dict[str, Memory]) -> dict[str, Memory]:
    """Return a detached exact registry and reject cross-name object aliases."""
    if type(memory) is not dict:
        raise TypeError("memory must be a dictionary or None")
    result: dict[str, Memory] = {}
    identities: dict[int, str] = {}
    for raw_name, implementation in memory.items():
        name = _require_name(raw_name, "memory policy name")
        if not isinstance(implementation, Memory):
            raise TypeError(
                f"memory policy {name!r} must implement the Memory protocol"
            )
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
    policy_names: tuple[str, ...],
) -> dict[str, dict[str, Any] | None]:
    """Return the per-policy restore payload, or fresh ``None`` payloads."""
    if snapshots is None:
        return {name: None for name in policy_names}
    if type(snapshots) is not tuple:
        raise TypeError("memory snapshots must be a tuple or None")
    restored: dict[str, dict[str, Any] | None] = {}
    supplied_names: list[str] = []
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
            raise ValueError("memory snapshot policy names must be unique")
        supplied_names.append(canonical.memory)
        restored[canonical.memory] = canonical.snapshot
    if tuple(supplied_names) != policy_names:
        raise ValueError(
            "memory snapshots must exactly match referenced policies in canonical order"
        )
    return restored


def _validate_session(
    value: object,
    policy_name: str,
    origin: OriginRef | None,
) -> MemorySession:
    """Require the narrow callable session surface without reading properties."""
    for attribute in ("open_frame", "commit", "close"):
        if not callable(getattr(value, attribute, None)):
            raise MemorySetupError(
                f"memory policy {policy_name!r} returned an invalid run session",
                origin=origin,
            )
    return value  # type: ignore[return-value]


def _validate_frame(
    value: object,
    policy_name: str,
    origin: OriginRef | None,
) -> MemoryFrame:
    """Require the narrow callable frame surface."""
    for attribute in ("recall", "learn"):
        if not callable(getattr(value, attribute, None)):
            raise MemorySetupError(
                f"memory policy {policy_name!r} returned an invalid frame",
                origin=origin,
            )
    return value  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class _OpenedFrame:
    """One effective frame paired with its opened backend handle and origin."""

    spec: MemoryFrameSpec
    frame: MemoryFrame
    origin: OriginRef | None


class _LearningProposalError(Exception):
    """One known frame-local proposal failure before any commit request."""

    def __init__(self, frame_ref: MemoryFrameRef, failure: MemoryOperationFailure) -> None:
        """Keep the owning frame and sanitized failure at the proposal boundary."""
        self.frame_ref = frame_ref
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
    """One opened set of policy sessions shared by execution and learning."""

    def __init__(
        self,
        *,
        layout: tuple[MemoryFrameSpec, ...],
        origins: dict[MemoryFrameRef, OriginRef | None],
        sessions: dict[str, MemorySession],
        frames: dict[MemoryFrameRef, _OpenedFrame],
        snapshots: tuple[MemorySnapshotEnvelope, ...],
        eligible: frozenset[str] | None = None,
    ) -> None:
        self._layout = layout
        self._origins = origins
        self._sessions = sessions
        self._frames = frames
        self._snapshots = snapshots
        self._eligible = eligible
        self._closed = False

    @property
    def layout(self) -> tuple[MemoryFrameSpec, ...]:
        """Return the immutable canonical frame inventory for this run."""
        return self._layout

    @property
    def snapshots(self) -> tuple[MemorySnapshotEnvelope, ...]:
        """Return the immutable public snapshots captured after frame opening."""
        return self._snapshots

    @classmethod
    def open(
        cls,
        *,
        layout: tuple[MemoryFrameSpec, ...],
        origins: dict[MemoryFrameRef, OriginRef | None],
        memory: dict[str, Memory],
        snapshots: tuple[MemorySnapshotEnvelope, ...] | None = None,
        memories: tuple[MemorySpec, ...] | None = None,
    ) -> "MemoryRuntime":
        """Validate and open one run-scoped memory authority.

        ``snapshots=None`` opens the implementations at their current state for a
        fresh run.  A tuple restores the exact public snapshots recorded by
        ``RunStarted`` during recovery.  ``memories`` is the recorded key table;
        when given, only a conversation-eligible memory may return a transcript,
        and at most one may.  Omitting it keeps the innermost-transcript rule.
        """
        if memories is not None:
            if type(memories) is not tuple or any(type(m) is not MemorySpec for m in memories):
                raise TypeError("memories must be a tuple of MemorySpec values")
            if {m.key for m in memories} != {spec.policy_name for spec in layout}:
                raise ValueError("memories must name exactly the layout's memory keys")
        _validate_layout(layout)
        canonical_origins = _validate_origins(layout, origins)
        registry = _validate_registry(memory)
        policy_names = tuple(sorted({spec.policy_name for spec in layout}))
        restore = _validate_snapshots(snapshots, policy_names)
        first_origin: dict[str, OriginRef | None] = {}
        for spec in layout:
            first_origin.setdefault(spec.policy_name, canonical_origins[spec.ref])
        for policy_name in policy_names:
            if policy_name not in registry:
                raise MemorySetupError(
                    f"memory policy {policy_name!r} was not supplied by the host",
                    origin=first_origin[policy_name],
                )

        opened_sessions: dict[str, MemorySession] = {}
        opened_frames: dict[MemoryFrameRef, _OpenedFrame] = {}
        try:
            for policy_name in policy_names:
                implementation = registry[policy_name]
                try:
                    session = implementation.open_run(
                        snapshot=restore[policy_name]
                    )
                    if callable(getattr(session, "close", None)):
                        opened_sessions[policy_name] = session  # type: ignore[assignment]
                    opened_sessions[policy_name] = _validate_session(
                        session,
                        policy_name,
                        first_origin[policy_name],
                    )
                except MemoryOperationError as exc:
                    raise MemorySetupError(
                        f"memory policy {policy_name!r} could not open its run session: "
                        f"{exc.message}",
                        origin=first_origin[policy_name],
                    ) from exc

            for spec in layout:
                try:
                    frame = opened_sessions[spec.policy_name].open_frame(spec)
                    opened_frames[spec.ref] = _OpenedFrame(
                        spec,
                        _validate_frame(
                            frame,
                            spec.policy_name,
                            canonical_origins[spec.ref],
                        ),
                        canonical_origins[spec.ref],
                    )
                except MemoryOperationError as exc:
                    raise MemorySetupError(
                        f"memory policy {spec.policy_name!r} could not open frame "
                        f"{spec.ref.static_path!r}: {exc.message}",
                        origin=canonical_origins[spec.ref],
                    ) from exc

            captured: list[MemorySnapshotEnvelope] = []
            for policy_name in policy_names:
                try:
                    captured.append(
                        MemorySnapshotEnvelope(
                            policy_name,
                            opened_sessions[policy_name].snapshot,
                        )
                    )
                except MemoryOperationError as exc:
                    raise MemorySetupError(
                        f"memory policy {policy_name!r} could not export its run "
                        f"snapshot: {exc.message}",
                        origin=first_origin[policy_name],
                    ) from exc
                except (AttributeError, TypeError, ValueError, RecursionError) as exc:
                    raise MemorySetupError(
                        f"memory policy {policy_name!r} exported an invalid run snapshot",
                        origin=first_origin[policy_name],
                    ) from exc
            if snapshots is not None and tuple(captured) != snapshots:
                differing = next(
                    (
                        expected.memory
                        for expected, actual in zip(snapshots, captured)
                        if expected != actual
                    ),
                    policy_names[0] if policy_names else None,
                )
                raise MemorySetupError(
                    "restored memory session did not reproduce its recorded snapshot",
                    origin=(
                        None if differing is None else first_origin[differing]
                    ),
                )
        except BaseException as primary:
            _close_sessions(opened_sessions, primary)
            raise

        return cls(
            layout=layout,
            origins=canonical_origins,
            sessions=opened_sessions,
            frames=opened_frames,
            snapshots=tuple(captured),
            eligible=(
                None
                if memories is None
                else frozenset(m.key for m in memories if m.conversation_eligible)
            ),
        )

    @contextmanager
    def open_context(
        self,
        scope: MemoryScope,
        request: MemoryContextRequest,
        *,
        check_stopped: CheckStopped,
    ) -> Iterator[MemoryContext]:
        """Recall each frame and admit the one transcript an eligible memory contributes.

        Notes retain outer-to-local ownership. Transcript selection uses only
        explicit recall contributions; losing sources are never opened. A live
        transcript requires the accepted invocation, while text-only contexts do
        not. The admitted backend resource lasts until this context exits.

        Args:
            scope (`MemoryScope`): Exact compiled consumer scope over opened frames.
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
            `RuntimeError`: The runtime is closed or a frame is not opened.
            `MemoryRecallError`: A policy rejects recall or returns invalid content,
                a memory that is not conversation-eligible returns a transcript, or
                a second memory returns a transcript.
            `Exception`: An unexpected recall or transcript backend error.

        Example:
            `with runtime.open_context(scope, request, check_stopped=check) as context:`
            admits all visible memory for one invocation.
        """
        if self._closed:
            raise RuntimeError("memory runtime is closed")
        if type(scope) is not MemoryScope:
            raise TypeError("memory context scope must be a MemoryScope")
        scope = MemoryScope(
            scope.root_flow_id, scope.static_path, scope.frame_stack,
            scope.occurrence_path,
        )
        if type(request) is not MemoryContextRequest:
            raise TypeError("memory request must be a MemoryContextRequest")
        request = MemoryContextRequest(
            request.recall, request.invocation, request.compatibility,
        )
        if not callable(check_stopped):
            raise TypeError("check_stopped must be callable")
        if request.recall.consumer_static_path != scope.static_path:
            raise ValueError("memory context request does not match the consumer scope")
        sections: list[MemorySection] = []
        selected = None
        for ref in scope.frame_stack:
            opened = self._frames.get(ref)
            if opened is None:
                raise RuntimeError("memory scope names a frame outside the opened layout")
            try:
                content = opened.frame.recall(request.recall)
            except MemoryOperationError as exc:
                raise MemoryRecallError(
                    f"memory policy {opened.spec.policy_name!r} recall failed: "
                    f"{exc.message}",
                    origin=opened.origin,
                ) from exc
            try:
                if type(content) is MemoryContent:
                    content = MemoryContent(content.text, content.transcript)
                else:
                    content = MemoryContent(content)
                section = MemorySection(ref, content.text)
            except (TypeError, ValueError, RecursionError) as exc:
                raise MemoryRecallError(
                    f"memory policy {opened.spec.policy_name!r} returned invalid "
                    "recall content",
                    origin=opened.origin,
                ) from exc
            if section.text:
                sections.append(section)
            if content.transcript is not None:
                key = opened.spec.policy_name
                if self._eligible is not None:
                    if key not in self._eligible:
                        raise MemoryRecallError(
                            f"memory {key!r} returned a transcript but is not "
                            "conversation-eligible; only a memory with one AGENT "
                            "owner may hold a conversation",
                            origin=opened.origin,
                        )
                    if selected is not None:
                        raise MemoryRecallError(
                            f"memory {key!r} returned a second transcript for "
                            f"{'/'.join(scope.static_path)!r}; one AGENT may hold "
                            "at most one conversation",
                            origin=opened.origin,
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

    def propose_learning(self, contexts: tuple[LearnContext, ...]) -> tuple[FrameProposal, ...]:
        """Stage every owned frame's proposal without writing records or memory.

        Args:
            contexts (`tuple[LearnContext, ...]`): Complete detached owner contexts
                in this runtime's recorded layout order, including empty frames.

        Returns:
            `tuple[FrameProposal, ...]`: Validated owned mutations in layout order.

        Raises:
            `TypeError`, `ValueError`: For invalid context coverage or carriers.
            `RuntimeError`: If this runtime is closed.
            `_LearningProposalError`: For a typed backend error or invalid proposal.
                Host controls and unexpected backend exceptions propagate unchanged.
        """
        if self._closed:
            raise RuntimeError("memory runtime is closed")
        if type(contexts) is not tuple or any(type(context) is not LearnContext for context in contexts):
            raise TypeError("learning contexts must be a tuple of exact LearnContext values")
        copied = tuple(deepcopy(LearnContext(
            context.run_id, context.frame_ref, context.episodes, context.feedback,
        )) for context in contexts)
        if tuple(context.frame_ref for context in copied) != tuple(spec.ref for spec in self._layout):
            raise ValueError("learning contexts must cover the complete recorded layout in order")
        if len({context.run_id for context in copied}) > 1:
            raise ValueError("learning contexts must belong to one run")

        proposals = []
        policy_ids: dict[str, set[str]] = {}
        for spec, context in zip(self._layout, copied):
            try:
                raw = self._frames[spec.ref].frame.learn(context)
            except MemoryOperationError as exc:
                _raise_host_error(exc)
                raise _LearningProposalError(spec.ref, _operation_failure("learn", exc)) from exc
            try:
                if type(raw) is not list:
                    raise TypeError("memory learn() must return an exact list")
                proposal = FrameProposal(spec.ref, tuple(raw))
                known_ids = policy_ids.setdefault(spec.policy_name, set())
                if any(mutation.mutation_id in known_ids for mutation in proposal.mutations):
                    raise ValueError("memory mutation ids must be unique within one policy commit")
            except (AttributeError, TypeError, ValueError) as exc:
                _raise_host_error(exc)
                raise _LearningProposalError(spec.ref, _invalid_result_failure("learn")) from exc
            known_ids.update(mutation.mutation_id for mutation in proposal.mutations)
            proposals.append(proposal)
        return tuple(proposals)

    def commit_learning(self, request: MemoryCommitRequestedRecord) -> MemoryOperationFailure | None:
        """Perform one already accepted backward request without writing its observation.

        Args:
            request (`MemoryCommitRequestedRecord`): The accepted attempt-aware
                request, with its exact policy, stable ID, and ordered batch.

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
            request.policy_name, request.commit_id, request.mutations, attempt_id=request.attempt_id,
        )
        if request.policy_name not in self._sessions:
            raise ValueError("backward commit requires an enabled policy and attempt")
        mutations = _validate_mutations(request.mutations)
        try:
            result = self._sessions[request.policy_name].commit(request.commit_id, mutations)
        except MemoryOperationError as exc:
            _raise_host_error(exc)
            return _operation_failure("commit", exc)
        return None if result is None else _invalid_result_failure("commit")

    def close(self) -> None:
        """Close every opened session in reverse policy order exactly once."""
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
    for policy_name in reversed(tuple(sessions)):
        try:
            sessions[policy_name].close()
        except BaseException as exc:
            target = primary_error if primary_error is not None else first_error
            if target is None:
                first_error = exc
                continue
            try:
                target.add_note(
                    f"memory policy {policy_name!r} session close also raised "
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
    """Return one stable final failure for a malformed policy return value."""
    return MemoryOperationFailure(
        operation,
        "memory.invalid_result",
        f"memory {operation} returned an invalid result",
        False,
    )


__all__ = [
    "MemoryRuntime",
]
