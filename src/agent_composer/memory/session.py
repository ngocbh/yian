"""Process-local conversations and learned notes behind the memory protocols.

This backend owns FIFO turn persistence and atomic learned-note commits. It
accepts an injected cooperative-stop check but owns no provider, scheduler,
run log, model call, or durable storage.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
import json
from threading import Condition
from types import TracebackType
from uuid import uuid4

import agent_composer.memory.contracts as c
from agent_composer.memory.conversation import (
    advance_conversation,
    conversation_content_digest,
    conversation_view_from_json,
    conversation_view_to_json,
)


def _error(
    code: str, message: str, *, stage: str = "conversation"
) -> c.MemoryOperationError:
    return c.MemoryOperationError(stage, "session." + code, message, code == "busy")


def _wire(value: object) -> c.JsonObject:
    return c._thaw_json_object(c._freeze_json_object(value, "session snapshot"))


def _canonical(value: c.JsonValue) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )


def _same(left: c.ConversationView, right: c.ConversationView) -> bool:
    return _canonical(conversation_view_to_json(left)) == _canonical(
        conversation_view_to_json(right)
    )


def _frame_json(ref: c.MemoryFrameRef) -> c.JsonObject:
    return {
        "root_flow_id": ref.root_flow_id,
        "static_path": list(ref.static_path),
        "owner": ref.owner.value,
    }


def _note_id(source: c.JsonObject) -> str:
    return hashlib.sha256(_canonical(source).encode("utf-8")).hexdigest()


def _admitted(
    views: dict[c.ConversationKey, c.ConversationView],
    key: c.ConversationKey,
    run_id: str,
    execution_seq: int,
) -> bool:
    view = views.get(key)
    identity = _canonical([run_id, execution_seq, 0])
    return view is not None and any(
        entry.operation_id == identity for entry in view.applied
    )


def _validate_note(value: object) -> tuple[c.JsonObject, c.ConversationKey]:
    """Detach a closed note and prove its source identity and lexical ownership."""
    note = _wire(value)
    if set(note) != {
        "note_id",
        "frame_ref",
        "consumer_static_path",
        "run_id",
        "execution_seq",
        "message_index",
        "text",
    }:
        raise ValueError("note fields do not match the supported shape")
    raw_ref = note["frame_ref"]
    if type(raw_ref) is not dict or set(raw_ref) != {
        "root_flow_id",
        "static_path",
        "owner",
    }:
        raise ValueError("invalid note frame identity")
    if (
        type(raw_ref["static_path"]) is not list
        or type(note["consumer_static_path"]) is not list
    ):
        raise TypeError("note paths must be arrays")
    c._require_string(raw_ref["owner"], "owner")
    ref = c.MemoryFrameRef(
        raw_ref["root_flow_id"],
        tuple(raw_ref["static_path"]),
        c.MemoryOwner(raw_ref["owner"]),
    )
    key = c.ConversationKey(ref, tuple(note["consumer_static_path"]))
    c._require_string(note["run_id"], "run_id")
    c._require_counter(note["execution_seq"], "execution_seq")
    c._require_counter(note["message_index"], "message_index")
    c._require_string(note["text"], "text", nonempty=False)
    c._require_string(note["note_id"], "note_id")
    if note["note_id"] != _note_id({k: v for k, v in note.items() if k != "note_id"}):
        raise ValueError("note identity does not match its source")
    return note, key


def _snapshot(
    namespace: str,
    generation: int,
    views: dict[c.ConversationKey, c.ConversationView],
    notes: tuple[c.JsonObject, ...],
) -> c.JsonObject:
    return _wire(
        {
            "version": 2,
            "namespace": namespace,
            "generation": generation,
            "conversations": [
                conversation_view_to_json(view) for view in views.values()
            ],
            "notes": list(notes),
        }
    )


def _zero(
    namespace: str, generation: int, key: c.ConversationKey, compatibility: c.JsonObject
) -> c.ConversationView:
    return c.ConversationView(
        namespace, generation, 0, key, compatibility, (), None, (), (), ()
    )


def _prefix(earlier: c.ConversationView, later: c.ConversationView) -> bool:
    """Prove a complete historical view against a newer ledger without guessing."""
    if (earlier.namespace, earlier.generation, earlier.key) != (
        later.namespace,
        later.generation,
        later.key,
    ) or _canonical(earlier.compatibility) != _canonical(later.compatibility):
        return False
    if earlier.revision > later.revision:
        return False
    if earlier.revision == later.revision:
        return _same(earlier, later)
    if earlier.applied != later.applied[: earlier.revision]:
        return False
    expected = (
        later.applied[earlier.revision - 1].after_digest
        if earlier.revision
        else later.applied[0].before_digest
    )
    return conversation_content_digest(earlier) == expected


def _parse_snapshot(
    value: object,
) -> tuple[
    c.JsonObject, dict[c.ConversationKey, c.ConversationView], tuple[c.JsonObject, ...]
]:
    """Decode the closed public envelope without normalizing its supplied order."""
    try:
        data = _wire(value)
        version = data.get("version")
        if type(version) is not int or version not in (1, 2):
            raise ValueError("snapshot version must be 1 or 2")
        fields = {"version", "namespace", "generation", "conversations"}
        if version == 2:
            fields.add("notes")
        if set(data) != fields:
            raise ValueError("snapshot fields do not match the supported version")
        c._require_string(data["namespace"], "namespace")
        c._require_counter(data["generation"], "generation")
        if type(data["conversations"]) is not list:
            raise TypeError("conversations must be an array")
        views = {}
        for raw in data["conversations"]:
            view = conversation_view_from_json(raw)
            if (view.namespace, view.generation) != (
                data["namespace"],
                data["generation"],
            ) or view.key in views:
                raise ValueError("snapshot contains foreign or duplicate conversations")
            views[view.key] = view
        raw_notes = data.get("notes", [])
        if type(raw_notes) is not list:
            raise TypeError("notes must be an array")
        notes, ids = [], set()
        for raw in raw_notes:
            note, key = _validate_note(raw)
            if note["note_id"] in ids or not _admitted(
                views, key, note["run_id"], note["execution_seq"]
            ):
                raise ValueError("snapshot contains duplicate or unadmitted notes")
            ids.add(note["note_id"])
            notes.append(note)
        return data, views, tuple(notes)
    except (TypeError, ValueError) as exc:
        raise _error(
            "invalid_snapshot", "invalid SessionMemory snapshot", stage="snapshot"
        ) from exc


@dataclass(eq=False)
class _Lease:
    session: _Session
    frame: _Frame
    request: c.ConversationRequest
    key: c.ConversationKey
    ticket: int
    active: bool = False
    released: bool = False
    before: c.ConversationView | None = None


class SessionMemory:
    """Own process-local authored-leaf conversations and learned self-feedback.

    Example:
        ``memory = SessionMemory()`` creates an isolated namespace. Injected
        conversation-capable callers use ``memory.open_run()``; runtime and
        provider integration are separate from this backend.
    """

    def __init__(self) -> None:
        """Create an empty namespace with no external resources."""
        self._condition = Condition()
        self._namespace = uuid4().hex
        self._generation = 0
        self._used = False
        self._views: dict[c.ConversationKey, c.ConversationView] = {}
        self._notes: tuple[c.JsonObject, ...] = ()
        self._applied_commits: dict[str, str] = {}
        self._queues: dict[c.ConversationKey, deque[_Lease]] = {}
        self._active: dict[c.ConversationKey, _Lease] = {}
        self._next_ticket = 0

    def open_run(self, *, snapshot: c.JsonObject | None = None) -> _Session:
        """Open a run with a detached, pinned public snapshot.

        Args:
            snapshot (`JsonObject | None`): Exact recorded snapshot to restore or
                replay; defaults to the current live inventory.

        Returns:
            `_Session`: A run-scoped memory session.

        Raises:
            `MemoryOperationError`: Snapshot shape or lineage is invalid. Only a
                pristine backend may adopt a snapshot's namespace and history.
        """
        parsed = None if snapshot is None else _parse_snapshot(snapshot)
        with self._condition:
            if parsed is None:
                pinned = _snapshot(
                    self._namespace, self._generation, self._views, self._notes
                )
            else:
                pinned, views, notes = parsed
                if not self._used:
                    self._namespace = pinned["namespace"]
                    self._generation = pinned["generation"]
                    self._views = views
                    self._notes = notes
                elif (
                    (pinned["namespace"], pinned["generation"])
                    != (
                        self._namespace,
                        self._generation,
                    )
                    or any(
                        key not in self._views or not _prefix(view, self._views[key])
                        for key, view in views.items()
                    )
                    or _canonical(list(notes))
                    != _canonical(list(self._notes[: len(notes)]))
                ):
                    raise _error(
                        "invalid_snapshot",
                        "snapshot is not proven history of this backend",
                        stage="snapshot",
                    )
            self._used = True
            return _Session(self, pinned)

    def reset(self) -> None:
        """Clear conversations and notes and invalidate older generation handles.

        Commit tombstones remain so exact retries cannot resurrect cleared notes.
        The operation is process-local and does not delete provider resources.

        Raises:
            `MemoryOperationError`: Any conversation has an active lease or queued
                ticket. Busy reset fails immediately without modifying state.
        """
        with self._condition:
            if self._active or self._queues:
                raise _error(
                    "busy", "cannot reset active or queued conversations", stage="reset"
                )
            self._views.clear()
            self._notes = ()
            self._generation += 1
            self._used = True
            self._condition.notify_all()


class _Session:
    """One run handle over a pinned snapshot and live conversation capabilities."""

    def __init__(self, memory: SessionMemory, pinned: c.JsonObject) -> None:
        self._memory = memory
        self._pinned = pinned
        self._namespace = memory._namespace
        self._generation = memory._generation
        self._closed = False
        self._frames: dict[c.MemoryFrameRef, _Frame] = {}
        self._leases: dict[c.ConversationInvocation, _Lease] = {}

    def _ensure_live(self) -> None:
        if self._closed or (self._namespace, self._generation) != (
            self._memory._namespace,
            self._memory._generation,
        ):
            raise _error(
                "stale", "memory session is closed or belongs to an old generation"
            )

    @property
    def snapshot(self) -> c.JsonObject:
        """Return a detached copy of the run's original snapshot.

        Returns:
            `JsonObject`: The exact pinned public envelope, unaffected by later
                frame openings or conversation checkpoints.

        Raises:
            `MemoryOperationError`: This run handle is closed or reset-stale.
        """
        with self._memory._condition:
            self._ensure_live()
            return _wire(self._pinned)

    def __enter__(self) -> _Session:
        """Enter this still-open, current-generation session."""
        with self._memory._condition:
            self._ensure_live()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the run handle without suppressing a body exception."""
        self.close()

    def close(self) -> None:
        """Close this run handle idempotently and remove its queued tickets.

        Active owners retain their leases until their turn finalizes, so closing
        a session cannot overlap an in-flight external effect with another turn.
        """
        with self._memory._condition:
            if self._closed:
                return
            self._closed = True
            for lease in self._leases.values():
                if not lease.active and not lease.released:
                    self._release(lease)
            self._memory._condition.notify_all()

    def _release(self, lease: _Lease) -> None:
        """Release a lease or ticket while holding the backend condition."""
        with self._memory._condition:
            memory = self._memory
            if memory._active.get(lease.key) is lease:
                del memory._active[lease.key]
            queue = memory._queues.get(lease.key)
            if queue is not None:
                if lease in queue:
                    queue.remove(lease)
                if not queue:
                    del memory._queues[lease.key]
            lease.active = False
            lease.released = True
            lease.before = None
            memory._condition.notify_all()

    def _reserve(
        self,
        frame: _Frame,
        request: c.ConversationRequest,
        check_stopped: c.CheckStopped,
    ) -> _Lease:
        """Reserve FIFO position before selecting the current live baseline."""
        memory = self._memory
        key = c.ConversationKey(frame._spec.ref, request.consumer_static_path)
        with memory._condition:
            self._ensure_live()
            if request.invocation in self._leases:
                raise _error("conflict", "invocation already has a session lease")
            memory._next_ticket += 1
            lease = _Lease(self, frame, request, key, memory._next_ticket)
            self._leases[request.invocation] = lease
            memory._queues.setdefault(key, deque()).append(lease)
            memory._condition.notify_all()
        try:
            while True:
                check_stopped()
                with memory._condition:
                    self._ensure_live()
                    if lease.released:
                        raise _error("stale", "conversation ticket was released")
                    queue = memory._queues[key]
                    if queue[0] is lease and key not in memory._active:
                        queue.popleft()
                        if not queue:
                            del memory._queues[key]
                        lease.active = True
                        memory._active[key] = lease
                        before = memory._views.get(key)
                        if before is None:
                            before = _zero(
                                self._namespace,
                                self._generation,
                                key,
                                request.compatibility,
                            )
                        elif _canonical(before.compatibility) != _canonical(
                            request.compatibility
                        ):
                            raise _error(
                                "incompatible",
                                "conversation compatibility changed; reset is required",
                            )
                        stamp = c.ConversationStamp(
                            frame._spec.policy_name,
                            key,
                            request.invocation,
                            self._namespace,
                            self._generation,
                            before.revision,
                            0,
                        )
                        advance_conversation(
                            before, c.ConversationAdmitted(stamp, before)
                        )
                        lease.before = before
                        return lease
                    memory._condition.wait(timeout=0.05)
        except BaseException:
            self._release(lease)
            raise

    def open_frame(self, spec: c.MemoryFrameSpec) -> _Frame:
        """Open one empty-config frame idempotently within this run.

        Args:
            spec (`MemoryFrameSpec`): Exact authored owner and actual host policy
                name. Configuration must be empty; reopening requires the same spec.

        Returns:
            `_Frame`: The same frame object for repeated identical openings.

        Raises:
            `TypeError`: The spec is not an exact frame carrier.
            `ValueError`: A malformed carrier contains invalid identity or JSON.
            `MemoryOperationError`: This handle is stale, configuration is nonempty,
                or the owner was opened with a different spec.
        """
        if type(spec) is not c.MemoryFrameSpec:
            raise TypeError("spec must be a MemoryFrameSpec")
        detached = c.MemoryFrameSpec(spec.ref, spec.policy_name, spec.config)
        with self._memory._condition:
            self._ensure_live()
            if detached.config:
                raise _error(
                    "invalid_config",
                    "SessionMemory frame config must be empty",
                    stage="frame",
                )
            previous = self._frames.get(detached.ref)
            if previous is not None:
                if previous._spec != detached:
                    raise _error(
                        "invalid_config",
                        "frame was already opened with another policy",
                        stage="frame",
                    )
                return previous
            frame = _Frame(self, detached)
            self._frames[detached.ref] = frame
            return frame

    def commit(self, commit_id: str, mutations: tuple[c.MemoryMutation, ...]) -> None:
        """Atomically append one exact idempotent batch of learned notes.

        Args:
            commit_id (`str`): Nonempty caller-owned batch identity. Exact replays
                remain inert even after reset; a different batch is a collision.
            mutations (`tuple[MemoryMutation, ...]`): Exact append-note proposals
                for frames opened by this run handle. Empty batches are allowed.

        Raises:
            `TypeError`: The identity or batch has the wrong exact type.
            `ValueError`: The identity is empty or invalid.
            `MemoryOperationError`: The batch is malformed, foreign, unadmitted,
                or collides with a prior commit; the handle is closed or stale.
        """
        c._require_string(commit_id, "commit_id")
        if type(mutations) is not tuple:
            raise TypeError("mutations must be a tuple")
        with self._memory._condition:
            self._ensure_live()
            try:
                mutations = c._validate_mutations(mutations)
                additions, batch = [], []
                for mutation in mutations:
                    if mutation.frame_ref not in self._frames:
                        raise ValueError(
                            "mutation frame was not opened by this session"
                        )
                    payload = mutation.payload
                    if (
                        type(payload) is not dict
                        or set(payload) != {"operation", "note"}
                        or payload["operation"] != "append_note"
                    ):
                        raise ValueError(
                            "mutation must be a closed append-note operation"
                        )
                    note, key = _validate_note(payload["note"])
                    if (
                        key.frame_ref != mutation.frame_ref
                        or note["note_id"] != mutation.mutation_id
                    ):
                        raise ValueError("mutation and note identities must match")
                    additions.append((note, key))
                    batch.append(
                        {
                            "mutation_id": mutation.mutation_id,
                            "frame_ref": _frame_json(mutation.frame_ref),
                            "payload": payload,
                        }
                    )
                digest = hashlib.sha256(_canonical(batch).encode("utf-8")).hexdigest()
            except (TypeError, ValueError) as exc:
                raise _error(
                    "invalid_batch", "mutation batch is invalid", stage="commit"
                ) from exc
            previous = self._memory._applied_commits.get(commit_id)
            if previous is not None:
                if previous == digest:
                    return
                raise _error(
                    "commit_collision",
                    "commit id identifies a different batch",
                    stage="commit",
                )
            try:
                notes = {note["note_id"]: note for note in self._memory._notes}
                for note, key in additions:
                    if not _admitted(
                        self._memory._views, key, note["run_id"], note["execution_seq"]
                    ):
                        raise ValueError("note has no matching conversation admission")
                    retained = notes.get(note["note_id"])
                    if retained is not None and _canonical(retained) != _canonical(
                        note
                    ):
                        raise ValueError("note id identifies different content")
                    notes[note["note_id"]] = note
                candidate = tuple(notes.values())
                _snapshot(
                    self._namespace, self._generation, self._memory._views, candidate
                )
            except (TypeError, ValueError) as exc:
                raise _error(
                    "invalid_batch", "mutation batch is invalid", stage="commit"
                ) from exc
            commits = {**self._memory._applied_commits, commit_id: digest}
            self._memory._notes = candidate
            self._memory._applied_commits = commits

    def _candidate_snapshot(self, view: c.ConversationView) -> None:
        """Prove that publishing a view cannot make public snapshots unrestorable."""
        views = dict(self._memory._views)
        views[view.key] = view
        _snapshot(self._namespace, self._generation, views, self._memory._notes)

    def _ensure_lease(
        self, lease: _Lease | None, *, allow_closed: bool = False
    ) -> None:
        if not allow_closed:
            self._ensure_live()
        if (
            (self._namespace, self._generation)
            != (self._memory._namespace, self._memory._generation)
            or lease is None
            or lease.session is not self
            or not lease.active
            or lease.released
            or self._memory._active.get(lease.key) is not lease
            or self._leases.get(lease.request.invocation) is not lease
            or self._frames.get(lease.frame._spec.ref) is not lease.frame
        ):
            raise _error("stale", "conversation lease is no longer active")


class _Frame:
    """A policy-named frame whose descendants own independent conversations."""

    def __init__(self, session: _Session, spec: c.MemoryFrameSpec) -> None:
        self._session = session
        self._spec = spec

    def recall(self, context: c.RecallContext | None = None) -> str:
        """Recall this frame's pinned notes for exactly one authored consumer.

        Args:
            context (`RecallContext | None`): Optional exact visible-context carrier;
                defaults to `None`, which yields no notes. Raw conversation
                history is not rendered as recall text.

        Returns:
            `str`: Notes from the run-opening snapshot, in retained order.

        Raises:
            `TypeError`: A supplied context is not an exact recall-context carrier.
            `MemoryOperationError`: This frame's run handle is closed or reset-stale.
        """
        if context is not None and type(context) is not c.RecallContext:
            raise TypeError("context must be a RecallContext or None")
        with self._session._memory._condition:
            self._session._ensure_live()
            if context is None:
                return ""
            context = c.RecallContext(
                context.base_system_prompt,
                context.prompt,
                context.inputs,
                context.consumer_static_path,
            )
            return "\n".join(
                note["text"]
                for note in self._session._pinned.get("notes", [])
                if note["frame_ref"] == _frame_json(self._spec.ref)
                and note["consumer_static_path"] == list(context.consumer_static_path)
            )

    def learn(self, context: c.LearnContext) -> list[c.MemoryMutation]:
        """Propose notes from self-feedback for this frame's admitted invocations.

        Args:
            context (`LearnContext`): Exact validated frame learning context.

        Returns:
            `list[MemoryMutation]`: Deterministic append-note proposals. Automatic
                history is unchanged, and no provider or model is called.

        Raises:
            `TypeError`: The context is not an exact learning-context carrier.
            `MemoryOperationError`: The context is invalid for this frame, or its
                run handle is closed or reset-stale.
        """
        if type(context) is not c.LearnContext:
            raise TypeError("context must be a LearnContext")
        with self._session._memory._condition:
            self._session._ensure_live()
            try:
                if context.frame_ref != self._spec.ref:
                    raise ValueError("learning context belongs to another frame")
                context = c.LearnContext(
                    context.run_id,
                    context.frame_ref,
                    context.episodes,
                    context.feedback,
                )
                episodes = {}
                for episode in context.episodes:
                    path = episode.owner_static_path
                    owner = self._spec.ref.static_path
                    if path[: len(owner)] != owner or (
                        self._spec.ref.owner is c.MemoryOwner.AGENT and path != owner
                    ):
                        raise ValueError("episode is outside its owning frame")
                    if episode.execution_seq is not None:
                        if episode.execution_seq in episodes:
                            raise ValueError(
                                "episode execution coordinates must be unique"
                            )
                        episodes[episode.execution_seq] = episode
                notes = {}
                for feedback in context.feedback:
                    if (
                        type(feedback) is not dict
                        or feedback.get("channel") != "self"
                        or feedback.get("root") is True
                    ):
                        continue
                    sequence = feedback.get("execution_seq")
                    if sequence is None:
                        continue
                    c._require_counter(sequence, "execution_seq")
                    episode = episodes.get(sequence)
                    if episode is None:
                        continue
                    # Structural flow episodes have no authored leaf conversation.
                    if (
                        self._spec.ref.owner is c.MemoryOwner.FLOW
                        and episode.owner_static_path == self._spec.ref.static_path
                    ):
                        continue
                    key = c.ConversationKey(self._spec.ref, episode.owner_static_path)
                    if not _admitted(
                        self._session._memory._views, key, context.run_id, sequence
                    ):
                        continue
                    source = {
                        "frame_ref": _frame_json(self._spec.ref),
                        "consumer_static_path": list(episode.owner_static_path),
                        "run_id": context.run_id,
                        "execution_seq": sequence,
                        "message_index": feedback.get("message_index"),
                        "text": feedback.get("value"),
                    }
                    note, _ = _validate_note({"note_id": _note_id(source), **source})
                    notes[note["note_id"]] = note
                return [
                    c.MemoryMutation(
                        note_id,
                        self._spec.ref,
                        {"operation": "append_note", "note": note},
                    )
                    for note_id, note in notes.items()
                ]
            except (TypeError, ValueError) as exc:
                raise _error(
                    "invalid_context", "learning context is invalid", stage="learn"
                ) from exc

    def open_turn(
        self,
        request: c.ConversationRequest,
        *,
        check_stopped: c.CheckStopped,
    ) -> _Turn:
        """Build a queued turn that persists directly through this backend.

        Args:
            request (`ConversationRequest`): Authored consumer, unique invocation,
                and exact compatibility descriptor. The frame determines ownership.
            check_stopped (`CheckStopped`): Cooperative check that raises on stop;
                checked during queuing and between effects, never before recording
                an actual observation. Failures propagate unchanged.

        Returns:
            `_Turn`: A single-entry context manager; entering waits in FIFO order
                for this authored leaf and persists admission before returning.

        Raises:
            `TypeError`: The request or callbacks have the wrong type.
            `ValueError`: A malformed request has invalid identity or JSON.
            `MemoryOperationError`: On entry, the handle is stale, compatibility
                changed, or the invocation conflicts with retained history.
        """
        if type(request) is not c.ConversationRequest:
            raise TypeError("request must be a ConversationRequest")
        if not callable(check_stopped):
            raise TypeError("check_stopped must be callable")
        request = c.ConversationRequest(
            request.consumer_static_path, request.invocation, request.compatibility
        )
        return _Turn(self, request, check_stopped)


def _secondary(primary: BaseException, cleanup: BaseException) -> None:
    """Preserve a primary host failure while reporting a secondary cleanup fault."""
    try:
        if hasattr(primary, "add_note"):
            primary.add_note(
                f"Conversation cleanup also failed: {type(cleanup).__name__}: {cleanup}"
            )
    except BaseException:
        pass


class _Turn:
    """One FIFO lease that persists requests and actual observations directly."""

    def __init__(
        self,
        frame: _Frame,
        request: c.ConversationRequest,
        check_stopped: c.CheckStopped,
    ) -> None:
        self._frame = frame
        self._session = frame._session
        self._request = request
        self._check_stopped = check_stopped
        self._lease: _Lease | None = None
        self._view: c.ConversationView | None = None
        self._ordinal = 0
        self._entered = False
        self._stop_error: BaseException | None = None

    def _check(self) -> None:
        if self._stop_error is not None:
            raise self._stop_error
        try:
            self._check_stopped()
        except BaseException as exc:
            self._stop_error = exc
            raise

    def _stamp(self) -> c.ConversationStamp:
        return c.ConversationStamp(
            self._frame._spec.policy_name,
            self._view.key,
            self._request.invocation,
            self._view.namespace,
            self._view.generation,
            self._view.revision,
            self._ordinal,
        )

    def _publish(self, checkpoint: c.ConversationCheckpoint) -> None:
        """Validate the whole candidate snapshot before atomically publishing."""
        with self._session._memory._condition:
            self._session._ensure_lease(
                self._lease,
                allow_closed=type(checkpoint)
                in (c.ConversationEffectObserved, c.ConversationClosed),
            )
            lease = self._lease
            stamp = checkpoint.stamp
            if (
                stamp.key != lease.key
                or stamp.invocation != lease.request.invocation
                or stamp.policy_name != lease.frame._spec.policy_name
                or (stamp.namespace, stamp.generation)
                != (self._session._namespace, self._session._generation)
            ):
                raise _error("conflict", "checkpoint does not belong to this lease")
            if _canonical(self._view.compatibility) != _canonical(
                lease.request.compatibility
            ):
                raise _error("incompatible", "turn compatibility changed")
            live = self._session._memory._views.get(lease.key, lease.before)
            if not _same(self._view, live):
                raise _error(
                    "conflict", "turn predecessor is not the current live view"
                )
            after = advance_conversation(self._view, checkpoint)
            self._session._candidate_snapshot(after)
            self._session._memory._views[after.key] = after
            self._view = after
            self._ordinal += 1

    def __enter__(self) -> _Turn:
        """Acquire the authored conversation and persist its admission."""
        if self._entered:
            raise _error("stale", "a turn context can be entered only once")
        self._entered = True
        self._lease = self._session._reserve(self._frame, self._request, self._check)
        try:
            with self._session._memory._condition:
                self._session._ensure_lease(self._lease)
                self._view = self._lease.before
            self._publish(c.ConversationAdmitted(self._stamp(), self._view))
            self._check()
            with self._session._memory._condition:
                self._session._ensure_lease(self._lease)
            return self
        except BaseException as exc:
            self._finish(exc)
            raise

    @property
    def view(self) -> c.ConversationView:
        """Return the last persisted detached state while the lease is live.

        Returns:
            `ConversationView`: Immutable carrier with defensive JSON properties.

        Raises:
            `MemoryOperationError`: The turn is not active or belongs to a stale run.
        """
        with self._session._memory._condition:
            self._session._ensure_lease(self._lease)
            return conversation_view_from_json(conversation_view_to_json(self._view))

    def request(
        self,
        kind: c.ConversationEffectKind,
        payload: c.JsonObject,
        *,
        items: tuple[c.ConversationItem, ...] = (),
    ) -> int:
        """Persist an effect request before allowing its external execution.

        Args:
            kind (`ConversationEffectKind`): `model`, `tool`, or `native_session`.
            payload (`JsonObject`): Exact finite JSON describing the external request.
            items (`tuple[ConversationItem, ...]`): Newly retained items with unique
                IDs; defaults to the empty tuple.

        Returns:
            `int`: Positive effect ordinal, only after persistence, live-lease,
                and stop checks. Stop failures propagate unchanged.

        Raises:
            `TypeError`: An argument has the wrong exact carrier or JSON type.
            `ValueError`: Kind, item identity, finite JSON, or nesting is invalid.
            `MemoryOperationError`: The lease is stale or its history conflicts.
        """
        with self._session._memory._condition:
            self._session._ensure_lease(self._lease)
        self._check()
        with self._session._memory._condition:
            self._session._ensure_lease(self._lease)
            checkpoint = c.ConversationEffectRequested(
                self._stamp(), kind, payload, items
            )
        self._publish(checkpoint)
        self._check()
        with self._session._memory._condition:
            self._session._ensure_lease(self._lease)
        return checkpoint.stamp.ordinal

    def observe(
        self,
        effect_ordinal: int,
        status: c.ConversationEffectStatus,
        *,
        items: tuple[c.ConversationItem, ...] = (),
        reference: c.ConversationReference | None = None,
        failure: c.MemoryOperationFailure | None = None,
    ) -> None:
        """Record actual evidence before honoring a stop that arrived in the effect.

        Args:
            effect_ordinal (`int`): Positive ordinal of a request in this invocation.
            status (`ConversationEffectStatus`): Actual `pending`, `completed`,
                `failed`, or `interrupted` disposition.
            items (`tuple[ConversationItem, ...]`): New evidence with unique IDs;
                defaults to the empty tuple.
            reference (`ConversationReference | None`): Opaque native association;
                defaults to `None`, which preserves the previous association.
            failure (`MemoryOperationFailure | None`): Required for `failed`,
                optional for `interrupted`, absent otherwise; defaults to `None`.

        Raises:
            `TypeError`: An argument has the wrong exact carrier or JSON type.
            `ValueError`: Ordinal, status, item, failure, or nesting shape is invalid.
            `MemoryOperationError`: The lease is stale or the effect transition
                is invalid. Stop failures propagate after actual truth is retained.
        """
        with self._session._memory._condition:
            self._session._ensure_lease(self._lease, allow_closed=True)
            checkpoint = c.ConversationEffectObserved(
                self._stamp(), effect_ordinal, status, items, reference, failure
            )
        self._publish(checkpoint)
        self._check()

    def _finish(self, primary: BaseException | None) -> None:
        lease = self._lease
        if lease is None or lease.released:
            return
        try:
            try:
                self._check()
            except BaseException as stop:
                if primary is None:
                    primary = stop
                elif primary is not stop:
                    _secondary(primary, stop)
            if self._ordinal:
                with self._session._memory._condition:
                    status = (
                        "interrupted"
                        if self._stop_error is not None
                        or self._session._closed
                        or (primary is not None and not isinstance(primary, Exception))
                        else "failed" if primary is not None else "completed"
                    )
                    self._publish(c.ConversationClosed(self._stamp(), status))
        except BaseException as cleanup:
            if primary is None:
                raise
            _secondary(primary, cleanup)
        finally:
            self._session._release(lease)
        if primary is not None:
            raise primary

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Persist a truthful close, release the lease, and preserve body errors."""
        self._finish(exc)
