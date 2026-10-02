"""Conversations and learned notes with optional single-owner SQLite storage.

This backend owns FIFO turn persistence and atomic learned-note commits. It
accepts an injected cooperative-stop check but owns no provider, scheduler,
run log, or model call. Durable values use the lower private storage adapter.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import os
from threading import Condition
from types import TracebackType
from uuid import uuid4

import agent_composer.memory.contracts as c
from agent_composer.memory.conversation import (
    _invocation_from_json,
    _invocation_json,
    advance_conversation,
    conversation_content_digest,
    conversation_view_from_json,
    conversation_view_to_json,
)
from agent_composer.memory.storage import SQLiteStorage
from agent_composer.memory.summary import Summary, admissions, prefix_digest


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


def _note_id(source: c.JsonObject) -> str:
    return hashlib.sha256(_canonical(source).encode("utf-8")).hexdigest()


def _admitted(
    view: c.ConversationView | None, run_id: str, execution_seq: int
) -> bool:
    identity = _canonical([run_id, execution_seq, 0])
    return view is not None and any(
        entry.operation_id == identity for entry in view.applied
    )


def _validate_note(value: object) -> c.JsonObject:
    """Detach a closed note and prove its source identity."""
    note = _wire(value)
    if set(note) != {
        "note_id",
        "consumer_static_path",
        "run_id",
        "execution_seq",
        "message_index",
        "text",
    }:
        raise ValueError("note fields do not match the supported shape")
    if type(note["consumer_static_path"]) is not list:
        raise TypeError("note consumer path must be an array")
    c._require_path(tuple(note["consumer_static_path"]), "consumer_static_path")
    c._require_string(note["run_id"], "run_id")
    c._require_counter(note["execution_seq"], "execution_seq")
    c._require_counter(note["message_index"], "message_index")
    c._require_string(note["text"], "text", nonempty=False)
    c._require_string(note["note_id"], "note_id")
    if note["note_id"] != _note_id({k: v for k, v in note.items() if k != "note_id"}):
        raise ValueError("note identity does not match its source")
    return note


def _snapshot(
    namespace: str,
    generation: int,
    view: c.ConversationView | None,
    notes: tuple[c.JsonObject, ...],
) -> c.JsonObject:
    return _wire(
        {
            "version": 3,
            "namespace": namespace,
            "generation": generation,
            "conversation": (
                None if view is None else conversation_view_to_json(view)
            ),
            "notes": list(notes),
        }
    )


def _zero(
    namespace: str, generation: int, compatibility: c.JsonObject
) -> c.ConversationView:
    return c.ConversationView(
        namespace, generation, 0, compatibility, (), (), (), ()
    )


def _prefix(earlier: c.ConversationView, later: c.ConversationView) -> bool:
    """Prove a complete historical view against a newer ledger without guessing."""
    if (earlier.namespace, earlier.generation) != (
        later.namespace,
        later.generation,
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
) -> tuple[c.JsonObject, c.ConversationView | None, tuple[c.JsonObject, ...]]:
    """Decode the closed version 3 envelope without normalizing note order."""
    try:
        data = _wire(value)
        if type(data.get("version")) is not int or data["version"] != 3:
            raise ValueError("snapshot version must be 3")
        if set(data) != {
            "version",
            "namespace",
            "generation",
            "conversation",
            "notes",
        }:
            raise ValueError("snapshot fields do not match the supported version")
        c._require_string(data["namespace"], "namespace")
        c._require_counter(data["generation"], "generation")
        view = None
        if data["conversation"] is not None:
            view = conversation_view_from_json(data["conversation"])
            if (view.namespace, view.generation) != (
                data["namespace"],
                data["generation"],
            ):
                raise ValueError("snapshot contains a foreign conversation")
        if type(data["notes"]) is not list:
            raise TypeError("notes must be an array")
        notes, ids = [], set()
        for raw in data["notes"]:
            note = _validate_note(raw)
            if note["note_id"] in ids or not _admitted(
                view, note["run_id"], note["execution_seq"]
            ):
                raise ValueError("snapshot contains duplicate or unadmitted notes")
            ids.add(note["note_id"])
            notes.append(note)
        return data, view, tuple(notes)
    except (TypeError, ValueError) as exc:
        raise _error(
            "invalid_snapshot", "invalid SessionMemory snapshot", stage="snapshot"
        ) from exc


def _summary_json(summary: Summary) -> c.JsonObject:
    return {
        "namespace": summary.namespace,
        "generation": summary.generation,
        "projection_version": summary.projection_version,
        "covered": [_invocation_json(value) for value in summary.covered],
        "digest": summary.digest,
        "text": summary.text,
    }


def _parse_inventory(value: object) -> tuple[
    c.JsonObject,
    c.ConversationView | None,
    tuple[c.JsonObject, ...],
    bool,
    dict[str, str],
    dict[str, Summary],
]:
    """Prove every saved value, including private cache ownership and lineage."""
    try:
        data = _wire(value)
        if set(data) != {"version", "snapshot", "used", "applied_commits", "summaries"}:
            raise ValueError("invalid inventory fields")
        if type(data["version"]) is not int or data["version"] != 1:
            raise ValueError("unsupported inventory version")
        snapshot, view, notes = _parse_snapshot(data["snapshot"])
        if type(data["used"]) is not bool:
            raise ValueError("invalid inventory use flag")
        commits = _wire(data["applied_commits"])
        for commit_id, digest in commits.items():
            c._require_string(commit_id, "commit_id")
            c._require_string(digest, "digest")
            if len(digest) != 64 or any(
                char not in "0123456789abcdef" for char in digest
            ):
                raise ValueError("invalid commit digest")
        if type(data["summaries"]) is not list:
            raise ValueError("summaries must be an array")
        summaries = {}
        for raw in data["summaries"]:
            if type(raw) is not dict or set(raw) != {
                "namespace",
                "generation",
                "projection_version",
                "covered",
                "digest",
                "text",
            }:
                raise ValueError("invalid summary fields")
            c._require_string(raw["namespace"], "namespace")
            c._require_counter(raw["generation"], "generation")
            c._require_string(raw["projection_version"], "projection_version")
            c._require_string(raw["digest"], "digest")
            if type(raw["covered"]) is not list:
                raise ValueError("covered invocations must be an array")
            summary = Summary(
                raw["namespace"],
                raw["generation"],
                raw["projection_version"],
                tuple(_invocation_from_json(v) for v in raw["covered"]),
                raw["digest"],
                raw["text"],
            )
            cache_key = summary.projection_version
            if (
                cache_key in summaries
                or view is None
                or len(summary.covered) >= len(admissions(view))
                or not summary.matches(
                    view, admissions(view), summary.projection_version
                )
            ):
                raise ValueError("summary has foreign or invalid raw-prefix evidence")
            summaries[cache_key] = summary
        if not data["used"] and (
            snapshot["generation"]
            or view is not None
            or notes
            or commits
            or summaries
        ):
            raise ValueError("unused inventory contains retained work")
        return snapshot, view, notes, data["used"], commits, summaries
    except c.MemoryOperationError as exc:
        if exc.code == "session.unsupported_history":
            raise
        raise _error(
            "invalid_storage", "stored session state is invalid", stage="storage"
        ) from None
    except (TypeError, ValueError, RecursionError):
        raise _error(
            "invalid_storage", "stored session state is invalid", stage="storage"
        ) from None


@dataclass(eq=False)
class _Lease:
    session: _Session
    frame: _Frame
    request: c.ConversationRequest
    ticket: int
    active: bool = False
    released: bool = False
    before: c.ConversationView | None = None


class SessionMemory:
    """Own one conversation and its notes, optionally in local SQLite.

    Example:
        ``memory = SessionMemory()`` creates an isolated namespace. Injected
        conversation-capable callers use ``memory.open_run()``; runtime and
        provider integration are separate from this backend.
        ``with SessionMemory(storage="memory.sqlite") as memory:`` retains values
        across restarts and releases the single file owner on context exit.
    """

    def __init__(
        self,
        *,
        history_budget_chars: int = 32_000,
        recent_turns: int = 4,
        storage: str | os.PathLike[str] | None = None,
    ) -> None:
        """Create an empty namespace with an optional API context budget.

        Args:
            history_budget_chars (`int`): Positive soft context character budget;
                defaults to 32,000. Raw history is never removed.
            recent_turns (`int`): Nonnegative number of whole prior turns kept
                unsummarized; defaults to four. The current turn is always kept.
            storage (`str | PathLike[str] | None`): Optional local SQLite file with
                an existing parent; defaults to `None` for process-local values.
                Ownership lasts until close. Current history settings control
                future preparation without changing stored raw history or pins.

        Raises:
            `TypeError`: A setting is not an exact integer.
            `ValueError`: A setting is outside its supported range.
            `MemoryOperationError`: Storage is unavailable, occupied, or invalid.
        """
        for name, value, minimum in (
            ("history_budget_chars", history_budget_chars, 1),
            ("recent_turns", recent_turns, 0),
        ):
            if type(value) is not int:
                raise TypeError(f"{name} must be an exact integer")
            if value < minimum:
                raise ValueError(f"{name} must be at least {minimum}")
        self._history_budget_chars = history_budget_chars
        self._recent_turns = recent_turns
        self._summaries: dict[str, Summary] = {}
        self._condition = Condition()
        self._pid = os.getpid()
        self._closed = False
        self._storage: SQLiteStorage | None = None
        self._namespace = uuid4().hex
        self._generation = 0
        self._used = False
        self._view: c.ConversationView | None = None
        self._notes: tuple[c.JsonObject, ...] = ()
        self._applied_commits: dict[str, str] = {}
        self._queue: deque[_Lease] = deque()
        self._active: _Lease | None = None
        self._next_ticket = 0
        if storage is not None:
            self._storage = SQLiteStorage(storage, "session")
            try:
                saved = self._storage.load()
                if saved is None:
                    self._save_state(
                        _snapshot(
                            self._namespace, self._generation, self._view, self._notes
                        )
                    )
                else:
                    pinned, view, notes, used, commits, summaries = _parse_inventory(
                        saved
                    )
                    self._namespace, self._generation = (
                        pinned["namespace"],
                        pinned["generation"],
                    )
                    self._view, self._notes, self._used = view, notes, used
                    self._applied_commits, self._summaries = commits, summaries
            except BaseException as primary:
                try:
                    self._storage.close()
                except BaseException as cleanup:
                    _secondary(primary, cleanup)
                raise

    def _check_pid(self) -> None:
        """Reject inherited durable handles before touching any Python lock."""
        if self._storage is not None and os.getpid() != self._pid:
            raise c.MemoryOperationError(
                "storage",
                "storage.forked",
                "Memory storage cannot be used after fork.",
                False,
            )

    @contextmanager
    def _locked(self, *, cleanup: bool = False) -> Iterator[None]:
        """Guard a value operation, allowing coordination cleanup after failure."""
        self._check_pid()
        with self._condition:
            if not cleanup:
                if self._closed:
                    raise _error("closed", "memory backend is closed", stage="session")
                if self._storage is not None:
                    self._storage.ensure_open()
            yield

    def _save_state(
        self,
        snapshot: c.JsonObject,
        *,
        used: bool | None = None,
        commits: dict[str, str] | None = None,
        summaries: dict[str, Summary] | None = None,
    ) -> None:
        """Save a complete validated candidate before publishing it in RAM."""
        if self._storage is None:
            return
        inventory = {
            "version": 1,
            "snapshot": snapshot,
            "used": self._used if used is None else used,
            "applied_commits": self._applied_commits if commits is None else commits,
            "summaries": [
                _summary_json(summary)
                for summary in (
                    self._summaries if summaries is None else summaries
                ).values()
            ],
        }
        _parse_inventory(inventory)
        self._storage.save(inventory)

    def __enter__(self) -> SessionMemory:
        """Enter this still-open backend.

        Returns:
            `SessionMemory`: This backend.

        Raises:
            `MemoryOperationError`: The backend or its storage is unusable.
        """
        with self._locked():
            return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the backend without replacing an initiating exception.

        Args:
            exc_type (`type[BaseException] | None`): Active exception type, if any.
            exc (`BaseException | None`): Original exception to preserve.
            traceback (`TracebackType | None`): Original exception traceback.

        Raises:
            `MemoryOperationError`: Close fails without an initiating exception.
        """
        try:
            self.close()
        except BaseException as cleanup:
            if exc is None:
                raise
            _secondary(exc, cleanup)

    def close(self) -> None:
        """Release storage and invalidate idle handles without interrupting work.

        Repeated closes are inert. Active turns and queued tickets retain their
        ownership even after their run handle closes.

        Raises:
            `MemoryOperationError`: Active work makes close immediately busy,
                or an inherited handle or storage cleanup fails.
        """
        self._check_pid()
        if not self._condition.acquire(blocking=False):
            raise _error("busy", "memory backend is active", stage="close")
        try:
            if self._closed:
                return
            if self._active is not None or self._queue:
                raise _error(
                    "busy", "cannot close active or queued conversations", stage="close"
                )
            if self._storage is not None:
                self._storage.close()
            self._closed = True
        finally:
            self._condition.release()

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
        with self._locked():
            namespace, generation, live_view, live_notes = (
                self._namespace,
                self._generation,
                self._view,
                self._notes,
            )
            if parsed is None:
                pinned = _snapshot(
                    self._namespace, self._generation, self._view, self._notes
                )
            else:
                pinned, view, notes = parsed
                if not self._used:
                    namespace, generation = pinned["namespace"], pinned["generation"]
                    live_view, live_notes = view, notes
                elif (
                    (pinned["namespace"], pinned["generation"])
                    != (
                        self._namespace,
                        self._generation,
                    )
                    or (
                        view is not None
                        and (self._view is None or not _prefix(view, self._view))
                    )
                    or _canonical(list(notes))
                    != _canonical(list(self._notes[: len(notes)]))
                ):
                    raise _error(
                        "invalid_snapshot",
                        "snapshot is not proven history of this backend",
                        stage="snapshot",
                    )
            if not self._used:
                self._save_state(
                    _snapshot(namespace, generation, live_view, live_notes), used=True
                )
            self._namespace, self._generation = namespace, generation
            self._view, self._notes = live_view, live_notes
            self._used = True
            return _Session(self, pinned)

    def reset(self) -> None:
        """Clear the conversation and notes and invalidate older generation handles.

        Commit tombstones remain so exact retries cannot resurrect cleared notes.
        The operation does not delete provider resources.

        Raises:
            `MemoryOperationError`: The conversation has an active lease or queued
                ticket. Busy reset fails immediately without modifying state.
        """
        self._check_pid()
        if not self._condition.acquire(blocking=False):
            raise _error("busy", "memory backend is active", stage="reset")
        try:
            with self._locked():
                if self._active is not None or self._queue:
                    raise _error(
                        "busy",
                        "cannot reset active or queued conversations",
                        stage="reset",
                    )
                self._save_state(
                    _snapshot(self._namespace, self._generation + 1, None, ()),
                    used=True,
                    summaries={},
                )
                self._view = None
                self._notes = ()
                self._summaries.clear()
                self._generation += 1
                self._used = True
                self._condition.notify_all()
        finally:
            self._condition.release()


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
        with self._memory._locked():
            self._ensure_live()
            return _wire(self._pinned)

    def __enter__(self) -> _Session:
        """Enter this still-open, current-generation session."""
        with self._memory._locked():
            self._ensure_live()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the run handle without suppressing a body exception."""
        try:
            self.close()
        except BaseException as cleanup:
            if exc is None:
                raise
            _secondary(exc, cleanup)

    def close(self) -> None:
        """Close this run handle idempotently and remove its queued tickets.

        Active owners retain their leases until their turn finalizes, so closing
        a session cannot overlap an in-flight external effect with another turn.
        """
        with self._memory._locked(cleanup=True):
            if self._closed:
                return
            self._closed = True
            for lease in self._leases.values():
                if not lease.active and not lease.released:
                    self._release(lease)
            self._memory._condition.notify_all()

    def _release(self, lease: _Lease) -> None:
        """Release a lease or ticket while holding the backend condition."""
        with self._memory._locked(cleanup=True):
            memory = self._memory
            if memory._active is lease:
                memory._active = None
            if lease in memory._queue:
                memory._queue.remove(lease)
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
        with memory._locked():
            self._ensure_live()
            if request.invocation in self._leases:
                raise _error("conflict", "invocation already has a session lease")
            memory._next_ticket += 1
            lease = _Lease(self, frame, request, memory._next_ticket)
            self._leases[request.invocation] = lease
            memory._queue.append(lease)
            memory._condition.notify_all()
        try:
            while True:
                check_stopped()
                with memory._locked():
                    self._ensure_live()
                    if lease.released:
                        raise _error("stale", "conversation ticket was released")
                    if memory._queue[0] is lease and memory._active is None:
                        memory._queue.popleft()
                        lease.active = True
                        memory._active = lease
                        before = memory._view
                        if before is None:
                            before = _zero(
                                self._namespace,
                                self._generation,
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
        except BaseException as primary:
            try:
                self._release(lease)
            except BaseException as cleanup:
                _secondary(primary, cleanup)
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
        with self._memory._locked():
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
        with self._memory._locked():
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
                    note = _validate_note(payload["note"])
                    if note["note_id"] != mutation.mutation_id:
                        raise ValueError("mutation and note identities must match")
                    additions.append(note)
                    batch.append(
                        {
                            "mutation_id": mutation.mutation_id,
                            "frame_ref": {
                                "root_flow_id": mutation.frame_ref.root_flow_id,
                                "static_path": list(mutation.frame_ref.static_path),
                                "owner": mutation.frame_ref.owner.value,
                            },
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
                for note in additions:
                    if not _admitted(
                        self._memory._view, note["run_id"], note["execution_seq"]
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
                    self._namespace, self._generation, self._memory._view, candidate
                )
            except (TypeError, ValueError) as exc:
                raise _error(
                    "invalid_batch", "mutation batch is invalid", stage="commit"
                ) from exc
            commits = {**self._memory._applied_commits, commit_id: digest}
            self._memory._save_state(
                _snapshot(
                    self._namespace, self._generation, self._memory._view, candidate
                ),
                commits=commits,
            )
            self._memory._notes = candidate
            self._memory._applied_commits = commits

    def _candidate_snapshot(self, view: c.ConversationView) -> None:
        """Validate and save a complete view before publishing it in RAM."""
        candidate = _snapshot(
            self._namespace, self._generation, view, self._memory._notes
        )
        self._memory._save_state(candidate)

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
            or self._memory._active is not lease
            or self._leases.get(lease.request.invocation) is not lease
            or self._frames.get(lease.frame._spec.ref) is not lease.frame
        ):
            raise _error("stale", "conversation lease is no longer active")


class _Frame:
    """A frame view over the instance's single conversation."""

    def __init__(self, session: _Session, spec: c.MemoryFrameSpec) -> None:
        self._session = session
        self._spec = spec

    def recall(self, context: c.RecallContext | None = None) -> c.MemoryContent:
        """Contribute pinned notes and the live transcript source for one consumer.

        Args:
            context (`RecallContext | None`): Optional exact visible-context carrier;
                defaults to `None`, which yields no notes. Raw conversation
                history is not rendered as recall text.

        Returns:
            `MemoryContent`: Pinned notes of the instance in retained order and the
                transcript source.

        Raises:
            `TypeError`: A supplied context is not an exact recall-context carrier.
            `MemoryOperationError`: This frame's run handle is closed or reset-stale.
        """
        if context is not None and type(context) is not c.RecallContext:
            raise TypeError("context must be a RecallContext or None")
        with self._session._memory._locked():
            self._session._ensure_live()
            if context is None:
                return c.MemoryContent(transcript=self)
            context = c.RecallContext(
                context.base_system_prompt,
                context.prompt,
                context.inputs,
                context.consumer_static_path,
            )
            notes = "\n".join(
                note["text"] for note in self._session._pinned.get("notes", [])
            )
            return c.MemoryContent(notes, self)

    def learn(self, context: c.LearnContext) -> list[c.MemoryMutation]:
        """Propose notes from self-feedback for the instance's admitted invocations.

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
        with self._session._memory._locked():
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
                    if not _admitted(
                        self._session._memory._view, context.run_id, sequence
                    ):
                        continue
                    source = {
                        "consumer_static_path": list(episode.owner_static_path),
                        "run_id": context.run_id,
                        "execution_seq": sequence,
                        "message_index": feedback.get("message_index"),
                        "text": feedback.get("value"),
                    }
                    note = _validate_note({"note_id": _note_id(source), **source})
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
                for the instance's conversation and persists admission before
                returning.

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
        self._interrupted = False

    def _check(self) -> None:
        self._session._memory._check_pid()
        if self._stop_error is not None:
            raise self._stop_error
        try:
            self._check_stopped()
        except BaseException as exc:
            self._stop_error = exc
            raise

    def _stamp(self) -> c.ConversationStamp:
        return c.ConversationStamp(
            self._request.invocation,
            self._view.namespace,
            self._view.generation,
            self._view.revision,
            self._ordinal,
        )

    def _publish(self, checkpoint: c.ConversationCheckpoint) -> None:
        """Validate the whole candidate snapshot before atomically publishing."""
        with self._session._memory._locked():
            self._session._ensure_lease(
                self._lease,
                allow_closed=type(checkpoint)
                in (c.ConversationEffectObserved, c.ConversationClosed),
            )
            lease = self._lease
            stamp = checkpoint.stamp
            if (
                stamp.invocation != lease.request.invocation
                or (stamp.namespace, stamp.generation)
                != (self._session._namespace, self._session._generation)
            ):
                raise _error("conflict", "checkpoint does not belong to this lease")
            if _canonical(self._view.compatibility) != _canonical(
                lease.request.compatibility
            ):
                raise _error("incompatible", "turn compatibility changed")
            live = self._session._memory._view or lease.before
            if not _same(self._view, live):
                raise _error(
                    "conflict", "turn predecessor is not the current live view"
                )
            after = advance_conversation(self._view, checkpoint)
            self._session._candidate_snapshot(after)
            self._session._memory._view = after
            self._view = after
            self._ordinal += 1

    def __enter__(self) -> _Turn:
        """Acquire the authored conversation and persist its admission."""
        if self._entered:
            raise _error("stale", "a turn context can be entered only once")
        self._entered = True
        self._lease = self._session._reserve(self._frame, self._request, self._check)
        try:
            with self._session._memory._locked():
                self._session._ensure_lease(self._lease)
                self._view = self._lease.before
            self._publish(c.ConversationAdmitted(self._stamp(), self._view))
            self._check()
            with self._session._memory._locked():
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
        with self._session._memory._locked():
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
            kind (`ConversationEffectKind`): `model` or `tool`.
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
        with self._session._memory._locked():
            self._session._ensure_lease(self._lease)
        self._check()
        with self._session._memory._locked():
            self._session._ensure_lease(self._lease)
            checkpoint = c.ConversationEffectRequested(
                self._stamp(), kind, payload, items
            )
        self._publish(checkpoint)
        self._check()
        with self._session._memory._locked():
            self._session._ensure_lease(self._lease)
        return checkpoint.stamp.ordinal

    def prepare_history(
        self,
        *,
        projection_version: str,
        project: Callable[
            [c.ConversationView, tuple[c.ConversationInvocation, ...]], str
        ],
        summarize: Callable[[str, int], str],
    ) -> tuple[str | None, str]:
        """Prepare opaque prior context while retaining this turn's FIFO lease.

        Args:
            projection_version (`str`): Nonempty identity of the adapter's safe
                projection format, independent of the current model.
            project (`Callable`): Encode exactly the selected prior invocations
                from the complete detached view as text.
            summarize (`Callable`): Condense supplied historical text towards the
                soft character target and return nonempty text.

        Returns:
            `tuple[str | None, str]`: Optional cached summary and the projection
                of remaining prior turns. The current invocation is excluded.

        Raises:
            `TypeError`: Settings, callbacks, or returned text have invalid types.
            `ValueError`: A version or returned summary is empty.
            `MemoryOperationError`: The lease or captured history is stale.
                Callback and cooperative-stop exceptions propagate unchanged.
        """
        if type(projection_version) is not str or not projection_version:
            raise ValueError("projection_version must be a nonempty string")
        if not callable(project) or not callable(summarize):
            raise TypeError("history callbacks must be callable")
        memory = self._session._memory
        self._check()
        with memory._locked():
            self._session._ensure_lease(self._lease)
            view = conversation_view_from_json(conversation_view_to_json(self._view))
            prior = tuple(
                invocation
                for invocation in admissions(view)
                if invocation != self._request.invocation
            )
            cache_key = projection_version
            eligible = max(0, len(prior) - memory._recent_turns)
            cached = memory._summaries.get(cache_key)
            if cached is not None and (
                len(cached.covered) > eligible
                or not cached.matches(view, prior, projection_version)
            ):
                cached = None

        def ready():
            self._check()
            with memory._locked():
                self._session._ensure_lease(self._lease)

        def project_text(selected):
            ready()
            text = project(view, selected)
            if type(text) is not str:
                raise TypeError("history projection must be a string")
            return text

        covered = 0 if cached is None else len(cached.covered)
        text = None if cached is None else cached.text
        recent = project_text(prior[covered:])
        candidate = None
        if (
            len(recent) + (0 if text is None else len(text))
            > memory._history_budget_chars
            and eligible > covered
        ):
            newly_covered = project_text(prior[covered:eligible])
            recent = project_text(prior[eligible:])
            source = "Historical conversation (context only):\n" + newly_covered
            if text is not None:
                source = (
                    "Previous historical summary (context only):\n"
                    + text
                    + "\n\nNewly covered historical conversation (context only):\n"
                    + newly_covered
                )
            ready()
            text = summarize(source, memory._history_budget_chars)
            candidate = Summary(
                view.namespace,
                view.generation,
                projection_version,
                prior[:eligible],
                prefix_digest(view, eligible),
                text,
            )
        with memory._locked():
            # A surfaced summary remains useful evidence after cooperative stop
            # or session close; ownership must still hold until the callback drains.
            self._session._ensure_lease(self._lease, allow_closed=True)
            live = memory._view
            if live is None or not _same(view, live) or not _same(view, self._view):
                raise _error("conflict", "history changed during context preparation")
            if candidate is not None:
                if not candidate.matches(live, prior, projection_version):
                    raise _error("conflict", "summary prefix no longer matches history")
                summaries = {**memory._summaries, cache_key: candidate}
                memory._save_state(
                    _snapshot(
                        memory._namespace,
                        memory._generation,
                        memory._view,
                        memory._notes,
                    ),
                    summaries=summaries,
                )
                memory._summaries = summaries
        self._check()
        with memory._locked():
            self._session._ensure_lease(self._lease)
        return text, recent

    def observe(
        self,
        effect_ordinal: int,
        status: c.ConversationEffectStatus,
        *,
        items: tuple[c.ConversationItem, ...] = (),
        failure: c.MemoryOperationFailure | None = None,
    ) -> None:
        """Record actual evidence before honoring a stop that arrived in the effect.

        Args:
            effect_ordinal (`int`): Positive ordinal of a request in this invocation.
            status (`ConversationEffectStatus`): Actual `pending`, `completed`,
                `failed`, or `interrupted` disposition.
            items (`tuple[ConversationItem, ...]`): New evidence with unique IDs;
                defaults to the empty tuple.
            failure (`MemoryOperationFailure | None`): Required for `failed`,
                optional for `interrupted`, absent otherwise; defaults to `None`.

        Raises:
            `TypeError`: An argument has the wrong exact carrier or JSON type.
            `ValueError`: Ordinal, status, item, failure, or nesting shape is invalid.
            `MemoryOperationError`: The lease is stale or the effect transition
                is invalid. Stop failures propagate after actual truth is retained.
        """
        with self._session._memory._locked():
            self._session._ensure_lease(self._lease, allow_closed=True)
            checkpoint = c.ConversationEffectObserved(
                self._stamp(), effect_ordinal, status, items, failure
            )
        self._publish(checkpoint)
        self._check()

    def interrupt(self) -> None:
        """Close this turn as `interrupted` when its body later returns normally."""
        self._interrupted = True

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
                with self._session._memory._locked():
                    status = (
                        "interrupted"
                        if self._stop_error is not None
                        or self._session._closed
                        or (primary is not None and not isinstance(primary, Exception))
                        or (primary is None and self._interrupted)
                        else "failed" if primary is not None else "completed"
                    )
                    self._publish(c.ConversationClosed(self._stamp(), status))
        except BaseException as cleanup:
            if primary is None:
                primary = cleanup
            else:
                _secondary(primary, cleanup)
        finally:
            try:
                self._session._release(lease)
            except BaseException as cleanup:
                if primary is None:
                    primary = cleanup
                else:
                    _secondary(primary, cleanup)
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
