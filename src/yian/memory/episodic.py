"""Episodic memory with snapshot-isolated runs and optional SQLite persistence.

The backend owns retained episode history behind the source-neutral memory
protocols.  This module establishes the session, instance, snapshot, and commit
boundaries without depending on the workflow engine or providers. Optional
durability uses the private single-owner storage adapter below this backend.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import os
from threading import RLock
from types import TracebackType
from typing import Iterator, cast
from uuid import uuid4

from yian.defaults import current
from yian.memory.contracts import (
    JsonObject,
    JsonValue,
    LearnContext,
    MemoryEpisode,
    MemoryEpisodeStatus,
    MemoryMutation,
    MemoryOperationError,
    MemoryOccurrenceKind,
    MemoryOccurrenceSegment,
    MemorySection,
    ModelTextInteraction,
    ModelValueInteraction,
    RecallContext,
    ToolCallInteraction,
    ToolResultInteraction,
    VisibleInteraction,
    _memory_nesting_error,
)
from yian.memory.storage import SQLiteStorage, check_storage_path

_SNAPSHOT_FORMAT = "yian.episodic_memory"
_SNAPSHOT_VERSION = 2
_POLICY_NAME = "episodic"


@dataclass(frozen=True, slots=True)
class _Limits:
    """Effective retention and recall limits of one episodic instance."""

    max_episodes: int
    recall_limit: int


@dataclass(frozen=True, slots=True)
class _State:
    """The limits and ordered retained entries of one episodic instance."""

    limits: _Limits
    episodes: tuple[JsonObject, ...]


def _operation_error(stage: str, code: str, message: str) -> MemoryOperationError:
    """Build one stable nonretryable backend failure."""
    return MemoryOperationError(stage, code, message, False)


def _positive_int(value: object, name: str) -> int:
    """Require one exact positive integer."""
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _resolve_limits(
    requested: tuple[int | None, int | None],
    recorded: _Limits | None,
    stage: str,
) -> _Limits:
    """Fill unset limits from recorded ones or defaults; reject explicit conflicts."""
    max_episodes, recall_limit = requested
    if recorded is not None and (
        max_episodes not in (None, recorded.max_episodes)
        or recall_limit not in (None, recorded.recall_limit)
    ):
        raise _operation_error(
            stage,
            "episodic.config_conflict",
            "configured limits differ from the recorded limits",
        )
    if recorded is None:
        values = current()
        base = _Limits(values.episodic_max_episodes, values.episodic_recall_limit)
    else:
        base = recorded
    return _Limits(
        base.max_episodes if max_episodes is None else max_episodes,
        base.recall_limit if recall_limit is None else recall_limit,
    )


def _json_string(value: object, name: str, *, nonempty: bool = False) -> str:
    """Require one exact string with a canonical UTF-8 representation."""
    if type(value) is not str:
        raise TypeError(f"{name} must be a string")
    if nonempty and not value:
        raise ValueError(f"{name} must be non-empty")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError("Unicode surrogate is not valid canonical JSON") from exc
    return value


def _copy_json(
    value: object,
    name: str,
    *,
    active: set[int] | None = None,
    nesting: int = 0,
    limit: int | None = None,
) -> JsonValue:
    """Validate and defensively copy one exact finite JSON value.

    `limit` is the memory JSON nesting bound; the top-level call reads it once.
    """
    value_type = type(value)
    if value is None or value_type in (bool, int):
        return cast(JsonValue, value)
    if value_type is str:
        return _json_string(value, name)
    if value_type is float:
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError(f"{name} JSON number must be finite")
        return value
    if value_type not in (list, dict):
        raise TypeError(f"{name} must contain exact JSON values")
    if limit is None:
        limit = current().memory_json_nesting
    if nesting >= limit:
        raise _memory_nesting_error(name, limit)
    active = set() if active is None else active
    identity = id(value)
    if identity in active:
        raise ValueError(f"{name} contains a circular JSON value")
    active.add(identity)
    try:
        if value_type is list:
            return [
                _copy_json(
                    item,
                    f"{name}[{index}]",
                    active=active,
                    nesting=nesting + 1,
                    limit=limit,
                )
                for index, item in enumerate(value)
            ]
        copied: JsonObject = {}
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError(f"{name} JSON object keys must be strings")
            _json_string(key, f"{name} JSON object key")
            copied[key] = _copy_json(
                item, f"{name}.{key}", active=active, nesting=nesting + 1, limit=limit
            )
        return copied
    finally:
        active.remove(identity)


def _copy_object(value: object, name: str) -> JsonObject:
    """Validate and copy one exact JSON object."""
    if type(value) is not dict:
        raise TypeError(f"{name} must be an exact JSON object")
    return cast(JsonObject, _copy_json(value, name))


def _canonical_json(value: JsonValue) -> str:
    """Return one stable compact JSON encoding."""
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _lexical_terms(value: str) -> frozenset[str]:
    """Split text into case-insensitive alphanumeric retrieval terms."""
    terms: list[str] = []
    current: list[str] = []
    for character in value.casefold():
        if character.isalnum():
            current.append(character)
        elif current:
            terms.append("".join(current))
            current = []
    if current:
        terms.append("".join(current))
    return frozenset(terms)


def _occurrence_path_json(path: tuple[MemoryOccurrenceSegment, ...]) -> list[JsonValue]:
    """Encode one root-relative occurrence path."""
    return [
        {
            "index": segment.index,
            "kind": segment.kind.value,
            "static_path": list(segment.static_path),
        }
        for segment in path
    ]


def _occurrence_path_from_json(
    value: object, name: str
) -> tuple[MemoryOccurrenceSegment, ...]:
    """Decode one exact root-relative occurrence path."""
    if type(value) is not list:
        raise TypeError(f"{name} must be an array")
    path: list[MemoryOccurrenceSegment] = []
    for index, raw in enumerate(value):
        item_name = f"{name}[{index}]"
        item = _copy_object(raw, item_name)
        if set(item) != {"index", "kind", "static_path"}:
            raise ValueError(f"{item_name} has invalid fields")
        kind_raw = _json_string(item["kind"], f"{item_name}.kind", nonempty=True)
        try:
            kind = MemoryOccurrenceKind(kind_raw)
        except ValueError as exc:
            raise ValueError(f"{item_name}.kind is invalid") from exc
        static_raw = item["static_path"]
        if type(static_raw) is not list or not static_raw:
            raise ValueError(f"{item_name}.static_path must be a non-empty array")
        static_path = tuple(
            _json_string(part, f"{item_name}.static_path[{part_index}]", nonempty=True)
            for part_index, part in enumerate(static_raw)
        )
        sequence = item["index"]
        if type(sequence) is not int or sequence < 0:
            raise ValueError(f"{item_name}.index is invalid")
        path.append(MemoryOccurrenceSegment(kind, static_path, sequence))
    return tuple(path)


def _interaction_json(interaction: object) -> JsonObject:
    """Encode one closed visible-interaction variant."""
    if type(interaction) is ModelTextInteraction:
        return {"kind": "model_text", "text": interaction.text}
    if type(interaction) is ModelValueInteraction:
        return {"kind": "model_value", "value": interaction.value}
    if type(interaction) is ToolCallInteraction:
        return {
            "arguments": interaction.arguments,
            "kind": "tool_call",
            "name": interaction.name,
        }
    if type(interaction) is ToolResultInteraction:
        return {
            "kind": "tool_result",
            "name": interaction.name,
            "text": interaction.text,
        }
    raise TypeError("episode contains an invalid interaction")


def _interaction_from_json(value: object, name: str) -> VisibleInteraction:
    """Decode one closed visible-interaction variant."""
    data = _copy_object(value, name)
    kind = data.get("kind")
    if kind == "model_text" and set(data) == {"kind", "text"}:
        return ModelTextInteraction(_json_string(data["text"], f"{name}.text"))
    if kind == "model_value" and set(data) == {"kind", "value"}:
        return ModelValueInteraction(data["value"])
    if kind == "tool_call" and set(data) == {"arguments", "kind", "name"}:
        return ToolCallInteraction(
            _json_string(data["name"], f"{name}.name", nonempty=True),
            _copy_object(data["arguments"], f"{name}.arguments"),
        )
    if kind == "tool_result" and set(data) == {"kind", "name", "text"}:
        return ToolResultInteraction(
            _json_string(data["name"], f"{name}.name", nonempty=True),
            _json_string(data["text"], f"{name}.text"),
        )
    raise ValueError(f"{name} has an invalid interaction variant")


def _episode_json(episode: MemoryEpisode) -> JsonObject:
    """Encode every source-neutral field of one accepted episode."""
    return {
        "base_system_prompt": episode.base_system_prompt,
        "execution_seq": episode.execution_seq,
        "failure": episode.failure,
        "inputs": episode.inputs,
        "interactions": [_interaction_json(item) for item in episode.interactions],
        "memory_sections": [
            {"memory": section.memory, "text": section.text}
            for section in episode.memory_sections
        ],
        "occurrence_path": _occurrence_path_json(episode.occurrence_path),
        "output": episode.output,
        "owner_static_path": list(episode.owner_static_path),
        "prompt": episode.prompt,
        "run_id": episode.run_id,
        "status": episode.status.value,
    }


def _episode_from_json(value: object, name: str) -> MemoryEpisode:
    """Decode one exact complete episode through its public carrier."""
    data = _copy_object(value, name)
    fields = {
        "base_system_prompt",
        "execution_seq",
        "failure",
        "inputs",
        "interactions",
        "memory_sections",
        "occurrence_path",
        "output",
        "owner_static_path",
        "prompt",
        "run_id",
        "status",
    }
    if set(data) != fields:
        raise ValueError(f"{name} has invalid fields")
    owner_raw = data["owner_static_path"]
    if type(owner_raw) is not list or not owner_raw:
        raise ValueError(f"{name}.owner_static_path must be a non-empty array")
    owner_path = tuple(
        _json_string(part, f"{name}.owner_static_path[{index}]", nonempty=True)
        for index, part in enumerate(owner_raw)
    )
    status_raw = _json_string(data["status"], f"{name}.status", nonempty=True)
    try:
        status = MemoryEpisodeStatus(status_raw)
    except ValueError as exc:
        raise ValueError(f"{name}.status is invalid") from exc
    sections_raw = data["memory_sections"]
    interactions_raw = data["interactions"]
    if type(sections_raw) is not list or type(interactions_raw) is not list:
        raise TypeError(f"{name} collections must be arrays")
    sections: list[MemorySection] = []
    for index, raw in enumerate(sections_raw):
        section_name = f"{name}.memory_sections[{index}]"
        section = _copy_object(raw, section_name)
        if set(section) != {"memory", "text"}:
            raise ValueError(f"{section_name} has invalid fields")
        sections.append(
            MemorySection(
                _json_string(section["memory"], f"{section_name}.memory", nonempty=True),
                _json_string(section["text"], f"{section_name}.text"),
            )
        )
    inputs = data["inputs"]
    failure = data["failure"]
    if inputs is not None and type(inputs) is not dict:
        raise TypeError(f"{name}.inputs must be an object or null")
    if failure is not None and type(failure) is not dict:
        raise TypeError(f"{name}.failure must be an object or null")
    execution_seq = data["execution_seq"]
    if execution_seq is not None and (
        type(execution_seq) is not int or execution_seq < 0
    ):
        raise ValueError(f"{name}.execution_seq is invalid")
    base_prompt = data["base_system_prompt"]
    prompt = data["prompt"]
    if base_prompt is not None:
        base_prompt = _json_string(base_prompt, f"{name}.base_system_prompt")
    if prompt is not None:
        prompt = _json_string(prompt, f"{name}.prompt")
    return MemoryEpisode(
        _json_string(data["run_id"], f"{name}.run_id", nonempty=True),
        _occurrence_path_from_json(data["occurrence_path"], f"{name}.occurrence_path"),
        owner_path,
        status,
        None if inputs is None else _copy_object(inputs, f"{name}.inputs"),
        data["output"],
        None if failure is None else _copy_object(failure, f"{name}.failure"),
        tuple(
            _interaction_from_json(item, f"{name}.interactions[{index}]")
            for index, item in enumerate(interactions_raw)
        ),
        base_prompt,
        prompt,
        tuple(sections),
        execution_seq=execution_seq,
    )


def _stored_entry(entry_id: str, episode: MemoryEpisode) -> JsonObject:
    """Build one attributed canonical retained entry."""
    return {"entry_id": entry_id, "episode": _episode_json(episode)}


def _validate_entry(value: object, name: str) -> JsonObject:
    """Validate one retained entry and return its canonical defensive copy."""
    data = _copy_object(value, name)
    if set(data) != {"entry_id", "episode"}:
        raise ValueError(f"{name} has invalid fields")
    entry_id = _json_string(data["entry_id"], f"{name}.entry_id", nonempty=True)
    episode = _episode_from_json(data["episode"], f"{name}.episode")
    canonical = _stored_entry(entry_id, episode)
    if _canonical_json(data) != _canonical_json(canonical):
        raise ValueError(f"{name} must be canonical")
    return canonical


def _copy_state(state: _State) -> _State:
    """Defensively copy one immutable instance state."""
    return _State(
        state.limits,
        tuple(_copy_object(episode, "episode") for episode in state.episodes),
    )


def _snapshot_json(
    namespace: str, epoch: int, revision: int, state: _State
) -> JsonObject:
    """Encode one canonical episodic-memory snapshot."""
    return {
        "config": {
            "max_episodes": state.limits.max_episodes,
            "recall_limit": state.limits.recall_limit,
        },
        "episodes": [_copy_object(episode, "episode") for episode in state.episodes],
        "epoch": epoch,
        "format": _SNAPSHOT_FORMAT,
        "namespace": namespace,
        "revision": revision,
        "version": _SNAPSHOT_VERSION,
    }


def _snapshot_error() -> MemoryOperationError:
    """Return the stable snapshot-validation failure."""
    return _operation_error(
        "open_run", "episodic.invalid_snapshot", "recorded memory snapshot is invalid"
    )


def _parse_snapshot(value: object) -> tuple[str, int, int, _State]:
    """Strictly decode one canonical episodic-memory snapshot."""
    try:
        data = _copy_object(value, "snapshot")
        if set(data) != {
            "config",
            "episodes",
            "epoch",
            "format",
            "namespace",
            "revision",
            "version",
        }:
            raise ValueError("snapshot has invalid fields")
        if data["format"] != _SNAPSHOT_FORMAT:
            raise ValueError("snapshot format is unsupported")
        if type(data["version"]) is not int or data["version"] != _SNAPSHOT_VERSION:
            raise ValueError("snapshot version is unsupported")
        namespace = _json_string(data["namespace"], "snapshot.namespace", nonempty=True)
        epoch = data["epoch"]
        revision = data["revision"]
        if type(epoch) is not int or epoch < 0:
            raise ValueError("snapshot.epoch must be a non-negative integer")
        if type(revision) is not int or revision < 0:
            raise ValueError("snapshot.revision must be a non-negative integer")
        config = _copy_object(data["config"], "snapshot.config")
        if set(config) != {"max_episodes", "recall_limit"}:
            raise ValueError("snapshot config has invalid fields")
        limits = _Limits(
            _positive_int(config["max_episodes"], "max_episodes"),
            _positive_int(config["recall_limit"], "recall_limit"),
        )
        episodes = data["episodes"]
        if type(episodes) is not list:
            raise ValueError("snapshot episodes must be an array")
        if len(episodes) > limits.max_episodes:
            raise ValueError("snapshot exceeds its retention limit")
        decoded = tuple(
            _validate_entry(item, f"snapshot.episodes[{index}]")
            for index, item in enumerate(episodes)
        )
        entry_ids = [cast(str, item["entry_id"]) for item in decoded]
        if len(entry_ids) != len(set(entry_ids)):
            raise ValueError("snapshot episode IDs must be unique")
        state = _State(limits, decoded)
        canonical = _snapshot_json(namespace, epoch, revision, state)
        if _canonical_json(data) != _canonical_json(canonical):
            raise ValueError("snapshot must be canonical")
        return namespace, epoch, revision, state
    except MemoryOperationError:
        raise
    except (TypeError, ValueError, RecursionError) as exc:
        raise _snapshot_error() from exc


def _parse_inventory(value: object) -> tuple[
    tuple[str, int, int, _State],
    dict[str, str],
    bool,
    bool,
    bool,
]:
    """Validate complete private storage state before adopting any of it."""
    try:
        data = _copy_object(value, "inventory")
        if set(data) != {
            "version",
            "snapshot",
            "applied_commits",
            "ever_opened",
            "ever_reset",
            "ever_committed",
        }:
            raise ValueError("invalid inventory fields")
        if type(data["version"]) is not int or data["version"] != 1:
            raise ValueError("unsupported inventory version")
        snapshot = _parse_snapshot(data["snapshot"])
        _, epoch, revision, state = snapshot
        opened, reset, committed = (
            data[key] for key in ("ever_opened", "ever_reset", "ever_committed")
        )
        if any(type(flag) is not bool for flag in (opened, reset, committed)):
            raise ValueError("invalid inventory flags")
        commits = _copy_object(data["applied_commits"], "applied_commits")
        for commit_id, digest in commits.items():
            _json_string(commit_id, "commit_id", nonempty=True)
            _json_string(digest, "digest", nonempty=True)
            if len(digest) != 64 or any(
                char not in "0123456789abcdef" for char in digest
            ):
                raise ValueError("invalid commit digest")
        if (
            not opened
            and (state.episodes or revision or committed or commits)
            or not committed
            and (revision or commits)
            or reset
            and epoch == 0
            or not opened
            and not reset
            and epoch != 0
            or committed
            and revision == 0
            and not reset
        ):
            raise ValueError("inconsistent inventory flags")
        return snapshot, cast(dict[str, str], commits), opened, reset, committed
    except (TypeError, ValueError, RecursionError, MemoryOperationError):
        raise _operation_error(
            "storage", "episodic.invalid_storage", "stored episodic state is invalid"
        ) from None


class EpisodicMemory:
    """Retain bounded canonical episodes, optionally in a local SQLite file.

    Examples:
        Inject one instance under the name selected by an authored
        ``long_term_memory: episodic`` declaration. Forward execution only
        retains history; the explicit backward call commits it for later runs.

        ```python
        from yian import run_flow
        from yian.memory.episodic import EpisodicMemory

        memory = EpisodicMemory(max_episodes=100, recall_limit=10)
        result = run_flow(
            flow,
            inputs,
            memory={"episodic": memory},
            learnable=True,
        )
        result.backward()
        ```

        Use ``with EpisodicMemory(storage="memory.sqlite") as memory:`` to retain
        episodes across restarts and explicitly release the single file owner.
        The file is opened at the first operation, not at construction.
        Closing idle run handles is not required before closing this backend.
    """

    def __init__(
        self,
        max_episodes: int | None = None,
        recall_limit: int | None = None,
        *,
        storage: str | os.PathLike[str] | None = None,
        create_parent: bool = False,
    ) -> None:
        """Create one isolated episodic-memory namespace.

        Args:
            max_episodes (`int | None`, defaults to `None`):
                Maximum retained episodes. None adopts the stored or snapshot
                value, else `YIAN_EPISODIC_MAX_EPISODES` (100).
            recall_limit (`int | None`, defaults to `None`):
                Maximum episodes returned by contextual recall. None adopts the
                stored or snapshot value, else `YIAN_EPISODIC_RECALL_LIMIT` (10).
            storage (`str | PathLike[str] | None`, defaults to `None`):
                Optional local SQLite file. Nothing is opened until the first
                operation; this backend then owns the file until close. None is
                process-local.
            create_parent (`bool`, defaults to `False`):
                Create the storage file's parent directory at first open.

        Raises:
            TypeError: Raised when either limit is not an exact integer.
            ValueError: Raised when either limit is not positive.
            MemoryOperationError: ``storage`` is not a local file path
                (`storage.path`).
        """
        if max_episodes is not None:
            _positive_int(max_episodes, "max_episodes")
        if recall_limit is not None:
            _positive_int(recall_limit, "recall_limit")
        self._requested = (max_episodes, recall_limit)
        self._lock = RLock()
        self._pid = os.getpid()
        self._storage: SQLiteStorage | None = None
        self._closed = False
        self._active_operations = 0
        self._namespace = uuid4().hex
        self._epoch = 0
        self._revision = 0
        self._state = _State(_resolve_limits(self._requested, None, "storage"), ())
        self._applied_commits: dict[str, str] = {}
        self._ever_opened = False
        self._ever_reset = False
        self._ever_committed = False
        if type(create_parent) is not bool:
            raise TypeError("create_parent must be a bool")
        self._storage_path = None if storage is None else check_storage_path(storage)
        self._create_parent = create_parent

    def _open_storage(self) -> None:
        """Open and load durable state once; the caller holds ``self._lock``."""
        if self._storage is not None or self._storage_path is None:
            return
        self._storage = SQLiteStorage(
            self._storage_path, _POLICY_NAME, create_parent=self._create_parent
        )
        try:
            saved = self._storage.load()
            if saved is None:
                self._save_state(
                    _snapshot_json(
                        self._namespace, self._epoch, self._revision, self._state
                    )
                )
            else:
                snapshot, commits, opened, reset, committed = _parse_inventory(
                    saved
                )
                saved_limits = snapshot[3].limits
                _resolve_limits(self._requested, saved_limits, "storage")
                self._namespace, self._epoch, self._revision, self._state = (
                    snapshot
                )
                self._applied_commits = commits
                self._ever_opened = opened
                self._ever_reset = reset
                self._ever_committed = committed
        except BaseException as primary:
            try:
                self._storage.close()
            except BaseException as cleanup:
                if hasattr(primary, "add_note"):
                    primary.add_note(
                        f"EpisodicMemory cleanup failed: {type(cleanup).__name__}"
                    )
            # Forget the failed handle so a later operation may retry the open.
            self._storage = None
            raise

    def _check_pid(self) -> None:
        """Reject inherited durable handles before touching any Python lock."""
        if self._storage_path is not None and os.getpid() != self._pid:
            raise MemoryOperationError(
                "storage",
                "storage.forked",
                "Memory storage cannot be used after fork.",
                False,
            )

    @contextmanager
    def _operation(self) -> Iterator[None]:
        """Hold one operation independently of run-handle lifetime."""
        self._check_pid()
        with self._lock:
            if self._closed:
                raise _operation_error(
                    "session", "episodic.closed", "memory backend is closed"
                )
            self._open_storage()
            if self._storage is not None:
                self._storage.ensure_open()
            self._active_operations += 1
            try:
                yield
            finally:
                self._active_operations -= 1

    def _save_state(
        self,
        snapshot: JsonObject,
        *,
        applied_commits: dict[str, str] | None = None,
        ever_opened: bool | None = None,
        ever_reset: bool | None = None,
        ever_committed: bool | None = None,
    ) -> None:
        """Validate and persist a complete candidate before live RAM publication."""
        if self._storage is None:
            return
        inventory = {
            "version": 1,
            "snapshot": snapshot,
            "applied_commits": (
                self._applied_commits if applied_commits is None else applied_commits
            ),
            "ever_opened": self._ever_opened if ever_opened is None else ever_opened,
            "ever_reset": self._ever_reset if ever_reset is None else ever_reset,
            "ever_committed": (
                self._ever_committed if ever_committed is None else ever_committed
            ),
        }
        _parse_inventory(inventory)
        self._storage.save(inventory)

    def __enter__(self) -> EpisodicMemory:
        """Enter this still-open backend; exit releases its storage ownership.

        Returns:
            `EpisodicMemory`: This backend.

        Raises:
            `MemoryOperationError`: The backend or storage handle is unusable.
        """
        with self._operation():
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
            if hasattr(exc, "add_note"):
                exc.add_note(f"EpisodicMemory cleanup failed: {type(cleanup).__name__}")

    def close(self) -> None:
        """Release storage and invalidate idle handles; repeated closes are inert.

        Raises:
            MemoryOperationError: Active work makes close busy without waiting or
                interrupting it; an inherited handle or storage cleanup fails.
        """
        self._check_pid()
        if not self._lock.acquire(blocking=False):
            raise MemoryOperationError(
                "close", "episodic.busy", "memory backend is active", True
            )
        try:
            if self._closed:
                return
            if self._active_operations:
                raise MemoryOperationError(
                    "close", "episodic.busy", "memory backend is active", True
                )
            if self._storage is not None:
                self._storage.close()
            self._closed = True
        finally:
            self._lock.release()

    def open_run(self, *, snapshot: JsonObject | None = None) -> _EpisodicSession:
        """Open a run session over one pinned memory snapshot.

        Args:
            snapshot (`JsonObject`, *optional*, defaults to `None`):
                Recorded episodic snapshot to restore or replay.

        Returns:
            `_EpisodicSession`: The isolated run-scoped session.

        Raises:
            MemoryOperationError: The snapshot is invalid or incompatible, the
                backend is closed, or durable state cannot be opened, read, or
                saved.
        """
        with self._operation():
            parsed = None if snapshot is None else _parse_snapshot(snapshot)
            if parsed is not None:
                recorded_limits = parsed[3].limits
                _resolve_limits(self._requested, recorded_limits, "open_run")
            adopt = False
            if parsed is None:
                namespace = self._namespace
                epoch = self._epoch
                revision = self._revision
                state = _copy_state(self._state)
            else:
                namespace, epoch, revision, state = parsed
                if namespace != self._namespace:
                    pristine = (
                        not self._ever_opened
                        and not self._ever_reset
                        and not self._ever_committed
                        and not self._state.episodes
                        and self._epoch == 0
                        and self._revision == 0
                    )
                    if not pristine:
                        raise _snapshot_error()
                    adopt = True
                else:
                    if epoch != self._epoch or revision > self._revision:
                        raise _snapshot_error()
                    if revision == self._revision and state != self._state:
                        raise _snapshot_error()
            if adopt or not self._ever_opened:
                adopted_state = _copy_state(state) if adopt else self._state
                candidate = (
                    _snapshot_json(namespace, epoch, revision, state)
                    if adopt
                    else _snapshot_json(
                        self._namespace, self._epoch, self._revision, self._state
                    )
                )
                self._save_state(
                    candidate,
                    ever_opened=True,
                    ever_committed=revision > 0 if adopt else self._ever_committed,
                )
                if adopt:
                    self._namespace = namespace
                    self._epoch = epoch
                    self._revision = revision
                    self._state = adopted_state
                    self._ever_committed = revision > 0
                self._ever_opened = True
            return _EpisodicSession(self, namespace, epoch, revision, state)

    def reset(self) -> None:
        """Clear episodes and begin a new generation, retaining commit tombstones.

        Raises:
            `MemoryOperationError`: The backend is closed, or durable state
                cannot be opened or saved.
        """
        with self._operation():
            cleared = _State(self._state.limits, ())
            self._save_state(
                _snapshot_json(self._namespace, self._epoch + 1, 0, cleared),
                ever_reset=True,
            )
            self._state = cleared
            self._epoch += 1
            self._revision = 0
            self._ever_reset = True

    def rewrite_query(self, context: RecallContext) -> str:
        """Derive deterministic lexical retrieval text from visible context."""
        with self._operation():
            if type(context) is not RecallContext:
                raise TypeError("context must be a RecallContext")
            return "\n".join(
                (
                    context.base_system_prompt,
                    context.prompt,
                    _canonical_json(context.inputs),
                )
            )


class _EpisodicSession:
    """One run-scoped view over a pinned episodic-memory generation."""

    def __init__(
        self,
        memory: EpisodicMemory,
        namespace: str,
        epoch: int,
        revision: int,
        state: _State,
    ) -> None:
        self._memory = memory
        self._namespace = namespace
        self._epoch = epoch
        self._revision = revision
        self._state = state
        self._closed = False

    def _ensure_open(self) -> None:
        """Reject every operation after the run boundary closes."""
        if self._closed:
            raise _operation_error(
                "session", "episodic.session_closed", "memory session is closed"
            )

    @property
    def snapshot(self) -> JsonObject:
        """Return a defensive public snapshot pinned for this run."""
        with self._memory._operation():
            self._ensure_open()
            return _snapshot_json(
                self._namespace, self._epoch, self._revision, self._state
            )

    def recall(self, context: RecallContext | None = None) -> str:
        """Render pinned retained history, optionally ranked by lexical relevance."""
        with self._memory._operation():
            self._ensure_open()
            if context is not None and type(context) is not RecallContext:
                raise _operation_error(
                    "recall", "episodic.invalid_context", "context is invalid"
                )
            entries = self._state.episodes
            if context is not None:
                try:
                    query = self._memory.rewrite_query(context)
                except Exception as exc:
                    raise _operation_error(
                        "recall", "episodic.invalid_query", "query rewriting failed"
                    ) from exc
                self._memory._check_pid()
                if type(query) is not str:
                    raise _operation_error(
                        "recall",
                        "episodic.invalid_query",
                        "query rewriting must return text",
                    )
                query_terms = _lexical_terms(query)
                ranked = sorted(
                    enumerate(entries),
                    key=lambda pair: (
                        -len(query_terms & _lexical_terms(_canonical_json(pair[1]))),
                        -pair[0],
                    ),
                )
                entries = tuple(
                    entry for _, entry in ranked[: self._state.limits.recall_limit]
                )
            return "\n".join(_canonical_json(entry) for entry in entries)

    def learn(self, context: LearnContext) -> list[MemoryMutation]:
        """Propose one deterministic append for every episode in the batch."""
        with self._memory._operation():
            self._ensure_open()
            if type(context) is not LearnContext:
                raise _operation_error(
                    "learn", "episodic.invalid_context", "learning context is invalid"
                )
            self._assert_current("learn")
            try:
                context = LearnContext(
                    context.run_id, context.memory, context.episodes, context.feedback
                )
            except (TypeError, ValueError) as exc:
                raise _operation_error(
                    "learn", "episodic.invalid_context", "learning context is invalid"
                ) from exc
            mutations: list[MemoryMutation] = []
            for ordinal, episode in enumerate(context.episodes):
                identity: JsonObject = {
                    "episode": _episode_json(episode),
                    "memory": context.memory,
                    "ordinal": ordinal,
                    "run_id": context.run_id,
                }
                entry_id = hashlib.sha256(
                    _canonical_json(identity).encode("utf-8")
                ).hexdigest()
                mutations.append(
                    MemoryMutation(
                        entry_id,
                        {"entry": _stored_entry(entry_id, episode), "operation": "append"},
                    )
                )
            return mutations

    def __enter__(self) -> _EpisodicSession:
        """Enter this still-open session."""
        with self._memory._operation():
            self._ensure_open()
            return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close this session without suppressing an exception."""
        try:
            self.close()
        except BaseException as cleanup:
            if exc is None:
                raise
            if hasattr(exc, "add_note"):
                exc.add_note(
                    f"Episodic session cleanup failed: {type(cleanup).__name__}"
                )

    def close(self) -> None:
        """Idempotently close this session, without closing its backend."""
        self._memory._check_pid()
        with self._memory._lock:
            self._closed = True

    def _assert_current(self, stage: str) -> None:
        """Require this session to remain the current memory generation."""
        if (
            self._namespace != self._memory._namespace
            or self._epoch != self._memory._epoch
            or self._revision != self._memory._revision
        ):
            raise _operation_error(
                stage, "episodic.stale_generation", "memory session is stale"
            )

    def _batch_digest(self, mutations: tuple[MemoryMutation, ...]) -> str:
        """Compute a canonical digest for one exact mutation tuple."""
        if type(mutations) is not tuple:
            raise _operation_error(
                "commit", "episodic.invalid_batch", "mutations must be an exact tuple"
            )
        encoded: list[JsonObject] = []
        for mutation in mutations:
            if type(mutation) is not MemoryMutation:
                raise _operation_error(
                    "commit",
                    "episodic.invalid_batch",
                    "mutation batch contains an invalid value",
                )
            encoded.append(
                {
                    "mutation_id": mutation.mutation_id,
                    "payload": mutation.payload,
                }
            )
        try:
            material = _canonical_json(cast(JsonValue, encoded))
        except (TypeError, ValueError) as exc:
            raise _operation_error(
                "commit",
                "episodic.invalid_batch",
                "mutation batch is not canonical JSON",
            ) from exc
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    def commit(self, commit_id: str, mutations: tuple[MemoryMutation, ...]) -> None:
        """Atomically append one exact idempotent mutation batch."""
        with self._memory._operation():
            self._ensure_open()
            try:
                commit_id = _json_string(commit_id, "commit_id", nonempty=True)
            except (TypeError, ValueError) as exc:
                raise _operation_error(
                    "commit", "episodic.invalid_commit_id", "commit id is invalid"
                ) from exc
            digest = self._batch_digest(mutations)
            applied = self._memory._applied_commits.get(commit_id)
            if applied is not None:
                if applied == digest:
                    return
                raise _operation_error(
                    "commit",
                    "episodic.commit_collision",
                    "commit id collision with a different batch",
                )
            self._assert_current("commit")
            state = _copy_state(self._memory._state)
            if state.limits != self._state.limits:
                raise _operation_error(
                    "commit",
                    "episodic.config_conflict",
                    "current memory limits changed",
                )
            seen_mutations: set[str] = set()
            additions: list[JsonObject] = []
            try:
                for index, mutation in enumerate(mutations):
                    if mutation.mutation_id in seen_mutations:
                        raise ValueError("mutation IDs must be unique")
                    seen_mutations.add(mutation.mutation_id)
                    payload = _copy_object(
                        mutation.payload, f"mutations[{index}].payload"
                    )
                    if (
                        set(payload) != {"entry", "operation"}
                        or payload["operation"] != "append"
                    ):
                        raise ValueError(
                            "mutation payload must be a closed append operation"
                        )
                    entry = _validate_entry(
                        payload["entry"], f"mutations[{index}].payload.entry"
                    )
                    if entry["entry_id"] != mutation.mutation_id:
                        raise ValueError("mutation and entry IDs must match")
                    additions.append(entry)
                existing_ids = {
                    cast(str, entry["entry_id"]) for entry in state.episodes
                }
                incoming_ids = {cast(str, entry["entry_id"]) for entry in additions}
                if existing_ids & incoming_ids:
                    raise ValueError("episode entry is already retained")
                retained = (state.episodes + tuple(additions))[
                    -state.limits.max_episodes :
                ]
                state = _State(state.limits, retained)
            except (TypeError, ValueError, RecursionError) as exc:
                raise _operation_error(
                    "commit", "episodic.invalid_batch", "mutation batch is invalid"
                ) from exc
            commits = {**self._memory._applied_commits, commit_id: digest}
            revision = self._memory._revision + 1
            self._memory._save_state(
                _snapshot_json(
                    self._memory._namespace, self._memory._epoch, revision, state
                ),
                applied_commits=commits,
                ever_committed=True,
            )
            self._memory._state = state
            self._memory._applied_commits = commits
            self._memory._revision = revision
            self._memory._ever_committed = True
