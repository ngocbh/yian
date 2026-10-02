"""Reflective memory with snapshot-isolated sessions and optional SQLite storage.

The backend keeps atomic lessons in memory, retrieves them deterministically,
and asks one tool-free structured model call to propose frame-local mutations.
It deliberately knows nothing about compiled graphs, nodes, or the scheduler.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import os
import re
from threading import RLock
from types import TracebackType
from typing import Any, Iterator, Literal, cast
from uuid import uuid4

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agent_composer.llm_clients.connection import legacy_connection_config, resolve_model_config, split_model_config
from agent_composer.llm_clients import LLMConfig, model_from_config
from agent_composer.memory.contracts import (
    JsonObject,
    JsonValue,
    LearnContext,
    MemoryEpisode,
    MemoryFrameRef,
    MemoryFrameSpec,
    MemoryMutation,
    MemoryOperationError,
    ModelTextInteraction,
    ModelValueInteraction,
    RecallContext,
    ToolCallInteraction,
    ToolResultInteraction,
    VisibleInteraction,
    _MAX_JSON_NESTING,
)
from agent_composer.memory.storage import SQLiteStorage

_SNAPSHOT_FORMAT = "agent_composer.reflective_memory"
_SNAPSHOT_VERSION = 2
_POLICY_NAME = "reflective"
_MAX_REFLECTION_PROMPT_CHARS = 32_000
_MAX_FRAME_MATERIAL_CHARS = 1_000
_MAX_LIMIT_MATERIAL_CHARS = 256
_MAX_LESSON_MATERIAL_CHARS = 8_000
_MAX_EPISODE_MATERIAL_CHARS = 20_000
_WORD = re.compile(r"\w+", re.UNICODE)


class _ReflectionAction(BaseModel):
    """One closed model-proposed lesson action."""

    model_config = ConfigDict(extra="forbid", strict=True)

    action: Literal["add", "revise", "remove", "noop"]
    lesson_id: str | None = None
    text: str | None = None


class _ReflectionResponse(BaseModel):
    """Strict structured result from one reflective model call."""

    model_config = ConfigDict(extra="forbid", strict=True)

    actions: list[_ReflectionAction] = Field(min_length=1)


@dataclass(frozen=True, slots=True)
class _Lesson:
    """One immutable atomic lesson in insertion order."""

    lesson_id: str
    text: str
    source: JsonObject


@dataclass(frozen=True, slots=True)
class _Limits:
    """Effective lesson and recall limits of one reflective instance."""

    max_items: int
    recall_limit: int


@dataclass(frozen=True, slots=True)
class _State:
    """The limits and ordered lessons of one reflective instance."""

    limits: _Limits
    lessons: tuple[_Lesson, ...]


_DEFAULT_LIMITS = _Limits(100, 20)


def _operation_error(
    stage: str,
    code: str,
    message: str,
    *,
    retryable: bool = False,
) -> MemoryOperationError:
    """Build one stable typed backend failure."""
    return MemoryOperationError(stage, code, message, retryable)


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
    max_items, recall_limit = requested
    if recorded is not None and (
        max_items not in (None, recorded.max_items)
        or recall_limit not in (None, recorded.recall_limit)
    ):
        raise _operation_error(
            stage,
            "reflective.config_conflict",
            "configured limits differ from the recorded limits",
            retryable=False,
        )
    base = _DEFAULT_LIMITS if recorded is None else recorded
    return _Limits(
        base.max_items if max_items is None else max_items,
        base.recall_limit if recall_limit is None else recall_limit,
    )


def _validated_json_string(value: object, name: str) -> str:
    """Return one exact string with a canonical UTF-8 representation."""
    if type(value) is not str:
        raise TypeError(f"{name} must be a string")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError("Unicode surrogate is not valid canonical JSON") from exc
    return value


def _copy_json(
    value: object,
    name: str = "value",
    *,
    active: set[int] | None = None,
    nesting: int = 0,
) -> JsonValue:
    """Validate and defensively copy one exact finite JSON value."""
    value_type = type(value)
    if value is None or value_type in (bool, int):
        return cast(JsonValue, value)
    if value_type is str:
        return _validated_json_string(value, name)
    if value_type is float:
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError(f"{name} JSON number must be finite")
        return value
    if value_type not in (list, dict):
        raise TypeError(f"{name} must contain exact JSON values")
    if nesting >= _MAX_JSON_NESTING:
        raise ValueError(
            f"{name} JSON nesting exceeds the conservative wire-safe limit "
            f"of {_MAX_JSON_NESTING}"
        )

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
                )
                for index, item in enumerate(value)
            ]
        copied: JsonObject = {}
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError(f"{name} JSON object keys must be strings")
            _validated_json_string(key, f"{name} JSON object key")
            copied[key] = _copy_json(
                item,
                f"{name}.{key}",
                active=active,
                nesting=nesting + 1,
            )
        return copied
    finally:
        active.remove(identity)


def _copy_json_object(value: object, name: str) -> JsonObject:
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


def _nonempty_string(value: object, name: str) -> str:
    """Require one exact nonempty string."""
    value = _validated_json_string(value, name)
    if not value:
        raise ValueError(f"{name} must be non-empty")
    return value


def _frame_ref_json(ref: MemoryFrameRef) -> JsonObject:
    """Encode one stable frame identity as canonical JSON."""
    return {
        "owner": ref.owner.value,
        "root_flow_id": ref.root_flow_id,
        "static_path": list(ref.static_path),
    }


def _source_json(run_id: str) -> JsonObject:
    """Return stable source metadata for one proposed lesson mutation."""
    return {"run_id": run_id}


def _source_from_json(value: object, name: str) -> JsonObject:
    """Decode one exact canonical lesson source."""
    source = _copy_json_object(value, name)
    if set(source) != {"run_id"}:
        raise ValueError(f"{name} has invalid fields")
    canonical: JsonObject = {
        "run_id": _nonempty_string(source["run_id"], f"{name}.run_id")
    }
    if _canonical_json(source) != _canonical_json(canonical):
        raise ValueError(f"{name} must be canonical")
    return canonical


def _copy_lesson(lesson: _Lesson) -> _Lesson:
    """Copy the sole mutable component of an immutable lesson."""
    return _Lesson(
        lesson.lesson_id, lesson.text, _copy_json_object(lesson.source, "source")
    )


def _copy_state(state: _State) -> _State:
    """Defensively copy one internal instance state."""
    return _State(state.limits, tuple(_copy_lesson(item) for item in state.lessons))


def _state_json(namespace: str, epoch: int, revision: int, state: _State) -> JsonObject:
    """Encode canonical lessons and identity without host model settings."""
    return {
        "config": {
            "max_items": state.limits.max_items,
            "recall_limit": state.limits.recall_limit,
        },
        "epoch": epoch,
        "lessons": [
            {
                "id": lesson.lesson_id,
                "source": _copy_json_object(lesson.source, "source"),
                "text": lesson.text,
            }
            for lesson in state.lessons
        ],
        "namespace": namespace,
        "revision": revision,
    }


def _snapshot_json(
    namespace: str,
    epoch: int,
    revision: int,
    state: _State,
    llm_config: JsonObject,
) -> JsonObject:
    """Encode one canonical session snapshot with its pinned model settings."""
    return {
        **_state_json(namespace, epoch, revision, state),
        "format": _SNAPSHOT_FORMAT,
        "llm_config": _copy_json_object(llm_config, "llm_config"),
        "version": _SNAPSHOT_VERSION,
    }


def _snapshot_error(message: str) -> MemoryOperationError:
    """Return a stable final snapshot-validation failure."""
    return _operation_error(
        "open_run", "reflective.invalid_snapshot", message, retryable=False
    )


def _parse_state(value: object) -> tuple[str, int, int, _State]:
    """Decode the shared semantic state without requiring a model configuration."""
    data = _copy_json_object(value, "state")
    if set(data) != {"config", "epoch", "lessons", "namespace", "revision"}:
        raise ValueError("state has invalid fields")
    namespace = _nonempty_string(data["namespace"], "snapshot.namespace")
    epoch = data["epoch"]
    revision = data["revision"]
    if type(epoch) is not int or epoch < 0:
        raise ValueError("snapshot.epoch must be a non-negative integer")
    if type(revision) is not int or revision < 0:
        raise ValueError("snapshot.revision must be a non-negative integer")

    config = _copy_json_object(data["config"], "snapshot.config")
    if set(config) != {"max_items", "recall_limit"}:
        raise ValueError("snapshot.config has invalid fields")
    limits = _Limits(
        _positive_int(config["max_items"], "snapshot.config.max_items"),
        _positive_int(config["recall_limit"], "snapshot.config.recall_limit"),
    )
    lessons_raw = data["lessons"]
    if type(lessons_raw) is not list:
        raise TypeError("snapshot.lessons must be an array")
    lessons: list[_Lesson] = []
    lesson_ids: set[str] = set()
    for lesson_index, raw_lesson in enumerate(lessons_raw):
        lesson = _copy_json_object(raw_lesson, f"snapshot.lessons[{lesson_index}]")
        if set(lesson) != {"id", "source", "text"}:
            raise ValueError("snapshot lesson has invalid fields")
        lesson_id = _nonempty_string(lesson["id"], "snapshot lesson id")
        text = _nonempty_string(lesson["text"], "snapshot lesson text")
        if not text.strip():
            raise ValueError("snapshot lesson text must not be blank")
        if lesson_id in lesson_ids:
            raise ValueError("snapshot lesson ids must be unique")
        lesson_ids.add(lesson_id)
        source = _source_from_json(lesson["source"], "snapshot lesson source")
        lessons.append(_Lesson(lesson_id, text, source))
    if len(lessons) > limits.max_items:
        raise ValueError("snapshot exceeds max_items")
    return namespace, epoch, revision, _State(limits, tuple(lessons))


def _parse_snapshot(
    value: object,
) -> tuple[str, int, int, _State, JsonObject]:
    """Strictly decode one canonical reflective-memory snapshot."""
    try:
        data = _copy_json_object(value, "snapshot")
        expected = {
            "config",
            "epoch",
            "format",
            "lessons",
            "llm_config",
            "namespace",
            "revision",
            "version",
        }
        if set(data) != expected:
            raise ValueError("snapshot has invalid fields")
        if type(data["format"]) is not str or data["format"] != _SNAPSHOT_FORMAT:
            raise ValueError("snapshot format is unsupported")
        if type(data["version"]) is not int or data["version"] != _SNAPSHOT_VERSION:
            raise ValueError("snapshot version is unsupported")
        llm_config = _copy_json_object(data["llm_config"], "snapshot.llm_config")
        # Preserve the exact historical snapshot for its recorded digest while
        # validating inert defaults removed from the current public carrier.
        retired_nulls = {key: None for key in ("sandbox", "cwd") if key in llm_config and llm_config[key] is None}
        validated_llm, connection = split_model_config({key: value for key, value in llm_config.items() if key not in retired_nulls})
        complete_llm = cast(
            JsonObject, validated_llm.model_dump(mode="json", exclude_none=False)
        )
        complete_llm.update(retired_nulls)
        if connection is not None:
            complete_llm["_connection"] = connection
        if _canonical_json(llm_config) != _canonical_json(complete_llm):
            raise ValueError("snapshot.llm_config must be complete and canonical")
        provider = _nonempty_string(
            complete_llm.get("provider"), "snapshot.llm_config.provider"
        )
        model = _nonempty_string(complete_llm.get("model"), "snapshot.llm_config.model")
        if not provider.strip() or not model.strip():
            raise ValueError("snapshot.llm_config must pin nonblank provider and model")

        namespace, epoch, revision, state = _parse_state(
            {
                key: data[key]
                for key in ("config", "epoch", "lessons", "namespace", "revision")
            }
        )
        return namespace, epoch, revision, state, complete_llm
    except MemoryOperationError:
        raise
    except (TypeError, ValueError, ValidationError, RecursionError) as exc:
        raise _snapshot_error("recorded memory snapshot is invalid") from exc


def _episode_json(episode: MemoryEpisode) -> JsonObject:
    """Encode one immutable episode for the reflector prompt."""
    occurrence_path: list[JsonValue] = []
    for segment in episode.occurrence.occurrence_path:
        occurrence_path.append(
            {
                "index": segment.index,
                "kind": segment.kind.value,
                "static_path": list(segment.static_path),
            }
        )
    return {
        "base_system_prompt": episode.base_system_prompt,
        "execution_seq": episode.execution_seq,
        "failure": episode.failure,
        "inputs": episode.inputs,
        "interactions": [
            _visible_interaction_json(interaction)
            for interaction in episode.interactions
        ],
        "memory_sections": [
            {
                "frame_ref": _frame_ref_json(section.frame_ref),
                "text": section.text,
            }
            for section in episode.memory_sections
        ],
        "occurrence": {
            "frame_ref": _frame_ref_json(episode.occurrence.frame_ref),
            "occurrence_path": occurrence_path,
        },
        "output": episode.output,
        "owner_static_path": list(episode.owner_static_path),
        "prompt": episode.prompt,
        "run_id": episode.run_id,
        "status": episode.status.value,
    }


def _visible_interaction_json(interaction: VisibleInteraction) -> JsonObject:
    """Encode one closed visible interaction for the reflector prompt."""
    if type(interaction) is ModelTextInteraction:
        return {"kind": "model_text", "text": interaction.text}
    if type(interaction) is ModelValueInteraction:
        return {"kind": "model_value", "value": interaction.value}
    if type(interaction) is ToolCallInteraction:
        return {
            "kind": "tool_call",
            "name": interaction.name,
            "arguments": interaction.arguments,
        }
    if type(interaction) is ToolResultInteraction:
        return {
            "kind": "tool_result",
            "name": interaction.name,
            "text": interaction.text,
        }
    raise TypeError("unsupported visible interaction")


def _bounded_json_material(value: JsonValue, limit: int) -> str:
    """Encode JSON under a deterministic cap while retaining both ends."""
    material = _canonical_json(value)
    if len(material) <= limit:
        return material
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()
    chunk_chars = max(0, (limit - 200) // 2)
    summary: JsonObject = {
        "original_chars": len(material),
        "prefix": material[:chunk_chars],
        "sha256": digest,
        "suffix": material[-chunk_chars:] if chunk_chars else "",
        "truncated": True,
    }
    encoded = _canonical_json(summary)
    while len(encoded) > limit and chunk_chars:
        reduction = max(1, (len(encoded) - limit + 1) // 2)
        chunk_chars = max(0, chunk_chars - reduction)
        summary["prefix"] = material[:chunk_chars]
        summary["suffix"] = material[-chunk_chars:] if chunk_chars else ""
        encoded = _canonical_json(summary)
    return encoded


def _bounded_episode_material(context: LearnContext) -> str:
    """Return deterministic episode material within its reserved prompt budget."""
    payload: JsonObject = {
        "episodes": [_episode_json(episode) for episode in context.episodes],
        "feedback": list(context.feedback),
    }
    return _bounded_json_material(payload, _MAX_EPISODE_MATERIAL_CHARS)


def _mutation_digest_payload(
    context: LearnContext,
    index: int,
    payload: JsonObject,
) -> JsonObject:
    """Return the complete deterministic mutation identity material."""
    return {
        "frame_ref": _frame_ref_json(context.frame_ref),
        "index": index,
        "payload": payload,
        "run_id": context.run_id,
    }


def _parse_inventory(value: object) -> tuple[
    tuple[str, int, int, _State],
    dict[str, str],
    bool,
    bool,
    bool,
]:
    """Validate the complete private inventory before restoring any RAM state."""
    try:
        data = _copy_json_object(value, "inventory")
        if set(data) != {
            "version",
            "state",
            "applied_commits",
            "ever_opened",
            "ever_reset",
            "ever_committed",
        }:
            raise ValueError("invalid inventory fields")
        if type(data["version"]) is not int or data["version"] != 1:
            raise ValueError("unsupported inventory version")
        state = _parse_state(data["state"])
        _, epoch, revision, instance = state
        opened, reset, committed = (
            data[key] for key in ("ever_opened", "ever_reset", "ever_committed")
        )
        if any(type(flag) is not bool for flag in (opened, reset, committed)):
            raise ValueError("invalid inventory flags")
        commits = _copy_json_object(data["applied_commits"], "applied_commits")
        for commit_id, digest in commits.items():
            _nonempty_string(commit_id, "commit_id")
            _nonempty_string(digest, "digest")
            if len(digest) != 64 or any(
                char not in "0123456789abcdef" for char in digest
            ):
                raise ValueError("invalid commit digest")
        has_lessons = bool(instance.lessons)
        if (
            not opened
            and (has_lessons or revision or committed or commits)
            or not committed
            and (revision or commits or has_lessons)
            or reset
            and epoch == 0
            or not opened
            and not reset
            and epoch != 0
            or committed
            and revision == 0
            and not reset
            and not has_lessons
        ):
            raise ValueError("inconsistent inventory flags")
        return state, cast(dict[str, str], commits), opened, reset, committed
    except (TypeError, ValueError, RecursionError) as exc:
        raise _operation_error(
            "storage",
            "reflective.invalid_storage",
            "stored reflective state is invalid",
        ) from exc


class ReflectiveMemory:
    """Learn atomic lessons through reflection, optionally in a local SQLite file.

    Examples:
        ``with ReflectiveMemory(storage="memory.sqlite") as memory:`` retains
        lessons across restarts and releases the single file owner on exit.
        Idle run handles do not need to close before the backend.
    """

    def __init__(
        self,
        llm_config: JsonObject | None = None,
        *,
        max_items: int | None = None,
        recall_limit: int | None = None,
        storage: str | os.PathLike[str] | None = None,
    ) -> None:
        """Create or reopen one reflective-memory namespace.

        Args:
            llm_config (`JsonObject | None`, defaults to `None`): Public model
                settings for new sessions, filled from ambient defaults when absent.
                Recorded sessions retain their exact pinned model configuration.
                ``api_key`` is rejected, so the learner always authenticates with
                environment or saved-connection credentials.
            max_items (`int | None`, defaults to `None`): Maximum retained
                lessons. None adopts the stored or snapshot value, else 100.
            recall_limit (`int | None`, defaults to `None`): Maximum lessons
                returned by contextual recall. None adopts the stored or snapshot
                value, else 20.
            storage (`str | PathLike[str] | None`, defaults to `None`): Optional
                local SQLite file with an existing parent, owned until close or
                context exit. None keeps state process-local.

        Raises:
            TypeError: Limits or model settings have invalid types.
            ValueError: Limits or model settings are invalid.
            MemoryOperationError: Storage is unavailable, occupied, or invalid,
                or an explicit limit differs from the stored limits
                (`reflective.config_conflict`).
        """
        if max_items is not None:
            _positive_int(max_items, "max_items")
        if recall_limit is not None:
            _positive_int(recall_limit, "recall_limit")
        self._requested = (max_items, recall_limit)
        raw_config = (
            {} if llm_config is None else _copy_json_object(llm_config, "llm_config")
        )
        validated = LLMConfig.model_validate(raw_config, strict=True)
        if validated.api_key is not None:
            # Snapshots record this config; a key would be dropped or persisted.
            raise ValueError(
                "llm_config.api_key is not supported by reflective memory, which "
                "records its model configuration"
            )
        if validated.model is not None and not validated.model.strip():
            raise ValueError("llm_config.model must be non-empty")
        self._llm_config = cast(
            JsonObject, validated.model_dump(mode="json", exclude_none=True)
        )
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
        if storage is not None:
            self._storage = SQLiteStorage(storage, _POLICY_NAME)
            try:
                saved = self._storage.load()
                if saved is None:
                    self._save_state(
                        _state_json(
                            self._namespace, self._epoch, self._revision, self._state
                        )
                    )
                else:
                    state, commits, opened, reset, committed = _parse_inventory(saved)
                    _resolve_limits(self._requested, state[3].limits, "storage")
                    self._namespace, self._epoch, self._revision, self._state = state
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
                            f"ReflectiveMemory cleanup failed: {type(cleanup).__name__}"
                        )
                raise

    def _check_pid(self) -> None:
        """Reject inherited durable handles before touching any Python lock."""
        if self._storage is not None and os.getpid() != self._pid:
            raise _operation_error(
                "storage", "storage.forked", "Memory storage cannot be used after fork."
            )

    @contextmanager
    def _operation(self, *, unlocked: bool = False) -> Iterator[None]:
        """Keep operations active independently of run handles and callback locks."""
        self._check_pid()
        with self._lock:
            if self._closed:
                raise _operation_error(
                    "session", "reflective.closed", "memory backend is closed"
                )
            if self._storage is not None:
                self._storage.ensure_open()
            self._active_operations += 1
            if not unlocked:
                try:
                    yield
                finally:
                    self._active_operations -= 1
                return
        primary: BaseException | None = None
        try:
            yield
        except BaseException as error:
            primary = error
            raise
        finally:
            try:
                self._check_pid()
            except MemoryOperationError:
                # A forked callback cannot clean inherited state or lose its fault.
                if primary is None:
                    raise
            else:
                with self._lock:
                    self._active_operations -= 1

    def _save_state(
        self,
        state: JsonObject,
        *,
        applied_commits: dict[str, str] | None = None,
        ever_opened: bool | None = None,
        ever_reset: bool | None = None,
        ever_committed: bool | None = None,
    ) -> None:
        """Persist a validated candidate before changing the live backend."""
        if self._storage is None:
            return
        inventory = {
            "version": 1,
            "state": state,
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

    def __enter__(self) -> ReflectiveMemory:
        """Enter this still-open backend.

        Returns:
            `ReflectiveMemory`: This backend.

        Raises:
            MemoryOperationError: The backend or storage handle is unusable.
        """
        with self._operation():
            return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Release ownership while preserving an initiating exception.

        Args:
            exc_type (`type[BaseException] | None`): Original exception type.
            exc (`BaseException | None`): Original exception, if any.
            traceback (`TracebackType | None`): Original exception traceback.

        Raises:
            MemoryOperationError: Close fails without an initiating exception.
        """
        try:
            self.close()
        except BaseException as cleanup:
            if exc is None:
                raise
            if hasattr(exc, "add_note"):
                exc.add_note(
                    f"ReflectiveMemory cleanup failed: {type(cleanup).__name__}"
                )

    def close(self) -> None:
        """Release storage and invalidate idle handles without waiting for work.

        Raises:
            MemoryOperationError: Active operations make close busy, the handle
                was inherited after fork, or storage cleanup fails.
        """
        self._check_pid()
        if not self._lock.acquire(blocking=False):
            raise _operation_error(
                "close", "reflective.busy", "memory backend is active", retryable=True
            )
        try:
            if self._closed:
                return
            if self._active_operations:
                raise _operation_error(
                    "close",
                    "reflective.busy",
                    "memory backend is active",
                    retryable=True,
                )
            if self._storage is not None:
                self._storage.close()
            self._closed = True
        finally:
            self._lock.release()

    def rewrite_query(self, context: RecallContext) -> str:
        """Build the default lexical query from the visible prompt and inputs."""
        with self._operation():
            if type(context) is not RecallContext:
                raise TypeError("context must be a RecallContext")
            return f"Task: {context.prompt}\nInputs: {_canonical_json(context.inputs)}"

    def _effective_llm_config(self) -> JsonObject:
        """Resolve ambient defaults exactly once for one fresh session."""
        config = _copy_json_object(self._llm_config, "llm_config")
        try:
            resolved = resolve_model_config(config)
            validated, connection = split_model_config(resolved)
            complete = cast(JsonObject, validated.model_dump(mode="json", exclude_none=False))
            if connection is not None:
                complete["_connection"] = connection
            _nonempty_string(complete.get("provider"), "llm_config.provider")
            model = _nonempty_string(complete.get("model"), "llm_config.model")
            if not model.strip():
                raise ValueError("llm_config.model must not be blank")
        except Exception as exc:
            raise _operation_error(
                "open_run",
                "reflective.config",
                "reflective model configuration is invalid",
                retryable=False,
            ) from exc
        return complete

    def open_run(self, *, snapshot: JsonObject | None = None) -> _ReflectiveSession:
        """Open one immutable-read session, optionally from a recorded snapshot.

        Args:
            snapshot (`JsonObject | None`, defaults to `None`): Recorded public
                snapshot to replay, including its pinned model settings. None
                takes current lessons and resolves current host model settings.

        Returns:
            `_ReflectiveSession`: An isolated run session.

        Raises:
            MemoryOperationError: The snapshot or model configuration is invalid,
                the backend is closed, or durable state cannot be saved.
        """
        with self._operation():
            parsed = None if snapshot is None else _parse_snapshot(snapshot)
            adopt = False
            if parsed is None:
                namespace = self._namespace
                epoch = self._epoch
                revision = self._revision
                state = _copy_state(self._state)
                llm_config = self._effective_llm_config()
            else:
                namespace, epoch, revision, state, llm_config = parsed
                _resolve_limits(self._requested, state.limits, "open_run")
                if namespace != self._namespace:
                    pristine = (
                        not self._ever_opened
                        and not self._ever_reset
                        and not self._ever_committed
                        and not self._state.lessons
                        and self._epoch == 0
                        and self._revision == 0
                    )
                    if not pristine:
                        raise _snapshot_error(
                            "snapshot namespace cannot replace a non-pristine memory"
                        )
                    adopt = True
                else:
                    if epoch != self._epoch:
                        raise _snapshot_error(
                            "snapshot generation is incompatible after reset"
                        )
                    if revision > self._revision:
                        raise _snapshot_error(
                            "snapshot revision is ahead of the memory state"
                        )
                    if revision == self._revision and state != self._state:
                        raise _snapshot_error(
                            "snapshot state does not match the current generation"
                        )
            if adopt or not self._ever_opened:
                adopted_state = _copy_state(state) if adopt else self._state
                committed = (
                    revision > 0 or bool(state.lessons)
                    if adopt
                    else self._ever_committed
                )
                candidate = (
                    _state_json(namespace, epoch, revision, adopted_state)
                    if adopt
                    else _state_json(
                        self._namespace, self._epoch, self._revision, self._state
                    )
                )
                self._save_state(candidate, ever_opened=True, ever_committed=committed)
                if adopt:
                    self._namespace = namespace
                    self._epoch = epoch
                    self._revision = revision
                    self._state = adopted_state
                    self._ever_committed = committed
                self._ever_opened = True
            return _ReflectiveSession(
                self,
                namespace,
                epoch,
                revision,
                state,
                llm_config,
            )

    def reset(self) -> None:
        """Clear lessons, start a new generation, and retain commit idempotency.

        Raises:
            MemoryOperationError: The backend is closed or durable state cannot
                be saved. A failed save leaves live lessons unchanged.
        """
        with self._operation():
            cleared = _State(self._state.limits, ())
            self._save_state(
                _state_json(self._namespace, self._epoch + 1, 0, cleared),
                ever_reset=True,
            )
            self._state = cleared
            self._epoch += 1
            self._revision = 0
            self._ever_reset = True


class _ReflectiveSession:
    """One run-scoped view over a pinned reflective-memory generation."""

    def __init__(
        self,
        memory: ReflectiveMemory,
        namespace: str,
        epoch: int,
        revision: int,
        state: _State,
        llm_config: JsonObject,
    ) -> None:
        self._memory = memory
        self._namespace = namespace
        self._epoch = epoch
        self._revision = revision
        self._state = state
        self._llm_config = _copy_json_object(llm_config, "llm_config")
        self._opened: dict[MemoryFrameRef, _ReflectiveFrame] = {}
        self._closed = False

    def _ensure_open(self) -> None:
        """Reject every operation once the run boundary has closed."""
        if self._closed:
            raise _operation_error(
                "session", "reflective.session_closed", "memory session is closed"
            )

    @property
    def snapshot(self) -> JsonObject:
        """Return the public durable source-neutral state pinned for this run."""
        with self._memory._operation():
            self._ensure_open()
            return _snapshot_json(
                self._namespace,
                self._epoch,
                self._revision,
                self._state,
                self._llm_config,
            )

    def __enter__(self) -> _ReflectiveSession:
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
        """Close this run session while preserving an initiating exception."""
        try:
            self.close()
        except BaseException as cleanup:
            if exc is None:
                raise
            if hasattr(exc, "add_note"):
                exc.add_note(
                    f"ReflectiveMemory session cleanup failed: {type(cleanup).__name__}"
                )

    def close(self) -> None:
        """Idempotently close this process-local session."""
        self._memory._check_pid()
        with self._memory._lock:
            self._closed = True

    def open_frame(self, spec: MemoryFrameSpec) -> _ReflectiveFrame:
        """Return the view of this instance's state for one frame reference.

        The frame's policy name and configuration are ignored. The instance
        limits come from the backend.
        """
        with self._memory._operation():
            self._ensure_open()
            if type(spec) is not MemoryFrameSpec:
                raise _operation_error(
                    "open_frame",
                    "reflective.invalid_frame",
                    "spec must be a MemoryFrameSpec",
                )
            opened = self._opened.get(spec.ref)
            if opened is None:
                opened = _ReflectiveFrame(self, spec.ref)
                self._opened[spec.ref] = opened
            return opened

    def _assert_current_generation(self, stage: str) -> None:
        """Require this proposal session to remain the current generation."""
        if (
            self._namespace != self._memory._namespace
            or self._epoch != self._memory._epoch
            or self._revision != self._memory._revision
        ):
            raise _operation_error(
                stage,
                "reflective.stale_generation",
                "memory session is stale",
                retryable=False,
            )

    def _batch_digest(self, mutations: tuple[MemoryMutation, ...]) -> str:
        """Compute a canonical digest before generation-dependent validation."""
        if type(mutations) is not tuple:
            raise _operation_error(
                "commit",
                "reflective.invalid_batch",
                "mutations must be an exact tuple",
            )
        encoded: list[JsonObject] = []
        for mutation in mutations:
            if type(mutation) is not MemoryMutation:
                raise _operation_error(
                    "commit",
                    "reflective.invalid_batch",
                    "mutation batch contains an invalid value",
                )
            encoded.append(
                {
                    "frame_ref": _frame_ref_json(mutation.frame_ref),
                    "mutation_id": mutation.mutation_id,
                    "payload": mutation.payload,
                }
            )
        try:
            material = _canonical_json(cast(JsonValue, encoded))
        except (TypeError, ValueError) as exc:
            raise _operation_error(
                "commit",
                "reflective.invalid_batch",
                "mutation batch cannot be encoded canonically",
            ) from exc
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    def _validate_batch_ownership(self, mutations: tuple[MemoryMutation, ...]) -> None:
        """Validate session ownership and per-batch mutation identity."""
        mutation_ids: set[str] = set()
        for mutation in mutations:
            if mutation.frame_ref not in self._opened:
                raise _operation_error(
                    "commit",
                    "reflective.invalid_batch",
                    "mutation belongs to a frame not opened by this session",
                )
            if mutation.mutation_id in mutation_ids:
                raise _operation_error(
                    "commit",
                    "reflective.invalid_batch",
                    "mutation ids must be unique within a batch",
                )
            mutation_ids.add(mutation.mutation_id)

    def _validated_action(
        self,
        mutation: MemoryMutation,
    ) -> tuple[str, str, str | None, JsonObject]:
        """Validate one persisted reflective mutation payload exactly."""
        payload = mutation.payload
        if type(payload) is not dict:
            raise _operation_error(
                "commit",
                "reflective.invalid_mutation",
                "mutation payload must be an object",
            )
        allowed = {"action", "lesson_id", "source", "text"}
        required = {"action", "lesson_id", "source"}
        if set(payload) - allowed or not required.issubset(payload):
            raise _operation_error(
                "commit",
                "reflective.invalid_mutation",
                "mutation payload has invalid fields",
            )
        action = payload["action"]
        if action not in ("add", "revise", "remove"):
            raise _operation_error(
                "commit",
                "reflective.invalid_mutation",
                "mutation action is invalid",
            )
        try:
            lesson_id = _nonempty_string(payload["lesson_id"], "lesson_id")
            source = _source_from_json(payload["source"], "source")
        except (TypeError, ValueError) as exc:
            raise _operation_error(
                "commit",
                "reflective.invalid_mutation",
                "mutation identity or source is invalid",
            ) from exc
        text: str | None = None
        if action in ("add", "revise"):
            try:
                text = _nonempty_string(payload.get("text"), "text")
            except (TypeError, ValueError) as exc:
                raise _operation_error(
                    "commit",
                    "reflective.invalid_mutation",
                    "mutation text is invalid",
                ) from exc
            if not text.strip():
                raise _operation_error(
                    "commit",
                    "reflective.invalid_mutation",
                    "text must not be blank",
                )
        elif "text" in payload:
            raise _operation_error(
                "commit",
                "reflective.invalid_mutation",
                "remove mutations cannot carry text",
            )
        return cast(str, action), lesson_id, text, source

    def commit(
        self,
        commit_id: str,
        mutations: tuple[MemoryMutation, ...],
    ) -> None:
        """Atomically apply one exact batch, with ledger lookup before generation."""
        with self._memory._operation():
            self._ensure_open()
            try:
                commit_id = _nonempty_string(commit_id, "commit_id")
            except (TypeError, ValueError) as exc:
                raise _operation_error(
                    "commit",
                    "reflective.invalid_commit_id",
                    "commit id must be a nonempty string",
                ) from exc
            digest = self._batch_digest(mutations)

            applied_digest = self._memory._applied_commits.get(commit_id)
            if applied_digest is not None:
                if applied_digest == digest:
                    return
                raise _operation_error(
                    "commit",
                    "reflective.commit_collision",
                    "commit id collision with a different batch",
                )

            self._validate_batch_ownership(mutations)
            self._assert_current_generation("commit")
            state = _copy_state(self._memory._state)
            lessons = list(state.lessons)
            touched: set[str] = set()
            for mutation in mutations:
                action, lesson_id, text, source = self._validated_action(mutation)
                if lesson_id in touched:
                    raise _operation_error(
                        "commit",
                        "reflective.invalid_mutation",
                        "a batch cannot target one lesson more than once",
                    )
                touched.add(lesson_id)
                position = next(
                    (
                        index
                        for index, lesson in enumerate(lessons)
                        if lesson.lesson_id == lesson_id
                    ),
                    None,
                )
                if action == "add":
                    if position is not None:
                        raise _operation_error(
                            "commit",
                            "reflective.invalid_mutation",
                            "add mutation targets an existing lesson",
                        )
                    lessons.append(_Lesson(lesson_id, cast(str, text), source))
                elif action == "revise":
                    if position is None:
                        raise _operation_error(
                            "commit",
                            "reflective.invalid_mutation",
                            "revise mutation targets a missing lesson",
                        )
                    lessons[position] = _Lesson(lesson_id, cast(str, text), source)
                else:
                    if position is None:
                        raise _operation_error(
                            "commit",
                            "reflective.invalid_mutation",
                            "remove mutation targets a missing lesson",
                        )
                    lessons.pop(position)

            state = _State(state.limits, tuple(lessons))
            if len(state.lessons) > state.limits.max_items:
                raise _operation_error(
                    "commit",
                    "reflective.max_items",
                    "mutation batch exceeds max_items",
                )

            commits = {**self._memory._applied_commits, commit_id: digest}
            self._memory._save_state(
                _state_json(
                    self._memory._namespace,
                    self._memory._epoch,
                    self._memory._revision + 1,
                    state,
                ),
                applied_commits=commits,
                ever_committed=True,
            )
            self._memory._state = state
            self._memory._applied_commits = commits
            self._memory._revision += 1
            self._memory._ever_committed = True


class _ReflectiveFrame:
    """One session-bound frame for deterministic recall and proposal generation."""

    def __init__(
        self,
        session: _ReflectiveSession,
        ref: MemoryFrameRef,
    ) -> None:
        self._session = session
        self._ref = ref

    def _lessons(self) -> tuple[_Lesson, ...]:
        """Return the instance's pinned immutable lessons."""
        return self._session._state.lessons

    @staticmethod
    def _format(lessons: list[_Lesson] | tuple[_Lesson, ...]) -> str:
        """Render atomic ids and text without an outer section header."""
        return "\n".join(f"[{lesson.lesson_id}] {lesson.text}" for lesson in lessons)

    def recall(self, context: RecallContext | None = None) -> str:
        """Return all pinned lessons or a deterministic lexical ranking."""
        with self._session._memory._operation(unlocked=True):
            return self._recall(context)

    def _recall(self, context: RecallContext | None) -> str:
        """Rank pinned lessons while query callbacks run outside the backend lock."""
        with self._session._memory._operation():
            self._session._ensure_open()
            lessons = self._lessons()
        if context is None:
            return self._format(lessons)
        if type(context) is not RecallContext:
            raise _operation_error(
                "recall",
                "reflective.invalid_context",
                "context must be a RecallContext or None",
            )
        try:
            query = self._session._memory.rewrite_query(context)
        except Exception as exc:
            raise _operation_error(
                "recall",
                "reflective.query",
                "memory query rewriting failed",
            ) from exc
        if type(query) is not str:
            raise _operation_error(
                "recall",
                "reflective.query",
                "rewrite_query() must return a string",
            )
        query_words = set(_WORD.findall(query.casefold()))
        ranked = sorted(
            enumerate(lessons),
            key=lambda item: (
                -len(query_words & set(_WORD.findall(item[1].text.casefold()))),
                item[0],
                item[1].lesson_id,
            ),
        )
        limit = self._session._state.limits.recall_limit
        selected = [lesson for _, lesson in ranked[:limit]]
        return self._format(selected)

    def _reflection_prompt(self, context: LearnContext) -> str:
        """Build one bounded deterministic human message for reflection."""
        lessons = [
            {
                "id": lesson.lesson_id,
                "source": lesson.source,
                "text": lesson.text,
            }
            for lesson in self._lessons()
        ]
        frame_material = _bounded_json_material(
            _frame_ref_json(self._ref), _MAX_FRAME_MATERIAL_CHARS
        )
        lesson_material = _bounded_json_material(
            cast(JsonValue, lessons), _MAX_LESSON_MATERIAL_CHARS
        )
        episode_material = _bounded_episode_material(context)
        limit_material = _bounded_json_material(
            {"max_items": self._session._state.limits.max_items},
            _MAX_LIMIT_MATERIAL_CHARS,
        )
        return (
            f"Frame: {frame_material}\n"
            f"Limits: {limit_material}\n"
            f"Current lessons: {lesson_material}\n"
            f"Episodes: {episode_material}"
        )

    def _validate_actions(
        self,
        context: LearnContext,
        response: _ReflectionResponse,
    ) -> list[MemoryMutation]:
        """Validate closed actions against pinned state and build proposals."""
        actions = response.actions
        noops = [action for action in actions if action.action == "noop"]
        if noops and len(actions) != 1:
            raise _operation_error(
                "learn",
                "reflective.contract",
                "a no-op action must be the only action",
            )
        if noops:
            noop = noops[0]
            if noop.lesson_id is not None or noop.text is not None:
                raise _operation_error(
                    "learn",
                    "reflective.contract",
                    "a no-op action cannot carry an id or text",
                )
            return []

        current = {lesson.lesson_id: lesson for lesson in self._lessons()}
        touched: set[str] = set()
        proposals: list[MemoryMutation] = []
        source = _source_json(context.run_id)
        for index, action in enumerate(actions):
            lesson_id = action.lesson_id
            if type(lesson_id) is not str or not lesson_id:
                raise _operation_error(
                    "learn",
                    "reflective.contract",
                    "lesson ids must be non-empty strings",
                )
            if lesson_id in touched:
                raise _operation_error(
                    "learn",
                    "reflective.contract",
                    "actions cannot target one lesson more than once",
                )
            touched.add(lesson_id)
            if action.action in ("add", "revise"):
                if type(action.text) is not str or not action.text.strip():
                    raise _operation_error(
                        "learn",
                        "reflective.contract",
                        "add and revise actions require non-blank text",
                    )
            elif action.text is not None:
                raise _operation_error(
                    "learn",
                    "reflective.contract",
                    "remove actions cannot carry text",
                )

            if action.action == "add":
                if lesson_id in current:
                    raise _operation_error(
                        "learn",
                        "reflective.contract",
                        "reflector add action targets an existing lesson",
                    )
                current[lesson_id] = _Lesson(lesson_id, cast(str, action.text), source)
            elif action.action == "revise":
                if lesson_id not in current:
                    raise _operation_error(
                        "learn",
                        "reflective.contract",
                        "reflector revise action targets a missing lesson",
                    )
                current[lesson_id] = _Lesson(lesson_id, cast(str, action.text), source)
            else:
                if lesson_id not in current:
                    raise _operation_error(
                        "learn",
                        "reflective.contract",
                        "reflector remove action targets a missing lesson",
                    )
                del current[lesson_id]

            payload: JsonObject = {
                "action": action.action,
                "lesson_id": lesson_id,
                "source": _copy_json_object(source, "source"),
            }
            if action.action in ("add", "revise"):
                payload["text"] = cast(str, action.text)
            identity = _mutation_digest_payload(context, index, payload)
            mutation_id = (
                "reflective:"
                + hashlib.sha256(_canonical_json(identity).encode("utf-8")).hexdigest()
            )
            proposals.append(MemoryMutation(mutation_id, context.frame_ref, payload))

        if len(current) > self._session._state.limits.max_items:
            raise _operation_error(
                "learn",
                "reflective.max_items",
                "proposed lessons exceed max_items",
            )
        return proposals

    def learn(self, context: LearnContext) -> list[MemoryMutation]:
        """Propose, but do not apply, one exact list of lesson mutations."""
        with self._session._memory._operation(unlocked=True):
            return self._learn(context)

    def _learn(self, context: LearnContext) -> list[MemoryMutation]:
        """Reflect outside the lock while retaining active backend ownership."""
        if type(context) is not LearnContext:
            raise _operation_error(
                "learn",
                "reflective.invalid_context",
                "context must be a LearnContext",
            )
        if context.frame_ref != self._ref:
            raise _operation_error(
                "learn",
                "reflective.invalid_context",
                "learning context belongs to a different frame",
            )
        with self._session._memory._operation():
            self._session._ensure_open()
            self._session._assert_current_generation("learn")
            llm_config = _copy_json_object(self._session._llm_config, "llm_config")
            prompt = self._reflection_prompt(context)

        system = SystemMessage(
            content=(
                "You maintain atomic workflow lessons. Return only the requested "
                "structured actions. Use add for a new id, revise/remove only for "
                "an existing id, or one noop when nothing should change. Never use tools."
            )
        )
        human = HumanMessage(content=prompt)
        try:
            model = model_from_config(legacy_connection_config(llm_config))
        except Exception as exc:
            raise _operation_error(
                "learn",
                "reflective.model_construction",
                "reflector model construction failed",
                retryable=False,
            ) from exc
        self._session._memory._check_pid()
        try:
            structured = model.with_structured_output(_ReflectionResponse)
        except Exception as exc:
            raise _operation_error(
                "learn",
                "reflective.structured_output",
                "reflector does not support the required structured output",
                retryable=False,
            ) from exc
        self._session._memory._check_pid()
        try:
            invoke = getattr(structured, "invoke", None)
            if not callable(invoke):
                raise TypeError("structured model must provide invoke()")
        except Exception as exc:
            raise _operation_error(
                "learn",
                "reflective.structured_output",
                "reflector does not support the required structured output",
                retryable=False,
            ) from exc
        self._session._memory._check_pid()
        try:
            raw_response = invoke([system, human])
        except (OutputParserException, ValidationError) as exc:
            raise _operation_error(
                "learn",
                "reflective.contract",
                "reflector returned invalid structured output",
                retryable=False,
            ) from exc
        except Exception as exc:
            raise _operation_error(
                "learn",
                "reflective.provider",
                "reflector invocation failed",
                retryable=True,
            ) from exc
        try:
            response = _ReflectionResponse.model_validate(raw_response, strict=True)
        except ValidationError as exc:
            raise _operation_error(
                "learn",
                "reflective.contract",
                "reflector returned invalid structured output",
                retryable=False,
            ) from exc
        proposals = self._validate_actions(context, response)
        with self._session._memory._operation():
            self._session._ensure_open()
            self._session._assert_current_generation("learn")
        return proposals


__all__ = ["ReflectiveMemory"]
