"""Process-local episodic memory with snapshot-isolated run sessions.

The backend owns retained episode history behind the source-neutral memory
protocols.  This module establishes the session, frame, snapshot, and commit
boundaries without depending on the workflow engine or storage services.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from threading import RLock
from types import TracebackType
from typing import cast
from uuid import uuid4

from agent_composer.memory.contracts import (
    JsonObject,
    JsonValue,
    LearnContext,
    MemoryEpisode,
    MemoryEpisodeStatus,
    MemoryFrameRef,
    MemoryFrameSpec,
    MemoryMutation,
    MemoryOperationError,
    MemoryOwner,
    MemoryOccurrenceKind,
    MemoryOccurrenceRef,
    MemoryOccurrenceSegment,
    MemorySection,
    ModelTextInteraction,
    ModelValueInteraction,
    RecallContext,
    ToolCallInteraction,
    ToolResultInteraction,
    VisibleInteraction,
    _MAX_JSON_NESTING,
)

_SNAPSHOT_FORMAT = "agent_composer.episodic_memory"
_SNAPSHOT_VERSION = 1
_POLICY_NAME = "episodic"


@dataclass(frozen=True, slots=True)
class _EffectiveFrameSpec:
    """Validated concrete configuration of one episodic frame."""

    ref: MemoryFrameRef
    max_episodes: int
    recall_limit: int


@dataclass(frozen=True, slots=True)
class _FrameState:
    """One frame specification and its ordered episode records."""

    spec: _EffectiveFrameSpec
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
) -> JsonValue:
    """Validate and defensively copy one exact finite JSON value."""
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
    if nesting >= _MAX_JSON_NESTING:
        raise ValueError(f"{name} JSON nesting is too deep")
    active = set() if active is None else active
    identity = id(value)
    if identity in active:
        raise ValueError(f"{name} contains a circular JSON value")
    active.add(identity)
    try:
        if value_type is list:
            return [
                _copy_json(item, f"{name}[{index}]", active=active, nesting=nesting + 1)
                for index, item in enumerate(value)
            ]
        copied: JsonObject = {}
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError(f"{name} JSON object keys must be strings")
            _json_string(key, f"{name} JSON object key")
            copied[key] = _copy_json(
                item, f"{name}.{key}", active=active, nesting=nesting + 1
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


def _frame_ref_json(ref: MemoryFrameRef) -> JsonObject:
    """Encode one stable frame identity as JSON."""
    return {
        "owner": ref.owner.value,
        "root_flow_id": ref.root_flow_id,
        "static_path": list(ref.static_path),
    }


def _frame_ref_from_json(value: object, name: str) -> MemoryFrameRef:
    """Decode one exact frame identity."""
    data = _copy_object(value, name)
    if set(data) != {"owner", "root_flow_id", "static_path"}:
        raise ValueError(f"{name} has invalid fields")
    root = _json_string(data["root_flow_id"], f"{name}.root_flow_id", nonempty=True)
    owner_raw = _json_string(data["owner"], f"{name}.owner", nonempty=True)
    try:
        owner = MemoryOwner(owner_raw)
    except ValueError as exc:
        raise ValueError(f"{name}.owner is invalid") from exc
    path_raw = data["static_path"]
    if type(path_raw) is not list or not path_raw:
        raise ValueError(f"{name}.static_path must be a non-empty array")
    path = tuple(
        _json_string(part, f"{name}.static_path[{index}]", nonempty=True)
        for index, part in enumerate(path_raw)
    )
    return MemoryFrameRef(root, path, owner)


def _frame_sort_key(ref: MemoryFrameRef) -> tuple[str, tuple[str, ...], str]:
    """Return the canonical static-frame order."""
    return (ref.root_flow_id, ref.static_path, ref.owner.value)


def _occurrence_json(occurrence: MemoryOccurrenceRef) -> JsonObject:
    """Encode one dynamic occurrence without losing its authored frame."""
    return {
        "frame_ref": _frame_ref_json(occurrence.frame_ref),
        "occurrence_path": [
            {
                "index": segment.index,
                "kind": segment.kind.value,
                "static_path": list(segment.static_path),
            }
            for segment in occurrence.occurrence_path
        ],
    }


def _occurrence_from_json(value: object, name: str) -> MemoryOccurrenceRef:
    """Decode one exact dynamic occurrence."""
    data = _copy_object(value, name)
    if set(data) != {"frame_ref", "occurrence_path"}:
        raise ValueError(f"{name} has invalid fields")
    ref = _frame_ref_from_json(data["frame_ref"], f"{name}.frame_ref")
    path_raw = data["occurrence_path"]
    if type(path_raw) is not list:
        raise TypeError(f"{name}.occurrence_path must be an array")
    path: list[MemoryOccurrenceSegment] = []
    for index, raw in enumerate(path_raw):
        item_name = f"{name}.occurrence_path[{index}]"
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
    return MemoryOccurrenceRef(ref, tuple(path))


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
            {"frame_ref": _frame_ref_json(section.frame_ref), "text": section.text}
            for section in episode.memory_sections
        ],
        "occurrence": _occurrence_json(episode.occurrence),
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
        "occurrence",
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
        if set(section) != {"frame_ref", "text"}:
            raise ValueError(f"{section_name} has invalid fields")
        sections.append(
            MemorySection(
                _frame_ref_from_json(section["frame_ref"], f"{section_name}.frame_ref"),
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
        _occurrence_from_json(data["occurrence"], f"{name}.occurrence"),
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


def _validate_episode_owner(
    episode: MemoryEpisode, expected_ref: MemoryFrameRef
) -> None:
    """Require one episode to belong to its containing memory frame."""
    if episode.occurrence.frame_ref != expected_ref:
        raise ValueError("episode occurrence belongs to a different frame")
    frame_path = expected_ref.static_path
    owner_path = episode.owner_static_path
    if expected_ref.owner is MemoryOwner.AGENT:
        valid_owner = owner_path == frame_path
    else:
        valid_owner = owner_path[: len(frame_path)] == frame_path
    if not valid_owner:
        raise ValueError("episode lies outside its owning frame")


def _validate_entry(
    value: object, name: str, expected_ref: MemoryFrameRef
) -> JsonObject:
    """Validate one retained entry and return its canonical defensive copy."""
    data = _copy_object(value, name)
    if set(data) != {"entry_id", "episode"}:
        raise ValueError(f"{name} has invalid fields")
    entry_id = _json_string(data["entry_id"], f"{name}.entry_id", nonempty=True)
    episode = _episode_from_json(data["episode"], f"{name}.episode")
    _validate_episode_owner(episode, expected_ref)
    canonical = _stored_entry(entry_id, episode)
    if _canonical_json(data) != _canonical_json(canonical):
        raise ValueError(f"{name} must be canonical")
    return canonical


def _copy_state(state: _FrameState) -> _FrameState:
    """Defensively copy one immutable frame state."""
    return _FrameState(
        state.spec,
        tuple(_copy_object(episode, "episode") for episode in state.episodes),
    )


def _snapshot_json(
    namespace: str,
    epoch: int,
    revision: int,
    frames: dict[MemoryFrameRef, _FrameState],
) -> JsonObject:
    """Encode one canonical episodic-memory snapshot."""
    encoded: list[JsonValue] = []
    for ref in sorted(frames, key=_frame_sort_key):
        state = frames[ref]
        encoded.append(
            {
                "config": {
                    "max_episodes": state.spec.max_episodes,
                    "recall_limit": state.spec.recall_limit,
                },
                "episodes": [
                    _copy_object(episode, "episode") for episode in state.episodes
                ],
                "policy_name": _POLICY_NAME,
                "ref": _frame_ref_json(ref),
            }
        )
    return {
        "epoch": epoch,
        "format": _SNAPSHOT_FORMAT,
        "frames": encoded,
        "namespace": namespace,
        "revision": revision,
        "version": _SNAPSHOT_VERSION,
    }


def _snapshot_error() -> MemoryOperationError:
    """Return the stable snapshot-validation failure."""
    return _operation_error(
        "open_run", "episodic.invalid_snapshot", "recorded memory snapshot is invalid"
    )


def _parse_snapshot(
    value: object,
) -> tuple[str, int, int, dict[MemoryFrameRef, _FrameState]]:
    """Strictly decode one canonical episodic-memory snapshot."""
    try:
        data = _copy_object(value, "snapshot")
        if set(data) != {
            "epoch",
            "format",
            "frames",
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
        frames_raw = data["frames"]
        if type(frames_raw) is not list:
            raise TypeError("snapshot.frames must be an array")
        frames: dict[MemoryFrameRef, _FrameState] = {}
        prior: tuple[str, tuple[str, ...], str] | None = None
        for index, raw in enumerate(frames_raw):
            frame = _copy_object(raw, f"snapshot.frames[{index}]")
            if set(frame) != {"config", "episodes", "policy_name", "ref"}:
                raise ValueError("snapshot frame has invalid fields")
            if frame["policy_name"] != _POLICY_NAME:
                raise ValueError("snapshot frame has invalid policy")
            ref = _frame_ref_from_json(frame["ref"], f"snapshot.frames[{index}].ref")
            key = _frame_sort_key(ref)
            if prior is not None and key <= prior:
                raise ValueError("snapshot.frames must be uniquely sorted")
            prior = key
            config = _copy_object(frame["config"], f"snapshot.frames[{index}].config")
            if set(config) != {"max_episodes", "recall_limit"}:
                raise ValueError("snapshot frame config has invalid fields")
            max_episodes = _positive_int(config["max_episodes"], "max_episodes")
            recall_limit = _positive_int(config["recall_limit"], "recall_limit")
            episodes = frame["episodes"]
            if type(episodes) is not list:
                raise ValueError("snapshot frame episodes must be an array")
            if len(episodes) > max_episodes:
                raise ValueError("snapshot frame exceeds its retention limit")
            decoded = tuple(
                _validate_entry(
                    item,
                    f"snapshot.frames[{index}].episodes[{episode_index}]",
                    ref,
                )
                for episode_index, item in enumerate(episodes)
            )
            entry_ids = [cast(str, item["entry_id"]) for item in decoded]
            if len(entry_ids) != len(set(entry_ids)):
                raise ValueError("snapshot frame episode IDs must be unique")
            spec = _EffectiveFrameSpec(ref, max_episodes, recall_limit)
            frames[ref] = _FrameState(spec, decoded)
        canonical = _snapshot_json(namespace, epoch, revision, frames)
        if _canonical_json(data) != _canonical_json(canonical):
            raise ValueError("snapshot must be canonical")
        return namespace, epoch, revision, frames
    except MemoryOperationError:
        raise
    except (TypeError, ValueError, RecursionError) as exc:
        raise _snapshot_error() from exc


def _compatible(
    current: dict[MemoryFrameRef, _FrameState],
    recorded: dict[MemoryFrameRef, _FrameState],
) -> bool:
    """Accept current state plus harmless recorded empty declarations."""
    for ref, state in current.items():
        if recorded.get(ref) != state:
            return False
    return all(ref in current or not state.episodes for ref, state in recorded.items())


class EpisodicMemory:
    """Process-local memory that retains bounded canonical episodes.

    Examples:
        Inject one instance under the name selected by an authored
        ``long_term_memory: episodic`` declaration. Forward execution only
        retains history; the explicit backward call commits it for later runs.

        ```python
        from agent_composer import run_flow
        from agent_composer.memory.episodic import EpisodicMemory

        memory = EpisodicMemory(max_episodes=100, recall_limit=10)
        result = run_flow(
            flow,
            inputs,
            memory={"episodic": memory},
            learnable=True,
        )
        result.backward()
        ```
    """

    def __init__(self, max_episodes: int = 100, recall_limit: int = 10) -> None:
        """Create one isolated episodic-memory namespace.

        Args:
            max_episodes (`int`, defaults to `100`):
                Maximum retained episodes for each frame.
            recall_limit (`int`, defaults to `10`):
                Maximum episodes returned by contextual recall.

        Raises:
            TypeError: Raised when either limit is not an exact integer.
            ValueError: Raised when either limit is not positive.
        """
        self._default_max_episodes = _positive_int(max_episodes, "max_episodes")
        self._default_recall_limit = _positive_int(recall_limit, "recall_limit")
        self._lock = RLock()
        self._namespace = uuid4().hex
        self._epoch = 0
        self._revision = 0
        self._frames: dict[MemoryFrameRef, _FrameState] = {}
        self._applied_commits: dict[str, str] = {}
        self._ever_opened = False
        self._ever_reset = False
        self._ever_committed = False

    def open_run(self, *, snapshot: JsonObject | None = None) -> _EpisodicSession:
        """Open a run session over one pinned memory snapshot.

        Args:
            snapshot (`JsonObject`, *optional*, defaults to `None`):
                Recorded episodic snapshot to restore or replay.

        Returns:
            `_EpisodicSession`: The isolated run-scoped session.

        Raises:
            MemoryOperationError: Raised when a snapshot is invalid or incompatible.
        """
        parsed = None if snapshot is None else _parse_snapshot(snapshot)
        with self._lock:
            if parsed is None:
                namespace = self._namespace
                epoch = self._epoch
                revision = self._revision
                frames = {
                    ref: _copy_state(state) for ref, state in self._frames.items()
                }
            else:
                namespace, epoch, revision, frames = parsed
                if namespace != self._namespace:
                    pristine = (
                        not self._ever_opened
                        and not self._ever_reset
                        and not self._ever_committed
                        and not self._frames
                        and self._epoch == 0
                        and self._revision == 0
                    )
                    if not pristine:
                        raise _snapshot_error()
                    self._namespace = namespace
                    self._epoch = epoch
                    self._revision = revision
                    self._frames = {
                        ref: _copy_state(state) for ref, state in frames.items()
                    }
                    self._ever_committed = revision > 0
                else:
                    if epoch != self._epoch or revision > self._revision:
                        raise _snapshot_error()
                    if revision == self._revision and not _compatible(
                        self._frames, frames
                    ):
                        raise _snapshot_error()
                frames = {ref: _copy_state(state) for ref, state in frames.items()}
            self._ever_opened = True
            return _EpisodicSession(self, namespace, epoch, revision, frames)

    def reset(self) -> None:
        """Clear episodes and begin a new memory generation."""
        with self._lock:
            self._frames = {}
            self._epoch += 1
            self._revision = 0
            self._ever_reset = True

    def rewrite_query(self, context: RecallContext) -> str:
        """Derive deterministic lexical retrieval text from visible context."""
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
        frames: dict[MemoryFrameRef, _FrameState],
    ) -> None:
        self._memory = memory
        self._namespace = namespace
        self._epoch = epoch
        self._revision = revision
        self._frames = frames
        self._opened: dict[MemoryFrameRef, _EpisodicFrame] = {}
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
        with self._memory._lock:
            self._ensure_open()
            frames = {ref: _copy_state(state) for ref, state in self._frames.items()}
            for ref, frame in self._opened.items():
                frames.setdefault(ref, _FrameState(frame._spec, ()))
            return _snapshot_json(self._namespace, self._epoch, self._revision, frames)

    def __enter__(self) -> _EpisodicSession:
        """Enter this still-open session."""
        with self._memory._lock:
            self._ensure_open()
            return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close this session without suppressing an exception."""
        self.close()

    def close(self) -> None:
        """Idempotently close this process-local session."""
        with self._memory._lock:
            self._closed = True

    def _normalize_spec(
        self, spec: MemoryFrameSpec, recorded: _FrameState | None
    ) -> _EffectiveFrameSpec:
        """Validate one frame and fill its local defaults."""
        if type(spec) is not MemoryFrameSpec:
            raise _operation_error(
                "open_frame", "episodic.invalid_frame", "spec must be a MemoryFrameSpec"
            )
        if spec.policy_name != _POLICY_NAME:
            raise _operation_error(
                "open_frame",
                "episodic.invalid_policy",
                "frame policy must be 'episodic'",
            )
        config = spec.config
        if set(config) - {"max_episodes", "recall_limit"}:
            raise _operation_error(
                "open_frame",
                "episodic.invalid_frame_config",
                "frame configuration contains unsupported fields",
            )
        try:
            max_default = (
                recorded.spec.max_episodes
                if recorded is not None
                else self._memory._default_max_episodes
            )
            recall_default = (
                recorded.spec.recall_limit
                if recorded is not None
                else self._memory._default_recall_limit
            )
            return _EffectiveFrameSpec(
                spec.ref,
                _positive_int(config.get("max_episodes", max_default), "max_episodes"),
                _positive_int(
                    config.get("recall_limit", recall_default), "recall_limit"
                ),
            )
        except (TypeError, ValueError) as exc:
            raise _operation_error(
                "open_frame",
                "episodic.invalid_frame_config",
                "frame configuration values are invalid",
            ) from exc

    def open_frame(self, spec: MemoryFrameSpec) -> _EpisodicFrame:
        """Open one frame idempotently against the pinned snapshot."""
        with self._memory._lock:
            self._ensure_open()
            if type(spec) is not MemoryFrameSpec:
                raise _operation_error(
                    "open_frame",
                    "episodic.invalid_frame",
                    "spec must be a MemoryFrameSpec",
                )
            recorded = self._frames.get(spec.ref)
            effective = self._normalize_spec(spec, recorded)
            opened = self._opened.get(effective.ref)
            if opened is not None:
                if opened._requested_spec != spec or opened._spec != effective:
                    raise _operation_error(
                        "open_frame",
                        "episodic.frame_conflict",
                        "the frame was already opened with a different specification",
                    )
                return opened
            if recorded is not None and recorded.spec != effective:
                raise _operation_error(
                    "open_frame",
                    "episodic.frame_conflict",
                    "the recorded frame has a different specification",
                )
            frame = _EpisodicFrame(self, effective, spec)
            self._opened[effective.ref] = frame
            return frame

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
                "episodic.invalid_batch",
                "mutation batch is not canonical JSON",
            ) from exc
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    def commit(self, commit_id: str, mutations: tuple[MemoryMutation, ...]) -> None:
        """Atomically append one exact idempotent mutation batch."""
        with self._memory._lock:
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
            frames = {
                ref: _copy_state(state) for ref, state in self._memory._frames.items()
            }
            for ref, opened in self._opened.items():
                current = frames.get(ref)
                if current is not None and current.spec != opened._spec:
                    raise _operation_error(
                        "commit",
                        "episodic.frame_conflict",
                        "current frame specification changed",
                    )
                frames.setdefault(ref, _FrameState(opened._spec, ()))
            seen_mutations: set[str] = set()
            additions: dict[MemoryFrameRef, list[JsonObject]] = {}
            try:
                for index, mutation in enumerate(mutations):
                    if mutation.mutation_id in seen_mutations:
                        raise ValueError("mutation IDs must be unique")
                    seen_mutations.add(mutation.mutation_id)
                    if mutation.frame_ref not in self._opened:
                        raise ValueError(
                            "mutation frame was not opened by this session"
                        )
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
                        payload["entry"],
                        f"mutations[{index}].payload.entry",
                        mutation.frame_ref,
                    )
                    if entry["entry_id"] != mutation.mutation_id:
                        raise ValueError("mutation and entry IDs must match")
                    additions.setdefault(mutation.frame_ref, []).append(entry)
                for ref, entries in additions.items():
                    state = frames[ref]
                    existing_ids = {
                        cast(str, entry["entry_id"]) for entry in state.episodes
                    }
                    incoming_ids = {cast(str, entry["entry_id"]) for entry in entries}
                    if existing_ids & incoming_ids:
                        raise ValueError("episode entry is already retained")
                    retained = state.episodes + tuple(entries)
                    retained = retained[-state.spec.max_episodes :]
                    frames[ref] = _FrameState(state.spec, retained)
            except (TypeError, ValueError, RecursionError) as exc:
                raise _operation_error(
                    "commit", "episodic.invalid_batch", "mutation batch is invalid"
                ) from exc
            self._memory._frames = frames
            self._memory._applied_commits[commit_id] = digest
            self._memory._revision += 1
            self._memory._ever_committed = True


class _EpisodicFrame:
    """One session-bound episodic memory frame."""

    def __init__(
        self,
        session: _EpisodicSession,
        spec: _EffectiveFrameSpec,
        requested_spec: MemoryFrameSpec,
    ) -> None:
        self._session = session
        self._spec = spec
        self._requested_spec = requested_spec

    def recall(self, context: RecallContext | None = None) -> str:
        """Render pinned retained history, optionally ranked by lexical relevance."""
        with self._session._memory._lock:
            self._session._ensure_open()
            if context is not None and type(context) is not RecallContext:
                raise _operation_error(
                    "recall", "episodic.invalid_context", "context is invalid"
                )
            state = self._session._frames.get(self._spec.ref)
            entries = () if state is None else state.episodes
            if context is not None:
                try:
                    query = self._session._memory.rewrite_query(context)
                except Exception as exc:
                    raise _operation_error(
                        "recall", "episodic.invalid_query", "query rewriting failed"
                    ) from exc
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
                entries = tuple(entry for _, entry in ranked[: self._spec.recall_limit])
            return "\n".join(_canonical_json(entry) for entry in entries)

    def learn(self, context: LearnContext) -> list[MemoryMutation]:
        """Propose one deterministic append for every owner-scoped episode."""
        with self._session._memory._lock:
            self._session._ensure_open()
            if type(context) is not LearnContext or context.frame_ref != self._spec.ref:
                raise _operation_error(
                    "learn", "episodic.invalid_context", "learning context is invalid"
                )
            self._session._assert_current("learn")
            try:
                context = LearnContext(
                    context.run_id,
                    context.frame_ref,
                    context.episodes,
                    context.feedback,
                )
                for episode in context.episodes:
                    _validate_episode_owner(episode, self._spec.ref)
            except (TypeError, ValueError) as exc:
                raise _operation_error(
                    "learn", "episodic.invalid_context", "learning context is invalid"
                ) from exc
            mutations: list[MemoryMutation] = []
            for ordinal, episode in enumerate(context.episodes):
                encoded = _episode_json(episode)
                identity: JsonObject = {
                    "episode": encoded,
                    "frame_ref": _frame_ref_json(self._spec.ref),
                    "ordinal": ordinal,
                    "run_id": context.run_id,
                }
                entry_id = hashlib.sha256(
                    _canonical_json(identity).encode("utf-8")
                ).hexdigest()
                entry = _stored_entry(entry_id, episode)
                mutations.append(
                    MemoryMutation(
                        entry_id,
                        self._spec.ref,
                        {"entry": entry, "operation": "append"},
                    )
                )
            return mutations
