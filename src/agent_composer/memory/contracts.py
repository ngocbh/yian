"""Source-neutral value types and protocols for injected workflow memory."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from types import NoneType, TracebackType
from typing import Callable, Iterator, Protocol, TypeAlias, runtime_checkable

from agent_composer.diagnostics import OriginRef


JsonScalar: TypeAlias = NoneType | bool | int | float | str
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]


class _FrozenJsonScalarKind(Enum):
    """Exact scalar types retained by a private frozen JSON snapshot."""

    NULL = "null"
    BOOLEAN = "boolean"
    INTEGER = "integer"
    NUMBER = "number"
    STRING = "string"


@dataclass(frozen=True, slots=True)
class _FrozenJsonScalar:
    """One private JSON scalar whose exact Python type remains significant."""

    kind: _FrozenJsonScalarKind
    value: JsonScalar


@dataclass(frozen=True, slots=True)
class _FrozenJsonArray:
    """One private immutable JSON-array snapshot."""

    values: tuple[_FrozenJson, ...]


@dataclass(frozen=True, slots=True)
class _FrozenJsonObject:
    """One private immutable JSON-object snapshot in canonical key order."""

    entries: tuple[tuple[str, _FrozenJson], ...]


_FrozenJson: TypeAlias = _FrozenJsonScalar | _FrozenJsonArray | _FrozenJsonObject


def _require_string(value: object, name: str, *, nonempty: bool = True) -> str:
    """Return one exact string or reject an ambiguous identifier."""
    if type(value) is not str:
        raise TypeError(f"{name} must be a string")
    if nonempty and not value:
        raise ValueError(f"{name} must be non-empty")
    return value


def _require_path(
    value: object,
    name: str,
    *,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    """Return one exact immutable path whose components are non-empty strings."""
    if type(value) is not tuple:
        raise TypeError(f"{name} must be a tuple")
    if not allow_empty and not value:
        raise ValueError(f"{name} must be non-empty")
    for part in value:
        if type(part) is not str:
            raise TypeError(f"{name} components must be strings")
        if not part:
            raise ValueError(f"{name} components must be non-empty")
    return value


def _freeze_json(
    value: object,
    *,
    name: str,
    active: set[int] | None = None,
) -> _FrozenJson:
    """Validate and snapshot one exact recursive JSON value."""
    value_type = type(value)
    if value is None:
        return _FrozenJsonScalar(_FrozenJsonScalarKind.NULL, None)
    if value_type is bool:
        return _FrozenJsonScalar(_FrozenJsonScalarKind.BOOLEAN, value)
    if value_type is int:
        return _FrozenJsonScalar(_FrozenJsonScalarKind.INTEGER, value)
    if value_type is str:
        return _FrozenJsonScalar(_FrozenJsonScalarKind.STRING, value)
    if value_type is float:
        if not isfinite(value):
            raise ValueError(f"{name} JSON number must be finite")
        return _FrozenJsonScalar(_FrozenJsonScalarKind.NUMBER, value)

    if value_type not in (list, dict):
        raise TypeError(f"{name} must contain exact JSON values")

    active = set() if active is None else active
    identity = id(value)
    if identity in active:
        raise ValueError(f"{name} contains a circular JSON value")
    active.add(identity)
    try:
        if value_type is list:
            return _FrozenJsonArray(
                tuple(
                    _freeze_json(item, name=f"{name}[{index}]", active=active)
                    for index, item in enumerate(value)
                )
            )

        entries: list[tuple[str, _FrozenJson]] = []
        for key in value:
            if type(key) is not str:
                raise TypeError(f"{name} JSON object keys must be strings")
        for key in sorted(value):
            entries.append(
                (
                    key,
                    _freeze_json(value[key], name=f"{name}.{key}", active=active),
                )
            )
        return _FrozenJsonObject(tuple(entries))
    finally:
        active.remove(identity)


def _freeze_json_object(value: object, name: str) -> _FrozenJsonObject:
    """Validate and snapshot one exact JSON object."""
    if type(value) is not dict:
        raise TypeError(f"{name} must be an exact JSON object")
    frozen = _freeze_json(value, name=name)
    if not isinstance(frozen, _FrozenJsonObject):  # pragma: no cover - type guard
        raise AssertionError("JSON object validation returned a non-object")
    return frozen


def _thaw_json(value: _FrozenJson) -> JsonValue:
    """Return a fresh plain-JSON copy of an immutable snapshot."""
    if isinstance(value, _FrozenJsonScalar):
        return value.value
    if isinstance(value, _FrozenJsonArray):
        return [_thaw_json(item) for item in value.values]
    if isinstance(value, _FrozenJsonObject):
        return {key: _thaw_json(item) for key, item in value.entries}
    return value


def _thaw_json_object(value: _FrozenJsonObject) -> JsonObject:
    """Return a fresh exact JSON-object copy of an immutable snapshot."""
    return {key: _thaw_json(item) for key, item in value.entries}


def _require_origin(value: object, name: str = "origin") -> OriginRef | None:
    """Return one optional opaque source identity."""
    if value is not None and not isinstance(value, OriginRef):
        raise TypeError(f"{name} must be an OriginRef or None")
    if value is not None:
        # A subclass can bypass OriginRef.__post_init__; replay the base contract.
        OriginRef(value.key)
    return value


class MemoryOwner(str, Enum):
    """Closed set of semantic entities that may own a memory frame."""

    FLOW = "flow"
    AGENT = "agent"


class MemoryOccurrenceKind(str, Enum):
    """Closed set of dynamic occurrence segments."""

    MAP_ITEM = "map_item"
    LOOP_ITERATION = "loop_iteration"


class MemoryBoundaryKind(str, Enum):
    """Closed set of boundaries around one memory-owning flow."""

    START = "start"
    END = "end"


class MemoryEpisodeStatus(str, Enum):
    """Closed set of authoritative episode outcomes."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class MemoryOccurrenceSegment:
    """One structured MAP-item or LOOP-iteration occurrence segment."""

    kind: MemoryOccurrenceKind
    static_path: tuple[str, ...]
    index: int

    def __post_init__(self) -> None:
        """Validate the occurrence kind, authored path, and exact index."""
        if type(self.kind) is not MemoryOccurrenceKind:
            raise TypeError("kind must be a MemoryOccurrenceKind")
        _require_path(self.static_path, "static_path")
        if type(self.index) is not int:
            raise TypeError("index must be an integer")
        if self.index < 0:
            raise ValueError("index must be non-negative")


@dataclass(frozen=True, slots=True)
class MemoryFrameRef:
    """Stable authored identity of one flow- or AGENT-owned memory frame."""

    root_flow_id: str
    static_path: tuple[str, ...]
    owner: MemoryOwner

    def __post_init__(self) -> None:
        """Validate the root, static path, and closed owner vocabulary."""
        _require_string(self.root_flow_id, "root_flow_id")
        _require_path(self.static_path, "static_path")
        if type(self.owner) is not MemoryOwner:
            raise TypeError("owner must be a MemoryOwner")


def _require_occurrence_path(value: object) -> tuple[MemoryOccurrenceSegment, ...]:
    """Return one exact immutable path of structured occurrence segments."""
    if type(value) is not tuple:
        raise TypeError("occurrence_path must be a tuple")
    for segment in value:
        if type(segment) is not MemoryOccurrenceSegment:
            raise TypeError(
                "occurrence_path items must be MemoryOccurrenceSegment values"
            )
    return value


@dataclass(frozen=True, slots=True)
class MemoryOccurrenceRef:
    """One dynamic execution occurrence of a stable authored memory frame."""

    frame_ref: MemoryFrameRef
    occurrence_path: tuple[MemoryOccurrenceSegment, ...] = ()

    def __post_init__(self) -> None:
        """Validate the frame identity and immutable occurrence path."""
        if type(self.frame_ref) is not MemoryFrameRef:
            raise TypeError("frame_ref must be a MemoryFrameRef")
        _require_occurrence_path(self.occurrence_path)


@dataclass(frozen=True, slots=True)
class MemoryBoundary:
    """A START or END boundary belonging to one memory-owning flow frame."""

    kind: MemoryBoundaryKind
    frame_ref: MemoryFrameRef

    def __post_init__(self) -> None:
        """Require a closed boundary kind and a flow-owned frame."""
        if type(self.kind) is not MemoryBoundaryKind:
            raise TypeError("kind must be a MemoryBoundaryKind")
        if type(self.frame_ref) is not MemoryFrameRef:
            raise TypeError("frame_ref must be a MemoryFrameRef")
        if self.frame_ref.owner is not MemoryOwner.FLOW:
            raise ValueError("a memory boundary must belong to a flow frame")


@dataclass(frozen=True, slots=True)
class MemoryScope:
    """The immutable memory context carried by one compiled execution node."""

    root_flow_id: str
    static_path: tuple[str, ...]
    frame_stack: tuple[MemoryFrameRef, ...] = ()
    occurrence_path: tuple[MemoryOccurrenceSegment, ...] = ()
    boundary: MemoryBoundary | None = None

    def __post_init__(self) -> None:
        """Validate paths and keep every nested identity within one root flow."""
        _require_string(self.root_flow_id, "root_flow_id")
        _require_path(self.static_path, "static_path")
        if type(self.frame_stack) is not tuple:
            raise TypeError("frame_stack must be a tuple")
        for frame in self.frame_stack:
            if type(frame) is not MemoryFrameRef:
                raise TypeError("frame_stack items must be MemoryFrameRef values")
            if frame.root_flow_id != self.root_flow_id:
                raise ValueError("frame_stack root_flow_id must match the scope")
        _require_occurrence_path(self.occurrence_path)
        if self.boundary is not None:
            if type(self.boundary) is not MemoryBoundary:
                raise TypeError("boundary must be a MemoryBoundary or None")
            if self.boundary.frame_ref.root_flow_id != self.root_flow_id:
                raise ValueError("boundary root_flow_id must match the scope")


@dataclass(frozen=True, slots=True, init=False)
class MemoryPolicySpec:
    """One authored named memory policy and its source-neutral configuration."""

    name: str
    _config: _FrozenJsonObject = field(repr=False)
    origin: OriginRef | None

    def __init__(
        self,
        name: str,
        config: JsonObject,
        origin: OriginRef | None = None,
    ) -> None:
        """Validate and snapshot one authored policy specification."""
        object.__setattr__(self, "name", _require_string(name, "name"))
        object.__setattr__(self, "_config", _freeze_json_object(config, "config"))
        object.__setattr__(self, "origin", _require_origin(origin))

    @property
    def config(self) -> JsonObject:
        """Return a fresh plain-JSON copy of the policy configuration."""
        return _thaw_json_object(self._config)


def _iter_memory_policy_origin(
    spec: MemoryPolicySpec,
    seen: set[int],
) -> Iterator[OriginRef]:
    """Yield one policy origin while preserving shared-carrier identity."""
    if type(spec) is not MemoryPolicySpec:
        raise TypeError("memory policy must be an exact MemoryPolicySpec")
    identity = id(spec)
    if identity in seen:
        return
    seen.add(identity)
    origin = _require_origin(spec.origin, "memory policy origin")
    if origin is not None:
        yield origin


def _rewrite_memory_policy_origin(
    spec: MemoryPolicySpec,
    mapper: Callable[[OriginRef], OriginRef],
    seen: set[int],
) -> None:
    """Rewrite one policy origin in a private copy without replacing aliases."""
    if type(spec) is not MemoryPolicySpec:
        raise TypeError("memory policy must be an exact MemoryPolicySpec")
    identity = id(spec)
    if identity in seen:
        return
    seen.add(identity)
    origin = _require_origin(spec.origin, "memory policy origin")
    if origin is None:
        return
    mapped = mapper(origin)
    if not isinstance(mapped, OriginRef):
        raise TypeError("origin mapper must return an OriginRef")
    object.__setattr__(spec, "origin", OriginRef(mapped.key))


@dataclass(frozen=True, slots=True, init=False)
class MemoryFrameSpec:
    """Origin-free effective policy configuration for one memory frame."""

    ref: MemoryFrameRef
    policy_name: str
    _config: _FrozenJsonObject = field(repr=False)

    def __init__(
        self,
        ref: MemoryFrameRef,
        policy_name: str,
        config: JsonObject,
    ) -> None:
        """Validate and snapshot one effective frame specification."""
        if type(ref) is not MemoryFrameRef:
            raise TypeError("ref must be a MemoryFrameRef")
        object.__setattr__(self, "ref", ref)
        object.__setattr__(
            self,
            "policy_name",
            _require_string(policy_name, "policy_name"),
        )
        object.__setattr__(self, "_config", _freeze_json_object(config, "config"))

    @property
    def config(self) -> JsonObject:
        """Return a fresh plain-JSON copy of the frame configuration."""
        return _thaw_json_object(self._config)


@dataclass(frozen=True, slots=True, init=False)
class RecallContext:
    """Visible AGENT context from which a memory policy may derive a query."""

    base_system_prompt: str
    prompt: str
    _inputs: _FrozenJsonObject = field(repr=False)
    consumer_static_path: tuple[str, ...]

    def __init__(
        self,
        base_system_prompt: str,
        prompt: str,
        inputs: JsonObject,
        consumer_static_path: tuple[str, ...],
    ) -> None:
        """Validate and snapshot one recall context."""
        object.__setattr__(
            self,
            "base_system_prompt",
            _require_string(base_system_prompt, "base_system_prompt", nonempty=False),
        )
        object.__setattr__(
            self,
            "prompt",
            _require_string(prompt, "prompt", nonempty=False),
        )
        object.__setattr__(self, "_inputs", _freeze_json_object(inputs, "inputs"))
        object.__setattr__(
            self,
            "consumer_static_path",
            _require_path(consumer_static_path, "consumer_static_path"),
        )

    @property
    def inputs(self) -> JsonObject:
        """Return a fresh plain-JSON copy of the visible node inputs."""
        return _thaw_json_object(self._inputs)


@dataclass(frozen=True, slots=True, init=False)
class MemoryEpisode:
    """One reducer-accepted source-neutral execution episode."""

    run_id: str
    occurrence: MemoryOccurrenceRef
    owner_static_path: tuple[str, ...]
    status: MemoryEpisodeStatus
    _inputs: _FrozenJsonObject = field(repr=False)
    _output: _FrozenJson = field(repr=False)
    _failure: _FrozenJsonObject | None = field(repr=False)
    _interactions: tuple[_FrozenJsonObject, ...] = field(repr=False)

    def __init__(
        self,
        run_id: str,
        occurrence: MemoryOccurrenceRef,
        owner_static_path: tuple[str, ...],
        status: MemoryEpisodeStatus,
        inputs: JsonObject,
        output: JsonValue | None = None,
        failure: JsonObject | None = None,
        interactions: tuple[JsonObject, ...] = (),
    ) -> None:
        """Validate one accepted success or failure episode and snapshot its JSON."""
        object.__setattr__(self, "run_id", _require_string(run_id, "run_id"))
        if type(occurrence) is not MemoryOccurrenceRef:
            raise TypeError("occurrence must be a MemoryOccurrenceRef")
        object.__setattr__(self, "occurrence", occurrence)
        object.__setattr__(
            self,
            "owner_static_path",
            _require_path(owner_static_path, "owner_static_path"),
        )
        if type(status) is not MemoryEpisodeStatus:
            raise TypeError("status must be a MemoryEpisodeStatus")
        object.__setattr__(self, "status", status)

        if status is MemoryEpisodeStatus.SUCCEEDED and failure is not None:
            raise ValueError("a succeeded episode cannot carry failure data")
        if status is MemoryEpisodeStatus.FAILED:
            if failure is None:
                raise ValueError("a failed episode must carry failure data")
            if output is not None:
                raise ValueError("a failed episode cannot carry output data")

        if type(interactions) is not tuple:
            raise TypeError("interactions must be a tuple")
        frozen_interactions: list[_FrozenJsonObject] = []
        for index, interaction in enumerate(interactions):
            frozen_interactions.append(
                _freeze_json_object(interaction, f"interactions[{index}]")
            )

        object.__setattr__(self, "_inputs", _freeze_json_object(inputs, "inputs"))
        object.__setattr__(self, "_output", _freeze_json(output, name="output"))
        object.__setattr__(
            self,
            "_failure",
            None if failure is None else _freeze_json_object(failure, "failure"),
        )
        object.__setattr__(self, "_interactions", tuple(frozen_interactions))

    @property
    def inputs(self) -> JsonObject:
        """Return a fresh plain-JSON copy of the episode inputs."""
        return _thaw_json_object(self._inputs)

    @property
    def output(self) -> JsonValue:
        """Return a fresh plain-JSON copy of the success output, or ``None``."""
        return _thaw_json(self._output)

    @property
    def failure(self) -> JsonObject | None:
        """Return a fresh plain-JSON copy of failure data when present."""
        return None if self._failure is None else _thaw_json_object(self._failure)

    @property
    def interactions(self) -> tuple[JsonObject, ...]:
        """Return fresh JSON copies in their authoritative interaction order."""
        return tuple(_thaw_json_object(item) for item in self._interactions)


@dataclass(frozen=True, slots=True, init=False)
class LearnContext:
    """One frame-local ordered batch presented to a memory policy."""

    run_id: str
    frame_ref: MemoryFrameRef
    episodes: tuple[MemoryEpisode, ...]
    _feedback: tuple[_FrozenJson, ...] = field(repr=False)

    def __init__(
        self,
        run_id: str,
        frame_ref: MemoryFrameRef,
        episodes: tuple[MemoryEpisode, ...],
        feedback: tuple[JsonValue, ...] = (),
    ) -> None:
        """Validate and snapshot one frame-local learning context."""
        object.__setattr__(self, "run_id", _require_string(run_id, "run_id"))
        if type(frame_ref) is not MemoryFrameRef:
            raise TypeError("frame_ref must be a MemoryFrameRef")
        object.__setattr__(self, "frame_ref", frame_ref)
        if type(episodes) is not tuple:
            raise TypeError("episodes must be a tuple")
        for episode in episodes:
            if type(episode) is not MemoryEpisode:
                raise TypeError("episodes items must be MemoryEpisode values")
            if episode.run_id != self.run_id:
                raise ValueError("episode run_id must match the learning context")
            if episode.occurrence.frame_ref != frame_ref:
                raise ValueError(
                    "episode occurrence frame_ref must match the learning context"
                )
        object.__setattr__(self, "episodes", episodes)
        if type(feedback) is not tuple:
            raise TypeError("feedback must be a tuple")
        object.__setattr__(
            self,
            "_feedback",
            tuple(
                _freeze_json(item, name=f"feedback[{index}]")
                for index, item in enumerate(feedback)
            ),
        )

    @property
    def feedback(self) -> tuple[JsonValue, ...]:
        """Return fresh JSON copies in their reserved downstream-feedback order."""
        return tuple(_thaw_json(item) for item in self._feedback)


@dataclass(frozen=True, slots=True, init=False)
class MemoryMutation:
    """One policy-proposed JSON mutation for a single owning memory frame."""

    mutation_id: str
    frame_ref: MemoryFrameRef
    _payload: _FrozenJson = field(repr=False)

    def __init__(
        self,
        mutation_id: str,
        frame_ref: MemoryFrameRef,
        payload: JsonValue,
    ) -> None:
        """Validate and snapshot one proposed mutation."""
        object.__setattr__(
            self,
            "mutation_id",
            _require_string(mutation_id, "mutation_id"),
        )
        if type(frame_ref) is not MemoryFrameRef:
            raise TypeError("frame_ref must be a MemoryFrameRef")
        object.__setattr__(self, "frame_ref", frame_ref)
        object.__setattr__(self, "_payload", _freeze_json(payload, name="payload"))

    @property
    def payload(self) -> JsonValue:
        """Return a fresh plain-JSON copy of the mutation payload."""
        return _thaw_json(self._payload)


class MemoryOperationError(Exception):
    """Typed ordinary failure raised by a memory policy or session operation."""

    def __init__(
        self,
        stage: str,
        code: str,
        message: str,
        retryable: bool,
    ) -> None:
        """Create one stable policy/session failure."""
        self.stage = _require_string(stage, "stage")
        self.code = _require_string(code, "code")
        self.message = _require_string(message, "message")
        if type(retryable) is not bool:
            raise TypeError("retryable must be a boolean")
        self.retryable = retryable
        super().__init__(message)


class MemorySetupError(ValueError):
    """Invalid host memory setup, optionally attributed to authored policy syntax."""

    def __init__(self, message: str, *, origin: OriginRef | None = None) -> None:
        """Create one setup failure with optional opaque source identity."""
        message = _require_string(message, "message", nonempty=False)
        self.origin = _require_origin(origin)
        super().__init__(message)


class MemoryRecallError(Exception):
    """Ordinary recall failure attributed to an authored memory policy site."""

    def __init__(self, message: str, *, origin: OriginRef | None = None) -> None:
        """Create one recall wrapper; callers retain causes with ``raise ... from``."""
        message = _require_string(message, "message", nonempty=False)
        self.origin = _require_origin(origin)
        super().__init__(message)


@runtime_checkable
class MemoryFrame(Protocol):
    """One opened frame through which a policy recalls and proposes learning."""

    def recall(self, context: RecallContext | None = None) -> str:
        """Return the memory text visible to one AGENT invocation."""
        ...

    def learn(self, context: LearnContext) -> list[MemoryMutation]:
        """Return this frame's exact list of proposed mutations."""
        ...


@runtime_checkable
class MemorySession(Protocol):
    """One immutable-snapshot run session over a host-provided memory object."""

    @property
    def snapshot(self) -> JsonObject:
        """Return a defensive copy of the pinned canonical run snapshot."""
        ...

    def __enter__(self) -> MemorySession:
        """Enter this session."""
        ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        """Close this session when its run boundary exits."""
        ...

    def close(self) -> None:
        """Release resources held by this run session."""
        ...

    def open_frame(self, spec: MemoryFrameSpec) -> MemoryFrame:
        """Open one effective frame idempotently within this session."""
        ...

    def commit(
        self,
        commit_id: str,
        mutations: tuple[MemoryMutation, ...],
    ) -> None:
        """Atomically commit one immutable policy mutation batch."""
        ...


@runtime_checkable
class Memory(Protocol):
    """Host-injected memory object that opens run sessions and can be reset."""

    def open_run(self, *, snapshot: JsonObject | None = None) -> MemorySession:
        """Open one run session, optionally restoring a recorded snapshot."""
        ...

    def reset(self) -> None:
        """Clear current memory state according to the backend's policy."""
        ...
