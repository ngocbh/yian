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

_MAX_JSON_NESTING = 128
"""Conservative memory JSON nesting bound that remains schema-4 wire-safe."""


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

_PRIVATE_CREDENTIAL_KEYS = frozenset(
    {
        "api_key",
        "access_token",
        "auth",
        "auth_token",
        "authorization",
        "basic_auth",
        "backend_token",
        "bearer_token",
        "client_secret",
        "connection_string",
        "credential",
        "credentials",
        "databricks_token",
        "docker_auth_config",
        "github_token",
        "google_application_credentials",
        "hf_token",
        "id_token",
        "mysql_pwd",
        "npm_token",
        "password",
        "passphrase",
        "passwd",
        "pgpassword",
        "private_key",
        "proxy_auth",
        "pwd",
        "refresh_token",
        "secret",
        "secret_access_key",
        "secret_key",
        "session_token",
        "shared_access_key",
        "shared_access_signature",
        "sig",
        "signature",
        "slack_bot_token",
        "sslpassword",
        "token",
        "webhook_signature",
        "webhook_url",
        "x_amz_security_token",
        "x_amz_signature",
        "x_goog_signature",
    }
)
_PRIVATE_CREDENTIAL_SUFFIXES = frozenset(
    {
        "account_key",
        "access_key",
        "access_key_id",
        "access_token",
        "api_key",
        "api_token",
        "application_credentials",
        "auth_key",
        "auth_token",
        "authorization",
        "bearer_key",
        "bearer_token",
        "connection_string",
        "credential",
        "credentials",
        "password",
        "passphrase",
        "private_key",
        "refresh_key",
        "refresh_token",
        "secret",
        "secret_access_key",
        "secret_key",
        "session_cookie",
        "session_key",
        "session_token",
        "shared_access_key",
        "shared_access_signature",
        "subscription_key",
        "webhook_url",
    }
)
_PRIVATE_TENANT_KEYS = frozenset({"tenant_id"})
_PRIVATE_PYTHON_KEYS = frozenset(
    {
        "class",
        "class_path",
        "import",
        "import_path",
        "module",
        "module_path",
        "python_class",
        "python_import",
        "python_module",
        "python_path",
    }
)
_PRIVATE_PYTHON_SUFFIXES = frozenset(
    {"class_path", "import_path", "module_path", "python_path"}
)
_PRIVATE_LOCATION_ROOTS = frozenset(
    {"backend", "database", "persistence", "storage"}
)
_PRIVATE_LOCATION_FIELDS = frozenset(
    {"connection", "connection_string", "dsn", "location", "path", "uri", "url"}
)
_PRIVATE_WEBHOOK_FIELDS = frozenset({"signature", "url"})
_PRIVATE_PYTHON_FIELDS = frozenset({"class", "import", "module", "path"})


def _require_string(value: object, name: str, *, nonempty: bool = True) -> str:
    """Return one exact string or reject an ambiguous identifier."""
    if type(value) is not str:
        raise TypeError(f"{name} must be a string")
    if nonempty and not value:
        raise ValueError(f"{name} must be non-empty")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError("Unicode surrogate is not valid canonical JSON") from exc
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
        _require_string(part, f"{name} components")
    return value


def _freeze_json(
    value: object,
    *,
    name: str,
    active: set[int] | None = None,
    nesting: int = 0,
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
        _require_string(value, name, nonempty=False)
        return _FrozenJsonScalar(_FrozenJsonScalarKind.STRING, value)
    if value_type is float:
        if not isfinite(value):
            raise ValueError(f"{name} JSON number must be finite")
        return _FrozenJsonScalar(_FrozenJsonScalarKind.NUMBER, value)

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
            return _FrozenJsonArray(
                tuple(
                    _freeze_json(
                        item,
                        name=f"{name}[{index}]",
                        active=active,
                        nesting=nesting + 1,
                    )
                    for index, item in enumerate(value)
                )
            )

        entries: list[tuple[str, _FrozenJson]] = []
        for key in value:
            if type(key) is not str:
                raise TypeError(f"{name} JSON object keys must be strings")
            _require_string(key, f"{name} JSON object key", nonempty=False)
        for key in sorted(value):
            entries.append(
                (
                    key,
                    _freeze_json(
                        value[key],
                        name=f"{name}.{key}",
                        active=active,
                        nesting=nesting + 1,
                    ),
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


def _normalize_config_key(value: str) -> str:
    """Return one case-, camel-, and punctuation-neutral authored key."""
    normalized: list[str] = []
    for index, character in enumerate(value):
        if not character.isalnum():
            if normalized and normalized[-1] != "_":
                normalized.append("_")
            continue
        previous = value[index - 1] if index else ""
        following = value[index + 1] if index + 1 < len(value) else ""
        if (
            character.isupper()
            and normalized
            and normalized[-1] != "_"
            and (
                previous.islower()
                or previous.isdigit()
                or (previous.isupper() and following.islower())
            )
        ):
            normalized.append("_")
        normalized.append(character.casefold())
    return "".join(normalized).strip("_")


def _matches_exact_private_key(value: str, vocabulary: frozenset[str]) -> bool:
    """Return whether one normalized key equals a spelling-neutral private key."""
    compact = value.replace("_", "")
    return any(
        value == item or compact == item.replace("_", "")
        for item in vocabulary
    )


def _matches_private_suffix(value: str, vocabulary: frozenset[str]) -> bool:
    """Return whether one key equals or ends in a high-confidence private term."""
    compact = value.replace("_", "")
    return any(
        value == item
        or value.endswith(f"_{item}")
        or compact == item.replace("_", "")
        for item in vocabulary
    )


def _reject_private_memory_keys(
    value: _FrozenJson,
    name: str,
    *,
    location_context: bool = False,
    python_context: bool = False,
    webhook_context: bool = False,
) -> None:
    """Reject recognized host-private fields from authored memory config."""
    if isinstance(value, _FrozenJsonArray):
        for index, item in enumerate(value.values):
            _reject_private_memory_keys(
                item,
                f"{name}[{index}]",
                location_context=location_context,
                python_context=python_context,
                webhook_context=webhook_context,
            )
        return
    if isinstance(value, _FrozenJsonObject):
        for key, item in value.entries:
            normalized = _normalize_config_key(key)
            tokens = frozenset(normalized.split("_"))
            private_location = location_context or bool(
                tokens & _PRIVATE_LOCATION_ROOTS
            )
            private_python = python_context or "python" in tokens
            private_webhook = webhook_context or "webhook" in tokens
            forbidden = (
                _matches_exact_private_key(normalized, _PRIVATE_CREDENTIAL_KEYS)
                or _matches_private_suffix(
                    normalized,
                    _PRIVATE_CREDENTIAL_SUFFIXES,
                )
                or _matches_private_suffix(normalized, _PRIVATE_TENANT_KEYS)
                or _matches_exact_private_key(normalized, _PRIVATE_PYTHON_KEYS)
                or _matches_private_suffix(normalized, _PRIVATE_PYTHON_SUFFIXES)
                or (
                    private_location
                    and _matches_private_suffix(
                        normalized,
                        _PRIVATE_LOCATION_FIELDS,
                    )
                )
                or (
                    private_python
                    and _matches_private_suffix(
                        normalized,
                        _PRIVATE_PYTHON_FIELDS,
                    )
                )
                or (
                    private_webhook
                    and _matches_private_suffix(
                        normalized,
                        _PRIVATE_WEBHOOK_FIELDS,
                    )
                )
            )
            if forbidden:
                raise ValueError(f"{name} contains forbidden private key {key!r}")
            _reject_private_memory_keys(
                item,
                f"{name}.{key}",
                location_context=private_location,
                python_context=private_python,
                webhook_context=private_webhook,
            )


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


@dataclass(frozen=True, slots=True, init=False)
class MemorySnapshotEnvelope:
    """One opaque public durable snapshot declared by a memory backend."""

    policy_name: str
    _snapshot: _FrozenJsonObject = field(repr=False)

    def __init__(self, policy_name: str, snapshot: JsonObject) -> None:
        """Validate the policy name and retain a defensive JSON snapshot."""
        frozen = _freeze_json_object(snapshot, "snapshot")
        object.__setattr__(
            self,
            "policy_name",
            _require_string(policy_name, "policy_name"),
        )
        object.__setattr__(
            self,
            "_snapshot",
            frozen,
        )

    @property
    def snapshot(self) -> JsonObject:
        """Return a fresh copy of the backend's source-neutral public export."""
        return _thaw_json_object(self._snapshot)


@dataclass(frozen=True, slots=True)
class MemoryOperationFailure:
    """Sanitized durable failure from one memory operation."""

    stage: str
    code: str
    message: str
    retryable: bool

    def __post_init__(self) -> None:
        """Require stable exact string fields and an exact retry flag."""
        _require_string(self.stage, "stage")
        _require_string(self.code, "code")
        _require_string(self.message, "message")
        if type(self.retryable) is not bool:
            raise TypeError("retryable must be a boolean")


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
        if self.static_path[0] != self.root_flow_id:
            raise ValueError("static_path must begin with root_flow_id")
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
        MemoryFrameRef(
            self.frame_ref.root_flow_id,
            self.frame_ref.static_path,
            self.frame_ref.owner,
        )
        _require_occurrence_path(self.occurrence_path)
        prior_path: tuple[str, ...] | None = None
        for segment in self.occurrence_path:
            MemoryOccurrenceSegment(segment.kind, segment.static_path, segment.index)
            if segment.static_path[0] != self.frame_ref.root_flow_id:
                raise ValueError("occurrence segment root must match the frame")
            if prior_path is None:
                frame_path = self.frame_ref.static_path
                comparable = (
                    frame_path[: len(segment.static_path)] == segment.static_path
                    or segment.static_path[: len(frame_path)] == frame_path
                )
                if not comparable:
                    raise ValueError(
                        "frame and first occurrence paths must be lexically comparable"
                    )
            if prior_path is not None and (
                len(segment.static_path) <= len(prior_path)
                or segment.static_path[: len(prior_path)] != prior_path
            ):
                raise ValueError(
                    "occurrence segments must follow lexical nesting order"
                )
            prior_path = segment.static_path


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
        MemoryFrameRef(
            self.frame_ref.root_flow_id,
            self.frame_ref.static_path,
            self.frame_ref.owner,
        )
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
        if self.static_path[0] != self.root_flow_id:
            raise ValueError("static_path must begin with root_flow_id")
        if type(self.frame_stack) is not tuple:
            raise TypeError("frame_stack must be a tuple")
        prior_path: tuple[str, ...] | None = None
        seen_frames: set[MemoryFrameRef] = set()
        for index, frame in enumerate(self.frame_stack):
            if type(frame) is not MemoryFrameRef:
                raise TypeError("frame_stack items must be MemoryFrameRef values")
            MemoryFrameRef(frame.root_flow_id, frame.static_path, frame.owner)
            if frame.root_flow_id != self.root_flow_id:
                raise ValueError("frame_stack root_flow_id must match the scope")
            if frame in seen_frames:
                raise ValueError("frame_stack must contain unique frame refs")
            if (
                prior_path is not None
                and frame.static_path[: len(prior_path)] != prior_path
            ):
                raise ValueError("frame_stack must follow lexical ownership order")
            if prior_path is not None and len(frame.static_path) == len(prior_path):
                raise ValueError("frame_stack ownership paths must be strictly nested")
            if frame.owner is MemoryOwner.AGENT and index != len(self.frame_stack) - 1:
                raise ValueError("an AGENT frame must be the final local frame")
            seen_frames.add(frame)
            prior_path = frame.static_path
        if self.frame_stack:
            inner = self.frame_stack[-1]
            if inner.owner is MemoryOwner.AGENT:
                if self.static_path != inner.static_path:
                    raise ValueError("an AGENT frame must own its exact node path")
            elif (
                len(self.static_path) <= len(inner.static_path)
                or self.static_path[: len(inner.static_path)] != inner.static_path
            ):
                raise ValueError("a FLOW frame must lexically contain the node path")
        _require_occurrence_path(self.occurrence_path)
        prior_occurrence_path: tuple[str, ...] | None = None
        for segment in self.occurrence_path:
            MemoryOccurrenceSegment(segment.kind, segment.static_path, segment.index)
            if segment.static_path[0] != self.root_flow_id:
                raise ValueError("occurrence segment root must match the scope")
            if prior_occurrence_path is not None and (
                len(segment.static_path) <= len(prior_occurrence_path)
                or segment.static_path[: len(prior_occurrence_path)]
                != prior_occurrence_path
            ):
                raise ValueError(
                    "occurrence segments must follow lexical nesting order"
                )
            prior_occurrence_path = segment.static_path
        if prior_occurrence_path is not None and (
            len(prior_occurrence_path) >= len(self.static_path)
            or self.static_path[: len(prior_occurrence_path)]
            != prior_occurrence_path
        ):
            raise ValueError(
                "the innermost occurrence segment must contain the scoped node"
            )
        if self.boundary is not None:
            if type(self.boundary) is not MemoryBoundary:
                raise TypeError("boundary must be a MemoryBoundary or None")
            MemoryBoundary(self.boundary.kind, self.boundary.frame_ref)
            if self.boundary.frame_ref.root_flow_id != self.root_flow_id:
                raise ValueError("boundary root_flow_id must match the scope")
            if not self.frame_stack or self.boundary.frame_ref != self.frame_stack[-1]:
                raise ValueError(
                    "boundary frame must be the innermost newly owned flow frame"
                )


@dataclass(frozen=True, slots=True)
class MemorySection:
    """One separately framed memory text section visible to an AGENT."""

    frame_ref: MemoryFrameRef
    text: str

    def __post_init__(self) -> None:
        """Require an exact frame identity and exact text value."""
        if type(self.frame_ref) is not MemoryFrameRef:
            raise TypeError("frame_ref must be a MemoryFrameRef")
        MemoryFrameRef(
            self.frame_ref.root_flow_id,
            self.frame_ref.static_path,
            self.frame_ref.owner,
        )
        _require_string(self.text, "text", nonempty=False)


@dataclass(frozen=True, slots=True)
class ModelTextInteraction:
    """One visible free-text model response."""

    text: str

    def __post_init__(self) -> None:
        """Require exact visible text without provider metadata."""
        _require_string(self.text, "text", nonempty=False)


@dataclass(frozen=True, slots=True, init=False)
class ModelValueInteraction:
    """One visible structured model response."""

    _value: _FrozenJson = field(repr=False)

    def __init__(self, value: JsonValue) -> None:
        """Validate and retain one defensive structured response."""
        object.__setattr__(self, "_value", _freeze_json(value, name="value"))

    @property
    def value(self) -> JsonValue:
        """Return a fresh plain-JSON copy of the visible response."""
        return _thaw_json(self._value)


@dataclass(frozen=True, slots=True, init=False)
class ToolCallInteraction:
    """One visible tool call name and JSON arguments."""

    name: str
    _arguments: _FrozenJsonObject = field(repr=False)

    def __init__(self, name: str, arguments: JsonObject) -> None:
        """Validate the name and retain defensive call arguments."""
        object.__setattr__(self, "name", _require_string(name, "name"))
        object.__setattr__(
            self,
            "_arguments",
            _freeze_json_object(arguments, "arguments"),
        )

    @property
    def arguments(self) -> JsonObject:
        """Return a fresh plain-JSON copy of the tool arguments."""
        return _thaw_json_object(self._arguments)


@dataclass(frozen=True, slots=True)
class ToolResultInteraction:
    """One visible tool result paired with its tool name."""

    name: str
    text: str

    def __post_init__(self) -> None:
        """Require an exact non-empty name and exact visible result text."""
        _require_string(self.name, "name")
        _require_string(self.text, "text", nonempty=False)


VisibleInteraction: TypeAlias = (
    ModelTextInteraction
    | ModelValueInteraction
    | ToolCallInteraction
    | ToolResultInteraction
)

_VISIBLE_INTERACTION_TYPES = (
    ModelTextInteraction,
    ModelValueInteraction,
    ToolCallInteraction,
    ToolResultInteraction,
)


def _revalidate_visible_interaction(interaction: VisibleInteraction) -> None:
    """Replay every invariant of one exact visible-interaction carrier."""
    if type(interaction) is ModelTextInteraction:
        ModelTextInteraction(interaction.text)
        return
    if type(interaction) is ModelValueInteraction:
        ModelValueInteraction(interaction.value)
        return
    if type(interaction) is ToolCallInteraction:
        ToolCallInteraction(interaction.name, interaction.arguments)
        return
    if type(interaction) is ToolResultInteraction:
        ToolResultInteraction(interaction.name, interaction.text)
        return
    raise TypeError("interactions items must be VisibleInteraction values")


@dataclass(frozen=True, slots=True, init=False)
class EvaluationEvidence:
    """Source-neutral visible evidence accepted with one node outcome."""

    scope: MemoryScope
    _bound_inputs: _FrozenJsonObject | None = field(repr=False)
    base_system_prompt: str | None
    prompt: str | None
    memory_sections: tuple[MemorySection, ...]
    interactions: tuple[VisibleInteraction, ...]

    def __init__(
        self,
        scope: MemoryScope,
        bound_inputs: JsonObject | None,
        base_system_prompt: str | None,
        prompt: str | None,
        memory_sections: tuple[MemorySection, ...] = (),
        interactions: tuple[VisibleInteraction, ...] = (),
    ) -> None:
        """Validate visible inputs, prompts, sections, and interactions."""
        if type(scope) is not MemoryScope:
            raise TypeError("scope must be a MemoryScope")
        MemoryScope(
            scope.root_flow_id,
            scope.static_path,
            scope.frame_stack,
            scope.occurrence_path,
            scope.boundary,
        )
        if not scope.frame_stack:
            raise ValueError("evaluation evidence requires a non-empty frame_stack")
        object.__setattr__(self, "scope", scope)
        if bound_inputs is not None and type(bound_inputs) is not dict:
            raise TypeError("bound_inputs must be an exact JSON object or None")
        object.__setattr__(
            self,
            "_bound_inputs",
            None
            if bound_inputs is None
            else _freeze_json_object(bound_inputs, "bound_inputs"),
        )
        if (base_system_prompt is None) != (prompt is None):
            raise ValueError(
                "base_system_prompt and prompt must both be present or absent"
            )
        if base_system_prompt is not None:
            _require_string(base_system_prompt, "base_system_prompt", nonempty=False)
            _require_string(prompt, "prompt", nonempty=False)
        object.__setattr__(self, "base_system_prompt", base_system_prompt)
        object.__setattr__(self, "prompt", prompt)

        if type(memory_sections) is not tuple:
            raise TypeError("memory_sections must be a tuple")
        stack_positions = {
            frame: index for index, frame in enumerate(scope.frame_stack)
        }
        seen_frames: set[MemoryFrameRef] = set()
        positions: list[int] = []
        for section in memory_sections:
            if type(section) is not MemorySection:
                raise TypeError("memory_sections items must be MemorySection values")
            MemorySection(section.frame_ref, section.text)
            if section.frame_ref in seen_frames:
                raise ValueError("memory_sections frame_ref values must be unique")
            if section.frame_ref not in stack_positions:
                raise ValueError(
                    "memory_sections frames must appear in scope.frame_stack"
                )
            seen_frames.add(section.frame_ref)
            positions.append(stack_positions[section.frame_ref])
        if positions != sorted(positions):
            raise ValueError("memory_sections must follow scope.frame_stack order")
        object.__setattr__(self, "memory_sections", memory_sections)

        if type(interactions) is not tuple:
            raise TypeError("interactions must be a tuple")
        for interaction in interactions:
            _revalidate_visible_interaction(interaction)
        object.__setattr__(self, "interactions", interactions)

    @property
    def bound_inputs(self) -> JsonObject | None:
        """Return bound inputs, or ``None`` when binding or safe capture was unavailable."""
        if self._bound_inputs is None:
            return None
        return _thaw_json_object(self._bound_inputs)


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
        frozen = _freeze_json_object(config, "config")
        _reject_private_memory_keys(frozen, "config")
        object.__setattr__(self, "_config", frozen)
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
        MemoryFrameRef(ref.root_flow_id, ref.static_path, ref.owner)
        object.__setattr__(self, "ref", ref)
        object.__setattr__(
            self,
            "policy_name",
            _require_string(policy_name, "policy_name"),
        )
        frozen = _freeze_json_object(config, "config")
        _reject_private_memory_keys(frozen, "config")
        object.__setattr__(self, "_config", frozen)

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
    _inputs: _FrozenJsonObject | None = field(repr=False)
    _output: _FrozenJson = field(repr=False)
    _failure: _FrozenJsonObject | None = field(repr=False)
    base_system_prompt: str | None
    prompt: str | None
    memory_sections: tuple[MemorySection, ...]
    interactions: tuple[VisibleInteraction, ...]

    def __init__(
        self,
        run_id: str,
        occurrence: MemoryOccurrenceRef,
        owner_static_path: tuple[str, ...],
        status: MemoryEpisodeStatus,
        inputs: JsonObject | None,
        output: JsonValue | None = None,
        failure: JsonObject | None = None,
        interactions: tuple[VisibleInteraction, ...] = (),
        base_system_prompt: str | None = None,
        prompt: str | None = None,
        memory_sections: tuple[MemorySection, ...] = (),
    ) -> None:
        """Validate one accepted success or failure episode and snapshot its JSON."""
        object.__setattr__(self, "run_id", _require_string(run_id, "run_id"))
        if type(occurrence) is not MemoryOccurrenceRef:
            raise TypeError("occurrence must be a MemoryOccurrenceRef")
        MemoryOccurrenceRef(occurrence.frame_ref, occurrence.occurrence_path)
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

        if inputs is not None and type(inputs) is not dict:
            raise TypeError("inputs must be an exact JSON object or None")
        object.__setattr__(
            self,
            "_inputs",
            None if inputs is None else _freeze_json_object(inputs, "inputs"),
        )
        object.__setattr__(self, "_output", _freeze_json(output, name="output"))
        object.__setattr__(
            self,
            "_failure",
            None if failure is None else _freeze_json_object(failure, "failure"),
        )
        if (base_system_prompt is None) != (prompt is None):
            raise ValueError(
                "base_system_prompt and prompt must both be present or absent"
            )
        if base_system_prompt is not None:
            _require_string(base_system_prompt, "base_system_prompt", nonempty=False)
            _require_string(prompt, "prompt", nonempty=False)
        object.__setattr__(self, "base_system_prompt", base_system_prompt)
        object.__setattr__(self, "prompt", prompt)
        if type(memory_sections) is not tuple:
            raise TypeError("memory_sections must be a tuple")
        for section in memory_sections:
            if type(section) is not MemorySection:
                raise TypeError("memory_sections items must be MemorySection values")
            MemorySection(section.frame_ref, section.text)
        object.__setattr__(self, "memory_sections", memory_sections)
        if type(interactions) is not tuple:
            raise TypeError("interactions must be a tuple")
        for interaction in interactions:
            _revalidate_visible_interaction(interaction)
        object.__setattr__(self, "interactions", interactions)

    @property
    def inputs(self) -> JsonObject | None:
        """Return inputs, or ``None`` when binding or safe capture was unavailable."""
        return None if self._inputs is None else _thaw_json_object(self._inputs)

    @property
    def output(self) -> JsonValue:
        """Return a fresh plain-JSON copy of the success output, or ``None``."""
        return _thaw_json(self._output)

    @property
    def failure(self) -> JsonObject | None:
        """Return a fresh plain-JSON copy of failure data when present."""
        return None if self._failure is None else _thaw_json_object(self._failure)

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
        MemoryFrameRef(frame_ref.root_flow_id, frame_ref.static_path, frame_ref.owner)
        object.__setattr__(self, "frame_ref", frame_ref)
        if type(episodes) is not tuple:
            raise TypeError("episodes must be a tuple")
        for episode in episodes:
            if type(episode) is not MemoryEpisode:
                raise TypeError("episodes items must be MemoryEpisode values")
            MemoryEpisode(
                run_id=episode.run_id,
                occurrence=episode.occurrence,
                owner_static_path=episode.owner_static_path,
                status=episode.status,
                inputs=episode.inputs,
                output=episode.output,
                failure=episode.failure,
                interactions=episode.interactions,
                base_system_prompt=episode.base_system_prompt,
                prompt=episode.prompt,
                memory_sections=episode.memory_sections,
            )
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
        MemoryFrameRef(frame_ref.root_flow_id, frame_ref.static_path, frame_ref.owner)
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
class EvaluationTrace(Protocol):
    """Write-only sink for visible evidence from one AGENT invocation.

    Implementations accept only the exact source-neutral carriers declared by
    this package.  The protocol deliberately exposes no read surface: the
    evaluator owns freezing the invocation-local trace into durable evidence.
    """

    def record_prompt(
        self,
        base_system_prompt: str,
        prompt: str,
        memory_sections: tuple[MemorySection, ...] = (),
    ) -> None:
        """Record the one visible model prompt established by an AGENT mode."""
        ...

    def record_interaction(self, interaction: VisibleInteraction) -> None:
        """Append one exact visible model or tool interaction."""
        ...


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
        """Return the backend-declared public durable source-neutral export.

        The export MUST omit host credentials, live handles, backend paths, and
        tenant identifiers. Its JSON is persisted verbatim for run recovery.
        """
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
