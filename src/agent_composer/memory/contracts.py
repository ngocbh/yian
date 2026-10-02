"""Source-neutral value types and protocols for injected workflow memory."""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass, field, replace
from enum import Enum
import json
from math import isfinite
from types import NoneType, TracebackType
from typing import Callable, Iterator, Literal, Protocol, TypeAlias, runtime_checkable

from agent_composer.diagnostics import OriginRef


JsonScalar: TypeAlias = NoneType | bool | int | float | str
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]

_MAX_JSON_NESTING = 128
"""Conservative memory JSON nesting bound for nested record payloads."""


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


def _with_high_confidence_plurals(
    vocabulary: frozenset[str],
) -> frozenset[str]:
    """Add predictable plural forms to one private-term vocabulary."""
    plurals: set[str] = set()
    for item in vocabulary:
        if item.endswith("class"):
            plurals.add(f"{item}es")
        elif item.endswith("directory"):
            plurals.add(f"{item[:-1]}ies")
        elif item.endswith("s"):
            plurals.add(item)
        else:
            plurals.add(f"{item}s")
    return vocabulary | frozenset(plurals)


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
_PRIVATE_CREDENTIAL_SUFFIXES = _with_high_confidence_plurals(
    frozenset(
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
            "secret_token",
            "session_cookie",
            "session_key",
            "session_token",
            "shared_access_key",
            "shared_access_signature",
            "subscription_key",
            "webhook_url",
        }
    )
)
_PRIVATE_TENANT_KEYS = frozenset({"tenant_id"})
_PRIVATE_PYTHON_KEYS = frozenset(
    {
        "class",
        "class_path",
        "import",
        "import_path",
        "imports",
        "module",
        "module_path",
        "python_class",
        "python_import",
        "python_module",
        "python_path",
    }
)
_PRIVATE_PYTHON_SUFFIXES = _with_high_confidence_plurals(
    frozenset({"class_path", "import_path", "imports", "module_path", "python_path"})
)
_PRIVATE_LOCATION_ROOTS = _with_high_confidence_plurals(
    frozenset({"backend", "database", "persistence", "storage"})
)
_PRIVATE_LOCATION_FIELDS = _with_high_confidence_plurals(
    frozenset(
        {
            "connection",
            "connection_string",
            "dir",
            "directory",
            "dsn",
            "file",
            "location",
            "path",
            "root",
            "uri",
            "url",
        }
    )
)
_PRIVATE_INDIRECTION_FIELDS = _with_high_confidence_plurals(
    frozenset({"dir", "directory", "file", "path"})
)
_PRIVATE_WEBHOOK_FIELDS = frozenset({"signature", "url"})
_PRIVATE_PYTHON_FIELDS = _with_high_confidence_plurals(
    frozenset({"class", "import", "module", "path"})
)


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
        or compact.endswith(item.replace("_", ""))
        for item in vocabulary
    )


def _matches_private_credential_indirection(value: str) -> bool:
    """Return whether one key names a file or path containing credentials."""
    compact = value.replace("_", "")
    for location_field in _PRIVATE_INDIRECTION_FIELDS:
        suffix = f"_{location_field}"
        if value.endswith(suffix):
            stem = value[: -len(suffix)]
        else:
            compact_field = location_field.replace("_", "")
            if not compact.endswith(compact_field):
                continue
            stem = compact[: -len(compact_field)]
        if stem and (
            _matches_exact_private_key(stem, _PRIVATE_CREDENTIAL_KEYS)
            or _matches_private_suffix(stem, _PRIVATE_CREDENTIAL_SUFFIXES)
        ):
            return True
    return False


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
                or _matches_private_credential_indirection(normalized)
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


@dataclass(frozen=True, slots=True)
class DeclSite:
    """The definition site of one memory declaration.

    ``scope`` is the import-alias path from the root file (``()`` for the root
    file). ``owner`` is the lexical flow path compose threads: ``("root",)``,
    ``("def", name)``, or an inline body's path. ``node`` is the declaring AGENT
    id, or ``None`` for a flow declaration.
    """

    scope: tuple[str, ...]
    owner: tuple[str, ...]
    node: str | None = None

    def __post_init__(self) -> None:
        """Require exact string paths and an optional non-empty node id."""
        _require_path(self.scope, "scope", allow_empty=True)
        _require_path(self.owner, "owner")
        if self.node is not None:
            _require_string(self.node, "node")

    def sort_key(self) -> tuple[tuple[str, ...], tuple[str, ...], bool, str]:
        """Return a total order that puts a flow site before AGENT sites."""
        return (self.scope, self.owner, self.node is not None, self.node or "")


class MemoryOwner(str, Enum):
    """Closed set of semantic entities that may own a memory frame."""

    FLOW = "flow"
    AGENT = "agent"


class MemoryOccurrenceKind(str, Enum):
    """Closed set of dynamic occurrence segments."""

    MAP_ITEM = "map_item"
    LOOP_ITERATION = "loop_iteration"


class MemoryEpisodeStatus(str, Enum):
    """Closed set of authoritative episode outcomes."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass(frozen=True, slots=True, init=False)
class MemorySnapshotEnvelope:
    """One opaque public durable snapshot declared by a memory backend."""

    memory: str
    _snapshot: _FrozenJsonObject = field(repr=False)

    def __init__(self, memory: str, snapshot: JsonObject) -> None:
        """Validate the memory key and retain a defensive JSON snapshot."""
        frozen = _freeze_json_object(snapshot, "snapshot")
        object.__setattr__(
            self,
            "memory",
            _require_string(memory, "memory"),
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
    """Return one exact, revalidated, lexically nested occurrence path."""
    if type(value) is not tuple:
        raise TypeError("occurrence_path must be a tuple")
    prior: tuple[str, ...] | None = None
    for segment in value:
        if type(segment) is not MemoryOccurrenceSegment:
            raise TypeError(
                "occurrence_path items must be MemoryOccurrenceSegment values"
            )
        MemoryOccurrenceSegment(segment.kind, segment.static_path, segment.index)
        if prior is not None and (
            len(segment.static_path) <= len(prior)
            or segment.static_path[: len(prior)] != prior
        ):
            raise ValueError("occurrence segments must follow lexical nesting order")
        prior = segment.static_path
    return value


def _require_occurrence_contains(
    occurrence_path: tuple[MemoryOccurrenceSegment, ...],
    static_path: tuple[str, ...],
) -> None:
    """Require the innermost occurrence segment to strictly contain ``static_path``."""
    if not occurrence_path:
        return
    inner = occurrence_path[-1].static_path
    if len(inner) >= len(static_path) or static_path[: len(inner)] != inner:
        raise ValueError(
            "the innermost occurrence segment must contain the scoped node"
        )


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
class MemoryScope:
    """The memory keys and occurrence path carried by one compiled node.

    ``static_path`` is the node's root-relative locator. ``memories`` lists
    the keys the node recalls and learns into, outermost first.
    """

    static_path: tuple[str, ...]
    memories: tuple[str, ...] = ()
    occurrence_path: tuple[MemoryOccurrenceSegment, ...] = ()

    def __post_init__(self) -> None:
        """Validate the locator, unique keys, and the nested occurrence path."""
        _require_path(self.static_path, "static_path")
        _require_path(self.memories, "memories", allow_empty=True)
        if len(set(self.memories)) != len(self.memories):
            raise ValueError("memories must be unique")
        _require_occurrence_path(self.occurrence_path)
        _require_occurrence_contains(self.occurrence_path, self.static_path)


@dataclass(frozen=True, slots=True)
class MemorySection:
    """One memory's recalled text, shown to an AGENT as its own section."""

    memory: str
    text: str

    def __post_init__(self) -> None:
        """Require a non-empty memory key and an exact text value."""
        _require_string(self.memory, "memory")
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

    scope: MemoryScope | None
    _bound_inputs: _FrozenJsonObject | None = field(repr=False)
    base_system_prompt: str | None
    prompt: str | None
    memory_sections: tuple[MemorySection, ...]
    interactions: tuple[VisibleInteraction, ...]

    def __init__(
        self,
        scope: MemoryScope | None,
        bound_inputs: JsonObject | None,
        base_system_prompt: str | None,
        prompt: str | None,
        memory_sections: tuple[MemorySection, ...] = (),
        interactions: tuple[VisibleInteraction, ...] = (),
    ) -> None:
        """Validate visible inputs, prompts, sections, and interactions."""
        if scope is not None:
            if type(scope) is not MemoryScope:
                raise TypeError("scope must be a MemoryScope or None")
            MemoryScope(scope.static_path, scope.memories, scope.occurrence_path)
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
            key: index
            for index, key in enumerate(scope.memories if scope is not None else ())
        }
        seen: set[str] = set()
        positions: list[int] = []
        for section in memory_sections:
            if type(section) is not MemorySection:
                raise TypeError("memory_sections items must be MemorySection values")
            MemorySection(section.memory, section.text)
            if section.memory in seen:
                raise ValueError("memory_sections memory values must be unique")
            if section.memory not in stack_positions:
                raise ValueError(
                    "memory_sections memories must appear in scope.memories"
                )
            seen.add(section.memory)
            positions.append(stack_positions[section.memory])
        if positions != sorted(positions):
            raise ValueError("memory_sections must follow scope.memories order")
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
class MemoryDecl:
    """One authored memory declaration: name, backend type, config, and site."""

    name: str
    type: str
    _config: _FrozenJsonObject = field(repr=False)
    site: DeclSite
    origin: OriginRef | None

    def __init__(
        self,
        name: str,
        type: str,
        config: JsonObject,
        site: DeclSite,
        origin: OriginRef | None = None,
    ) -> None:
        """Validate and snapshot one declaration."""
        object.__setattr__(self, "name", _require_string(name, "name"))
        object.__setattr__(self, "type", _require_string(type, "type"))
        frozen = _freeze_json_object(config, "config")
        _reject_private_memory_keys(frozen, "config")
        object.__setattr__(self, "_config", frozen)
        if site.__class__ is not DeclSite:
            raise TypeError("site must be a DeclSite")
        object.__setattr__(self, "site", DeclSite(site.scope, site.owner, site.node))
        object.__setattr__(self, "origin", _require_origin(origin))

    @property
    def config(self) -> JsonObject:
        """Return a fresh plain-JSON copy of the declaration configuration."""
        return _thaw_json_object(self._config)

    @property
    def key(self) -> str:
        """Return the rendered memory key ``".".join((*site.scope, name))``."""
        return ".".join((*self.site.scope, self.name))


def _iter_memory_decl_origins(
    decls: tuple[MemoryDecl, ...],
    seen: set[int],
) -> Iterator[OriginRef]:
    """Yield declaration origins while preserving shared-carrier identity."""
    if type(decls) is not tuple:
        raise TypeError("memories must be a tuple")
    for decl in decls:
        if type(decl) is not MemoryDecl:
            raise TypeError("memories items must be exact MemoryDecl values")
        identity = id(decl)
        if identity in seen:
            continue
        seen.add(identity)
        origin = _require_origin(decl.origin, "memory declaration origin")
        if origin is not None:
            yield origin


def _rewrite_memory_decl_origins(
    decls: tuple[MemoryDecl, ...],
    mapper: Callable[[OriginRef], OriginRef],
    seen: set[int],
) -> None:
    """Rewrite declaration origins in a private copy without replacing aliases."""
    if type(decls) is not tuple:
        raise TypeError("memories must be a tuple")
    for decl in decls:
        if type(decl) is not MemoryDecl:
            raise TypeError("memories items must be exact MemoryDecl values")
        identity = id(decl)
        if identity in seen:
            continue
        seen.add(identity)
        origin = _require_origin(decl.origin, "memory declaration origin")
        if origin is None:
            continue
        mapped = mapper(origin)
        if not isinstance(mapped, OriginRef):
            raise TypeError("origin mapper must return an OriginRef")
        object.__setattr__(decl, "origin", OriginRef(mapped.key))


@dataclass(frozen=True, slots=True, init=False)
class MemorySpec:
    """One recorded memory: key, type, effective config, and owning sites."""

    key: str
    type: str
    _config: _FrozenJsonObject = field(repr=False)
    owners: tuple[DeclSite, ...]
    conversation_eligible: bool

    def __init__(
        self,
        key: str,
        type: str,
        config: JsonObject,
        owners: tuple[DeclSite, ...],
        conversation_eligible: bool,
    ) -> None:
        """Validate one table entry; eligibility must match its owners."""
        object.__setattr__(self, "key", _require_string(key, "key"))
        object.__setattr__(self, "type", _require_string(type, "type"))
        frozen = _freeze_json_object(config, "config")
        _reject_private_memory_keys(frozen, "config")
        object.__setattr__(self, "_config", frozen)
        if owners.__class__ is not tuple:
            raise TypeError("owners must be a tuple")
        if not owners:
            raise ValueError("owners must be non-empty")
        copied = []
        for owner in owners:
            if owner.__class__ is not DeclSite:
                raise TypeError("owners items must be DeclSite values")
            copied.append(DeclSite(owner.scope, owner.owner, owner.node))
        keys = [owner.sort_key() for owner in copied]
        if keys != sorted(set(keys)):
            raise ValueError("owners must be sorted and unique by DeclSite.sort_key()")
        object.__setattr__(self, "owners", tuple(copied))
        if conversation_eligible.__class__ is not bool:
            raise TypeError("conversation_eligible must be a boolean")
        expected = len(copied) == 1 and copied[0].node is not None
        if conversation_eligible != expected:
            raise ValueError(
                "conversation_eligible must be true exactly for one AGENT owner"
            )
        object.__setattr__(self, "conversation_eligible", conversation_eligible)

    @property
    def config(self) -> JsonObject:
        """Return a fresh plain-JSON copy of the effective configuration."""
        return _thaw_json_object(self._config)


@dataclass(frozen=True, slots=True)
class MemoryStack:
    """The ordered memory keys one static target recalls and learns into."""

    locator: tuple[str, ...]
    memories: tuple[str, ...]

    def __post_init__(self) -> None:
        """Require a non-empty locator and unique non-empty keys."""
        _require_path(self.locator, "locator")
        _require_path(self.memories, "memories", allow_empty=True)
        if len(set(self.memories)) != len(self.memories):
            raise ValueError("memories must be unique")


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
    """One reducer-accepted source-neutral execution episode.

    ``execution_seq`` optionally identifies the physical accepted node-start
    record; it is independent of episode order and outcome-record position.
    Host-created episodes may omit it, and source coordinate zero is valid.
    """

    run_id: str
    occurrence_path: tuple[MemoryOccurrenceSegment, ...]
    owner_static_path: tuple[str, ...]
    status: MemoryEpisodeStatus
    _inputs: _FrozenJsonObject | None = field(repr=False)
    _output: _FrozenJson = field(repr=False)
    _failure: _FrozenJsonObject | None = field(repr=False)
    base_system_prompt: str | None
    prompt: str | None
    memory_sections: tuple[MemorySection, ...]
    interactions: tuple[VisibleInteraction, ...]
    execution_seq: int | None

    def __init__(
        self,
        run_id: str,
        occurrence_path: tuple[MemoryOccurrenceSegment, ...],
        owner_static_path: tuple[str, ...],
        status: MemoryEpisodeStatus,
        inputs: JsonObject | None,
        output: JsonValue | None = None,
        failure: JsonObject | None = None,
        interactions: tuple[VisibleInteraction, ...] = (),
        base_system_prompt: str | None = None,
        prompt: str | None = None,
        memory_sections: tuple[MemorySection, ...] = (),
        *,
        execution_seq: int | None = None,
    ) -> None:
        """Validate one accepted success or failure episode and snapshot its JSON.

        ``execution_seq`` is an optional exact nonnegative integer. Malformed
        fields raise ``TypeError``; inconsistent values or excessive JSON nesting
        raise ``ValueError``. Existing positional host construction is preserved.
        """
        object.__setattr__(self, "run_id", _require_string(run_id, "run_id"))
        if execution_seq is not None:
            if type(execution_seq) is not int:
                raise TypeError("execution_seq must be an integer or None")
            if execution_seq < 0:
                raise ValueError("execution_seq must be non-negative")
        object.__setattr__(self, "execution_seq", execution_seq)
        object.__setattr__(
            self, "occurrence_path", _require_occurrence_path(occurrence_path)
        )
        object.__setattr__(
            self,
            "owner_static_path",
            _require_path(owner_static_path, "owner_static_path"),
        )
        _require_occurrence_contains(self.occurrence_path, self.owner_static_path)
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
            MemorySection(section.memory, section.text)
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
    """One memory's ordered learning batch."""

    run_id: str
    memory: str
    episodes: tuple[MemoryEpisode, ...]
    _feedback: tuple[_FrozenJson, ...] = field(repr=False)

    def __init__(
        self,
        run_id: str,
        memory: str,
        episodes: tuple[MemoryEpisode, ...],
        feedback: tuple[JsonValue, ...] = (),
    ) -> None:
        """Validate and snapshot one memory's learning context."""
        object.__setattr__(self, "run_id", _require_string(run_id, "run_id"))
        object.__setattr__(self, "memory", _require_string(memory, "memory"))
        if type(episodes) is not tuple:
            raise TypeError("episodes must be a tuple")
        for episode in episodes:
            if type(episode) is not MemoryEpisode:
                raise TypeError("episodes items must be MemoryEpisode values")
            MemoryEpisode(
                run_id=episode.run_id,
                occurrence_path=episode.occurrence_path,
                owner_static_path=episode.owner_static_path,
                status=episode.status,
                inputs=episode.inputs,
                output=episode.output,
                failure=episode.failure,
                interactions=episode.interactions,
                base_system_prompt=episode.base_system_prompt,
                prompt=episode.prompt,
                memory_sections=episode.memory_sections,
                execution_seq=episode.execution_seq,
            )
            if episode.run_id != self.run_id:
                raise ValueError("episode run_id must match the learning context")
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
    """One JSON mutation a memory proposed."""

    mutation_id: str
    _payload: _FrozenJson = field(repr=False)

    def __init__(
        self,
        mutation_id: str,
        payload: JsonValue,
    ) -> None:
        """Validate and snapshot one proposed mutation."""
        object.__setattr__(
            self,
            "mutation_id",
            _require_string(mutation_id, "mutation_id"),
        )
        object.__setattr__(self, "_payload", _freeze_json(payload, name="payload"))

    @property
    def payload(self) -> JsonValue:
        """Return a fresh plain-JSON copy of the mutation payload."""
        return _thaw_json(self._payload)


def _validate_mutations(
    mutations: tuple[MemoryMutation, ...],
) -> tuple[MemoryMutation, ...]:
    """Revalidate and detach a mutation batch, preserving its semantic order."""
    if type(mutations) is not tuple:
        raise TypeError("mutations must be a tuple")
    copied = []
    for mutation in mutations:
        if type(mutation) is not MemoryMutation:
            raise TypeError("mutations items must be MemoryMutation values")
        copied.append(MemoryMutation(mutation.mutation_id, mutation.payload))
    ids = tuple(mutation.mutation_id for mutation in mutations)
    if len(ids) != len(set(ids)):
        raise ValueError("mutation IDs must be unique")
    return tuple(copied)


@dataclass(frozen=True, slots=True)
class LearningFailure:
    """One memory's learn or commit failure in a run result."""

    memory: str
    operation: str
    failure: MemoryOperationFailure

    def __post_init__(self) -> None:
        """Require exact attribution and one revalidated durable failure."""
        _require_string(self.memory, "memory")
        _require_string(self.operation, "operation")
        if self.operation not in ("learn", "commit"):
            raise ValueError("operation must be 'learn' or 'commit'")
        if type(self.failure) is not MemoryOperationFailure:
            raise TypeError("failure must be a MemoryOperationFailure")
        MemoryOperationFailure(
            self.failure.stage,
            self.failure.code,
            self.failure.message,
            self.failure.retryable,
        )


@dataclass(frozen=True, slots=True)
class PolicyCommitOutcome:
    """One policy's confirmed or still-unresolved commit disposition.

    Attributes:
        memory (`str`): Non-empty memory key.
        status (`str`): `unrequested`, `unknown`, `succeeded`, or `failed`.
        commit_id (`str | None`): Nonempty protocol identity after a request;
            defaults to `None` and must be absent when unrequested.
        failure (`MemoryOperationFailure | None`): Defaults to `None`; required
            only for a confirmed failed commit. Backend stage and retryability
            remain unchanged.
    """

    memory: str
    status: str
    commit_id: str | None = None
    failure: MemoryOperationFailure | None = None

    def __post_init__(self) -> None:
        """Validate exact identities and detach the optional failure carrier."""
        _require_string(self.memory, "memory")
        _require_string(self.status, "status")
        if self.status not in ("unrequested", "unknown", "succeeded", "failed"):
            raise ValueError("invalid policy commit status")
        if self.commit_id is not None:
            _require_string(self.commit_id, "commit_id")
        if self.failure is not None:
            if type(self.failure) is not MemoryOperationFailure:
                raise TypeError("failure must be a MemoryOperationFailure or None")
            object.__setattr__(self, "failure", replace(self.failure))
        if self.status == "unrequested":
            if self.commit_id is not None or self.failure is not None:
                raise ValueError("unrequested policy cannot carry commit data")
        elif self.commit_id is None:
            raise ValueError("requested policy requires commit_id")
        if (self.status == "failed") != (self.failure is not None):
            raise ValueError("only failed policy commits require a failure")


@dataclass(frozen=True, slots=True)
class LearningResult:
    """Separate learning completion from backend retryability hints.

    Attributes:
        status (`str`): `succeeded`, `failed`, or `incomplete`. Success requires
            no failures and only successful supplied policies. Failure requires
            an actual attributed failure. Incomplete does not infer completion
            from the known policy dispositions.
        failures (`tuple[LearningFailure, ...]`): Unique failure attributions in
            supplied order; defaults to an empty tuple.
        policies (`tuple[PolicyCommitOutcome, ...]`): Unique memory keys in
            supplied order; defaults to an empty tuple. Recorded run layout
            owns configured membership and order.
    """

    status: str
    failures: tuple[LearningFailure, ...] = ()
    policies: tuple[PolicyCommitOutcome, ...] = ()

    def __post_init__(self) -> None:
        """Validate supplied dispositions without inferring attempt completion."""
        _require_string(self.status, "status")
        if self.status not in ("succeeded", "failed", "incomplete"):
            raise ValueError(
                "learning status must be 'succeeded', 'failed', or 'incomplete'"
            )
        if type(self.failures) is not tuple:
            raise TypeError("learning failures must be a tuple")
        failures = []
        identities: set[tuple[str, str]] = set()
        for failure in self.failures:
            if type(failure) is not LearningFailure:
                raise TypeError("failures items must be LearningFailure values")
            failures.append(replace(failure))
            identity = (failure.memory, failure.operation)
            if identity in identities:
                raise ValueError("learning failure attribution must be unique")
            identities.add(identity)
        if type(self.policies) is not tuple:
            raise TypeError("learning policies must be a tuple")
        policies = []
        names: set[str] = set()
        for policy in self.policies:
            if type(policy) is not PolicyCommitOutcome:
                raise TypeError("policies items must be PolicyCommitOutcome values")
            policies.append(replace(policy))
            if policy.memory in names:
                raise ValueError("learning memory keys must be unique")
            names.add(policy.memory)
        object.__setattr__(self, "policies", tuple(policies))
        object.__setattr__(self, "failures", tuple(failures))
        if self.status == "succeeded" and self.failures:
            raise ValueError("succeeded learning cannot carry failures")
        if self.status == "succeeded" and any(p.status != "succeeded" for p in policies):
            raise ValueError("succeeded learning requires succeeded policies")
        if self.status == "failed" and not self.failures:
            raise ValueError("failed learning requires at least one failure")


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

    def recall(self, context: RecallContext | None = None) -> str | MemoryContent:
        """Return notes and an optional transcript source for one invocation."""
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

    def recall(self, context: RecallContext | None = None) -> str | MemoryContent:
        """Return notes and an optional transcript source for one invocation.

        ``None`` asks for the whole memory, as a host inspection does.
        """
        ...

    def learn(self, context: LearnContext) -> list[MemoryMutation]:
        """Return this memory's exact list of proposed mutations for one batch."""
        ...

    def commit(
        self,
        commit_id: str,
        mutations: tuple[MemoryMutation, ...],
    ) -> None:
        """Atomically commit one immutable policy mutation batch.

        ``commit_id`` is stable within one backward attempt, so a repeated
        commit of that attempt may be a no-op. Every later attempt that
        requests commits gets a distinct ID, even for an identical batch.
        """
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


def _require_counter(value: object, name: str, *, positive: bool = False) -> int:
    """Require an exact integer counter without accepting booleans."""
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if value < (1 if positive else 0):
        raise ValueError(f"{name} must be {'positive' if positive else 'non-negative'}")
    return value


def _copy_invocation(value: object) -> ConversationInvocation:
    """Validate and detach one exact invocation identity."""
    if type(value) is not ConversationInvocation:
        raise TypeError("invocation must be a ConversationInvocation")
    return replace(value)


@dataclass(frozen=True, slots=True)
class ConversationInvocation:
    """One run's accepted node start, independent of its conversation owner."""

    run_id: str
    start_sequence: int

    def __post_init__(self) -> None:
        """Require a nonempty run identifier and positive start sequence."""
        _require_string(self.run_id, "run_id")
        _require_counter(self.start_sequence, "start_sequence", positive=True)


@dataclass(frozen=True, slots=True, init=False)
class ConversationRequest:
    """An invocation's authored consumer and detached compatibility descriptor."""

    consumer_static_path: tuple[str, ...]
    invocation: ConversationInvocation
    _compatibility: _FrozenJsonObject = field(repr=False)

    def __init__(
        self,
        consumer_static_path: tuple[str, ...],
        invocation: ConversationInvocation,
        compatibility: JsonObject,
    ) -> None:
        """Validate identity and snapshot the opaque JSON descriptor.

        Args:
            consumer_static_path (`tuple[str, ...]`): Nonempty authored leaf path.
            invocation (`ConversationInvocation`): Exact accepted node-start identity.
            compatibility (`JsonObject`): Exact JSON compatibility descriptor.

        Raises:
            TypeError: An argument has the wrong exact carrier or JSON type.
            ValueError: A path, identifier, or JSON value is invalid.
        """
        object.__setattr__(
            self,
            "consumer_static_path",
            _require_path(consumer_static_path, "consumer_static_path"),
        )
        object.__setattr__(self, "invocation", _copy_invocation(invocation))
        object.__setattr__(
            self, "_compatibility", _freeze_json_object(compatibility, "compatibility")
        )

    @property
    def compatibility(self) -> JsonObject:
        """Return a fresh JSON copy of the requested compatibility descriptor."""
        return _thaw_json_object(self._compatibility)


@dataclass(frozen=True, slots=True, init=False)
class ConversationItem:
    """One immutable, uniquely identified opaque conversation payload."""

    item_id: str
    format: str
    _payload: _FrozenJsonObject = field(repr=False)

    def __init__(self, item_id: str, format: str, payload: JsonObject) -> None:
        """Require stable nonempty identifiers and detach the JSON payload.

        Args:
            item_id (`str`): Nonempty stable identifier unique in its conversation.
            format (`str`): Nonempty opaque payload format identifier.
            payload (`JsonObject`): Exact JSON data, not a provider SDK object.

        Raises:
            TypeError: An identifier or payload has the wrong exact type.
            ValueError: An identifier or JSON value is invalid.
        """
        object.__setattr__(self, "item_id", _require_string(item_id, "item_id"))
        object.__setattr__(self, "format", _require_string(format, "format"))
        object.__setattr__(self, "_payload", _freeze_json_object(payload, "payload"))

    @property
    def payload(self) -> JsonObject:
        """Return a fresh JSON copy without exposing retained state."""
        return _thaw_json_object(self._payload)


@dataclass(frozen=True, slots=True)
class ConversationStamp:
    """The conversation generation, predecessor revision, and operation identity."""

    invocation: ConversationInvocation
    namespace: str
    generation: int
    expected_revision: int
    ordinal: int

    def __post_init__(self) -> None:
        """Detach exact identity carriers and validate nonnegative counters."""
        _require_string(self.namespace, "namespace")
        object.__setattr__(self, "invocation", _copy_invocation(self.invocation))
        _require_counter(self.generation, "generation")
        _require_counter(self.expected_revision, "expected_revision")
        _require_counter(self.ordinal, "ordinal")


ConversationEffectKind: TypeAlias = Literal["model", "tool"]
ConversationEffectStatus: TypeAlias = Literal[
    "pending", "completed", "failed", "interrupted"
]
ConversationTurnStatus: TypeAlias = Literal["completed", "failed", "interrupted"]


def _require_effect_kind(value: object) -> ConversationEffectKind:
    """Require one exact member of the closed effect vocabulary."""
    _require_string(value, "kind")
    if value not in ("model", "tool"):
        raise ValueError("unsupported conversation effect kind")
    return value


def _require_turn_status(value: object) -> ConversationTurnStatus:
    """Require a terminal disposition without interpreting its cause."""
    _require_string(value, "status")
    if value not in ("completed", "failed", "interrupted"):
        raise ValueError("invalid conversation turn status")
    return value


def _copy_effect_failure(
    status: object,
    failure: object,
) -> MemoryOperationFailure | None:
    """Validate an effect disposition and detach its optional known failure."""
    _require_string(status, "status")
    if status not in ("pending", "completed", "failed", "interrupted"):
        raise ValueError("invalid conversation effect status")
    if failure is not None:
        if type(failure) is not MemoryOperationFailure:
            raise TypeError("failure must be a MemoryOperationFailure or None")
        failure = replace(failure)
    if status in ("pending", "completed") and failure is not None:
        raise ValueError("pending or completed effect cannot carry a failure")
    if status == "failed" and failure is None:
        raise ValueError("failed effect requires a failure")
    return failure


@dataclass(frozen=True, slots=True, init=False)
class ConversationEffectState:
    """One requested effect with its actual ordered observation disposition."""

    invocation: ConversationInvocation
    effect_ordinal: int
    kind: ConversationEffectKind
    _request: _FrozenJsonObject = field(repr=False)
    status: ConversationEffectStatus
    observation_ordinals: tuple[int, ...]
    failure: MemoryOperationFailure | None

    def __init__(
        self,
        invocation: ConversationInvocation,
        effect_ordinal: int,
        kind: ConversationEffectKind,
        request: JsonObject,
        status: ConversationEffectStatus,
        observation_ordinals: tuple[int, ...],
        failure: MemoryOperationFailure | None,
    ) -> None:
        """Snapshot the request and validate ordered, later observation ordinals.

        Args:
            invocation (`ConversationInvocation`): Exact requesting invocation.
            effect_ordinal (`int`): Positive request ordinal within that invocation.
            kind (`ConversationEffectKind`): Model or tool effect.
            request (`JsonObject`): Detached original effect request.
            status (`ConversationEffectStatus`): Actual latest disposition.
            observation_ordinals (`tuple[int, ...]`): Strictly increasing positive
                ordinals after the request; required for a terminal disposition.
            failure (`MemoryOperationFailure | None`): Required for failed, optional
                for interrupted, and absent for pending or completed effects.

        Raises:
            TypeError: An argument has the wrong exact carrier or JSON type.
            ValueError: Identity, ordering, JSON, or disposition rules are violated.
        """
        object.__setattr__(self, "invocation", _copy_invocation(invocation))
        object.__setattr__(
            self,
            "effect_ordinal",
            _require_counter(effect_ordinal, "effect_ordinal", positive=True),
        )
        object.__setattr__(self, "kind", _require_effect_kind(kind))
        object.__setattr__(self, "_request", _freeze_json_object(request, "request"))
        object.__setattr__(self, "failure", _copy_effect_failure(status, failure))
        if type(observation_ordinals) is not tuple:
            raise TypeError("observation_ordinals must be a tuple")
        previous = effect_ordinal
        for ordinal in observation_ordinals:
            _require_counter(ordinal, "observation ordinal", positive=True)
            if ordinal <= previous:
                raise ValueError("observation ordinals must increase after the request")
            previous = ordinal
        if status != "pending" and not observation_ordinals:
            raise ValueError("terminal effect requires an observation")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "observation_ordinals", observation_ordinals)

    @property
    def request(self) -> JsonObject:
        """Return a fresh JSON copy of the original effect request."""
        return _thaw_json_object(self._request)


@dataclass(frozen=True, slots=True)
class ConversationCompletion:
    """An invocation's actual closed ordinal and terminal disposition."""

    invocation: ConversationInvocation
    closed_ordinal: int
    status: ConversationTurnStatus

    def __post_init__(self) -> None:
        """Require an exact invocation, positive ordinal, and terminal status."""
        object.__setattr__(self, "invocation", _copy_invocation(self.invocation))
        _require_counter(self.closed_ordinal, "closed_ordinal", positive=True)
        _require_turn_status(self.status)


@dataclass(frozen=True, slots=True)
class ConversationApplied:
    """Content evidence for one exactly identified applied checkpoint."""

    operation_id: str
    event_digest: str
    before_digest: str
    after_digest: str

    def __post_init__(self) -> None:
        """Validate canonical operation identity and lowercase SHA-256 shape."""
        _require_string(self.operation_id, "operation_id")
        parts = json.loads(self.operation_id)
        if type(parts) is not list or len(parts) != 3:
            raise ValueError(
                "operation_id must encode [run_id, start_sequence, ordinal]"
            )
        _require_string(parts[0], "run_id")
        _require_counter(parts[1], "start_sequence", positive=True)
        _require_counter(parts[2], "ordinal")
        if (
            json.dumps(parts, ensure_ascii=False, separators=(",", ":"))
            != self.operation_id
        ):
            raise ValueError("operation_id must be canonical JSON")
        for name in ("event_digest", "before_digest", "after_digest"):
            digest = _require_string(getattr(self, name), name)
            if len(digest) != 64 or any(
                char not in "0123456789abcdef" for char in digest
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _copy_conversation_items(value: object) -> tuple[ConversationItem, ...]:
    """Detach an exact item tuple while preserving order and unique IDs."""
    if type(value) is not tuple:
        raise TypeError("items must be a tuple")
    result = []
    seen: set[str] = set()
    for item in value:
        if type(item) is not ConversationItem:
            raise TypeError("items must contain ConversationItem values")
        detached = ConversationItem(item.item_id, item.format, item.payload)
        if detached.item_id in seen:
            raise ValueError("conversation item IDs must be unique")
        seen.add(detached.item_id)
        result.append(detached)
    return tuple(result)


@dataclass(frozen=True, slots=True, init=False)
class ConversationView:
    """Complete detached conversation state with shaped application evidence.

    Construction checks carrier shape and identity uniqueness. The pure
    conversation fold separately proves cross-event ordering and content digests.
    """

    namespace: str
    generation: int
    revision: int
    _compatibility: _FrozenJsonObject = field(repr=False)
    items: tuple[ConversationItem, ...]
    effects: tuple[ConversationEffectState, ...]
    closed_invocations: tuple[ConversationCompletion, ...]
    applied: tuple[ConversationApplied, ...]

    def __init__(
        self,
        namespace: str,
        generation: int,
        revision: int,
        compatibility: JsonObject,
        items: tuple[ConversationItem, ...],
        effects: tuple[ConversationEffectState, ...],
        closed_invocations: tuple[ConversationCompletion, ...],
        applied: tuple[ConversationApplied, ...],
    ) -> None:
        """Detach all supplied state and reject malformed or duplicate identities.

        Args:
            namespace (`str`): Nonempty backend namespace identity.
            generation (`int`): Nonnegative reset generation.
            revision (`int`): Nonnegative count of accepted checkpoints.
            compatibility (`JsonObject`): Retained opaque compatibility descriptor.
            items (`tuple[ConversationItem, ...]`): Ordered items with unique IDs.
            effects (`tuple[ConversationEffectState, ...]`): Unique invocation and
                request-ordinal pairs in retained order.
            closed_invocations (`tuple[ConversationCompletion, ...]`): One actual
                completion per closed invocation, in retained order.
            applied (`tuple[ConversationApplied, ...]`): One unique operation entry
                per revision. Digest-chain proof belongs to the conversation fold.

        Raises:
            TypeError: An argument has the wrong exact carrier or JSON type.
            ValueError: Shape, identity, uniqueness, or revision rules are violated.
        """
        object.__setattr__(self, "namespace", _require_string(namespace, "namespace"))
        object.__setattr__(
            self, "generation", _require_counter(generation, "generation")
        )
        object.__setattr__(self, "revision", _require_counter(revision, "revision"))
        object.__setattr__(
            self, "_compatibility", _freeze_json_object(compatibility, "compatibility")
        )
        object.__setattr__(self, "items", _copy_conversation_items(items))
        if type(effects) is not tuple:
            raise TypeError("effects must be a tuple")
        detached_effects = []
        effect_ids: set[tuple[ConversationInvocation, int]] = set()
        for effect in effects:
            if type(effect) is not ConversationEffectState:
                raise TypeError("effects must contain ConversationEffectState values")
            detached = ConversationEffectState(
                effect.invocation,
                effect.effect_ordinal,
                effect.kind,
                effect.request,
                effect.status,
                effect.observation_ordinals,
                effect.failure,
            )
            identity = (detached.invocation, detached.effect_ordinal)
            if identity in effect_ids:
                raise ValueError("conversation effect identities must be unique")
            effect_ids.add(identity)
            detached_effects.append(detached)
        object.__setattr__(self, "effects", tuple(detached_effects))
        if type(closed_invocations) is not tuple:
            raise TypeError("closed_invocations must be a tuple")
        completions = []
        closed_ids: set[ConversationInvocation] = set()
        for completion in closed_invocations:
            if type(completion) is not ConversationCompletion:
                raise TypeError(
                    "closed_invocations must contain ConversationCompletion values"
                )
            detached = replace(completion)
            if detached.invocation in closed_ids:
                raise ValueError("closed invocation identities must be unique")
            closed_ids.add(detached.invocation)
            completions.append(detached)
        object.__setattr__(self, "closed_invocations", tuple(completions))
        if type(applied) is not tuple:
            raise TypeError("applied must be a tuple")
        entries = []
        operation_ids: set[str] = set()
        for entry in applied:
            if type(entry) is not ConversationApplied:
                raise TypeError("applied must contain ConversationApplied values")
            detached = replace(entry)
            if detached.operation_id in operation_ids:
                raise ValueError("applied operation IDs must be unique")
            operation_ids.add(detached.operation_id)
            entries.append(detached)
        object.__setattr__(self, "applied", tuple(entries))
        if len(applied) != revision:
            raise ValueError(
                "conversation revision requires one applied entry per checkpoint"
            )
        if revision == 0 and (items or effects or closed_invocations):
            raise ValueError("revision-zero conversation must have empty history")

    @property
    def compatibility(self) -> JsonObject:
        """Return a fresh JSON copy of the retained compatibility descriptor."""
        return _thaw_json_object(self._compatibility)


def _copy_conversation_view(value: object) -> ConversationView:
    """Reconstruct one exact view without retaining any nested input carrier."""
    if type(value) is not ConversationView:
        raise TypeError("baseline must be a ConversationView")
    return ConversationView(
        value.namespace,
        value.generation,
        value.revision,
        value.compatibility,
        value.items,
        value.effects,
        value.closed_invocations,
        value.applied,
    )


def _copy_conversation_stamp(
    value: object, *, admission: bool = False
) -> ConversationStamp:
    """Detach an exact stamp and require its operation's ordinal domain."""
    if type(value) is not ConversationStamp:
        raise TypeError("stamp must be a ConversationStamp")
    stamp = replace(value)
    if (stamp.ordinal == 0) != admission:
        raise ValueError("only conversation admission has ordinal zero")
    return stamp


@dataclass(frozen=True, slots=True)
class ConversationAdmitted:
    """Admission checkpoint with the complete predecessor conversation view."""

    stamp: ConversationStamp
    baseline: ConversationView

    def __post_init__(self) -> None:
        """Detach the baseline and require exact stamped predecessor identity."""
        stamp = _copy_conversation_stamp(self.stamp, admission=True)
        baseline = _copy_conversation_view(self.baseline)
        if (
            stamp.namespace != baseline.namespace
            or stamp.generation != baseline.generation
            or stamp.expected_revision != baseline.revision
        ):
            raise ValueError("admission stamp must match its baseline")
        object.__setattr__(self, "stamp", stamp)
        object.__setattr__(self, "baseline", baseline)


@dataclass(frozen=True, slots=True, init=False)
class ConversationEffectRequested:
    """One effect request carrying only newly retained conversation items."""

    stamp: ConversationStamp
    kind: ConversationEffectKind
    _payload: _FrozenJsonObject = field(repr=False)
    items: tuple[ConversationItem, ...]

    def __init__(
        self,
        stamp: ConversationStamp,
        kind: ConversationEffectKind,
        payload: JsonObject,
        items: tuple[ConversationItem, ...],
    ) -> None:
        """Detach an exact positive-ordinal request, JSON payload, and new items.

        Args:
            stamp (`ConversationStamp`): Exact identity with a positive ordinal.
            kind (`ConversationEffectKind`): Model or tool effect.
            payload (`JsonObject`): Detached external-effect request data.
            items (`tuple[ConversationItem, ...]`): Newly retained items only;
                duplicates within this tuple are rejected.

        Raises:
            TypeError: An argument has the wrong exact carrier or JSON type.
            ValueError: Identity, ordinal, JSON, or item uniqueness is invalid.
        """
        object.__setattr__(self, "stamp", _copy_conversation_stamp(stamp))
        object.__setattr__(self, "kind", _require_effect_kind(kind))
        object.__setattr__(self, "_payload", _freeze_json_object(payload, "payload"))
        object.__setattr__(self, "items", _copy_conversation_items(items))

    @property
    def payload(self) -> JsonObject:
        """Return a fresh JSON copy of the requested external effect."""
        return _thaw_json_object(self._payload)


@dataclass(frozen=True, slots=True)
class ConversationEffectObserved:
    """An actual observation of a prior requested effect within one invocation."""

    stamp: ConversationStamp
    effect_ordinal: int
    status: ConversationEffectStatus
    items: tuple[ConversationItem, ...]
    failure: MemoryOperationFailure | None

    def __post_init__(self) -> None:
        """Detach observation evidence without inventing a terminal disposition."""
        stamp = _copy_conversation_stamp(self.stamp)
        _require_counter(self.effect_ordinal, "effect_ordinal", positive=True)
        if self.effect_ordinal >= stamp.ordinal:
            raise ValueError("observed effect ordinal must precede the observation")
        object.__setattr__(self, "stamp", stamp)
        object.__setattr__(self, "items", _copy_conversation_items(self.items))
        object.__setattr__(
            self, "failure", _copy_effect_failure(self.status, self.failure)
        )


@dataclass(frozen=True, slots=True)
class ConversationClosed:
    """A positive-ordinal checkpoint closing one admitted invocation."""

    stamp: ConversationStamp
    status: ConversationTurnStatus

    def __post_init__(self) -> None:
        """Detach the exact stamp and require an actual terminal disposition."""
        object.__setattr__(self, "stamp", _copy_conversation_stamp(self.stamp))
        _require_turn_status(self.status)


ConversationCheckpoint: TypeAlias = (
    ConversationAdmitted
    | ConversationEffectRequested
    | ConversationEffectObserved
    | ConversationClosed
)


def _copy_conversation_checkpoint(value: object) -> ConversationCheckpoint:
    """Detach one member of the closed checkpoint sum."""
    if type(value) is ConversationEffectRequested:
        return ConversationEffectRequested(
            value.stamp, value.kind, value.payload, value.items
        )
    if type(value) in (
        ConversationAdmitted,
        ConversationEffectObserved,
        ConversationClosed,
    ):
        return replace(value)
    raise TypeError("checkpoints must contain exact ConversationCheckpoint values")


CheckStopped: TypeAlias = Callable[[], None]


class ConversationTurn(Protocol):
    """A leased invocation that directly retains requests and actual observations.

    Callers probe two optional capabilities with `getattr`; neither is part of
    this protocol. `prepare_history(...)` projects and summarizes retained history
    for an API request. `interrupt()` asks the turn to close as `interrupted`
    when its body later returns normally, as a steerable AGENT does after a host
    interrupt; a turn without it closes as usual. SessionMemory turns have both.
    """

    @property
    def view(self) -> ConversationView:
        """Return the latest persisted detached conversation view."""
        ...

    def request(
        self,
        kind: ConversationEffectKind,
        payload: JsonObject,
        *,
        items: tuple[ConversationItem, ...] = (),
    ) -> int:
        """Persist one request before returning its positive effect ordinal.

        Args:
            kind (`ConversationEffectKind`): The external-effect category.
            payload (`JsonObject`): Opaque source-neutral request data.
            items (`tuple[ConversationItem, ...]`): New retained items; defaults
                to an empty tuple.

        Returns:
            `int`: Positive request ordinal, only after atomic persistence.
        """
        ...

    def observe(
        self,
        effect_ordinal: int,
        status: ConversationEffectStatus,
        *,
        items: tuple[ConversationItem, ...] = (),
        failure: MemoryOperationFailure | None = None,
    ) -> None:
        """Persist only actual observation evidence for a requested effect.

        Args:
            effect_ordinal (`int`): Positive request ordinal from this turn.
            status (`ConversationEffectStatus`): The actual observed disposition.
            items (`tuple[ConversationItem, ...]`): New retained evidence; defaults
                to an empty tuple.
            failure (`MemoryOperationFailure | None`): Known failure, required for
                failed and optional for interrupted; defaults to `None`.
        """
        ...


@runtime_checkable
class TranscriptSource(Protocol):
    """Explicit recall contribution that admits a serialized session transcript."""

    def open_turn(
        self,
        request: ConversationRequest,
        *,
        check_stopped: CheckStopped,
    ) -> AbstractContextManager[ConversationTurn]:
        """Acquire one turn with an injected cooperative-stop check.

        Args:
            request (`ConversationRequest`): Authored consumer, invocation, and
                compatibility descriptor without backend-owned identity.
            check_stopped (`CheckStopped`): Raise when cooperative stop is requested.

        Returns:
            `AbstractContextManager[ConversationTurn]`: The queued turn lease.
        """
        ...


@dataclass(frozen=True, slots=True)
class MemoryContent:
    """One frame's transient notes and optional structured transcript source.

    Args:
        text (`str`, defaults to `""`): Exact UTF-8 notes; empty text contributes
            no visible section.
        transcript (`TranscriptSource | None`, defaults to `None`): Explicit live
            source with callable `open_turn`; retained only for this admission.

    Raises:
        `TypeError`: Text is not an exact string or the source lacks `open_turn`.
        `ValueError`: Text is not valid UTF-8.

    Example:
        Return `MemoryContent("Prefer concise answers.")` from a notes frame, or
        `MemoryContent(transcript=self)` from a frame that owns a session.
    """

    text: str = ""
    transcript: TranscriptSource | None = None

    def __post_init__(self) -> None:
        """Validate visible notes and the explicitly contributed source."""
        _require_string(self.text, "text", nonempty=False)
        if self.transcript is not None and not callable(
            getattr(self.transcript, "open_turn", None)
        ):
            raise TypeError("transcript must supply open_turn or be None")


@dataclass(frozen=True, slots=True, init=False)
class MemoryContextRequest:
    """Detached recall inputs and accepted invocation for one context admission."""

    recall: RecallContext
    invocation: ConversationInvocation | None
    _compatibility: _FrozenJsonObject = field(repr=False)

    def __init__(
        self,
        recall: RecallContext,
        invocation: ConversationInvocation | None,
        compatibility: JsonObject,
    ) -> None:
        """Validate and detach recall inputs, identity, and compatibility.

        Args:
            recall (`RecallContext`): Exact visible prompts, canonical inputs,
                and authored consumer path.
            invocation (`ConversationInvocation | None`): Accepted node-start
                identity; `None` permits text-only context without a transcript.
            compatibility (`JsonObject`): Exact finite JSON descriptor used by
                a selected transcript source. Nested values are detached.

        Raises:
            `TypeError`: A carrier or JSON value has the wrong exact type.
            `ValueError`: An identity, path, string, or JSON value is invalid.

        Example:
            `MemoryContextRequest(recall, None, {})` supports direct evaluation
            of text-only policies. A transcript contribution requires an identity.
        """
        if type(recall) is not RecallContext:
            raise TypeError("recall must be a RecallContext")
        object.__setattr__(self, "recall", RecallContext(
            recall.base_system_prompt, recall.prompt, recall.inputs,
            recall.consumer_static_path,
        ))
        object.__setattr__(self, "invocation", (
            None if invocation is None else _copy_invocation(invocation)
        ))
        object.__setattr__(self, "_compatibility", _freeze_json_object(
            compatibility, "compatibility",
        ))

    @property
    def compatibility(self) -> JsonObject:
        """Return a fresh JSON copy of the opaque compatibility descriptor."""
        return _thaw_json_object(self._compatibility)


@dataclass(frozen=True, slots=True)
class MemoryContext:
    """Ordered visible notes and an optional admitted live transcript.

    Args:
        sections (`tuple[MemorySection, ...]`, defaults to `()`): Exact sections
            in outer-to-local ownership order. Admission validates scope order.
        transcript (`ConversationTurn | None`, defaults to `None`): Live leased
            transcript whose lifetime is owned by the enclosing context manager.

    Raises:
        `TypeError`: Sections are not an exact tuple of exact section carriers.
        `ValueError`: A section's identity or text is invalid.

    Example:
        `MemoryContext()` is the empty context for an invocation without memory.
    """

    sections: tuple[MemorySection, ...] = ()
    transcript: ConversationTurn | None = None

    def __post_init__(self) -> None:
        """Require exact immutable text sections; retain only a transient lease."""
        if type(self.sections) is not tuple:
            raise TypeError("sections must be an exact tuple")
        for section in self.sections:
            if type(section) is not MemorySection:
                raise TypeError("sections must contain MemorySection values")
            MemorySection(section.memory, section.text)
