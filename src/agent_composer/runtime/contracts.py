"""Immutable forward captures and backward outcomes shared by runtime consumers.

These values depend only on lower node, expression, memory, and diagnostic contracts. They
never import the runtime log, state, trace, driver, or public run boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
from typing import Literal

from agent_composer.diagnostics import Failure, OriginRef
from agent_composer.expr.lineage import (
    FeedbackValue, SourceRef, ValueRoute, validate_feedback,
)
from agent_composer.memory.contracts import (
    LearningResult,
    MemoryFrameRef,
    MemoryMutation,
    MemoryOperationFailure,
    _validate_mutations,
    JsonObject,
    JsonValue,
)
from agent_composer.nodes.base import (
    BackwardResult, _copy_backward_value, _validate_declared_names, _validate_input_output_paths,
)
from agent_composer.nodes.contracts import ExecutionScope


def _validate_source(source: SourceRef) -> None:
    """Validate a source identity without normalizing forged carrier fields."""
    if type(source) is not SourceRef:
        raise TypeError("source must be an exact SourceRef")
    if type(source.run_id) is not str or not source.run_id:
        raise ValueError("source run_id must be non-empty")
    if type(source.execution_seq) is not int or source.execution_seq < 0:
        raise ValueError("source execution_seq must be a non-negative integer")
    _validate_route_path(source.path)


def _validate_route_path(path: tuple[str | int, ...]) -> None:
    """Require an immutable field/index path with exact scalar components."""
    if type(path) is not tuple:
        raise TypeError("route path must be a tuple")
    if any(type(part) is not str and (type(part) is not int or part < 0) for part in path):
        raise ValueError("route path components must be strings or non-negative integers")


def _route_object_without_duplicates(pairs: list[tuple[str, JsonValue]]) -> JsonObject:
    """Reject ambiguous object keys inside an embedded route result snapshot."""
    result: JsonObject = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f"duplicate JSON object key: {name!r}")
        result[name] = value
    return result


def _validate_route(route: ValueRoute) -> None:
    """Validate every route field and variant at the durable capture boundary."""
    if type(route) is not ValueRoute:
        raise TypeError("route must be an exact ValueRoute")
    variants = {
        "literal": {"value_json", "result_type", "snapshot"},
        "source": {"source"},
        "selected": {"children"},
        "list": {"children"},
        "record": {"children", "names"},
        "concat": {"children", "sizes"},
        "repeat": {"children", "sizes", "count", "content_index"},
        "scalar": {"children", "value_json", "result_type", "snapshot"},
        "interpolation": {"children", "value_json", "spans"},
        "opaque": {"children", "names", "value_json", "result_type", "snapshot"},
        "project": {"children", "path"},
    }
    if type(route.kind) is not str or route.kind not in variants:
        raise ValueError("unknown route kind")
    if type(route.expression) is not str:
        raise TypeError("route expression must be a string")
    defaults = ValueRoute(route.kind)
    for name in (
        "source", "value_json", "children", "names", "sizes", "count",
        "content_index", "spans", "path", "result_type", "snapshot",
    ):
        if name not in variants[route.kind] and getattr(route, name) != getattr(defaults, name):
            raise ValueError(f"route {route.kind} cannot carry {name}")
    for name in ("children", "names", "sizes", "spans", "path"):
        if type(getattr(route, name)) is not tuple:
            raise TypeError(f"route {name} must be a tuple")
    for name in ("count", "content_index"):
        if type(getattr(route, name)) is not int:
            raise TypeError(f"route {name} must be an integer")
    _validate_route_path(route.path)
    for child in route.children:
        _validate_route(child)
    if any(type(name) is not str for name in route.names):
        raise ValueError("route names must be strings")
    if route.kind == "record" and len(set(route.names)) != len(route.names):
        raise ValueError("record route names must be unique")
    if any(type(size) is not int or size < 0 for size in route.sizes):
        raise ValueError("route sizes must be non-negative integers")
    if route.kind == "source":
        _validate_source(route.source)
    if route.kind in ("selected", "project") and len(route.children) != 1:
        raise ValueError("selected and project routes require one child")
    if route.kind == "project" and not route.path:
        raise ValueError("project route requires a path")
    if route.kind in ("record", "opaque") and len(route.names) != len(route.children):
        raise ValueError(f"{route.kind} route names must match its children")
    if route.kind == "concat" and (len(route.children) != 2 or len(route.sizes) != 2):
        raise ValueError("concat route requires two children and sizes")
    if route.kind == "repeat" and (
        len(route.children) != 2 or len(route.sizes) != 1 or route.content_index not in (0, 1)
    ):
        raise ValueError("repeat route requires two children, one size and a content index")
    if route.kind == "scalar" and not route.children:
        raise ValueError("scalar route requires operands")
    result = None
    if route.value_json is not None:
        if type(route.value_json) is not str:
            raise TypeError("route value_json must be a string or None")
        result = json.loads(route.value_json, object_pairs_hook=_route_object_without_duplicates)
        _copy_backward_value(result, "route result")
        json.dumps(result, ensure_ascii=False, allow_nan=False).encode("utf-8")
    if route.result_type is not None and (type(route.result_type) is not str or not route.result_type):
        raise ValueError("route result_type must be a non-empty string or None")
    if route.kind in ("literal", "scalar", "opaque"):
        if (route.value_json is None) == (route.result_type is None):
            raise ValueError("route requires either a JSON result or an unavailable type")
    if route.snapshot is not None:
        if route.result_type is None or route.value_json is not None:
            raise ValueError("partial snapshot requires an unavailable result")
        _validate_route(route.snapshot)
        pending = [route.snapshot]
        while pending:
            current = pending.pop()
            if current.kind not in ("literal", "list", "record"):
                raise ValueError("partial snapshot cannot contain source routes")
            pending.extend(current.children)
    if route.kind == "interpolation":
        if type(result) is not str or not route.children or len(route.spans) != len(route.children):
            raise ValueError("interpolation requires a string result and one span per child")
        previous_end = 0
        for span in route.spans:
            if type(span) is not tuple or len(span) != 2 or any(type(i) is not int for i in span):
                raise TypeError("interpolation spans must be integer pairs")
            start, end = span
            if not previous_end <= start <= end <= len(result):
                raise ValueError("interpolation spans must be ordered within the result")
            previous_end = end


@dataclass(frozen=True, slots=True, init=False)
class BackwardCapture:
    """Transient detached inventories, scope, and node material from one invocation.

    This worker-owned snapshot becomes durable only through an accepted
    ForwardCapture. It contains neither bound values nor outcomes.
    """

    input_names: tuple[str, ...]
    variable_names: tuple[str, ...]
    scope: ExecutionScope | None
    _node_data: JsonObject = field(repr=False)

    def __init__(
        self,
        input_names: tuple[str, ...] = (),
        variable_names: tuple[str, ...] = (),
        scope: ExecutionScope | None = None,
        node_data: JsonObject | None = None,
    ) -> None:
        """Validate common declarations and copy optional node-owned invocation data.

        Args:
            input_names (`tuple[str, ...]`, *optional*, defaults to `()`): Distinct
                eligible input names, snapshotted before binding can fail.
            variable_names (`tuple[str, ...]`, *optional*, defaults to `()`):
                Distinct local variable names in their declaration order.
            scope (`ExecutionScope | None`, *optional*, defaults to `None`):
                This invocation's authored path and dynamic positions, if known.
            node_data (`JsonObject | None`, *optional*, defaults to `None`):
                Finite JSON static material and optional actual render data.
                `None` uses `{}`, including on early binding failure.

        Raises:
            `TypeError`: If inventories, scope, or JSON containers are malformed.
            `ValueError`: If names repeat, scope or output paths are invalid,
                or node data contains non-finite numbers.
        """
        _validate_declared_names(input_names, "input_names")
        _validate_declared_names(variable_names, "variable_names")
        if scope is not None:
            if type(scope) is not ExecutionScope:
                raise TypeError("scope must be an ExecutionScope or None")
            ExecutionScope(scope.static_path, scope.flow_path, scope.occurrence_path, scope.boundary)
        data = {} if node_data is None else node_data
        if type(data) is not dict:
            raise TypeError("node_data must be a dictionary")
        _validate_input_output_paths(data, input_names)
        object.__setattr__(self, "input_names", input_names)
        object.__setattr__(self, "variable_names", variable_names)
        object.__setattr__(self, "scope", scope)
        object.__setattr__(self, "_node_data", _copy_backward_value(data, "node_data"))

    @property
    def node_data(self) -> JsonObject:
        """Return detached invocation data without exposing the worker's snapshot."""
        return _copy_backward_value(self._node_data, "node_data")


@dataclass(frozen=True, slots=True, init=False)
class ForwardCapture:
    """Detached canonical inputs and routes from one actual binding operation.

    A missing input record marks unfinished binding; safe partial routes remain
    inspectable. Item captures retain explicit indices and actual second-template
    evaluation flags. Outcomes and visible interactions stay on their accepted
    record. Scope may be absent for synthetic nodes awaiting instantiation.
    """

    _bound_inputs: JsonObject | None = field(repr=False)
    route: ValueRoute
    evaluated_keys: frozenset[str]
    items: tuple[tuple[int, ForwardCapture], ...]
    scope: ExecutionScope | None
    _node_data: JsonObject | None = field(repr=False)
    input_names: tuple[str, ...]
    variable_names: tuple[str, ...]

    def __init__(
        self,
        bound_inputs: JsonObject | None,
        route: ValueRoute,
        evaluated_keys: frozenset[str] = frozenset(),
        items: tuple[tuple[int, ForwardCapture], ...] = (),
        scope: ExecutionScope | None = None,
        node_data: JsonObject | None = None,
        *,
        input_names: tuple[str, ...] = (),
        variable_names: tuple[str, ...] = (),
    ) -> None:
        """Validate immutable routes and detach canonical invocation data.

        Args:
            bound_inputs (`JsonObject | None`): Actual completed input record;
                `None` marks unfinished binding and forbids usable item inputs.
            route (`ValueRoute`): Record-shaped routes for the captured fields.
            evaluated_keys (`frozenset[str]`, *optional*, defaults to empty):
                Captured keys whose template expressions were evaluated here.
            items (`tuple[tuple[int, ForwardCapture], ...]`, *optional*, defaults
                to `()`): Per-item bindings in increasing explicit index order.
            scope (`ExecutionScope | None`, *optional*, defaults to `None`):
                Captured authored location and occurrence, when available.
            node_data (`JsonObject | None`, *optional*, defaults to `None`):
                Detached node-owned static backward material, if captured.
            input_names (`tuple[str, ...]`, *optional*, defaults to `()`):
                Distinct eligible input names retained for this invocation.
            variable_names (`tuple[str, ...]`, *optional*, defaults to `()`):
                Distinct local variable names retained for this invocation.

        Raises:
            `TypeError`: If a carrier, immutable container, or JSON value has an
                invalid type.
            `ValueError`: If routes, inventory names, scope, output paths, or
                input/item presence violate the capture contract.
        """
        _validate_route(route)
        if route.kind != "record":
            raise ValueError("input route must be a record")
        if type(evaluated_keys) is not frozenset or any(type(key) is not str for key in evaluated_keys):
            raise TypeError("evaluated_keys must be a frozenset of strings")
        if not evaluated_keys <= set(route.names):
            raise ValueError("evaluated_keys must name captured routes")
        if bound_inputs is not None and type(bound_inputs) is not dict:
            raise TypeError("bound_inputs must be a JSON object or None")
        if bound_inputs is not None and set(bound_inputs) != set(route.names):
            raise ValueError("bound_inputs keys must match input routes")
        if type(items) is not tuple:
            raise TypeError("items must be a tuple")
        previous_index = -1
        for pair in items:
            if type(pair) is not tuple or len(pair) != 2:
                raise TypeError("item captures must be index/capture pairs")
            index, capture = pair
            if type(index) is not int or index <= previous_index:
                raise ValueError("item indices must be non-negative and increasing")
            if type(capture) is not ForwardCapture:
                raise TypeError("item capture must be a ForwardCapture")
            ForwardCapture(capture.bound_inputs, capture.route, capture.evaluated_keys,
                           capture.items, capture.scope, capture.node_data,
                           input_names=capture.input_names, variable_names=capture.variable_names)
            if bound_inputs is None and capture.bound_inputs is not None:
                raise ValueError("unfinished binding cannot retain usable item inputs")
            previous_index = index
        if scope is not None:
            if type(scope) is not ExecutionScope:
                raise TypeError("scope must be an ExecutionScope or None")
            ExecutionScope(scope.static_path, scope.flow_path, scope.occurrence_path, scope.boundary)
        _validate_declared_names(input_names, "input_names")
        _validate_declared_names(variable_names, "variable_names")
        if node_data is not None:
            if type(node_data) is not dict:
                raise TypeError("node_data must be a dictionary or None")
            _validate_input_output_paths(node_data, input_names)
        object.__setattr__(self, "_bound_inputs", _copy_backward_value(bound_inputs, "bound_inputs"))
        object.__setattr__(self, "route", route)
        object.__setattr__(self, "evaluated_keys", evaluated_keys)
        object.__setattr__(self, "items", items)
        object.__setattr__(self, "scope", scope)
        object.__setattr__(self, "_node_data", _copy_backward_value(node_data, "node_data"))
        object.__setattr__(self, "input_names", input_names)
        object.__setattr__(self, "variable_names", variable_names)

    @property
    def bound_inputs(self) -> JsonObject | None:
        """Return a detached canonical input record, or an unfinished marker."""
        return _copy_backward_value(self._bound_inputs, "bound_inputs")

    @property
    def node_data(self) -> JsonObject | None:
        """Return detached optional static data for the node's backward hook."""
        return _copy_backward_value(self._node_data, "node_data")


@dataclass(frozen=True, slots=True)
class BackwardAttribution:
    """Reference one input message transfer and its sender's incoming positions.

    Sender/input/message identify a returned input batch slot; contribution is
    its ordinal in the existing pure route transfer. Parents index the sender's
    ordered received messages. A root seed has no sender or parents. These
    coordinates retain expression stops and contexts without copying source data.
    """

    sender: SourceRef | None
    input_name: str | None
    message_index: int
    contribution_index: int | None
    parents: tuple[int, ...]

    def __post_init__(self) -> None:
        """Reject malformed root/transfer coordinates and mutable inventories."""
        if type(self.message_index) is not int or self.message_index < 0:
            raise ValueError("message_index must be a non-negative integer")
        if type(self.parents) is not tuple or any(type(i) is not int or i < 0 for i in self.parents):
            raise TypeError("parents must be a tuple of non-negative integer positions")
        if tuple(sorted(set(self.parents))) != self.parents:
            raise ValueError("parents must be distinct ordered positions")
        if self.sender is None:
            if (self.input_name is not None or self.message_index != 0
                    or self.contribution_index is not None or self.parents):
                raise ValueError("root attribution cannot carry sender coordinates")
        else:
            _validate_source(self.sender)
            if self.sender.execution_seq == 0 or self.sender.path:
                raise ValueError("attribution sender must identify an accepted occurrence")
            if type(self.input_name) is not str or not self.input_name:
                raise ValueError("transfer attribution requires an input name")
            if type(self.contribution_index) is not int or self.contribution_index < 0:
                raise ValueError("transfer attribution requires a non-negative contribution index")
            if not self.parents:
                raise ValueError("transfer attribution requires incoming parent positions")


@dataclass(frozen=True, slots=True, init=False)
class AttributedBackwardResult:
    """Detached feedback and a result for every accepted forward occurrence."""

    source: SourceRef
    _feedback: tuple[FeedbackValue, ...] = field(repr=False)
    attribution: tuple[BackwardAttribution, ...]
    result: BackwardResult
    invoked: bool

    def __init__(
        self, source: SourceRef, feedback: tuple[FeedbackValue, ...],
        attribution: tuple[BackwardAttribution, ...], result: BackwardResult, invoked: bool,
    ) -> None:
        """Validate aligned attribution and copy mutable message containers.

        Args:
            source (`SourceRef`): Accepted forward occurrence identity, never
                source zero.
            feedback (`tuple[FeedbackValue, ...]`): Ordered ordinary messages
                delivered to its callback.
            attribution (`tuple[BackwardAttribution, ...]`): One source-neutral
                transfer reference per message.
            result (`BackwardResult`): Its validated three-channel result, empty
                when unreached.
            invoked (`bool`): Whether the fold invoked this occurrence's callback.

        Raises:
            `TypeError`: For noncanonical carriers or mutable batch containers.
            `ValueError`: For invalid coordinates, misalignment, or nonempty
                material on an unreached occurrence.
        """
        _validate_source(source)
        if source.execution_seq == 0 or source.path:
            raise ValueError("backward result source must identify an accepted occurrence")
        if type(feedback) is not tuple or type(attribution) is not tuple:
            raise TypeError("feedback and attribution must be tuples")
        if len(feedback) != len(attribution):
            raise ValueError("feedback and attribution must align")
        for message in feedback:
            validate_feedback(message, message)
        for item in attribution:
            if type(item) is not BackwardAttribution:
                raise TypeError("attribution requires exact BackwardAttribution values")
            BackwardAttribution.__post_init__(item)
        if type(result) is not BackwardResult or type(invoked) is not bool:
            raise TypeError("result and invoked must be a BackwardResult and bool")
        checked = BackwardResult(result.input_feedback, result.self_feedback, result.variable_feedback)
        if not invoked and (feedback or checked != BackwardResult()):
            raise ValueError("unreached occurrences cannot carry backward material")
        if invoked and not feedback:
            raise ValueError("invoked occurrences require received feedback")
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "_feedback", tuple(_copy_backward_value(m, "feedback") for m in feedback))
        object.__setattr__(self, "attribution", attribution)
        object.__setattr__(self, "result", checked)
        object.__setattr__(self, "invoked", invoked)

    @property
    def feedback(self) -> tuple[FeedbackValue, ...]:
        """Return an isolated copy of the ordered ordinary message batch."""
        return tuple(_copy_backward_value(message, "feedback") for message in self._feedback)


@dataclass(frozen=True, slots=True)
class FrameProposal:
    """Detached ordered mutations proposed for one recorded owning frame.

    Attributes:
        frame_ref (`MemoryFrameRef`): Exact ownership identity; policy membership
            comes from the recorded layout.
        mutations (`tuple[MemoryMutation, ...]`): Frame-local mutations with unique
            IDs, retained in proposal order. An empty batch is valid.
    """

    frame_ref: MemoryFrameRef
    mutations: tuple[MemoryMutation, ...]

    def __post_init__(self) -> None:
        """Validate ownership and retain independent canonical mutation payloads."""
        if type(self.frame_ref) is not MemoryFrameRef:
            raise TypeError("frame_ref must be a MemoryFrameRef")
        object.__setattr__(self, "frame_ref", replace(self.frame_ref))
        object.__setattr__(
            self, "mutations", _validate_mutations(self.mutations, frame_ref=self.frame_ref),
        )


@dataclass(frozen=True, slots=True)
class BackwardFailure:
    """An attributed failure from one backward or memory operation.

    Attributes:
        stage (`str`): Non-empty operation stage, such as backward, learn, or commit.
        code (`str`): Stable, non-empty semantic failure code.
        message (`str`): Human-readable detail; it may be empty.
        scope (`ExecutionScope | None`): Responsible authored owner and child position.
        run_id (`str | None`): Recorded run identity, when available.
        node_id (`str | None`): Executing node's display ID, when applicable.
        node_started_seq (`int | None`): Positive accepted start sequence identifying
            a node occurrence together with its run ID.
        policy_name (`str | None`): Responsible memory policy, when applicable.
        frame_ref (`MemoryFrameRef | None`): Responsible owned memory frame, if any.
        cause (`Failure | MemoryOperationFailure | None`): Underlying source-neutral
            diagnostic, including a backend retryability hint when one exists.
            Failure origins retain only their canonical keys.
        cause_type (`str | None`): Python exception class name retained for debugging.
        traceback (`str | None`): Opaque debugging traceback text.
    """

    stage: str
    code: str
    message: str
    scope: ExecutionScope | None = None
    run_id: str | None = None
    node_id: str | None = None
    node_started_seq: int | None = None
    policy_name: str | None = None
    frame_ref: MemoryFrameRef | None = None
    cause: Failure | MemoryOperationFailure | None = None
    cause_type: str | None = None
    traceback: str | None = None

    def __post_init__(self) -> None:
        """Require stable attribution and data-only diagnostic causes."""
        for name, value in (("stage", self.stage), ("code", self.code)):
            if type(value) is not str:
                raise TypeError(f"backward failure {name} must be a string")
            if not value:
                raise ValueError(f"backward failure {name} must be non-empty")
        if type(self.message) is not str:
            raise TypeError("backward failure message must be a string")
        for name, value in (
            ("run_id", self.run_id),
            ("node_id", self.node_id),
            ("policy_name", self.policy_name),
            ("cause_type", self.cause_type),
            ("traceback", self.traceback),
        ):
            if value is not None and type(value) is not str:
                raise TypeError(f"backward failure {name} must be a string or None")
        if self.node_started_seq is not None:
            if type(self.node_started_seq) is not int:
                raise TypeError("node_started_seq must be an integer or None")
            if self.node_started_seq <= 0:
                raise ValueError("node_started_seq must be positive")
        if self.scope is not None:
            if type(self.scope) is not ExecutionScope:
                raise TypeError("scope must be an ExecutionScope or None")
            ExecutionScope.__post_init__(self.scope)
        if self.frame_ref is not None:
            if type(self.frame_ref) is not MemoryFrameRef:
                raise TypeError("frame_ref must be a MemoryFrameRef or None")
            MemoryFrameRef.__post_init__(self.frame_ref)
        if self.cause is not None:
            if type(self.cause) is Failure:
                Failure.__post_init__(self.cause)
                object.__setattr__(
                    self,
                    "cause",
                    replace(
                        self.cause,
                        trace=tuple(OriginRef(origin.key) for origin in self.cause.trace),
                    ),
                )
            elif type(self.cause) is MemoryOperationFailure:
                MemoryOperationFailure.__post_init__(self.cause)
            else:
                raise TypeError("cause must be a Failure, MemoryOperationFailure, or None")


@dataclass(frozen=True, slots=True)
class BackwardOutcome:
    """A backward attempt's disposition, separate from its forward run result.

    Attributes:
        status (`Literal["succeeded", "failed"]`): Status of this backward attempt.
        feedback (`BackwardResult | None`): Completed flow-level feedback, when
            available, including when a later learning operation fails.
        failure (`BackwardFailure | None`): The failed attempt's diagnostic;
            absent on success.
        learning (`LearningResult | None`): Memory learning disposition, or
            `None` when no memory policies participate.
    """

    status: Literal["succeeded", "failed"]
    feedback: BackwardResult | None = None
    failure: BackwardFailure | None = None
    learning: LearningResult | None = None

    def __post_init__(self) -> None:
        """Validate the disposition and detach the completed feedback result."""
        if type(self.status) is not str or self.status not in ("succeeded", "failed"):
            raise ValueError("backward status must be 'succeeded' or 'failed'")
        if self.feedback is not None:
            if type(self.feedback) is not BackwardResult:
                raise TypeError("feedback must be a BackwardResult or None")
            object.__setattr__(
                self,
                "feedback",
                BackwardResult(
                    self.feedback.input_feedback,
                    self.feedback.self_feedback,
                    self.feedback.variable_feedback,
                ),
            )
        if self.failure is not None:
            if type(self.failure) is not BackwardFailure:
                raise TypeError("failure must be a BackwardFailure or None")
            object.__setattr__(self, "failure", replace(self.failure))
        if self.status == "succeeded" and self.failure is not None:
            raise ValueError("succeeded backward cannot carry a failure")
        if self.status == "failed" and self.failure is None:
            raise ValueError("failed backward requires a failure")
        if self.learning is not None:
            if type(self.learning) is not LearningResult:
                raise TypeError("learning must be a LearningResult or None")
            object.__setattr__(self, "learning", replace(self.learning))
            if self.status == "succeeded" and self.learning.status != "succeeded":
                raise ValueError("succeeded backward requires succeeded learning")


__all__ = [
    "AttributedBackwardResult", "BackwardAttribution", "BackwardCapture",
    "BackwardFailure", "BackwardOutcome", "ForwardCapture", "FrameProposal",
]
