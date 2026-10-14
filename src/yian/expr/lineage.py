"""Recorded expression composition and source-relative feedback transfer.

Feedback addresses recorded values without depending on nodes, memory, or the
runtime. A list is one positional selection; callers use tuples to group
independent messages. Routes contain immutable mappings from the same expression
evaluation, with source values supplied by identity when feedback is transferred.
Registered function boundaries retain stopped feedback for inspection.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Literal, TypeAlias

import yian.json_codec as json_codec

FeedbackValue: TypeAlias = (
    str | dict[str, "FeedbackValue"] | list["FeedbackValue"] | None
)


@dataclass(frozen=True)
class SourceRef:
    """One accepted execution's value, or a path inside that value.

    Sequence zero addresses run inputs; other sequences identify the accepted
    node-start record. Values themselves remain in the caller's source store.
    """

    run_id: str
    execution_seq: int
    path: tuple[str | int, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", tuple(self.path))


@dataclass(frozen=True)
class ValueRoute:
    """Immutable expression composition recorded alongside its forward value.

    Source leaves retain identity only. Literal and opaque results use JSON text
    to isolate their snapshots from later mutation without mutable route fields.
    Non-JSON temporaries retain an unavailable-result type marker, without
    calling their display methods or retaining the live object. Mixed containers
    keep available shape in `snapshot`, using these same literal/list/record
    forms around unavailable leaves. Ordinary JSON snapshots stay complete.
    Partial capture walks only exact builtins with bounded nesting; subclasses
    and recursive/deeper temporary leaves stay unavailable.
    """

    kind: Literal[
        "literal",
        "source",
        "selected",
        "list",
        "record",
        "concat",
        "repeat",
        "scalar",
        "interpolation",
        "opaque",
        "project",
    ]
    source: SourceRef | None = None
    value_json: str | None = None
    children: tuple[ValueRoute, ...] = ()
    expression: str = ""
    names: tuple[str, ...] = ()
    sizes: tuple[int, ...] = ()
    count: int = 0
    content_index: int = 0
    spans: tuple[tuple[int, int], ...] = ()
    path: tuple[str | int, ...] = ()
    result_type: str | None = None
    snapshot: ValueRoute | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "children", tuple(self.children))
        object.__setattr__(self, "names", tuple(self.names))
        object.__setattr__(self, "sizes", tuple(self.sizes))
        object.__setattr__(self, "spans", tuple(tuple(span) for span in self.spans))
        object.__setattr__(self, "path", tuple(self.path))

    @classmethod
    def literal(cls, value: Any) -> ValueRoute:
        """Capture a literal snapshot or an unavailable-result type marker.

        Args:
            value (`Any`): A value evaluated without a producer. Non-JSON leaves
                retain their type names; ordinary containing lists and records
                preserve available structure.

        Returns:
            `ValueRoute`: A literal snapshot with no upstream feedback route.
        """
        if _uses_only_json_builtins(value):
            try:
                encoded = json_codec.dumps(value, ensure_ascii=False, allow_nan=False)
                encoded.encode("utf-8")
            except (ValueError, RecursionError):
                pass
            else:
                return cls("literal", value_json=encoded, expression=encoded)
        return _capture_literal(value, set())

    @classmethod
    def record(cls, fields: Iterable[tuple[str, ValueRoute]]) -> ValueRoute:
        """Retain the named child routes of a record assembled by a binder.

        Args:
            fields (`Iterable[tuple[str, ValueRoute]]`): Unique field names and
                their routes in construction order.

        Returns:
            `ValueRoute`: The record constructor and its named child routes.
        """
        entries = tuple(fields)
        return cls(
            "record",
            children=tuple(route for _, route in entries),
            names=tuple(name for name, _ in entries),
            expression="{"
            + ", ".join(f"{name}: {route.expression}" for name, route in entries)
            + "}",
        )

    def project(self, path: tuple[str | int, ...]) -> ValueRoute:
        """Address an existing field or item of this composition's output.

        Args:
            path (`tuple[str | int, ...]`): Valid field names and nonnegative
                list indices, relative to this route. Empty selects the root.

        Returns:
            `ValueRoute`: A projection that retains the parent composition.
        """
        path = tuple(path)
        if not path:
            return self
        suffix = "".join(
            f".{part}" if isinstance(part, str) else f"[{part}]" for part in path
        )
        if self.kind == "source":
            source = self.source
            return ValueRoute(
                "source",
                source=SourceRef(
                    source.run_id, source.execution_seq, (*source.path, *path)
                ),
                expression=self.expression + suffix,
            )
        return ValueRoute(
            "project", children=(self,), path=path, expression=self.expression + suffix
        )


@dataclass(frozen=True)
class ResolvedValue:
    """One read with actual presence and a producer or prior composition route.

    In traced evaluation this also represents the local MAP item scope. A
    present null keeps its source even though forward's missing/null behavior
    still lets default and coalesce fall through.
    """

    value: Any
    present: bool
    source: SourceRef | ValueRoute | None = None


@dataclass(frozen=True)
class TracedValue:
    """The actual evaluated value together with its immutable routing recipe."""

    value: Any
    route: ValueRoute


@dataclass(frozen=True)
class RouteContext:
    """Interpretation data accompanying unchanged feedback text.

    `positions` maps each content item to its output positions. `span` is a
    half-open range of list positions for concat, or characters for interpolation.
    A projection's `operand` is its selected path; other operands are field names
    or input positions. Source contexts reference their result through `sources`
    instead of copying accepted producer values.
    `sources` preserves evaluated source occurrences, including duplicates;
    source values remain in the caller's accepted snapshot store.
    """

    expression: str
    result_json: str | None
    sources: tuple[SourceRef, ...]
    role: str = "operand"
    operand: int | str | tuple[str | int, ...] | None = None
    positions: tuple[tuple[int, ...], ...] = ()
    span: tuple[int, int] | None = None
    result_type: str | None = None
    snapshot: ValueRoute | None = None

    @property
    def result(self) -> Any:
        """Return an isolated result, or None for a source or unavailable result.

        A recorded JSON null has `result_json == "null"` and no `result_type`;
        an unavailable temporary has no JSON snapshot and retains its type name.
        Partial container structure remains in `snapshot`. Source contexts keep
        the exact accepted value reference in `sources` instead of a JSON copy.
        """
        return json_codec.loads(self.result_json) if self.result_json is not None else None


@dataclass(frozen=True)
class SourceFeedback:
    """One independent feedback value lifted to an execution's root output."""

    source: SourceRef
    feedback: FeedbackValue
    context: tuple[RouteContext, ...] = ()


@dataclass(frozen=True)
class StoppedFeedback:
    """Feedback retained at an opaque expression instead of sent to its inputs."""

    feedback: FeedbackValue
    context: tuple[RouteContext, ...]


@dataclass(frozen=True)
class RoutedFeedback:
    """Ordered source contributions and opaque stops from one transfer."""

    contributions: tuple[SourceFeedback, ...] = ()
    stopped: tuple[StoppedFeedback, ...] = ()


def _at_path(value: Any, path: tuple[str | int, ...]) -> Any:
    for step in path:
        value = value[step]
    return value


def _lift_feedback(
    feedback: FeedbackValue, value: Any, path: tuple[str | int, ...]
) -> FeedbackValue:
    if not path:
        return feedback
    head, *rest = path
    lifted = _lift_feedback(feedback, value[head], tuple(rest))
    if isinstance(head, str):
        return {head: lifted}
    result: list[FeedbackValue] = [None] * len(value)
    result[head] = lifted
    return result


_UNAVAILABLE_RESULT = object()
# Diagnostic retention must remain finite even when a later operation discards
# an arbitrarily deep temporary value.
_MAX_SNAPSHOT_DEPTH = 32


def _uses_only_json_builtins(value: Any) -> bool:
    """Check encoder safety without recursion or invoking subclass hooks."""
    pending = [value]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        value_type = type(current)
        if any(value_type is kind for kind in (type(None), bool, int, float, str)):
            continue
        if (
            value_type is not list
            and value_type is not tuple
            and value_type is not dict
        ):
            return False
        if id(current) in seen:
            continue
        seen.add(id(current))
        if value_type is dict:
            if any(type(name) is not str for name in current):
                return False
            pending.extend(current.values())
        else:
            pending.extend(current)
    return True


def _capture_literal(value: Any, active: set[int], depth: int = 0) -> ValueRoute:
    """Capture bounded exact-builtin structure without calling subclass hooks."""
    value_type = type(value)
    result_type = f"{value_type.__module__}.{value_type.__qualname__}"
    unavailable = ValueRoute(
        "literal", result_type=result_type, expression=f"<{result_type}>"
    )
    if value_type in (type(None), bool, int, float, str):
        try:
            encoded = json_codec.dumps(value, ensure_ascii=False, allow_nan=False)
            encoded.encode("utf-8")
        except (ValueError, RecursionError):
            return unavailable
        return ValueRoute("literal", value_json=encoded, expression=encoded)
    if value_type not in (list, tuple, dict):
        return unavailable
    if depth >= _MAX_SNAPSHOT_DEPTH or id(value) in active:
        return unavailable
    if value_type is dict:
        for name in value:
            if type(name) is not str:
                return unavailable
            try:
                name.encode("utf-8")
            except UnicodeEncodeError:
                return unavailable
    active.add(id(value))
    try:
        if value_type is dict:
            snapshot = ValueRoute.record(
                (name, _capture_literal(item, active, depth + 1))
                for name, item in value.items()
            )
        else:
            snapshot = ValueRoute(
                "list",
                children=tuple(
                    _capture_literal(item, active, depth + 1) for item in value
                ),
            )
    finally:
        active.remove(id(value))
    if any(child.value_json is None for child in snapshot.children):
        return ValueRoute(
            "literal",
            result_type=result_type,
            expression=f"<{result_type}>",
            snapshot=snapshot,
        )
    if value_type is dict:
        encoded = (
            "{"
            + ", ".join(
                json.dumps(name, ensure_ascii=False) + ": " + child.value_json
                for name, child in zip(snapshot.names, snapshot.children)
            )
            + "}"
        )
    else:
        encoded = "[" + ", ".join(child.value_json for child in snapshot.children) + "]"
    return ValueRoute("literal", value_json=encoded, expression=encoded)


def _recorded_at_path(value: Any, path: tuple[str | int, ...]) -> Any:
    """Select available snapshot structure without disguising ordinary path faults."""
    for step in path:
        if value is _UNAVAILABLE_RESULT:
            return _UNAVAILABLE_RESULT
        value = value[step]
    return value


def _route_value(route: ValueRoute, lookup: Callable[[SourceRef], Any]) -> Any:
    if route.kind == "source":
        source = route.source
        return _at_path(
            lookup(SourceRef(source.run_id, source.execution_seq)), source.path
        )
    if route.kind in ("literal", "scalar", "interpolation", "opaque"):
        if route.snapshot is not None:
            return _route_value(route.snapshot, lookup)
        return (
            json_codec.loads(route.value_json)
            if route.value_json is not None
            else _UNAVAILABLE_RESULT
        )
    if route.kind == "selected":
        return _route_value(route.children[0], lookup)
    if route.kind == "project":
        return _recorded_at_path(_route_value(route.children[0], lookup), route.path)
    if route.kind == "list":
        return [_route_value(child, lookup) for child in route.children]
    if route.kind == "record":
        return {
            name: _route_value(child, lookup)
            for name, child in zip(route.names, route.children)
        }
    if route.kind == "concat":
        values = []
        unavailable = False
        # Inspect known siblings too: unavailable snapshots must not mask real faults.
        for child in route.children:
            value = _route_value(child, lookup)
            if value is _UNAVAILABLE_RESULT:
                unavailable = True
            else:
                values.extend(value)
        return _UNAVAILABLE_RESULT if unavailable else values
    if route.kind == "repeat":
        content = _route_value(route.children[route.content_index], lookup)
        if content is _UNAVAILABLE_RESULT and route.count > 0:
            return _UNAVAILABLE_RESULT
        return [item for _ in range(max(0, route.count)) for item in content]
    raise ValueError(f"unknown route kind {route.kind!r}")


def _source_refs(route: ValueRoute) -> tuple[SourceRef, ...]:
    if route.kind == "source":
        return (route.source,)
    return tuple(source for child in route.children for source in _source_refs(child))


def route_feedback(
    route: ValueRoute,
    feedback: FeedbackValue,
    lookup: Callable[[SourceRef], Any],
    *,
    recorded_value: Any = _UNAVAILABLE_RESULT,
) -> RoutedFeedback:
    """Transfer feedback using recorded composition and accepted source values.

    Args:
        route (`ValueRoute`): The route retained during forward evaluation.
        feedback (`FeedbackValue`): One message selecting this route's result.
        lookup (`Callable[[SourceRef], Any]`): Returns an accepted root value for
            an identity with an empty path. It must use the original snapshot.
        recorded_value (`Any`, *optional*): The accepted canonical result, when
            available. It supplies complete feedback shape when a supported
            forward value had an unavailable temporary snapshot. Omission uses
            route snapshots. This value supplies no producer mappings and is
            propagated only through recorded structural selections.

    Returns:
        `RoutedFeedback`: Ordered, independent source-shaped contributions and
            stopped feedback with expression/result/source interpretation data.

    Raises:
        `TypeError`: Raised for unsupported feedback data.
        `ValueError`: Raised for feedback that does not match the result shape.
    """
    if recorded_value is not _UNAVAILABLE_RESULT:
        validate_feedback(feedback, recorded_value)
    contributions: list[SourceFeedback] = []
    stopped: list[StoppedFeedback] = []

    def visit(
        current: ValueRoute,
        message: FeedbackValue,
        context: tuple[RouteContext, ...] = (),
        known_value: Any = _UNAVAILABLE_RESULT,
        selection: tuple[str | int, ...] = (),
    ) -> None:
        if message is None:
            return
        value = _route_value(current, lookup)
        selected_value = _recorded_at_path(value, selection)
        validate_feedback(
            message, selected_value if known_value is _UNAVAILABLE_RESULT else known_value,
        )
        if not has_feedback(message):
            return
        if current.kind == "source":
            source = current.source
            root = SourceRef(source.run_id, source.execution_seq)
            original = lookup(root)
            path = (*source.path, *selection)
            validate_feedback(message, _at_path(original, path))
            lifted = _lift_feedback(message, original, path)
            validate_feedback(lifted, original)
            selected_source = SourceRef(source.run_id, source.execution_seq, path)
            detail = RouteContext(
                current.expression,
                None,
                (selected_source,),
                "projection" if path else "source",
                path or None,
            )
            contributions.append(SourceFeedback(root, lifted, (*context, detail)))
            return
        elif current.kind == "selected":
            visit(current.children[0], message, context, known_value, selection)
            return
        elif current.kind == "literal":
            return

        snapshot = (
            current
            if current.kind in ("scalar", "interpolation", "opaque")
            else ValueRoute.literal(value)
        )
        sources = _source_refs(current)

        def with_context(role, operand=None, positions=(), span=None):
            return (
                *context,
                RouteContext(
                    current.expression,
                    snapshot.value_json,
                    sources,
                    role,
                    operand,
                    positions,
                    span,
                    snapshot.result_type,
                    snapshot.snapshot,
                ),
            )

        if current.kind == "project":
            visit(
                current.children[0], message, with_context("projection", current.path),
                known_value, (*current.path, *selection),
            )
        elif current.kind in ("list", "record"):
            for index, child in enumerate(current.children):
                operand = index if current.kind == "list" else current.names[index]
                if selection and operand != selection[0]:
                    continue
                if selection or isinstance(message, str):
                    selected = message
                elif current.kind == "list":
                    selected = message[index]
                else:
                    selected = message.get(operand)
                child_value = (
                    known_value if selection or known_value is _UNAVAILABLE_RESULT
                    else known_value[operand]
                )
                positions = ((index,),) if current.kind == "list" else ()
                visit(
                    child,
                    selected,
                    with_context(
                        "item" if current.kind == "list" else "field",
                        operand,
                        positions,
                    ),
                    child_value, selection[1:],
                )
        elif current.kind == "concat":
            if selection and (type(selection[0]) is not int
                              or not 0 <= selection[0] < sum(current.sizes)):
                raise IndexError("concat projection index is outside its recorded sizes")
            offset = 0
            for index, (child, size) in enumerate(zip(current.children, current.sizes)):
                if selection and not offset <= selection[0] < offset + size:
                    offset += size
                    continue
                child_value = (
                    known_value if selection or known_value is _UNAVAILABLE_RESULT
                    else known_value[offset : offset + size]
                )
                visit(
                    child,
                    (
                        message
                        if selection or isinstance(message, str)
                        else message[offset : offset + size]
                    ),
                    with_context(
                        "concat",
                        index,
                        tuple((offset + item,) for item in range(size)),
                        (offset, offset + size),
                    ),
                    child_value,
                    (selection[0] - offset, *selection[1:]) if selection else (),
                )
                offset += size
        elif current.kind == "repeat":
            content_index = current.content_index
            content = current.children[content_index]
            size = current.sizes[0]
            copies = max(0, current.count)
            if selection and (type(selection[0]) is not int
                              or not 0 <= selection[0] < copies * size):
                raise IndexError("repeat projection index is outside its recorded copies")
            if selection:
                offset = (selection[0] // size) * size
                visit(
                    content, message,
                    with_context("content", content_index, tuple((offset + i,) for i in range(size))),
                    known_value, (selection[0] % size, *selection[1:]),
                )
            elif isinstance(message, str):
                positions = tuple(
                    tuple(copy * size + i for copy in range(copies))
                    for i in range(size)
                )
                for index, child in enumerate(current.children):
                    role = "content" if index == content_index else "count"
                    child_value = (
                        known_value[:size]
                        if index == content_index and copies > 0
                        and known_value is not _UNAVAILABLE_RESULT else _UNAVAILABLE_RESULT
                    )
                    visit(child, message, with_context(role, index, positions), child_value)
            else:
                for copy in range(copies):
                    offset = copy * size
                    positions = tuple((offset + i,) for i in range(size))
                    visit(
                        content,
                        message[offset : offset + size],
                        with_context(
                            "content",
                            content_index,
                            positions,
                        ),
                        known_value[offset : offset + size]
                        if known_value is not _UNAVAILABLE_RESULT else _UNAVAILABLE_RESULT,
                    )
        elif current.kind in ("scalar", "interpolation"):
            if selection:
                message = _lift_feedback(message, value, selection)
            for index, child in enumerate(current.children):
                span = current.spans[index] if current.kind == "interpolation" else None
                visit(
                    child,
                    message,
                    with_context(
                        "interpolation" if span is not None else "operand",
                        index,
                        span=span,
                    ),
                )
        elif current.kind == "opaque":
            if selection and _recorded_at_path(value, selection[:-1]) is not _UNAVAILABLE_RESULT:
                message = _lift_feedback(message, value, selection)
            stopped.append(StoppedFeedback(message, with_context("opaque")))

    visit(route, feedback, known_value=recorded_value)
    return RoutedFeedback(tuple(contributions), tuple(stopped))


def validate_feedback(
    feedback: FeedbackValue,
    value: Any,
    *,
    path: tuple[str | int, ...] = (),
) -> None:
    """Require feedback to select the exact shape of a recorded value.

    Args:
        feedback (`FeedbackValue`): A string, absent feedback, a partial field
            selection, or a positional list with the recorded list's length.
        value (`Any`): The recorded JSON-like value addressed by the feedback.
        path (`tuple[str | int, ...]`, *optional*, defaults to `()`):
            Address of this value, used to locate validation failures.

    Raises:
        `TypeError`: Raised for an unsupported feedback value or non-string key.
        `ValueError`: Raised for an unknown field or a mismatched container shape.
    """
    if feedback is None or type(feedback) is str:
        return

    location = "feedback" + "".join(f"[{part!r}]" for part in path)
    if type(feedback) is dict:
        if type(value) is not dict:
            raise ValueError(f"{location} requires a recorded object")
        for name, message in feedback.items():
            if type(name) is not str:
                raise TypeError(f"{location} object keys must be strings")
            if name not in value:
                raise ValueError(f"{location} selects unknown field {name!r}")
            validate_feedback(message, value[name], path=(*path, name))
        return

    if type(feedback) is list:
        if type(value) is not list:
            raise ValueError(f"{location} requires a recorded list")
        if len(feedback) != len(value):
            raise ValueError(
                f"{location} must have exactly {len(value)} items; "
                f"got {len(feedback)}"
            )
        for index, (message, item) in enumerate(zip(feedback, value)):
            validate_feedback(message, item, path=(*path, index))
        return

    raise TypeError(f"{location} must be a string, object, list, or None")


def has_feedback(feedback: FeedbackValue) -> bool:
    """Return whether validated feedback contains any supplied string.

    Args:
        feedback (`FeedbackValue`): One ordinary feedback value. Its container
            shape must already have been validated against the recorded value.

    Returns:
        `bool`: Whether a string occurs, including the empty string.

    Raises:
        `TypeError`: Raised when a visited value has an unsupported type.
    """
    if feedback is None:
        return False
    if type(feedback) is str:
        return True
    if type(feedback) is dict:
        return any(has_feedback(message) for message in feedback.values())
    if type(feedback) is list:
        return any(has_feedback(message) for message in feedback)
    raise TypeError("feedback must be a string, object, list, or None")


__all__ = [
    "FeedbackValue",
    "ResolvedValue",
    "RouteContext",
    "RoutedFeedback",
    "SourceFeedback",
    "SourceRef",
    "StoppedFeedback",
    "TracedValue",
    "ValueRoute",
    "has_feedback",
    "route_feedback",
    "validate_feedback",
]
