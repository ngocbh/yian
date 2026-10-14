"""Graph-stage: reject a MAP `over:` or WAIT `until:` source whose type can never fit.

A MAP node maps over the list its `over:` source resolves to, and a WAIT node waits
until the date or datetime its `until:` source resolves to; the node rejects any
other value at run time. A wrong literal is already rejected where the node is
parsed, with the authored text at hand. This load pass rejects a reference or
expression whose inferred type proves the run-time check would fail:

- `over:` must infer a list kind (or `Any`), nullability aside. The test is on the
  kind rather than the type relation, so any list loads whatever its element type;
- `until:` must infer a type that overlaps `date` or `datetime`. A `str` overlaps
  both (the node parses text at run time), so a text reference loads.

A literal source is left to the parse stage, and an unproven (`None`) inferred type
loads and is left to the run-time check. A tag on these keys is never consumed, so
the source never carries a declared contract and is always inferred from its text.

It runs after the type tables are filled, so an `isinstance` call in `over:` or
`until:` keeps its more specific error, and before the cycle check, so a wrong-typed
source in a cyclic flow reports the type error first.

Imports flow down only: `compile.validation` (the type relation), `compose.helpers`
(literal detection, tag labels, load diagnostics), `compose.nodes.base` (build results),
`compose.validate.narrowing` and `compose.validate.typed_bindings` (narrowed views and
binding type inference), `nodes`, `typesys`, and `diagnostics`.
"""

from __future__ import annotations

from typing import Any, Optional

from yian.compile.validation import type_relation
from yian.compose.helpers.errors import LoadDiagnostics, LoadError
from yian.compose.helpers.output_contracts import NOT_LITERAL, TypeFn, literal_value
from yian.compose.helpers.tags import named_type_label
from yian.compose.nodes.base import BuildResult
from yian.compose.validate.narrowing import Narrowing
from yian.compose.validate.typed_bindings import ResolveOrigin, infer_binding_type
from yian.diagnostics import SourceRange
from yian.nodes.contracts import Binding, OutputContract, ValueContract, binding_source
from yian.nodes.map import MapNode
from yian.nodes.wait import WaitNode
from yian.typesys import Type, TypeRegistry, ValueKind

_DATE = Type.scalar(ValueKind.DATE)
_DATETIME = Type.scalar(ValueKind.DATETIME)


def check_reserved_sources(
    build_results: dict[str, BuildResult],
    flow_wiring: dict[str, dict[str, Any]],
    producers: dict[str, OutputContract],
    flow_inputs: dict[str, ValueContract],
    *,
    narrowing: Narrowing | None = None,
    type_fn: TypeFn | None = None,
    registry: TypeRegistry | None = None,
    resolve_origin: ResolveOrigin | None = None,
    diagnostics: LoadDiagnostics | None = None,
) -> None:
    """Reject a MAP `over:` or WAIT `until:` source whose inferred type can never fit.

    Args:
        build_results (`dict[str, BuildResult]`): Built nodes with their transient
            value ranges, the fallback location.
        flow_wiring (`dict[str, dict[str, Any]]`): Final binding maps keyed by node id
            and wiring key; a MAP node's `over` and a WAIT node's `until` hold the
            source.
        producers (`dict[str, OutputContract]`): Exact or partial node output
            contracts keyed by producer id.
        flow_inputs (`dict[str, ValueContract]`): Exact flow input contracts keyed by
            input name.
        narrowing (`Narrowing | None`, *optional*, defaults to `None`): The union
            narrowing each node reads against, from the CASE branches that route it;
            `None` reads every source against the unchanged maps.
        type_fn (`TypeFn | None`, *optional*, defaults to `None`): Resolves a
            constructor name to its type for inference.
        registry (`TypeRegistry | None`, *optional*, defaults to `None`): The flow's
            typedefs, so a message names a declared record by its typedef.
        resolve_origin (`ResolveOrigin | None`, *optional*, defaults to `None`):
            Resolves a binding's registered origin to its source frame, the
            preferred location.
        diagnostics (`LoadDiagnostics | None`, *optional*, defaults to `None`):
            Factory for located load errors.

    Raises:
        `LoadError`: A MAP `over:` source infers a type that is not a list, or a WAIT
            `until:` source infers a type disjoint from both `date` and `datetime`.
    """
    diagnostics = diagnostics or LoadDiagnostics()
    for node_id, result in build_results.items():
        node = result.node
        if isinstance(node, MapNode):
            key, kind = "over", "map"
        elif isinstance(node, WaitNode):
            key, kind = "until", "wait"
        else:
            continue
        source = flow_wiring.get(node_id, {}).get(key)
        if source is None or literal_value(binding_source(source)) is not NOT_LITERAL:
            continue  # absent, or a literal the parse stage already checked
        node_producers, node_inputs = (
            narrowing.for_node(node_id) if narrowing is not None else (producers, flow_inputs)
        )
        inferred = infer_binding_type(source, node_producers, node_inputs, type_fn=type_fn)
        if inferred is None:
            continue  # unproven: the run-time check covers it
        label = named_type_label(inferred, registry or {})
        if key == "over":
            if inferred.kind.is_list() or inferred.kind is ValueKind.ANY:
                continue
            message = (
                f"`over:` must be a list, but the value is {label}; "
                "write a reference to a list"
            )
        else:
            if (
                type_relation(inferred, _DATE) != "disjoint"
                or type_relation(inferred, _DATETIME) != "disjoint"
            ):
                continue
            message = (
                f"`until:` must be a date or datetime, but the value is {label}; "
                "write a reference to a date, datetime, or text"
            )
        raise _located_error(
            f"node {node_id!r} (kind={kind}): {message}",
            source,
            result.field_ranges.get(("field", key)) or result.node_range,
            resolve_origin,
            diagnostics,
        )


def _located_error(
    message: str,
    source: Any,
    fallback: Optional[SourceRange],
    resolve_origin: ResolveOrigin | None,
    diagnostics: LoadDiagnostics,
) -> LoadError:
    """Build the load error, located at the source's registered origin when it resolves.

    Args:
        message (`str`): The error message.
        source (`Any`): The rejected source, a `Binding` when it carries an origin.
        fallback (`SourceRange | None`): The location used when no origin resolves.
        resolve_origin (`ResolveOrigin | None`): Resolves a registered origin to its
            source frame.
        diagnostics (`LoadDiagnostics`): Factory for located load errors.

    Returns:
        `LoadError`: The located error.
    """
    origin = source.origin if isinstance(source, Binding) else None
    frame = (
        resolve_origin(origin) if resolve_origin is not None and origin is not None else None
    )
    if frame is not None:
        return diagnostics.error(message, frames=(frame,))
    return diagnostics.error(message, range=fallback)
