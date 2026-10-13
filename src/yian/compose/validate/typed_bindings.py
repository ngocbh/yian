"""Graph-stage: reject typed bindings whose value can never fit the declared type.

A typed binding (`name: Type = value`, on a node input or a flow output) states the
type its value must have; the binder enforces it at runtime. A literal right side is
already checked against its type when the binding is split, where the authored text
is at hand. This load pass rejects the remaining bindings that could never pass the
runtime check:

- a string source with no `${...}` span (an escaped `$${x}`) is literal text, so it
  is checked exactly with the same write-boundary check the binder uses;
- a reference or expression is inferred from its raw source over the producer and
  flow-input contracts, and rejected only when that type is disjoint from the
  declared one. A narrower inferred type (an Optional source, a str into a Literal)
  loads and is left to the runtime check, as is an unproven source;
- a lone `${item}` or `${item.field}` read on a node that binds per item (MAP) is
  inferred from the element type of its `over` source, when that is a proven list.

Node inputs are checked as soon as producer contracts exist, before the CALL, MAP,
and LOOP checks that compare the declared type with a child's parameters; flow
outputs are checked once their contracts are attached. A node input error locates at
the binding's registered value origin, else its `input:` entry range, else the node
range; a flow output error at its registered origin, else the `output:` range.

Imports flow down only: `compile.validation` (the type relation), `expr` (template
scanning), `typesys`, and `compose.helpers` (output-contract inference and load
diagnostics). Only the loader (and the compose package facade) import this back.
"""

from __future__ import annotations

import json
from math import isfinite
from typing import Any, Callable, Optional

from yian.compile.model import FlowOutput
from yian.compile.validation import type_relation
from yian.compose.helpers.errors import LoadDiagnostics, LoadError
from yian.compose.helpers.output_contracts import (
    direct_reference_path,
    project_output_contract,
    resolve_output_binding_contract,
)
from yian.compose.nodes.base import BuildResult
from yian.compose.validate.narrowing import Narrowing
from yian.diagnostics import DiagnosticFrame, OriginRef, SourceRange
from yian.expr import ExpressionError, eval_binding, scan_template
from yian.expr.template import Span
from yian.nodes.contracts import Binding, OutputContract, ValueContract, binding_source
from yian.typesys import (
    Type,
    TypeCheckError,
    ValueKind,
    build_value,
    build_value_as,
    canonical_type_label,
)

_NOT_LITERAL = object()

ResolveOrigin = Callable[[OriginRef], DiagnosticFrame | None]


def check_typed_inputs(
    flow_wiring: dict[str, dict[str, Any]],
    producers: dict[str, OutputContract],
    flow_inputs: dict[str, ValueContract],
    build_results: dict[str, BuildResult],
    *,
    narrowing: Narrowing | None = None,
    resolve_origin: ResolveOrigin | None = None,
    diagnostics: LoadDiagnostics | None = None,
) -> None:
    """Reject every typed node input that can never fit its declared type.

    Args:
        flow_wiring (`dict[str, dict[str, Any]]`): Final binding maps keyed by node id
            and parameter name; typed entries are `Binding`s with a contract.
        producers (`dict[str, OutputContract]`): Exact or partial node output
            contracts keyed by producer id.
        flow_inputs (`dict[str, ValueContract]`): Exact flow input contracts keyed by
            input name.
        build_results (`dict[str, BuildResult]`): Built nodes with their transient
            value ranges, the fallback location for a node input.
        narrowing (`Narrowing | None`, *optional*, defaults to `None`):
            The union narrowing each param reads against, from the CASE branches
            that route its node; `None` reads every param against the unchanged maps.
        resolve_origin (`ResolveOrigin | None`, *optional*, defaults to `None`):
            Resolves a binding's registered origin to its source frame, the
            preferred location.
        diagnostics (`LoadDiagnostics | None`, *optional*, defaults to `None`):
            Factory for located load errors.

    Raises:
        `LoadError`: A literal-text source does not fit its declared type, or a
            source's inferred type is disjoint from it.
    """
    diagnostics = diagnostics or LoadDiagnostics()
    for node_id, wiring in flow_wiring.items():
        result = build_results.get(node_id)
        node_producers, node_inputs = (
            narrowing.for_node(node_id) if narrowing is not None else (producers, flow_inputs)
        )
        item = _item_contract(result, wiring, node_producers, node_inputs)
        for name, binding in wiring.items():
            param_producers, param_inputs = (
                narrowing.for_param(node_id, name)
                if narrowing is not None
                else (node_producers, node_inputs)
            )
            message = _binding_error(binding, param_producers, param_inputs, item=item)
            if message is None:
                continue
            fallback = None
            if result is not None:
                fallback = result.field_ranges.get(("input", name)) or result.node_range
            raise _located_error(
                f"node {node_id!r} input {name!r}: {message}",
                binding,
                fallback,
                resolve_origin,
                diagnostics,
            )


def check_typed_outputs(
    flow_outputs: list[FlowOutput],
    producers: dict[str, OutputContract],
    flow_inputs: dict[str, ValueContract],
    *,
    value_mode: bool = False,
    outputs_range: Optional[SourceRange] = None,
    resolve_origin: ResolveOrigin | None = None,
    diagnostics: LoadDiagnostics | None = None,
) -> None:
    """Reject every typed flow output that can never fit its declared type.

    Args:
        flow_outputs (`list[FlowOutput]`): The flow's outputs, typed ones carrying
            their declared contract on the binding.
        producers (`dict[str, OutputContract]`): Exact or partial node output
            contracts keyed by producer id.
        flow_inputs (`dict[str, ValueContract]`): Exact flow input contracts keyed by
            input name.
        value_mode (`bool`, *optional*, defaults to `False`): Whether the flow has a
            single `output:` value, whose internal name the author never wrote, so
            errors name it `output` instead.
        outputs_range (`SourceRange | None`, *optional*, defaults to `None`): The
            authored `output:` range, the fallback location.
        resolve_origin (`ResolveOrigin | None`, *optional*, defaults to `None`):
            Resolves a binding's registered origin to its source frame, the
            preferred location.
        diagnostics (`LoadDiagnostics | None`, *optional*, defaults to `None`):
            Factory for located load errors.

    Raises:
        `LoadError`: A literal-text source does not fit its declared type, or a
            source's inferred type is disjoint from it.
    """
    diagnostics = diagnostics or LoadDiagnostics()
    for output in flow_outputs:
        message = _binding_error(output.binding, producers, flow_inputs)
        if message is not None:
            label = "output" if value_mode else f"output {output.name!r}"
            raise _located_error(
                f"{label}: {message}",
                output.binding,
                outputs_range,
                resolve_origin,
                diagnostics,
            )


def infer_binding_type(
    binding: Any,
    producers: dict[str, OutputContract],
    flow_inputs: dict[str, ValueContract],
) -> Type | None:
    """Return the type one binding is proven to produce, else `None`.

    A typed binding has its declared type; a literal has its exact type (a one-tag
    `Literal[...]` for non-empty text, one-point bounds for a number); a reference or
    expression has the exact type inferred over the producer and flow-input contracts.

    Args:
        binding (`Any`): A wiring binding, raw or a `Binding`.
        producers (`dict[str, OutputContract]`): Node output contracts by producer id.
        flow_inputs (`dict[str, ValueContract]`): Flow input contracts by name.

    Returns:
        `Type | None`: The proven type, or `None` when it is unproven or only partial.
    """
    if isinstance(binding, Binding) and binding.contract is not None:
        return binding.contract.type
    literal = _literal_value(binding_source(binding))
    if literal is not _NOT_LITERAL:
        return _literal_type(literal)
    inferred = resolve_output_binding_contract(binding, producers, flow_inputs)
    return inferred.type if type(inferred) is ValueContract else None


def _literal_type(value: Any) -> Type | None:
    """Return the exact type of a literal value, else `None`.

    Args:
        value (`Any`): The bound literal.

    Returns:
        `Type | None`: The literal's type, or `None` for a value with no type.
    """
    # bool is an int subclass, so it is matched first.
    if type(value) is bool:
        return Type.scalar(ValueKind.BOOLEAN)
    if type(value) is int:
        return Type(kind=ValueKind.INTEGER, bounds=(value, value))
    if type(value) is float:
        if not isfinite(value):
            return None
        return Type(kind=ValueKind.NUMBER, bounds=(value, value))
    if type(value) is str:
        # A tag must be non-empty, so empty text is plain str.
        if not value:
            return Type.scalar(ValueKind.STRING)
        return Type(kind=ValueKind.STRING, tags=frozenset({value}))
    if value is None:
        return Type.scalar(ValueKind.NONE)
    try:
        return Type.scalar(build_value(value).kind)
    except TypeCheckError:
        return None


def _located_error(
    message: str,
    binding: Binding,
    fallback: Optional[SourceRange],
    resolve_origin: ResolveOrigin | None,
    diagnostics: LoadDiagnostics,
) -> LoadError:
    """Build the load error, located at the binding's origin when it resolves.

    Args:
        message (`str`): The error message.
        binding (`Binding`): The rejected binding.
        fallback (`SourceRange | None`): The location used when the origin does not
            resolve.
        resolve_origin (`ResolveOrigin | None`): Resolves a registered origin to its
            source frame.
        diagnostics (`LoadDiagnostics`): Factory for located load errors.

    Returns:
        `LoadError`: The located error.
    """
    frame = (
        resolve_origin(binding.origin)
        if resolve_origin is not None and binding.origin is not None
        else None
    )
    if frame is not None:
        return diagnostics.error(message, frames=(frame,))
    return diagnostics.error(message, range=fallback)


def _item_contract(
    result: BuildResult | None,
    wiring: dict[str, Any],
    producers: dict[str, OutputContract],
    flow_inputs: dict[str, ValueContract],
) -> ValueContract | None:
    """Return the exact element contract `${item}` reads on a per-item node, else `None`.

    Args:
        result (`BuildResult | None`): The built node; only a node with an item
            source key (MAP) has an item scope.
        wiring (`dict[str, Any]`): The node's binding map, holding the item source
            under that key.
        producers (`dict[str, OutputContract]`): Node output contracts by producer id.
        flow_inputs (`dict[str, ValueContract]`): Flow input contracts by name.

    Returns:
        `ValueContract | None`: The element contract when the item source is an
            exact list with a known element type, else `None`.
    """
    key = result.node.item_source_key if result is not None else None
    if key is None or key not in wiring:
        return None
    source = resolve_output_binding_contract(wiring[key], producers, flow_inputs)
    if type(source) is not ValueContract:
        return None
    if not source.type.kind.is_list() or source.type.element is None:
        return None
    return ValueContract(source.type.element)


def _binding_error(
    binding: Any,
    producers: dict[str, OutputContract],
    flow_inputs: dict[str, ValueContract],
    *,
    item: ValueContract | None = None,
) -> str | None:
    """Return why one typed binding can never fit its declared type, else `None`.

    Args:
        binding (`Any`): A wiring or output binding; untyped ones are skipped.
        producers (`dict[str, OutputContract]`): Node output contracts by producer id.
        flow_inputs (`dict[str, ValueContract]`): Flow input contracts by name.
        item (`ValueContract | None`, *optional*, defaults to `None`): The element
            contract a lone `${item...}` reference reads, when the node binds per
            item over a proven list.

    Returns:
        `str | None`: The error message, without the binding's name, or `None` when
            the binding may fit.
    """
    if not isinstance(binding, Binding) or binding.contract is None:
        return None
    declared = binding.contract.type
    label = canonical_type_label(declared)
    literal = _literal_value(binding.source)
    if literal is not _NOT_LITERAL:
        try:
            build_value_as(declared, literal)
        except TypeCheckError as exc:
            message = f"declared {label}, but the value {_shown(literal)} does not fit"
            if exc.path:
                message += f" ({exc})"
            return message
        return None
    # Infer from the raw source: the Binding itself would resolve to its own
    # declared contract and always fit.
    inferred = _item_reference_contract(binding.source, item)
    if inferred is None:
        inferred = resolve_output_binding_contract(
            binding.source, producers, flow_inputs
        )
    if type(inferred) is not ValueContract:
        return None  # unproven or partial: the runtime check covers it
    if type_relation(inferred.type, declared) != "disjoint":
        return None
    return f"declared {label}, but the value is {canonical_type_label(inferred.type)}"


def _item_reference_contract(
    source: Any, item: ValueContract | None
) -> OutputContract | None:
    """Return the contract a lone `${item...}` reference reads, else `None`.

    Args:
        source (`Any`): The binding's raw source.
        item (`ValueContract | None`): The node's element contract, or `None` when
            the node has no proven item scope.

    Returns:
        `OutputContract | None`: The element contract projected through the
            reference's fields, or `None` for any other source or an unknown field
            (reported by reference validation).
    """
    if item is None:
        return None
    path = direct_reference_path(source)
    if path is None:
        return None
    head, *fields = path.split(".")
    if head != "item":
        return None
    return project_output_contract(item, fields)


def _shown(value: Any) -> str:
    """Render a literal as YAML would write it (`true`, `null`, `"text"`).

    Args:
        value (`Any`): The literal to render.

    Returns:
        `str`: JSON text for a JSON-able value, else its `repr`.
    """
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return repr(value)


def _literal_value(source: Any) -> Any:
    """Return the value a literal source binds, or `_NOT_LITERAL` for a template.

    A non-string is bound as is; a string with no `${...}` span binds its text after
    `$$` unescaping, exactly as the binder evaluates it.

    Args:
        source (`Any`): The binding's raw source.

    Returns:
        `Any`: The bound literal value, or `_NOT_LITERAL`.
    """
    if not isinstance(source, str):
        return source
    try:
        segments = scan_template(source)
        if any(isinstance(segment, Span) for segment in segments):
            return _NOT_LITERAL
        return eval_binding(source, lambda path: None)
    except ExpressionError:
        return _NOT_LITERAL  # malformed text is reported by reference validation
