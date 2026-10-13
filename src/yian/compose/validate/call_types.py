"""Graph-stage: the cross-flow TYPE check for every `call`/`map` binding.

A loader post-build pass (run after the flow's leaves + child signatures are assembled).
Each binding's SOURCE contract is resolved from its reference over the PARENT producer +
flow-input contracts (a MAP's `${item...}` from the element type of `over:`), then its
`Type` is checked with `types_compatible` against the CHILD input's declared `Type`. A
literal, or the literal fallback of a `:-` escape, is validated by value instead, as the
child's start validates it. Its
range anchor comes from the call node's transient Compose
`BuildResult.field_ranges[("field", "inputs")]` (the coarse `input:` mapping span).

Imports flow down only: `compile.validation` (the type comparator), `expr` (the binding-null
analysis and escape parsing), `compose.helpers` (shared output-contract inference and load
diagnostics), `compose.validate.typed_bindings` (literal and `${item}` helpers), and
`nodes.binding` (the child start's string conversion).
Only the loader (and the compose package facade) import this back.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Optional

from yian.compile.validation import types_compatible
from yian.compose.helpers.errors import LoadDiagnostics
from yian.compose.helpers.output_contracts import (
    direct_reference_parts,
    direct_reference_path,
    project_output_contract,
    reference_contract,
    resolve_output_binding_contract,
)
from yian.compose.nodes.base import BuildResult
from yian.compose.validate.narrowing import Narrowing
from yian.compose.validate.typed_bindings import (
    NOT_LITERAL,
    item_contract,
    literal_value,
    shown_literal,
)
from yian.diagnostics import SourceRange
from yian.expr import (
    ExpressionError,
    binding_co_skips,
    eval_expr,
    expr_refs,
    scan_template,
)
from yian.expr.template import Span
from yian.nodes.binding import coerce_param
from yian.nodes.contracts import (
    Binding,
    OutputContract,
    Parameter,
    PartialRecordContract,
    ValueContract,
    binding_source,
)
from yian.typesys import Type, TypeCheckError, build_value_as, canonical_type_label


def check_call_map_types(
    build_results: dict[str, BuildResult],
    producers: dict[str, OutputContract],
    flow_inputs: dict[str, ValueContract],
    flow_wiring: dict[str, dict[str, Any]],
    *,
    narrowing: Narrowing | None = None,
    diagnostics: LoadDiagnostics | None = None,
) -> None:
    """Cross-flow type check for every `call`/`map` binding (a loader post-build pass).

    Each binding must fit the CHILD input's declared `Type`, read over the PARENT
    producer + flow-input contracts as the node's `narrowing` view sees them when given:
    a reference by its resolved contract under `types_compatible` (a lone `${item...}`
    through the element type of a proven `over:` list), a literal by value, and a
    `${ref :- fallback}`/`${ref :? message}` by its de-nulled reference and its
    fallback. An unproven source, or an opaque/absent child input (incl. the reserved
    `over` key), stays lenient (skipped). Loud + located at the node's Compose-owned
    `input:` mapping range, else its authored node range. Iterates the WIRING (not the
    params) so a mis-named binding still surfaces here when its child decl is absent
    rather than being silently skipped."""
    diagnostics = diagnostics or LoadDiagnostics()
    for label, contracts in (
        ("output producer", producers),
        ("flow input", flow_inputs),
    ):
        for name, contract in contracts.items():
            allowed = (
                (ValueContract, PartialRecordContract)
                if label == "output producer"
                else (ValueContract,)
            )
            if type(contract) not in allowed:
                required = (
                    "ValueContract or PartialRecordContract"
                    if label == "output producer"
                    else "exact ValueContract"
                )
                raise TypeError(
                    f"{label} contracts must be {required} values, "
                    f"got {type(contract).__name__} for {name!r}"
                )
    for nid, result in build_results.items():
        node = result.node
        # CALL/MAP carry `child_inputs` to type-check against; a LOOP also carries a child but
        # its `'a -> 'a` contract is checked by check_loop_type_contract/carried_types instead.
        # A leaf has no `child_inputs` attr, and a zero-input child gives [] (falsy), so skip.
        if node.is_loop or not getattr(node, "child_inputs", None):
            continue
        child_decls = {d.name: d for d in node.child_inputs}
        node_producers, node_inputs = (
            narrowing.for_node(nid) if narrowing is not None else (producers, flow_inputs)
        )
        wiring = flow_wiring.get(nid, {})
        item = item_contract(result, wiring, node_producers, node_inputs)
        for param, src in wiring.items():
            decl = child_decls.get(param)
            if decl is None or decl.contract is None:
                continue  # `over`/name-caught-in-build/opaque child input -> lenient
            message = _binding_mismatch(src, decl, node_producers, node_inputs, item)
            if message is not None:
                rng: Optional[SourceRange] = (
                    result.field_ranges.get(("field", "inputs"))
                    or result.node_range
                )
                raise diagnostics.error(
                    f"call node {nid!r}: binding {param!r} — {message}",
                    range=rng,
                )


def _binding_mismatch(
    src: Any,
    decl: Parameter,
    producers: dict[str, OutputContract],
    flow_inputs: dict[str, ValueContract],
    item: ValueContract | None,
) -> str | None:
    """Return why one binding can't fill the child input `decl`, else `None`.

    A literal is validated by value exactly as the child's start validates it, after
    the start's string conversion, so `"3"` fits an `int` input, a quoted date fits a
    `date` input, and `"abc"` fits neither. A lone `${ref :- fallback}` or
    `${ref :? message}` checks the de-nulled reference, then a literal fallback by value
    and a reference fallback by its contract. Any other source is checked by its
    resolved contract, a lone `${item...}` read through the element type of `over:`.

    Args:
        src (`Any`): The binding, raw or a `Binding`.
        decl (`Parameter`): The child input, with a declared contract.
        producers (`dict[str, OutputContract]`): Node output contracts by producer id.
        flow_inputs (`dict[str, ValueContract]`): Flow input contracts by name.
        item (`ValueContract | None`): The element contract `${item}` reads on a MAP
            over a proven list, else `None`.

    Returns:
        `str | None`: The mismatch, without the node and binding names, or `None` when
            the binding fits or its type is unproven.
    """
    child_type = decl.contract.type
    source = binding_source(src)
    if not (isinstance(src, Binding) and src.contract is not None):
        literal = literal_value(source)
        if literal is not NOT_LITERAL:
            return _value_mismatch("value", literal, decl)
        escape = _null_escape(source)
        if escape is not None:
            head, fallback = escape
            message = _contract_mismatch(
                _path_contract(head, producers, flow_inputs, item),
                child_type,
                non_null=True,
            )
            if message is not None or fallback is None:
                return message
            if isinstance(fallback, str):
                return _contract_mismatch(
                    _path_contract(fallback, producers, flow_inputs, item),
                    child_type,
                    non_null=False,
                )
            return _value_mismatch("default", fallback[0], decl)
        path = direct_reference_path(source)
        if path is not None and path.split(".")[0] == "item":
            return _contract_mismatch(
                _path_contract(path, producers, flow_inputs, item),
                child_type,
                non_null=False,
            )
    # A nullable source is fine for a non-nullable child input ONLY when the binding
    # GUARANTEES a non-null value — a non-null literal operand (`${a | 5}`) or text
    # around a span. A present-null source SHADOWS the child's own `default:`
    # (apply_defaults fills only OMITTED inputs, never a bound null), so it doesn't
    # cover a null. `not binding_co_skips` is exactly that non-null guarantee.
    return _contract_mismatch(
        resolve_output_binding_contract(src, producers, flow_inputs),
        child_type,
        non_null=not binding_co_skips(source),
    )


def _null_escape(source: Any) -> tuple[str, tuple[Any] | str | None] | None:
    """Split a lone `${ref :- fallback}` or `${ref :? message}` into its parts.

    Args:
        source (`Any`): The binding's raw source.

    Returns:
        `tuple[str, tuple[Any] | str | None] | None`: The head reference path and the
            fallback: a one-item tuple holding a literal fallback's value, a reference
            fallback's path, or `None` for `:?` or a computed fallback. `None` when the
            source is not such an escape over a lone reference.
    """
    if not isinstance(source, str):
        return None
    try:
        segments = scan_template(source)
    except ExpressionError:
        return None
    if not (len(segments) == 1 and isinstance(segments[0], Span)):
        return None
    tree = segments[0].tree
    kind = getattr(tree, "data", None)
    if kind not in ("default_expr", "required_expr"):
        return None
    head_parts = direct_reference_parts(tree.children[0])
    if head_parts is None:
        return None
    head = ".".join(head_parts)
    if kind == "required_expr":
        return head, None
    fallback = tree.children[1]
    fallback_parts = direct_reference_parts(fallback)
    if fallback_parts is not None:
        return head, ".".join(fallback_parts)
    if expr_refs(fallback):
        return head, None
    try:
        return head, (eval_expr(fallback, lambda _path: None),)
    except ExpressionError:
        return head, None  # a malformed constant fails when the binding is evaluated


def _path_contract(
    path: str,
    producers: dict[str, OutputContract],
    flow_inputs: dict[str, ValueContract],
    item: ValueContract | None,
) -> OutputContract | None:
    """Return the contract one reference path reads, `item...` through `item`.

    Args:
        path (`str`): The dotted reference path.
        producers (`dict[str, OutputContract]`): Node output contracts by producer id.
        flow_inputs (`dict[str, ValueContract]`): Flow input contracts by name.
        item (`ValueContract | None`): The node's element contract, if any.

    Returns:
        `OutputContract | None`: The projected contract, or `None` when unproven.
    """
    head, *fields = path.split(".")
    if head == "item":
        return None if item is None else project_output_contract(item, fields)
    return reference_contract(path, producers, flow_inputs)


def _contract_mismatch(
    contract: OutputContract | None, child_type: Type, *, non_null: bool
) -> str | None:
    """Return why a source contract can't fill `child_type`, else `None`.

    Args:
        contract (`OutputContract | None`): The source's resolved contract.
        child_type (`Type`): The child input's declared type.
        non_null (`bool`): Whether the binding guarantees a non-null value, so a
            nullable source is compared without its null.

    Returns:
        `str | None`: The mismatch, or `None` when it fits or is unproven (`None`
            contract, or a whole partial record with no exact type).
    """
    if contract is None or type(contract) is PartialRecordContract:
        return None
    if type(contract) is not ValueContract:
        raise TypeError(
            "resolved source contract must be ValueContract, "
            "PartialRecordContract, or None"
        )
    source = contract.type
    if source.nullable and not child_type.nullable and non_null:
        source = replace(source, nullable=False)
    if types_compatible(source, child_type):
        return None
    return (
        f"child expects {canonical_type_label(child_type)!r}, source is "
        f"{canonical_type_label(source)!r}"
    )


def _value_mismatch(what: str, value: Any, decl: Parameter) -> str | None:
    """Return why a literal can't fill the child input `decl`, else `None`.

    Args:
        what (`str`): What the literal is in the binding (`value` or `default`).
        value (`Any`): The literal.
        decl (`Parameter`): The child input, with a declared contract.

    Returns:
        `str | None`: The mismatch, or `None` when the value fits.
    """
    child_type = decl.contract.type
    try:
        # The child's start converts a present string before validating it.
        build_value_as(child_type, coerce_param(decl, value))
    except TypeCheckError as exc:
        message = (
            f"child expects {canonical_type_label(child_type)!r}, but the {what} "
            f"{shown_literal(value)} does not fit"
        )
        if exc.path:
            message += f" ({exc})"
        return message
    return None
