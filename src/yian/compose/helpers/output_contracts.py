"""Infer canonical value contracts for Compose output bindings.

This module owns the Compose-specific envelope around the generic expression
engine: template shape, reference-to-contract lookup, record-field projection,
and provenance retention. Direct references re-export their exact or partial
contract; computed expressions receive a source-free exact contract only when
``expr`` proves their result type.

Knows about: ``expr`` AST result inference, semantic binding/contract carriers,
and source-neutral ``typesys`` values.
Never imports: YAML/source documents, node kinds, graph-stage passes, the loader,
or runtime.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Iterable

from lark import Token, Tree

from yian.expr import ExpressionError, expr_refs, infer_expr_type
from yian.expr.template import Span, scan_template, template_refs
from yian.nodes.contracts import (
    Binding,
    ContractOrigins,
    OutputContract,
    PartialRecordContract,
    ValueContract,
    binding_source,
    output_contract_field_error,
    output_contract_type,
)
from yian.typesys.values import FieldRead, Type, ValueKind, union_field


def project_output_contract(
    contract: OutputContract,
    fields: list[str],
) -> OutputContract | None:
    """Project a proven path from an exact or partial record output contract.

    A field read on a union projects to the members' shared field type; a field
    only some members have, or whose types differ, projects to `None`.
    """
    if type(contract) is PartialRecordContract:
        contract.validate_shape()
        if not fields:
            return contract
        field_name, *remaining = fields
        field_contracts = dict(contract.fields)
        if field_name not in field_contracts:
            return None
        field_contract = field_contracts[field_name]
        if field_contract is None:
            return None
        return project_output_contract(field_contract, remaining)
    if type(contract) is not ValueContract:
        raise TypeError(
            "output contract must be an exact ValueContract or "
            f"PartialRecordContract, got {type(contract).__name__}"
        )
    if not fields:
        return contract
    typ = contract.type
    path_nullable = typ.nullable
    for field in fields:
        if typ.variants is not None:
            read = union_field(typ, field)
            if not isinstance(read, FieldRead) or read.type is None:
                return None
            # The shared type is already nullable when any member's field may be absent.
            path_nullable = path_nullable or read.type.nullable
            typ = read.type
            continue
        if typ.fields is None:
            return None
        field_type = typ.fields.get(field)
        if field_type is None:
            return None
        path_nullable = (
            path_nullable
            or field not in (typ.required or frozenset())
            or field_type.nullable
        )
        typ = field_type
    if path_nullable and not typ.nullable:
        typ = replace(typ, nullable=True)
    prefix = tuple(fields)
    entries = tuple(
        (path[len(prefix) :], origin)
        for path, origin in contract.origins.entries
        if len(path) > len(prefix) and path[: len(prefix)] == prefix
    )
    return ValueContract(
        type=typ,
        origins=ContractOrigins(
            default=contract.origins.at(prefix),
            entries=entries,
        ),
    )


def _reference_contract(
    path: str,
    producers: dict[str, OutputContract],
    flow_inputs: dict[str, ValueContract],
) -> OutputContract | None:
    """Resolve one supported reference path to its exact or projected contract."""
    parts = path.split(".")
    head = parts[0]
    if head == "input" and len(parts) >= 2:
        contract = flow_inputs.get(parts[1])
        if contract is not None and type(contract) is not ValueContract:
            raise TypeError(
                "flow input contracts must be exact ValueContract values, "
                f"got {type(contract).__name__} for {parts[1]!r}"
            )
        if contract is None:
            return None
        return project_output_contract(contract, parts[2:])
    if len(parts) < 2 or parts[1] != "output":
        return None

    producer_id = parts[0]
    contract = producers.get(producer_id)
    if contract is not None and type(contract) not in (
        ValueContract,
        PartialRecordContract,
    ):
        raise TypeError(
            "output producer contracts must be ValueContract or "
            "PartialRecordContract values, "
            f"got {type(contract).__name__} for {producer_id!r}"
        )
    if contract is None:
        return None
    return project_output_contract(contract, parts[2:])


def output_contract_path_error(
    contract: OutputContract,
    fields: list[str],
    path: str,
    *,
    partial_union_fields: bool = False,
) -> str | None:
    """Return the field-read error for an output path, else ``None``.

    The error is an undeclared record field, any field on a scalar or list, or a
    field only some members of a union have (unless `partial_union_fields`).
    An opaque field inside a partial record remains lenient below that field,
    while the partial record's closed root name set is still enforced.
    """
    return output_contract_field_error(
        contract, fields, path, partial_union_fields=partial_union_fields
    )


def resolve_output_binding_contract(
    binding: Any,
    producers: dict[str, OutputContract],
    flow_inputs: dict[str, ValueContract],
) -> OutputContract | None:
    """Resolve one Compose output binding to its canonical value contract.

    A typed binding (a ``Binding`` whose ``contract`` is set) resolves to its
    declared contract, since that type is what the binding guarantees at runtime.
    Otherwise a direct whole-value reference retains the referenced contract, including
    provenance.  An explicit null default ``${ref :- null}`` over such a reference retains
    it as ``Optional``; ``${(a | b) :- null}`` over references of one exact type receives a
    source-free ``Optional`` contract.  A computed whole-value expression receives a new
    source-free contract only when the expression engine proves its result type.  Embedded
    or multiple template spans always produce ``str``; malformed, literal-text, and
    unsupported computed bindings remain untyped.

    Args:
        binding (`Any`):
            A raw binding value or exact semantic ``Binding`` carrier, possibly
            carrying a declared contract.
        producers (`dict[str, OutputContract]`):
            Exact or partial node output contracts keyed by producer id. A
            proven field of a partial record remains projectable.
        flow_inputs (`dict[str, ValueContract]`):
            Exact flow input contracts keyed by input name.

    Returns:
        `OutputContract | None`:
            The proven canonical exact or partial contract, or ``None`` when the
            binding's result shape is not statically proven.

    Raises:
        `TypeError`:
            A referenced producer is not an ``OutputContract``, or a flow-input
            contract is not an exact ``ValueContract``.
    """
    if isinstance(binding, Binding) and binding.contract is not None:
        return binding.contract
    source = binding_source(binding)
    if not isinstance(source, str):
        return None
    try:
        segments = scan_template(source)
    except ExpressionError:
        return None
    spans = [segment for segment in segments if isinstance(segment, Span)]
    if not spans:
        return None
    refs = template_refs(segments)
    reference_contracts = {
        path: _reference_contract(path, producers, flow_inputs) for path in refs
    }
    if not (len(segments) == 1 and isinstance(segments[0], Span)):
        return ValueContract(Type.scalar(ValueKind.STRING))

    expression = segments[0].tree
    if _is_direct_reference(expression):
        if len(refs) != 1:
            return None
        return reference_contracts[refs[0]]
    null_default = _null_default_contract(expression, reference_contracts)
    if null_default is not None:
        return null_default

    def reference_type(path: str) -> Type | None:
        """Project one referenced contract to the type requested by ``expr``."""
        contract = reference_contracts.get(path)
        if path not in reference_contracts:
            contract = _reference_contract(path, producers, flow_inputs)
        return output_contract_type(contract)

    inferred = infer_expr_type(expression, reference_type)
    return None if inferred is None else ValueContract(inferred)


def _null_default_contract(
    expression: Any,
    reference_contracts: dict[str, OutputContract | None],
) -> ValueContract | None:
    """Type `head :- null` as Optional when `head` is a reference or a coalesce of them.

    Args:
        expression (`Any`): The parsed tree of a template's single span.
        reference_contracts (`dict[str, OutputContract | None]`):
            The resolved contract of every reference path the span reads.

    Returns:
        `ValueContract | None`:
            The head reference's exact contract made nullable with its origins, a
            source-free nullable contract for a coalesce of references of one exact
            type, or ``None`` for any other expression.
    """
    if not (
        getattr(expression, "data", None) == "default_expr"
        and len(expression.children) == 2
        and getattr(expression.children[1], "type", None) == "NULL"
    ):
        return None
    head = expression.children[0]
    if getattr(head, "data", None) == "coalesce":
        operands = [
            _direct_value_contract(operand, reference_contracts)
            for operand in head.children
        ]
        if any(operand is None for operand in operands):
            return None
        types = [replace(operand.type, nullable=False) for operand in operands]
        if any(typ != types[0] for typ in types[1:]):
            return None
        return ValueContract(replace(types[0], nullable=True))
    contract = _direct_value_contract(head, reference_contracts)
    if contract is None:
        return None
    return replace(contract, type=replace(contract.type, nullable=True))


def _direct_value_contract(
    operand: Any,
    reference_contracts: dict[str, OutputContract | None],
) -> ValueContract | None:
    """Return the exact contract of an operand that is one direct reference.

    Args:
        operand (`Any`): One parsed operand of a span's expression tree.
        reference_contracts (`dict[str, OutputContract | None]`):
            The resolved contract of every reference path the span reads.

    Returns:
        `ValueContract | None`:
            The referenced exact contract, or ``None`` when the operand is not a lone
            reference or its contract is partial or unknown.
    """
    if not _is_direct_reference(operand):
        return None
    contract = reference_contracts.get(expr_refs(operand)[0])
    return contract if type(contract) is ValueContract else None


def direct_reference_path(binding: Any) -> str | None:
    """Return the path a binding reads when it is one lone whole-value reference.

    Args:
        binding (`Any`): A raw binding value or semantic ``Binding`` carrier.

    Returns:
        `str | None`: The dotted reference path (``item``, ``item.name``,
            ``a.output``), or ``None`` for a literal, an embedded or computed
            template, a call, or malformed text.
    """
    source = binding_source(binding)
    if not isinstance(source, str):
        return None
    try:
        segments = scan_template(source)
    except ExpressionError:
        return None
    if not (len(segments) == 1 and isinstance(segments[0], Span)):
        return None
    if not _is_direct_reference(segments[0].tree):
        return None
    refs = template_refs(segments)
    return refs[0] if len(refs) == 1 else None


def _is_direct_reference(expression: Any) -> bool:
    """Return whether a whole-value span is a bare reference, not a call or operator.

    Args:
        expression (`Any`): The parsed tree of a template's single span.

    Returns:
        `bool`: ``True`` for a wrapped or dotted reference with no call suffix.
    """
    return getattr(expression, "type", None) == "WRAPPED_REF" or (
        getattr(expression, "data", None) == "refcall"
        and not any(
            getattr(child, "data", None) == "call_suffix"
            for child in getattr(expression, "children", ())
        )
    )


def _origin_path_sort_key(
    entry: tuple[tuple[str | int, ...], Any],
) -> tuple[tuple[int, str | int], ...]:
    """Return the same total path ordering required by ``ContractOrigins``."""
    return tuple(
        (0, segment) if type(segment) is str else (1, segment) for segment in entry[0]
    )


def combine_output_contracts(
    fields: dict[str, OutputContract | None],
) -> OutputContract:
    """Combine named outputs into one exact or partial closed-record contract.

    Args:
        fields (`dict[str, OutputContract | None]`):
            Output field names mapped to proven contracts or ``None`` for opaque
            fields. A nested partial contract cannot be represented inside one
            root field and is therefore retained as opaque at this boundary.
            Insertion order is retained.

    Returns:
        `OutputContract`:
            One exact closed record when every field is exact; otherwise one
            partial record that retains the closed name set and every exact
            sibling contract.
    """
    for name, contract in fields.items():
        if contract is not None and type(contract) not in (
            ValueContract,
            PartialRecordContract,
        ):
            raise TypeError(
                "output fields must contain ValueContract, PartialRecordContract, "
                f"or None values, got {type(contract).__name__} for {name!r}"
            )
    if any(type(contract) is not ValueContract for contract in fields.values()):
        return PartialRecordContract(
            tuple(
                (
                    name,
                    contract if type(contract) is ValueContract else None,
                )
                for name, contract in fields.items()
            )
        )

    return ValueContract(
        type=Type(
            kind=ValueKind.OBJECT,
            fields={name: contract.type for name, contract in fields.items()},
            required=frozenset(fields),
        ),
        origins=record_field_origins(fields.items()),
    )


def record_field_origins(
    fields: Iterable[tuple[str, ValueContract | None]],
) -> ContractOrigins:
    """Index each field contract's origins under its field name, as one record's origins.

    Args:
        fields (`Iterable[tuple[str, ValueContract | None]]`):
            Record field names paired with exact contracts; `None` (an opaque field)
            contributes no origin.

    Returns:
        `ContractOrigins`:
            No root origin; each field's root origin at `(name,)` and its nested origins
            below it, in canonical path order.
    """
    origins: list[tuple[tuple[str | int, ...], Any]] = []
    for name, contract in fields:
        if contract is None:
            continue
        if contract.origins.default is not None:
            origins.append(((name,), contract.origins.default))
        origins.extend(
            ((name, *path), origin) for path, origin in contract.origins.entries
        )
    origins.sort(key=_origin_path_sort_key)
    return ContractOrigins(entries=tuple(origins))


def direct_reference_parts(expression: Tree | Token) -> tuple[str, ...] | None:
    """Return the path of a lone plain reference, excluding calls and expressions.

    Args:
        expression (`Tree | Token`):
            One parsed `${...}` interior, as `Span.tree`.

    Returns:
        `tuple[str, ...] | None`:
            The dotted path's names, such as `("d", "output", "action")`, or `None`
            when the expression is not a plain `refcall` of names (an index, a call,
            an escape such as `:-`, or any operator).
    """
    if not isinstance(expression, Tree) or expression.data != "refcall":
        return None

    children = expression.children
    if (
        not children
        or not isinstance(children[0], Token)
        or str(children[0].type) != "NAME"
    ):
        return None

    parts = [str(children[0])]
    for child in children[1:]:
        if (
            not isinstance(child, Tree)
            or child.data != "trailer"
            or len(child.children) != 1
            or not isinstance(child.children[0], Token)
            or str(child.children[0].type) != "NAME"
        ):
            return None
        parts.append(str(child.children[0]))
    return tuple(parts)


__all__ = [
    "combine_output_contracts",
    "direct_reference_parts",
    "output_contract_path_error",
    "project_output_contract",
    "record_field_origins",
    "resolve_output_binding_contract",
]
