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
from typing import Any

from yian.expr import ExpressionError, infer_expr_type
from yian.expr.template import Span, scan_template, template_refs
from yian.nodes.contracts import (
    ContractOrigins,
    OutputContract,
    PartialRecordContract,
    ValueContract,
    binding_source,
    output_contract_field_error,
    output_contract_type,
)
from yian.typesys.values import Type, ValueKind


def project_output_contract(
    contract: OutputContract,
    fields: list[str],
) -> OutputContract | None:
    """Project a proven path from an exact or partial record output contract."""
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
) -> str | None:
    """Return the field-read error for an output path, else ``None``.

    The error is an undeclared record field or any field on a scalar or list.
    An opaque field inside a partial record remains lenient below that field,
    while the partial record's closed root name set is still enforced.
    """
    return output_contract_field_error(contract, fields, path)


def resolve_output_binding_contract(
    binding: Any,
    producers: dict[str, OutputContract],
    flow_inputs: dict[str, ValueContract],
) -> OutputContract | None:
    """Resolve one Compose output binding to its canonical value contract.

    A direct whole-value reference retains the referenced contract, including
    provenance.  A computed whole-value expression receives a new source-free
    contract only when the expression engine proves its result type.  Embedded
    or multiple template spans always produce ``str``; malformed, literal-text,
    and unsupported computed bindings remain untyped.

    Args:
        binding (`Any`):
            A raw binding value or exact semantic ``Binding`` carrier.
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
    is_direct_reference = getattr(expression, "type", None) == "WRAPPED_REF" or (
        getattr(expression, "data", None) == "refcall"
        and not any(
            getattr(child, "data", None) == "call_suffix"
            for child in getattr(expression, "children", ())
        )
    )
    if is_direct_reference:
        if len(refs) != 1:
            return None
        return reference_contracts[refs[0]]

    def reference_type(path: str) -> Type | None:
        """Project one referenced contract to the type requested by ``expr``."""
        contract = reference_contracts.get(path)
        if path not in reference_contracts:
            contract = _reference_contract(path, producers, flow_inputs)
        return output_contract_type(contract)

    inferred = infer_expr_type(expression, reference_type)
    return None if inferred is None else ValueContract(inferred)


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

    origins: list[tuple[tuple[str | int, ...], Any]] = []
    for name, contract in fields.items():
        assert type(contract) is ValueContract
        if contract.origins.default is not None:
            origins.append(((name,), contract.origins.default))
        origins.extend(
            ((name, *path), origin) for path, origin in contract.origins.entries
        )
    origins.sort(key=_origin_path_sort_key)
    return ValueContract(
        type=Type(
            kind=ValueKind.OBJECT,
            fields={name: contract.type for name, contract in fields.items()},
            required=frozenset(fields),
        ),
        origins=ContractOrigins(entries=tuple(origins)),
    )


__all__ = [
    "combine_output_contracts",
    "output_contract_path_error",
    "project_output_contract",
    "resolve_output_binding_contract",
]
