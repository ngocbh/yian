"""Infer canonical value contracts for Compose output bindings.

This module owns the Compose-specific envelope around the generic expression
engine: template shape, reference-to-contract lookup, record-field projection,
and provenance retention.  Direct references re-export their exact contract;
computed expressions receive a source-free contract only when ``expr`` proves
their result type.

Knows about: ``expr`` AST result inference, semantic binding/contract carriers,
and source-neutral ``typesys`` values.
Never imports: YAML/source documents, node kinds, graph-stage passes, the loader,
or runtime.
"""

from __future__ import annotations

from typing import Any

from agent_composer.expr import ExpressionError, infer_expr_type
from agent_composer.expr.template import Span, scan_template, template_refs
from agent_composer.nodes.contracts import (
    ContractOrigins,
    ValueContract,
    binding_source,
)
from agent_composer.typesys.values import Type, ValueKind


def _project_output_contract(
    contract: ValueContract,
    fields: list[str],
) -> ValueContract | None:
    """Project a nested record contract and rebase its origins to the field."""
    if not fields:
        return contract
    typ = contract.type
    for field in fields:
        if typ.fields is None:
            return None
        typ = typ.fields.get(field)
        if typ is None:
            return None
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
    producers: dict[str, ValueContract],
    flow_inputs: dict[str, ValueContract],
) -> ValueContract | None:
    """Resolve one supported reference path to its exact or projected contract."""
    parts = path.split(".")
    head = parts[0]
    if head == "input" and len(parts) == 2:
        contract = flow_inputs.get(parts[1])
        if contract is not None and type(contract) is not ValueContract:
            raise TypeError(
                "flow input contracts must be exact ValueContract values, "
                f"got {type(contract).__name__} for {parts[1]!r}"
            )
        return contract
    if len(parts) < 2 or parts[1] != "output":
        return None

    producer_id = parts[0]
    contract = producers.get(producer_id)
    if contract is not None and type(contract) is not ValueContract:
        raise TypeError(
            "output producer contracts must be exact ValueContract values, "
            f"got {type(contract).__name__} for {producer_id!r}"
        )
    if contract is None:
        return None
    return _project_output_contract(contract, parts[2:])


def resolve_output_binding_contract(
    binding: Any,
    producers: dict[str, ValueContract],
    flow_inputs: dict[str, ValueContract],
) -> ValueContract | None:
    """Resolve one Compose output binding to its canonical value contract.

    A direct whole-value reference retains the referenced contract, including
    provenance.  A computed whole-value expression receives a new source-free
    contract only when the expression engine proves its result type.  Embedded
    or multiple template spans always produce ``str``; malformed, literal-text,
    and unsupported computed bindings remain untyped.

    Args:
        binding (`Any`):
            A raw binding value or exact semantic ``Binding`` carrier.
        producers (`dict[str, ValueContract]`):
            Exact node output contracts keyed by producer id.
        flow_inputs (`dict[str, ValueContract]`):
            Exact flow input contracts keyed by input name.

    Returns:
        `ValueContract | None`:
            The proven canonical contract, or ``None`` when the binding's result
            type is not statically proven.

    Raises:
        `TypeError`:
            A referenced producer or flow-input contract is not an exact
            ``ValueContract``.
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
    if not (len(segments) == 1 and isinstance(segments[0], Span)):
        return ValueContract(Type.scalar(ValueKind.STRING))

    expression = segments[0].tree
    refs = template_refs(segments)
    is_direct_reference = getattr(expression, "data", None) == "refcall" and not any(
        getattr(child, "data", None) == "call_suffix"
        for child in getattr(expression, "children", ())
    )
    if is_direct_reference:
        if len(refs) != 1:
            return None
        return _reference_contract(refs[0], producers, flow_inputs)

    def reference_type(path: str) -> Type | None:
        """Project one referenced contract to the type requested by ``expr``."""
        contract = _reference_contract(path, producers, flow_inputs)
        return None if contract is None else contract.type

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
    fields: dict[str, ValueContract | None],
) -> ValueContract:
    """Combine named output contracts into one closed-record contract.

    Args:
        fields (`dict[str, ValueContract | None]`):
            Output field names mapped to proven contracts or ``None`` for opaque
            fields. Insertion order is retained in the resulting record type.

    Returns:
        `ValueContract`:
            One closed record whose proven field origins are prefixed by their
            output names.
    """
    origins: list[tuple[tuple[str | int, ...], Any]] = []
    for name, contract in fields.items():
        if contract is None:
            continue
        if contract.origins.default is not None:
            origins.append(((name,), contract.origins.default))
        origins.extend(
            ((name, *path), origin) for path, origin in contract.origins.entries
        )
    origins.sort(key=_origin_path_sort_key)
    return ValueContract(
        type=Type(
            kind=ValueKind.OBJECT,
            fields={
                name: (
                    contract.type
                    if contract is not None
                    else Type.scalar(ValueKind.NONE)
                )
                for name, contract in fields.items()
            },
            required=frozenset(
                name for name, contract in fields.items() if contract is not None
            ),
        ),
        origins=ContractOrigins(entries=tuple(origins)),
    )


__all__ = ["combine_output_contracts", "resolve_output_binding_contract"]
