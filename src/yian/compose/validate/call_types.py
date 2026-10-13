"""Graph-stage: the cross-flow TYPE check for every `call`/`map` binding.

A loader post-build pass (run after the flow's leaves + child signatures are assembled).
Each binding's SOURCE contract is resolved from its `${<id>.output}`/`${input.X}` ref over
the PARENT producer + flow-input contracts, then its `Type` is projected and checked with
`types_compatible` against the CHILD input's declared `Type`. Its range anchor comes from
the call node's transient Compose
`BuildResult.field_ranges[("field", "inputs")]` (the coarse `input:` mapping span).

Imports flow down only: `compile.validation` (the type comparator), `expr` (the binding-null
analysis), and `compose.helpers` (shared output-contract inference and load diagnostics).
Only the loader (and the compose package facade) import this back.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Optional

from yian.compile.validation import types_compatible
from yian.compose.helpers.errors import LoadDiagnostics
from yian.compose.helpers.output_contracts import (
    resolve_output_binding_contract,
)
from yian.compose.nodes.base import BuildResult
from yian.compose.validate.narrowing import Narrowing
from yian.diagnostics import SourceRange
from yian.expr import binding_co_skips
from yian.nodes.contracts import (
    OutputContract,
    PartialRecordContract,
    ValueContract,
    binding_source,
)
from yian.typesys import canonical_type_label


def check_call_map_types(
    build_results: dict[str, BuildResult],
    producers: dict[str, OutputContract],
    flow_inputs: dict[str, ValueContract],
    flow_wiring: dict[str, dict[str, Any]],
    *,
    narrowing: Narrowing | None = None,
    diagnostics: LoadDiagnostics | None = None,
) -> None:
    """Cross-flow type check for every `call` binding (a loader post-build pass).

    Each binding's SOURCE contract (resolved from its `${<id>.output}`/`${input.X}` ref
    over the PARENT producer + flow-input contracts, as the call node's `narrowing` view
    sees them when given) is projected to its `Type` and must
    satisfy `types_compatible` against the CHILD input's declared `Type`. A
    `${item}`/coalesce/literal/opaque source — or an opaque/absent child input
    (incl. the reserved `over` key) — stays lenient (skipped). Loud + located at the
    node's Compose-owned `input:` mapping range, else its authored node range.
    Iterates the WIRING (not the params)
    so a mis-named binding still surfaces here when its child decl is absent rather than
    being silently skipped."""
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
        for param, src in flow_wiring.get(nid, {}).items():
            decl = child_decls.get(param)
            if decl is None or decl.contract is None:
                continue  # `over`/name-caught-in-build/opaque child input -> lenient
            child_type = decl.contract.type
            source_contract = resolve_output_binding_contract(
                src,
                node_producers,
                node_inputs,
            )
            if source_contract is None:
                continue  # ${item}/coalesce/literal/opaque source -> lenient
            if type(source_contract) is PartialRecordContract:
                continue  # a whole partial record has no exact Type to compare
            if type(source_contract) is not ValueContract:
                raise TypeError(
                    "resolved source contract must be ValueContract, "
                    "PartialRecordContract, or None"
                )
            source = source_contract.type
            # A nullable source is fine for a non-nullable child input ONLY when the
            # binding GUARANTEES a non-null value — a non-null literal escape or a `:?`
            # required (both fire on null, present-null included). A ref-default `:-${y}`
            # may itself be null, and the child's own `default:` does NOT cover it: a
            # present-null source SHADOWS the child default (apply_defaults fills only
            # OMITTED inputs, never a bound null). `not binding_co_skips` is exactly that
            # non-null guarantee (a `|` coalesce of refs stays strict).
            if (
                source.nullable
                and not child_type.nullable
                and not binding_co_skips(binding_source(src))
            ):
                source = replace(source, nullable=False)
            if not types_compatible(source, child_type):
                rng: Optional[SourceRange] = (
                    result.field_ranges.get(("field", "inputs"))
                    or result.node_range
                )
                raise diagnostics.error(
                    f"call node {nid!r}: binding {param!r} — child expects "
                    f"{canonical_type_label(child_type)!r}, source is "
                    f"{canonical_type_label(source)!r}",
                    range=rng,
                )
