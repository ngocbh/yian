"""Graph-stage: the TYPE-level half of a `loop`'s `'a -> 'a` contract.

A loader post-build pass (run after `check_call_map_types`). `nodes.loop.build` checked only the
carried field-NAME sets — it lacks the resolved carried-record Types, which depend on the parent
producers and flow inputs assembled by the parent build. This pass resolves each returned carried
field's contract from its bound seed source (`${<id>.output}`/`${input.X}` over the parent
producer + flow-input contracts, via `resolve_output_binding_contract`), projects its Type, and
checks it with `types_compatible` against the body OUTPUT's same-named field Type. A field the
body omits carries over unchanged, so it has nothing to compare. The node's canonical output
contract already retains independently proven body fields when no sound exact aggregate
exists. This pass carries a range anchor from
`BuildResult.field_ranges[("field", "inputs")]` (the coarse `input:` mapping span).

Imports flow down only: `compile.validation` (the type comparator) and
`compose.helpers` (shared output-contract inference and load diagnostics). Only
the loader (and the compose package facade) import this back.
"""

from __future__ import annotations

from typing import Any, Optional

from yian.compile.validation import types_compatible
from yian.compose.helpers.errors import LoadDiagnostics
from yian.compose.helpers.output_contracts import (
    resolve_output_binding_contract,
)
from yian.compose.nodes.base import BuildResult
from yian.diagnostics import SourceRange
from yian.nodes.contracts import (
    OutputContract,
    PartialRecordContract,
    ValueContract,
)
from yian.typesys import canonical_type_label


def check_loop_type_contract(
    build_results: dict[str, BuildResult],
    producers: dict[str, OutputContract],
    flow_inputs: dict[str, ValueContract],
    flow_wiring: dict[str, dict[str, Any]],
    *,
    diagnostics: LoadDiagnostics | None = None,
) -> None:
    """The TYPE-level half of a `loop`'s `'a -> 'a` contract (a loader post-build pass).

    `nodes.loop.build` checked only the field-NAME sets (it lacks the resolved carried-record
    Types because the seed sources' types depend on parent producers and flow inputs).
    This pass, run after `check_call_map_types`, resolves each returned carried field's Type
    from its bound seed source (`${<id>.output}`/`${input.X}` over the parent producer +
    flow-input contracts, exactly as `check_call_map_types` via
    `resolve_output_binding_contract`),
    and checks it with `types_compatible` against the body
    OUTPUT's same-named field Type. A fully proven body reads those fields from the
    aggregate `output_contract`; a partially typed record reads its independently
    proven field contracts from the same canonical output-contract sum. A field outside
    the node's `returned_fields` (omitted by the body, so carried over unchanged) is
    skipped. A literal/coalesce/opaque seed source, or an opaque body output field,
    stays lenient (skipped).
    Loud + located at the loop node's `input:` mapping range
    (else the Compose-owned node range) on a
    same-names/different-TYPES mismatch."""
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
        if not node.is_loop:
            continue
        body_contract = node.output_contract
        if body_contract is not None and type(body_contract) not in (
            ValueContract,
            PartialRecordContract,
        ):
            raise TypeError(
                f"node {nid!r} output_contract must be an exact ValueContract, "
                "PartialRecordContract, or None, "
                f"got {type(body_contract).__name__}"
            )
        body_out = (
            body_contract.type
            if type(body_contract) is ValueContract
            else None
        )
        if body_out is not None and body_out.fields is not None:
            body_fields = body_out.fields
        elif type(body_contract) is PartialRecordContract:
            body_fields = {
                field_name: field_contract.type
                for field_name, field_contract in body_contract.fields
                if field_contract is not None
            }
        else:
            body_fields = {}
        # A non-record / absent body output was already caught by the field-name half in
        # nodes.loop.build. With no independently proven fields, nothing remains to compare.
        if not body_fields:
            continue
        returned = node.returned_fields
        for field_name, src in flow_wiring.get(nid, {}).items():
            if returned is not None and field_name not in returned:
                continue  # omitted by the body: carried over, nothing returned to compare
            body_field = body_fields.get(field_name)
            if body_field is None:
                continue  # name mismatch caught in build; nothing to type-check
            source_contract = resolve_output_binding_contract(
                src,
                producers,
                flow_inputs,
            )
            if source_contract is None:
                continue  # literal/coalesce/opaque seed source -> lenient
            if type(source_contract) is PartialRecordContract:
                continue  # a whole partial record has no exact Type to compare
            if type(source_contract) is not ValueContract:
                raise TypeError(
                    "resolved loop source contract must be ValueContract, "
                    "PartialRecordContract, or None"
                )
            source = source_contract.type
            if not types_compatible(source, body_field):
                rng: Optional[SourceRange] = (
                    result.field_ranges.get(("field", "inputs"))
                    or result.node_range
                )
                raise diagnostics.error(
                    f"loop node {nid!r}: carried field {field_name!r} — body output is "
                    f"{canonical_type_label(body_field)!r}, carried source is "
                    f"{canonical_type_label(source)!r} "
                    "(a returned field must keep its carried type)",
                    range=rng,
                )
