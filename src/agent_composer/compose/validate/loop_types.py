"""Graph-stage: the TYPE-level half of a `loop`'s `'a -> 'a` contract.

A loader post-build pass (run after `check_ref_map_types`). `nodes.loop.build` checked only the
carried field-NAME sets — it lacks the resolved carried-record Types, which depend on the parent
producers/flow inputs assembled LATER. This pass resolves each carried field's contract from its
bound seed source (`${<id>.output}`/`${input.X}` over the parent producer + flow-input
contracts, via `resolve_output_binding_contract`), projects its Type, and checks it is structurally
compatible (`types_compatible`, C-EQUIV) with the body OUTPUT's same-named field Type. It
carries a range anchor from the loop's transient Compose
`BuildResult.field_ranges[("field", "inputs")]` (the coarse `input:` mapping span).

Imports flow down only: `compile.validation` (the type comparator) and
`compose.helpers` (shared output-contract inference and load diagnostics). Only
the loader (and the compose package facade) import this back.
"""

from __future__ import annotations

from typing import Any, Optional

from agent_composer.compile.validation import types_compatible
from agent_composer.compose.helpers.errors import LoadDiagnostics
from agent_composer.compose.helpers.output_contracts import (
    resolve_output_binding_contract,
)
from agent_composer.compose.nodes.base import BuildResult
from agent_composer.diagnostics import SourceRange
from agent_composer.nodes.contracts import ValueContract
from agent_composer.typesys import canonical_type_label


def check_loop_type_contract(
    build_results: dict[str, BuildResult],
    producers: dict[str, ValueContract],
    flow_inputs: dict[str, ValueContract],
    flow_wiring: dict[str, dict[str, Any]],
    *,
    diagnostics: LoadDiagnostics | None = None,
) -> None:
    """The TYPE-level half of a `loop`'s `'a -> 'a` contract (a loader post-build pass).

    `nodes.loop.build` checked only the field-NAME sets (it lacks the resolved carried-record
    Types — the seed sources' types depend on the parent producers/flow inputs assembled
    LATER). This pass, run after `check_ref_map_types`, resolves each carried field's Type
    from its bound seed source (`${<id>.output}`/`${input.X}` over the parent producer +
    flow-input contracts, exactly as `check_ref_map_types` via
    `resolve_output_binding_contract`),
    and checks it is structurally compatible (`types_compatible`, C-EQUIV) with the body
    OUTPUT's same-named field Type (the body `output_contract.type`). A
    literal/coalesce/opaque seed source, or an opaque body output field, stays lenient
    (skipped). Loud + located at the loop node's `input:` mapping range
    (else the Compose-owned node range) on a
    same-names/different-TYPES mismatch."""
    diagnostics = diagnostics or LoadDiagnostics()
    for label, contracts in (
        ("output producer", producers),
        ("flow input", flow_inputs),
    ):
        for name, contract in contracts.items():
            if type(contract) is not ValueContract:
                raise TypeError(
                    f"{label} contracts must be exact ValueContract values, "
                    f"got {type(contract).__name__} for {name!r}"
                )
    for nid, result in build_results.items():
        node = result.node
        if not node.is_loop:
            continue
        body_contract = node.output_contract
        if body_contract is not None and type(body_contract) is not ValueContract:
            raise TypeError(
                f"node {nid!r} output_contract must be an exact ValueContract or None, "
                f"got {type(body_contract).__name__}"
            )
        body_out = None if body_contract is None else body_contract.type
        # A non-record / absent body output was already caught by the field-name half in
        # nodes.loop.build; nothing typed to compare against here.
        if body_out is None or body_out.fields is None:
            continue
        for field_name, src in flow_wiring.get(nid, {}).items():
            body_field = body_out.fields.get(field_name)
            if body_field is None:
                continue  # name mismatch caught in build; nothing to type-check
            source_contract = resolve_output_binding_contract(
                src,
                producers,
                flow_inputs,
            )
            if source_contract is None:
                continue  # literal/coalesce/opaque seed source -> lenient
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
                    "(the body must return the same 'a it reads)",
                    range=rng,
                )
