"""Case `on:`-form enum-exhaustiveness — the validate-phase check.

The compile-time coverage law for `case` nodes: when a `case`'s `on:` ref resolves to an ENUM
producer, every enum tag must be covered by a case `when:` value OR by a present `else:`. A
missing tag with no `else:` is a `LoadError`.

This lives in the validate phase (not in `CaseDescriptor.build`) because it needs the flow's
`producers` map — each producer node's `output_contract` — which only exists after every node is
built. `.build()` lowers the case unconditionally; the loader runs `check_case_exhaustive` in
the validate phase, alongside the other cross-node checks.

Imports flow down/peer only: `expr` (ref scan), `nodes.contracts` (the output-contract
sum), `typesys` (Type), and the compose peers `helpers.output_contracts`/`errors` and
`parse.file`. Nothing in the engine imports this back.
"""

from __future__ import annotations

from typing import Optional

from yian.compose.helpers.output_contracts import (
    resolve_output_binding_contract,
)
from yian.expr import ExpressionError, expr_refs_of
from yian.nodes.contracts import (
    OutputContract,
    PartialRecordContract,
    ValueContract,
)
from yian.typesys.values import Type
from yian.compose.helpers.errors import LoadDiagnostics
from yian.compose.parse.file import CaseDescriptor
from yian.compose.validate.narrowing import Narrowing


def check_case_exhaustive(
    descriptors,
    producers: dict[str, OutputContract],
    *,
    narrowing: Narrowing | None = None,
    diagnostics: LoadDiagnostics | None = None,
):
    """Enum-exhaustiveness over every `case` descriptor's on:-form. Raises LoadError
    (range-stamped at the case node) on a non-exhaustive enum with no else:.

    `descriptors` is the flow's `{node_id -> NodeDescriptor}`; `producers` maps producer-node
    id -> its exact or partial output contract. A known field projected from either arm
    can drive enum coverage. With `narrowing`, each case resolves its `on:` ref against
    its own view of the producers, so a case inside a branch covers only the tags that
    can reach it. The descriptor's own range is authoritative.
    """
    diagnostics = diagnostics or LoadDiagnostics()
    for producer_id, contract in producers.items():
        if type(contract) not in (ValueContract, PartialRecordContract):
            raise TypeError(
                "output producer contracts must be ValueContract or "
                "PartialRecordContract values, "
                f"got {type(contract).__name__} for {producer_id!r}"
            )
    for desc in descriptors.values():
        if not isinstance(desc, CaseDescriptor):
            continue
        case_producers = (
            narrowing.for_node(desc.id)[0] if narrowing is not None else producers
        )
        _check_exhaustive(desc, _on_covered(desc), case_producers, diagnostics)


def _on_covered(desc: CaseDescriptor) -> set[str]:
    """The set of enum tags an on:-form case explicitly matches (its `when:` values)."""
    if desc.on is None:
        return set()
    return {str(c.get("when")) for c in desc.cases if c.get("when") is not None}


def _resolve_on_type(
    on_ref: str,
    producers: dict[str, OutputContract],
) -> Optional[Type]:
    """The `Type` an `on: ${<id>.output[.<field>…]}` ref resolves to, else None.

    Resolves the binding through the shared output-contract projector, so an exact
    contract or a known field of a partial record can supply the enum Type. A
    non-node-first head, opaque result, or non-record step returns ``None`` (lenient:
    no exhaustiveness check).
    """
    try:
        refs = expr_refs_of(on_ref)
    except ExpressionError:
        return None
    if len(refs) != 1:
        return None
    parts = refs[0].split(".")
    # Singular only: `<id>.output[.<field>…]`.
    if len(parts) < 2 or parts[1] != "output":
        return None
    contract = resolve_output_binding_contract(on_ref, producers, {})
    return contract.type if type(contract) is ValueContract else None


def _check_exhaustive(
    desc: CaseDescriptor,
    covered: set[str],
    producers: dict[str, OutputContract],
    diagnostics: LoadDiagnostics | None = None,
) -> None:
    """Enum-exhaustiveness for the `on:` form.

    Resolves the `on:` ref to a `Type`; if it carries `.tags` (an enum), every tag must
    be a `when:` match value OR be covered by a present `else:`. A missing tag with no
    `else:` is a `LoadError`. A non-enum / unresolved producer is lenient (no check).
    """
    diagnostics = diagnostics or LoadDiagnostics()
    if desc.on is None or desc.else_ is not None:
        return  # searched form, or an else: that satisfies coverage
    typ = _resolve_on_type(desc.on, producers)
    if typ is None or typ.tags is None:
        return  # not a checked enum producer -> lenient
    missing = sorted(typ.tags - covered)
    if missing:
        raise diagnostics.error(
            f"case node {desc.id!r} on {desc.on}: non-exhaustive — enum tag(s) "
            f"{missing} not covered by a case `when:` or an `else:`",
            range=desc.range,
        )
