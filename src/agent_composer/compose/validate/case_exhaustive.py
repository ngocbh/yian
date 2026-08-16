"""Case `on:`-form enum-exhaustiveness — the validate-phase check.

The compile-time coverage law for `case` nodes: when a `case`'s `on:` ref resolves to an ENUM
producer, every enum tag must be covered by a case `when:` value OR by a present `else:`. A
missing tag with no `else:` is a `LoadError`.

This lives in the validate phase (not in `CaseDescriptor.build`) because it needs the flow's
`producers` map — each producer node's `output_contract` — which only exists after every node is
built. `.build()` lowers the case unconditionally; the loader runs `check_case_exhaustive` in
the validate phase, alongside the other cross-node checks.

Imports flow down/peer only: `expr` (ref scan), `nodes.contracts` (ValueContract),
`typesys` (Type), `compile.validation` (`_walk_record_fields`, a representation-neutral leaf
checker, imported locally inside the resolver to avoid coupling the walk twice), and the
compose peers `helpers.errors`, `parse.file`. Nothing in the engine imports this back.
"""

from __future__ import annotations

from typing import Optional

from agent_composer.expr import ExpressionError, expr_refs_of
from agent_composer.nodes.contracts import ValueContract
from agent_composer.typesys.values import Type
from agent_composer.compose.helpers.errors import LoadDiagnostics
from agent_composer.compose.parse.file import CaseDescriptor


def check_case_exhaustive(
    descriptors,
    producers: dict[str, ValueContract],
    *,
    diagnostics: LoadDiagnostics | None = None,
):
    """Enum-exhaustiveness over every `case` descriptor's on:-form. Raises LoadError
    (range-stamped at the case node) on a non-exhaustive enum with no else:.

    `descriptors` is the flow's `{node_id -> NodeDescriptor}`; `producers` maps producer-node
    id -> its exact `ValueContract` (whose semantic type drives the dotted-field walk).
    The descriptor's own range is authoritative.
    """
    diagnostics = diagnostics or LoadDiagnostics()
    for producer_id, contract in producers.items():
        if type(contract) is not ValueContract:
            raise TypeError(
                "output producer contracts must be exact ValueContract values, "
                f"got {type(contract).__name__} for {producer_id!r}"
            )
    for desc in descriptors.values():
        if not isinstance(desc, CaseDescriptor):
            continue
        _check_exhaustive(desc, _on_covered(desc), producers, diagnostics)


def _on_covered(desc: CaseDescriptor) -> set[str]:
    """The set of enum tags an on:-form case explicitly matches (its `when:` values)."""
    if desc.on is None:
        return set()
    return {str(c.get("when")) for c in desc.cases if c.get("when") is not None}


def _resolve_on_type(
    on_ref: str,
    producers: dict[str, ValueContract],
) -> Optional[Type]:
    """The `Type` an `on: ${<id>.output[.<field>…]}` ref resolves to, else None.

    Projects the producing node's exact `ValueContract` to its semantic type at
    the dotted-field algorithm boundary, then walks it field by field
    (`_walk_record_fields`) into a record to reach the named field. A
    non-node-first head / opaque producer / non-record step -> None (lenient: no
    exhaustiveness check). Imported here to avoid coupling the walk logic twice.
    """
    from agent_composer.compile.validation import _walk_record_fields

    try:
        refs = expr_refs_of(on_ref)
    except ExpressionError:
        return None
    if len(refs) != 1:
        return None
    parts = refs[0].split(".")
    # Singular only: `<id>.output[.<field>…]`.
    if len(parts) >= 2 and parts[1] == "output":
        producer_id = parts[0]
        fields = parts[2:]
    else:
        return None
    contract = producers.get(producer_id)
    if contract is None:
        return None
    if type(contract) is not ValueContract:
        raise TypeError(
            "output producer contracts must be exact ValueContract values, "
            f"got {type(contract).__name__} for {producer_id!r}"
        )
    typ = contract.type
    # Reuse the dotted-field walk only to FAIL on a bad dotted field; then re-walk to the Type.
    if _walk_record_fields(typ, fields, refs[0]) is not None:
        return None  # bad field -> the reference-wiring pass reports it; skip exhaustiveness
    for f in fields:
        if typ is None or typ.fields is None:
            return None
        typ = typ.fields.get(f)
    return typ


def _check_exhaustive(
    desc: CaseDescriptor,
    covered: set[str],
    producers: dict[str, ValueContract],
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
