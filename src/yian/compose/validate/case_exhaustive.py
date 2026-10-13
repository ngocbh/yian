"""Case exhaustiveness — the validate-phase check.

Two coverage laws for `case` nodes:

- Enum coverage, for every case: when an `on:` ref resolves to an ENUM node output, every
  enum tag must be covered by a case `when:` value OR by a present `else:`. A missing tag
  with no `else:` is a `LoadError`.
- Value-producing cases: a case that produces a value (it declares `output:`, is read
  through `${<case>.output}`, or is a branch target of such a case) must be exhaustive. It
  has an `else:`, or it is an `on:` form whose `on:` type is a non-nullable enum with every
  tag listed in a `when:`. Its `on:` resolves against node outputs and flow inputs, under
  the case's narrowing view. Otherwise it is a `LoadError`. A routing-only case with no
  match runs no branch, so only enum coverage applies to it.

This lives in the validate phase (not in `CaseDescriptor.build`) because it needs the flow's
`producers` map — each producer node's `output_contract` — which only exists after every node is
built. `.build()` lowers the case unconditionally; the loader runs `check_case_exhaustive` in
the validate phase, alongside the other cross-node checks.

Imports flow down/peer only: `expr` (ref scan), `nodes.contracts` (the output-contract
sum), `typesys` (Type), and the compose peers `helpers.output_contracts`/`errors`,
`parse.file`, and `validate.narrowing`. Nothing in the engine imports this back.
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
from yian.typesys.values import Type, canonical_type_label
from yian.compose.helpers.errors import LoadDiagnostics
from yian.compose.parse.file import CaseDescriptor, NodeDescriptor
from yian.compose.validate.narrowing import Narrowing


def check_case_exhaustive(
    descriptors,
    producers: dict[str, OutputContract],
    *,
    narrowing: Narrowing | None = None,
    value_cases: frozenset[str] = frozenset(),
    diagnostics: LoadDiagnostics | None = None,
):
    """Check every case's enum coverage, and that each value-producing case is exhaustive.

    A known field projected from either arm of a partial record can drive enum coverage.
    With `narrowing`, each case resolves its `on:` ref against its own view of the
    producers and flow inputs, so a case inside a branch covers only the tags that can
    reach it. Errors are located at the case node.

    Args:
        descriptors (`dict`): The flow's node descriptors keyed by node id.
        producers (`dict[str, OutputContract]`): Producer node id -> its exact or partial
            output contract.
        narrowing (`Narrowing | None`): Each node's narrowed producer and flow-input
            views. Without it, a case sees `producers` and no flow-input contracts.
        value_cases (`frozenset[str]`): The ids of the value-producing cases, which must
            have an `else:` or list every tag of a non-nullable enum `on:` type.
        diagnostics (`LoadDiagnostics | None`): Source ownership for located errors.

    Raises:
        `LoadError`: A case leaves an enum tag uncovered with no `else:`, or a
            value-producing case is not exhaustive.
        `TypeError`: A producer contract is not a `ValueContract` or
            `PartialRecordContract`.
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
        case_producers, case_inputs = (
            narrowing.for_node(desc.id) if narrowing is not None else (producers, {})
        )
        if desc.id in value_cases:
            _check_value_exhaustive(desc, case_producers, case_inputs, diagnostics)
        _check_exhaustive(desc, _on_covered(desc), case_producers, diagnostics)


def _check_value_exhaustive(
    desc: CaseDescriptor,
    producers: dict[str, OutputContract],
    inputs: dict[str, ValueContract],
    diagnostics: LoadDiagnostics,
) -> None:
    """Require a value-producing case to have an `else:` or to list every enum tag.

    Args:
        desc (`CaseDescriptor`): The value-producing case.
        producers (`dict[str, OutputContract]`): The producer contracts the case sees.
        inputs (`dict[str, ValueContract]`): The flow-input contracts the case sees.
        diagnostics (`LoadDiagnostics`): Source ownership for located errors.

    Raises:
        `LoadError`: The case has no `else:` and its `on:` type is unknown, not an enum,
            nullable, or has a tag no `when:` lists (or it is a searched case).
    """
    if case_is_exhaustive(desc, producers, inputs):
        return
    raise diagnostics.error(
        f"case node {desc.id!r} produces a value, so it must be exhaustive: "
        f"add an `else:` ({_not_exhaustive_reason(desc, producers, inputs)})",
        range=desc.range,
    )


def _not_exhaustive_reason(
    desc: CaseDescriptor,
    producers: dict[str, OutputContract],
    inputs: dict[str, ValueContract],
) -> str:
    """Return why a case that `case_is_exhaustive` rejected may match no branch."""
    if desc.on is None:
        return "a searched case needs an else:"
    typ = _resolve_on_type(desc.on, producers, inputs)
    if typ is None:
        return f"the type of on: {desc.on} is not known"
    if typ.tags is None:
        return f"on: {desc.on} is {canonical_type_label(typ)}, not a Literal"
    if typ.nullable:
        return f"{desc.on} may be null"
    return f"enum tag(s) {sorted(typ.tags - _on_covered(desc))} not covered"


def case_is_exhaustive(
    desc: CaseDescriptor,
    producers: dict[str, OutputContract],
    flow_inputs: dict[str, ValueContract] | None = None,
) -> bool:
    """Return whether a `case` always takes a branch when it runs.

    Args:
        desc (`CaseDescriptor`):
            The `case` descriptor.
        producers (`dict[str, OutputContract]`):
            Producer node id -> output contract, as the case sees them.
        flow_inputs (`dict[str, ValueContract] | None`, *optional*, defaults to `None`):
            Flow input name -> contract, as the case sees them. Without them an `on:` over
            a flow input is never exhaustive.

    Returns:
        `bool`:
            True when the case has an `else:`, or it is the `on:` form over a non-nullable
            `Literal` node output or flow input whose every tag is a `when:` value. A
            searched case without `else:` is never exhaustive.
    """
    if desc.else_ is not None:
        return True
    if desc.on is None:
        return False
    typ = _resolve_on_type(desc.on, producers, flow_inputs or {})
    if typ is None or typ.tags is None or typ.nullable:
        return False
    return typ.tags <= _on_covered(desc)


def exhaustive_cases(
    descriptors: dict[str, NodeDescriptor],
    producers: dict[str, OutputContract],
    flow_inputs: dict[str, ValueContract] | None = None,
    *,
    narrowing: Narrowing | None = None,
) -> set[str]:
    """Return the ids of the CASE descriptors that always take a branch when they run.

    Args:
        descriptors (`dict[str, NodeDescriptor]`):
            The level's node id -> descriptor. Non-CASE descriptors are ignored.
        producers (`dict[str, OutputContract]`):
            Producer node id -> output contract.
        flow_inputs (`dict[str, ValueContract] | None`, *optional*, defaults to `None`):
            Flow input name -> contract.
        narrowing (`Narrowing | None`, *optional*, defaults to `None`):
            Per-node narrowed views; each CASE resolves its `on:` against its own view.

    Returns:
        `set[str]`:
            The exhaustive CASE ids (see `case_is_exhaustive`).
    """
    exhaustive = set()
    for desc in descriptors.values():
        if not isinstance(desc, CaseDescriptor):
            continue
        views = narrowing.for_node(desc.id) if narrowing is not None else (producers, flow_inputs)
        if case_is_exhaustive(desc, *views):
            exhaustive.add(desc.id)
    return exhaustive


def _on_covered(desc: CaseDescriptor) -> set[str]:
    """The set of enum tags an on:-form case explicitly matches (its `when:` values)."""
    if desc.on is None:
        return set()
    return {str(c.get("when")) for c in desc.cases if c.get("when") is not None}


def _resolve_on_type(
    on_ref: str,
    producers: dict[str, OutputContract],
    flow_inputs: dict[str, ValueContract] | None = None,
) -> Optional[Type]:
    """Return the `Type` an `on:` ref resolves to, else None.

    Resolves the binding through the shared output-contract projector, so an exact
    contract or a known field of a partial record can supply the enum Type.

    Args:
        on_ref (`str`):
            The case's `on:` template.
        producers (`dict[str, OutputContract]`):
            Producer node id -> output contract.
        flow_inputs (`dict[str, ValueContract] | None`, *optional*, defaults to `None`):
            Flow input name -> contract. With them an `${input.<name>[.<field>…]}` ref
            resolves too; without them only `${<id>.output[.<field>…]}` does.

    Returns:
        `Optional[Type]`:
            The resolved type, or ``None`` for an unsupported head, an opaque result, or a
            non-record step (lenient: no exhaustiveness check).
    """
    try:
        refs = expr_refs_of(on_ref)
    except ExpressionError:
        return None
    if len(refs) != 1:
        return None
    parts = refs[0].split(".")
    node_output = len(parts) >= 2 and parts[1] == "output"
    flow_input = flow_inputs is not None and len(parts) >= 2 and parts[0] == "input"
    if not (node_output or flow_input):
        return None
    contract = resolve_output_binding_contract(on_ref, producers, flow_inputs or {})
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
