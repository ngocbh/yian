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
For an inline body, whose parameters are generated from the seeds, each proven returned
field must also fit its generated parameter; that error points at the seed binding.
A literal seed stays lenient here (its contract is not inferred); `check_call_map_types`
checks every seed against the body's declared input.

Imports flow down only: `compile.validation` (the type comparator) and
`compose.helpers` (shared output-contract inference and load diagnostics). Only
the loader (and the compose package facade) import this back.
"""

from __future__ import annotations

import json
from collections.abc import Collection
from dataclasses import replace
from typing import Any, Optional

from yian.compile.validation import types_compatible
from yian.compose.helpers.errors import LoadDiagnostics
from yian.compose.helpers.isinstance_types import constructor_type_fn
from yian.compose.helpers.tags import any_source_hint, suggest_tag
from yian.compose.helpers.output_contracts import (
    resolve_output_binding_contract,
)
from yian.compose.nodes.base import BuildResult
from yian.diagnostics import SourceRange
from yian.nodes.contracts import (
    Binding,
    OutputContract,
    PartialRecordContract,
    ValueContract,
    ValueTemplate,
    binding_source,
)
from yian.typesys import canonical_type_label
from yian.typesys.types import TypeRegistry
from yian.typesys.values import Type, value_repr


def check_loop_type_contract(
    build_results: dict[str, BuildResult],
    producers: dict[str, OutputContract],
    flow_inputs: dict[str, ValueContract],
    flow_wiring: dict[str, dict[str, Any]],
    *,
    inline_loops: Collection[str] = (),
    registry: TypeRegistry | None = None,
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
    OUTPUT's same-named field Type, read from the node's carried-record `output_contract`
    (the body's own contract for a total loop, the merged returned-plus-seeded contract
    for a partial one). An exact record reads those fields from its aggregate type; a
    partially typed record reads its independently proven field contracts. A field outside
    the node's `returned_fields` (omitted by the body, so carried over unchanged) is
    skipped. A literal/coalesce/opaque seed source, or an opaque body output field,
    stays lenient (skipped); `check_call_map_types` checks every seed against the
    body's declared input.
    Loud + located at the loop node's `input:` mapping range
    (else the Compose-owned node range) on a
    same-names/different-TYPES mismatch.

    An inline body's parameters are generated from the seeds, and each returned field is
    bound to its parameter again on the next iteration, so for a loop in `inline_loops`
    every statically proven returned field must also fit its generated parameter.

    Args:
        build_results (`dict[str, BuildResult]`): The built nodes keyed by node id.
        producers (`dict[str, OutputContract]`): Each producer node's output contract.
        flow_inputs (`dict[str, ValueContract]`): The flow's declared input contracts.
        flow_wiring (`dict[str, dict[str, Any]]`): Each node's `param -> source` bindings.
        inline_loops (`Collection[str]`, *optional*, defaults to `()`): The ids of loop
            nodes whose body is written inline.
        registry (`TypeRegistry | None`, *optional*, defaults to `None`): The flow's
            typedefs, used to spell a suggested seed tag by a typedef name and to type
            a `T(...)` constructor seed as `T`.
        diagnostics (`LoadDiagnostics | None`, *optional*, defaults to `None`): Factory
            for located load errors.

    Raises:
        `LoadError`: A returned field does not keep its carried type (with advice to
            tag the seed or narrow it with `isinstance` when it is `Any`), or an inline
            body returns a field that cannot fit its generated parameter.
        `TypeError`: A producer, flow input, or loop contract has an unexpected type."""
    diagnostics = diagnostics or LoadDiagnostics()
    type_fn = constructor_type_fn(registry) if registry is not None else None
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
        # The node's carried-record contract: the body's own contract for a total loop, or
        # the merged returned-plus-seeded contract for a partial loop.
        loop_contract = node.output_contract
        if loop_contract is not None and type(loop_contract) not in (
            ValueContract,
            PartialRecordContract,
        ):
            raise TypeError(
                f"node {nid!r} output_contract must be an exact ValueContract, "
                "PartialRecordContract, or None, "
                f"got {type(loop_contract).__name__}"
            )
        loop_out = (
            loop_contract.type
            if type(loop_contract) is ValueContract
            else None
        )
        if loop_out is not None and loop_out.fields is not None:
            carried_fields = loop_out.fields
        elif type(loop_contract) is PartialRecordContract:
            carried_fields = {
                field_name: field_contract.type
                for field_name, field_contract in loop_contract.fields
                if field_contract is not None
            }
        else:
            carried_fields = {}
        # A non-record / absent carried contract was already caught by the field-name half in
        # nodes.loop.build. With no independently proven fields, nothing remains to compare.
        if not carried_fields:
            continue
        returned = node.returned_fields
        for field_name, src in flow_wiring.get(nid, {}).items():
            if returned is not None and field_name not in returned:
                continue  # omitted by the body: carried over, nothing returned to compare
            # A returned field's carried contract is the body's own field contract.
            body_field = carried_fields.get(field_name)
            if body_field is None:
                continue  # name mismatch caught in build; nothing to type-check
            source_contract = resolve_output_binding_contract(
                src,
                producers,
                flow_inputs,
                type_fn=type_fn,
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
                message = (
                    f"loop node {nid!r}: carried field {field_name!r} — body output is "
                    f"{canonical_type_label(body_field)!r}, carried source is "
                    f"{canonical_type_label(source)!r} "
                    "(a returned field must keep its carried type)"
                )
                hint = any_source_hint(field_name, source, body_field, registry=registry)
                if hint is not None:
                    message += f"; {hint}"
                raise diagnostics.error(message, range=rng)
        if nid in inline_loops:
            _check_inline_returned_fields(
                result,
                carried_fields,
                flow_wiring.get(nid, {}),
                registry or {},
                diagnostics,
            )


def _check_inline_returned_fields(
    result: BuildResult,
    carried_fields: dict[str, Type],
    wiring: dict[str, Any],
    registry: TypeRegistry,
    diagnostics: LoadDiagnostics,
) -> None:
    """Require each returned field of an inline loop body to fit its generated parameter.

    Over a typed seed, a nullable returned field whose non-null type fits the declared
    parameter is accepted; a null it returns fails the next iteration's input check.

    Args:
        result (`BuildResult`): The built loop node and its source ranges.
        carried_fields (`dict[str, Type]`): The statically proven carried field types;
            a returned field's entry is the body's own field type.
        wiring (`dict[str, Any]`): The loop's seed bindings keyed by carried field.
        registry (`TypeRegistry`): The flow's typedefs, for the suggested seed tag.
        diagnostics (`LoadDiagnostics`): Factory for located load errors.

    Raises:
        `LoadError`: A returned field's type cannot fit the parameter generated from its
            seed; the error names both types and suggests typing the seed.
    """
    node = result.node
    returned = node.returned_fields
    parameters = {
        parameter.name: parameter.contract
        for parameter in node.child_inputs
        if parameter.contract is not None
    }
    for field_name, seed in wiring.items():
        if returned is not None and field_name not in returned:
            continue  # omitted by the body: never bound to its parameter again
        body_field = carried_fields.get(field_name)
        parameter = parameters.get(field_name)
        if body_field is None or parameter is None:
            continue  # an opaque returned field has nothing proven to compare
        if types_compatible(body_field, parameter.type):
            continue
        typed = isinstance(seed, Binding) and seed.contract is not None
        # The next iteration's input check rejects a null.
        if typed and types_compatible(replace(body_field, nullable=False), parameter.type):
            continue
        returned_label = canonical_type_label(body_field)
        raise diagnostics.error(
            f"loop node {node.id!r}: the body returns {field_name!r} as {returned_label}, "
            "but the carried field is "
            + (
                f"declared {canonical_type_label(parameter.type)}"
                if typed
                else f"typed {canonical_type_label(parameter.type)} from its seed"
            )
            + "; declare the seed's type: "
            + suggest_tag(
                field_name,
                body_field,
                _seed_text(binding_source(seed)),
                registry=registry,
            ),
            range=result.field_ranges.get(("input", field_name))
            or result.field_ranges.get(("field", "inputs"))
            or result.node_range,
        )


def _seed_text(value: Any) -> str:
    """Render a seed value as the right side of a typed binding.

    Args:
        value (`Any`): The seed's binding value: template or directive text, or a literal.

    Returns:
        `str`: Template and directive text as written; a literal or a value template
            in YAML flow style (a nested binding is shown by its value).
    """
    if isinstance(value, str) and ("${" in value or value.startswith("call(")):
        return value
    if isinstance(value, ValueTemplate):
        value = _untyped_tree(value.value)
    # JSON is YAML flow style, and a quoted string stays literal text.
    try:
        return json.dumps(value, default=str)
    except ValueError:
        # An int past Python's digit limit can't be written; the hint is a message.
        return value_repr(value)


def _untyped_tree(node: Any) -> Any:
    """Return a template tree with each nested binding replaced by its value."""
    if isinstance(node, Binding):
        return _untyped_tree(node.source)
    if type(node) is dict:
        return {key: _untyped_tree(item) for key, item in node.items()}
    if type(node) is list:
        return [_untyped_tree(item) for item in node]
    return node
