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
Each proven returned field must also fit the body parameter it is bound to on the next
iteration (a named body's declared input, or the parameter an inline body generates from
its seed), read as nullable where the loop's `while:` proves the field not null; that
error points at the seed binding.
A literal seed stays lenient here (its contract is not inferred); `check_call_map_types`
checks every seed against the body's declared input.

Imports flow down only: `compile.validation` (the type comparator),
`compose.helpers` (shared output-contract inference and load diagnostics), and
`compose.nodes.loop` (a body's inputs as the carried values enter them, and the
carried field types). Only the loader (and the compose package facade) import this back.
"""

from __future__ import annotations

import json
from collections.abc import Collection
from dataclasses import replace
from typing import Any, Optional

from yian.compile.validation import types_compatible
from yian.compose.helpers.errors import LoadDiagnostics
from yian.compose.helpers.isinstance_types import constructor_type_fn
from yian.compose.helpers.tags import any_source_hint, suggest_tag, tag_text
from yian.compose.helpers.output_contracts import (
    assert_exact_contracts,
    resolve_output_binding_contract,
)
from yian.compose.nodes.base import BuildResult
from yian.compose.nodes.loop import carried_field_types, loop_entry_inputs
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
from yian.typesys.values import Type, ValueKind


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

    Each returned field is bound to the body's parameter again on the next iteration (a
    named body's declared input, or the parameter an inline body generates from its seed),
    so every statically proven returned field must also fit that parameter, read as
    nullable where the loop's `while:` proves the field not null. Over a typed seed, a
    nullable returned field whose non-null type fits is left to the runtime input check.

    Args:
        build_results (`dict[str, BuildResult]`): The built nodes keyed by node id.
        producers (`dict[str, OutputContract]`): Each producer node's output contract.
        flow_inputs (`dict[str, ValueContract]`): The flow's declared input contracts.
        flow_wiring (`dict[str, dict[str, Any]]`): Each node's `param -> source` bindings.
        inline_loops (`Collection[str]`, *optional*, defaults to `()`): The ids of loop
            nodes whose body is written inline; it selects the wording of the error for
            a returned field that does not fit its parameter.
        registry (`TypeRegistry | None`, *optional*, defaults to `None`): The flow's
            typedefs, used to spell a suggested seed tag by a typedef name and to type
            a `T(...)` constructor seed as `T`.
        diagnostics (`LoadDiagnostics | None`, *optional*, defaults to `None`): Factory
            for located load errors.

    Raises:
        `LoadError`: A returned field does not keep its carried type (with advice to
            tag the seed or narrow it with `isinstance` when it is `Any`), or the body
            returns a field that cannot fit the parameter it is bound to next.
        `TypeError`: A producer, flow input, or loop contract has an unexpected type."""
    diagnostics = diagnostics or LoadDiagnostics()
    type_fn = constructor_type_fn(registry) if registry is not None else None
    assert_exact_contracts(producers, flow_inputs)
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
        carried_fields = carried_field_types(loop_contract)
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
        _check_returned_fields(
            result,
            carried_fields,
            flow_wiring.get(nid, {}),
            registry or {},
            diagnostics,
            inline=nid in inline_loops,
        )


def _check_returned_fields(
    result: BuildResult,
    carried_fields: dict[str, Type],
    wiring: dict[str, Any],
    registry: TypeRegistry,
    diagnostics: LoadDiagnostics,
    *,
    inline: bool,
) -> None:
    """Require each returned field of a loop body to fit the parameter it is bound to next.

    The parameter is the body's declared input for a named body, or the one generated
    from the seed for an inline body. A path the loop's `while:` proves not null is never
    null when the body starts, so the parameter is read as nullable there and at each
    record above it (see `loop_entry_inputs`); the error still names the declared type.
    Otherwise, over a typed seed, a nullable returned field whose non-null type fits the
    parameter is accepted; a null it returns fails the next iteration's input check.

    Args:
        result (`BuildResult`): The built loop node and its source ranges.
        carried_fields (`dict[str, Type]`): The statically proven carried field types;
            a returned field's entry is the body's own field type.
        wiring (`dict[str, Any]`): The loop's seed bindings keyed by carried field.
        registry (`TypeRegistry`): The flow's typedefs, for the suggested types and tags.
        diagnostics (`LoadDiagnostics`): Factory for located load errors.
        inline (`bool`): Whether the body is written inline, which selects the wording.

    Raises:
        `LoadError`: A returned field's type cannot fit its parameter; the error names
            both types and says how to make them agree.
    """
    node = result.node
    returned = node.returned_fields
    parameters = {
        parameter.name: parameter.contract
        for parameter in node.child_inputs or ()
        if parameter.contract is not None
    }
    # The comparison reads the inputs as values enter the body; messages name the
    # declared input, which the body's start checks values against.
    entry_types = {
        parameter.name: parameter.contract.type
        for parameter in loop_entry_inputs(node)
        if parameter.contract is not None
    }
    for field_name, seed in wiring.items():
        if returned is not None and field_name not in returned:
            continue  # omitted by the body: never bound to its parameter again
        body_field = carried_fields.get(field_name)
        parameter = parameters.get(field_name)
        if body_field is None or parameter is None:
            continue  # opaque on either side, or not a body input: nothing to compare
        entry_type = entry_types[field_name]
        if types_compatible(body_field, entry_type):
            continue
        typed = isinstance(seed, Binding) and seed.contract is not None
        non_null_fits = types_compatible(replace(body_field, nullable=False), entry_type)
        # The next iteration's input check rejects a null.
        if typed and non_null_fits:
            continue
        returned_label = canonical_type_label(body_field)
        parameter_label = canonical_type_label(parameter.type)
        seed_text = _seed_text(binding_source(seed))
        if inline:
            message = (
                f"loop node {node.id!r}: the body returns {field_name!r} as {returned_label}, "
                "but the carried field is "
                + (
                    f"declared {parameter_label}"
                    if typed
                    else f"typed {parameter_label} from its seed"
                )
                + "; declare the seed's type: "
                + suggest_tag(field_name, body_field, seed_text, registry=registry)
            )
        else:
            message = (
                f"loop node {node.id!r}: the body {node.flow_id!r} returns {field_name!r} "
                f"as {returned_label}, but takes {field_name!r} as {parameter_label}; "
            )
            if non_null_fits:
                # Only an untyped seed gets here, so tagging it is a real alternative.
                nullable = replace(parameter.type, nullable=True)
                nullable_label = tag_text(
                    nullable, registry=registry
                ) or canonical_type_label(nullable)
                message += (
                    f"make the input {nullable_label}, or tag the seed to leave the check "
                    "to runtime: "
                    + suggest_tag(field_name, parameter.type, seed_text, registry=registry)
                )
            elif body_field.kind is ValueKind.ANY:
                message += (
                    "narrow it in the body with isinstance in a case, "
                    "or declare its producer's type"
                )
            else:
                message += "change the body's input or output type so they agree"
        raise diagnostics.error(
            message,
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
    return json.dumps(value, default=str)


def _untyped_tree(node: Any) -> Any:
    """Return a template tree with each nested binding replaced by its value."""
    if isinstance(node, Binding):
        return _untyped_tree(node.source)
    if type(node) is dict:
        return {key: _untyped_tree(item) for key, item in node.items()}
    if type(node) is list:
        return [_untyped_tree(item) for item in node]
    return node
