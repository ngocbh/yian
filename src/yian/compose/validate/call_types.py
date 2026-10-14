"""Graph-stage: the cross-flow TYPE check for `call`/`map` bindings and constant `loop` seeds.

A loader post-build pass (run after the flow's leaves + child signatures are assembled).
Each binding's SOURCE contract is resolved from its reference over the PARENT producer +
flow-input contracts (a MAP's `${item...}` from the element type of `over:`), then its
`Type` is checked with `types_compatible` against the CHILD input's declared `Type` (a
LOOP's body input for a seed). Only a LOOP's untyped seeds that read no reference are
checked. A literal, a constant operand of a `:-`/`|` escape, or each element of a literal
`over:` list read through `${item...}`, is validated by value instead, as the child's
start validates it: a CALL/MAP/LOOP edge is not a flow boundary, so text is never
converted to a number or bool across it. Its range anchor comes from the call, map, or
loop node's transient Compose `BuildResult.field_ranges[("field", "inputs")]` (the coarse
`input:` mapping span).

Imports flow down only: `compile.validation` (the type comparator), `expr` (the binding-null
analysis and escape parsing), `compose.helpers` (shared output-contract inference and load
diagnostics), `compose.validate.typed_bindings` (literal and `${item}` helpers), and
`nodes.binding` (the run boundary's string conversion, named in a hint only).
Only the loader (and the compose package facade) import this back.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date
from math import isfinite
from typing import Any, Optional

from lark import Token, Tree

from yian.compile.validation import types_compatible
from yian.compose.helpers.errors import LoadDiagnostics
from yian.compose.helpers.output_contracts import (
    TypeFn,
    direct_reference_parts,
    project_output_contract,
    reference_contract,
    resolve_output_binding_contract,
)
from yian.compose.helpers.tags import yaml_quoted
from yian.compose.helpers.yaml_words import former_boolean_fix
from yian.compose.nodes.base import BuildResult
from yian.compose.validate.narrowing import Narrowing
from yian.compose.validate.typed_bindings import (
    NOT_LITERAL,
    item_contract,
    literal_value,
    shown_literal,
)
from yian.diagnostics import SourceRange
from yian.expr import (
    ExpressionError,
    binding_co_skips,
    eval_expr,
    expr_refs,
    expr_refs_of,
    scan_template,
)
from yian.expr.template import Span
from yian.nodes.binding import coerce_param
from yian.nodes.contracts import (
    Binding,
    OutputContract,
    Parameter,
    PartialRecordContract,
    ValueContract,
    binding_source,
    iter_source_strings,
)
from yian.typesys import Type, TypeCheckError, build_value_as, canonical_type_label


def check_call_map_types(
    build_results: dict[str, BuildResult],
    producers: dict[str, OutputContract],
    flow_inputs: dict[str, ValueContract],
    flow_wiring: dict[str, dict[str, Any]],
    *,
    narrowing: Narrowing | None = None,
    type_fn: TypeFn | None = None,
    diagnostics: LoadDiagnostics | None = None,
) -> None:
    """Cross-flow type check for every `call`/`map` binding (a loader post-build pass).

    Each binding must fit the CHILD input's declared `Type`, read over the PARENT
    producer + flow-input contracts as the node's `narrowing` view sees them when given:
    a reference by its resolved contract under `types_compatible` (a lone `${item...}`
    through the element type of a proven `over:` list), a literal by value, and a
    `${ref :- fallback}`/`${ref :? message}` by its de-nulled reference and its
    fallback. An unproven source, or an opaque/absent child input (incl. the reserved
    `over` key), stays lenient (skipped). Loud + located at the node's Compose-owned
    `input:` mapping range, else its authored node range. Iterates the WIRING (not the
    params) so a mis-named binding still surfaces here when its child decl is absent
    rather than being silently skipped. A LOOP's untyped seeds that read no reference
    are checked the same way against its body's inputs; its typed and reference seeds
    are not checked against the body's inputs. ``type_fn`` types a `T(...)` constructor
    source as `T`."""
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
        # CALL/MAP/LOOP carry `child_inputs` to type-check against. A LOOP's seeds fill its
        # body's START like CALL arguments, but only an untyped seed that reads no reference
        # is checked here. Checking reference and typed seeds would bring CALL's `Optional`
        # rule to LOOP, a separate change; check_loop_type_contract checks returned fields
        # against the body output.
        # A leaf has no `child_inputs` attr, and a zero-input child gives [] (falsy), so skip.
        if not getattr(node, "child_inputs", None):
            continue
        kind = "loop" if node.is_loop else "call"
        child_decls = {d.name: d for d in node.child_inputs}
        node_producers, node_inputs = (
            narrowing.for_node(nid) if narrowing is not None else (producers, flow_inputs)
        )
        wiring = flow_wiring.get(nid, {})
        scope = _Scope(
            node_producers,
            node_inputs,
            item_contract(result, wiring, node_producers, node_inputs),
            _literal_items(result, wiring),
            type_fn,
        )
        for param, src in wiring.items():
            decl = child_decls.get(param)
            if decl is None or decl.contract is None:
                continue  # `over`/name-caught-in-build/opaque child input -> lenient
            if node.is_loop and not _constant_seed(src):
                continue
            message = _binding_mismatch(src, decl, scope)
            if message is not None:
                rng: Optional[SourceRange] = (
                    result.field_ranges.get(("field", "inputs"))
                    or result.node_range
                )
                raise diagnostics.error(
                    f"{kind} node {nid!r}: binding {param!r} — {message}",
                    range=rng,
                )


@dataclass(frozen=True)
class _Scope:
    """What a CALL/MAP/LOOP node's bindings read, for the load check.

    Attributes:
        producers (`dict[str, OutputContract]`): Node output contracts by producer id.
        flow_inputs (`dict[str, ValueContract]`): Flow input contracts by name.
        item (`ValueContract | None`): The element contract `${item}` reads on a MAP
            over a proven list, else `None`.
        items (`tuple[Any, ...] | None`): The elements of a MAP's literal `over:`
            list, which `${item}` reads by value, else `None`.
        type_fn (`TypeFn | None`): Types a `T(...)` constructor source as `T`.
    """

    producers: dict[str, OutputContract]
    flow_inputs: dict[str, ValueContract]
    item: ValueContract | None
    items: tuple[Any, ...] | None
    type_fn: TypeFn | None = None


def _literal_items(result: BuildResult, wiring: dict[str, Any]) -> tuple[Any, ...] | None:
    """Return the elements of a MAP's literal `over:` list, else `None`.

    Args:
        result (`BuildResult`): The built node.
        wiring (`dict[str, Any]`): The node's binding map.

    Returns:
        `tuple[Any, ...] | None`: The literal list's elements, or `None` when the node
            has no item scope or its `over:` is not a literal list.
    """
    key = getattr(result.node, "item_source_key", None)
    if key is None or key not in wiring:
        return None
    over = literal_value(binding_source(wiring[key]))
    return tuple(over) if isinstance(over, list) else None


def _binding_mismatch(src: Any, decl: Parameter, scope: _Scope) -> str | None:
    """Return why one binding can't fill the child input `decl`, else `None`.

    A literal is validated by value with no string conversion, as the child's start
    validates it: a quoted date fits a `date` input but `"3"` does not fit an `int`
    input. An unquoted YAML date, which YAML reads as a date and not text, is rejected
    too, as it is in a typed binding or a default. A lone `${item...}`, a
    `${ref :- fallback}` or `${ref :? message}` over a reference, and a `|` coalesce are
    checked operand by operand (see `_operand_mismatch`). Any other source is checked
    by its resolved contract.

    Args:
        src (`Any`): The binding, raw or a `Binding`.
        decl (`Parameter`): The child input, with a declared contract.
        scope (`_Scope`): What the node's bindings read.

    Returns:
        `str | None`: The mismatch, without the node and binding names, or `None` when
            the binding fits or its type is unproven.
    """
    child_type = decl.contract.type
    source = binding_source(src)
    if not (isinstance(src, Binding) and src.contract is not None):
        literal = literal_value(source)
        if literal is not NOT_LITERAL:
            return _value_mismatch("value", literal, decl)
        tree = _lone_span(source)
        path = None if tree is None else _operand_path(tree)
        kind = getattr(tree, "data", None)
        if (path is not None and path.split(".")[0] == "item") or (
            kind in ("default_expr", "required_expr")
            and _operand_path(tree.children[0]) is not None
        ):
            return _operand_mismatch(tree, decl, scope, non_null=False)
        if kind == "coalesce":
            message = _operand_mismatch(tree, decl, scope, non_null=False)
            if message is not None:
                return message
    # A nullable source is fine for a non-nullable child input ONLY when the binding
    # GUARANTEES a non-null value — a non-null literal operand (`${a | 5}`) or text
    # around a span. A present-null source SHADOWS the child's own `default:`
    # (apply_defaults fills only OMITTED inputs, never a bound null), so it doesn't
    # cover a null. `not binding_co_skips` is exactly that non-null guarantee.
    return _contract_mismatch(
        resolve_output_binding_contract(
            src, scope.producers, scope.flow_inputs, type_fn=scope.type_fn
        ),
        child_type,
        non_null=not binding_co_skips(source),
    )


def _lone_span(source: Any) -> Tree | Token | None:
    """Return the parsed expression of a source that is one whole `${...}`, else `None`."""
    if not isinstance(source, str):
        return None
    try:
        segments = scan_template(source)
    except ExpressionError:
        return None
    if not (len(segments) == 1 and isinstance(segments[0], Span)):
        return None
    return segments[0].tree


def _constant_seed(src: Any) -> bool:
    """Return whether a binding is untyped and reads no reference.

    Each string nested in a value template counts as a read of its own.
    """
    if isinstance(src, Binding) and src.contract is not None:
        return False
    try:
        return not any(expr_refs_of(text) for _, text in iter_source_strings(src))
    except ExpressionError:
        return False  # malformed text is reported by reference validation


def _operand_path(node: Tree | Token) -> str | None:
    """Return the dotted path of a plain reference operand (`a.b`), else `None`."""
    parts = direct_reference_parts(node)
    return None if parts is None else ".".join(parts)


def _operand_mismatch(
    node: Tree | Token, decl: Parameter, scope: _Scope, *, non_null: bool
) -> str | None:
    """Return why one operand of a lone `${...}` can't fill `decl`, else `None`.

    A reference is checked by its contract (`item...` through the element type of
    `over:`, or each element of a literal `over:` by value). `head :- fallback` checks
    `head` without its null, then `fallback`; `head :? message` checks `head` without
    its null; `a | b | c` checks every operand but the last without its null. A
    constant is checked by value. A computed operand is left to the child's start.

    Args:
        node (`Tree | Token`): The parsed operand.
        decl (`Parameter`): The child input, with a declared contract.
        scope (`_Scope`): What the node's bindings read.
        non_null (`bool`): Whether a null from this operand falls through to another
            one, so only its non-null values reach the child.

    Returns:
        `str | None`: The mismatch, or `None` when it fits or is unproven.
    """
    path = _operand_path(node)
    if path is not None:
        return _reference_mismatch(path, decl, scope, non_null=non_null)
    kind = getattr(node, "data", None)
    if kind == "default_expr":
        head, fallback = node.children
        return _operand_mismatch(head, decl, scope, non_null=True) or _operand_mismatch(
            fallback, decl, scope, non_null=non_null
        )
    if kind == "required_expr":
        return _operand_mismatch(node.children[0], decl, scope, non_null=True)
    if kind == "coalesce":
        *heads, last = node.children
        for head in heads:
            message = _operand_mismatch(head, decl, scope, non_null=True)
            if message is not None:
                return message
        return _operand_mismatch(last, decl, scope, non_null=non_null)
    if expr_refs(node):
        return None
    try:
        value = eval_expr(node, lambda _path: None)
    except ExpressionError:
        return None  # a malformed constant fails when the binding is evaluated
    if value is None and non_null:
        return None
    # Only a single token was written as the value shown; a computed constant such as
    # `2026-01-01` (arithmetic here) gets no rewrite hint.
    text = str(node) if isinstance(node, Token) and node.type == "NUMBER" else None
    return _value_mismatch(
        "default", value, decl, hint=isinstance(node, Token), text=text
    )


def _reference_mismatch(
    path: str, decl: Parameter, scope: _Scope, *, non_null: bool
) -> str | None:
    """Return why the value one reference path reads can't fill `decl`, else `None`.

    Args:
        path (`str`): The dotted reference path.
        decl (`Parameter`): The child input, with a declared contract.
        scope (`_Scope`): What the node's bindings read.
        non_null (`bool`): Whether a null read here falls through to another operand.

    Returns:
        `str | None`: The mismatch, or `None` when it fits or is unproven.
    """
    head, *fields = path.split(".")
    if head != "item":
        contract = reference_contract(path, scope.producers, scope.flow_inputs)
        return _contract_mismatch(contract, decl.contract.type, non_null=non_null)
    if scope.item is not None:
        contract = project_output_contract(scope.item, fields)
        return _contract_mismatch(contract, decl.contract.type, non_null=non_null)
    for element in scope.items or ():
        value = element
        for field in fields:
            if not isinstance(value, dict) or field not in value:
                break  # a missing field fails when the element is bound
            value = value[field]
        else:
            if value is None and non_null:
                continue
            message = _value_mismatch("element", value, decl)
            if message is not None:
                return message
    return None


def _contract_mismatch(
    contract: OutputContract | None, child_type: Type, *, non_null: bool
) -> str | None:
    """Return why a source contract can't fill `child_type`, else `None`.

    Args:
        contract (`OutputContract | None`): The source's resolved contract.
        child_type (`Type`): The child input's declared type.
        non_null (`bool`): Whether the binding guarantees a non-null value, so a
            nullable source is compared without its null.

    Returns:
        `str | None`: The mismatch, or `None` when it fits or is unproven (`None`
            contract, or a whole partial record with no exact type).
    """
    if contract is None or type(contract) is PartialRecordContract:
        return None
    if type(contract) is not ValueContract:
        raise TypeError(
            "resolved source contract must be ValueContract, "
            "PartialRecordContract, or None"
        )
    source = contract.type
    if source.nullable and not child_type.nullable and non_null:
        source = replace(source, nullable=False)
    if types_compatible(source, child_type):
        return None
    return (
        f"child expects {canonical_type_label(child_type)!r}, source is "
        f"{canonical_type_label(source)!r}"
    )


def _value_mismatch(
    what: str,
    value: Any,
    decl: Parameter,
    *,
    hint: bool = True,
    text: str | None = None,
) -> str | None:
    """Return why a literal can't fill the child input `decl`, else `None`.

    Args:
        what (`str`): What the literal is in the binding (`value`, `default`, or
            `element`).
        value (`Any`): The literal.
        decl (`Parameter`): The child input, with a declared contract.
        hint (`bool`, *optional*, defaults to `True`): Whether to suggest a rewrite;
            off for a computed constant, whose text is not one literal.
        text (`str | None`, *optional*, defaults to `None`): The literal as written,
            when it differs from how `shown_literal` renders the value.

    Returns:
        `str | None`: The mismatch, with a hint when rewriting the literal would fix
            it, or `None` when the value fits.
    """
    child_type = decl.contract.type
    try:
        build_value_as(child_type, value)
    except TypeCheckError as exc:
        message = (
            f"child expects {canonical_type_label(child_type)!r}, but the {what} "
            f"{text or shown_literal(value)} does not fit"
        )
        if exc.path:
            message += f" ({exc})"
        suggestion = _literal_hint(value, decl, text, yaml=what != "default") if hint else None
        return message if suggestion is None else f"{message}; {suggestion}"
    return None


def _literal_hint(
    value: Any, decl: Parameter, text: str | None, *, yaml: bool
) -> str | None:
    """Suggest how to rewrite a literal that doesn't fit `decl`, else `None`.

    A YAML 1.1 boolean spelling such as `yes` gets the `true`/`false` to write when a
    boolean fits. Other text that a run boundary would convert to a finite number or a
    bool is written unquoted instead. A value YAML read as a non-string, such as an
    unquoted number or date, or a list or record holding dates, is quoted when its text
    would fit.

    Args:
        value (`Any`): The literal that failed.
        decl (`Parameter`): The child input, with a declared contract.
        text (`str | None`): The literal as written, if known.
        yaml (`bool`): Whether YAML read the literal, rather than the `${...}`
            expression syntax; only a YAML literal gets the note on how YAML reads
            booleans.

    Returns:
        `str | None`: The hint, or `None` when rewriting the literal can't help.
    """
    child_type = decl.contract.type
    fix = former_boolean_fix(value, child_type)
    if fix is not None:
        return fix[1]
    if isinstance(value, str):
        candidate = coerce_param(replace(decl, coerce_strings=True), value)
        if isinstance(candidate, str) or not _fits(child_type, candidate):
            return None
        # YAML reads an unquoted `nan` or `inf` as text, so writing it doesn't help.
        if isinstance(candidate, float) and not isfinite(candidate):
            return None
        written = shown_literal(build_value_as(child_type, candidate).to_object())
        return f"text is not converted between flows, so write {written} without quotes"
    if isinstance(value, bool):
        if not _fits(child_type, "true"):
            return None
        if not yaml:
            return "quote it to keep it as text"
        return (
            "quote it to keep it as text "
            "(YAML reads unquoted true/false as booleans)"
        )
    if isinstance(value, (int, float, date)):
        written = text or shown_literal(value)
        if not _fits(child_type, written):
            return None
        return f"quote it to keep it as text: {yaml_quoted(written)}"
    if isinstance(value, (list, dict)):
        quoted = _dates_as_text(value)
        if quoted == value or not _fits(child_type, quoted):
            return None
        return "quote each date to keep it as text"
    return None


def _dates_as_text(value: Any) -> Any:
    """Return `value` with every date or datetime inside it replaced by its ISO text."""
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, list):
        return [_dates_as_text(item) for item in value]
    if isinstance(value, dict):
        return {key: _dates_as_text(item) for key, item in value.items()}
    return value


def _fits(child_type: Type, value: Any) -> bool:
    """Return whether `value` validates against `child_type`."""
    try:
        build_value_as(child_type, value)
    except TypeCheckError:
        return False
    return True
