"""Infer canonical value contracts for Compose output bindings.

This module owns the Compose-specific envelope around the generic expression
engine: template shape, reference-to-contract lookup, record-field projection,
and provenance retention. Direct references re-export their exact or partial
contract; computed expressions receive a source-free exact contract only when
``expr`` proves their result type.

Knows about: ``expr`` AST result inference, semantic binding/contract carriers,
and source-neutral ``typesys`` values.
Never imports: YAML/source documents, node kinds, graph-stage passes, the loader,
or runtime.
"""

from __future__ import annotations

from dataclasses import replace
from math import isfinite
from typing import Any, Callable, Iterable

from lark import Token, Tree

from yian.expr import ExpressionError, eval_binding, expr_refs, infer_expr_type
from yian.expr.template import Span, scan_template, template_refs
from yian.nodes.contracts import (
    Binding,
    ContractOrigins,
    OutputContract,
    PartialRecordContract,
    ValueContract,
    ValueTemplate,
    binding_source,
    output_contract_field_error,
    output_contract_type,
)
from yian.typesys import TypeCheckError, build_value
from yian.typesys.values import (
    FieldRead,
    Type,
    ValueKind,
    file_field_type,
    list_kind_for_element,
    union_field,
)

NOT_LITERAL = object()
"""Marks a binding source that is a template, not a literal, in `literal_value`."""

TypeFn = Callable[[str], "Type | None"]
"""Resolves a `T(...)` constructor name to its type, or `None` for a name that is none."""


def project_output_contract(
    contract: OutputContract,
    fields: list[str],
) -> OutputContract | None:
    """Project a proven path from an exact or partial record output contract.

    A field read on a union projects to the members' shared field type; a field
    only some members have, or whose types differ, projects to `None`. A file's
    `uri`, `mime` and `name` project to their `file_field_type`. A projected type is
    `Optional` when anything on the path may be absent.
    """
    if type(contract) is PartialRecordContract:
        contract.validate_shape()
        if not fields:
            return contract
        field_name, *remaining = fields
        field_contracts = dict(contract.fields)
        if field_name not in field_contracts:
            return None
        field_contract = field_contracts[field_name]
        if field_contract is None:
            return None
        return project_output_contract(field_contract, remaining)
    if type(contract) is not ValueContract:
        raise TypeError(
            "output contract must be an exact ValueContract or "
            f"PartialRecordContract, got {type(contract).__name__}"
        )
    if not fields:
        return contract
    typ = contract.type
    path_nullable = typ.nullable
    for field in fields:
        if typ.variants is not None:
            read = union_field(typ, field)
            if not isinstance(read, FieldRead) or read.type is None:
                return None
            # The shared type is already nullable when any member's field may be absent.
            path_nullable = path_nullable or read.type.nullable
            typ = read.type
            continue
        if typ.fields is None:
            field_type = file_field_type(field) if typ.kind is ValueKind.FILE else None
            if field_type is None:
                return None
            path_nullable = path_nullable or field_type.nullable
            typ = field_type
            continue
        field_type = typ.fields.get(field)
        if field_type is None:
            return None
        path_nullable = (
            path_nullable
            or field not in (typ.required or frozenset())
            or field_type.nullable
        )
        typ = field_type
    if path_nullable and not typ.nullable:
        typ = replace(typ, nullable=True)
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


def reference_contract(
    path: str,
    producers: dict[str, OutputContract],
    flow_inputs: dict[str, ValueContract],
) -> OutputContract | None:
    """Resolve one `input.X...` or `<id>.output...` path to its exact or projected contract.

    Args:
        path (`str`): The dotted reference path.
        producers (`dict[str, OutputContract]`): Node output contracts by producer id.
        flow_inputs (`dict[str, ValueContract]`): Flow input contracts by name.

    Returns:
        `OutputContract | None`: The projected contract, or `None` for an unknown or
            unsupported path.

    Raises:
        `TypeError`: A referenced producer is not an `OutputContract`, or a flow-input
            contract is not an exact `ValueContract`.
    """
    parts = path.split(".")
    head = parts[0]
    if head == "input" and len(parts) >= 2:
        contract = flow_inputs.get(parts[1])
        if contract is not None and type(contract) is not ValueContract:
            raise TypeError(
                "flow input contracts must be exact ValueContract values, "
                f"got {type(contract).__name__} for {parts[1]!r}"
            )
        if contract is None:
            return None
        return project_output_contract(contract, parts[2:])
    if len(parts) < 2 or parts[1] != "output":
        return None

    producer_id = parts[0]
    contract = producers.get(producer_id)
    if contract is not None and type(contract) not in (
        ValueContract,
        PartialRecordContract,
    ):
        raise TypeError(
            "output producer contracts must be ValueContract or "
            "PartialRecordContract values, "
            f"got {type(contract).__name__} for {producer_id!r}"
        )
    if contract is None:
        return None
    return project_output_contract(contract, parts[2:])


def output_contract_path_error(
    contract: OutputContract,
    fields: list[str],
    path: str,
    *,
    partial_union_fields: bool = False,
) -> str | None:
    """Return the field-read error for an output path, else ``None``.

    The error is an undeclared record field, any field on a scalar or list, a field
    of a `file` other than `uri`, `mime` and `name`, or a field only some members of
    a union have (unless `partial_union_fields`).
    An opaque field inside a partial record remains lenient below that field,
    while the partial record's closed root name set is still enforced.
    """
    return output_contract_field_error(
        contract, fields, path, partial_union_fields=partial_union_fields
    )


def resolve_output_binding_contract(
    binding: Any,
    producers: dict[str, OutputContract],
    flow_inputs: dict[str, ValueContract],
    *,
    type_fn: TypeFn | None = None,
) -> OutputContract | None:
    """Resolve one Compose output binding to its canonical value contract.

    A typed binding (a ``Binding`` whose ``contract`` is set) resolves to its
    declared contract, since that type is what the binding guarantees at runtime.
    Otherwise a direct whole-value reference retains the referenced contract, including
    provenance.  An explicit null default ``${ref :- null}`` over such a reference retains
    it as ``Optional``; ``${(a | b) :- null}`` over references of one exact type receives a
    source-free ``Optional`` contract.  A computed whole-value expression receives a new
    source-free contract only when the expression engine proves its result type.  Embedded
    or multiple template spans always produce ``str``; malformed, literal-text, and
    unsupported computed bindings remain untyped.  A `T(...)` constructor has type
    ``T`` (``type_fn`` resolves it).  A value template's contract is inferred from its
    structure (``template_contract``), each string by these rules and each literal by
    its kind.

    Args:
        binding (`Any`):
            A raw binding value or exact semantic ``Binding`` carrier, possibly
            carrying a declared contract.
        producers (`dict[str, OutputContract]`):
            Exact or partial node output contracts keyed by producer id. A
            proven field of a partial record remains projectable.
        flow_inputs (`dict[str, ValueContract]`):
            Exact flow input contracts keyed by input name.
        type_fn (`TypeFn | None`, *optional*, defaults to `None`):
            Resolves a constructor name to its type; ``None`` leaves constructors
            untyped.

    Returns:
        `OutputContract | None`:
            The proven canonical exact or partial contract, or ``None`` when the
            binding's result shape is not statically proven.

    Raises:
        `TypeError`:
            A referenced producer is not an ``OutputContract``, or a flow-input
            contract is not an exact ``ValueContract``.
    """
    if isinstance(binding, Binding) and binding.contract is not None:
        return binding.contract
    source = binding_source(binding)
    if isinstance(source, ValueTemplate):
        return template_contract(
            source,
            lambda text: resolve_output_binding_contract(
                text, producers, flow_inputs, type_fn=type_fn
            ),
        )
    if not isinstance(source, str):
        return None
    try:
        segments = scan_template(source)
    except ExpressionError:
        return None
    spans = [segment for segment in segments if isinstance(segment, Span)]
    if not spans:
        return None
    refs = template_refs(segments)
    reference_contracts = {
        path: reference_contract(path, producers, flow_inputs) for path in refs
    }
    if not (len(segments) == 1 and isinstance(segments[0], Span)):
        return ValueContract(Type.scalar(ValueKind.STRING))

    expression = segments[0].tree
    if _is_direct_reference(expression):
        if len(refs) != 1:
            return None
        return reference_contracts[refs[0]]
    null_default = _null_default_contract(expression, reference_contracts)
    if null_default is not None:
        return null_default

    def reference_type(path: str) -> Type | None:
        """Project one referenced contract to the type requested by ``expr``."""
        contract = reference_contracts.get(path)
        if path not in reference_contracts:
            contract = reference_contract(path, producers, flow_inputs)
        return output_contract_type(contract)

    inferred = infer_expr_type(expression, reference_type, type_fn)
    return None if inferred is None else ValueContract(inferred)


def template_contract(
    template: ValueTemplate,
    string_contract: Callable[[str], OutputContract | None],
    *,
    exact_literals: bool = False,
) -> OutputContract | None:
    """Infer a value template's contract from its structure.

    A mapping is a closed record of its fields' contracts: exact when every field is
    exact, else partial. A list is a list of its items' joined type when every item
    is exact and the items are equal up to `Optional` (or become equal once literal
    refinements are widened); otherwise, and for an empty list, it is unproven. A
    nested typed binding has its declared contract, a template string the contract
    `string_contract` gives it, and a literal (including a string with no `${...}`
    span) its kind, as a top-level literal would leave it open to change.

    Args:
        template (`ValueTemplate`): The template to infer.
        string_contract (`Callable[[str], OutputContract | None]`): Infers one
            template string's contract.
        exact_literals (`bool`, *optional*, defaults to `False`): Give each literal
            its exact type (a one-tag `Literal`, one-point bounds), for checking the
            value against a declared type, where a wrong literal must not fit.

    Returns:
        `OutputContract | None`: The inferred exact or partial contract, or `None`
            when the value's shape is not proven.
    """
    return _tree_contract(template.value, string_contract, exact_literals)


def _tree_contract(
    node: Any,
    string_contract: Callable[[str], OutputContract | None],
    exact_literals: bool,
) -> OutputContract | None:
    """Infer the contract of one node of a template tree (see `template_contract`)."""
    if isinstance(node, Binding):
        if node.contract is not None:
            return node.contract
        return _tree_contract(node.source, string_contract, exact_literals)
    if type(node) is dict:
        return combine_output_contracts(
            {
                key: _tree_contract(item, string_contract, exact_literals)
                for key, item in node.items()
            }
        )
    if type(node) is list:
        return _list_contract(
            [_tree_contract(item, string_contract, exact_literals) for item in node]
        )
    literal = literal_value(node)
    if literal is NOT_LITERAL:
        return string_contract(node)
    typ = literal_type(literal)
    if typ is None:
        return None
    return ValueContract(typ if exact_literals else _widened(typ))


def literal_contract(value: Any) -> OutputContract | None:
    """Infer the contract of a span-free plain literal, as written out.

    A mapping is a closed record of its keys' contracts, a list is a list of its
    items' joined type (as in `template_contract`), and a scalar has its kind's
    widened type. A nested mapping with an unproven field, or a list whose items
    don't join, is opaque as a whole, since a partial record can't nest. Every
    string is `str`: the value is already read, so `$$` escapes are undone and a
    string is never scanned for `${...}` spans.

    Args:
        value (`Any`): The literal, with no template or binding inside it.

    Returns:
        `OutputContract | None`: The exact or partial contract, or `None` when the
            value's shape is unproven (a non-text key, which only a programmatic
            caller can pass, a non-finite number, a list whose items don't join).
    """
    # Not `_tree_contract`: it would undo `$$` escapes a second time and read a `${...}`
    # left by an escape as a span, but this value's strings are final text.
    if type(value) is dict:
        # A loaded flow's mapping keys are text; this serves programmatic callers, whose
        # mapping may have another key that no record field could name.
        if any(type(key) is not str for key in value):
            return None
        return combine_output_contracts(
            {key: literal_contract(item) for key, item in value.items()}
        )
    if type(value) is list:
        return _list_contract([literal_contract(item) for item in value])
    typ = literal_type(value)
    return None if typ is None else ValueContract(_widened(typ))


def _list_contract(items: list[OutputContract | None]) -> ValueContract | None:
    """Return the list contract of exact items with one joined type, else `None`."""
    # Unlike an inline body's strict element join, this leaves a list it can't join
    # unproven (the runtime check covers it) instead of failing the load.
    if not items or any(type(item) is not ValueContract for item in items):
        return None
    types = [item.type for item in items]
    element = _joined(types)
    if element is None:
        element = _joined([_widened(typ) for typ in types])
    if element is None:
        return None
    return ValueContract(Type(kind=list_kind_for_element(element), element=element))


def _joined(types: list[Type]) -> Type | None:
    """Return the types' common type when they are equal up to `Optional`, else `None`."""
    bare = [replace(typ, nullable=False) for typ in types]
    if any(typ != bare[0] for typ in bare[1:]):
        return None
    return replace(bare[0], nullable=any(typ.nullable for typ in types))


def _widened(typ: Type) -> Type:
    """Return `typ` with literal refinements (tags, bounds) dropped, at every depth."""
    return replace(
        typ,
        tags=None,
        bounds=None,
        fields=(
            None
            if typ.fields is None
            else {name: _widened(field) for name, field in typ.fields.items()}
        ),
        element=None if typ.element is None else _widened(typ.element),
    )


def literal_value(source: Any) -> Any:
    """Return the value a literal binding source binds, or `NOT_LITERAL` for a template.

    A string with no `${...}` span binds its text after `$$` unescaping, exactly as
    the binder evaluates it; a value template is a template; any other value is
    bound as is.

    Args:
        source (`Any`): The binding's raw source.

    Returns:
        `Any`: The bound literal value, or `NOT_LITERAL`.
    """
    if isinstance(source, ValueTemplate):
        return NOT_LITERAL
    if not isinstance(source, str):
        return source
    try:
        segments = scan_template(source)
        if any(isinstance(segment, Span) for segment in segments):
            return NOT_LITERAL
        return eval_binding(source, lambda path: None)
    except ExpressionError:
        return NOT_LITERAL  # malformed text is reported by reference validation


def literal_type(value: Any) -> Type | None:
    """Return the exact type of a literal value, else `None`.

    Args:
        value (`Any`): The bound literal.

    Returns:
        `Type | None`: A one-tag `Literal` for non-empty text, one-point bounds for a
            finite number, the value's kind otherwise; `None` for a non-finite number
            or a value with no type.
    """
    if type(value) is int:
        return Type(kind=ValueKind.INTEGER, bounds=(value, value))
    if type(value) is float:
        if not isfinite(value):
            return None
        return Type(kind=ValueKind.NUMBER, bounds=(value, value))
    if type(value) is str:
        # A tag must be non-empty, so empty text is plain str.
        if not value:
            return Type.scalar(ValueKind.STRING)
        return Type(kind=ValueKind.STRING, tags=frozenset({value}))
    try:
        return Type.scalar(build_value(value).kind)
    except TypeCheckError:
        return None


def _null_default_contract(
    expression: Any,
    reference_contracts: dict[str, OutputContract | None],
) -> ValueContract | None:
    """Type `head :- null` as Optional when `head` is a reference or a coalesce of them.

    Args:
        expression (`Any`): The parsed tree of a template's single span.
        reference_contracts (`dict[str, OutputContract | None]`):
            The resolved contract of every reference path the span reads.

    Returns:
        `ValueContract | None`:
            The head reference's exact contract made nullable with its origins, a
            source-free nullable contract for a coalesce of references of one exact
            type, or ``None`` for any other expression.
    """
    if not (
        getattr(expression, "data", None) == "default_expr"
        and len(expression.children) == 2
        and getattr(expression.children[1], "type", None) == "NULL"
    ):
        return None
    head = expression.children[0]
    if getattr(head, "data", None) == "coalesce":
        operands = [
            _direct_value_contract(operand, reference_contracts)
            for operand in head.children
        ]
        if any(operand is None for operand in operands):
            return None
        types = [replace(operand.type, nullable=False) for operand in operands]
        if any(typ != types[0] for typ in types[1:]):
            return None
        return ValueContract(replace(types[0], nullable=True))
    contract = _direct_value_contract(head, reference_contracts)
    if contract is None:
        return None
    return replace(contract, type=replace(contract.type, nullable=True))


def _direct_value_contract(
    operand: Any,
    reference_contracts: dict[str, OutputContract | None],
) -> ValueContract | None:
    """Return the exact contract of an operand that is one direct reference.

    Args:
        operand (`Any`): One parsed operand of a span's expression tree.
        reference_contracts (`dict[str, OutputContract | None]`):
            The resolved contract of every reference path the span reads.

    Returns:
        `ValueContract | None`:
            The referenced exact contract, or ``None`` when the operand is not a lone
            reference or its contract is partial or unknown.
    """
    if not _is_direct_reference(operand):
        return None
    contract = reference_contracts.get(expr_refs(operand)[0])
    return contract if type(contract) is ValueContract else None


def direct_reference_path(binding: Any) -> str | None:
    """Return the path a binding reads when it is one lone whole-value reference.

    Args:
        binding (`Any`): A raw binding value or semantic ``Binding`` carrier.

    Returns:
        `str | None`: The dotted reference path (``item``, ``item.name``,
            ``a.output``), or ``None`` for a literal, an embedded or computed
            template, a call, or malformed text.
    """
    source = binding_source(binding)
    if not isinstance(source, str):
        return None
    try:
        segments = scan_template(source)
    except ExpressionError:
        return None
    if not (len(segments) == 1 and isinstance(segments[0], Span)):
        return None
    if not _is_direct_reference(segments[0].tree):
        return None
    refs = template_refs(segments)
    return refs[0] if len(refs) == 1 else None


def _is_direct_reference(expression: Any) -> bool:
    """Return whether a whole-value span is a bare reference, not a call or operator.

    Args:
        expression (`Any`): The parsed tree of a template's single span.

    Returns:
        `bool`: ``True`` for a bare or dotted reference with no call suffix.
    """
    return getattr(expression, "data", None) == "refcall" and not any(
        getattr(child, "data", None) == "call_suffix"
        for child in getattr(expression, "children", ())
    )


def _origin_path_sort_key(
    entry: tuple[tuple[str | int, ...], Any],
) -> tuple[tuple[int, str | int], ...]:
    """Return the same total path ordering required by ``ContractOrigins``."""
    return tuple(
        (0, segment) if type(segment) is str else (1, segment) for segment in entry[0]
    )


def combine_output_contracts(
    fields: dict[str, OutputContract | None],
) -> OutputContract:
    """Combine named outputs into one exact or partial closed-record contract.

    Args:
        fields (`dict[str, OutputContract | None]`):
            Output field names mapped to proven contracts or ``None`` for opaque
            fields. A nested partial contract cannot be represented inside one
            root field and is therefore retained as opaque at this boundary.
            Insertion order is retained.

    Returns:
        `OutputContract`:
            One exact closed record when every field is exact; otherwise one
            partial record that retains the closed name set and every exact
            sibling contract.
    """
    for name, contract in fields.items():
        if contract is not None and type(contract) not in (
            ValueContract,
            PartialRecordContract,
        ):
            raise TypeError(
                "output fields must contain ValueContract, PartialRecordContract, "
                f"or None values, got {type(contract).__name__} for {name!r}"
            )
    if any(type(contract) is not ValueContract for contract in fields.values()):
        return PartialRecordContract(
            tuple(
                (
                    name,
                    contract if type(contract) is ValueContract else None,
                )
                for name, contract in fields.items()
            )
        )

    return ValueContract(
        type=Type(
            kind=ValueKind.OBJECT,
            fields={name: contract.type for name, contract in fields.items()},
            required=frozenset(fields),
        ),
        origins=record_field_origins(fields.items()),
    )


def record_field_origins(
    fields: Iterable[tuple[str, ValueContract | None]],
) -> ContractOrigins:
    """Index each field contract's origins under its field name, as one record's origins.

    Args:
        fields (`Iterable[tuple[str, ValueContract | None]]`):
            Record field names paired with exact contracts; `None` (an opaque field)
            contributes no origin.

    Returns:
        `ContractOrigins`:
            No root origin; each field's root origin at `(name,)` and its nested origins
            below it, in canonical path order.
    """
    origins: list[tuple[tuple[str | int, ...], Any]] = []
    for name, contract in fields:
        if contract is None:
            continue
        if contract.origins.default is not None:
            origins.append(((name,), contract.origins.default))
        origins.extend(
            ((name, *path), origin) for path, origin in contract.origins.entries
        )
    origins.sort(key=_origin_path_sort_key)
    return ContractOrigins(entries=tuple(origins))


def direct_reference_parts(expression: Tree | Token) -> tuple[str, ...] | None:
    """Return the path of a lone plain reference, excluding calls and expressions.

    Args:
        expression (`Tree | Token`):
            One parsed `${...}` interior, as `Span.tree`.

    Returns:
        `tuple[str, ...] | None`:
            The dotted path's names, such as `("d", "output", "action")`, or `None`
            when the expression is not a plain `refcall` of names (an index, a call,
            an escape such as `:-`, or any operator).
    """
    if not isinstance(expression, Tree) or expression.data != "refcall":
        return None

    children = expression.children
    if (
        not children
        or not isinstance(children[0], Token)
        or str(children[0].type) != "NAME"
    ):
        return None

    parts = [str(children[0])]
    for child in children[1:]:
        if (
            not isinstance(child, Tree)
            or child.data != "trailer"
            or len(child.children) != 1
            or not isinstance(child.children[0], Token)
            or str(child.children[0].type) != "NAME"
        ):
            return None
        parts.append(str(child.children[0]))
    return tuple(parts)


__all__ = [
    "NOT_LITERAL",
    "combine_output_contracts",
    "direct_reference_parts",
    "literal_type",
    "literal_value",
    "output_contract_path_error",
    "project_output_contract",
    "reference_contract",
    "record_field_origins",
    "reference_contract",
    "resolve_output_binding_contract",
    "template_contract",
    "TypeFn",
]
