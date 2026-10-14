"""Strict type inference for generated inline-composite input declarations.

Inline composite bodies become private child flows, so every outer binding must
produce a concrete child-parameter type. This module infers only types that the
Compose source proves structurally; it never falls back to the runtime value
builder's lenient ``Any`` behavior. Top-level strings and the leaf strings of a
value template are binding templates; a span-free literal container's strings are
ordinary strings.

Knows about: ``expr`` template syntax, node-owned semantic binding/output-contract
carriers, ``typesys`` structural types, and Compose source ranges/errors.
Never imports: ``compose.loader``, ``compile``, ``runtime``, or node kinds.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import Any

from lark import Token, Tree

from yian.compose.helpers.errors import LoadDiagnostics, LoadError
from yian.compose.helpers.output_contracts import (
    TypeFn,
    direct_reference_parts,
    output_contract_path_error,
    project_output_contract,
)
from yian.diagnostics import SourceRange
from yian.expr.builtins import TEMPLATE_FNS
from yian.expr.expressions import ISINSTANCE, ExpressionError
from yian.expr.template import Span, scan_template
from yian.nodes.contracts import (
    Binding,
    OutputContract,
    PartialRecordContract,
    ValueContract,
    ValueTemplate,
    binding_source,
)
from yian.typesys.values import (
    Type,
    ValueKind,
    canonical_type_label,
    list_kind_for_element,
    union_field,
)


def explicit_type_hint(binding_label: str) -> str:
    """Return the advice for an inline binding whose type can't be proven.

    Args:
        binding_label (`str`): The binding's node/input path, such as
            ``"each.input.text"``; its last segment names the slot.

    Returns:
        `str`: Advice to tag the value with its type or use a typed named body.
    """
    name = binding_label.rsplit(".", 1)[-1]
    return (
        f"declare its type with a tag (`{name}: !<type> ...`) or use an "
        "explicitly typed named `defs:` body"
    )

ProducerType = Callable[[str, str, SourceRange | None], Type | OutputContract]
ItemType = Callable[[str, SourceRange | None], Type]

_SCALAR_TOKEN_KINDS: dict[str, ValueKind] = {
    "NULL": ValueKind.NONE,
    "BOOL": ValueKind.BOOLEAN,
    "STRING": ValueKind.STRING,
}

_LIST_KINDS: dict[ValueKind, ValueKind] = {
    ValueKind.STRING: ValueKind.LIST_STRING,
    ValueKind.INTEGER: ValueKind.LIST_INTEGER,
    ValueKind.NUMBER: ValueKind.LIST_NUMBER,
    ValueKind.BOOLEAN: ValueKind.LIST_BOOLEAN,
    ValueKind.OBJECT: ValueKind.LIST_OBJECT,
}


def infer_inline_binding_type(
    value: Any,
    *,
    parent_input_types: Mapping[str, Type],
    producer_type: ProducerType,
    item_type: ItemType | None = None,
    type_fn: TypeFn | None = None,
    binding_label: str,
    binding_range: SourceRange | None = None,
    diagnostics: LoadDiagnostics | None = None,
) -> Type:
    """Infer one generated child parameter's strict structural type.

    ``binding_label`` names the outer binding for diagnostics, such as
    ``"each.input.text"``. A string passed directly as ``value`` and each leaf
    string of a ``ValueTemplate`` are scanned as binding templates; a typed binding
    nested in a template gives its declared type. A plain mapping/list value is a
    literal, so its strings infer as ``str`` without scanning. A lone direct
    reference obtains its root type from the parent signature or an injected
    producer/MAP-item provider, then strictly walks record fields. A whole-value
    `T(...)` constructor has type `T` (``type_fn`` resolves it), with field reads on
    the result walked the same way.

    Examples:
        ```python
        inferred = infer_inline_binding_type(
            3,
            parent_input_types={},
            producer_type=producer_type,
            binding_label="call.input.limit",
        )
        assert inferred == Type.scalar(ValueKind.INTEGER)
        ```

    Args:
        value (`Any`):
            The authored outer binding value.
        parent_input_types (`Mapping[str, Type]`):
            Parent flow input declarations available through ``${input.<name>}``.
        producer_type (`ProducerType`):
            Resolves a node id to its output type or canonical output contract.
            Receives the requesting binding label/range so lazy-build failures
            retain their source site.
        item_type (`ItemType | None`, *optional*, defaults to `None`):
            Lazily supplies a MAP element type for a direct ``${item...}`` read.
            ``None`` rejects item references outside an outer MAP input.
        type_fn (`TypeFn | None`, *optional*, defaults to `None`):
            Resolves a constructor name to its type; ``None`` leaves constructors
            unproven.
        binding_label (`str`):
            The author-facing node/input path included in inference errors.
        binding_range (`SourceRange | None`):
            The binding value's source range, or ``None`` for spanless input.

    Returns:
        `Type`:
            The exact scalar or recursively structural type proven by ``value``.

    Raises:
        `LoadError`:
            The value has no statically proven type under the inline-body rules.
    """
    diagnostics = diagnostics or LoadDiagnostics()
    value = binding_source(value)
    if isinstance(value, str):
        inferred = _infer_binding_string(
            value,
            parent_input_types=parent_input_types,
            producer_type=producer_type,
            item_type=item_type,
            type_fn=type_fn,
            binding_label=binding_label,
            binding_range=binding_range,
            diagnostics=diagnostics,
        )
    elif isinstance(value, ValueTemplate):
        inferred = _infer_template(
            value.value,
            parent_input_types=parent_input_types,
            producer_type=producer_type,
            item_type=item_type,
            type_fn=type_fn,
            binding_label=binding_label,
            binding_range=binding_range,
            path=binding_label,
            diagnostics=diagnostics,
        )
    else:
        inferred = _infer_literal(
            value,
            binding_label=binding_label,
            binding_range=binding_range,
            path=binding_label,
            active_containers=set(),
            diagnostics=diagnostics,
        )
    return require_proven_type(
        inferred,
        binding_label=binding_label,
        binding_range=binding_range,
        diagnostics=diagnostics,
    )


def require_proven_type(
    typ: Type,
    *,
    binding_label: str,
    binding_range: SourceRange | None = None,
    diagnostics: LoadDiagnostics | None = None,
) -> Type:
    """Require every structural refinement in ``typ`` to be statically proven.

    A refined record is closed: each declared field must either be required or
    nullable, and every child type is checked recursively. A union requires each
    member to be proven. An unrefined object remains valid as a whole value. Every
    list storage kind, including ``list(Optional(Any))`` with no element, requires a
    recursively proven structural element.

    Args:
        typ (`Type`):
            The type to prove without copying or mutating it.
        binding_label (`str`):
            The author-facing node/input path included in inference errors.
        binding_range (`SourceRange | None`, *optional*, defaults to `None`):
            The binding value's source range, or ``None`` for spanless input.

    Returns:
        `Type`:
            The exact same object passed as ``typ`` after successful proof.

    Raises:
        `LoadError`:
            A record or list refinement is incomplete or internally invalid.
    """
    diagnostics = diagnostics or LoadDiagnostics()
    if typ.fields is not None:
        required = typ.required or frozenset()
        missing = required - typ.fields.keys()
        if missing:
            raise _inference_error(
                binding_label,
                binding_range,
                f"record requires unknown fields {sorted(missing)!r}",
                diagnostics,
            )
        for name, field_type in typ.fields.items():
            if name not in required and not field_type.nullable:
                raise _inference_error(
                    binding_label,
                    binding_range,
                    f"record field {name!r} is neither required nor nullable",
                    diagnostics,
                )
            require_proven_type(
                field_type,
                binding_label=binding_label,
                binding_range=binding_range,
                diagnostics=diagnostics,
            )

    for member in typ.variants or ():
        require_proven_type(
            member,
            binding_label=binding_label,
            binding_range=binding_range,
            diagnostics=diagnostics,
        )

    if typ.kind.is_list():
        if typ.element is None:
            raise _inference_error(
                binding_label,
                binding_range,
                f"list type {canonical_type_label(typ)!r} has no proven element type",
                diagnostics,
            )
        require_proven_type(
            typ.element,
            binding_label=binding_label,
            binding_range=binding_range,
            diagnostics=diagnostics,
        )

    return typ


def canonicalize_inline_type(typ: Type) -> Type:
    """Return a recursively copied, deterministically ordered inline type.

    Record fields use raw Unicode code-point order. Their exact valid required
    membership is retained, including required nullable fields; a refined
    record with ``required=None`` normalizes to an empty frozen set. Other type
    metadata is copied without changing its structural meaning.

    Args:
        typ (`Type`): The already-proven type to canonicalize.

    Returns:
        `Type`: A deep canonical copy; ``typ`` is never mutated.
    """
    if typ.fields is None:
        fields = None
        required = (
            None
            if typ.required is None
            else frozenset(name for name in sorted(typ.required))
        )
    else:
        fields = {
            name: canonicalize_inline_type(typ.fields[name])
            for name in sorted(typ.fields)
        }
        source_required = typ.required or frozenset()
        required = frozenset(name for name in fields if name in source_required)

    tags = None if typ.tags is None else frozenset(tag for tag in sorted(typ.tags))
    element = None if typ.element is None else canonicalize_inline_type(typ.element)
    # Union members are already in canonical order; only their contents are copied.
    variants = (
        None
        if typ.variants is None
        else tuple(canonicalize_inline_type(member) for member in typ.variants)
    )
    return Type(
        kind=typ.kind,
        fields=fields,
        required=required,
        tags=tags,
        element=element,
        nullable=typ.nullable,
        bounds=typ.bounds,
        variants=variants,
    )


def _infer_binding_string(
    source: str,
    *,
    parent_input_types: Mapping[str, Type],
    producer_type: ProducerType,
    item_type: ItemType | None,
    type_fn: TypeFn | None,
    binding_label: str,
    binding_range: SourceRange | None,
    diagnostics: LoadDiagnostics,
) -> Type:
    """Infer a top-level string using binding-template semantics."""
    try:
        segments = scan_template(source)
    except ExpressionError as exc:
        raise _inference_error(
            binding_label,
            binding_range,
            f"invalid binding template: {exc}",
            diagnostics,
        ) from exc

    if len(segments) != 1 or not isinstance(segments[0], Span):
        return Type.scalar(ValueKind.STRING)

    expression = segments[0].tree
    direct_parts = direct_reference_parts(expression)
    if direct_parts is not None:
        return _infer_direct_reference(
            direct_parts,
            parent_input_types=parent_input_types,
            producer_type=producer_type,
            item_type=item_type,
            binding_label=binding_label,
            binding_range=binding_range,
            diagnostics=diagnostics,
        )

    constructed = _constructor_type(expression, type_fn)
    if constructed is not None:
        typ, name, fields = constructed
        return _walk_record_type(
            typ,
            fields,
            path=".".join((f"{name}(...)", *fields)),
            binding_label=binding_label,
            binding_range=binding_range,
            diagnostics=diagnostics,
        )

    if not isinstance(expression, Token):
        raise _inference_error(
            binding_label,
            binding_range,
            "a whole-value expression has no supported static scalar type",
            diagnostics,
        )

    token_kind = str(expression.type)
    if token_kind == "NUMBER":
        kind = ValueKind.NUMBER if "." in str(expression) else ValueKind.INTEGER
        return Type.scalar(kind)
    if token_kind in _SCALAR_TOKEN_KINDS:
        return Type.scalar(_SCALAR_TOKEN_KINDS[token_kind])

    raise _inference_error(
        binding_label,
        binding_range,
        "a whole-value expression has no supported static scalar type",
        diagnostics,
    )


def _constructor_type(
    expression: Any, type_fn: TypeFn | None
) -> tuple[Type, str, tuple[str, ...]] | None:
    """Return `(T, name, fields)` for a whole-value `T(...)` or `T(...).f`, else `None`."""
    if type_fn is None or not isinstance(expression, Tree) or expression.data != "refcall":
        return None
    children = expression.children
    if not any(isinstance(c, Tree) and c.data == "call_suffix" for c in children):
        return None
    name = str(children[0])
    if name == ISINSTANCE or name in TEMPLATE_FNS:
        return None
    typ = type_fn(name)
    if typ is None:
        return None
    fields = tuple(
        str(c.children[0]) for c in children if isinstance(c, Tree) and c.data == "trailer"
    )
    return typ, name, fields


def _infer_direct_reference(
    parts: tuple[str, ...],
    *,
    parent_input_types: Mapping[str, Type],
    producer_type: ProducerType,
    item_type: ItemType | None,
    binding_label: str,
    binding_range: SourceRange | None,
    diagnostics: LoadDiagnostics,
) -> Type:
    """Resolve one direct reference root, then apply the shared strict field walk."""
    path = ".".join(parts)
    head = parts[0]

    if head == "input":
        if len(parts) < 2:
            raise _inference_error(
                binding_label,
                binding_range,
                f"reference ${{{path}}} must be input.<name>[.field]",
                diagnostics,
            )
        input_name = parts[1]
        if input_name not in parent_input_types:
            raise _inference_error(
                binding_label,
                binding_range,
                f"reference ${{{path}}} points to unknown parent input {input_name!r}",
                diagnostics,
            )
        root = parent_input_types[input_name]
        fields = parts[2:]
    elif head == "item":
        if item_type is None:
            raise _inference_error(
                binding_label,
                binding_range,
                f"reference ${{{path}}} is only typed inside an outer MAP input",
                diagnostics,
            )
        root = item_type(binding_label, binding_range)
        fields = parts[1:]
    elif head in ("inputs", "outputs"):
        raise _inference_error(
            binding_label,
            binding_range,
            f"reference ${{{path}}} uses retired plural head {head!r}",
            diagnostics,
        )
    elif head == "system":
        raise _inference_error(
            binding_label,
            binding_range,
            f"reference ${{{path}}} has no statically proven system type",
            diagnostics,
        )
    else:
        if len(parts) < 2 or parts[1] != "output":
            raise _inference_error(
                binding_label,
                binding_range,
                f"reference ${{{path}}} must select {head}.output[.field]",
                diagnostics,
            )
        root = producer_type(head, binding_label, binding_range)
        fields = parts[2:]

        if type(root) in (ValueContract, PartialRecordContract):
            path_error = output_contract_path_error(root, list(fields), path)
            if path_error is not None:
                raise _inference_error(
                    binding_label,
                    binding_range,
                    path_error,
                    diagnostics,
                )
            projected = project_output_contract(root, list(fields))
            if type(projected) is not ValueContract:
                raise _inference_error(
                    binding_label,
                    binding_range,
                    f"reference ${{{path}}} has no statically proven exact type",
                    diagnostics,
                )
            root = projected.type
            fields = ()

    proven_root = require_proven_type(
        root,
        binding_label=binding_label,
        binding_range=binding_range,
        diagnostics=diagnostics,
    )
    return _walk_record_type(
        proven_root,
        fields,
        path=path,
        binding_label=binding_label,
        binding_range=binding_range,
        diagnostics=diagnostics,
    )


def _walk_record_type(
    root: Type,
    fields: tuple[str, ...],
    *,
    path: str,
    binding_label: str,
    binding_range: SourceRange | None,
    diagnostics: LoadDiagnostics,
) -> Type:
    """Walk proven record fields and return a fresh result with path nullability.

    A union field is read through the shared output-contract walk, so its rules and
    messages match every other reference check.
    """
    current = root
    path_nullable = root.nullable
    for index, field in enumerate(fields):
        if current.variants is not None:
            union_error = output_contract_path_error(
                ValueContract(current), list(fields[index:]), path
            )
            if union_error is not None:
                raise _inference_error(
                    binding_label, binding_range, union_error, diagnostics
                )
            read_type = union_field(current, field).type
            if read_type is None:
                raise _inference_error(
                    binding_label,
                    binding_range,
                    f"reference ${{{path}}} has no statically proven exact type",
                    diagnostics,
                )
            current = read_type
            path_nullable = path_nullable or current.nullable
            continue
        if current.fields is None:
            raise _inference_error(
                binding_label,
                binding_range,
                f"reference ${{{path}}} traverses {field!r} without proven record fields",
                diagnostics,
            )
        if field not in current.fields:
            raise _inference_error(
                binding_label,
                binding_range,
                f"reference ${{{path}}} reads unknown record field {field!r}",
                diagnostics,
            )
        current = current.fields[field]
        path_nullable = path_nullable or current.nullable

    return replace(current, nullable=path_nullable)


def _infer_template(
    node: Any,
    *,
    parent_input_types: Mapping[str, Type],
    producer_type: ProducerType,
    item_type: ItemType | None,
    type_fn: TypeFn | None,
    binding_label: str,
    binding_range: SourceRange | None,
    path: str,
    diagnostics: LoadDiagnostics,
) -> Type:
    """Infer one node of a value template: strings are templates, typed bindings declared."""
    if isinstance(node, Binding):
        if node.contract is not None:
            return node.contract.type
        node = node.source
    if isinstance(node, str):
        return _infer_binding_string(
            node,
            parent_input_types=parent_input_types,
            producer_type=producer_type,
            item_type=item_type,
            type_fn=type_fn,
            binding_label=binding_label,
            binding_range=binding_range,
            diagnostics=diagnostics,
        )

    def infer(child: Any, child_path: str) -> Type:
        return _infer_template(
            child,
            parent_input_types=parent_input_types,
            producer_type=producer_type,
            item_type=item_type,
            type_fn=type_fn,
            binding_label=binding_label,
            binding_range=binding_range,
            path=child_path,
            diagnostics=diagnostics,
        )

    if type(node) is dict:
        fields = {key: infer(node[key], f"{path}[{key!r}]") for key in sorted(node)}
        return Type(kind=ValueKind.OBJECT, fields=fields, required=frozenset(fields))
    if type(node) is list:
        if not node:
            raise _inference_error(
                binding_label,
                binding_range,
                f"empty literal list at {path} has no provable element type",
                diagnostics,
            )
        element = _common_element_type(
            [infer(item, f"{path}[{index}]") for index, item in enumerate(node)],
            binding_label=binding_label,
            binding_range=binding_range,
            path=path,
            noun="list",
            diagnostics=diagnostics,
        )
        return Type(kind=list_kind_for_element(element), element=element)
    return _infer_literal(
        node,
        binding_label=binding_label,
        binding_range=binding_range,
        path=path,
        active_containers=set(),
        diagnostics=diagnostics,
    )


def _infer_literal(
    value: Any,
    *,
    binding_label: str,
    binding_range: SourceRange | None,
    path: str,
    active_containers: set[int],
    diagnostics: LoadDiagnostics,
) -> Type:
    """Infer a recursive literal; nested strings never enter the template parser."""
    if value is None:
        return Type.scalar(ValueKind.NONE)
    if isinstance(value, bool):
        return Type.scalar(ValueKind.BOOLEAN)
    if isinstance(value, int):
        return Type.scalar(ValueKind.INTEGER)
    if isinstance(value, float):
        return Type.scalar(ValueKind.NUMBER)
    if isinstance(value, str):
        return Type.scalar(ValueKind.STRING)

    if isinstance(value, dict):
        return _infer_mapping(
            value,
            binding_label=binding_label,
            binding_range=binding_range,
            path=path,
            active_containers=active_containers,
            diagnostics=diagnostics,
        )
    if isinstance(value, list):
        return _infer_list(
            value,
            binding_label=binding_label,
            binding_range=binding_range,
            path=path,
            active_containers=active_containers,
            diagnostics=diagnostics,
        )

    raise _inference_error(
        binding_label,
        binding_range,
        f"literal at {path} has unsupported type {type(value).__name__!r}",
        diagnostics,
    )


def _infer_mapping(
    value: dict[Any, Any],
    *,
    binding_label: str,
    binding_range: SourceRange | None,
    path: str,
    active_containers: set[int],
    diagnostics: LoadDiagnostics,
) -> Type:
    """Infer a sorted closed record from one literal mapping."""
    container_id = _enter_container(
        value,
        binding_label=binding_label,
        binding_range=binding_range,
        path=path,
        active_containers=active_containers,
        diagnostics=diagnostics,
    )
    try:
        for key in value:
            if not isinstance(key, str):
                raise _inference_error(
                    binding_label,
                    binding_range,
                    f"literal mapping at {path} requires string keys, got {key!r}",
                    diagnostics,
                )

        fields = {
            key: _infer_literal(
                value[key],
                binding_label=binding_label,
                binding_range=binding_range,
                path=f"{path}[{key!r}]",
                active_containers=active_containers,
                diagnostics=diagnostics,
            )
            for key in sorted(value)
        }
        return Type(
            kind=ValueKind.OBJECT,
            fields=fields,
            required=frozenset(fields),
        )
    finally:
        active_containers.remove(container_id)


def _infer_list(
    value: list[Any],
    *,
    binding_label: str,
    binding_range: SourceRange | None,
    path: str,
    active_containers: set[int],
    diagnostics: LoadDiagnostics,
) -> Type:
    """Infer one homogeneous structural list, widening only int/float scalars."""
    container_id = _enter_container(
        value,
        binding_label=binding_label,
        binding_range=binding_range,
        path=path,
        active_containers=active_containers,
        diagnostics=diagnostics,
    )
    try:
        if not value:
            raise _inference_error(
                binding_label,
                binding_range,
                f"empty literal list at {path} has no provable element type",
                diagnostics,
            )

        element_types = [
            _infer_literal(
                item,
                binding_label=binding_label,
                binding_range=binding_range,
                path=f"{path}[{index}]",
                active_containers=active_containers,
                diagnostics=diagnostics,
            )
            for index, item in enumerate(value)
        ]
        element = _common_element_type(
            element_types,
            binding_label=binding_label,
            binding_range=binding_range,
            path=path,
            diagnostics=diagnostics,
        )
        return Type(
            kind=_LIST_KINDS.get(element.kind, ValueKind.LIST_ANY),
            element=element,
        )
    finally:
        active_containers.remove(container_id)


def _common_element_type(
    element_types: list[Type],
    *,
    binding_label: str,
    binding_range: SourceRange | None,
    path: str,
    noun: str = "literal list",
    diagnostics: LoadDiagnostics,
) -> Type:
    """Return the common type up to `Optional`, or the scalar numeric widening, for a list."""
    # A generated parameter must have one concrete type, so unlike a template's
    # output contract (left unproven) a list with no common type fails the load.
    bare = [replace(element, nullable=False) for element in element_types]
    if all(element == bare[0] for element in bare[1:]):
        return replace(bare[0], nullable=any(e.nullable for e in element_types))

    integer = Type.scalar(ValueKind.INTEGER)
    number = Type.scalar(ValueKind.NUMBER)
    if all(element == integer or element == number for element in element_types):
        return number

    raise _inference_error(
        binding_label,
        binding_range,
        f"{noun} at {path} has heterogeneous element types",
        diagnostics,
    )


def _enter_container(
    value: dict[Any, Any] | list[Any],
    *,
    binding_label: str,
    binding_range: SourceRange | None,
    path: str,
    active_containers: set[int],
    diagnostics: LoadDiagnostics,
) -> int:
    """Add one container to the active recursion path and reject true cycles."""
    container_id = id(value)
    if container_id in active_containers:
        raise _inference_error(
            binding_label,
            binding_range,
            f"cyclic literal container encountered at {path}",
            diagnostics,
        )
    active_containers.add(container_id)
    return container_id


def _inference_error(
    binding_label: str,
    binding_range: SourceRange | None,
    reason: str,
    diagnostics: LoadDiagnostics,
) -> LoadError:
    """Create the one located error shape used by strict inline inference."""
    return diagnostics.error(
        f"cannot infer type for inline binding {binding_label!r}: "
        f"{reason}; {explicit_type_hint(binding_label)}",
        range=binding_range,
    )


__all__ = [
    "ItemType",
    "ProducerType",
    "canonicalize_inline_type",
    "explicit_type_hint",
    "infer_inline_binding_type",
    "require_proven_type",
]
