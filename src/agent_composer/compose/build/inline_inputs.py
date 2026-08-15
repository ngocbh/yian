"""Strict type inference for generated inline-composite input declarations.

Inline composite bodies become private child flows, so every outer binding must
produce a concrete child-parameter type. This module infers only types that the
Compose source proves structurally; it never falls back to the runtime value
builder's lenient ``Any`` behavior. Top-level strings retain binding-template
semantics, while strings nested in literal containers remain ordinary strings.

Knows about: ``expr`` template syntax, ``typesys`` structural types, and Compose
source ranges/errors.
Never imports: ``compose.loader``, ``compile``, ``runtime``, or node kinds.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import Any

from lark import Token, Tree

from agent_composer.compose.helpers.errors import LoadError
from agent_composer.events import SourceRange
from agent_composer.expr.expressions import ExpressionError
from agent_composer.expr.template import Span, scan_template
from agent_composer.typesys.values import Type, ValueKind


_EXPLICIT_TYPE_HINT = "use an explicitly typed named `defs:` body"

ProducerType = Callable[[str, str, SourceRange | None], Type]
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
    binding_label: str,
    binding_range: SourceRange | None = None,
) -> Type:
    """Infer one generated child parameter's strict structural type.

    ``binding_label`` names the outer binding for diagnostics, such as
    ``"each.input.text"``. Only a string passed directly as ``value`` is scanned
    as a binding template. Recursive mapping/list values are literals, so nested
    strings containing ``${...}`` or ``$$`` infer as ``str`` without scanning. A
    lone direct reference obtains its root type from the parent signature or an
    injected producer/MAP-item provider, then strictly walks record fields.

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
            Resolves a node id to its output type. Receives the requesting
            binding label/range so lazy-build failures retain their source site.
        item_type (`ItemType | None`, *optional*, defaults to `None`):
            Lazily supplies a MAP element type for a direct ``${item...}`` read.
            ``None`` rejects item references outside an outer MAP input.
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
    if isinstance(value, str):
        return _infer_binding_string(
            value,
            parent_input_types=parent_input_types,
            producer_type=producer_type,
            item_type=item_type,
            binding_label=binding_label,
            binding_range=binding_range,
        )
    return _infer_literal(
        value,
        binding_label=binding_label,
        binding_range=binding_range,
        path=binding_label,
        active_containers=set(),
    )


def _infer_binding_string(
    source: str,
    *,
    parent_input_types: Mapping[str, Type],
    producer_type: ProducerType,
    item_type: ItemType | None,
    binding_label: str,
    binding_range: SourceRange | None,
) -> Type:
    """Infer a top-level string using binding-template semantics."""
    try:
        segments = scan_template(source)
    except ExpressionError as exc:
        raise _inference_error(
            binding_label,
            binding_range,
            f"invalid binding template: {exc}",
        ) from exc

    if len(segments) != 1 or not isinstance(segments[0], Span):
        return Type.scalar(ValueKind.STRING)

    expression = segments[0].tree
    direct_parts = _direct_reference_parts(expression)
    if direct_parts is not None:
        return _infer_direct_reference(
            direct_parts,
            parent_input_types=parent_input_types,
            producer_type=producer_type,
            item_type=item_type,
            binding_label=binding_label,
            binding_range=binding_range,
        )

    if not isinstance(expression, Token):
        raise _inference_error(
            binding_label,
            binding_range,
            "a whole-value expression has no supported static scalar type",
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
    )


def _direct_reference_parts(expression: Tree | Token) -> tuple[str, ...] | None:
    """Return an exact lone ``refcall`` path, excluding calls and expressions."""
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


def _infer_direct_reference(
    parts: tuple[str, ...],
    *,
    parent_input_types: Mapping[str, Type],
    producer_type: ProducerType,
    item_type: ItemType | None,
    binding_label: str,
    binding_range: SourceRange | None,
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
            )
        input_name = parts[1]
        if input_name not in parent_input_types:
            raise _inference_error(
                binding_label,
                binding_range,
                f"reference ${{{path}}} points to unknown parent input {input_name!r}",
            )
        root = parent_input_types[input_name]
        fields = parts[2:]
    elif head == "item":
        if item_type is None:
            raise _inference_error(
                binding_label,
                binding_range,
                f"reference ${{{path}}} is only typed inside an outer MAP input",
            )
        root = item_type(binding_label, binding_range)
        fields = parts[1:]
    elif head in ("inputs", "outputs"):
        raise _inference_error(
            binding_label,
            binding_range,
            f"reference ${{{path}}} uses retired plural head {head!r}",
        )
    elif head == "system":
        raise _inference_error(
            binding_label,
            binding_range,
            f"reference ${{{path}}} has no statically proven system type",
        )
    else:
        if len(parts) < 2 or parts[1] != "output":
            raise _inference_error(
                binding_label,
                binding_range,
                f"reference ${{{path}}} must select {head}.output[.field]",
            )
        root = producer_type(head, binding_label, binding_range)
        fields = parts[2:]

    return _walk_record_type(
        root,
        fields,
        path=path,
        binding_label=binding_label,
        binding_range=binding_range,
    )


def _walk_record_type(
    root: Type,
    fields: tuple[str, ...],
    *,
    path: str,
    binding_label: str,
    binding_range: SourceRange | None,
) -> Type:
    """Walk proven record fields and return a fresh result with path nullability."""
    current = root
    path_nullable = root.nullable
    for field in fields:
        if current.fields is None:
            raise _inference_error(
                binding_label,
                binding_range,
                f"reference ${{{path}}} traverses {field!r} without proven record fields",
            )
        if field not in current.fields:
            raise _inference_error(
                binding_label,
                binding_range,
                f"reference ${{{path}}} reads unknown record field {field!r}",
            )
        current = current.fields[field]
        path_nullable = path_nullable or current.nullable

    return replace(current, nullable=path_nullable)


def _infer_literal(
    value: Any,
    *,
    binding_label: str,
    binding_range: SourceRange | None,
    path: str,
    active_containers: set[int],
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
        )
    if isinstance(value, list):
        return _infer_list(
            value,
            binding_label=binding_label,
            binding_range=binding_range,
            path=path,
            active_containers=active_containers,
        )

    raise _inference_error(
        binding_label,
        binding_range,
        f"literal at {path} has unsupported type {type(value).__name__!r}",
    )


def _infer_mapping(
    value: dict[Any, Any],
    *,
    binding_label: str,
    binding_range: SourceRange | None,
    path: str,
    active_containers: set[int],
) -> Type:
    """Infer a sorted closed record from one literal mapping."""
    container_id = _enter_container(
        value,
        binding_label=binding_label,
        binding_range=binding_range,
        path=path,
        active_containers=active_containers,
    )
    try:
        for key in value:
            if not isinstance(key, str):
                raise _inference_error(
                    binding_label,
                    binding_range,
                    f"literal mapping at {path} requires string keys, got {key!r}",
                )

        fields = {
            key: _infer_literal(
                value[key],
                binding_label=binding_label,
                binding_range=binding_range,
                path=f"{path}[{key!r}]",
                active_containers=active_containers,
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
) -> Type:
    """Infer one homogeneous structural list, widening only int/float scalars."""
    container_id = _enter_container(
        value,
        binding_label=binding_label,
        binding_range=binding_range,
        path=path,
        active_containers=active_containers,
    )
    try:
        if not value:
            raise _inference_error(
                binding_label,
                binding_range,
                f"empty literal list at {path} has no provable element type",
            )

        element_types = [
            _infer_literal(
                item,
                binding_label=binding_label,
                binding_range=binding_range,
                path=f"{path}[{index}]",
                active_containers=active_containers,
            )
            for index, item in enumerate(value)
        ]
        element = _common_element_type(
            element_types,
            binding_label=binding_label,
            binding_range=binding_range,
            path=path,
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
) -> Type:
    """Return the exact common type, or the scalar numeric widening, for a list."""
    first = element_types[0]
    if all(element == first for element in element_types[1:]):
        return first

    integer = Type.scalar(ValueKind.INTEGER)
    number = Type.scalar(ValueKind.NUMBER)
    if all(element == integer or element == number for element in element_types):
        return number

    raise _inference_error(
        binding_label,
        binding_range,
        f"literal list at {path} has heterogeneous element types",
    )


def _enter_container(
    value: dict[Any, Any] | list[Any],
    *,
    binding_label: str,
    binding_range: SourceRange | None,
    path: str,
    active_containers: set[int],
) -> int:
    """Add one container to the active recursion path and reject true cycles."""
    container_id = id(value)
    if container_id in active_containers:
        raise _inference_error(
            binding_label,
            binding_range,
            f"cyclic literal container encountered at {path}",
        )
    active_containers.add(container_id)
    return container_id


def _inference_error(
    binding_label: str,
    binding_range: SourceRange | None,
    reason: str,
) -> LoadError:
    """Create the one located error shape used by strict inline inference."""
    return LoadError(
        f"cannot infer type for inline binding {binding_label!r}: "
        f"{reason}; {_EXPLICIT_TYPE_HINT}",
        range=binding_range,
    )


__all__ = ["ItemType", "ProducerType", "infer_inline_binding_type"]
