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

from typing import Any

from lark import Token

from agent_composer.compose.helpers.errors import LoadError
from agent_composer.events import SourceRange
from agent_composer.expr.expressions import ExpressionError
from agent_composer.expr.template import Span, scan_template
from agent_composer.typesys.values import Type, ValueKind


_EXPLICIT_TYPE_HINT = "use an explicitly typed named `defs:` body"

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
    binding_label: str,
    binding_range: SourceRange | None = None,
) -> Type:
    """Infer one generated child parameter's fully structural literal type.

    ``binding_label`` names the outer binding for diagnostics, such as
    ``"each.input.text"``. Only a string passed directly as ``value`` is scanned
    as a binding template. Recursive mapping/list values are literals, so nested
    strings containing ``${...}`` or ``$$`` infer as ``str`` without scanning.

    Examples:
        ```python
        inferred = infer_inline_binding_type(3, binding_label="call.input.limit")
        assert inferred == Type.scalar(ValueKind.INTEGER)
        ```

    Args:
        value (`Any`):
            The authored outer binding value.
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


__all__ = ["infer_inline_binding_type"]
