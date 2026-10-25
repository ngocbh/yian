"""The type names `isinstance(x, T)` and `T(...)` constructors name, and their resolution.

`isinstance` and constructors read their type from the evaluating node's `types` table.
Compose fills that table from the names a node's expressions use, resolved against the
flow's own `typedefs:` registry, so each flow (root, `defs:`, inline bodies, `uses:`
children) sees its own types. `isinstance` also takes a builtin scalar name; a
constructor takes only a `typedefs:` name. `isinstance_members` also says which members of
a union an `isinstance` check narrows it to.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable

from yian.expr import constructor_names, isinstance_type_names, parse_condition
from yian.expr.expressions import unknown_function_message
from yian.expr.template import Span, scan_template
from yian.nodes.contracts import binding_source, iter_source_strings
from yian.typesys import TypeCheckError
from yian.typesys.types import TypeRegistry, type_for
from yian.typesys.values import Type, canonical_union_order, types_overlap

__all__ = [
    "collect_constructor_names",
    "collect_isinstance_names",
    "constructor_type_fn",
    "expression_trees",
    "isinstance_members",
    "resolve_constructor_type",
    "resolve_isinstance_type",
]


# The scalar names `isinstance` accepts besides a `typedefs:` name.
_ISINSTANCE_SCALAR_NAMES = ("str", "int", "float", "bool", "date", "datetime", "object")


def collect_isinstance_names(source: Any, *, condition: bool = False) -> list[str]:
    """Return the type names the `isinstance(x, T)` calls in `source` name.

    Args:
        source (`Any`):
            A binding source (raw or carrier-wrapped) or a condition string. Every
            string nested in a value template is scanned; a literal names no types.
        condition (`bool`, *optional*, defaults to `False`):
            `True` parses `source` as one condition expression (`when:`, `while:`, an
            assert), where a bare `isinstance(x, T)` is an expression. `False` scans it as
            a template, where only `${...}` spans are expressions.

    Returns:
        `list[str]`:
            The names in first-seen order, without duplicates.

    Raises:
        `ExpressionError`:
            Raised when `source` does not parse.

    Example:
        ```python
        collect_isinstance_names("${isinstance(d.output, Buy)}")  # ["Buy"]
        collect_isinstance_names("isinstance(x, int)", condition=True)  # ["int"]
        ```
    """
    return _collect(source, condition, isinstance_type_names)


def collect_constructor_names(source: Any, *, condition: bool = False) -> list[str]:
    """Return the names the `T(...)` constructor calls in `source` use.

    Args:
        source (`Any`):
            A binding source or a condition string, as in `collect_isinstance_names`.
        condition (`bool`, *optional*, defaults to `False`):
            `True` parses `source` as one condition expression; `False` scans it as a
            template.

    Returns:
        `list[str]`:
            The names in first-seen order, without duplicates.

    Raises:
        `ExpressionError`:
            Raised when `source` does not parse.

    Example:
        ```python
        collect_constructor_names("${Buy(d.output).conviction}")  # ["Buy"]
        ```
    """
    return _collect(source, condition, constructor_names)


def expression_trees(source: Any, *, condition: bool = False) -> list[Any]:
    """Return the parsed expressions in `source`.

    Args:
        source (`Any`):
            A binding source or a condition string, as in `collect_isinstance_names`.
        condition (`bool`, *optional*, defaults to `False`):
            `True` parses `source` as one condition expression; `False` parses each
            `${...}` span of every string in it.

    Returns:
        `list[Any]`:
            The parse trees in source order.

    Raises:
        `ExpressionError`:
            Raised when `source` does not parse.
    """
    if condition:
        source = binding_source(source)
        return [parse_condition(source)] if isinstance(source, str) else []
    return [
        seg.tree
        for _, text in iter_source_strings(source)
        for seg in scan_template(text)
        if isinstance(seg, Span)
    ]


def _collect(source: Any, condition: bool, names_of: Callable[[Any], list[str]]) -> list[str]:
    """Union `names_of` over the expressions in `source`, in first-seen order."""
    names: list[str] = []
    for tree in expression_trees(source, condition=condition):
        for name in names_of(tree):
            if name not in names:
                names.append(name)
    return names


def resolve_isinstance_type(name: str, registry: TypeRegistry) -> Type:
    """Resolve one `isinstance` type name against a flow's `typedefs:` registry.

    Args:
        name (`str`):
            The bare type name written as the second `isinstance` argument.
        registry (`TypeRegistry`):
            The flow's `read_typedefs(...)` table.

    Returns:
        `Type`:
            The resolved type.

    Raises:
        `TypeCheckError`:
            Raised when `name` is neither a `typedefs:` name nor a builtin scalar name.
    """
    if name not in registry and name not in _ISINSTANCE_SCALAR_NAMES:
        raise TypeCheckError(
            f"unknown type {name!r}; isinstance takes a typedefs: name or one of "
            + ", ".join(_ISINSTANCE_SCALAR_NAMES)
        )
    return type_for(name, registry)


def resolve_constructor_type(name: str, registry: TypeRegistry) -> Type:
    """Resolve one constructor name against a flow's `typedefs:` registry.

    Args:
        name (`str`):
            The callee of a `T(...)` call that is no builtin.
        registry (`TypeRegistry`):
            The flow's `read_typedefs(...)` table.

    Returns:
        `Type`:
            The resolved type.

    Raises:
        `TypeCheckError`:
            Raised when `name` is not a `typedefs:` name; a builtin type name (`int`)
            gets a hint that a constructor is no conversion.
    """
    if name not in registry:
        raise TypeCheckError(unknown_function_message(name))
    return type_for(name, registry)


def constructor_type_fn(registry: TypeRegistry) -> Callable[[str], Type | None]:
    """Return the static-inference lookup of constructor types for one flow.

    Args:
        registry (`TypeRegistry`):
            The flow's `read_typedefs(...)` table.

    Returns:
        `Callable[[str], Type | None]`:
            Maps a `typedefs:` name to its type and any other name to `None`.
    """
    resolved: dict[str, Type] = {}

    def type_fn(name: str) -> Type | None:
        if name not in registry:
            return None
        if name not in resolved:
            resolved[name] = type_for(name, registry)
        return resolved[name]

    return type_fn


def isinstance_members(base_type: Type, typ: Type) -> tuple[Type, ...] | None:
    """Return the members of a union `base_type` that pass `isinstance(x, typ)`, if known.

    The answer is the base members equal to a member of `typ` (ignoring nullability),
    and only when no other base member could pass the check. It does not check whether
    every `base_type` value already fits `typ`; callers check that first.

    Args:
        base_type (`Type`):
            The type of the value being tested.
        typ (`Type`):
            The type the check names. A union names each of its members.

    Returns:
        `tuple[Type, ...] | None`:
            The shared members in canonical union order, or `None` when `base_type` is
            no union, shares no member with `typ`, or has another member that `typ` may
            admit.

    Example:
        For `Union[Buy, Hold, Sell]`, `Union[Hold, Sell, Cancel]` gives `(Hold, Sell)`
        and `Buy` gives `(Buy,)`; a `typ` whose member admits some `Buy` values but is
        no base member gives `None`.
    """
    members = base_type.variants
    if members is None:
        return None
    strict = replace(typ, nullable=False)
    wanted = strict.variants or (strict,)
    shared = [member for member in members if any(member == w for w in wanted)]
    # The runtime check is the write-boundary check, so a base member that is not a
    # member of `typ` but overlaps it can still pass. Leaving it out would be unsound
    # (the branch could see a value the answer excludes), and keeping it could make the
    # answer less permissive than `typ` (its values need not fit `typ`), so give no answer.
    passing = [member for member in members if types_overlap(member, strict)]
    if shared and shared == passing:
        return canonical_union_order(shared)
    return None
