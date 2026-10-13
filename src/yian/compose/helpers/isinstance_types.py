"""The type names an `isinstance(x, T)` call names, and their resolution at load.

`isinstance` reads its type from the evaluating node's `types` table. Compose fills that
table from the names a node's expressions use, resolved against the flow's own `typedefs:`
registry, so each flow (root, `defs:`, inline bodies, `uses:` children) sees its own types.
"""

from __future__ import annotations

from typing import Any

from yian.expr import isinstance_type_names, parse_condition
from yian.expr.template import Span, scan_template
from yian.nodes.contracts import binding_source
from yian.typesys import TypeCheckError
from yian.typesys.types import TypeRegistry, type_for
from yian.typesys.values import Type

__all__ = ["BUILTIN_TYPE_NAMES", "collect_isinstance_names", "resolve_isinstance_type"]


# The scalar names `isinstance` accepts besides a `typedefs:` name.
BUILTIN_TYPE_NAMES = ("str", "int", "float", "bool", "date", "datetime", "object")


def collect_isinstance_names(source: Any, *, condition: bool = False) -> list[str]:
    """Return the type names the `isinstance(x, T)` calls in `source` name.

    Args:
        source (`Any`):
            A binding source (raw or carrier-wrapped) or a condition string. A non-string
            source names no types.
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
    source = binding_source(source)
    if not isinstance(source, str):
        return []
    if condition:
        trees = [parse_condition(source)]
    else:
        trees = [seg.tree for seg in scan_template(source) if isinstance(seg, Span)]
    names: list[str] = []
    for tree in trees:
        for name in isinstance_type_names(tree):
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
    if name not in registry and name not in BUILTIN_TYPE_NAMES:
        raise TypeCheckError(
            f"unknown type {name!r}; isinstance takes a typedefs: name or one of "
            + ", ".join(BUILTIN_TYPE_NAMES)
        )
    return type_for(name, registry)
