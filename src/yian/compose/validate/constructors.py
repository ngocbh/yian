"""Graph-stage: reject `${T(...)}` constructor calls that can never succeed.

A constructor `T(x)` checks the value of `x` against the `typedefs:` type `T`; the
keyword form `T(a=..., b=...)` builds a record from exactly the given fields and checks
it. Most of what can go wrong is visible in the call itself:

- the call mixes keyword and positional arguments, or has several positional ones;
- the keyword form names a union (construct a member instead) or a type that is no
  record, a field `T` doesn't have, or leaves out a required field;
- a field read on the result (`T(x).f`) names a field `T` can't have.

`builtin_field_errors` rejects a field read on a builtin whose result type is known
(`upper(x).f` reads a field of text), the same way.

Given the types of the references an expression reads, a positional argument or a
keyword field whose inferred type is disjoint from the declared one is an error too.
A narrower or unproven value loads and is left to the runtime check.

Imports flow down only: `compile.validation` (the type relation), `expr`, `typesys`,
`nodes.contracts` (the field-read message), and `compose.helpers`. Callers are the
reference validator and the pipeline's type-table pass.
"""

from __future__ import annotations

from typing import Callable

from lark import Token, Tree

from yian.compile.validation import type_relation
from yian.compose.helpers.isinstance_types import constructor_type_fn
from yian.compose.helpers.output_contracts import literal_type
from yian.expr import infer_expr_type
from yian.expr.builtins import TEMPLATE_FNS, builtin_result_type
from yian.expr.expressions import ISINSTANCE, constructor_shape_error
from yian.nodes.contracts import ValueContract, output_contract_field_error
from yian.typesys import canonical_type_label
from yian.typesys.types import AliasDef, RefExpr, TypeRegistry, UnionExpr
from yian.typesys.values import Type, parse_int_text

__all__ = ["builtin_field_errors", "constructor_errors"]

RefType = Callable[[str], "Type | None"]


def constructor_errors(
    tree: Tree | Token,
    registry: TypeRegistry,
    ref_type: RefType | None = None,
) -> list[str]:
    """Return why each constructor call in one parsed expression can never succeed.

    Args:
        tree (`Tree | Token`):
            A parsed `${...}` interior or condition.
        registry (`TypeRegistry`):
            The flow's `typedefs:` registry; a callee outside it is no constructor and
            is reported elsewhere.
        ref_type (`RefType | None`, *optional*, defaults to `None`):
            Resolves a reference path to its proven type. `None` checks only what the
            call itself shows.

    Returns:
        `list[str]`:
            One message per problem, in source order.
    """
    if not isinstance(tree, Tree):
        return []
    type_fn = constructor_type_fn(registry)
    errors: list[str] = []
    for node in tree.iter_subtrees_topdown():
        if node.data != "refcall":
            continue
        call_suffix = next(
            (c for c in node.children if isinstance(c, Tree) and c.data == "call_suffix"),
            None,
        )
        name = str(node.children[0])
        if call_suffix is None or name == ISINSTANCE or name in TEMPLATE_FNS:
            continue
        typ = type_fn(name)
        if typ is None:
            continue
        args = call_suffix.children
        shape_error = constructor_shape_error(name, call_suffix)
        if shape_error is not None:
            errors.append(shape_error)
            continue
        if len(args) == 1 and len(args[0].children) == 1:
            error = _value_error(name, typ, args[0].children[0], ref_type, type_fn)
            if error is not None:
                errors.append(error)
        else:
            errors.extend(_record_errors(name, typ, args, registry, ref_type, type_fn))
        error = _result_field_error(node, name, typ)
        if error is not None:
            errors.append(error)
    return errors


def builtin_field_errors(tree: Tree | Token) -> list[str]:
    """Return why each field read on a builtin call's result in one expression fails.

    A builtin whose result type is unknown (`builtin_result_type` gives `None`) is left
    unchecked.

    Args:
        tree (`Tree | Token`):
            A parsed `${...}` interior or condition.

    Returns:
        `list[str]`:
            One message per field read the result type can't have, in source order.
    """
    if not isinstance(tree, Tree):
        return []
    errors: list[str] = []
    for node in tree.iter_subtrees_topdown():
        if node.data != "refcall":
            continue
        # A `${upper.f}` read of an input named `upper` is no call.
        if not any(isinstance(c, Tree) and c.data == "call_suffix" for c in node.children):
            continue
        name = str(node.children[0])
        typ = builtin_result_type(name)
        if typ is None:
            continue
        error = _result_field_error(node, name, typ)
        if error is not None:
            errors.append(error)
    return errors


def _result_field_error(node: Tree, name: str, typ: Type) -> str | None:
    """Return why the field reads trailing call `name(...)`, whose result is `typ`, fail."""
    fields = [
        str(child.children[0])
        for child in node.children
        if isinstance(child, Tree) and child.data == "trailer"
    ]
    if not fields:
        return None
    return output_contract_field_error(
        ValueContract(typ), fields, f"{name}(...)." + ".".join(fields)
    )


def _value_error(
    name: str,
    typ: Type,
    arg: Tree | Token,
    ref_type: RefType | None,
    type_fn: Callable[[str], Type | None],
) -> str | None:
    """Return why `T(x)`'s value can never be a `T`, else `None`."""
    inferred = _argument_type(arg, ref_type, type_fn)
    if inferred is None or type_relation(inferred, typ) != "disjoint":
        return None
    return (
        f"{name}(...): the value is {canonical_type_label(inferred)}, which can never "
        f"be a {name}"
    )


def _record_errors(
    name: str,
    typ: Type,
    args: list,
    registry: TypeRegistry,
    ref_type: RefType | None,
    type_fn: Callable[[str], Type | None],
) -> list[str]:
    """Return why `T(a=..., b=...)` can never build a valid `T`."""
    if typ.variants is not None:
        members = _union_member_names(name, registry)
        listed = f" ({' or '.join(f'{m}(...)' for m in members)})" if members else ""
        return [
            f"{name} is a union; construct a member{listed}, or check a value with "
            f"{name}(x)"
        ]
    if typ.fields is None:
        return [
            f"{name} is not a record, so {name}(...) takes one value to check, as in "
            f"{name}(x)"
        ]
    given = {str(arg.children[0]): arg.children[1] for arg in args}
    errors: list[str] = []
    for field_name in given:
        if field_name not in typ.fields:
            errors.append(
                f"{name} has no field {field_name!r} (fields: {', '.join(typ.fields)})"
            )
    missing = [
        field_name
        for field_name in typ.fields
        if field_name in (typ.required or frozenset()) and field_name not in given
    ]
    if missing:
        noun = "field" if len(missing) == 1 else "fields"
        errors.append(
            f"{name}(...) is missing required {noun} "
            + ", ".join(f"`{field_name}`" for field_name in missing)
        )
    for field_name, value in given.items():
        declared = typ.fields.get(field_name)
        if declared is None:
            continue
        inferred = _argument_type(value, ref_type, type_fn)
        if inferred is not None and type_relation(inferred, declared) == "disjoint":
            errors.append(
                f"{name}(...) field {field_name!r}: declared "
                f"{canonical_type_label(declared)}, but the value is "
                f"{canonical_type_label(inferred)}"
            )
    return errors


def _argument_type(
    arg: Tree | Token,
    ref_type: RefType | None,
    type_fn: Callable[[str], Type | None],
) -> Type | None:
    """Infer one argument's type; a literal has its exact type, so a wrong one is disjoint."""
    if isinstance(arg, Token) and arg.type in ("STRING", "NUMBER", "BOOL"):
        text = str(arg)
        if arg.type == "STRING":
            value: object = text[1:-1]
        elif arg.type == "NUMBER":
            # `parse_expr` rejected a long int token; parse_int_text keeps yian's limit
            # and message should one reach here anyway.
            value = float(text) if "." in text else parse_int_text(text)
        else:
            value = text.lower() == "true"
        return literal_type(value)
    return infer_expr_type(arg, ref_type or (lambda _path: None), type_fn)


def _union_member_names(name: str, registry: TypeRegistry) -> list[str]:
    """Return the member names a union typedef lists, or `[]` when they aren't all names."""
    definition = registry.get(name)
    if not isinstance(definition, AliasDef) or not isinstance(definition.target, UnionExpr):
        return []
    members = definition.target.members
    if not all(isinstance(member, RefExpr) for member in members):
        return []
    return [member.name for member in members]
