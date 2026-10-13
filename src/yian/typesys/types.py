"""Authoring TypeExpr grammar → runtime Type.

Parses an authored type expression into a `TypeExpr` AST, and
resolves a `TypeExpr` against the per-flow type registry into a leaf `Type`
(`yian.typesys.values`). This is the bridge from the authoring contract
to the typed value system.

Imports the peer ``values`` leaf and is consumed by compose/compile type
resolution. Runtime receives only the resolved ``Type`` values.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, replace
from typing import Union

from yian.typesys.values import (
    TypeCheckError,
    ValueKind,
    Type,
    _validate_numeric_bounds,
    canonical_union_order,
    types_overlap,
    union_tag_fields,
)


# ---------- TypeExpr AST ----------


@dataclass(frozen=True)
class ScalarExpr:
    name: str  # a scalar keyword (see _SCALAR_KEYWORDS)
    bounds: tuple[int | float, int | float] | None = None


@dataclass(frozen=True)
class ListExpr:
    element: "TypeExpr"


@dataclass(frozen=True)
class RefExpr:
    name: str  # a registry name (record or variant)


@dataclass(frozen=True)
class OptionalExpr:
    inner: "TypeExpr"  # Optional[X] — the resolved Type is X's, marked nullable


@dataclass(frozen=True)
class EnumExpr:
    tags: tuple  # an inline Literal[...] enum (a tag-only variant)


@dataclass(frozen=True)
class DictExpr:
    # dict[K, V] resolves to a lenient object; key/value types are not enforced.
    pass


@dataclass(frozen=True)
class UnionExpr:
    members: tuple  # the authored `Union[...]` members, in authored order


TypeExpr = Union[ScalarExpr, ListExpr, RefExpr, OptionalExpr, EnumExpr, DictExpr, UnionExpr]

_SCALAR_KEYWORDS = {
    "str", "int", "float", "bool", "date", "datetime", "object", "file", "None", "none",
}

# Names a typedef must not shadow (typing constructors + the generic builtins).
_TYPING_CONSTRUCTORS = {"Optional", "Literal", "Union", "List", "Dict", "list", "dict", "Any"}

# Names a type expression may start with besides typedefs, derived from the sets
# above so they cannot drift. `topics` is a retired alias, so it is not listed.
BUILTIN_TYPE_NAMES = frozenset(_SCALAR_KEYWORDS | _TYPING_CONSTRUCTORS | {"any"})


def parse_type(s: str) -> TypeExpr:
    """Parse a type reference string into a `TypeExpr` AST via Python `ast` (eval mode).

    Accepts the one Python-surface vocabulary (`str`/`int`/`float`/`bool`/`Any`/`date`/
    `datetime`/`object`/`file`/`None`, `int[min,max]`/`float[min,max]`,
    `list[X]`/`Optional[X]`/`dict[K,V]`/`Literal[...]`/`Union[A, B, ...]`,
    `topics`). A bare name that is neither a scalar nor a constructor is a registry
    reference. The legacy engine spelling (`string`/`integer`/`number`/`boolean`) no
    longer parses (it resolves as an unknown registry RefExpr). `Union[...]` members are
    only checked to be records when the type is resolved.

    Args:
        s (`str`):
            The authored type expression, e.g. `"list[Signal]"` or `"Optional[date]"`.

    Returns:
        `TypeExpr`:
            The parsed AST node (`ScalarExpr`, `ListExpr`, `RefExpr`, `OptionalExpr`,
            `EnumExpr`, `DictExpr`, or `UnionExpr`) — still unresolved against any registry.

    Raises:
        `TypeCheckError`:
            The string is not a valid expression (including a lone Unicode surrogate or an
            empty `Literal` member), uses an unknown constructor, spells a
            union as `A | B` (the hint suggests `Union[A, B]`), or lists `None` as a
            `Union[...]` member (the hint suggests `Optional[Union[...]]`).

    Example:
        `parse_type("Union[Buy, Sell]")` gives
        `UnionExpr((RefExpr("Buy"), RefExpr("Sell")))`.
    """
    s = s.strip()
    try:
        node = ast.parse(s, mode="eval").body
    except SyntaxError as exc:
        raise TypeCheckError(f"malformed type expression {s!r}: {exc}") from exc
    except UnicodeEncodeError as exc:
        # `ast.parse` encodes its source as UTF-8, so a lone surrogate stops it here.
        raise TypeCheckError(
            f"type expression {s!r} contains a Unicode surrogate (types must be valid UTF-8 text)"
        ) from exc
    return _type_from_ast(node, s)


def _type_from_ast(node: "ast.expr", src: str) -> TypeExpr:
    # `None` arrives from `ast` as a constant, not a Name — it never reaches the Name arm.
    if isinstance(node, ast.Constant) and node.value is None:
        return ScalarExpr("None")
    if isinstance(node, ast.Name):
        nm = node.id
        if nm in ("Any", "any"):
            return ScalarExpr("object")
        if nm == "topics":
            return ListExpr(ScalarExpr("str"))  # domain alias (to be retired)
        if nm == "none":
            return ScalarExpr("None")  # tolerate the historical lowercase token
        if nm in _SCALAR_KEYWORDS:
            return ScalarExpr(nm)
        return RefExpr(nm)
    if isinstance(node, ast.Subscript):
        if not isinstance(node.value, ast.Name):
            raise TypeCheckError(f"malformed type expression {src!r}")
        ctor = node.value.id
        sl = node.slice
        if ctor in ("int", "float"):
            if not isinstance(sl, ast.Tuple) or len(sl.elts) != 2:
                raise TypeCheckError(f"numeric bounds require two endpoints in {src!r}")
            bounds = tuple(_numeric_endpoint(endpoint, src) for endpoint in sl.elts)
            try:
                _validate_numeric_bounds(_SCALAR_SEG[ctor], bounds)
            except (TypeError, ValueError) as exc:
                raise TypeCheckError(f"invalid bounds in {src!r}: {exc}") from exc
            return ScalarExpr(ctor, bounds=bounds)
        if ctor in ("list", "List"):
            return ListExpr(_type_from_ast(sl, src))
        if ctor == "Optional":
            return OptionalExpr(_type_from_ast(sl, src))
        if ctor == "Literal":
            return EnumExpr(_literal_tags(sl, src))
        if ctor in ("dict", "Dict"):
            return DictExpr()
        if ctor == "Union":
            elts = sl.elts if isinstance(sl, ast.Tuple) else [sl]
            if not elts:
                raise TypeCheckError(f"malformed type expression {src!r}")
            members = tuple(_type_from_ast(e, src) for e in elts)
            if ScalarExpr("None") in members:
                raise TypeCheckError(
                    f"None is not a union member in {src!r}; use Optional[Union[...]]"
                )
            return UnionExpr(members)
        raise TypeCheckError(f"unknown type constructor {ctor!r} in {src!r}")
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        operands = ", ".join(ast.unparse(operand) for operand in _bar_operands(node))
        raise TypeCheckError(
            f"malformed type expression {src!r}: use Union[{operands}] (| is not a type operator)"
        )
    raise TypeCheckError(f"malformed type expression {src!r}")


def _bar_operands(node: "ast.expr") -> list:
    """Flatten an `A | B | C` chain into its operands, left to right."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return _bar_operands(node.left) + _bar_operands(node.right)
    return [node]


def _numeric_endpoint(node: ast.expr, src: str) -> int | float:
    """Read one numeric literal with an optional unary sign, without evaluation."""
    sign = 1
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        sign = -1 if isinstance(node.op, ast.USub) else 1
        node = node.operand
    if not isinstance(node, ast.Constant) or type(node.value) not in (int, float):
        raise TypeCheckError(f"numeric bounds require literal endpoints in {src!r}")
    return sign * node.value


def _literal_tags(sl: "ast.expr", src: str) -> tuple:
    elts = sl.elts if isinstance(sl, ast.Tuple) else [sl]
    tags = []
    for e in elts:
        if isinstance(e, ast.Constant) and isinstance(e.value, str):
            if not e.value:
                raise TypeCheckError(
                    f"empty Literal member in {src!r} (tags must be non-empty names or quoted strings)"
                )
            if not encodes_as_utf8(e.value):
                # An escape such as "\uD800" in the quoted member decodes to a lone surrogate.
                raise TypeCheckError(
                    f"Literal member {e.value!r} in {src!r} contains a Unicode surrogate "
                    f"(tags must be valid UTF-8 text)"
                )
            tags.append(e.value)  # quoted member: Literal["red"]
        elif isinstance(e, ast.Name):
            tags.append(e.id)  # unquoted member: Literal[red]
        else:
            raise TypeCheckError(f"invalid Literal member in {src!r} (use names or quoted strings)")
    return tuple(tags)


def encodes_as_utf8(name: str) -> bool:
    """Return whether `name` encodes as UTF-8, i.e. holds no lone Unicode surrogate."""
    try:
        name.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _is_shadow(name: str) -> bool:
    """A typedef name must not shadow a scalar keyword or a typing constructor."""
    return name in _SCALAR_KEYWORDS or name.lower() in _SCALAR_KEYWORDS or name in _TYPING_CONSTRUCTORS


# ---------- Registry + resolution ----------


@dataclass(frozen=True)
class RecordDef:
    fields: dict[str, str]  # field name -> type reference string


@dataclass(frozen=True)
class VariantDef:
    tags: tuple[str, ...]


@dataclass(frozen=True)
class AliasDef:
    target: "TypeExpr"  # a transparent type abbreviation; resolved transitively (cycle-guarded)


TypeRegistry = dict[str, Union[RecordDef, VariantDef, AliasDef]]

_SCALAR_SEG = {
    "str": ValueKind.STRING,
    "int": ValueKind.INTEGER,
    "float": ValueKind.NUMBER,
    "bool": ValueKind.BOOLEAN,
    "date": ValueKind.DATE,
    "datetime": ValueKind.DATETIME,
    "object": ValueKind.OBJECT,
    "None": ValueKind.NONE,
    "file": ValueKind.FILE,
}

_LIST_SEG_FOR_ELEMENT = {
    ValueKind.STRING: ValueKind.LIST_STRING,
    ValueKind.INTEGER: ValueKind.LIST_INTEGER,
    ValueKind.NUMBER: ValueKind.LIST_NUMBER,
    ValueKind.BOOLEAN: ValueKind.LIST_BOOLEAN,
    ValueKind.OBJECT: ValueKind.LIST_OBJECT,
}


def resolve_type(t: TypeExpr, registry: TypeRegistry, _seen: frozenset[str] = frozenset()) -> Type:
    """Resolve a `TypeExpr` against the registry into a runtime `Type`.

    Expands records field-by-field and follows aliases transitively, guarding against
    reference cycles via `_seen`. A `Union[...]` flattens nested unions and aliases,
    drops duplicate members, and requires every member to be a non-nullable record. Its
    members must be told apart by a tag field or by shapes that can't overlap. A union
    with one distinct member is that member.

    Args:
        t (`TypeExpr`):
            A parsed type AST from [`parse_type`][yian.typesys.types.parse_type].
        registry (`TypeRegistry`):
            The per-flow `name -> RecordDef | VariantDef | AliasDef` table a `RefExpr`
            resolves against.
        _seen (`frozenset[str]`, *optional*, defaults to `frozenset()`):
            Registry names already being resolved on the current path — the cycle guard.
            Callers pass nothing; it is threaded internally through nested resolution.

    Returns:
        `Type`:
            The leaf structural type: a scalar, a typed list, a closed record, a
            tag-constrained variant, or a union of records with members in canonical
            order.

    Raises:
        `TypeCheckError`:
            A `RefExpr` names an unknown type, a record/alias reference is recursive, a
            union member is not a record or is nullable, or two union members can match
            the same value.

    Example:
        With `Buy: {action: Literal[buy], qty: int}` and
        `Sell: {action: Literal[sell], price: float}` registered,
        `resolve_type(parse_type("Union[Sell, Buy]"), registry)` gives a union whose
        members are `Buy` then `Sell`.
    """
    if isinstance(t, ScalarExpr):
        kind = _SCALAR_SEG[t.name]
        if t.bounds is not None:
            _validate_numeric_bounds(kind, t.bounds)
        return Type(kind=kind, bounds=t.bounds)

    if isinstance(t, OptionalExpr):
        return replace(resolve_type(t.inner, registry, _seen), nullable=True)

    if isinstance(t, EnumExpr):
        return Type(kind=ValueKind.STRING, tags=frozenset(t.tags))

    if isinstance(t, DictExpr):
        return Type.scalar(ValueKind.OBJECT)  # lenient (no key/value typing yet)

    if isinstance(t, UnionExpr):
        return _resolve_union(t, registry, _seen)

    if isinstance(t, ListExpr):
        elem = resolve_type(t.element, registry, _seen)
        if elem.tags is not None:  # list of variant -> list of strings
            list_seg = ValueKind.LIST_STRING
        elif elem.fields is not None:  # list of record -> list of objects
            list_seg = ValueKind.LIST_OBJECT
        else:
            list_seg = _LIST_SEG_FOR_ELEMENT.get(elem.kind, ValueKind.LIST_ANY)
        return Type(kind=list_seg, element=elem)

    # RefExpr
    name = t.name
    if name in _seen:
        raise TypeCheckError(f"recursive type reference: {name!r}")
    defn = registry.get(name)
    if defn is None:
        raise TypeCheckError(f"unknown type {name!r}")
    if isinstance(defn, AliasDef):
        return resolve_type(defn.target, registry, _seen | {name})  # transitive (cycle-guarded)
    if isinstance(defn, VariantDef):
        return Type(kind=ValueKind.STRING, tags=frozenset(defn.tags))
    fields = {
        f: resolve_type(parse_type(ts), registry, _seen | {name})
        for f, ts in defn.fields.items()
    }
    # an Optional[X] field (its resolved Type is nullable) is NOT required.
    required = frozenset(f for f, sh in fields.items() if not sh.nullable)
    return Type(kind=ValueKind.OBJECT, fields=fields, required=required)


def _resolve_union(t: UnionExpr, registry: TypeRegistry, _seen: frozenset[str]) -> Type:
    """Resolve a `Union[...]` into a record, or a union of distinguishable records."""
    label = _expr_label(t)
    members: list[tuple[str, str | None, Type]] = []
    for name, origin, member in _union_parts(t, registry, _seen):
        if member.nullable:
            raise TypeCheckError(
                f"{label}: member {name!r} is nullable; use Optional[Union[...]] instead"
            )
        if member.fields is None:
            raise TypeCheckError(f"{label}: member {name!r} is not a record")
        # `Type` holds dicts, so it is unhashable; dedupe by equality.
        same = next((m for m in members if m[2] == member), None)
        if same is None:
            members.append((name, origin, member))
        elif same[1] != origin:
            # Two differently named records with one shape can't be told apart.
            raise _overlap_error(label, same[0], name)
    if len(members) == 1:
        return members[0][2]
    union = Type(
        kind=ValueKind.OBJECT,
        variants=canonical_union_order(member for _, _, member in members),
    )
    if not union_tag_fields(union):
        for i, (a_name, _, a) in enumerate(members):
            for b_name, _, b in members[i + 1:]:
                if types_overlap(a, b):
                    raise _overlap_error(label, a_name, b_name)
    return union


def _overlap_error(label: str, a_name: str, b_name: str) -> TypeCheckError:
    """Build the error for two union members that can match the same value."""
    return TypeCheckError(
        f"{label}: members {a_name!r} and {b_name!r} can match the same value; add a tag "
        f"field every member declares as a single-value Literal, with a different value "
        f"in each member, e.g. kind: Literal[{a_name.lower()}]"
    )


def _union_parts(
    t: TypeExpr,
    registry: TypeRegistry,
    _seen: frozenset[str],
    name: str | None = None,
) -> list[tuple[str, str | None, Type]]:
    """Return a union's flattened `(display name, origin, resolved member)` triples.

    Nested unions and aliases are expanded so error messages can name each member the
    way it was written. The origin is the registry name an alias chain ends at, or
    `None` when the member is not a reference; it tells a repeated member from a
    differently named record of the same shape.
    """
    if isinstance(t, UnionExpr):
        parts: list[tuple[str, str | None, Type]] = []
        for member in t.members:
            parts.extend(_union_parts(member, registry, _seen))
        return parts
    if isinstance(t, RefExpr):
        if t.name in _seen:
            raise TypeCheckError(f"recursive type reference: {t.name!r}")
        defn = registry.get(t.name)
        if isinstance(defn, AliasDef):
            return _union_parts(defn.target, registry, _seen | {t.name}, name or t.name)
    origin = t.name if isinstance(t, RefExpr) else None
    return [(name or _expr_label(t), origin, resolve_type(t, registry, _seen))]


def _expr_label(t: TypeExpr) -> str:
    """Render a parsed type back to its authored spelling, for messages."""
    if isinstance(t, ScalarExpr):
        if t.bounds is None:
            return t.name
        return f"{t.name}[{t.bounds[0]}, {t.bounds[1]}]"
    if isinstance(t, RefExpr):
        return t.name
    if isinstance(t, ListExpr):
        return f"list[{_expr_label(t.element)}]"
    if isinstance(t, OptionalExpr):
        return f"Optional[{_expr_label(t.inner)}]"
    if isinstance(t, EnumExpr):
        return "Literal[" + ", ".join(t.tags) + "]"
    if isinstance(t, UnionExpr):
        return "Union[" + ", ".join(_expr_label(m) for m in t.members) + "]"
    return "dict"


def type_for(type_str: str, registry: TypeRegistry) -> Type:
    """Parse a type string and resolve it against the registry in one step.

    Args:
        type_str (`str`):
            The authored type expression.
        registry (`TypeRegistry`):
            The per-flow type table the reference resolves against.

    Returns:
        `Type`:
            The resolved leaf structural type.
    """
    return resolve_type(parse_type(type_str), registry)


# ---------- typedefs: registry builder ----------


def _refs_in_type(t: TypeExpr) -> set:
    """The registry names a parsed TypeExpr references (for the eager alias-cycle check)."""
    if isinstance(t, RefExpr):
        return {t.name}
    if isinstance(t, ListExpr):
        return _refs_in_type(t.element)
    if isinstance(t, OptionalExpr):
        return _refs_in_type(t.inner)
    if isinstance(t, UnionExpr):
        return set().union(*(_refs_in_type(m) for m in t.members))
    return set()  # scalars / enums / dict-placeholder reference nothing


def _typedef_parts(
    defn: Union[RecordDef, VariantDef, AliasDef],
) -> list[tuple[str | None, str | TypeExpr]]:
    """Return a typedef's `(field, type)` parts: each record field, or the alias target."""
    if isinstance(defn, RecordDef):
        return list(defn.fields.items())
    if isinstance(defn, AliasDef):
        return [(None, defn.target)]
    return []  # an enum's tags are already checked by parse_type


def _typedef_error(name: str, field: str | None, exc: TypeCheckError) -> TypeCheckError:
    """Prefix an error with the typedef (and field) it belongs to, and locate it there."""
    if field is None:
        return TypeCheckError(f"typedef {name!r}: {exc}", path=(name,))
    return TypeCheckError(f"typedef {name!r} field {field!r}: {exc}", path=(name, field))


def _check_local(t: TypeExpr, registry: TypeRegistry) -> None:
    """Check what a type expression says on its own: names it uses exist, bounds are valid."""
    if isinstance(t, ScalarExpr):
        if t.bounds is not None:
            _validate_numeric_bounds(_SCALAR_SEG[t.name], t.bounds)
    elif isinstance(t, RefExpr):
        if t.name not in registry:
            raise TypeCheckError(f"unknown type {t.name!r}")
    elif isinstance(t, ListExpr):
        _check_local(t.element, registry)
    elif isinstance(t, OptionalExpr):
        _check_local(t.inner, registry)
    elif isinstance(t, UnionExpr):
        for member in t.members:
            _check_local(member, registry)


def _check_typedefs(registry: TypeRegistry) -> None:
    """Resolve every typedef, so a broken one fails even when nothing uses it."""
    parsed: dict[str, list[tuple[str | None, TypeExpr]]] = {}
    # First each typedef's own text, so an error names the typedef that holds the
    # mistake rather than the first typedef that happens to use it.
    for name, defn in registry.items():
        parts = []
        for field, written in _typedef_parts(defn):
            try:
                t = parse_type(written) if isinstance(written, str) else written
                _check_local(t, registry)
            except TypeCheckError as exc:
                raise _typedef_error(name, field, exc) from exc
            parts.append((field, t))
        parsed[name] = parts

    # Then the checks that follow references (recursion, union members), dependencies
    # first for the same reason.
    order: list[str] = []
    visited: set[str] = set()

    def visit(name: str) -> None:
        if name in visited:
            return
        visited.add(name)
        for _, t in parsed[name]:
            for ref in sorted(_refs_in_type(t)):
                visit(ref)
        order.append(name)

    for name in registry:
        visit(name)
    for name in order:
        for field, t in parsed[name]:
            try:
                resolve_type(t, registry, frozenset({name}))
            except TypeCheckError as exc:
                raise _typedef_error(name, field, exc) from exc


def _check_alias_cycles(registry: TypeRegistry) -> None:
    """Reject an alias->alias cycle, located at the alias it was found from (record
    recursion is caught by `_check_typedefs`)."""
    def visit(start: str, name: str, path: frozenset) -> None:
        defn = registry.get(name)
        if not isinstance(defn, AliasDef):
            return
        for ref in _refs_in_type(defn.target):
            if ref in path:
                raise TypeCheckError(f"alias cycle through {ref!r} in typedefs", path=(start,))
            if ref in registry:
                visit(start, ref, path | {ref})

    for name, defn in registry.items():
        if isinstance(defn, AliasDef):
            visit(name, name, frozenset({name}))


def read_typedefs(raw: dict) -> TypeRegistry:
    """Build a `TypeRegistry` from a raw `typedefs:` mapping (Python-typing surface).

    A mapping value is a RECORD (`RecordDef`; field type-strings kept raw). A
    `Literal[...]` string is an ENUM (`VariantDef`, tag-only). Any other string is an
    ALIAS (`AliasDef`). A **sequence** is the dropped tagged-union spelling and is
    rejected: an all-bare-tags sequence must be a `Literal[...]` enum; a payload
    sequence must be a `Union[...]` of records. Names must be PascalCase and must not
    shadow a scalar/typing constructor. Every typedef is then resolved, even one
    nothing uses, so a bad field type, an unknown name, a recursive record, an alias
    cycle, or a bad `Union[...]` fails here.

    Args:
        raw (`dict`):
            The raw `typedefs:` mapping (`name -> definition`), or falsy for none.

    Returns:
        `TypeRegistry`:
            The `name -> RecordDef | VariantDef | AliasDef` table used by `resolve_type`.

    Raises:
        `TypeCheckError`:
            A name is not PascalCase or shadows a builtin, a field name is empty, not a
            string, or holds a Unicode surrogate, a field type is not a string, a value is
            a union sequence, or a typedef does not resolve (an unknown constructor or
            name, bad bounds, a recursive record, an alias cycle, or a bad `Union[...]`).
            The message names the typedef and field, and `path` is `(typedef,)` or
            `(typedef, field)` when the error belongs to one entry.

    Example:
        `read_typedefs({"Buy": {"action": "Literal[buy]"}, "Sell": {"action":
        "Literal[sell]"}, "Decision": "Union[Buy, Sell]"})` registers two records and an
        alias to their union.
    """
    registry: TypeRegistry = {}
    for name, value in (raw or {}).items():
        if not (isinstance(name, str) and name.isidentifier() and name[0].isupper()):
            raise TypeCheckError(
                f"typedef name {name!r} must be PascalCase (an uppercase identifier)",
                path=(name,) if isinstance(name, str) else (),
            )
        if _is_shadow(name):
            raise TypeCheckError(
                f"typedef name {name!r} shadows a builtin scalar or typing constructor",
                path=(name,),
            )
        if isinstance(value, dict):
            for fname, ftype in value.items():
                if type(fname) is not str or not fname:
                    raise TypeCheckError(
                        f"typedef {name!r} field {fname!r}: field names must be non-empty strings"
                    )
                if not encodes_as_utf8(fname):
                    raise TypeCheckError(
                        f"typedef {name!r} field {fname!r}: field names must not contain a "
                        f"Unicode surrogate"
                    )
                if not isinstance(ftype, str):
                    raise TypeCheckError(
                        f"typedef {name!r} field {fname!r}: type must be a string expression "
                        f"(got {type(ftype).__name__})",
                        path=(name, fname) if isinstance(fname, str) else (name,),
                    )
            registry[name] = RecordDef(fields=dict(value))
        elif isinstance(value, str):
            try:
                parsed = parse_type(value)
            except TypeCheckError as exc:
                raise _typedef_error(name, None, exc) from exc
            if isinstance(parsed, EnumExpr):
                registry[name] = VariantDef(tags=parsed.tags)
            else:
                registry[name] = AliasDef(target=parsed)
        elif isinstance(value, (list, tuple)):
            if all(isinstance(m, str) for m in value):
                members = ", ".join(str(m) for m in value)
                raise TypeCheckError(
                    f"typedef {name!r}: a tag-only union must be a Literal[...] enum, not a "
                    f"sequence (e.g. {name}: Literal[{members}])",
                    path=(name,),
                )
            raise TypeCheckError(
                f"typedef {name!r}: a tagged/payload union sequence is not supported; declare "
                f"each member as a record and write {name}: Union[A, B, ...]",
                path=(name,),
            )
        else:
            raise TypeCheckError(
                f"typedef {name!r}: unsupported definition of type {type(value).__name__}",
                path=(name,),
            )
    _check_alias_cycles(registry)
    _check_typedefs(registry)
    return registry
