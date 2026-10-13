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


TypeExpr = Union[ScalarExpr, ListExpr, RefExpr, OptionalExpr, EnumExpr, DictExpr]

_SCALAR_KEYWORDS = {
    "str", "int", "float", "bool", "date", "datetime", "object", "file", "None", "none",
}

# Names a typedef must not shadow (typing constructors + the generic builtins).
_TYPING_CONSTRUCTORS = {"Optional", "Literal", "Union", "List", "Dict", "list", "dict", "Any"}


def parse_type(s: str) -> TypeExpr:
    """Parse a type reference string into a `TypeExpr` AST via Python `ast` (eval mode).

    Accepts the one Python-surface vocabulary (`str`/`int`/`float`/`bool`/`Any`/`date`/
    `datetime`/`object`/`file`/`None`, `int[min,max]`/`float[min,max]`,
    `list[X]`/`Optional[X]`/`dict[K,V]`/`Literal[...]`,
    `topics`). A bare name that is neither a scalar nor a constructor is a registry
    reference. The legacy engine spelling (`string`/`integer`/`number`/`boolean`) no
    longer parses (it resolves as an unknown registry RefExpr).

    Args:
        s (`str`):
            The authored type expression, e.g. `"list[Signal]"` or `"Optional[date]"`.

    Returns:
        `TypeExpr`:
            The parsed AST node (`ScalarExpr`, `ListExpr`, `RefExpr`, `OptionalExpr`,
            `EnumExpr`, or `DictExpr`) — still unresolved against any registry.

    Raises:
        `TypeCheckError`:
            The string is not a valid expression, uses an unknown constructor, or is a
            `Union[...]` (unsupported — model a tagged variant as a discriminated record).
    """
    s = s.strip()
    try:
        node = ast.parse(s, mode="eval").body
    except SyntaxError as exc:
        raise TypeCheckError(f"malformed type expression {s!r}: {exc}") from exc
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
            raise TypeCheckError(
                f"Union types are not supported ({src!r}); model a tagged variant as a "
                f"discriminated record {{tag: Literal[...], ...}} routed by `case ... on tag`"
            )
        raise TypeCheckError(f"unknown type constructor {ctor!r} in {src!r}")
    raise TypeCheckError(f"malformed type expression {src!r}")


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
            tags.append(e.value)  # quoted member: Literal["red"]
        elif isinstance(e, ast.Name):
            tags.append(e.id)  # unquoted member: Literal[red]
        else:
            raise TypeCheckError(f"invalid Literal member in {src!r} (use names or quoted strings)")
    return tuple(tags)


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
    reference cycles via `_seen`.

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
            The leaf structural type: a scalar, a typed list, a closed record, or a
            tag-constrained variant.

    Raises:
        `TypeCheckError`:
            A `RefExpr` names an unknown type, or a record/alias reference is recursive.
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
    return set()  # scalars / enums / dict-placeholder reference nothing


def _check_alias_cycles(registry: TypeRegistry) -> None:
    """Reject an alias->alias cycle eagerly, located at the typedefs block (records are
    caught lazily by resolve_type's `_seen`)."""
    def visit(name: str, path: frozenset) -> None:
        defn = registry.get(name)
        if not isinstance(defn, AliasDef):
            return
        for ref in _refs_in_type(defn.target):
            if ref in path:
                raise TypeCheckError(f"alias cycle through {ref!r} in typedefs")
            if ref in registry:
                visit(ref, path | {ref})

    for name, defn in registry.items():
        if isinstance(defn, AliasDef):
            visit(name, frozenset({name}))


def read_typedefs(raw: dict) -> TypeRegistry:
    """Build a `TypeRegistry` from a raw `typedefs:` mapping (Python-typing surface).

    A mapping value is a RECORD (`RecordDef`; field type-strings kept raw, resolved
    lazily). A `Literal[...]` string is an ENUM (`VariantDef`, tag-only). Any other
    string is an ALIAS (`AliasDef`). A **sequence** is the dropped tagged-union spelling
    and is rejected: an all-bare-tags sequence must be a `Literal[...]` enum; a payload
    sequence must be a discriminated record routed by `case ... on tag`. Names must be
    PascalCase and must not shadow a scalar/typing constructor. Alias cycles are rejected
    eagerly.

    Args:
        raw (`dict`):
            The raw `typedefs:` mapping (`name -> definition`), or falsy for none.

    Returns:
        `TypeRegistry`:
            The `name -> RecordDef | VariantDef | AliasDef` table used by `resolve_type`.

    Raises:
        `TypeCheckError`:
            A name is not PascalCase or shadows a builtin, a field type is not a string,
            a value is a union sequence, or an alias cycle is detected.
    """
    registry: TypeRegistry = {}
    for name, value in (raw or {}).items():
        if not (isinstance(name, str) and name.isidentifier() and name[0].isupper()):
            raise TypeCheckError(f"typedef name {name!r} must be PascalCase (an uppercase identifier)")
        if _is_shadow(name):
            raise TypeCheckError(
                f"typedef name {name!r} shadows a builtin scalar or typing constructor"
            )
        if isinstance(value, dict):
            for fname, ftype in value.items():
                if not isinstance(ftype, str):
                    raise TypeCheckError(
                        f"typedef {name!r} field {fname!r}: type must be a string expression "
                        f"(got {type(ftype).__name__})"
                    )
            registry[name] = RecordDef(fields={str(k): v for k, v in value.items()})
        elif isinstance(value, str):
            parsed = parse_type(value)
            if isinstance(parsed, EnumExpr):
                registry[name] = VariantDef(tags=parsed.tags)
            else:
                registry[name] = AliasDef(target=parsed)
        elif isinstance(value, (list, tuple)):
            if all(isinstance(m, str) for m in value):
                members = ", ".join(str(m) for m in value)
                raise TypeCheckError(
                    f"typedef {name!r}: a tag-only union must be a Literal[...] enum, not a "
                    f"sequence (e.g. {name}: Literal[{members}])"
                )
            raise TypeCheckError(
                f"typedef {name!r}: a tagged/payload union sequence is not supported; model it "
                f"as a discriminated record {{tag: Literal[...], ...}} routed by `case ... on tag`"
            )
        else:
            raise TypeCheckError(
                f"typedef {name!r}: unsupported definition of type {type(value).__name__}"
            )
    _check_alias_cycles(registry)
    return registry
