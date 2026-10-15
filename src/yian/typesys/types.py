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
import functools
import io
import re
import tokenize
from dataclasses import dataclass, replace
from typing import Union

from yian.defaults import Defaults, current
from yian.typesys.values import (
    _BUILTIN_CONSTANTS,
    _NAME_PATTERN,
    IntTooLongError,
    TypeCheckError,
    ValueKind,
    Type,
    _validate_numeric_bounds,
    canonical_union_order,
    int_digits,
    literal_label,
    parse_int_text,
    type_levels,
    types_overlap,
    union_tag_fields,
)


# ---------- TypeExpr AST ----------


@dataclass(frozen=True)
class ScalarExpr:
    name: str  # a scalar keyword (see _SCALAR_KEYWORDS)
    bounds: tuple[int | float | None, int | float | None] | None = None


@dataclass(frozen=True)
class ListExpr:
    element: "TypeExpr"


@dataclass(frozen=True)
class RefExpr:
    name: str  # a registry name (record or variant)


@dataclass(frozen=True)
class OptionalExpr:
    inner: "TypeExpr"  # Optional(X) — the resolved Type is X's, marked nullable


@dataclass(frozen=True)
class EnumExpr:
    tags: tuple  # an inline Literal(...) enum (a tag-only variant)


@dataclass(frozen=True)
class DictExpr:
    # dict[K, V] resolves to a lenient object; key/value types are not enforced.
    pass


@dataclass(frozen=True)
class UnionExpr:
    members: tuple  # the authored `Union(...)` members, in authored order


TypeExpr = Union[ScalarExpr, ListExpr, RefExpr, OptionalExpr, EnumExpr, DictExpr, UnionExpr]

_SCALAR_KEYWORDS = {
    "str", "int", "float", "bool", "date", "datetime", "object", "file", "None", "none", "Any",
}

# Names a typedef must not shadow (typing constructors + the generic builtins).
_TYPING_CONSTRUCTORS = {"Optional", "Literal", "Union", "List", "Dict", "list", "dict", "Any"}

# Names a type expression may start with besides typedefs, derived from the sets
# above so they cannot drift. `topics` is a retired alias, so it is not listed.
BUILTIN_TYPE_NAMES = frozenset(_SCALAR_KEYWORDS | _TYPING_CONSTRUCTORS | {"any"})


def parse_type(s: str) -> TypeExpr:
    """Parse an authored type expression into a `TypeExpr` AST.

    The vocabulary is `str`/`int`/`float`/`bool`/`Any`/`date`/`datetime`/`object`/
    `file`/`None`, where `Any` is any JSON value except null and `object` is a JSON
    object, inclusive ranges `int(1..12)`/`float(0.0..1.0)`, and the
    constructors `list(X)`/`Optional(X)`/`dict(K, V)`/`Literal(a, "b c")`/
    `Union(A, B, ...)`, plus the `topics` alias. A bare name that is neither a
    scalar nor a constructor is a registry reference. `Union(...)` members are only
    checked to be records when the type is resolved. Text in the older bracket
    spelling (`Optional[str]`, `int[1, 12]`) is an error that shows the parenthesized
    spelling.

    Args:
        s (`str`):
            The authored type expression, e.g. `"list(Signal)"` or `"Optional(date)"`.

    Returns:
        `TypeExpr`:
            The parsed AST node (`ScalarExpr`, `ListExpr`, `RefExpr`, `OptionalExpr`,
            `EnumExpr`, `DictExpr`, or `UnionExpr`) — still unresolved against any registry.

    Raises:
        `TypeCheckError`:
            The string is not a valid expression (including unbalanced parentheses, a
            lone Unicode surrogate, or an empty `Literal` member), uses an unknown
            constructor or the lowercase `any`, gives a numeric range with no end, no
            `..`, or as two arguments, spells a union as `A | B` (the hint suggests
            `Union(A, B)`), or lists `None` as a `Union(...)` member (the hint suggests
            `Optional(Union(...))`), or holds a `[` outside a quoted string (the message
            shows the parenthesized spelling).

    Example:
        `parse_type("Union(Buy, Sell)")` gives
        `UnionExpr((RefExpr("Buy"), RefExpr("Sell")))`.
    """
    s = s.strip()
    if _has_unquoted_bracket(s):
        raise _bracket_error(s)
    if not encodes_as_utf8(s):
        raise TypeCheckError(
            f"type expression {s!r} contains a Unicode surrogate (types must be valid UTF-8 text)"
        )
    return _TypeParser(s).parse()


def _bracket_error(s: str) -> TypeCheckError:
    """Explain that types use parentheses, showing `s` in that spelling when it parses."""
    try:
        spelled = legacy_respelling(s)
    except TypeCheckError as exc:
        return TypeCheckError(f"types use parentheses, not brackets ({exc})")
    return TypeCheckError(f"types use parentheses: `{spelled}`")


def legacy_respelling(s: str) -> str:
    """Spell bracket-style type text in the parenthesized spelling, for error messages.

    A `dict[K, V]` becomes `object`, the lenient mapping it stood for.

    Args:
        s (`str`): Type text in the older bracket spelling, e.g. `"Optional[str]"`.

    Returns:
        `str`: The same type in the parenthesized spelling, e.g. `"Optional(str)"`.

    Raises:
        `TypeCheckError`: `s` is not a type in the bracket spelling either.
    """
    return format_type_expr(_dict_as_object(_parse_legacy_type(s)))


def _dict_as_object(expr: TypeExpr) -> TypeExpr:
    """Replace every `dict(K, V)` in a parsed type with `object`."""
    if isinstance(expr, DictExpr):
        return ScalarExpr("object")
    if isinstance(expr, ListExpr):
        return ListExpr(_dict_as_object(expr.element))
    if isinstance(expr, OptionalExpr):
        return OptionalExpr(_dict_as_object(expr.inner))
    if isinstance(expr, UnionExpr):
        return UnionExpr(tuple(_dict_as_object(member) for member in expr.members))
    return expr


def _has_unquoted_bracket(s: str) -> bool:
    """Whether `s` holds a `[` outside quoted strings, i.e. is in the bracket spelling."""
    quote = None
    escaped = False
    for char in s:
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
        elif char in "\"'":
            quote = char
        elif char == "[":
            return True
    return False


# NUMBER needs a digit after a dot, so `0..1` reads as `0`, `..`, `1`. Its digits are
# ASCII only, so `int(١..٣)` is not a range.
_TYPE_TOKEN = re.compile(
    r"""\s*(?:
        (?P<number>[+-]?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)
      | (?P<name>""" + _NAME_PATTERN + r""")
      | (?P<string>"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')
      | (?P<punct>\.\.|[(),|])
    )""",
    re.VERBOSE,
)

class _TypeParser:
    """Recursive-descent parser for the parenthesized type grammar."""

    def __init__(self, src: str) -> None:
        self.src = src
        self.tokens: list[tuple[str, str]] = []
        pos = 0
        while pos < len(src):
            match = _TYPE_TOKEN.match(src, pos)
            if match is None:
                # An unclosed quote often means a `{...}` mapping's comma split the type.
                raise self._unbalanced() if self._opens_unclosed() else self._malformed()
            kind = match.lastgroup
            self.tokens.append((kind, match.group(kind)))
            pos = match.end()
        self.index = 0
        self.depth = 0
        self.max_depth = current().max_type_nesting

    def parse(self) -> TypeExpr:
        t = self._type()
        if self._peek() == ")":
            raise self._unbalanced()
        if self._peek() is not None:
            raise self._malformed()
        return t

    def _peek(self) -> str | None:
        """The next token's text, or `None` at the end."""
        return self.tokens[self.index][1] if self.index < len(self.tokens) else None

    def _peek_kind(self) -> str | None:
        return self.tokens[self.index][0] if self.index < len(self.tokens) else None

    def _next(self) -> tuple[str, str]:
        if self.index >= len(self.tokens):
            raise self._ended()
        token = self.tokens[self.index]
        self.index += 1
        return token

    def _expect(self, text: str) -> None:
        if self._peek() is None and text == ")":
            raise self._unbalanced()
        if self._next()[1] != text:
            raise self._malformed()

    def _malformed(self, detail: str = "") -> TypeCheckError:
        suffix = f": {detail}" if detail else ""
        return TypeCheckError(f"malformed type expression {self.src!r}{suffix}")

    def _opens_unclosed(self) -> bool:
        """Whether the text opens more parentheses than it closes."""
        return self.src.count("(") > self.src.count(")")

    def _ended(self) -> TypeCheckError:
        """The error for text that ends before the type does."""
        return self._unbalanced() if self._opens_unclosed() else self._malformed()

    def _trailing_comma(self, ctor: str) -> None:
        """Reject a `,` directly before `)`."""
        if self._peek() == ")":
            raise self._malformed(f"remove the trailing comma in {ctor}(...)")

    def _unbalanced(self) -> TypeCheckError:
        return self._malformed(
            "unbalanced parentheses — if this type sits inside a `{...}` mapping, its comma "
            "split it; quote it or use block style"
        )

    def _type(self) -> TypeExpr:
        t = self._primary()
        if self._peek() == "|":
            operands = [t]
            while self._peek() == "|":
                self._next()
                operands.append(self._primary())
            names = ", ".join(format_type_expr(operand) for operand in operands)
            raise self._malformed(f"use Union({names}) (| is not a type operator)")
        return t

    def _primary(self) -> TypeExpr:
        kind, name = self._next()
        # `None` is the null type; `True` and `False` are no types.
        if kind != "name" or (name in _BUILTIN_CONSTANTS and name != "None"):
            raise self._malformed()
        if self._peek() != "(":
            return _named_type(name)
        self._next()
        self.depth += 1
        if self.depth > self.max_depth:
            raise TypeCheckError(
                f"type nested too deeply in {self.src[:40]!r}... "
                f"(at most {self.max_depth} levels)"
            )
        t = self._constructed(name)
        self.depth -= 1
        return t

    def _constructed(self, name: str) -> TypeExpr:
        """Read the arguments of constructor `name` after its `(`."""
        if name in ("int", "float"):
            return self._range(name)
        if name in ("list", "List"):
            return ListExpr(self._one_type(name))
        if name == "Optional":
            return OptionalExpr(self._one_type(name))
        if name == "Literal":
            return EnumExpr(self._literal_tags())
        if name in ("dict", "Dict"):
            self._args(name, 2)
            return DictExpr()
        if name == "Union":
            members = tuple(self._args(name, None))
            if ScalarExpr("None") in members:
                raise TypeCheckError(
                    f"None is not a union member in {self.src!r}; use Optional(Union(...))"
                )
            return UnionExpr(members)
        raise TypeCheckError(f"unknown type constructor {name!r} in {self.src!r}")

    def _one_type(self, ctor: str) -> TypeExpr:
        (t,) = self._args(ctor, 1)
        return t

    def _args(self, ctor: str, count: int | None) -> list[TypeExpr]:
        """Read comma-separated types up to `)`: exactly `count`, or one or more."""
        if self._peek() == ")":
            raise self._malformed(f"{ctor}(...) needs a type")
        args = [self._type()]
        while self._peek() == ",":
            self._next()
            self._trailing_comma(ctor)
            args.append(self._type())
        self._expect(")")
        if count is not None and len(args) != count:
            wanted = "one type" if count == 1 else f"{count} types"
            raise self._malformed(f"{ctor}(...) takes {wanted}")
        return args

    def _range(self, ctor: str) -> ScalarExpr:
        """Read `lo..hi)` after `int(` or `float(`: an inclusive range; either end may be open."""
        example = "int(1..12)" if ctor == "int" else "float(0.0..1.0)"
        low = self._endpoint()
        if self._peek() == ",":
            self._next()
            high = self._endpoint()
            bounds = (low, high)
            if bounds != (None, None) and _valid_bounds(ctor, bounds):
                example = f"{ctor}({'' if low is None else low}..{'' if high is None else high})"
            raise self._malformed(f"write a range: {example}")
        dots = self._peek() == ".."
        high = None
        if dots:
            self._next()
            high = self._endpoint()
        if self._peek() is None:
            raise self._ended()
        if self._peek() in (",", ".."):
            raise self._malformed(f"a range takes one lo..hi, e.g. {example}")
        if self._peek() != ")":
            raise TypeCheckError(f"numeric bounds require literal endpoints in {self.src!r}")
        if not dots:
            raise self._malformed(_range_needs_dots(ctor))
        if low is None and high is None:
            raise self._malformed(f"a range needs at least one end; write {ctor} for any {ctor}")
        self._expect(")")
        bounds = (low, high)
        try:
            _validate_numeric_bounds(_SCALAR_SEG[ctor], bounds)
        except (TypeError, ValueError) as exc:
            raise TypeCheckError(f"invalid bounds in {self.src!r}: {exc}") from exc
        return ScalarExpr(ctor, bounds=bounds)

    def _endpoint(self) -> int | float | None:
        """Read a numeric endpoint, or return `None` when the next token is not a number."""
        if self._peek_kind() != "number":
            return None
        text = self._next()[1]
        if re.fullmatch(r"[+-]?[0-9]+", text):
            return _int_endpoint(text)
        return float(text)

    def _literal_tags(self) -> tuple:
        tags = []
        while True:
            kind, text = self._next()
            if kind == "string":
                tags.append(_literal_string(text, self.src))
            elif kind == "name" and text not in _BUILTIN_CONSTANTS and self._peek() != "(":
                tags.append(text)
            elif text == ")" and not tags:
                raise self._malformed("Literal(...) needs a member")
            else:
                raise TypeCheckError(
                    f"invalid Literal member in {self.src!r} (use names or quoted strings)"
                )
            if self._peek() == ",":
                self._next()
                self._trailing_comma("Literal")
                continue
            self._expect(")")
            return tuple(tags)


def _range_needs_dots(ctor: str) -> str:
    """Spell the hint for a numeric range written without `..`."""
    if ctor == "int":
        examples = "int(1..12), int(0..), or int(..100)"
    else:
        examples = "float(0.0..1.0), float(0.0..), or float(..1.0)"
    return f"a numeric range needs `..`, e.g. {examples}"


def _valid_bounds(ctor: str, bounds: tuple) -> bool:
    """Whether `bounds` make a valid range for `ctor` (`int` or `float`)."""
    try:
        _validate_numeric_bounds(_SCALAR_SEG[ctor], bounds)
    except (TypeError, ValueError):
        return False
    return True


def _named_type(name: str) -> TypeExpr:
    """Read a bare type name: a scalar keyword, an alias, or a registry reference."""
    if name == "any":
        # `any` stays a builtin name (the expression function), so it never reads as a
        # registry reference.
        raise TypeCheckError("unknown type 'any'; write Any")
    if name == "topics":
        return ListExpr(ScalarExpr("str"))  # domain alias (to be retired)
    if name == "none":
        return ScalarExpr("None")  # tolerate the historical lowercase token
    if name in _SCALAR_KEYWORDS:
        return ScalarExpr(name)
    return RefExpr(name)


def _literal_string(text: str, src: str) -> str:
    """Decode one quoted `Literal` member and check it is a usable tag."""
    try:
        value = ast.literal_eval(text)
    except (SyntaxError, ValueError) as exc:
        raise TypeCheckError(f"malformed type expression {src!r}: {exc}") from exc
    _check_literal_tag(value, src)
    return value


def _check_literal_tag(value: str, src: str) -> None:
    """Reject an empty quoted `Literal` member or one holding a Unicode surrogate."""
    if not value:
        raise TypeCheckError(
            f"empty Literal member in {src!r} (tags must be non-empty names or quoted strings)"
        )
    if not encodes_as_utf8(value):
        # An escape such as "\uD800" in the quoted member decodes to a lone surrogate.
        raise TypeCheckError(
            f"Literal member {value!r} in {src!r} contains a Unicode surrogate "
            f"(tags must be valid UTF-8 text)"
        )


def _parse_legacy_type(s: str) -> TypeExpr:
    """Parse the older bracket spelling (`Optional[str]`, `int[1, 12]`) via Python `ast`.

    Only migration errors use it, to show the parenthesized spelling.
    """
    s = s.strip()
    _reject_long_int_tokens(s)
    try:
        node = ast.parse(s, mode="eval").body
    except SyntaxError as exc:
        raise TypeCheckError(f"malformed type expression {s!r}: {exc}") from exc
    except UnicodeEncodeError as exc:
        # `ast.parse` encodes its source as UTF-8, so a lone surrogate stops it here.
        raise TypeCheckError(
            f"type expression {s!r} contains a Unicode surrogate (types must be valid UTF-8 text)"
        ) from exc
    except (RecursionError, MemoryError) as exc:
        raise TypeCheckError(f"type nested too deeply in {s[:40]!r}...") from exc
    try:
        return _type_from_ast(node, s)
    except RecursionError as exc:
        raise TypeCheckError(f"type nested too deeply in {s[:40]!r}...") from exc


def _reject_long_int_tokens(s: str) -> None:
    """Raise if bracket-spelled type text holds an int literal past `YIAN_MAX_INT_DIGITS`.

    Runs before `ast.parse`, which would fail on a long decimal literal with Python's
    own digit-limit message, or build it when that limit is off.

    Args:
        s (`str`): Type text in the older bracket spelling.

    Raises:
        `TypeCheckError`: An int literal has more than `YIAN_MAX_INT_DIGITS` digits.
    """
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(s).readline))
    except (tokenize.TokenError, SyntaxError):
        return  # `ast.parse` reports the malformed text
    for token in tokens:
        if token.type != tokenize.NUMBER:
            continue
        text = token.string
        if text[:2].lower() in ("0x", "0o", "0b"):
            digits = int_digits(int(text, 0))
            if digits > current().max_int_digits:
                raise _long_range_end(IntTooLongError(digits))
        elif re.fullmatch(r"[0-9_]+", text):
            _int_endpoint(text)


def _int_endpoint(text: str) -> int:
    """Read an int range end, rejecting one with more than `YIAN_MAX_INT_DIGITS` digits.

    Args:
        text (`str`): The end's text, a decimal int with an optional sign.

    Returns:
        `int`: The end's value.

    Raises:
        `TypeCheckError`: The int is too long, raised from the `IntTooLongError`; the
            message leaves out the type text, which would repeat every digit.
    """
    try:
        return parse_int_text(text)
    except IntTooLongError as exc:
        raise _long_range_end(exc)


def _long_range_end(cause: IntTooLongError) -> TypeCheckError:
    """Build the error for a too-long int range end, with `cause` as its cause.

    Args:
        cause (`IntTooLongError`): The digit-count error.

    Returns:
        `TypeCheckError`: An error whose message leaves out the type text, which
            would repeat every digit.
    """
    error = TypeCheckError(f"numeric range end: {cause}")
    error.__cause__ = cause
    return error


def _type_from_ast(node: "ast.expr", src: str) -> TypeExpr:
    """Convert one node of a bracket-spelled type, parsed by `ast`, into a `TypeExpr`."""
    # `None` arrives from `ast` as a constant, not a Name — it never reaches the Name arm.
    if isinstance(node, ast.Constant) and node.value is None:
        return ScalarExpr("None")
    if isinstance(node, ast.Name):
        return _named_type(node.id)
    if isinstance(node, ast.Subscript):
        if not isinstance(node.value, ast.Name):
            raise TypeCheckError(f"malformed type expression {src!r}")
        ctor = node.value.id
        sl = node.slice
        if ctor in ("int", "float"):
            if not isinstance(sl, ast.Tuple) or len(sl.elts) != 2:
                raise TypeCheckError(
                    f"malformed type expression {src!r}: {_range_needs_dots(ctor)}"
                )
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
                    f"None is not a union member in {src!r}; use Optional(Union(...))"
                )
            return UnionExpr(members)
        raise TypeCheckError(f"unknown type constructor {ctor!r} in {src!r}")
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        operands = ", ".join(_legacy_label(operand, src) for operand in _bar_operands(node))
        raise TypeCheckError(
            f"malformed type expression {src!r}: use Union({operands}) (| is not a type operator)"
        )
    raise TypeCheckError(f"malformed type expression {src!r}")


def _bar_operands(node: "ast.expr") -> list:
    """Flatten an `A | B | C` chain into its operands, left to right."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return _bar_operands(node.left) + _bar_operands(node.right)
    return [node]


def _legacy_label(node: "ast.expr", src: str) -> str:
    """Spell one `A | B` operand in the parenthesized syntax, or as written if it is no type."""
    try:
        return format_type_expr(_type_from_ast(node, src))
    except TypeCheckError:
        return ast.unparse(node)


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
    """Read the members of a bracket-spelled `Literal[...]`."""
    elts = sl.elts if isinstance(sl, ast.Tuple) else [sl]
    tags = []
    for e in elts:
        if isinstance(e, ast.Constant) and isinstance(e.value, str):
            _check_literal_tag(e.value, src)
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
    "Any": ValueKind.ANY,
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
    reference cycles via `_seen`. A `Union(...)` flattens nested unions and aliases,
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
        With `Buy: {action: Literal(buy), qty: int}` and
        `Sell: {action: Literal(sell), price: float}` registered,
        `resolve_type(parse_type("Union(Sell, Buy)"), registry)` gives a union whose
        members are `Buy` then `Sell`.
    """
    if isinstance(t, ScalarExpr):
        kind = _SCALAR_SEG[t.name]
        if t.bounds is not None:
            _validate_numeric_bounds(kind, t.bounds)
        return Type(kind=kind, bounds=t.bounds)

    if isinstance(t, OptionalExpr):
        # Peel the whole chain in a loop so nested Optionals cost one frame, not one each.
        while isinstance(t, OptionalExpr):
            t = t.inner
        return replace(resolve_type(t, registry, _seen), nullable=True)

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
        f: resolve_type(_parse_field(ts), registry, _seen | {name})
        for f, ts in defn.fields.items()
    }
    # an Optional[X] field (its resolved Type is nullable) is NOT required.
    required = frozenset(f for f, sh in fields.items() if not sh.nullable)
    return Type(kind=ValueKind.OBJECT, fields=fields, required=required)


def _resolve_union(t: UnionExpr, registry: TypeRegistry, _seen: frozenset[str]) -> Type:
    """Resolve a `Union(...)` into a record, or a union of distinguishable records."""
    # The label is rendered only for an error: rendering recurses through the whole
    # expression, which a deeply nested union would pay at every level of resolution.
    members: list[tuple[str, str | None, Type]] = []
    for name, origin, member in _union_parts(t, registry, _seen):
        if member.nullable:
            raise TypeCheckError(
                f"{_expr_label(t)}: member {name!r} is nullable; use Optional(Union(...)) "
                f"instead"
            )
        if member.fields is None:
            raise TypeCheckError(f"{_expr_label(t)}: member {name!r} is not a record")
        # `Type` holds dicts, so it is unhashable; dedupe by equality.
        same = next((m for m in members if m[2] == member), None)
        if same is None:
            members.append((name, origin, member))
        elif same[1] != origin:
            # Two differently named records with one shape can't be told apart.
            raise _overlap_error(_expr_label(t), same[0], name)
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
                    raise _overlap_error(
                        _expr_label(t), a_name, b_name, files=_holds_file(a) or _holds_file(b)
                    )
    return union


def _holds_file(t: Type) -> bool:
    """Whether a record field of `t`, at any record depth, is typed `file`.

    Args:
        t (`Type`):
            A resolved union member, or one of its field types.

    Returns:
        `bool`:
            `True` when `t` is `file`, or a record or union with such a field. List
            elements are not walked, since any two lists already overlap.
    """
    if t.kind is ValueKind.FILE:
        return True
    if t.fields is not None:
        return any(_holds_file(field) for field in t.fields.values())
    return t.variants is not None and any(_holds_file(member) for member in t.variants)


def _overlap_error(
    label: str, a_name: str, b_name: str, *, files: bool = False
) -> TypeCheckError:
    """Build the error for two union members that can match the same value.

    Args:
        label (`str`):
            The union as written, e.g. `Union(Bull, Bear)`.
        a_name (`str`):
            One overlapping member's name.
        b_name (`str`):
            The other overlapping member's name.
        files (`bool`, *optional*, defaults to `False`):
            Whether either member holds a `file` field, which overlaps more than its
            name suggests, so the message says why.

    Returns:
        `TypeCheckError`:
            The load error naming both members and the tag field that separates them.
    """
    why = (
        " (a `file` slot reads `{uri, mime, name}`, so `file` overlaps `object`, `Any` "
        "and a record of that shape)" if files else ""
    )
    return TypeCheckError(
        f"{label}: members {a_name!r} and {b_name!r} can match the same value{why}; add a "
        f"tag field every member declares as a single-value Literal, with a different "
        f"value in each member, e.g. kind: {literal_label([a_name.lower()])}"
    )


def _union_parts(
    t: TypeExpr, registry: TypeRegistry, _seen: frozenset[str]
) -> list[tuple[str, str | None, Type]]:
    """Return a union's flattened `(display name, origin, resolved member)` triples.

    Nested unions and aliases are expanded so error messages can name each member the
    way it was written: a member reached through an alias chain is named by the chain's
    first alias, unless a union lies between. The origin is the registry name an alias
    chain ends at, or `None` when the member is not a reference; it tells a repeated
    member from a differently named record of the same shape.
    """
    # An explicit stack, so flattening deeply nested unions and long alias chains costs
    # no frames.
    # Each entry is (expression, names on its alias path, display name so far).
    parts: list[tuple[str, str | None, Type]] = []
    stack: list[tuple[TypeExpr, frozenset[str], str | None]] = [(t, _seen, None)]
    while stack:
        t, seen, name = stack.pop()
        if isinstance(t, UnionExpr):
            stack.extend((member, seen, None) for member in reversed(t.members))
            continue
        if isinstance(t, RefExpr):
            if t.name in seen:
                raise TypeCheckError(f"recursive type reference: {t.name!r}")
            defn = registry.get(t.name)
            if isinstance(defn, AliasDef):
                stack.append((defn.target, seen | {t.name}, name or t.name))
                continue
        origin = t.name if isinstance(t, RefExpr) else None
        parts.append((name or _expr_label(t), origin, resolve_type(t, registry, seen)))
    return parts


def format_type_expr(t: TypeExpr) -> str:
    """Render a parsed type in the parenthesized spelling `parse_type` reads back.

    Args:
        t (`TypeExpr`):
            A parsed type AST.

    Returns:
        `str`:
            The type text, e.g. `"Optional(list(Buy))"` or `"int(1..12)"`. A `dict(K, V)`
            renders as `dict`, since its key and value types are not kept.

    Example:
        `format_type_expr(parse_type("Optional(Union(Buy,Sell))"))` gives
        `"Optional(Union(Buy, Sell))"`.
    """
    if isinstance(t, ScalarExpr):
        if t.bounds is None:
            return t.name
        low, high = ("" if end is None else end for end in t.bounds)
        return f"{t.name}({low}..{high})"
    if isinstance(t, RefExpr):
        return t.name
    if isinstance(t, ListExpr):
        return f"list({format_type_expr(t.element)})"
    if isinstance(t, OptionalExpr):
        return f"Optional({format_type_expr(t.inner)})"
    if isinstance(t, EnumExpr):
        return literal_label(t.tags)
    if isinstance(t, UnionExpr):
        return "Union(" + ", ".join(format_type_expr(m) for m in t.members) + ")"
    return "dict"


def _expr_label(t: TypeExpr) -> str:
    """Render a parsed type back to its authored spelling, for messages."""
    return format_type_expr(t)


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

    Raises:
        `TypeCheckError`:
            `type_str` does not parse or resolve, or the resolved type nests more record
            and list levels than `max_type_nesting`: text within the limit can still
            name a typedef that brings it past.
    """
    typ = resolve_type(parse_type(type_str), registry)
    limit = current().max_type_nesting
    if type_levels(typ) > limit:
        raise TypeCheckError(
            f"type nests deeper than {limit} levels once its typedef names are expanded, "
            f"the limit for a type (each record and list is a level)"
        )
    return typ


# ---------- typedefs: registry builder ----------


def _parse_field(text: str) -> TypeExpr:
    """Parse a record field's type text, cached per text and defaults.

    Args:
        text (`str`):
            The field's authored type expression.

    Returns:
        `TypeExpr`:
            What [`parse_type`][yian.typesys.types.parse_type] returns for `text`.

    Raises:
        `TypeCheckError`:
            `text` is not a valid type expression (errors are not cached).
    """
    # The registry check parses every field near the top of the stack, so resolution,
    # which reaches a field deep inside nested records, finds it parsed instead of paying
    # the parser's frames there. Sharing a result is safe: type ASTs are frozen. The key
    # holds the whole `Defaults`, since every default `parse_type` reads (the nesting and
    # integer digit limits) must be part of it.
    return _parse_field_within(text, current())


@functools.lru_cache(maxsize=65536)
def _parse_field_within(text: str, defaults: Defaults) -> TypeExpr:
    """Parse `text`, cached per text and defaults."""
    return parse_type(text)


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


def _expr_levels(t: TypeExpr, levels: dict[str, int]) -> int:
    """Return how many record and list levels a parsed type nests.

    Each `list(...)` is a level, `Optional(...)` adds none, a union counts as its deepest
    member, and a name counts as `levels[name]` (0 when not yet measured, which happens
    only on a reference cycle). Other scalars, `Literal(...)` and `dict(...)` are 0.
    """
    # Levels are the most lists on any path to a leaf plus that leaf's levels, so an
    # explicit stack of (expression, lists above it) needs no recursion.
    deepest = 0
    stack: list[tuple[TypeExpr, int]] = [(t, 0)]
    while stack:
        t, above = stack.pop()
        if isinstance(t, ListExpr):
            stack.append((t.element, above + 1))
        elif isinstance(t, OptionalExpr):
            stack.append((t.inner, above))
        elif isinstance(t, UnionExpr):
            stack.extend((member, above) for member in t.members)
        else:
            leaf = levels.get(t.name, 0) if isinstance(t, RefExpr) else 0
            deepest = max(deepest, above + leaf)
    return deepest


def _measure_typedef(
    name: str,
    defn: Union[RecordDef, VariantDef, AliasDef],
    parts: list[tuple[str | None, TypeExpr]],
    levels: dict[str, int],
    chain: dict[str, int],
) -> tuple[int, int, str | None, str | None]:
    """Measure a typedef from the typedefs it names, and record the result.

    Args:
        name (`str`):
            The typedef's name; its results are stored under it in `levels` and `chain`.
        defn (`RecordDef | VariantDef | AliasDef`):
            The typedef's definition.
        parts (`list[tuple[str | None, TypeExpr]]`):
            Its parsed `(field, type)` parts: each record field, or the alias target.
        levels (`dict[str, int]`):
            The levels of the typedefs measured so far; a name not in it counts 0.
        chain (`dict[str, int]`):
            The chains of the typedefs measured so far; a name not in it counts 0.

    Returns:
        `tuple[int, int, str | None, str | None]`:
            Its levels (a record is 1 more than its deepest field, an alias its target's,
            an enum 0; equal to `type_levels` of the resolved type), its chain (the
            typedefs on the longest path of names from it, itself included), the first
            field with the most levels, and the first field with the longest chain. The
            fields are `None` for an alias or an enum.
    """
    field_levels = [(field, _expr_levels(t, levels)) for field, t in parts]
    field_chain = [
        (field, max((chain.get(ref, 0) for ref in _refs_in_type(t)), default=0))
        for field, t in parts
    ]
    deepest = max((d for _, d in field_levels), default=0)
    longest = max((c for _, c in field_chain), default=0)
    # An enum has no parts, so it measures 0 levels like the string it resolves to.
    levels[name] = (1 if isinstance(defn, RecordDef) else 0) + deepest
    chain[name] = 1 + longest
    levels_field = next((f for f, d in field_levels if d == deepest), None)
    chain_field = next((f for f, c in field_chain if c == longest), None)
    return levels[name], chain[name], levels_field, chain_field


def _check_measure(
    name: str,
    defn: Union[RecordDef, VariantDef, AliasDef],
    parts: list[tuple[str | None, TypeExpr]],
    levels: dict[str, int],
    chain: dict[str, int],
) -> None:
    """Measure a typedef and reject one whose levels or chain exceed `max_type_nesting`.

    Args:
        name (`str`):
            The typedef's name.
        defn (`RecordDef | VariantDef | AliasDef`):
            Its definition.
        parts (`list[tuple[str | None, TypeExpr]]`):
            Its parsed `(field, type)` parts.
        levels (`dict[str, int]`):
            The levels memo of [`_measure_typedef`][yian.typesys.types._measure_typedef].
        chain (`dict[str, int]`):
            The chain memo of [`_measure_typedef`][yian.typesys.types._measure_typedef].

    Raises:
        `TypeCheckError`:
            The typedef nests more record and list levels than the limit, or ends a
            longer chain of typedefs each naming the next. `path` is `(typedef, field)`
            for the record field reported, else `(typedef,)`.
    """
    limit = current().max_type_nesting
    depth, length, levels_field, chain_field = _measure_typedef(
        name, defn, parts, levels, chain
    )
    if depth > limit:
        message = (
            f"type nests deeper than {limit} levels, the limit for a type (each record and "
            f"list is a level)"
        )
        if isinstance(defn, RecordDef):
            message += f"; its type {defn.fields[levels_field]!r} adds {depth - 1}"
            raise _typedef_error(name, levels_field, TypeCheckError(message))
        raise _typedef_error(name, None, TypeCheckError(message))
    if length > limit:
        raise _typedef_error(
            name,
            chain_field,
            TypeCheckError(
                f"it ends a chain of {length} typedefs, each naming the next, more than "
                f"{limit}, the limit for a type; write its type with fewer typedefs in "
                f"between"
            ),
        )


def _check_typedefs(registry: TypeRegistry) -> None:
    """Resolve every typedef, so a broken one fails even when nothing uses it.

    Typedefs are checked dependencies first, each one's fields in order. Each typedef is
    first measured: its levels and its chain of typedef names must stay within
    `max_type_nesting`, so resolution's stack stays bounded by the limit. A cycle is
    reported at the field that closes it, naming the typedef that holds that field: with
    `A: {x: B}` and `B: {y: A}` the error is `typedef 'B' field 'y': recursive type
    reference: 'B'`.
    """
    parsed: dict[str, list[tuple[str | None, TypeExpr]]] = {}
    # First each typedef's own text, so an error names the typedef that holds the
    # mistake rather than the first typedef that happens to use it.
    for name, defn in registry.items():
        parts = []
        for field, written in _typedef_parts(defn):
            try:
                t = _parse_field(written) if isinstance(written, str) else written
                _check_local(t, registry)
            except TypeCheckError as exc:
                raise _typedef_error(name, field, exc) from exc
            parts.append((field, t))
        parsed[name] = parts

    # Then the checks that follow references (recursion, union members), dependencies
    # first for the same reason. The walk is iterative so a long chain can't overflow
    # the stack; a ref still on the walk's stack (gray) when met is a back edge, so the
    # typedef holding it lies on a cycle.
    order: list[str] = []
    finished: set[str] = set()
    back_refs: dict[str, set[str]] = {}

    def refs_of(name: str):
        return (ref for _, t in parsed[name] for ref in sorted(_refs_in_type(t)))

    # A typedef becomes a key of `back_refs` when the walk reaches it, so the keys are
    # also the set of typedefs already discovered.
    for root in registry:
        if root in back_refs:
            continue
        back_refs[root] = set()
        stack = [(root, refs_of(root))]
        while stack:
            name, refs = stack[-1]
            for ref in refs:
                if ref not in back_refs:
                    back_refs[ref] = set()
                    stack.append((ref, refs_of(ref)))
                    break
                if ref not in finished:
                    back_refs[name].add(ref)
            else:
                stack.pop()
                finished.add(name)
                order.append(name)

    levels: dict[str, int] = {}
    chain: dict[str, int] = {}
    for name in order:
        _check_measure(name, registry[name], parsed[name], levels, chain)
        for field, t in parsed[name]:
            if _refs_in_type(t) & back_refs[name]:
                exc = TypeCheckError(f"recursive type reference: {name!r}")
                raise _typedef_error(name, field, exc)
            try:
                resolve_type(t, registry, frozenset({name}))
            except TypeCheckError as exc:
                raise _typedef_error(name, field, exc) from exc


def _check_alias_cycles(registry: TypeRegistry) -> None:
    """Reject an alias->alias cycle, located at the alias it was found from (record
    recursion is caught by `_check_typedefs`)."""
    # An iterative walk, so a long alias chain can't overflow the stack; an alias whose
    # walk finished without a cycle is not walked again, so a chain costs linear time.
    finished: set[str] = set()

    def alias_refs(defn: AliasDef):
        return iter(sorted(_refs_in_type(defn.target)))

    for start, defn in registry.items():
        if not isinstance(defn, AliasDef) or start in finished:
            continue
        on_path = {start}
        stack = [(start, alias_refs(defn))]
        while stack:
            name, refs = stack[-1]
            for ref in refs:
                if ref in on_path:
                    raise TypeCheckError(f"alias cycle through {ref!r} in typedefs", path=(start,))
                target = registry.get(ref)
                if isinstance(target, AliasDef) and ref not in finished:
                    on_path.add(ref)
                    stack.append((ref, alias_refs(target)))
                    break
            else:
                stack.pop()
                on_path.discard(name)
                finished.add(name)


def read_typedefs(raw: dict) -> TypeRegistry:
    """Build a `TypeRegistry` from a raw `typedefs:` mapping (Python-typing surface).

    A mapping value is a RECORD (`RecordDef`; field type-strings kept raw). A
    `Literal(...)` string is an ENUM (`VariantDef`, tag-only). Any other string is an
    ALIAS (`AliasDef`). A **sequence** is the dropped tagged-union spelling and is
    rejected: an all-bare-tags sequence must be a `Literal(...)` enum; a payload
    sequence must be a `Union(...)` of records. Names must be PascalCase and must not
    shadow a scalar/typing constructor. Every typedef is then resolved, even one
    nothing uses, so a bad field type, an unknown name, a recursive record, an alias
    cycle, or a bad `Union(...)` fails here.

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
            a union sequence, a typedef nests more record and list levels than
            `max_type_nesting` or ends a longer chain of typedefs each naming the next, or
            a typedef does not resolve (an unknown constructor or name, bad bounds, a
            recursive record, an alias cycle, or a bad `Union(...)`).
            The message names the typedef and field, and `path` is `(typedef,)` or
            `(typedef, field)` when the error belongs to one entry.

    Example:
        `read_typedefs({"Buy": {"action": "Literal(buy)"}, "Sell": {"action":
        "Literal(sell)"}, "Decision": "Union(Buy, Sell)"})` registers two records and an
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
                raise TypeCheckError(
                    f"typedef {name!r}: a tag-only union must be a Literal(...) enum, not a "
                    f"sequence (e.g. {name}: {literal_label(value)})",
                    path=(name,),
                )
            raise TypeCheckError(
                f"typedef {name!r}: a tagged/payload union sequence is not supported; declare "
                f"each member as a record and write {name}: Union(A, B, ...)",
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
