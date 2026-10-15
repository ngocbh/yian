"""`${...}` evaluation plus conservative compile-time type/reference analysis.

Variable resolution goes through `VariablePool.resolve`, so expressions read typed
segments and can traverse into object outputs such as `${x.output.output.ratio}`.

Grammar: comparisons (`==` `!=` `<` `<=` `>` `>=` `in` `not in`) and arithmetic over
references and literals, combined with `and` / `or` / `not` and parentheses. The same
parsed tree feeds `eval_expr`, conservative `infer_expr_type`, and the reference walkers.
Conditions route through the ONE unified engine; a bare reference with no comparison
operator is a truthiness test (not an error). `isinstance(x, T)` checks a value against a
type named in the `types` table every evaluation entry point accepts, and a constructor
`T(x)` / `T(a=..., b=...)` checks (or builds and checks) a value of that type.
"""

import inspect
import operator
import re
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Callable

from lark import Token, Tree

from yian.expr.lineage import ResolvedValue, SourceRef, TracedValue, ValueRoute
from yian.typesys.pool import VariablePool
from yian.typesys.types import BUILTIN_TYPE_NAMES
from yian.typesys.values import (
    FieldRead,
    IntTooLongError,
    Type,
    TypeCheckError,
    ValueKind,
    build_value_as,
    field_of,
    normalize_files,
    parse_int_text,
    union_field,
    value_repr,
    write_as_text,
)


class ExpressionError(ValueError):
    """A `when:` expression could not be parsed or evaluated."""


class RequiredError(ExpressionError):
    """A `${ref:?message}` whose ref was unbound (the binder maps it to BindingError)."""


class ConstructorTypeError(ExpressionError):
    """A `${T(...)}` constructor's value does not fit `T` (the binder's `binding.type`)."""


# The type-test special form `isinstance(x, T)`: its second argument is a type name
# looked up in the node's type table, never a reference, so the evaluator, the reference
# walkers and rewriters, and type inference treat only its first argument as an expression.
ISINSTANCE = "isinstance"


def conversion_hint(name: str) -> str:
    """Return the hint for calling a builtin type name, else `""`.

    `int(x)` reads like a conversion, but constructors are `typedefs:` names that only
    check.

    Args:
        name (`str`): The callee.

    Returns:
        `str`: `"; types in expressions are typedef names; ..."` for a builtin type name
            (`int`, `list`, `Optional`), else empty.
    """
    if name not in BUILTIN_TYPE_NAMES:
        return ""
    return f"; types in expressions are typedef names; `{name}(...)` is not a conversion"


def unknown_function_message(name: str) -> str:
    """Return the error text for a call to `name`, which is no builtin or type.

    Args:
        name (`str`): The callee.

    Returns:
        `str`: The message, with `conversion_hint` for a builtin type name.
    """
    return f"unknown expression function {name!r}" + conversion_hint(name)


def constructor_shape_error(name: str, call_suffix: "Tree") -> "str | None":
    """Return why a call to constructor `name` has the wrong arguments, else `None`.

    A constructor takes keyword fields (`Buy(action="buy", conviction=x)`) or exactly one
    positional value (`Buy(x)`), never both.

    Args:
        name (`str`): The constructor name.
        call_suffix (`Tree`): The call's parsed `call_suffix`.

    Returns:
        `str | None`: The message, or `None` for a well-formed call.
    """
    args = call_suffix.children
    keywords = sum(1 for arg in args if len(arg.children) == 2)
    positional = len(args) - keywords
    if positional > 1 or (positional and keywords):
        return (
            f"{name}(...) takes keyword fields, as in {name}(name=value, ...), or one "
            f"value to check, as in {name}(x)"
        )
    return None


_NUMBER_RE = re.compile(r"^-?\d+(\.\d+)?$")


def resolve_reference(path: str, pool: VariablePool) -> Any:
    """Resolve a `${path}` reference against the pool (missing -> None)."""
    parts = [p for p in path.strip().split(".")]
    if not parts or any(not p for p in parts):
        raise ExpressionError(f"invalid variable path: '${{{path}}}'")
    head, *rest = parts
    return pool.resolve(head, rest)


def _resolve_in_record(path: str, record: dict) -> Any:
    """Resolve a `${path}` against a node's bound input record (dotted walk).

    Bare local-input refs only. A missing input, dotted miss, or `None` step returns
    `None`, which gives `when:` its missing-is-falsy behavior. Compose rejects
    undeclared names everywhere, and field reads a known type can't have in bindings,
    CASE conditions, LOOP predicates, and flow outputs. Asserts check less (a node
    assert's reads of its inputs and a flow assert's `${input.X.f}` check only the head
    name), so there a `None` can also be a misspelled field.
    """
    parts = [p.strip() for p in path.strip().split(".")]
    if not parts or any(not p for p in parts):
        raise ExpressionError(f"invalid variable path: '${{{path}}}'")
    value: Any = record.get(parts[0])
    for step in parts[1:]:
        value = field_of(value, step)[1]
    return value


def _int_literal(text: str) -> int:
    """Convert an int literal's text, rejecting one with more than `YIAN_MAX_INT_DIGITS` digits.

    `parse_expr` already rejects a long `NUMBER` token; this keeps a tree or text that
    skipped it from failing with Python's own digit-limit error.

    Args:
        text (`str`): Decimal digits with an optional ``-`` sign.

    Returns:
        `int`: The literal's value.

    Raises:
        `ExpressionError`: The literal has more than `YIAN_MAX_INT_DIGITS` significant digits,
            raised from the `IntTooLongError`.
    """
    try:
        return parse_int_text(text)
    except IntTooLongError as exc:
        raise ExpressionError(str(exc)) from exc


def _parse_literal(token: str) -> Any:
    t = token.strip()
    if not t:
        raise ExpressionError("empty literal")
    low = t.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    if low in ("null", "none"):
        return None
    if _NUMBER_RE.match(t):
        return float(t) if "." in t else _int_literal(t)
    if (t.startswith('"') and t.endswith('"')) or (t.startswith("'") and t.endswith("'")):
        return t[1:-1]
    raise ExpressionError(f"cannot parse literal {token!r}")


def _operand_repr(value: Any) -> str:
    """Return `value_repr(value)`: the repr cut for a message, or a placeholder."""
    return value_repr(value)


def _eval_comparison(op: str, lhs: Any, rhs: Any) -> bool:
    try:
        if op == "==":
            return lhs == rhs
        if op == "!=":
            return lhs != rhs
        if op in ("in", "not in"):
            try:
                contained = lhs in rhs
            except TypeError as exc:
                raise ExpressionError(f"`in` rhs is not iterable: {exc}") from exc
            return contained if op == "in" else not contained
        # ordered comparisons: None on either side -> False (propagate falsy)
        if lhs is None or rhs is None:
            return False
        if op == "<":
            return lhs < rhs
        if op == "<=":
            return lhs <= rhs
        if op == ">":
            return lhs > rhs
        if op == ">=":
            return lhs >= rhs
    except TypeError as exc:
        raise ExpressionError(
            f"type error evaluating {_operand_repr(lhs)} {op} {_operand_repr(rhs)}: {exc}"
        ) from exc
    raise ExpressionError(f"unsupported operator: {op!r}")


def parse_condition(
    expression: str, types: "Mapping[str, Any] | None" = None
) -> "Tree | Token":
    """Parse an expression-field value (`when:`/`on:`/an `asserts:` entry) via the ONE
    unified grammar, honouring the both-spellings entry rule.

    An expression field may be written two ways, which PARSE (and later evaluate)
    IDENTICALLY:
      - a bare expression `a > 5` (no braces) — canonical,
      - a whole `${a > 5}` (braces around the whole expression).
    If the WHOLE trimmed value is a single `${...}` span, strip the outer braces and
    parse the interior; otherwise parse the whole value directly. A `${...}` inside the
    expression (`${a} > 5`) is an error that shows the bare rewrite.

    Every call to a builtin (`TEMPLATE_FNS`) must pass arguments it takes. Any other
    callee except the `isinstance` special form is a `T(...)` constructor: with `types`,
    it must name a type in it; without, its name is left to the caller (compose resolves
    it against the flow's `typedefs:`). The check is static: it applies even to a call
    `and`/`or` would never evaluate.

    Args:
        expression (`str`):
            The field value, in either spelling.
        types (`Mapping[str, Any] | None`, *optional*, defaults to `None`):
            The type table constructors are looked up in, or `None` when it is not known
            yet.

    Returns:
        `Tree | Token`:
            The parsed expression.

    Raises:
        `ExpressionError`:
            Raised when the value does not parse or holds a `${...}` inside it, calls a
            builtin with arguments it doesn't take, or, with `types`, calls a function that
            is neither a builtin nor a type in `types`.
    """
    from yian.expr.grammar import parse_expr, whole_span_interior

    interior = whole_span_interior(expression)
    tree = parse_expr(expression.strip() if interior is None else interior)
    _check_calls(tree, types)
    return tree


def _check_calls(tree: "Tree | Token", types: "Mapping[str, Any] | None" = None) -> None:
    """Require every builtin call in a parsed expression to pass arguments it takes.

    `isinstance` is skipped: the grammar already checks its shape. Any other callee that
    is no builtin is a `T(...)` constructor, checked against `types` when given.

    Args:
        tree (`Tree | Token`):
            A tree from `parse_expr`.
        types (`Mapping[str, Any] | None`, *optional*, defaults to `None`):
            The type table constructor names must be in, or `None` to leave them to the
            caller.

    Raises:
        `ExpressionError`:
            Raised when a builtin's signature can't bind the call's positional count and
            keyword names, or, with `types`, a callee is neither a builtin nor in it.
    """
    from yian.expr.builtins import TEMPLATE_FNS

    if not isinstance(tree, Tree):
        return
    for node in tree.iter_subtrees_topdown():
        if node.data != "refcall":
            continue
        call_suffix = next(
            (c for c in node.children if isinstance(c, Tree) and c.data == "call_suffix"), None
        )
        callee = str(node.children[0])
        if call_suffix is None or callee == ISINSTANCE:
            continue
        fn = TEMPLATE_FNS.get(callee)
        if fn is None:
            # A constructor: one unknown-function message, here or where compose resolves
            # the name against the flow's typedefs.
            if types is not None and callee not in types:
                raise ExpressionError(unknown_function_message(callee))
            continue
        try:
            signature = inspect.signature(fn)
        except (TypeError, ValueError):
            continue  # e.g. a C builtin a host registered; it is checked when called
        positional = [arg for arg in call_suffix.children if len(arg.children) == 1]
        keywords = [str(arg.children[0]) for arg in call_suffix.children if len(arg.children) == 2]
        try:
            signature.bind(*([None] * len(positional)), **dict.fromkeys(keywords))
        except TypeError as exc:
            raise ExpressionError(
                f"expression function {callee!r} takes ({_parameter_list(signature)}): {exc}"
            ) from None


def _parameter_list(signature: inspect.Signature) -> str:
    """Render a builtin's parameters for a message: `value, indent=2`.

    Args:
        signature (`inspect.Signature`):
            The builtin's signature.

    Returns:
        `str`:
            The parameter names, comma-separated, with `name=<repr default>` for optional
            ones and `*name`/`**name` for variadics.
    """
    parts = []
    for p in signature.parameters.values():
        if p.kind is p.VAR_POSITIONAL:
            parts.append(f"*{p.name}")
        elif p.kind is p.VAR_KEYWORD:
            parts.append(f"**{p.name}")
        elif p.default is p.empty:
            parts.append(p.name)
        else:
            parts.append(f"{p.name}={p.default!r}")
    return ", ".join(parts)


def condition_refs(expression: str) -> list[str]:
    """The reference PATHS a condition-field value reads (`when:`/`until:`/`while:`/`asserts:`).

    The condition analog of the template ref-walk ([`expr_refs_of`][yian.expr.template.expr_refs_of])
    and the prompt ref-walk ([`prompt_refs`][yian.expr.template.prompt_refs]): parse the
    WHOLE value as ONE expression (via `parse_condition`, honouring both spellings — bare
    `a > 5` and whole `${a > 5}`) and collect its reference leaves via the shared
    [`expr_refs`][yian.expr.expressions.expr_refs]. Every compile-time consumer of a
    condition's refs (flow/node `asserts:`, a loop `while:`/`until:` predicate, a searched `case`
    `when:`) routes here, so both spellings extract IDENTICALLY — unlike a flat `${...}` regex,
    which reads a whole-span as one bogus path and a bare form as no refs at all.

    Args:
        expression (`str`): the condition source — any expression the grammar accepts, any spelling.

    Returns:
        `list[str]`: the reference paths (dotted, `#`/`/` preserved) in source order, NOT deduped.

    Raises:
        `ExpressionError`: on a malformed condition, or one that calls a function that isn't a
            builtin or calls a builtin with arguments it doesn't take (the caller frames it as a
            located load error).
    """
    return expr_refs(parse_condition(expression))



def _evaluate(
    expression: str,
    resolve: Callable[[str], Any],
    types: "dict[str, Type] | None" = None,
) -> bool:
    """Parse + evaluate a `when:`/`asserts:` expression on the UNIFIED engine, resolving
    each reference via `resolve` and each `isinstance` type name via `types`; the result
    is coerced to `bool`.

    Runs the unified `eval_expr` in `CONDITION_FALSY` mode — a missing reference is `None`
    and therefore falsy through comparisons, which lets a CASE's `else:` route handle
    absent values. The truthiness fold of `and`/`or` is coerced back to a
    plain `bool` here, so a `when:`/`asserts:` predicate always yields a boolean.
    """
    tree = parse_condition(expression, types or {})
    return bool(eval_expr(tree, resolve, mode=ResolveMode.CONDITION_FALSY, types=types))


def evaluate_when(
    expression: str, pool: VariablePool, types: "dict[str, Type] | None" = None,
) -> bool:
    """
    Evaluate a `when:` boolean expression against the variable pool.

    Parses and folds the expression on the ONE unified engine — comparisons
    (`==` `!=` `<` `<=` `>` `>=` `in` `not in`) and arithmetic over references and literals,
    combined with `and` / `or` / `not` — and coerces the result to a single boolean. This is
    the public pool-based path. Record-scoped CASE and LOOP predicates use
    `evaluate_when_record` instead.

    The value may be written two ways, evaluating IDENTICALLY: a bare expression `a > 5`
    or a whole `${a > 5}` (braces around the whole thing). A bare reference with no
    comparison operator is a TRUTHINESS test (coerced to bool), not an error.

    Args:
        expression (`str`):
            The `when:` source — any expression the unified grammar accepts, in either
            spelling above.
        pool (`VariablePool`):
            The variable pool each reference resolves against (`node`/`system` namespaces).
            A reference that resolves to `None` participates as a falsy value.
        types (`dict[str, Type] | None`, *optional*, defaults to `None`):
            The type table `isinstance(x, T)` looks `T` up in. `None` means no types.

    Returns:
        `bool`:
            The truth value of the expression.

    Raises:
        `ExpressionError`:
            If the expression is malformed, references an invalid path, applies an
            operator Python rejects for its operand types (e.g. `"a" + 1`) or can't
            complete (overflow), or names a type absent from `types`. A call to a
            function that isn't a builtin, or to a builtin with arguments it doesn't take,
            raises even where `and`/`or` would not evaluate the call.

    Example:
        ```python
        evaluate_when("score.output >= 0.5 and flag.output", pool)
        ```
    """
    return _evaluate(expression, lambda path: resolve_reference(path, pool), types)


def first_failing_assert(exprs, pool: VariablePool):
    """The first assert expression in `exprs` that is false against `pool`, else None.

    A compatibility helper for callers that evaluate several pool-scoped expressions.
    Each expression uses the `when:`/`asserts:` boolean grammar."""
    for expr in exprs:
        if not evaluate_when(expr, pool):
            return expr
    return None


def evaluate_when_record(
    expression: str, record: dict, types: "dict[str, Type] | None" = None,
) -> bool:
    """Evaluate a strict-CASE `when:` against the node's bound input record.

    `name` / `name.path` resolve against `record`; a miss -> None, compared as a
    plain value: `==` and ordered comparisons against a non-null value are false, and
    `!=` is true. A bare `name` miss is falsy, which gives `when:` its
    missing-is-falsy behavior. A None left side of `in`/`not in` is an ordinary value
    (`miss in [...]` is false); a None right side raises, since None is no container.
    Bare local-input refs only; pool namespaces are not in scope (they belong in the
    node's input `from:` bindings).

    cf. `render_template_record` (strict AGENT), which RAISES on the same dotted miss.
    `when:` deliberately stays falsy so the CASE's `else:` route can fire — so a dotted-key
    typo on a declared dict input routes to `else:` silently until the `types:` registry can
    validate dotted paths at compile time (the head is checked today, not the path).

    `types` is the table `isinstance(x, T)` looks `T` up in (`None` means no types).

    A call to a function that isn't a builtin, or to a builtin with arguments it doesn't take,
    raises `ExpressionError` even where `and`/`or` would not evaluate the call.
    """
    return _evaluate(expression, lambda path: _resolve_in_record(path, record), types)


# --------------------------------------------------------------------------- #
# The ONE unified evaluator — walks a tree from `expr.grammar.parse_expr` against
# a `resolve` callable. It is the single `${...}` evaluator: the binding path
# (coalesce/default/required), the `when:`/`asserts:` boolean path, and the prompt
# builtin-call path all parse with `parse_expr` and evaluate here.
# --------------------------------------------------------------------------- #


class ResolveMode(Enum):
    """How a MISSING reference (one `resolve` maps to `None`) is treated.

    - `STRICT_RAISE`: a missing ref is an error — raise `ExpressionError`. For a
      context where every reference must be present (a strict render). A miss on a
      non-final operand of `|`, `:-`, or `:?` is still `None`, so the operator fires.
    - `BINDING_NONE`: a missing ref is `None` — the binding contract (a coalesce /
      default may then fire, or the whole value is `None`).
    - `CONDITION_FALSY`: a missing ref is `None`, which is falsy through comparisons
      to preserve `when:` missing-is-falsy behavior. Same None as `BINDING_NONE`; the
      distinct name marks the intent at the call site (a `when:`/`asserts:` predicate).
    """

    STRICT_RAISE = "strict-raise"
    BINDING_NONE = "binding-none"
    CONDITION_FALSY = "condition-falsy"


# A sentinel telling a `refcall` miss apart from a resolved `None`, so `default`/
# `required` can fire on a genuine miss. Non-strict modes turn it into `None` at the
# operator boundary; strict mode raises.
class _Missing:
    """Marker for an unresolved reference (distinct from a resolved `None` value)."""


_MISSING = _Missing()


def _value_add(lhs: Any, rhs: Any) -> Any:
    """`+` on resolved VALUES: `str+str` concatenates and `list+list` extends.
    Arithmetic over `None` (a missing ref in a non-strict mode) propagates Python's
    `TypeError`, wrapped by the caller."""
    return lhs + rhs


# One step of an arithmetic chain, by operator symbol.
_ARITH_STEPS: dict[str, Callable[[Any, Any], Any]] = {
    "+": _value_add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv,
    "%": operator.mod,
    "**": operator.pow,
}

# The operator tokens between the operands of a `sum` / `product` / `power` chain. A literal
# operand is a `Token` too, so operators are told apart by these types, never by `Token`.
_ARITH_OP_TOKEN_TYPES = frozenset({"ADD_OP", "MUL_OP", "POW_OP"})


def _split_chain(node: Tree) -> tuple[list, list[str]]:
    """Split an arithmetic chain node into its operands and operator symbols.

    Args:
        node (`Tree`):
            A `sum`, `product` or `power` node from `parse_expr`.

    Returns:
        `tuple[list, list[str]]`:
            The operand subtrees and tokens in source order, and the operator symbols
            between them (one fewer than the operands).
    """
    operands, operators = [], []
    for child in node.children:
        if isinstance(child, Token) and child.type in _ARITH_OP_TOKEN_TYPES:
            operators.append(str(child))
        else:
            operands.append(child)
    return operands, operators


def _chain_text(operators: list[str], routes: "list[ValueRoute] | tuple[ValueRoute, ...]") -> str:
    """Describe a chain from its operand routes, e.g. `(a + b - c)`.

    Args:
        operators (`list[str]`):
            The operator symbols between the operands.
        routes (`list[ValueRoute] | tuple[ValueRoute, ...]`):
            The operand routes, one more than `operators`.

    Returns:
        `str`:
            The operand expressions joined by their operators, in parentheses.
    """
    parts = [routes[0].expression]
    for symbol, route in zip(operators, routes[1:]):
        parts += [symbol, route.expression]
    return "(" + " ".join(parts) + ")"


def _balanced_concat(routes: list[ValueRoute], sizes: list[int]) -> ValueRoute:
    """Trace a `+` chain of lists as a balanced tree of two-child `concat` routes.

    Two-child routes keep the run-record format of a single `left + right`, and the
    balanced split keeps a long chain's route about `log2(n)` deep.

    Args:
        routes (`list[ValueRoute]`):
            The operand routes, in order.
        sizes (`list[int]`):
            Each operand list's length.

    Returns:
        `ValueRoute`:
            The lone route for one operand, else a `concat` of the two halves, whose
            expression is the halves' operands joined by `+`.
    """
    if len(routes) == 1:
        return routes[0]
    middle = len(routes) // 2
    return ValueRoute(
        "concat",
        children=(
            _balanced_concat(routes[:middle], sizes[:middle]),
            _balanced_concat(routes[middle:], sizes[middle:]),
        ),
        expression=_chain_text(["+"] * (len(routes) - 1), routes),
        sizes=(sum(sizes[:middle]), sum(sizes[middle:])),
    )


def _repeat_route(
    operators: list[str], values: list, running: list, routes: list[ValueRoute], expression: str,
) -> "ValueRoute | None":
    """Trace a `* / %` chain that repeats one list as a single two-child `repeat` route.

    The chain repeats a list when exactly one operand is an exact `list`, every operator
    from the one just before it onward is `*`, every later operand is an `int`, and the
    running value before it (if any) is an `int`: then every binary step from the list
    onward is a list times an `int`.

    Args:
        operators (`list[str]`):
            The chain's operator symbols.
        values (`list`):
            The operand values, in order.
        running (`list`):
            The running value after each operand.
        routes (`list[ValueRoute]`):
            The operand routes, in order.
        expression (`str`):
            The whole chain's description.

    Returns:
        `ValueRoute | None`:
            The `repeat` route, whose count child is the lone other operand's route or a
            `scalar` over the other operands' routes valued at the effective count; or
            `None` when the chain does not repeat a list.
    """
    lists = [index for index, value in enumerate(values) if type(value) is list]
    if len(lists) != 1:
        return None
    at = lists[0]
    if any(symbol != "*" for symbol in operators[max(at - 1, 0):]):
        return None
    if not all(isinstance(value, int) for value in values[at + 1:]):
        return None
    if at > 0 and not isinstance(running[at - 1], int):
        return None
    # `int.__int__` reads the value without calling an `int` subclass's conversion hooks.
    others = [index for index in range(len(values)) if index != at]
    if len(values) == 2:
        count = int.__int__(values[others[0]])
        count_route = routes[others[0]]
    else:
        # Python clamps a negative count at every step (`[x] * -1 * -1 == []`), so the
        # count is the product of the clamped counts in evaluation order.
        count = max(int.__int__(running[at - 1]), 0) if at > 0 else 1
        for value in values[at + 1:]:
            count *= max(int.__int__(value), 0)
        count_routes = [routes[index] for index in others]
        count_route = replace(
            ValueRoute.literal(count), kind="scalar", children=tuple(count_routes),
            expression=_chain_text([operators[index - 1] for index in others[1:]], count_routes),
        )
    content_index = 0 if at == 0 else 1
    return ValueRoute(
        "repeat",
        children=(routes[at], count_route) if at == 0 else (count_route, routes[at]),
        expression=expression, sizes=(len(values[at]),), count=count,
        content_index=content_index,
    )


@dataclass
class _TraceFrame:
    """Routes collected by the current expression evaluation frame."""

    children: list[ValueRoute] = field(default_factory=list)
    route: ValueRoute | None = None


class _EvalExpr:
    """Recursive walker for a `parse_expr` tree, parameterised by `resolve`, `item`,
    and `mode`. Hand-written (not a Lark `Transformer`) so coalesce / default /
    required can SHORT-CIRCUIT — evaluate the fallback only when the head misses.

    A `refcall` reference evaluates to `_MISSING` on an unresolved path; every value
    consumer (operators, comparisons, the top-level entry) maps `_MISSING` to `None`
    for non-strict modes or raises for `STRICT_RAISE`."""

    def __init__(
        self, resolve: Callable[[str], Any], item: Any, mode: ResolveMode,
        *, trace: bool = False, types: "dict[str, Type] | None" = None,
    ):
        self._resolve = resolve
        self._item = item
        self._mode = mode
        self._types = types
        self._trace = trace
        self._frames: list[_TraceFrame] = []
        self._last_route: ValueRoute | None = None
        self._miss_path: str | None = None

    # --- top-level entry: coerce a `_MISSING` head to a value per mode --- #
    def eval(self, node: "Tree | Token") -> Any:
        value = self._as_value(self._walk(node))
        return TracedValue(value, self._last_route) if self._trace else value

    def _as_value(self, v: Any) -> Any:
        """Turn an internal `_MISSING` into a mode-appropriate value: `None` for the
        non-strict modes, a raised `ExpressionError` for `STRICT_RAISE`."""
        if v is _MISSING:
            if self._mode is ResolveMode.STRICT_RAISE:
                where = f" ${{{self._miss_path}}}" if self._miss_path else ""
                raise ExpressionError(f"unresolved reference{where} in expression")
            return None
        return v

    @staticmethod
    def _or_none(v: Any) -> Any:
        """Turn `_MISSING` into `None` in every mode: a `|` / `:-` / `:?` operand whose
        miss the operator itself handles."""
        return None if v is _MISSING else v

    # --- dispatch --- #
    def _walk(self, node: "Tree | Token") -> Any:
        if not self._trace:
            return self._dispatch(node)
        frame = _TraceFrame()
        self._frames.append(frame)
        try:
            value = self._dispatch(node)
            route = frame.route
            if route is None:
                captured = None if value is _MISSING else value
                route = replace(
                    ValueRoute.literal(captured), kind="scalar",
                    children=tuple(frame.children), expression=self._describe(node, frame.children),
                ) if frame.children else ValueRoute.literal(captured)
        finally:
            self._frames.pop()
        self._last_route = route
        if self._frames:
            self._frames[-1].children.append(route)
        return value

    def _describe(self, node: "Tree | Token", children: list[ValueRoute]) -> str:
        """Describe the evaluated operands without another expression walk."""
        kind = str(node.data)
        if kind in ("sum", "product", "power"):
            return _chain_text(_split_chain(node)[1], children)
        symbol = {
            "or_expr": "or", "and_expr": "and",
            "compare_in": "in", "compare_notin": "not in",
        }.get(kind)
        if kind == "compare":
            symbol = str(node.children[1])
        if symbol is not None:
            return "(" + f" {symbol} ".join(child.expression for child in children) + ")"
        if kind in ("neg", "negate"):
            prefix = "-" if kind == "neg" else "not "
            return f"({prefix}{children[0].expression})"
        return f"{kind}({', '.join(child.expression for child in children)})"

    def _dispatch(self, node: "Tree | Token") -> Any:
        if isinstance(node, Token):
            return self._token(node)
        handler = getattr(self, f"_do_{node.data}", None)
        if handler is None:
            raise ExpressionError(f"unsupported expression node {node.data!r}")
        return handler(node)

    def _token(self, tok: Token) -> Any:
        """A lone literal that inlined up to a top-level token."""
        if tok.type == "NUMBER":
            s = str(tok)
            return float(s) if "." in s else _int_literal(s)
        if tok.type == "STRING":
            return str(tok)[1:-1]
        if tok.type == "BOOL":
            return str(tok).lower() == "true"
        if tok.type == "NULL":
            return None
        raise ExpressionError(f"unexpected token {tok.type} {tok!r}")

    # --- references / calls (`refcall`) --- #
    def _do_refcall(self, node: Tree) -> Any:
        """A `refcall` is a bare NAME + `trailer`s (a reference) OR a NAME +
        `call_suffix` + trailing `trailer`s (a builtin call, dotted access on its
        result). See `grammar` — the leading child is the NAME; trailers before any
        `call_suffix` are already rejected by `parse_expr`."""
        children = node.children
        name = str(children[0])
        call_suffix = next(
            (c for c in children if isinstance(c, Tree) and c.data == "call_suffix"), None
        )
        if call_suffix is None:
            segments = [str(c.children[0]) for c in children[1:]]  # each trailer's NAME
            return self._reference(".".join([name, *segments]))
        # a builtin call: dispatch through TEMPLATE_FNS, then dotted access on the result.
        # A `.field` walk that misses yields `_MISSING` — the SAME sentinel a plain-ref
        # dotted miss yields (below) — so every value consumer treats a call-result miss
        # exactly as a ref miss (strict raises, non-strict maps to None). Without this the
        # miss leaked a plain `None`, which a multi-span concat stringified to "" (a silent
        # blank that broke the STRICT_RAISE floor).
        constructor = self._is_constructor(name)
        if name == ISINSTANCE:
            value = self._isinstance(call_suffix)
        elif constructor:
            value = self._construct(name, call_suffix)
        else:
            value = self._call_builtin(name, call_suffix)
        after = [c for c in children if isinstance(c, Tree) and c.data == "trailer"]
        for tr in after:
            value = self._dotted_step(value, str(tr.children[0]))
        if value is _MISSING:
            self._miss_path = f"{name}(...)" + "".join(f".{tr.children[0]}" for tr in after)
        if self._trace and constructor:
            self._frames[-1].route = self._constructor_route(name, call_suffix).project(
                tuple(str(tr.children[0]) for tr in after)
            )
        elif self._trace:
            routes = tuple(self._frames[-1].children)
            names = tuple(
                str(arg.children[0]) if len(arg.children) == 2 else ""
                for arg in _value_args(name, call_suffix)
            )
            args = ", ".join(
                f"{key}={route.expression}" if key else route.expression
                for key, route in zip(names, routes)
            )
            if name == ISINSTANCE:
                args += f", {_isinstance_type_name(call_suffix)}"
            suffix = "".join(f".{tr.children[0]}" for tr in after)
            self._frames[-1].route = replace(
                ValueRoute.literal(None if value is _MISSING else value),
                kind="opaque", children=routes, names=names,
                expression=f"{name}({args}){suffix}",
            )
        return value

    def _reference(self, path: str) -> Any:
        """Resolve a dotted reference path. An `item`-headed path walks the MAP-body
        `item` scope locally (dict keys and file fields — see `_dotted_step`); every
        other path is handed WHOLE to `resolve` (the pool-agnostic seam — the
        resolver owns the dotted walk, exactly as `resolve_reference` / `_resolve_in_record`
        do today). A miss (`item` miss, or `resolve` -> None) yields `_MISSING`."""
        parts = path.split(".")
        if parts[0] == "item":  # MAP-body-local scope (not a resolver head)
            item = self._item
            if self._trace and isinstance(item, ResolvedValue):
                present, value = item.present, item.value
                source = item.source
            else:
                present, value = item is not None, item
                source = None
            if value is None or not present:
                value = _MISSING
            for step in parts[1:]:
                if self._trace and source is not None:
                    # A dict's presence is read off its stored keys, so the subclass `get`
                    # that `_dotted_step` calls runs once per step and nothing else does.
                    present = present and (
                        dict.__contains__(value, step) if isinstance(value, dict)
                        else field_of(value, step)[0]
                    )
                value = self._dotted_step(value, step)
            if self._trace and source is None:
                self._frames[-1].route = replace(
                    ValueRoute.literal(None if value is _MISSING else value),
                    expression=path,
                )
            elif self._trace and present:
                route = self._resolved_route(item, "item")
                self._frames[-1].route = route.project(tuple(parts[1:]))
            if value is _MISSING:
                self._miss_path = path
            return value  # `_dotted_step` already yields `_MISSING` on a walk miss
        resolved = self._resolve(path)
        if self._trace:
            if not isinstance(resolved, ResolvedValue):
                raise TypeError("a traced resolver must return ResolvedValue")
            if resolved.present:
                self._frames[-1].route = self._resolved_route(resolved, path)
            resolved = resolved.value if resolved.present else None
        if resolved is None:
            self._miss_path = path
            return _MISSING
        return resolved

    def _resolved_route(self, resolved: ResolvedValue, path: str) -> ValueRoute:
        source = resolved.source
        if isinstance(source, SourceRef):
            return ValueRoute("source", source=source, expression=path)
        if isinstance(source, ValueRoute):
            return ValueRoute("selected", children=(source,), expression=path)
        return ValueRoute.literal(resolved.value)

    def _dotted_step(self, value: Any, step: str) -> Any:
        """One dotted step — `field_of` ONLY (a dict key, or a file's `uri` / `mime` /
        `name`), never `getattr` (SAFETY-CRITICAL: so `${x.__class__}` can never reach a
        Python attribute). Any other value, a missing field, or a `None`-valued field ->
        `_MISSING` (the unresolved-reference sentinel), so a dotted miss on ANY object — a
        plain ref value OR a builtin-call result — is treated identically by every value
        consumer (strict raise / non-strict None / falsy). Also used to walk into an
        already-`_MISSING` value (a chained miss stays `_MISSING`)."""
        stepped = field_of(value, step)[1]
        return _MISSING if stepped is None else stepped

    def _isinstance(self, call_suffix: Tree) -> bool:
        """`isinstance(x, T)`: whether the value of `x` satisfies the type named `T`.

        `T` is looked up in the type table, never resolved as a reference. The check is
        the variable pool's write-boundary check, so a value passes exactly when it
        could be written to a slot of type `T`. A missing `x` is checked as `None` in the
        non-strict modes and raises in `STRICT_RAISE`."""
        type_name = _isinstance_type_name(call_suffix)
        typ = (self._types or {}).get(type_name)
        if typ is None:
            raise ExpressionError(f"isinstance: unknown type {type_name!r}")
        value = self._as_value(self._walk(call_suffix.children[0].children[0]))
        try:
            build_value_as(typ, value)
        except TypeCheckError:
            return False
        return True

    def _is_constructor(self, name: str) -> bool:
        """Whether a call to `name` is a `${T(...)}` constructor.

        A builtin wins over a type of the same name, and a builtin type name
        (`int`) is never a constructor even when an `isinstance` put it in the table.
        """
        from yian.expr.builtins import TEMPLATE_FNS

        return (
            name != ISINSTANCE
            and name not in TEMPLATE_FNS
            and name not in BUILTIN_TYPE_NAMES
            and name in (self._types or {})
        )

    def _construct(self, name: str, call_suffix: Tree) -> Any:
        """`T(x)` checks the value of `x` against `T`; `T(a=..., b=...)` builds a record.

        The keyword form's record holds exactly the given fields. Either value is
        checked with the variable pool's write-boundary check and given back with
        only its file and object representation read as `T` reads it
        (`normalize_files`), so it is exactly what `!T ${x}` delivers. A missing `x`
        is checked as `None` in the non-strict modes and raises in `STRICT_RAISE`.

        Raises:
            ExpressionError: The call mixes keyword and positional arguments or has
                several positional ones (compose rejects this at load).
            ConstructorTypeError: The value does not fit `T`.
        """
        shape_error = constructor_shape_error(name, call_suffix)
        if shape_error is not None:
            raise ExpressionError(shape_error)
        args = call_suffix.children
        if len(args) == 1 and len(args[0].children) == 1:
            value = self._as_value(self._walk(args[0].children[0]))
        else:
            value = {
                str(arg.children[0]): self._as_value(self._walk(arg.children[1]))
                for arg in args
            }
        try:
            build_value_as(self._types[name], value)
        except TypeCheckError as exc:
            raise ConstructorTypeError(f"{name}(...): {exc}") from exc
        return normalize_files(self._types[name], value)

    def _constructor_route(self, name: str, call_suffix: Tree) -> ValueRoute:
        """The route of a constructor call, from its argument routes.

        `T(x)` selects its argument's route, so feedback passes through to `x`; the
        keyword form is a record route, so feedback on a field reaches that field's
        argument.
        """
        routes = tuple(self._frames[-1].children)
        args = call_suffix.children
        if len(args) == 1 and len(args[0].children) == 1:
            return ValueRoute(
                "selected", children=routes, expression=f"{name}({routes[0].expression})",
            )
        names = [str(arg.children[0]) for arg in args]
        expression = ", ".join(f"{key}={route.expression}" for key, route in zip(names, routes))
        return replace(ValueRoute.record(zip(names, routes)), expression=f"{name}({expression})")

    def _call_builtin(self, callee: str, call_suffix: Tree) -> Any:
        """Dispatch a builtin call through `TEMPLATE_FNS`. Positional and keyword
        `arg`s are distinguished by shape (a kwarg carries a leading NAME token).
        A builtin failure is wrapped as `ExpressionError`."""
        from yian.expr.builtins import TEMPLATE_FNS

        fn = TEMPLATE_FNS.get(callee)
        if fn is None:
            raise ExpressionError(unknown_function_message(callee))
        pos: list = []
        kw: dict = {}
        for arg in call_suffix.children:  # each child is an `arg` subtree
            kids = arg.children
            if len(kids) == 2:  # kwarg: leading NAME token, then the value expr
                kw[str(kids[0])] = self._as_value(self._walk(kids[1]))
            else:  # positional: a single value expr
                pos.append(self._as_value(self._walk(kids[0])))
        try:
            return fn(*pos, **kw)
        except ExpressionError:
            raise
        except Exception as exc:  # a builtin blew up (bad arity/type) — surface loudly
            raise ExpressionError(f"expression function {callee!r} failed: {exc}") from exc

    # --- coalesce / default / required (short-circuit) --- #
    def _select_route(self, expression: str) -> None:
        if self._trace:
            self._frames[-1].route = ValueRoute(
                "selected", children=(self._last_route,), expression=expression,
            )

    def _do_coalesce(self, node: Tree) -> Any:
        """`a | b | c` — first operand that resolves to a non-None value wins; a
        present falsy value (0 / "" / False) wins, only `None`/miss falls through. Only
        the last operand's miss is subject to `STRICT_RAISE`."""
        last = len(node.children) - 1
        for i, child in enumerate(node.children):
            walked = self._walk(child)
            v = self._as_value(walked) if i == last else self._or_none(walked)
            if v is not None:
                self._select_route("|")
                return v
        if self._trace:
            self._frames[-1].route = ValueRoute.literal(None)
        return None

    def _do_default_expr(self, node: Tree) -> Any:
        """`head :- fallback` — the fallback is evaluated ONLY when `head` misses or
        resolves to None. Only the fallback's miss is subject to `STRICT_RAISE`."""
        head, fallback = node.children
        v = self._or_none(self._walk(head))
        if v is None:
            v = self._as_value(self._walk(fallback))
        self._select_route(":-")
        return v

    def _do_required_expr(self, node: Tree) -> Any:
        """`head :? message` — raise `RequiredError` (with `message`) when `head`
        misses / is None, in every mode."""
        head, message = node.children
        v = self._or_none(self._walk(head))
        if v is None:
            raise RequiredError(self._as_value(self._walk(message)))
        self._select_route(":?")
        return v

    # --- boolean combinators --- #
    # The operator KEYWORD tokens interleaved among the operands. A value operand may ALSO
    # be a top-level `Token` (a bare NUMBER/STRING/BOOL/NULL literal),
    # so a blanket "drop every Token" filter would wrongly discard the operand — only these
    # keyword token TYPES are the operators to skip.
    _OP_TOKEN_TYPES = frozenset({"NOT", "AND", "OR"})

    def _operands(self, node: Tree) -> list:
        """The operand children of a boolean combinator node (drop the operator keywords)."""
        return [
            c for c in node.children
            if not (isinstance(c, Token) and c.type in self._OP_TOKEN_TYPES)
        ]

    def _do_negate(self, node: Tree) -> Any:
        return not self._as_value(self._walk(self._operands(node)[-1]))

    def _do_or_expr(self, node: Tree) -> Any:
        # Operands fold by Python truthiness, not bool-only, so non-boolean values such
        # as `0` and `"hit"` participate directly.
        return any(self._as_value(self._walk(c)) for c in self._operands(node))

    def _do_and_expr(self, node: Tree) -> Any:
        # Operands fold by Python truthiness, not bool-only (see `_do_or_expr`).
        return all(self._as_value(self._walk(c)) for c in self._operands(node))

    # --- comparisons (reuse `_eval_comparison` missing-is-falsy behavior) --- #
    def _do_compare(self, node: Tree) -> Any:
        lhs, op, rhs = node.children
        return _eval_comparison(str(op), self._as_value(self._walk(lhs)), self._as_value(self._walk(rhs)))

    def _do_compare_in(self, node: Tree) -> Any:
        lhs, _op, rhs = node.children
        return _eval_comparison("in", self._as_value(self._walk(lhs)), self._as_value(self._walk(rhs)))

    def _do_compare_notin(self, node: Tree) -> Any:
        lhs, _op, rhs = node.children
        return _eval_comparison("not in", self._as_value(self._walk(lhs)), self._as_value(self._walk(rhs)))

    # --- arithmetic — Python operators on resolved VALUES --- #
    def _apply_step(self, symbol: str, lhs: Any, rhs: Any) -> Any:
        """Apply one operator of an arithmetic chain to two resolved values.

        Args:
            symbol (`str`):
                The operator: `+`, `-`, `*`, `/`, `%` or `**`.
            lhs (`Any`):
                The left value: the running value of a left fold, or an operand.
            rhs (`Any`):
                The right value.

        Returns:
            `Any`:
                Python's result of the operator.

        Raises:
            `ExpressionError`:
                If Python raises, or `%` formats an int over the digit limit; the
                message names the operator and both values.
        """
        try:
            if symbol == "%" and isinstance(lhs, str):
                # Formatting writes the right side as text.
                return write_as_text(lambda right: lhs % right, rhs)
            return _ARITH_STEPS[symbol](lhs, rhs)
        except ZeroDivisionError as exc:
            l, r = _operand_repr(lhs), _operand_repr(rhs)
            raise ExpressionError(f"division by zero: {l} {symbol} {r}") from exc
        except TypeError as exc:
            l, r = _operand_repr(lhs), _operand_repr(rhs)
            raise ExpressionError(f"type error evaluating {l} {symbol} {r}: {exc}") from exc
        except OverflowError as exc:
            l, r = _operand_repr(lhs), _operand_repr(rhs)
            # Float `**` overflow carries an (errno, message) pair; args[-1] is the message.
            detail = exc.args[-1] if exc.args else exc
            raise ExpressionError(f"cannot compute {l} {symbol} {r}: {detail}") from exc
        except KeyError as exc:
            l, r = _operand_repr(lhs), _operand_repr(rhs)
            raise ExpressionError(f"cannot compute {l} {symbol} {r}: missing key {exc}") from exc
        except ValueError as exc:
            l, r = _operand_repr(lhs), _operand_repr(rhs)
            raise ExpressionError(f"cannot compute {l} {symbol} {r}: {exc}") from exc

    def _do_sum(self, node: Tree) -> Any:
        """A `+ -` or `* / %` chain, folded left to right.

        Each operand is walked just before its step is applied, so evaluation order,
        error precedence and each step's message (naming the running value) are those
        of the left-nested binary steps. The fold is a loop, so a chain of any length
        adds no recursion.
        """
        operands, operators = _split_chain(node)
        acc = self._as_value(self._walk(operands[0]))
        # The operand values and running values, kept only when tracing (for the route).
        values, running = [acc], [acc]
        for symbol, operand in zip(operators, operands[1:]):
            rhs = self._as_value(self._walk(operand))
            acc = self._apply_step(symbol, acc, rhs)
            if self._trace:
                values.append(rhs)
                running.append(acc)
        if self._trace:
            routes = list(self._frames[-1].children)
            if node.data == "sum" and all(symbol == "+" for symbol in operators) and all(
                type(value) is list for value in values
            ):
                self._frames[-1].route = _balanced_concat(routes, [len(v) for v in values])
            elif node.data == "product":
                route = _repeat_route(
                    operators, values, running, routes, self._describe(node, routes),
                )
                if route is not None:
                    self._frames[-1].route = route
        return acc

    _do_product = _do_sum

    def _do_power(self, node: Tree) -> Any:
        """A `**` chain: every operand walked left to right, then folded right to left.

        This keeps the right-nested binary steps' order (`a ** (b ** c)`): a strict
        miss in an earlier operand wins before a later one is walked, and the innermost
        step's error comes first.
        """
        values = [self._as_value(self._walk(operand)) for operand in _split_chain(node)[0]]
        acc = values[-1]
        for lhs in reversed(values[:-1]):
            acc = self._apply_step("**", lhs, acc)
        return acc

    def _do_neg(self, node: Tree) -> Any:
        v = self._as_value(self._walk(node.children[0]))
        try:
            return -v
        except TypeError as exc:
            raise ExpressionError(f"unary minus on non-number {_operand_repr(v)}: {exc}") from exc
        except (ValueError, OverflowError) as exc:
            raise ExpressionError(f"cannot compute -{_operand_repr(v)}: {exc}") from exc

    def _do_list_lit(self, node: Tree) -> Any:
        value = [self._as_value(self._walk(c)) for c in node.children]
        if self._trace:
            children = tuple(self._frames[-1].children)
            self._frames[-1].route = ValueRoute(
                "list", children=children,
                expression="[" + ", ".join(child.expression for child in children) + "]",
            )
        return value


def eval_expr(
    tree: "Tree | Token",
    resolve: Callable[[str], Any],
    item: Any = None,
    mode: ResolveMode = ResolveMode.BINDING_NONE,
    *,
    trace: bool = False,
    types: "dict[str, Type] | None" = None,
) -> Any:
    """
    Evaluate a unified `${...}` expression tree (from `expr.grammar.parse_expr`).

    The ONE evaluator for every `${...}` construct: references (bare / dotted /
    `${}`-wrapped / `item`-scoped), arithmetic and comparisons over VALUES (so
    `str+str` concatenates and `list+list` extends), boolean combinators, list
    literals, coalesce / default / required, pure builtin calls, the
    `isinstance(x, T)` type test, and `T(...)` constructors. The operators are Python's
    (`%` formats text, a bool counts as 0/1).

    Three semantics support a CASE's `else:` routing:
    a missing reference is `None` (in the non-strict modes); arithmetic over that
    `None` raises a loud `ExpressionError` (a wrapped `TypeError`); an ordered
    comparison with a `None` operand is `False` (missing->falsy).

    Dotted access reads only a dict key or a file's `uri`, `mime` or `name`
    (`field_of`) — never `getattr` — so `${x.__class__}` can never reach a Python
    attribute (SAFETY-CRITICAL). Builtin calls dispatch through
    `expr.builtins.TEMPLATE_FNS` only (no arbitrary callables).

    Args:
        tree (`Tree | Token`):
            The parsed expression. A lone atom (`${a}`, a bare literal) inlines to a
            top-level `Token`; every other construct is a `Tree` — both are handled.
        resolve (`Callable[[str], Any]`):
            Resolves a reference path to its value; a path it maps to `None` is a MISS
            (the pool-agnostic seam). `item`-headed paths bypass this (see `item`).
            With `trace=True`, returns one `ResolvedValue` containing the value,
            actual presence, and a `SourceRef` or existing `ValueRoute`.
        item (`Any`, *optional*, defaults to `None`):
            The MAP-body-local scope for `item` / `item.path` (dict keys and file
            fields). `None` means no item scope. With tracing enabled, a
            `ResolvedValue` may supply the item's presence and route through a composed
            collection.
        mode (`ResolveMode`, *optional*, defaults to `BINDING_NONE`):
            How a missing reference is treated — raise (`STRICT_RAISE`) or become
            `None` (`BINDING_NONE` / `CONDITION_FALSY`).
        trace (`bool`, *optional*, defaults to `False`):
            Internal retention switch. Collect a compositional route during this
            evaluation and return `TracedValue(value, route)`.
        types (`dict[str, Type] | None`, *optional*, defaults to `None`):
            The type table `isinstance(x, T)` and a `T(...)` constructor look `T` up
            in, keyed by type name. `None` means no types, so any `isinstance` raises
            and no call is a constructor.

    Returns:
        `Any`:
            The evaluated value (typed): a number / string / bool / list / dict, or
            `None`. With tracing enabled, the same value inside `TracedValue`.

    Raises:
        `ExpressionError`:
            On an unsupported node, an unknown builtin, a builtin failure, arithmetic
            over an incompatible operand (incl. `None`), an operator Python can't
            complete (overflow, a repetition count too large, a missing `%(key)s`), a
            strict-mode miss, or an `isinstance` type name absent from `types`.
        `ConstructorTypeError`:
            When a `T(...)` constructor's value does not fit `T`.
        `RequiredError`:
            When a `head :? message` required atom's head misses / is None.
        `TypeError`:
            When a traced resolver does not return `ResolvedValue`.
    """
    return _EvalExpr(resolve, item, mode, trace=trace, types=types).eval(tree)


# --------------------------------------------------------------------------- #
# Conservative compile-time type inference over the unified expression tree.
# --------------------------------------------------------------------------- #


def infer_expr_type(
    tree: "Tree | Token",
    ref_type_fn: Callable[[str], "Type | None"],
    type_fn: "Callable[[str], Type | None] | None" = None,
) -> "Type | None":
    """Return the statically proven type of one parsed expression, if any.

    Inference is deliberately conservative: literal tokens, exact references,
    comparisons, boolean combinators, and `isinstance(x, T)` have fixed result types,
    and a `T(...)` constructor has type `T` (its value is checked against `T`), with
    field reads on the result typed through `T`. Other forms remain unknown until their
    runtime value semantics can be modeled without guessing. Reference and constructor
    lookup are injected so this source-neutral expression layer does not know about a
    flow graph or Compose document.

    Args:
        tree (`Tree | Token`):
            A tree or token returned by ``parse_expr``.
        ref_type_fn (`Callable[[str], Type | None]`):
            Resolves one dotted reference path to its known type, or returns
            ``None`` when the type is not statically known. Any exception raised
            by the callback propagates unchanged.
        type_fn (`Callable[[str], Type | None] | None`, *optional*, defaults to `None`):
            Resolves a constructor name to its type, or returns ``None`` for a name
            that is no constructor. ``None`` leaves every constructor unknown.

    Returns:
        `Type | None`:
            The proven expression type, or ``None`` when inference is
            inconclusive.

    Examples:
        ```python
        inferred = infer_expr_type(
            parse_expr("answer == 42"),
            lambda _path: Type.scalar(ValueKind.INTEGER),
        )
        assert inferred == Type.scalar(ValueKind.BOOLEAN)
        ```
    """
    if isinstance(tree, Token):
        if tree.type == "NUMBER":
            kind = ValueKind.NUMBER if "." in str(tree) else ValueKind.INTEGER
            return Type.scalar(kind)
        literal_kinds = {
            "STRING": ValueKind.STRING,
            "BOOL": ValueKind.BOOLEAN,
            "NULL": ValueKind.NONE,
        }
        kind = literal_kinds.get(str(tree.type))
        return None if kind is None else Type.scalar(kind)

    if tree.data == "refcall":
        children = tree.children
        if any(
            isinstance(child, Tree) and child.data == "call_suffix"
            for child in children
        ):
            name = str(children[0])
            if name == ISINSTANCE:
                # `parse_expr` rejects dotted access on an `isinstance(...)` result.
                return Type.scalar(ValueKind.BOOLEAN)
            from yian.expr.builtins import TEMPLATE_FNS

            typ = None if type_fn is None or name in TEMPLATE_FNS else type_fn(name)
            trailers = [
                str(child.children[0])
                for child in children
                if isinstance(child, Tree) and child.data == "trailer"
            ]
            for field_name in trailers:
                if typ is None:
                    return None
                typ = _field_read_type(typ, field_name)
            return typ
        path = ".".join(
            [str(children[0]), *(str(child.children[0]) for child in children[1:])]
        )
        return ref_type_fn(path)

    if tree.data in {
        "compare",
        "compare_in",
        "compare_notin",
        "negate",
        "and_expr",
        "or_expr",
    }:
        return Type.scalar(ValueKind.BOOLEAN)

    return None


def _field_read_type(typ: Type, name: str) -> "Type | None":
    """Return the type of reading field `name` on a value of type `typ`, if proven.

    A record's field is `Optional` when the record is, or the field is not required;
    a union's field is the members' shared type when every member has it.
    """
    if typ.variants is not None:
        read = union_field(typ, name)
        if not isinstance(read, FieldRead) or read.type is None:
            return None
        field = read.type
    elif typ.fields is not None and name in typ.fields:
        field = typ.fields[name]
        if name not in (typ.required or frozenset()):
            field = replace(field, nullable=True)
    else:
        return None
    return replace(field, nullable=True) if typ.nullable else field


# --------------------------------------------------------------------------- #
# The ONE unified compile-time ref-walk — collects every reference-leaf path a
# `parse_expr` tree reads. It is the single ref collector: the binding ref-walk
# (`template.expr_refs_of`), the prompt ref-walk (`template.prompt_refs`), and the
# condition ref-collection all scan spans and delegate here.
# --------------------------------------------------------------------------- #


def _value_args(callee: str, call_suffix: Tree) -> list:
    """The `arg` subtrees of a call whose values are expressions.

    Every argument of a builtin is an expression; `isinstance(x, T)` has only `x`."""
    args = call_suffix.children
    return args[:1] if callee == ISINSTANCE else args


def _isinstance_type_name(call_suffix: Tree) -> str:
    """The type name of a validated `isinstance(x, T)` call suffix."""
    return str(call_suffix.children[1].children[0].children[0])


def isinstance_type_names(tree: "Tree | Token") -> list[str]:
    """Return the type names the `isinstance(x, T)` calls in a parsed expression name.

    Args:
        tree (`Tree | Token`):
            A tree from [`parse_expr`][yian.expr.grammar.parse_expr] or `parse_condition`,
            which have already checked each `isinstance` call's shape.

    Returns:
        `list[str]`:
            Each `T` in source order, without duplicates. `[]` for a lone token.

    Example:
        ```python
        isinstance_type_names(parse_expr("isinstance(x, Buy) or isinstance(y, int)"))
        # ["Buy", "int"]
        ```
    """
    if not isinstance(tree, Tree):
        return []
    names: list[str] = []
    for node in tree.iter_subtrees_topdown():
        if node.data != "refcall" or str(node.children[0]) != ISINSTANCE:
            continue
        call_suffix = node.children[-1]
        if isinstance(call_suffix, Tree) and call_suffix.data == "call_suffix":
            name = _isinstance_type_name(call_suffix)
            if name not in names:
                names.append(name)
    return names


def constructor_names(tree: "Tree | Token") -> list[str]:
    """Return the callees in a parsed expression that can only be `T(...)` constructors.

    A callee that is neither a builtin nor `isinstance` is a constructor name; compose
    resolves each against the flow's `typedefs:` and rejects any other name at load.

    Args:
        tree (`Tree | Token`):
            A tree from [`parse_expr`][yian.expr.grammar.parse_expr] or `parse_condition`.

    Returns:
        `list[str]`:
            Each name in source order, without duplicates. `[]` for a lone token.

    Example:
        ```python
        constructor_names(parse_expr("[Buy(d.output), upper(x), Sell(price=p)]"))
        # ["Buy", "Sell"]
        ```
    """
    from yian.expr.builtins import TEMPLATE_FNS

    if not isinstance(tree, Tree):
        return []
    names: list[str] = []
    for node in tree.iter_subtrees_topdown():
        if node.data != "refcall" or not any(
            isinstance(child, Tree) and child.data == "call_suffix" for child in node.children
        ):
            continue
        name = str(node.children[0])
        if name != ISINSTANCE and name not in TEMPLATE_FNS and name not in names:
            names.append(name)
    return names


def isinstance_subject(tree: "Tree | Token") -> "Tree | Token | None":
    """Return the tested expression `x` of an `isinstance(x, T)` call.

    Args:
        tree (`Tree | Token`):
            A parsed expression from [`parse_expr`][yian.expr.grammar.parse_expr] or
            `parse_condition`, which have already checked each `isinstance` call's shape.

    Returns:
        `Tree | Token | None`:
            The first argument's expression when `tree` itself is an `isinstance` call,
            else `None`.

    Example:
        ```python
        isinstance_subject(parse_expr("isinstance(d.output, Buy)"))  # the `d.output` refcall
        isinstance_subject(parse_expr("isinstance(d.output, Buy) and x"))  # None
        ```
    """
    if (
        not isinstance(tree, Tree)
        or tree.data != "refcall"
        or len(tree.children) != 2
        or str(tree.children[0]) != ISINSTANCE
    ):
        return None
    call_suffix = tree.children[1]
    if not isinstance(call_suffix, Tree) or call_suffix.data != "call_suffix":
        return None
    return call_suffix.children[0].children[0]


def expr_refs(tree: "Tree | Token") -> list[str]:
    """
    Collect every reference-leaf PATH a unified `${...}` expression tree reads.

    The single compile-time ref-walk over a tree from
    [`parse_expr`][yian.expr.grammar.parse_expr]. A `refcall` WITHOUT a
    `call_suffix` is a reference — its dotted path (leading NAME joined with each
    `trailer` segment) is collected. A `refcall` WITH a `call_suffix` is a builtin
    CALL: the callee name contributes NO ref, but each `arg` VALUE's refs ARE
    collected (recursively). `isinstance(x, T)` collects only `x`'s refs; `T` is a
    type name. Literals contribute nothing. Coalesce / default /
    required refs are all collected, including a default's refs.

    `item`-headed refs ARE collected here — matching the evaluator, which resolves
    them from the MAP-body-local scope. The CALLER (compose `_ref_producer`,
    `compose/edges/producers.py`) is responsible for skipping `item`-headed refs when minting
    edges; that rule is unchanged and stays at the caller.

    Args:
        tree (`Tree | Token`):
            The parsed expression. A lone atom (`${a}`, a bare literal) inlines to a
            top-level `Token`; every other construct is a `Tree` — both are handled.

    Returns:
        `list[str]`:
            The reference paths (dotted, `#` / `/` preserved) in source/traversal
            order. NOT deduped — every span-scanning caller (`expr_refs_of`,
            `prompt_refs`) unions these in source order.
    """
    refs: list[str] = []
    _collect_expr_refs(tree, refs)
    return refs


def _collect_expr_refs(node: "Tree | Token", refs: list[str]) -> None:
    """Recursive walk collecting reference paths into `refs` (source order, no dedupe)."""
    if isinstance(node, Token):
        return  # a lone NUMBER / STRING / BOOL / NULL literal names no reference
    if node.data == "refcall":
        # KEEP IN SYNC with `_do_refcall`'s refcall decomposition (and the condition-read walk
        # in `template._condition_read_leaves`) — all duplicate this
        # call_suffix-finding + path-join (leading NAME + trailer segments) logic.
        children = node.children
        call_suffix = next(
            (c for c in children if isinstance(c, Tree) and c.data == "call_suffix"), None
        )
        if call_suffix is None:  # a reference: leading NAME + each trailer's NAME
            segments = [str(c.children[0]) for c in children[1:]]
            refs.append(".".join([str(children[0]), *segments]))
            return
        # a builtin call: the callee contributes NO ref; walk each arg VALUE only.
        for arg in _value_args(str(children[0]), call_suffix):  # empty `fn()` -> skipped
            # a kwarg carries a leading NAME token; the value expr is the last child.
            _collect_expr_refs(arg.children[-1], refs)
        return
    for child in node.children:
        _collect_expr_refs(child, refs)


# --------------------------------------------------------------------------- #
# rewrite — the position-splicing analog of the ref-walk (rename reference leaves)
#
# `expr_refs` COLLECTS reference paths; these functions REWRITE them. The compile-time
# passes that re-namespace a source (call/loop/map inlining) or rebind refs to node-local
# params (searched `case`) route through the shared parse tree: only the
# reference LEAVES move, every operator / literal / builtin-callee / whitespace stays
# verbatim (spliced by source position), and both condition spellings rewrite identically.
# --------------------------------------------------------------------------- #


def rewrite_expr_refs(text: str, rename: Callable[[str], "str | None"]) -> str:
    """Rewrite the reference-leaf PATHS of ONE unified `${...}` expression `text`.

    The rewrite analog of [`expr_refs`][yian.expr.expressions.expr_refs]: it walks
    the SAME parse tree (`refcall` reference / builtin-call args), but instead
    of collecting each reference path it asks `rename(path)` for a replacement and splices it
    in by source position. `rename` returns the new dotted path, or `None` to leave that
    reference untouched. Everything that is NOT a reference leaf — operators, numbers, string
    and list literals, a builtin CALL's callee name, parentheses, whitespace — is preserved
    verbatim, as is the type name of an `isinstance(x, T)`. A `refcall` (`a.b`) is
    replaced with the new path.

    Args:
        text (`str`): the expression source (a bare expression, or a span interior).
        rename (`Callable[[str], str | None]`): `path -> new_path`, or `None` to keep.

    Returns:
        `str`: `text` with the renamed reference leaves spliced in.

    Raises:
        `ExpressionError`: if `text` does not parse (same front end as `parse_expr`).
    """
    from yian.expr.grammar import parse_expr

    tree = parse_expr(text)
    edits: list[tuple[int, int, str]] = []
    _collect_ref_edits(tree, rename, edits)
    # Splice right-to-left so earlier offsets stay valid as later ones are replaced.
    for start, end, replacement in sorted(edits, key=lambda e: e[0], reverse=True):
        text = text[:start] + replacement + text[end:]
    return text


def _collect_ref_edits(
    node: "Tree | Token",
    rename: Callable[[str], "str | None"],
    edits: "list[tuple[int, int, str]]",
) -> None:
    """Walk `node` collecting `(start, end, replacement)` splices for each renamed
    reference leaf. KEEP IN SYNC with `_collect_expr_refs` (and `template._condition_read_leaves`,
    which splices leaves the same way) — same refcall/builtin-arg
    decomposition, but capturing token source positions to edit in place."""
    if isinstance(node, Token):
        return  # a lone literal holds no reference
    if node.data == "refcall":
        children = node.children
        call_suffix = next(
            (c for c in children if isinstance(c, Tree) and c.data == "call_suffix"), None
        )
        if call_suffix is None:  # a reference: leading NAME + each trailer's NAME
            segments = [str(c.children[0]) for c in children[1:]]
            path = ".".join([str(children[0]), *segments])
            new = rename(path)
            if new is not None:
                start = children[0].start_pos
                last = children[-1]  # the last trailer Tree, or the leading NAME token itself
                end = last.children[0].end_pos if isinstance(last, Tree) else last.end_pos
                edits.append((start, end, new))
            return
        # a builtin call: the callee is NOT a reference; rewrite each arg VALUE only.
        for arg in _value_args(str(children[0]), call_suffix):
            _collect_ref_edits(arg.children[-1], rename, edits)
        return
    for child in node.children:
        _collect_ref_edits(child, rename, edits)


def rewrite_condition_refs(expression: str, rename: Callable[[str], "str | None"]) -> str:
    """Rewrite the reference leaves of a condition-field value (`when:`/`until:`/`while:`/
    `asserts:`/searched-`case` `when:`), honouring both spellings.

    The rewrite analog of [`condition_refs`][yian.expr.expressions.condition_refs]:
    a whole-span `${a > 5}` keeps its outer braces and rewrites the interior; a bare
    `a > 5` is rewritten in place. `rename(path)` returns the replacement
    path (or `None` to keep). The value is trimmed first (matching `parse_condition`)."""
    from yian.expr.grammar import whole_span_interior

    interior = whole_span_interior(expression)
    if interior is not None:
        return "${" + rewrite_expr_refs(interior, rename) + "}"
    return rewrite_expr_refs(expression.strip(), rename)
