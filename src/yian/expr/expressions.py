"""`${...}` evaluation plus conservative compile-time type/reference analysis.

Variable resolution goes through `VariablePool.resolve`, so expressions read typed
segments and can traverse into object outputs such as `${x.output.output.ratio}`.

Grammar: comparisons (`==` `!=` `<` `<=` `>` `>=` `in` `not in`) and arithmetic over
references and literals, combined with `and` / `or` / `not` and parentheses. The same
parsed tree feeds `eval_expr`, conservative `infer_expr_type`, and the reference walkers.
Conditions route through the ONE unified engine; a bare reference with no comparison
operator is a truthiness test (not an error). `isinstance(x, T)` checks a value against a
type named in the `types` table every evaluation entry point accepts.
"""

import inspect
import re
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Callable

from lark import Token, Tree

import yian.json_codec as json_codec
from yian.expr.lineage import ResolvedValue, SourceRef, TracedValue, ValueRoute
from yian.typesys.pool import VariablePool
from yian.typesys.values import Type, TypeCheckError, ValueKind, build_value_as


class ExpressionError(ValueError):
    """A `when:` expression could not be parsed or evaluated."""


class RequiredError(ExpressionError):
    """A `${ref:?message}` whose ref was unbound (the binder maps it to BindingError)."""


# The type-test special form `isinstance(x, T)`: its second argument is a type name
# looked up in the node's type table, never a reference, so the evaluator, the reference
# walkers and rewriters, and type inference treat only its first argument as an expression.
ISINSTANCE = "isinstance"

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
        value = value.get(step) if isinstance(value, dict) else None
    return value


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
        return float(t) if "." in t else json_codec.parse_int(t)
    if (t.startswith('"') and t.endswith('"')) or (t.startswith("'") and t.endswith("'")):
        return t[1:-1]
    raise ExpressionError(f"cannot parse literal {token!r}")


def _operand_repr(value: Any) -> str:
    """Return `repr(value)`, or a placeholder naming its type when it can't be rendered."""
    try:
        return repr(value)
    except ValueError:
        # repr of an int past Python's str-digit limit raises, at any nesting depth.
        return f"<{type(value).__name__} too large to display>"


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


def parse_condition(expression: str) -> "Tree | Token":
    """Parse an expression-field value (`when:`/`on:`/an `asserts:` entry) via the ONE
    unified grammar, honouring the both-spellings entry rule.

    An expression field may be written two ways, which PARSE (and later evaluate)
    IDENTICALLY:
      - a bare expression `a > 5` (no braces) — canonical,
      - a whole `${a > 5}` (braces around the whole expression).
    If the WHOLE trimmed value is a single `${...}` span, strip the outer braces and
    parse the interior; otherwise parse the whole value directly. A `${...}` inside the
    expression (`${a} > 5`) is an error that shows the bare rewrite.

    Every call in the parsed expression must name a builtin (`TEMPLATE_FNS`, or the
    `isinstance` special form) that takes its arguments. The check is static: it applies
    even to a call `and`/`or` would never evaluate.

    Args:
        expression (`str`):
            The field value, in either spelling.

    Returns:
        `Tree | Token`:
            The parsed expression.

    Raises:
        `ExpressionError`:
            Raised when the value does not parse or holds a `${...}` inside it, calls a
            function that isn't a builtin, or calls a builtin with arguments it doesn't take.
    """
    from yian.expr.grammar import parse_expr, whole_span_interior

    interior = whole_span_interior(expression)
    tree = parse_expr(expression.strip() if interior is None else interior)
    _check_condition_calls(tree)
    return tree


def _check_condition_calls(tree: "Tree | Token") -> None:
    """Require every call in a parsed condition to name a builtin that takes its arguments.

    `isinstance` is skipped: the grammar already checks its shape.

    Args:
        tree (`Tree | Token`): A tree from `parse_expr`.

    Raises:
        `ExpressionError`: If a callee is not a builtin, or a builtin's signature can't bind
            the call's positional count and keyword names.
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
            raise ExpressionError(f"unknown expression function {callee!r}")
        try:
            signature = inspect.signature(fn)
        except (TypeError, ValueError):
            continue  # e.g. a C builtin a host registered; it is checked when called
        positional = [arg for arg in call_suffix.children if len(arg.children) == 1]
        keywords = [str(arg.children[0]) for arg in call_suffix.children if len(arg.children) == 2]
        repeated = next((k for i, k in enumerate(keywords) if k in keywords[:i]), None)
        if repeated is not None:  # the grammar accepts `f(a=1, a=2)`; binding would collapse it
            raise ExpressionError(f"expression function {callee!r} got keyword {repeated!r} twice")
        try:
            signature.bind(*([None] * len(positional)), **dict.fromkeys(keywords))
        except TypeError as exc:
            raise ExpressionError(
                f"expression function {callee!r} takes ({_parameter_list(signature)}): {exc}"
            ) from None


def _parameter_list(signature: inspect.Signature) -> str:
    """Render a builtin's parameters for a message: `value, indent=2`.

    Args:
        signature (`inspect.Signature`): The builtin's signature.

    Returns:
        `str`: The parameter names, comma-separated, with `name=<repr default>` for optional
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
    tree = parse_condition(expression)
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
        symbol = {
            "add": "+", "sub": "-", "mul": "*", "div": "/", "mod": "%",
            "power": "**", "or_expr": "or", "and_expr": "and",
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
            return float(s) if "." in s else json_codec.parse_int(s)
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
        if name == ISINSTANCE:
            value = self._isinstance(call_suffix)
        else:
            value = self._call_builtin(name, call_suffix)
        after = [c for c in children if isinstance(c, Tree) and c.data == "trailer"]
        for tr in after:
            value = self._dotted_step(value, str(tr.children[0]))
        if value is _MISSING:
            self._miss_path = f"{name}(...)" + "".join(f".{tr.children[0]}" for tr in after)
        if self._trace:
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
        `item` scope locally (dict-only — see `_dotted_step`); every other path is
        handed WHOLE to `resolve` (the pool-agnostic seam — the
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
                    present = (
                        present and isinstance(value, dict)
                        and dict.__contains__(value, step)
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
        """One dotted step — dict `.get` ONLY, never `getattr` (SAFETY-CRITICAL: so
        `${x.__class__}` can never reach a Python attribute). A non-dict, a missing key,
        or a `None`-valued key -> `_MISSING` (the unresolved-reference sentinel), so a
        dotted miss on ANY object — a plain ref value OR a builtin-call result — is treated
        identically by every value consumer (strict raise / non-strict None / falsy). Also
        used to walk into an already-`_MISSING` value (a chained miss stays `_MISSING`)."""
        if isinstance(value, dict):
            stepped = value.get(step)
            return _MISSING if stepped is None else stepped
        return _MISSING

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

    def _call_builtin(self, callee: str, call_suffix: Tree) -> Any:
        """Dispatch a builtin call through `TEMPLATE_FNS`. Positional and keyword
        `arg`s are distinguished by shape (a kwarg carries a leading NAME token).
        A builtin failure is wrapped as `ExpressionError`."""
        from yian.expr.builtins import TEMPLATE_FNS

        fn = TEMPLATE_FNS.get(callee)
        if fn is None:
            raise ExpressionError(f"unknown expression function {callee!r}")
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
    def _binop(self, node: Tree, op: Callable[[Any, Any], Any], symbol: str) -> Any:
        lhs = self._as_value(self._walk(node.children[0]))
        rhs = self._as_value(self._walk(node.children[1]))
        try:
            value = op(lhs, rhs)
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
        if self._trace and symbol == "+" and type(lhs) is list and type(rhs) is list:
            self._frames[-1].route = ValueRoute(
                "concat", children=tuple(self._frames[-1].children),
                expression=self._describe(node, self._frames[-1].children), sizes=(len(lhs), len(rhs)),
            )
        elif self._trace and symbol == "*" and (
            (type(lhs) is list and isinstance(rhs, int))
            or (isinstance(lhs, int) and type(rhs) is list)
        ):
            content_index = 0 if type(lhs) is list else 1
            content, count = (lhs, rhs) if content_index == 0 else (rhs, lhs)
            self._frames[-1].route = ValueRoute(
                "repeat", children=tuple(self._frames[-1].children),
                expression=self._describe(node, self._frames[-1].children),
                sizes=(len(content),), count=int.__int__(count),
                content_index=content_index,
            )
        return value

    def _do_add(self, node: Tree) -> Any:
        return self._binop(node, _value_add, "+")

    def _do_sub(self, node: Tree) -> Any:
        return self._binop(node, lambda a, b: a - b, "-")

    def _do_mul(self, node: Tree) -> Any:
        return self._binop(node, lambda a, b: a * b, "*")

    def _do_div(self, node: Tree) -> Any:
        return self._binop(node, lambda a, b: a / b, "/")

    def _do_mod(self, node: Tree) -> Any:
        return self._binop(node, lambda a, b: a % b, "%")

    def _do_power(self, node: Tree) -> Any:
        return self._binop(node, lambda a, b: a ** b, "**")

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
    literals, coalesce / default / required, pure builtin calls, and the
    `isinstance(x, T)` type test. The operators are Python's (`%` formats text, a
    bool counts as 0/1).

    Three semantics support a CASE's `else:` routing:
    a missing reference is `None` (in the non-strict modes); arithmetic over that
    `None` raises a loud `ExpressionError` (a wrapped `TypeError`); an ordered
    comparison with a `None` operand is `False` (missing->falsy).

    Dotted access is dict-key lookup ONLY — never `getattr` — so `${x.__class__}`
    can never reach a Python attribute (SAFETY-CRITICAL). Builtin calls dispatch
    through `expr.builtins.TEMPLATE_FNS` only (no arbitrary callables).

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
            The MAP-body-local scope for `item` / `item.path` (dict-only walk). `None`
            means no item scope. With tracing enabled, a `ResolvedValue` may
            supply the item's presence and route through a composed collection.
        mode (`ResolveMode`, *optional*, defaults to `BINDING_NONE`):
            How a missing reference is treated — raise (`STRICT_RAISE`) or become
            `None` (`BINDING_NONE` / `CONDITION_FALSY`).
        trace (`bool`, *optional*, defaults to `False`):
            Internal retention switch. Collect a compositional route during this
            evaluation and return `TracedValue(value, route)`.
        types (`dict[str, Type] | None`, *optional*, defaults to `None`):
            The type table `isinstance(x, T)` looks `T` up in, keyed by type name.
            `None` means no types, so any `isinstance` raises.

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
) -> "Type | None":
    """Return the statically proven type of one parsed expression, if any.

    Inference is deliberately conservative: literal tokens, exact references,
    comparisons, boolean
    combinators, and `isinstance(x, T)` have fixed result types. Other forms remain
    unknown until their runtime value semantics can be modeled without guessing.
    Reference lookup is injected so this source-neutral expression layer does not
    know about a flow graph or Compose document.

    Args:
        tree (`Tree | Token`):
            A tree or token returned by ``parse_expr``.
        ref_type_fn (`Callable[[str], Type | None]`):
            Resolves one dotted reference path to its known type, or returns
            ``None`` when the type is not statically known. Any exception raised
            by the callback propagates unchanged.

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
            # `parse_expr` rejects dotted access on an `isinstance(...)` result.
            is_type_test = str(children[0]) == ISINSTANCE
            return Type.scalar(ValueKind.BOOLEAN) if is_type_test else None
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
