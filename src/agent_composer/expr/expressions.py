"""`${...}` evaluation plus conservative compile-time type/reference analysis.

Ported from the legacy engine's lark grammar, with one change: variable
resolution now goes through `VariablePool.resolve`, so it reads typed
segments (and can traverse into object outputs — `${x.output.output.ratio}` —
which the old dict-of-str pool could not).

Grammar: comparisons (`==` `!=` `<` `<=` `>` `>=` `in` `not in`) and arithmetic over
references and literals, combined with `and` / `or` / `not` and parentheses. The same
parsed tree feeds `eval_expr`, conservative `infer_expr_type`, and the reference walkers.
Conditions route through the ONE unified engine; a bare reference with no comparison
operator is a truthiness test (not an error).
"""

import re
from enum import Enum
from typing import Any, Callable

from lark import Token, Tree

from agent_composer.typesys.pool import VariablePool
from agent_composer.typesys.values import Type, ValueKind


class ExpressionError(ValueError):
    """A `when:` expression could not be parsed or evaluated."""


class RequiredError(ExpressionError):
    """A `${ref:?message}` whose ref was unbound (the binder maps it to BindingError)."""


_VAR_RE = re.compile(r"\$\{([^}]+)\}")
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

    A missing input / dotted miss / None step -> None: the LOCKED `when:` missing->falsy
    contract (the compile-time strict check rejects undeclared names, so a None here is a
    legitimate empty value, not an authoring error). Bare local-input refs only.
    """
    parts = [p.strip() for p in path.strip().split(".")]
    if not parts or any(not p for p in parts):
        raise ExpressionError(f"invalid variable path: '${{{path}}}'")
    value: Any = record.get(parts[0])
    for step in parts[1:]:
        value = value.get(step) if isinstance(value, dict) else None
    return value


def render_template(text: str, pool: VariablePool) -> str:
    """Substitute every `${ref}` in `text` with its resolved value (str), against
    the whole pool (`node`/`system` namespaces).

    No production caller as of slice 5 — strict AGENT renders prompts against the
    bound input record via `render_template_record`. Retained (and test-pinned)
    for the open-namespace render path a future templated `HUMAN_INPUT` prompt
    will need. An UNRESOLVED reference (resolves to None) RAISES `ExpressionError`
    rather than silently rendering "" (the runtime floor for refs the compile-time
    check can't prove total).
    """

    def _sub(match: "re.Match[str]") -> str:
        ref = match.group(1)
        value = resolve_reference(ref, pool)
        if value is None:
            raise ExpressionError(f"unresolved reference ${{{ref}}} in template")
        return str(value)

    return _VAR_RE.sub(_sub, text)


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
        return float(t) if "." in t else int(t)
    if (t.startswith('"') and t.endswith('"')) or (t.startswith("'") and t.endswith("'")):
        return t[1:-1]
    raise ExpressionError(f"cannot parse literal {token!r}")


def _arith(op: str, lhs: Any, rhs: Any) -> Any:
    """A binary arithmetic op over NUMBERS ONLY. A non-numeric operand —
    string/bool/null/None — is a loud `ExpressionError` (bool is excluded: it is
    `type bool`, not `int`)."""
    if type(lhs) not in (int, float) or type(rhs) not in (int, float):
        raise ExpressionError(f"arithmetic `{op}` requires numbers, got {lhs!r} {op} {rhs!r}")
    try:
        if op == "+":
            return lhs + rhs
        if op == "-":
            return lhs - rhs
        if op == "*":
            return lhs * rhs
        if op == "/":
            return lhs / rhs
        if op == "%":
            return lhs % rhs
    except ZeroDivisionError as exc:
        raise ExpressionError(f"division by zero: {lhs!r} {op} {rhs!r}") from exc
    raise ExpressionError(f"unsupported arithmetic operator: {op!r}")


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
        raise ExpressionError(f"type error evaluating {lhs!r} {op} {rhs!r}: {exc}") from exc
    raise ExpressionError(f"unsupported operator: {op!r}")


_WHOLE_SPAN_RE = re.compile(r"^\$\{(.*)\}$", re.DOTALL)


def _parse_condition(expression: str) -> "Tree | Token":
    """Parse an expression-field value (`when:`/`on:`/an `asserts:` entry) via the ONE
    unified grammar, honouring the both-spellings entry rule.

    An expression field may be written three ways, all of which must PARSE (and later
    evaluate) IDENTICALLY:
      - a bare expression `a > 5` (no braces) — canonical,
      - a legacy `${a} > 5` (braces around a ref, operators outside),
      - a whole `${a > 5}` (braces around the whole expression).
    If the WHOLE trimmed value is a single `${...}` span, strip the outer braces and
    parse the interior; otherwise parse the whole value directly. The grammar's ref atom
    already admits a `${...}`-wrapped ref, so the mixed `${a} > 5` form parses as-is.
    """
    from agent_composer.expr.grammar import parse_expr

    text = expression.strip()
    m = _WHOLE_SPAN_RE.match(text)
    # A whole-span `${...}` with no unescaped `}` inside strips to its interior; a value
    # with an interior `}` (e.g. `${a} > 5`) is NOT a single span and parses whole.
    if m is not None and "}" not in m.group(1):
        return parse_expr(m.group(1))
    return parse_expr(text)


def condition_refs(expression: str) -> list[str]:
    """The reference PATHS a condition-field value reads (`when:`/`until:`/`while:`/`asserts:`).

    The condition analog of the template ref-walk ([`expr_refs_of`][agent_composer.expr.template.expr_refs_of])
    and the prompt ref-walk ([`prompt_refs`][agent_composer.expr.template.prompt_refs]): parse the
    WHOLE value as ONE expression (via `_parse_condition`, honouring all three spellings — bare
    `a > 5`, mixed `${a} > 5`, whole `${a > 5}`) and collect its reference leaves via the shared
    [`expr_refs`][agent_composer.expr.expressions.expr_refs]. Every compile-time consumer of a
    condition's refs (flow/node `asserts:`, a loop `while:`/`until:` predicate, a searched `case`
    `when:`) routes here, so the three spellings extract IDENTICALLY — unlike a flat `${...}` regex,
    which reads a whole-span as one bogus path and a bare form as no refs at all.

    Args:
        expression (`str`): the condition source — any expression the grammar accepts, any spelling.

    Returns:
        `list[str]`: the reference paths (dotted, `#`/`/` preserved) in source order, NOT deduped.

    Raises:
        `ExpressionError`: on a malformed condition (the caller frames it as a located load error).
    """
    return expr_refs(_parse_condition(expression))



def _evaluate(expression: str, resolve: Callable[[str], Any]) -> bool:
    """Parse + evaluate a `when:`/`asserts:` expression on the UNIFIED engine, resolving
    each reference via `resolve`; the result is coerced to `bool`.

    Runs the unified `eval_expr` in `CONDITION_FALSY` mode — a missing reference is `None`
    (falsy through comparisons, the LOCKED `when:` missing->falsy contract that `case default`
    routing depends on). The truthiness fold of `and`/`or` (unified) is coerced back to a
    plain `bool` here, so a `when:`/`asserts:` predicate always yields a boolean.
    """
    tree = _parse_condition(expression)
    return bool(eval_expr(tree, resolve, mode=ResolveMode.CONDITION_FALSY))


def evaluate_when(expression: str, pool: VariablePool) -> bool:
    """
    Evaluate a `when:` boolean expression against the variable pool.

    Parses and folds the expression on the ONE unified engine — comparisons
    (`==` `!=` `<` `<=` `>` `>=` `in` `not in`) and arithmetic over references and literals,
    combined with `and` / `or` / `not` — and coerces the result to a single boolean. This is
    the pool-based path, kept for manifest parse-checks and the deferred LOOP `while:` predicate
    seam; strict `CASE` uses the record-based path.

    The value may be written three ways, all evaluating IDENTICALLY: a bare expression
    `a > 5`, a legacy `${a} > 5` (braces on the ref), or a whole `${a > 5}` (braces around
    the whole thing). A bare reference with no comparison operator is a TRUTHINESS test
    (coerced to bool), not an error.

    Args:
        expression (`str`):
            The `when:` source — any expression the unified grammar accepts, in any of the
            three brace spellings above.
        pool (`VariablePool`):
            The variable pool each reference resolves against (`node`/`system` namespaces).
            A reference that resolves to `None` participates as a falsy value.

    Returns:
        `bool`:
            The truth value of the expression.

    Raises:
        `ExpressionError`:
            If the expression is malformed, references an invalid path, or mixes
            incompatible types (e.g. arithmetic over a non-number).

    Example:
        ```python
        evaluate_when("${score.output} >= 0.5 and ${flag.output}", pool)
        ```
    """
    return _evaluate(expression, lambda path: resolve_reference(path, pool))


def first_failing_assert(exprs, pool: VariablePool):
    """The first assert expression in `exprs` that is false against `pool`, else None.

    The shared enforcement primitive for a flow's `asserts:` (boundary or post): `run_flow`
    uses it at the top-level boundary, and the REF/MAP child seam uses it to enforce a def's
    child-boundary asserts. Each expr is the `when:`/`asserts:` boolean grammar."""
    for expr in exprs:
        if not evaluate_when(expr, pool):
            return expr
    return None


def evaluate_when_record(expression: str, record: dict) -> bool:
    """Evaluate a strict-CASE `when:` against the node's bound input record.

    `${name}` / `${name.path}` resolve against `record`; a miss -> None, which is falsy
    through ==/!=/ordered comparisons (the locked `when:` missing->falsy contract).
    Note: `in`/`not in` with a None operand still raises (unchanged from the pool path).
    Bare local-input refs only; pool namespaces are not in scope (they belong in the
    node's input `from:` bindings).

    cf. `render_template_record` (strict AGENT), which RAISES on the same dotted miss.
    `when:` deliberately stays falsy so `default` can fire — so a dotted-key typo on a
    declared dict input routes to `default` silently until the `types:` registry can
    validate dotted paths at compile time (the head is checked today, not the path).
    """
    return _evaluate(expression, lambda path: _resolve_in_record(path, record))


# --------------------------------------------------------------------------- #
# The ONE unified evaluator — walks a tree from `expr.grammar.parse_expr` against
# a `resolve` callable. It is the single `${...}` evaluator: the binding path
# (coalesce/default/required), the `when:`/`asserts:` boolean path, and the prompt
# builtin-call path all parse with `parse_expr` and evaluate here.
# --------------------------------------------------------------------------- #


class ResolveMode(Enum):
    """How a MISSING reference (one `resolve` maps to `None`) is treated.

    - `STRICT_RAISE`: a missing ref is an error — raise `ExpressionError`. For a
      context where every reference must be present (a strict render).
    - `BINDING_NONE`: a missing ref is `None` — the binding contract (a coalesce /
      default may then fire, or the whole value is `None`).
    - `CONDITION_FALSY`: a missing ref is `None`, which is falsy through comparisons
      (the locked `when:` missing->falsy contract). Same None as `BINDING_NONE`; the
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
    """`+` on resolved VALUES (NOT the number-only `_arith` gate): so `str+str`
    concatenates and `list+list` extends. Arithmetic over `None` (a missing ref in a
    non-strict mode) propagates Python's `TypeError`, wrapped by the caller."""
    return lhs + rhs


class _EvalExpr:
    """Recursive walker for a `parse_expr` tree, parameterised by `resolve`, `item`,
    and `mode`. Hand-written (not a Lark `Transformer`) so coalesce / default /
    required can SHORT-CIRCUIT — evaluate the fallback only when the head misses.

    A `refcall` reference evaluates to `_MISSING` on an unresolved path; every value
    consumer (operators, comparisons, the top-level entry) maps `_MISSING` to `None`
    for non-strict modes or raises for `STRICT_RAISE`."""

    def __init__(self, resolve: Callable[[str], Any], item: Any, mode: ResolveMode):
        self._resolve = resolve
        self._item = item
        self._mode = mode

    # --- top-level entry: coerce a `_MISSING` head to a value per mode --- #
    def eval(self, node: "Tree | Token") -> Any:
        return self._as_value(self._walk(node))

    def _as_value(self, v: Any) -> Any:
        """Turn an internal `_MISSING` into a mode-appropriate value: `None` for the
        non-strict modes, a raised `ExpressionError` for `STRICT_RAISE`."""
        if v is _MISSING:
            if self._mode is ResolveMode.STRICT_RAISE:
                raise ExpressionError("unresolved reference in expression")
            return None
        return v

    # --- dispatch --- #
    def _walk(self, node: "Tree | Token") -> Any:
        if isinstance(node, Token):
            return self._token(node)
        handler = getattr(self, f"_do_{node.data}", None)
        if handler is None:
            raise ExpressionError(f"unsupported expression node {node.data!r}")
        return handler(node)

    def _token(self, tok: Token) -> Any:
        """A lone atom that inlined up to a top-level token: `${...}`, or a literal."""
        if tok.type == "WRAPPED_REF":
            return self._reference(str(tok)[2:-1])
        if tok.type == "NUMBER":
            s = str(tok)
            return float(s) if "." in s else int(s)
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
        value = self._call_builtin(name, call_suffix)
        after = [c for c in children if isinstance(c, Tree) and c.data == "trailer"]
        for tr in after:
            value = self._dotted_step(value, str(tr.children[0]))
        return value

    def _reference(self, path: str) -> Any:
        """Resolve a dotted reference path. An `item`-headed path walks the MAP-body
        `item` scope locally (dict-only — see `_dotted_step`); every other path is
        handed WHOLE to `resolve` (the pool-agnostic seam — the
        resolver owns the dotted walk, exactly as `resolve_reference` / `_resolve_in_record`
        do today). A miss (`item` miss, or `resolve` -> None) yields `_MISSING`."""
        parts = path.split(".")
        if parts[0] == "item":  # MAP-body-local scope (not a resolver head)
            if self._item is None:
                return _MISSING
            value: Any = self._item
            for step in parts[1:]:
                value = self._dotted_step(value, step)
            return value  # `_dotted_step` already yields `_MISSING` on a walk miss
        resolved = self._resolve(path)
        return _MISSING if resolved is None else resolved

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

    def _call_builtin(self, callee: str, call_suffix: Tree) -> Any:
        """Dispatch a builtin call through `TEMPLATE_FNS`. Positional and keyword
        `arg`s are distinguished by shape (a kwarg carries a leading NAME token).
        A builtin failure is wrapped as `ExpressionError`."""
        from agent_composer.expr.builtins import TEMPLATE_FNS

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
    def _do_coalesce(self, node: Tree) -> Any:
        """`a | b | c` — first operand that resolves to a non-None value wins; a
        present falsy value (0 / "" / False) wins, only `None`/miss falls through."""
        for child in node.children:
            v = self._as_value(self._walk(child))
            if v is not None:
                return v
        return None

    def _do_default_expr(self, node: Tree) -> Any:
        """`head :- fallback` — the fallback is evaluated ONLY when `head` misses or
        resolves to None."""
        head, fallback = node.children
        v = self._as_value(self._walk(head))
        if v is not None:
            return v
        return self._as_value(self._walk(fallback))

    def _do_required_expr(self, node: Tree) -> Any:
        """`head :? message` — raise `RequiredError` (with `message`) when `head`
        misses / is None."""
        head, message = node.children
        v = self._as_value(self._walk(head))
        if v is None:
            raise RequiredError(self._as_value(self._walk(message)))
        return v

    # --- boolean combinators --- #
    # The operator KEYWORD tokens interleaved among the operands. A value operand may ALSO
    # be a top-level `Token` (a bare `${ref}` WRAPPED_REF, or a bare NUMBER/STRING/BOOL/NULL),
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
        # Operands fold by PYTHON truthiness, not bool-only (so `0 or "hit"` is truthy):
        # a deliberate change from the legacy `when:` bool-filter, matching the design
        # "operators delegate to Python semantics on resolved values".
        return any(self._as_value(self._walk(c)) for c in self._operands(node))

    def _do_and_expr(self, node: Tree) -> Any:
        # Operands fold by Python truthiness, not bool-only (see `_do_or_expr`): a
        # deliberate change from the legacy `when:` bool-filter.
        return all(self._as_value(self._walk(c)) for c in self._operands(node))

    # --- comparisons (reuse the locked `_eval_comparison` None-> False contract) --- #
    def _do_compare(self, node: Tree) -> Any:
        lhs, op, rhs = node.children
        return _eval_comparison(str(op), self._as_value(self._walk(lhs)), self._as_value(self._walk(rhs)))

    def _do_compare_in(self, node: Tree) -> Any:
        lhs, _op, rhs = node.children
        return _eval_comparison("in", self._as_value(self._walk(lhs)), self._as_value(self._walk(rhs)))

    def _do_compare_notin(self, node: Tree) -> Any:
        lhs, _op, rhs = node.children
        return _eval_comparison("not in", self._as_value(self._walk(lhs)), self._as_value(self._walk(rhs)))

    # --- arithmetic — Python operators on VALUES (NOT the number-only `_arith`) --- #
    def _binop(self, node: Tree, op: Callable[[Any, Any], Any], symbol: str) -> Any:
        lhs = self._as_value(self._walk(node.children[0]))
        rhs = self._as_value(self._walk(node.children[1]))
        try:
            return op(lhs, rhs)
        except ZeroDivisionError as exc:
            raise ExpressionError(f"division by zero: {lhs!r} {symbol} {rhs!r}") from exc
        except TypeError as exc:
            raise ExpressionError(f"type error evaluating {lhs!r} {symbol} {rhs!r}: {exc}") from exc

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
            raise ExpressionError(f"unary minus on non-number {v!r}: {exc}") from exc

    def _do_list_lit(self, node: Tree) -> Any:
        return [self._as_value(self._walk(c)) for c in node.children]


def eval_expr(
    tree: "Tree | Token",
    resolve: Callable[[str], Any],
    item: Any = None,
    mode: ResolveMode = ResolveMode.BINDING_NONE,
) -> Any:
    """
    Evaluate a unified `${...}` expression tree (from `expr.grammar.parse_expr`).

    The ONE evaluator for every `${...}` construct: references (bare / dotted /
    `${}`-wrapped / `item`-scoped), arithmetic and comparisons over VALUES (so
    `str+str` concatenates and `list+list` extends), boolean combinators, list
    literals, coalesce / default / required, and pure builtin calls.

    Three semantics are LOCKED (they are load-bearing for `case default` routing):
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
        item (`Any`, *optional*, defaults to `None`):
            The MAP-body-local scope for `item` / `item.path` (dict-only walk). `None`
            means no item scope.
        mode (`ResolveMode`, *optional*, defaults to `BINDING_NONE`):
            How a missing reference is treated — raise (`STRICT_RAISE`) or become
            `None` (`BINDING_NONE` / `CONDITION_FALSY`).

    Returns:
        `Any`:
            The evaluated value (typed): a number / string / bool / list / dict, or
            `None`.

    Raises:
        `ExpressionError`:
            On an unsupported node, an unknown builtin, a builtin failure, arithmetic
            over an incompatible operand (incl. `None`), or a strict-mode miss.
        `RequiredError`:
            When a `head :? message` required atom's head misses / is None.
    """
    return _EvalExpr(resolve, item, mode).eval(tree)


# --------------------------------------------------------------------------- #
# Conservative compile-time type inference over the unified expression tree.
# --------------------------------------------------------------------------- #


def infer_expr_type(
    tree: "Tree | Token",
    ref_type_fn: Callable[[str], "Type | None"],
) -> "Type | None":
    """Return the statically proven type of one parsed expression, if any.

    Inference is deliberately conservative: literal tokens, exact references,
    comparisons, and boolean combinators have fixed result types. Other forms
    remain unknown until their runtime value semantics can be modeled without
    guessing. Reference lookup is injected so this source-neutral expression
    layer does not know about a flow graph or Compose document.

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
            return None
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


def expr_refs(tree: "Tree | Token") -> list[str]:
    """
    Collect every reference-leaf PATH a unified `${...}` expression tree reads.

    The single compile-time ref-walk over a tree from
    [`parse_expr`][agent_composer.expr.grammar.parse_expr]. A `refcall` WITHOUT a
    `call_suffix` is a reference — its dotted path (leading NAME joined with each
    `trailer` segment) is collected. A `refcall` WITH a `call_suffix` is a builtin
    CALL: the callee name contributes NO ref, but each `arg` VALUE's refs ARE
    collected (recursively). Literals contribute nothing. Coalesce / default /
    required refs are all collected, including the nested-default ref.

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
        # A lone atom that inlined to a top-level token: only a `${...}` wrapped-ref
        # names a reference; a bare NUMBER / STRING / BOOL / NULL contributes nothing.
        if node.type == "WRAPPED_REF":
            refs.append(str(node)[2:-1])
        return
    if node.data == "refcall":
        # KEEP IN SYNC with `_do_refcall`'s refcall decomposition — both duplicate this
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
        for arg in call_suffix.children:  # each child is an `arg` subtree; empty `fn()` -> no children -> skipped
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
# params (searched `case`) used to `.sub` a flat `${...}` regex — which mishandles a
# whole-span `${a > 5}` (rewrites the whole interior as one path) and a bare `a > 5` (no
# `${}`, no match). Routing them through the ONE parse tree here fixes both: only the
# reference LEAVES move, every operator / literal / builtin-callee / whitespace stays
# verbatim (spliced by source position), and all three spellings rewrite identically.
# --------------------------------------------------------------------------- #


def rewrite_expr_refs(text: str, rename: Callable[[str], "str | None"]) -> str:
    """Rewrite the reference-leaf PATHS of ONE unified `${...}` expression `text`.

    The rewrite analog of [`expr_refs`][agent_composer.expr.expressions.expr_refs]: it walks
    the SAME parse tree (`refcall` reference / `WRAPPED_REF` / builtin-call args), but instead
    of collecting each reference path it asks `rename(path)` for a replacement and splices it
    in by source position. `rename` returns the new dotted path, or `None` to leave that
    reference untouched. Everything that is NOT a reference leaf — operators, numbers, string
    and list literals, a builtin CALL's callee name, parentheses, whitespace — is preserved
    verbatim. A bare `refcall` (`a.b`) is replaced with the bare new path; a `${a}`
    `WRAPPED_REF` leaf is replaced with the braced `${new}` (its wrapping preserved).

    Args:
        text (`str`): the expression source (a bare expression, or a span interior).
        rename (`Callable[[str], str | None]`): `path -> new_path`, or `None` to keep.

    Returns:
        `str`: `text` with the renamed reference leaves spliced in.

    Raises:
        `ExpressionError`: if `text` does not parse (same front end as `parse_expr`).
    """
    from agent_composer.expr.grammar import parse_expr

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
    reference leaf. KEEP IN SYNC with `_collect_expr_refs` — same refcall/WRAPPED_REF/
    builtin-arg decomposition, but capturing token source positions to edit in place."""
    if isinstance(node, Token):
        if node.type == "WRAPPED_REF":
            new = rename(str(node)[2:-1])
            if new is not None:
                edits.append((node.start_pos, node.end_pos, "${" + new + "}"))
        return
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
        for arg in call_suffix.children:
            _collect_ref_edits(arg.children[-1], rename, edits)
        return
    for child in node.children:
        _collect_ref_edits(child, rename, edits)


def rewrite_condition_refs(expression: str, rename: Callable[[str], "str | None"]) -> str:
    """Rewrite the reference leaves of a condition-field value (`when:`/`until:`/`while:`/
    `asserts:`/searched-`case` `when:`), honouring all three spellings.

    The rewrite analog of [`condition_refs`][agent_composer.expr.expressions.condition_refs]:
    a whole-span `${a > 5}` keeps its outer braces and rewrites the interior; a mixed
    `${a} > 5` / bare `a > 5` are rewritten in place. `rename(path)` returns the replacement
    path (or `None` to keep). The value is trimmed first (matching `_parse_condition`)."""
    text = expression.strip()
    m = _WHOLE_SPAN_RE.match(text)
    if m is not None and "}" not in m.group(1):
        return "${" + rewrite_expr_refs(m.group(1), rename) + "}"
    return rewrite_expr_refs(text, rename)
