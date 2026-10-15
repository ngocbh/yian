r"""One unified `${...}` expression grammar — PARSE ONLY (no evaluation here).

This single grammar parses binding coalesce/default expressions, `when:`
boolean/arithmetic expressions, and prompt builtin calls into one Lark tree.
Evaluation deliberately lives in `expr.expressions`.

## What it parses

- References: bare `a`, dotted `a.b.c`, and — for the runtime graph-expansion
  separators — segments containing `#` / `/` (`node#0.output`, `def/child.output`).
- Arithmetic (`+ - * / %` and `**`), unary minus, comparisons
  (`== != < <= > >=`), `and` / `or` / `not`, `in` / `not in`. A chain of
  same-precedence arithmetic operators is one flat node, like `and` / `or`:
  `a + b - c` is one `sum`, `a * b / c` one `product`, `a ** b ** c` one `power`,
  each holding its operator tokens between the operands.
- List literals (`[1, "x"]`) and parenthesised sub-expressions.
- Pure builtin calls (`upper(name)`, `join(items, ", ")`), with dotted access on
  the result (`fn(x).field`) — mirroring the prompt call form.
- Coalesce (`a | b | c`), default (`a :- d`), required (`a :? "msg"`).
- The `isinstance(x, T)` special form, whose second argument is a bare type name.
- Constructor calls `T(x)` / `T(a=..., b=...)`, parsed like any other call; whether a
  callee is a constructor (and so takes keyword fields or one value) is known only once
  compose resolves it against the flow's `typedefs:`.

`${...}` marks an expression inside a YAML string, so it never appears inside one:
values are named directly and grouped with parentheses. `parse_expr` rejects a
`${...}` outside a string literal with an error that shows the rewrite
(`a :- ${b}` -> `a :- b`).

## Shared terminal design for references and calls

A naive grammar with a SEPARATE bare-ref terminal AND a separate
`call: NAME "(" ... ")"` rule does NOT parse under LALR: the lexer matches the
bare-ref terminal on `upper` before it can ever see the `(`, so the call rule is
unreachable and `upper(name)` fails. One shared `NAME` terminal with an optional
call-suffix avoids that ambiguity: a `refcall` is a `NAME`, then dotted
`trailer`s, then an OPTIONAL `call_suffix`. No suffix => a reference (path = the
NAME joined with its trailer segments); a suffix present => a builtin call (callee
= the leading NAME, any trailers AFTER the parens are dotted access on the
result). A `call_suffix` WITH a leading trailer (`a.b(...)`) is rejected in
`parse_expr` — builtins are bare-callee only, matching the prompt grammar.

## Charset decision — why `-` is NOT in `NAME` (and `#` / `/` are)

`NAME` is `/[A-Za-z_][A-Za-z0-9_#\/]*/`. `-` is deliberately EXCLUDED: were it in
the terminal, `a - b` would lex as the single identifier `a-b` and subtraction
would break. Consequence: references inside `${}` use this non-hyphen charset, so
a hyphenated identifier is unavailable INSIDE an expression. That only matters for
the whole-value `call(...)` directive, whose flow-id may be hyphenated — but that is a
loader-side string match (a directive recognizer), NOT this grammar, so it is
unaffected. `#` and `/` ARE allowed because runtime graph expansion mints segments
like `node#0` and `def/child`. Node ids are checked to be identifiers where the node
is defined, so a hyphenated node id fails there rather than at a reference.

## Keyword priority

The keyword terminals (`and` `or` `not` `in` `not in` `true` `false` `null`
`none`) carry a lexer priority higher than `NAME`, so they win when an operator is
valid. Lark's contextual lexer may still fall back to `NAME` where only an operand
is valid; `parse_expr` therefore rejects these four operator words as exact
reference roots after parsing. Dotted field segments retain the ordinary `NAME`
token and remain legal.
"""

import re

from lark import Lark, Token, Tree
from lark.exceptions import LarkError

# Reuse the existing expression error — `expressions.ExpressionError` is already
# the base of `expressions.RequiredError`, so parse failures here stay in one family.
from yian.expr.expressions import ISINSTANCE, ExpressionError
from yian.defaults import current
from yian.typesys.values import IntTooLongError, parse_int_text

# The unified grammar. Precedence, lowest to highest:
#   coalesce (`|`) -> default/required (`:-`/`:?`) -> or -> and -> not
#   -> comparison (incl. in / not in) -> sum (`+ -`) -> product (`* / %`)
#   -> unary minus -> power (`**`, folded right to left) -> atom.
# `?rule` inlines a single-child parse to keep the tree flat (Lark convention).
_GRAMMAR = r"""
?start: coalesce

// coalesce (`|`) — lowest precedence. `a :- b | c` => `(a:-b) | c`. The two
// alternatives keep the `?` inline effective: a lone operand stays its own tree
// (no wrapper), and a `coalesce` node appears ONLY when a `|` is actually present.
?coalesce: default_expr
         | default_expr ("|" default_expr)+   -> coalesce

// default / required bind tighter than `|`, looser than boolean/arithmetic.
?default_expr: or_expr
             | or_expr ":-" or_expr   -> default_expr
             | or_expr ":?" or_expr   -> required_expr

?or_expr: and_expr (OR and_expr)*
?and_expr: not_expr (AND not_expr)*
?not_expr: NOT not_expr   -> negate
         | comparison
?comparison: sum
           | sum COMP_OP sum   -> compare
           | sum IN sum        -> compare_in
           | sum NOT_IN sum    -> compare_notin

// A chain of same-precedence operators is ONE flat node, like `and` / `or`:
// `a + b - c` is `sum[a, +, b, -, c]`, with the operator tokens kept between the operands.
// The evaluator folds `sum` and `product` left to right and `power` right to left, so a
// long chain needs no recursion per operator.
?sum: product (ADD_OP product)*
?product: unary (MUL_OP unary)*
// unary minus binds looser than `**` (Python: `-x ** 2` == `-(x ** 2)`).
?unary: power
      | "-" unary   -> neg
// `**` binds tighter than unary minus, yet its right operand may be negated
// (`2 ** -a ** b` == `2 ** (-(a ** b))`). Only the last operand can be, so the rule
// stays free of LALR conflicts.
?power: atom (POW_OP atom)* (POW_OP neg_tail)?
?neg_tail: "-" unary   -> neg

?atom: refcall
     | NUMBER | STRING | BOOL | NULL
     | list_lit
     | "(" coalesce ")"

// One shared NAME terminal disambiguates references from builtin calls:
//   no call_suffix  => a reference (NAME + trailer segments)
//   a call_suffix   => a builtin call (leading NAME is the callee; trailers
//                      AFTER the parens are dotted access on the result). A
//                      trailer BEFORE the parens (`a.b(...)`) is a dotted callee
//                      and is rejected in `parse_expr`.
refcall: NAME trailer* (call_suffix trailer*)?
trailer: "." NAME
call_suffix: "(" [arg ("," arg)*] ")"
arg: [NAME "="] coalesce

// list literal: elements are full expressions (refs / literals / arithmetic).
list_lit: "[" [coalesce ("," coalesce)*] "]"

ADD_OP: "+" | "-"
MUL_OP: "*" | "/" | "%"
POW_OP: "**"
COMP_OP: "==" | "!=" | "<=" | ">=" | "<" | ">"
STRING: /"[^"]*"/ | /'[^']*'/
NUMBER: /\d+(\.\d+)?/

// `#` and `/` ARE in NAME (graph-expansion segments); `-` is NOT (else `a - b`
// lexes as one ident and subtraction breaks — see the module docstring).
NAME: /[A-Za-z_][A-Za-z0-9_#\/]*/

// Keyword terminals outrank NAME where both are valid. The contextual lexer may
// still emit NAME in an operand-only state; parse_expr rejects reserved ref roots.
NOT_IN.5: /not\s+in\b/
IN.4: /in\b/
AND.4: /and\b/
OR.4: /or\b/
NOT.4: /not\b(?![#\/])/
BOOL.3: /true\b/ | /false\b/
NULL.3: /null\b/ | /none\b/

%import common.WS
%ignore WS
"""

# The shared LALR parser instance (built once at import). `maybe_placeholders`
# False so optional branches drop rather than inserting `None` placeholders into
# the tree.
_PARSER = Lark(_GRAMMAR, parser="lalr", maybe_placeholders=False)

# Parse nodes that only group the parts of a reference or call; they add no level.
_WRAPPER_NODES = frozenset({"trailer", "call_suffix", "arg"})

_RESERVED_REFERENCE_ROOTS = frozenset({"and", "or", "not", "in"})

# The keyword terminals in `_GRAMMAR` above; none of them can be a reference root.
EXPRESSION_KEYWORDS = _RESERVED_REFERENCE_ROOTS | {"true", "false", "null", "none"}


def find_span_end(s: str, start: int) -> int | None:
    """Index of the `}` closing a `${...}` span whose interior begins at `start`
    (brace-depth and quote aware), or None if unbalanced."""
    depth, quote, i, n = 1, None, start, len(s)
    while i < n:
        c = s[i]
        if quote is not None:
            if c == quote:
                quote = None
        elif c in ("'", '"'):
            quote = c
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return None


def whole_span_interior(text: str) -> str | None:
    """The interior of `text` when the trimmed text is exactly one `${...}` span, else None.

    Quote aware, so `${x == "}"}` is one span and `${a} > ${b}` is not.
    """
    text = text.strip()
    if not text.startswith("${"):
        return None
    end = find_span_end(text, 2)
    return text[2:end] if end == len(text) - 1 else None


# A character that would fuse with an unwrapped name: `not${x}` must become `not x`.
_NAME_CHAR = re.compile(r"[A-Za-z0-9_#/]")


def _parses(text: str) -> bool:
    """True when `text` parses under the grammar within the nesting limit (a rewrite worth
    suggesting)."""
    try:
        _reject_deep_expression(_PARSER.parse(text))
    except (LarkError, ExpressionError):
        return False
    return True


def _is_lone_atom(text: str) -> bool:
    """True when `text` parses as one reference or one literal (no operator or call)."""
    try:
        tree = _PARSER.parse(text)
    except LarkError:
        return False
    if isinstance(tree, Token):
        return True
    return tree.data == "refcall" and not any(
        isinstance(child, Tree) and child.data == "call_suffix" for child in tree.children
    )


def _unwrap_spans(text: str, depth: int = 0, limit: int | None = None) -> str | None:
    """`text` with each `${X}` outside a string literal replaced by `X`, or `(X)` when `X`
    is more than one reference or literal; None when a `${` is unbalanced or spans nest
    deeper than `limit` (`depth` counts the spans around `text`; `limit` defaults to the
    value nesting limit)."""
    if limit is None:
        limit = current().max_value_nesting
    if depth > limit:
        return None
    out: list[str] = []
    quote, i, n = None, 0, len(text)
    while i < n:
        c = text[i]
        if quote is not None:
            out.append(c)
            if c == quote:
                quote = None
            i += 1
        elif c in ("'", '"'):
            quote = c
            out.append(c)
            i += 1
        elif c == "$" and text[i + 1 : i + 2] == "{":
            end = find_span_end(text, i + 2)
            inner = None if end is None else _unwrap_spans(text[i + 2 : end], depth + 1, limit)
            if inner is None:
                return None
            inner = inner.strip()
            piece = inner if _is_lone_atom(inner) else f"({inner})"
            if out and _NAME_CHAR.match(out[-1]):
                piece = " " + piece
            if _NAME_CHAR.match(text[end + 1 : end + 2]):
                piece += " "
            out.append(piece)
            i = end + 1
        else:
            out.append(c)
            i += 1
    return "".join(out)


def _has_span(text: str) -> bool:
    """True when `text` has a `${` outside a string literal."""
    quote = None
    for i, c in enumerate(text):
        if quote is not None:
            if c == quote:
                quote = None
        elif c in ("'", '"'):
            quote = c
        elif c == "$" and text[i + 1 : i + 2] == "{":
            return True
    return False


def reject_nested_spans(text: str, *, in_span: bool = False) -> None:
    """Raise if the expression `text` holds a `${...}`, showing the rewrite.

    `${...}` marks an expression inside a YAML string, so inside an expression values are
    named directly and grouped with parentheses. A `${...}` inside a quoted string literal
    is text and allowed.

    Args:
        text (`str`):
            One expression: a span interior or a bare condition.
        in_span (`bool`, *optional*, defaults to `False`):
            `text` is the interior of a template span, so the rewrite is shown as the whole
            fixed span (`${a :- b}`) rather than the bare expression (`a :- b`).

    Raises:
        `ExpressionError`:
            If `text` holds a `${...}` outside a string literal. The message shows the
            rewrite when the unwrapped text parses.
    """
    if not _has_span(text):
        return
    message = "`${...}` can't appear inside an expression"
    rewrite = _unwrap_spans(text)
    if rewrite is not None and _parses(rewrite):
        rewrite = rewrite.strip()
        message += f"; write `{'${' + rewrite + '}' if in_span else rewrite}`"
    raise ExpressionError(message)


def _reject_long_int_literals(tree: Tree | Token) -> None:
    """Raise if a parsed expression holds an int literal with more than `YIAN_MAX_INT_DIGITS` digits.

    Leading zeros don't count. The walk is iterative, like `_reject_deep_expression`.

    Args:
        tree (`Tree | Token`):
            A parse from `_PARSER`.

    Raises:
        `ExpressionError`:
            If a `NUMBER` token without a `.` has more than `YIAN_MAX_INT_DIGITS` significant
            digits, raised from the `IntTooLongError`; the message gives the digit
            count, never the digits.
    """
    stack: list[Tree | Token] = [tree]
    while stack:
        node = stack.pop()
        if isinstance(node, Tree):
            stack.extend(node.children)
        # A token no longer than the limit can't be over it, so skip the parse.
        elif (
            isinstance(node, Token)
            and node.type == "NUMBER"
            and len(node) > current().max_int_digits
            and "." not in node
        ):
            try:
                parse_int_text(str(node))
            except IntTooLongError as exc:
                raise ExpressionError(str(exc)) from exc


def _reject_deep_expression(tree: Tree | Token) -> None:
    """Raise if a parsed expression nests deeper than the value nesting limit.

    An expression nests no deeper than a value, so one limit,
    `current().max_value_nesting`, bounds both.

    Each operator chain (`sum`, `product`, `power`, `and_expr`, `or_expr`), unary operator,
    comparison, escape (`|`, `:-`, `:?`), list literal, and call is one level; a plain
    reference and the parts of a reference or call add none, and parentheses make no node.
    The walk is iterative, so it is safe on a tree deep enough to overflow a recursive one.

    Args:
        tree (`Tree | Token`):
            A parse from `_PARSER`.

    Raises:
        `ExpressionError`:
            If any path from the root passes more than the limit.
    """
    limit = current().max_value_nesting
    stack: list[tuple[Tree | Token, int]] = [(tree, 0)]
    while stack:
        node, depth = stack.pop()
        if not isinstance(node, Tree):
            continue
        if node.data not in _WRAPPER_NODES and (
            node.data != "refcall"
            or any(
                isinstance(child, Tree) and child.data == "call_suffix"
                for child in node.children
            )
        ):
            depth += 1
            if depth > limit:
                raise ExpressionError(
                    f"expression nests deeper than {limit} levels, the "
                    "limit for a yian expression (each call, list, unary operator, "
                    "comparison, and chain of operators is a level); split it into smaller "
                    "bindings or steps"
                )
        stack.extend((child, depth) for child in node.children)


def _reject_reserved_reference_roots(tree: Tree | Token) -> None:
    """Reject operator words parsed contextually as reference roots.

    A `refcall` with a `call_suffix` is a builtin call rather than a reference, so
    its callee is outside this rule. Trailer segments and keyword-argument labels
    are likewise not roots.
    """
    if not isinstance(tree, Tree):
        return
    for node in tree.iter_subtrees():
        if node.data != "refcall":
            continue
        if any(
            isinstance(child, Tree) and child.data == "call_suffix"
            for child in node.children
        ):
            continue
        head = node.children[0]
        if isinstance(head, Token) and head.type == "NAME" and head in _RESERVED_REFERENCE_ROOTS:
            raise ExpressionError(
                f"reserved expression keyword {str(head)!r} cannot be used as a reference root"
            )


def _reject_dotted_callee(tree: Tree | Token) -> None:
    """Raise if any `refcall` is a call (`call_suffix`) with a leading `trailer`.

    Builtins are bare-callee only: `a.b(x)` — a dotted head applied like a call —
    is not a valid builtin call and must be rejected, matching prompt syntax.
    A call whose ONLY trailers follow the parens (`fn(x).field`, dotted
    access on the result) is fine; those trailers are parsed after `call_suffix`,
    so the guard checks for a trailer appearing BEFORE the suffix in the child
    order.

    A lone atom (e.g. a bare `NUMBER`/`STRING`) inlines all the way up to
    a top-level `Token` now that no spurious `coalesce` wraps it — such a top has
    no subtrees to check, so it is trivially accepted.
    """
    if not isinstance(tree, Tree):
        return
    for node in tree.iter_subtrees():
        if node.data != "refcall":
            continue
        has_call = False
        trailer_before_call = False
        for child in node.children:
            if isinstance(child, Tree) and child.data == "call_suffix":
                has_call = True
            elif isinstance(child, Tree) and child.data == "trailer" and not has_call:
                trailer_before_call = True
        if has_call and trailer_before_call:
            raise ExpressionError(
                "builtin call must have a bare callee (no dotted access before "
                "'('): a form like 'a.b(x)' is not allowed"
            )


def is_bare_name(node: Tree | Token) -> bool:
    """True for a `refcall` that is one non-keyword `NAME` with no trailers or call suffix."""
    return (
        isinstance(node, Tree)
        and node.data == "refcall"
        and len(node.children) == 1
        and isinstance(node.children[0], Token)
        and node.children[0].type == "NAME"
        and str(node.children[0]) not in _RESERVED_REFERENCE_ROOTS
    )


def _validate_isinstance(tree: Tree | Token) -> None:
    """Require every `isinstance(...)` call to be `isinstance(<expr>, <TypeName>)`.

    The second argument names a type, so it must be a bare `NAME`: no dotted path,
    call, or literal. Keyword arguments and dotted access on the
    boolean result (`isinstance(x, T).f`) are not accepted.
    """
    if not isinstance(tree, Tree):
        return
    for node in tree.iter_subtrees():
        if node.data != "refcall" or str(node.children[0]) != ISINSTANCE:
            continue
        call_suffix = next(
            (c for c in node.children if isinstance(c, Tree) and c.data == "call_suffix"),
            None,
        )
        if call_suffix is None:
            continue
        if node.children[-1] is not call_suffix:
            raise ExpressionError("isinstance(...) gives a bool; it has no fields to read")
        args = call_suffix.children
        if any(len(arg.children) == 2 for arg in args):
            raise ExpressionError("isinstance takes positional arguments, not keyword arguments")
        if len(args) != 2:
            raise ExpressionError(
                f"isinstance takes exactly two arguments (a value and a type name), got {len(args)}"
            )
        if not is_bare_name(args[1].children[0]):
            raise ExpressionError(
                "isinstance's second argument must be a bare type name, e.g. isinstance(x, Buy)"
            )


def _validate_call_arguments(tree: Tree | Token) -> None:
    """Reject a call that repeats a keyword argument, which no callee accepts."""
    if not isinstance(tree, Tree):
        return
    for node in tree.iter_subtrees():
        if node.data != "refcall":
            continue
        call_suffix = next(
            (c for c in node.children if isinstance(c, Tree) and c.data == "call_suffix"),
            None,
        )
        if call_suffix is None:
            continue
        keywords = [str(arg.children[0]) for arg in call_suffix.children if len(arg.children) == 2]
        repeated = next((name for i, name in enumerate(keywords) if name in keywords[:i]), None)
        if repeated is not None:
            raise ExpressionError(f"{node.children[0]}(...) repeats the keyword {repeated!r}")


def parse_expr(text: str) -> Tree | Token:
    """
    Parse one unified `${...}` expression into a Lark tree (PARSE ONLY).

    This does no evaluation and touches no variable pool — it is the shared front
    end for the evaluator in `expr.expressions`. A dotted-callee builtin call (`a.b(x)`) and
    a malformed `isinstance(x, T)` are rejected here; every other construct in the module
    docstring parses.

    Args:
        text (`str`):
            The expression source (the interior of a `${...}`, or a bare
            expression). May be any construct the grammar accepts.

    Returns:
        `Tree | Token`:
            The Lark parse tree. Its shape is the grammar above: `refcall` for a
            ref-or-call, `call_suffix` marking a call, `coalesce` (present ONLY
            when a `|` actually appears) / `default_expr` / `required_expr` at the
            top, and the boolean chains below. An arithmetic chain is one flat
            `sum`, `product` or `power` node whose operands alternate with their
            `ADD_OP` / `MUL_OP` / `POW_OP` tokens. A LONE literal — a bare `NUMBER`,
            `STRING`, `BOOL`, or `NULL` — inlines all the way up and comes back as a
            top-level `Token` (no wrapper node), since no spurious one-child
            `coalesce` is inserted.

    Raises:
        `ExpressionError`:
            If `text` holds a `${...}` (see `reject_nested_spans`), does not parse,
            nests deeper than `current().max_value_nesting` levels, holds an int literal
            with more than `current().max_int_digits` digits, is a dotted-callee builtin
            call, uses a reserved operator word as a reference root, calls
            `isinstance` with anything but a value and a bare type name, or repeats a
            keyword argument.
    """
    reject_nested_spans(text)
    try:
        tree = _PARSER.parse(text)
    except LarkError as exc:
        raise ExpressionError(f"could not parse expression {text!r}: {exc}") from exc
    # First, so no later walk meets a tree deep enough to overflow it.
    _reject_deep_expression(tree)
    _reject_long_int_literals(tree)
    _reject_dotted_callee(tree)
    # Before the reserved-root check, so `isinstance(x, in)` reports the type-name rule.
    _validate_isinstance(tree)
    _validate_call_arguments(tree)
    _reject_reserved_reference_roots(tree)
    return tree
