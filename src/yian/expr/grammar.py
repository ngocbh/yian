r"""One unified `${...}` expression grammar — PARSE ONLY (no evaluation here).

This single grammar parses binding coalesce/default expressions, `when:`
boolean/arithmetic expressions, and prompt builtin calls into one Lark tree.
Evaluation deliberately lives in `expr.expressions`.

## What it parses

- References: bare `a`, dotted `a.b.c`, and — for the runtime graph-expansion
  separators — segments containing `#` / `/` (`node#0.output`, `def/child.output`).
- The `${...}`-wrapped back-compat ref form (`${a}`), so old sources keep parsing.
- Arithmetic (`+ - * / %` and `**`), unary minus, comparisons
  (`== != < <= > >=`), `and` / `or` / `not`, `in` / `not in`.
- List literals (`[1, "x"]`) and parenthesised sub-expressions.
- Pure builtin calls (`upper(name)`, `join(items, ", ")`), with dotted access on
  the result (`fn(x).field`) — mirroring the prompt call form.
- Coalesce (`a | b | c`), default (`a :- d`), required (`a :? "msg"`), including a
  wrapped-ref default RHS (`a :- ${b}`).
- The `isinstance(x, T)` special form, whose second argument is a bare type name.

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

from lark import Lark, Token, Tree
from lark.exceptions import LarkError

# Reuse the existing expression error — `expressions.ExpressionError` is already
# the base of `expressions.RequiredError`, so parse failures here stay in one family.
from yian.expr.expressions import ISINSTANCE, ExpressionError

# The unified grammar. Precedence, lowest to highest:
#   coalesce (`|`) -> default/required (`:-`/`:?`) -> or -> and -> not
#   -> comparison (incl. in / not in) -> sum -> product -> unary minus
#   -> power (`**`, right-assoc) -> atom.
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

?sum: product
    | sum "+" product   -> add
    | sum "-" product   -> sub
?product: unary
        | product "*" unary   -> mul
        | product "/" unary   -> div
        | product "%" unary   -> mod
// unary minus binds looser than `**` (Python: `-x ** 2` == `-(x ** 2)`).
?unary: power
      | "-" unary   -> neg
// `**` is right-associative and binds tighter than unary minus.
?power: atom
      | atom "**" unary   -> power

?atom: refcall
     | NUMBER | STRING | BOOL | NULL
     | list_lit
     | WRAPPED_REF
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

COMP_OP: "==" | "!=" | "<=" | ">=" | "<" | ">"
WRAPPED_REF: /\$\{[^}]+\}/
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

_RESERVED_REFERENCE_ROOTS = frozenset({"and", "or", "not", "in"})

# The keyword terminals in `_GRAMMAR` above; none of them can be a reference root.
EXPRESSION_KEYWORDS = frozenset({"and", "or", "not", "in", "true", "false", "null", "none"})


def _raise_if_reserved_reference_root(path: str) -> None:
    """Reject an exact operator word at the root of one reference path."""
    root = path.strip().split(".", 1)[0].strip()
    if root in _RESERVED_REFERENCE_ROOTS:
        raise ExpressionError(
            f"reserved expression keyword {root!r} cannot be used as a reference root"
        )


def _reject_reserved_reference_roots(tree: Tree | Token) -> None:
    """Reject operator words parsed contextually as reference roots.

    A `refcall` with a `call_suffix` is a builtin call rather than a reference, so
    its callee is outside this rule. Trailer segments and keyword-argument labels
    are likewise not roots. `WRAPPED_REF` is a single token, so inspect its path
    explicitly both at the top level and when nested inside a larger expression.
    """
    if isinstance(tree, Token):
        wrapped_refs = (tree,) if tree.type == "WRAPPED_REF" else ()
    else:
        for node in tree.iter_subtrees():
            if node.data != "refcall":
                continue
            if any(
                isinstance(child, Tree) and child.data == "call_suffix"
                for child in node.children
            ):
                continue
            head = node.children[0]
            if isinstance(head, Token) and head.type == "NAME":
                _raise_if_reserved_reference_root(str(head))
        wrapped_refs = tree.scan_values(
            lambda value: isinstance(value, Token) and value.type == "WRAPPED_REF"
        )

    for token in wrapped_refs:
        _raise_if_reserved_reference_root(str(token)[2:-1])


def _reject_dotted_callee(tree: Tree | Token) -> None:
    """Raise if any `refcall` is a call (`call_suffix`) with a leading `trailer`.

    Builtins are bare-callee only: `a.b(x)` — a dotted head applied like a call —
    is not a valid builtin call and must be rejected, matching prompt syntax.
    A call whose ONLY trailers follow the parens (`fn(x).field`, dotted
    access on the result) is fine; those trailers are parsed after `call_suffix`,
    so the guard checks for a trailer appearing BEFORE the suffix in the child
    order.

    A lone atom (e.g. `${a}`, a bare `NUMBER`/`STRING`) inlines all the way up to
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
    call, literal, or `${...}` wrapping. Keyword arguments and dotted access on the
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
            top, and the arithmetic/boolean chain below. A LONE atom — a bare
            `${a}` wrapped-ref, or a bare `NUMBER` / `STRING` — inlines all the
            way up and comes back as a top-level `Token` (no wrapper node), since
            no spurious one-child `coalesce` is inserted.

    Raises:
        `ExpressionError`:
            If `text` does not parse, is a dotted-callee builtin call, uses a
            reserved operator word as a reference root, or calls `isinstance`
            with anything but a value and a bare type name.
    """
    try:
        tree = _PARSER.parse(text)
    except LarkError as exc:
        raise ExpressionError(f"could not parse expression {text!r}: {exc}") from exc
    _reject_dotted_callee(tree)
    # Before the reserved-root check, so `isinstance(x, in)` reports the type-name rule.
    _validate_isinstance(tree)
    _reject_reserved_reference_roots(tree)
    return tree
