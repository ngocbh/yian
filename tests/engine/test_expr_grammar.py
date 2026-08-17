"""Parse-only tests for the unified `${...}` expression grammar (`expr.grammar`).

These assert parse success and coarse tree shape only; no value is computed here.
The shared-terminal tests cover the exact LALR-collision cases that a naive
two-terminal grammar (a bare `REF` terminal plus a separate
`call: NAME "(" ... ")"` rule) fails to parse,
because the lexer matches the bare-ref terminal before it can see the `(`. The
grammar instead uses one shared `NAME` terminal with an optional call-suffix.
"""

import pytest
from lark import Token, Tree

from agent_composer.expr.expressions import ExpressionError
from agent_composer.expr.grammar import parse_expr


def _tree(text: str) -> Tree:
    """Parse `text` and assert we got a Lark `Tree` back (a parse, not an error)."""
    result = parse_expr(text)
    assert isinstance(result, Tree), f"{text!r} did not parse to a Tree"
    return result


def _kinds(tree: Tree) -> set[str]:
    """The set of rule/data names anywhere in `tree` — the coarse shape probe."""
    names: set[str] = set()
    for node in tree.iter_subtrees():
        names.add(node.data)
    return names


# --------------------------------------------------------------------------- #
# References
# --------------------------------------------------------------------------- #


def test_bare_ref():
    assert "refcall" in _kinds(_tree("a"))


def test_lone_expr_not_coalesce_wrapped():
    # A single operand (no `|`) must NOT be wrapped in a spurious `coalesce`
    # node: the top of a lone ref is the ref itself, and arithmetic stays bare.
    assert _tree("a").data == "refcall"
    assert _tree("a").data != "coalesce"
    assert _tree("1 + 2").data == "add"
    assert _tree("1 + 2").data != "coalesce"


def test_dotted_ref():
    tree = _tree("a.b.c")
    assert "refcall" in _kinds(tree)
    # a.b.c is ONE dotted ref: one refcall with trailers, not nested calls.
    refcalls = [n for n in tree.iter_subtrees() if n.data == "refcall"]
    assert len(refcalls) == 1


def test_ref_with_hash():
    assert "refcall" in _kinds(_tree("node#0.output"))


def test_ref_with_slash():
    assert "refcall" in _kinds(_tree("def/child.output"))


def test_wrapped_ref():
    # `${a}` back-compat form must still parse. It is a lone atom, so with no
    # spurious wrapper it inlines up to a bare WRAPPED_REF token (not a Tree).
    result = parse_expr("${a}")
    assert isinstance(result, Token)
    assert result.type == "WRAPPED_REF"


def test_item_ref():
    assert "refcall" in _kinds(_tree("item.title"))


# --------------------------------------------------------------------------- #
# Arithmetic / comparisons / booleans
# --------------------------------------------------------------------------- #


def test_arithmetic():
    _tree("1 + 2 * 3 - 4 / 5 % 6")


def test_power():
    assert "power" in _kinds(_tree("2 ** 3"))


def test_unary_power_precedence():
    # `-x ** 2` must parse (evaluator-side it is `-(x ** 2)`, per Python).
    _tree("-x ** 2")


def test_comparisons():
    for op in ("==", "!=", "<", "<=", ">", ">="):
        _tree(f"a {op} b")


def test_and_or_not():
    _tree("a and b or not c")


def test_in_and_not_in():
    _tree("a in b")
    _tree("a not in b")


def test_list_literal():
    _tree('[1, "x"]')


def test_parens():
    _tree("(a + b) * c")


def test_unary_minus():
    _tree("-x")


def test_subtract_of_negative():
    # `3 - -x`: subtraction of a unary-negated ref.
    _tree("3 - -x")


# --------------------------------------------------------------------------- #
# Builtin calls
# --------------------------------------------------------------------------- #


def test_pure_builtin_call():
    tree = _tree("upper(name)")
    assert "call_suffix" in _kinds(tree)


def test_builtin_call_with_string_arg():
    tree = _tree('join(items, ", ")')
    assert "call_suffix" in _kinds(tree)


def test_call_result_dotted_access():
    # dotted access on a call result (mirrors the prompt `_PromptCall.trailing`).
    _tree("fn(x).field")


def test_dotted_callee_rejected():
    # builtins are bare-callee only: `a.b(...)` must be rejected.
    with pytest.raises(ExpressionError):
        parse_expr("a.b(x)")


def test_empty_arg_call():
    # `fn()` — a call with no args — must parse: a `call_suffix` with no `arg`.
    tree = _tree("fn()")
    assert "call_suffix" in _kinds(tree)
    assert "arg" not in _kinds(tree)


def test_kwarg_distinguishable_from_positional():
    # A kwarg `foo(a=1)` and a positional `foo(1)` both parse, and their `arg`
    # nodes are distinguishable by shape: a kwarg carries a leading NAME token
    # (2 children), a positional does not (1 child, the expression) — the evaluator
    # uses this shape to tell keyword from positional arguments.
    kw_arg = next(n for n in _tree("foo(a=1)").iter_subtrees() if n.data == "arg")
    pos_arg = next(n for n in _tree("foo(1)").iter_subtrees() if n.data == "arg")

    # kwarg: 2 children, leading NAME token naming the parameter.
    assert len(kw_arg.children) == 2
    assert isinstance(kw_arg.children[0], Token)
    assert kw_arg.children[0].type == "NAME"
    assert str(kw_arg.children[0]) == "a"

    # positional: a single child (the value expr) with no leading NAME marker.
    assert len(pos_arg.children) == 1
    only = pos_arg.children[0]
    assert not (isinstance(only, Token) and only.type == "NAME")


# --------------------------------------------------------------------------- #
# Coalesce / default / required
# --------------------------------------------------------------------------- #


def test_coalesce():
    tree = _tree("a | b | c")
    # `a | b | c` DOES build a `coalesce` node (multiple `|` operands) — and it
    # is the top node, since coalesce is lowest precedence.
    assert "coalesce" in _kinds(tree)
    assert tree.data == "coalesce"


def test_default_non_chaining():
    # `:-` takes a SINGLE default only — multi-fallback is what `|` is for. A
    # chained `a :- b :- c` must be a parse error.
    with pytest.raises(ExpressionError):
        parse_expr("a :- b :- c")


def test_required_non_chaining():
    # `:?` is likewise single-only (same reasoning as `:-`).
    with pytest.raises(ExpressionError):
        parse_expr("a :? b :? c")


def test_default():
    tree = _tree('a :- "d"')
    assert "default_expr" in _kinds(tree)


def test_required():
    tree = _tree('a :? "msg"')
    assert "required_expr" in _kinds(tree)


def test_nested_default():
    # `a :- ${b}`: a wrapped-ref default RHS.
    _tree("a :- ${b}")


def test_default_binds_tighter_than_coalesce():
    # `a :- b | c` parses as `(a :- b) | c`: the coalesce is the top node with
    # two operands, the first being the default.
    tree = _tree("a :- b | c")
    coalesce = next(n for n in tree.iter_subtrees() if n.data == "coalesce")
    assert len(coalesce.children) == 2
    assert isinstance(coalesce.children[0], Tree)
    assert coalesce.children[0].data == "default_expr"


def test_coalesce_below_arithmetic():
    # `a | b + 1` parses as `a | (b + 1)`: the coalesce's second operand is a sum.
    tree = _tree("a | b + 1")
    coalesce = next(n for n in tree.iter_subtrees() if n.data == "coalesce")
    assert len(coalesce.children) == 2
    kinds_rhs = _kinds(coalesce.children[1]) if isinstance(coalesce.children[1], Tree) else set()
    assert "add" in kinds_rhs


# --------------------------------------------------------------------------- #
# Shared NAME terminal disambiguation
# --------------------------------------------------------------------------- #


def test_shared_name_terminal_distinguishes_builtin_call():
    """`upper(name)` -> a call (callee `upper`, one ref arg), NOT bare-ref + junk."""
    tree = _tree("upper(name)")
    assert "call_suffix" in _kinds(tree)


def test_shared_name_terminal_parses_mixed_call_arguments():
    """`join(items, ", ")` -> callee + ref arg + string-literal arg."""
    tree = _tree('join(items, ", ")')
    assert "call_suffix" in _kinds(tree)
    assert "arg" in _kinds(tree)


def test_shared_name_terminal_keeps_dotted_reference_single():
    """`a.b.c` -> a single dotted ref (one refcall)."""
    tree = _tree("a.b.c")
    refcalls = [n for n in tree.iter_subtrees() if n.data == "refcall"]
    assert len(refcalls) == 1


def test_name_terminal_accepts_hash_in_reference_segment():
    """`node#0.output` -> parses (`#` in a segment)."""
    _tree("node#0.output")


def test_name_terminal_excludes_hyphen_for_subtraction():
    """`a - b` -> subtraction of two refs, NOT one identifier `a-b`.

    The trap: if `-` is in the shared NAME terminal, `a - b` lexes as a single
    ident and this fails. Two refcalls under a `sub` node proves the split.
    """
    tree = _tree("a - b")
    assert "sub" in _kinds(tree)
    refcalls = [n for n in tree.iter_subtrees() if n.data == "refcall"]
    assert len(refcalls) == 2
