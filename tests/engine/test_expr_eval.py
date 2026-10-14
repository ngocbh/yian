"""Tests for the ONE unified expression evaluator (`expr.expressions.eval_expr`).

These drive `eval_expr(tree, resolve, item, mode)` — the single evaluator that
walks a tree from `expr.grammar.parse_expr` against a `resolve` callable. Unlike
the parse-only grammar tests (`test_expr_grammar.py`), these assert computed
VALUES, the three missing-ref-by-mode behaviors, and the load-bearing semantics
(missing->None, arith-over-None->loud, cmp-with-None->False) that a CASE's
`else:` routing depends on.
"""

import sys

import pytest

from yian.expr.expressions import (
    ExpressionError,
    RequiredError,
    ResolveMode,
    eval_expr,
)
from yian.expr.grammar import parse_expr

# Python 3.10.0-3.10.6 has no int-to-str digit limit, so a huge int's repr works there.
needs_digit_limit = pytest.mark.skipif(
    not hasattr(sys, "get_int_max_str_digits"), reason="no int-to-str digit limit"
)


def _eval(text, resolve=None, item=None, mode=ResolveMode.BINDING_NONE):
    """Parse `text` then evaluate it, defaulting to an empty resolver (all miss)."""
    resolve = resolve or (lambda _path: None)
    return eval_expr(parse_expr(text), resolve, item=item, mode=mode)


def _from(mapping):
    """A `resolve` callable backed by a nested dict, with the dict-ONLY dotted walk
    the production resolvers do (mirrors `_resolve_in_record`): head lookup then
    `.get` per step, never `getattr`. A miss / non-dict step -> None."""

    def _resolve(path):
        parts = path.split(".")
        value = mapping.get(parts[0])
        for step in parts[1:]:
            value = value.get(step) if isinstance(value, dict) else None
        return value

    return _resolve


# --------------------------------------------------------------------------- #
# Operators on resolved Python values
# --------------------------------------------------------------------------- #


def test_int_addition():
    assert _eval("1 + 2") == 3


def test_string_concatenation():
    # str + str concatenates — the number-only gate must be bypassed.
    assert _eval('"a" + "b"') == "ab"


def test_list_concatenation():
    # list + list extends.
    assert _eval("x + y", _from({"x": [1], "y": [2]})) == [1, 2]


def test_power():
    assert _eval("2 ** 3") == 8


def test_subtraction_and_product_and_div():
    assert _eval("10 - 3") == 7
    assert _eval("4 * 5") == 20
    assert _eval("8 / 2") == 4
    assert _eval("7 % 3") == 1


def test_unary_minus():
    assert _eval("-x", _from({"x": 5})) == -5


# --------------------------------------------------------------------------- #
# Errors Python raises while computing an operator become ExpressionError
# --------------------------------------------------------------------------- #


class _Boom:
    """A resolved value whose negation raises the given exception."""

    def __init__(self, exc):
        self.exc = exc

    def __neg__(self):
        raise self.exc


def test_float_power_overflow_is_an_expression_error():
    with pytest.raises(ExpressionError, match=r"\*\*") as info:
        _eval("2.0 ** n", _from({"n": 5000}))
    assert "Result too large" in str(info.value)
    assert "(34" not in str(info.value)


def test_int_too_large_for_float_division_is_an_expression_error():
    with pytest.raises(ExpressionError):
        _eval("n / 3", _from({"n": 10**400}))


def test_string_repetition_overflow_is_an_expression_error():
    with pytest.raises(ExpressionError):
        _eval('"a" * n', _from({"n": 10**20}))


def test_bad_format_directive_is_an_expression_error():
    with pytest.raises(ExpressionError):
        _eval('"%z" % 1')


def test_missing_format_key_is_an_expression_error():
    with pytest.raises(ExpressionError, match="missing key 'k'"):
        _eval('"%(k)s" % r', _from({"r": {"j": 1}}))


@needs_digit_limit
def test_type_error_with_a_huge_int_names_the_operand_type():
    with pytest.raises(ExpressionError, match="<int too large to display>"):
        _eval('n + "a"', _from({"n": 10**5000}))


@needs_digit_limit
def test_division_by_zero_with_a_huge_int():
    with pytest.raises(ExpressionError, match="division by zero: <int too large to display>"):
        _eval("n / 0", _from({"n": 10**5000}))


@needs_digit_limit
def test_comparison_type_error_with_a_huge_int():
    with pytest.raises(ExpressionError, match="<int too large to display>"):
        _eval('n < "a"', _from({"n": 10**5000}))


@needs_digit_limit
def test_unary_minus_on_a_list_holding_a_huge_int():
    with pytest.raises(ExpressionError, match="unary minus on non-number <list too large"):
        _eval("-xs", _from({"xs": [10**5000]}))


@pytest.mark.parametrize("exc", [OverflowError("boom"), ValueError("boom")])
def test_unary_minus_errors_are_expression_errors(exc):
    with pytest.raises(ExpressionError, match="boom"):
        _eval("-x", lambda p: _Boom(exc) if p == "x" else None)


def test_division_by_zero_message_is_unchanged():
    with pytest.raises(ExpressionError, match="division by zero"):
        _eval("1 / 0")


def test_type_error_message_is_unchanged():
    with pytest.raises(ExpressionError, match=r"type error evaluating 1 \+ 'a'"):
        _eval('1 + "a"')


# --------------------------------------------------------------------------- #
# Dotted access — dict-only, NEVER getattr (SAFETY-CRITICAL)
# --------------------------------------------------------------------------- #


def test_dotted_access_dict_key():
    assert _eval("x.k", _from({"x": {"k": 42}})) == 42


def test_dotted_access_never_getattr():
    # `obj.__class__` must MISS (dict has no key "__class__") — never reach the
    # Python attribute. Proves dotted access is dict-key lookup only.
    assert _eval("obj.__class__", _from({"obj": {"a": 1}})) is None


# --------------------------------------------------------------------------- #
# Missing-reference behavior by resolution mode
# --------------------------------------------------------------------------- #


def test_binding_none_missing_is_none():
    # A missing ref resolves to None in the non-strict binding mode.
    assert _eval("a", mode=ResolveMode.BINDING_NONE) is None


def test_condition_falsy_missing_ordered_compare_is_false():
    # An ordered comparison with a missing (None) operand returns False.
    assert _eval("missing > 5", mode=ResolveMode.CONDITION_FALSY) is False


def test_condition_falsy_arith_over_none_is_loud():
    # Arithmetic over a missing (None) operand propagates a loud TypeError,
    # wrapped as ExpressionError (NOT silently None/0).
    with pytest.raises(ExpressionError):
        _eval("missing + 1 > 5", mode=ResolveMode.CONDITION_FALSY)


def test_strict_raise_missing_raises():
    with pytest.raises(ExpressionError):
        _eval("a", mode=ResolveMode.STRICT_RAISE)


def test_binding_none_wrapped_ref_missing_is_none():
    assert _eval("${a}", mode=ResolveMode.BINDING_NONE) is None


# --------------------------------------------------------------------------- #
# Coalesce / default / required
# --------------------------------------------------------------------------- #


def test_coalesce_first_non_none():
    assert _eval("a | b | c", _from({"b": "second", "c": "third"})) == "second"


def test_coalesce_present_falsy_wins():
    # A present falsy value (0 / "" / False) is NOT None, so it wins the coalesce.
    assert _eval("a | b", _from({"a": 0})) == 0


def test_coalesce_short_circuits():
    # fallback (1/0) must NOT be evaluated when the head is present.
    assert _eval("a | 1 / 0", _from({"a": "present"})) == "present"


def test_default_when_missing():
    assert _eval('a :- "d"') == "d"


def test_default_short_circuits_fallback():
    # fallback (1/0) must NOT be evaluated when the head is present — a vacuous
    # inert literal fallback would pass even if short-circuit broke.
    assert _eval("a :- 1 / 0", _from({"a": "present"})) == "present"


def test_nested_default_wrapped_ref():
    # `a :- ${b}`: the default RHS is itself a ref, resolved when `a` is missing.
    assert _eval("a :- ${b}", _from({"b": "fallback"})) == "fallback"


def test_required_raises_when_missing():
    with pytest.raises(RequiredError):
        _eval('a :? "m"')


def test_required_short_circuits_fallback():
    # message (1/0) must NOT be evaluated when the head is present.
    assert _eval("a :? 1 / 0", _from({"a": "here"})) == "here"


# --------------------------------------------------------------------------- #
# Pure builtin dispatch through TEMPLATE_FNS
# --------------------------------------------------------------------------- #


def test_builtin_upper():
    assert _eval("upper(name)", _from({"name": "hi"})) == "HI"


def test_builtin_join():
    assert _eval('join(items, ", ")', _from({"items": ["a", "b"]})) == "a, b"


def test_builtin_kwarg():
    assert _eval("upper(s=name)", _from({"name": "hi"})) == "HI"


@pytest.mark.parametrize(
    ("text", "values", "message"),
    [
        ("upper(name)", {}, "upper\\(\\) got null"),
        ("lower(name)", {}, "lower\\(\\) got null"),
        ("upper(n)", {"n": 3}, "upper\\(\\) takes str, got int"),
        ("lower(xs)", {"xs": ["A"]}, "lower\\(\\) takes str, got list"),
        ("render_as_json(name)", {}, "render_as_json\\(\\) got null"),
        ('join(s, "-")', {"s": "abc"}, "join\\(\\) takes a list, got str"),
        ('join(xs, "-")', {}, "join\\(\\) got null"),
        ('join(xs, "-")', {"xs": ["a", None]}, "join\\(\\) got a null element"),
    ],
)
def test_builtin_rejects_null_and_wrong_types(text, values, message):
    # A builtin never turns null or a wrong-typed value into text (`upper(null)` is not
    # "NONE"); it fails like arithmetic over null does.
    with pytest.raises(ExpressionError, match=message):
        _eval(text, _from(values))


def test_builtin_condition_over_null_fails_instead_of_routing():
    with pytest.raises(ExpressionError, match="upper\\(\\) got null"):
        _eval('upper(name) == "NONE"', _from({}), mode=ResolveMode.CONDITION_FALSY)


def test_builtin_join_stringifies_numbers_and_render_keeps_nested_null():
    assert _eval('join(xs, "-")', _from({"xs": [1, 2]})) == "1-2"
    assert _eval("render_as_json(d, 0)", _from({"d": {"k": None}})) == '{\n"k": null\n}'


def test_builtin_call_result_dotted_access(monkeypatch):
    # dotted access on a call result: the builtin returns a dict, `.field` reads it.
    from yian.expr import builtins as _bi

    monkeypatch.setitem(_bi.TEMPLATE_FNS, "identity", lambda v: v)
    assert _eval("identity(d).k", _from({"d": {"k": "v"}})) == "v"


def test_unknown_builtin_raises():
    with pytest.raises(ExpressionError):
        _eval("no_such_fn(x)", _from({"x": 1}))


# --------------------------------------------------------------------------- #
# `item` scope (MAP-body-local)
# --------------------------------------------------------------------------- #


def test_item_scope():
    assert _eval("item.title", item={"title": "T"}) == "T"


def test_item_head_bare():
    assert _eval("item", item={"a": 1}) == {"a": 1}


def test_item_scope_dict_only():
    # dotted access under `item` is also dict-only, never getattr.
    assert _eval("item.__class__", item={"a": 1}) is None


# --------------------------------------------------------------------------- #
# `and` / `or` fold by Python truthiness
# --------------------------------------------------------------------------- #


def test_or_folds_by_python_truthiness():
    # `x or y` with x a FALSY non-bool (0) and y truthy: the non-bool operand
    # participates. The result is True because y is truthy.
    assert _eval("x or y", _from({"x": 0, "y": "hit"})) is True


def test_and_folds_by_python_truthiness():
    # `x and y` with x a falsy non-bool (""): the empty string participates as
    # falsy, so the `and` is False (not filtered out as a non-bool).
    assert _eval("x and y", _from({"x": "", "y": "hit"})) is False


# --------------------------------------------------------------------------- #
# Membership, unary-missing, and list edge cases
# --------------------------------------------------------------------------- #


def test_not_in_list_true():
    assert _eval('"z" not in items', _from({"items": ["a", "b"]})) is True


def test_not_in_list_false():
    assert _eval('"a" not in items', _from({"items": ["a", "b"]})) is False


def test_unary_minus_on_missing_ref_raises_loudly():
    # `-missing` in binding-none: the miss coerces to None, and unary minus over
    # None raises a loud (wrapped) ExpressionError — NOT a silent value.
    with pytest.raises(ExpressionError):
        _eval("-missing", mode=ResolveMode.BINDING_NONE)


def test_list_lit_with_missing_element():
    # `[a, 1]` with `a` missing (binding-none): the miss becomes None inside the list.
    assert _eval("[a, 1]", mode=ResolveMode.BINDING_NONE) == [None, 1]


# --------------------------------------------------------------------------- #
# Coalesce / default / required PRECEDENCE — pinned against the grammar's grouping
# (split on `|` at the top, then parse each operand).
# Every bare name is a reference and a literal must be quoted, so in
# `a:-b | c` the names `a`, `b`, `c` are all references. The grouping MUST be
# `(a:-b) | c` and `a | (b + 1)` — coalesce (`|`) is the lowest-precedence,
# n-ary, flat operator; `:-`/`:?` and arithmetic bind tighter.
# --------------------------------------------------------------------------- #


def _ref_name(node):
    """The NAME of a bare-reference operand: a `refcall` Tree with a single NAME
    child. Raises if `node` is not that shape."""
    from lark import Tree

    assert isinstance(node, Tree) and node.data == "refcall", node
    return str(node.children[0])


def test_default_binds_tighter_than_coalesce_grouping_tree():
    # STRUCTURAL proof of `(a:-b) | c` (NOT `a:-(b | c)`): the parse tree is a
    # `coalesce` whose FIRST operand is a `default_expr(a, b)` and whose SECOND
    # operand is the ref `c`. Under `a:-(b | c)` the top node would instead be a
    # `default_expr` wrapping a `coalesce` — a shape this asserts against.
    from lark import Tree

    tree = parse_expr("a:-b | c")
    assert isinstance(tree, Tree) and tree.data == "coalesce"
    first, second = tree.children
    assert isinstance(first, Tree) and first.data == "default_expr"
    assert _ref_name(first.children[0]) == "a" and _ref_name(first.children[1]) == "b"
    assert _ref_name(second) == "c"


def test_default_coalesce_grouping_default_supplies_then_short_circuits():
    # `(a:-b) | c` with `a` missing, `b`=10: the default supplies `b` (a absent),
    # then coalesce short-circuits on the non-None 10 — `c` is never consulted.
    # `c` is poisoned to None to make "never consulted" observable either way, but
    # the value pins the grouping: the default fired on `b`, not on `b | c`.
    assert _eval("a:-b | c", _from({"b": 10, "c": None})) == 10


def test_default_coalesce_grouping_falls_through_to_c():
    # `(a:-b) | c` with `a` AND `b` missing, `c`=20: `a:-b` yields None (both
    # absent), so the coalesce falls through to `c` -> 20.
    assert _eval("a:-b | c", _from({"c": 20})) == 20


def test_coalesce_arithmetic_grouping_tree():
    # STRUCTURAL proof of `a | (b + 1)`: a `coalesce` whose second operand is an
    # `add` node (arithmetic binds tighter than `|`), NOT `(a | b) + 1`.
    from lark import Tree

    tree = parse_expr("a | b + 1")
    assert isinstance(tree, Tree) and tree.data == "coalesce"
    first, second = tree.children
    assert _ref_name(first) == "a"
    assert isinstance(second, Tree) and second.data == "add"


def test_coalesce_arithmetic_grouping_value():
    # `a | (b + 1)` with `a` missing, `b`=4 -> 5 (the arithmetic operand supplies it).
    assert _eval("a | b + 1", _from({"b": 4})) == 5


def test_coalesce_arithmetic_short_circuits_arm():
    # `a | (b + 1)` with `a`=7 -> 7; the `b + 1` arm is never evaluated even though
    # `b` is missing (arithmetic over a missing ref would otherwise raise loudly).
    assert _eval("a | b + 1", _from({"a": 7})) == 7


def test_coalesce_nary_flat_tree():
    # `a | b | c` parses to ONE flat 3-operand `coalesce`, not nested pairs.
    from lark import Tree

    tree = parse_expr("a | b | c")
    assert isinstance(tree, Tree) and tree.data == "coalesce"
    assert [_ref_name(c) for c in tree.children] == ["a", "b", "c"]


def test_coalesce_nary_all_arms_reachable():
    # A flat 3-way coalesce: the third arm is reached when the first two miss.
    assert _eval("a | b | c", _from({"c": 3})) == 3
    # And each arm wins in turn when it is the first present one.
    assert _eval("a | b | c", _from({"b": 2, "c": 3})) == 2
    assert _eval("a | b | c", _from({"a": 1, "b": 2, "c": 3})) == 1


def test_nested_one_level_default_ref():
    # `${x:-${y}}` — one level of nesting is allowed: `x` missing, `y`=9 -> 9.
    from yian.expr.template import eval_binding

    assert eval_binding("${x:-${y}}", _from({"y": 9})) == 9
    # Both missing -> None in the binding (BINDING_NONE) mode.
    assert eval_binding("${x:-${y}}", _from({})) is None


def test_two_level_nested_default_rejected():
    # `${x:-${y:-${z}}}` — two levels of nesting is rejected by the scanner/grammar
    # in the ExpressionError family.
    from yian.expr.template import scan_template

    with pytest.raises(ExpressionError):
        scan_template("${x:-${y:-${z}}}")
