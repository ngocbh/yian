"""Tests for the position-splicing ref REWRITERS — the write-side analog of the
`expr_refs` walk.

`rewrite_expr_refs(text, rename)` reparses one unified `${...}` expression and splices
each reference-leaf path through `rename(path)` (return a new path, or `None` to keep),
leaving every operator / literal / builtin-callee / whitespace verbatim.
`rewrite_condition_refs` wraps it for the two condition spellings (bare / whole-span);
`rewrite_template_refs` wraps it for a multi-span binding/template value.

These support call/loop/map inlining and searched-case rebinding across whole-span and
bare expressions.
"""

import pytest

from yian.expr.expressions import ExpressionError, rewrite_condition_refs, rewrite_expr_refs
from yian.expr.template import rewrite_template_refs


# A rename that upper-cases the head segment, e.g. `a.b` -> `A.b`; keeps `keep.*`.
def _bump(path: str):
    if path.startswith("keep"):
        return None
    head, _, rest = path.partition(".")
    return head.upper() + (("." + rest) if rest else "")


# --------------------------------------------------------------------------- #
# rewrite_expr_refs — leaves spliced, everything else verbatim
# --------------------------------------------------------------------------- #


def test_rewrite_bare_refcall_leaf():
    assert rewrite_expr_refs("a > 5", _bump) == "A > 5"


def test_rewrite_dotted_refcall_preserves_tail():
    assert rewrite_expr_refs("a.b.c >= foo.output", _bump) == "A.b.c >= FOO.output"


def test_rewrite_leaves_operators_and_literals_verbatim():
    assert rewrite_expr_refs('a in ["x", "y"] and b < 3', _bump) == 'A in ["x", "y"] and B < 3'


def test_rewrite_none_keeps_reference():
    assert rewrite_expr_refs("keep.x > a", _bump) == "keep.x > A"


def test_rewrite_builtin_callee_not_renamed_but_args_are():
    # the callee `upper` is not a reference; the arg ref `name` is rewritten.
    assert rewrite_expr_refs("upper(name) == foo", _bump) == "upper(NAME) == FOO"


def test_rewrite_default_rhs():
    # `x :- y` — both the head and the default ref are rewritten.
    assert rewrite_expr_refs("x :- y", _bump) == "X :- Y"


def test_rewrite_hash_slash_segments():
    assert rewrite_expr_refs("node#0.output", lambda p: "ns/" + p) == "ns/node#0.output"


# --------------------------------------------------------------------------- #
# rewrite_condition_refs — both spellings rewrite identically
# --------------------------------------------------------------------------- #


def test_condition_bare():
    assert rewrite_condition_refs("a > 5", _bump) == "A > 5"


def test_condition_with_a_nested_span_is_rejected():
    with pytest.raises(ExpressionError, match=r"write `a > 5`"):
        rewrite_condition_refs("${a} > 5", _bump)


def test_condition_whole_span_keeps_outer_braces():
    assert rewrite_condition_refs("${a > 5}", _bump) == "${A > 5}"


def test_condition_whole_span_multi_ref():
    assert (
        rewrite_condition_refs("${a > 5 and b < 3}", _bump) == "${A > 5 and B < 3}"
    )


# --------------------------------------------------------------------------- #
# rewrite_template_refs — spans rewritten, literal text + $$ preserved
# --------------------------------------------------------------------------- #


def test_template_whole_span():
    assert rewrite_template_refs("${a.output}", _bump) == "${A.output}"


def test_template_embedded_text_preserved():
    assert (
        rewrite_template_refs("pre ${a} mid ${b.x} post", _bump)
        == "pre ${A} mid ${B.x} post"
    )


def test_template_dollar_escape_preserved():
    # `$$` is a literal `$` — it must survive verbatim (not be read as a span start).
    assert rewrite_template_refs("cost is $$${a}", _bump) == "cost is $$${A}"


def test_template_no_span_returned_unchanged():
    assert rewrite_template_refs("plain text", _bump) == "plain text"


def test_template_non_string_passthrough():
    assert rewrite_template_refs(5, _bump) == 5


def test_template_default_span():
    assert rewrite_template_refs("${x :- y}", _bump) == "${X :- Y}"
