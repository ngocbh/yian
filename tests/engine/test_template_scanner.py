"""Tests for the template scanner over the ONE unified parser.

`scan_template(text)` splits a text string into a list of `TypedValue`s — a LITERAL
run (raw text, `$$` already collapsed to `$`) or a SPAN whose interior has been
parsed via `expr.grammar.parse_expr`. **Only span interiors are parsed** — operator
characters (`|`, `+`, `[`, ...) in literal text are NEVER parsed as expression
operators, which is what makes free text like `"stance (positive|negative|neutral)"`
safe.

`eval_template(segments, resolve, item, mode)` returns the TYPED value of a
whole-single-span template (a float stays a float), else stringifies each span's
result and concatenates it with the literal runs. `template_refs(segments)` unions
the span refs in source order.
"""

import sys

import pytest

from yian.expr.expressions import ExpressionError, ResolveMode
from yian.expr.template import (
    Literal,
    Span,
    eval_template,
    render_template_record,
    scan_template,
    template_refs,
)


def _from(mapping):
    """A `resolve` callable backed by a nested dict, dict-only dotted walk (mirrors
    the production resolvers / test_expr_eval's `_from`)."""

    def _resolve(path):
        parts = path.split(".")
        value = mapping.get(parts[0])
        for step in parts[1:]:
            value = value.get(step) if isinstance(value, dict) else None
        return value

    return _resolve


def _eval(text, resolve=None, item=None, mode=ResolveMode.BINDING_NONE):
    resolve = resolve or (lambda _path: None)
    return eval_template(scan_template(text), resolve, item=item, mode=mode)


# --------------------------------------------------------------------------- #
# Whole-single-span preserves type
# --------------------------------------------------------------------------- #


def test_whole_single_span_preserves_float_type():
    # `"${score}"` where score is a float -> returns a FLOAT (not the str "0.5").
    result = _eval("${score}", _from({"score": 0.5}))
    assert result == 0.5
    assert isinstance(result, float)


def test_whole_single_span_preserves_list_type():
    result = _eval("${items}", _from({"items": [1, 2, 3]}))
    assert result == [1, 2, 3]
    assert isinstance(result, list)


def test_whole_single_span_missing_is_none():
    assert _eval("${gone}", mode=ResolveMode.BINDING_NONE) is None


# --------------------------------------------------------------------------- #
# Embedded span stringifies
# --------------------------------------------------------------------------- #


def test_embedded_span_stringifies():
    # `"n=${score}"` with score=0.5 -> the string "n=0.5" (span coerced to str).
    assert _eval("n=${score}", _from({"score": 0.5})) == "n=0.5"


def test_two_embedded_spans_with_literal_runs():
    # `"a ${b} c ${d}"` -> two spans with the literal runs preserved between them.
    result = _eval("a ${b} c ${d}", _from({"b": "B", "d": "D"}))
    assert result == "a B c D"


def test_embedded_missing_span_stringifies_to_empty():
    # A missing embedded span stringifies to "" (binding stringification of None).
    assert _eval("x=${gone}!", mode=ResolveMode.BINDING_NONE) == "x=!"


# --------------------------------------------------------------------------- #
# Literal text is never parsed — operator characters stay literal
# --------------------------------------------------------------------------- #


def test_pure_literal_with_operator_chars_untouched():
    # No `${}` at all: `|`, `(`, `)` are NOT expression operators — returned verbatim.
    text = "stance (positive|negative|neutral)"
    assert _eval(text) == text


def test_dollar_dollar_collapses_to_single_dollar():
    assert _eval("$$") == "$"


def test_lone_dollar_stays_literal():
    # A `$` not starting a `${` span stays literal.
    assert _eval("cost is $5") == "cost is $5"


def test_operator_chars_outside_span_stay_literal_with_span_present():
    # KEY property: a literal `|` / `+` / `[` OUTSIDE a span is untouched even when a
    # span exists elsewhere. Only the span interior is parsed.
    result = _eval("pick a|b|c: ${choice} + [x]", _from({"choice": "b"}))
    assert result == "pick a|b|c: b + [x]"


def test_span_interior_is_parsed():
    # Inside the span the `|` IS a coalesce operator — proves interiors go through
    # the unified grammar (the pool value wins the coalesce).
    assert _eval("${a | b}", _from({"b": "fallback"})) == "fallback"


# --------------------------------------------------------------------------- #
# scan_template segment shapes
# --------------------------------------------------------------------------- #


def test_scan_pure_literal_is_one_literal_segment():
    segs = scan_template("just text")
    assert len(segs) == 1
    assert isinstance(segs[0], Literal)
    assert segs[0].text == "just text"


def test_scan_dollar_dollar_collapsed_in_literal():
    segs = scan_template("a$$b")
    assert segs == [Literal("a$b")]


def test_scan_two_spans_and_literals():
    segs = scan_template("a ${b} c ${d}")
    kinds = [type(s).__name__ for s in segs]
    assert kinds == ["Literal", "Span", "Literal", "Span"]
    assert segs[0].text == "a "
    assert segs[2].text == " c "


def test_scan_whole_single_span_is_one_span():
    segs = scan_template("${score}")
    assert len(segs) == 1
    assert isinstance(segs[0], Span)


def test_scan_unbalanced_span_raises():
    with pytest.raises(ExpressionError):
        scan_template("${a")


def test_scan_empty_string_is_no_segments():
    # An empty template scans to NO segments; eval_template of it returns "".
    assert scan_template("") == []
    assert _eval("") == ""


def test_scan_empty_span_raises():
    # `${}` / `${   }` — an empty/whitespace-only interior fails parse_expr.
    with pytest.raises(ExpressionError):
        scan_template("${}")
    with pytest.raises(ExpressionError):
        scan_template("${   }")


def test_scan_adjacent_spans_are_two_spans_and_concatenate():
    # `${a}${b}` is TWO Span segments (NOT one whole-span) — so eval stringifies each
    # and concatenates (NOT the typed value of a single span). Pins the
    # whole-single-span-vs-embedded boundary.
    segs = scan_template("${a}${b}")
    assert [type(s).__name__ for s in segs] == ["Span", "Span"]
    assert _eval("${a}${b}", _from({"a": 1, "b": 2})) == "12"


# --------------------------------------------------------------------------- #
# template_refs
# --------------------------------------------------------------------------- #


def test_template_refs_source_order_no_dedupe():
    segs = scan_template("${a} and ${b.c} and ${a}")
    # Union in source order, NO dedupe (matching expr_refs convention).
    assert template_refs(segs) == ["a", "b.c", "a"]


def test_template_refs_pure_literal_is_empty():
    assert template_refs(scan_template("no refs (a|b)")) == []


def test_template_refs_ignores_literals_between_spans():
    segs = scan_template("x ${one} y ${two}")
    assert template_refs(segs) == ["one", "two"]


# --------------------------------------------------------------------------- #
# Item scope and mode pass-through
# --------------------------------------------------------------------------- #


def test_eval_template_item_scope():
    assert _eval("${item.title}", item={"title": "T"}) == "T"


def test_eval_template_strict_mode_raises_on_miss():
    with pytest.raises(ExpressionError):
        _eval("${gone}", mode=ResolveMode.STRICT_RAISE)


# --------------------------------------------------------------------------- #
# Rendering an embedded span's value as text
# --------------------------------------------------------------------------- #

# Python 3.10.0-3.10.6 has no int-to-str digit limit, so a huge int renders there.
needs_digit_limit = pytest.mark.skipif(
    not hasattr(sys, "get_int_max_str_digits"), reason="no int-to-str digit limit"
)


@needs_digit_limit
def test_eval_template_embedded_int_too_large_for_text_raises_expression_error():
    with pytest.raises(ExpressionError, match="as text"):
        eval_template(scan_template("x=${n}"), lambda p: 10**5000 if p == "n" else None)


@needs_digit_limit
def test_eval_template_embedded_list_holding_a_huge_int_raises_expression_error():
    with pytest.raises(ExpressionError, match="as text"):
        eval_template(scan_template("x=${n}"), lambda p: [10**5000])


@needs_digit_limit
def test_eval_template_whole_single_span_keeps_a_huge_int_typed():
    value = eval_template(scan_template("${n}"), lambda p: 10**5000 if p == "n" else None)
    assert value == 10**5000


@needs_digit_limit
def test_render_template_record_whole_span_int_too_large_for_text_raises_expression_error():
    with pytest.raises(ExpressionError, match="as text"):
        render_template_record("${n}", {"n": 10**5000})
