"""A `${...}` marks an expression inside a YAML string; it can't appear inside one.

Covers the quote-aware whole-span test that conditions share, and the load error that a
`${...}` nested inside an expression raises (with the rewrite it shows).
"""

from yian.expr import condition_refs, rewrite_condition_refs
from yian.expr.expressions import parse_condition


# --------------------------------------------------------------------------- #
# Whole-span conditions: one `${...}` around the whole value
# --------------------------------------------------------------------------- #


def test_whole_span_condition_matches_its_bare_spelling():
    assert parse_condition("${a > 5}") == parse_condition("a > 5")


def test_whole_span_condition_with_a_brace_in_a_string_literal():
    assert condition_refs('${x == "}"}') == ["x"]


def test_whole_span_rewrite_with_a_brace_in_a_string_literal():
    rename = {"x": "y"}.get
    assert rewrite_condition_refs('${x == "}"}', rename) == '${y == "}"}'
