import pytest

from yian.expr.grammar import parse_expr
from yian.expr.expressions import ExpressionError
from yian.compose.desugar.case_outputs import _CASE_OUTPUT_INTERIOR


def test_namespaced_path_parses():
    # a namespaced expansion ref: ${each#0/score.output.output} must parse (no ExpressionError)
    parse_expr("each#0/score.output.output")
    parse_expr("a/b.output.output")
    parse_expr("each#0/score.output")
    parse_expr("a/b.output")


def test_existing_dotted_paths_still_parse():
    parse_expr("research.output.report")
    parse_expr("outputs.research.report")


def test_genuinely_bad_path_still_rejected_unchanged():
    with pytest.raises(ExpressionError, match="could not parse expression"):
        parse_expr("a..b")          # empty segment still bad


def test_case_output_interior_accepts_separators():
    # the case-output regex accepts the new node-first
    # `<id>.output[.<seg>…]` shape with namespaced ids (`each#0/leaf`).
    assert _CASE_OUTPUT_INTERIOR.match("each#0/leaf.output")
    assert _CASE_OUTPUT_INTERIOR.match("research.output.report")
