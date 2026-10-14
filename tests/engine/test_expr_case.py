"""Unit tests for the expression evaluator + the real CASE node."""

import pytest

from yian.expr.expressions import (
    ExpressionError,
    evaluate_when,
    evaluate_when_record,
    resolve_reference,
)
from yian.compile.model import END_ID, START_ID, Edge, CompiledFlow
from yian.nodes.end import EndNode
from yian.nodes.case import ELSE_HANDLE, Case, CaseNode
from yian.nodes.start import StartNode
from yian.typesys.pool import VariablePool
from tests.engine._fakes import FuncNode, RecordNode, derive_wiring, stamp_reads
from tests.engine._graph_builder import _engine


def _with_boundary(node_map: dict) -> dict:
    # inject the real START_ID/END_ID boundary NODES so a manual `from_parts` flow runs.
    node_map = dict(node_map)
    node_map.setdefault(START_ID, StartNode(START_ID, params=[]))
    node_map.setdefault(END_ID, EndNode.value(END_ID, output_name=None))
    return node_map


# --- expression evaluation -------------------------------------------------- #


def _pool_with(**outputs):
    pool = VariablePool()
    for node_id, value in outputs.items():
        # `.output` is a SYNTACTIC discriminator the resolver SKIPS; store the
        # node's value directly. A `${<id>.output}` ref evaluates to `value` here.
        pool.set(node_id, value)
    return pool


def test_resolve_reference_structured():
    pool = VariablePool()
    # `.output` is a syntactic discriminator the resolver skips, NOT a stored key.
    # Store the node value directly; `resolve("a", ["output", "pe"])` traverses store["a"]["pe"].
    pool.set("a", {"pe": 21.0})
    assert resolve_reference("a.output.pe", pool) == 21.0
    assert resolve_reference("a.output", pool) == {"pe": 21.0}


def test_when_numeric_and_boolean_logic():
    pool = _pool_with(score=0.8)
    assert evaluate_when("score.output > 0.5", pool) is True
    assert evaluate_when("score.output > 0.5 and score.output < 1", pool) is True
    assert evaluate_when("score.output > 0.9 or score.output == 0.8", pool) is True
    assert evaluate_when("not score.output > 0.9", pool) is True


def test_when_string_and_in():
    pool = _pool_with(label="approve")
    assert evaluate_when("label.output == 'approve'", pool) is True
    assert evaluate_when("label.output in 'approve/reject'", pool) is True


def test_when_missing_reference_is_falsy_not_error():
    pool = VariablePool()
    assert evaluate_when("ghost.output > 5", pool) is False


def test_when_bare_reference_is_truthiness():
    # A bare ref in a `when:` is a truthiness test, coerced to bool at the boundary.
    assert evaluate_when("${x.output}", _pool_with(x=1)) is True
    assert evaluate_when("${x.output}", _pool_with(x=0)) is False
    assert evaluate_when("${missing.output}", VariablePool()) is False


# --- the three when: spellings evaluate identically (unified entry rule) ----- #


def test_when_both_spellings_evaluate_identically():
    # An expression field may be a bare expr or one whole `${expr}`. Both evaluate
    # identically through the unified entry (whole-span strip else parse whole).
    pool = _pool_with(a=8)
    assert evaluate_when("a.output > 5", pool) is True
    assert evaluate_when("${a.output > 5}", pool) is True
    low = _pool_with(a=3)
    assert evaluate_when("a.output > 5", low) is False
    assert evaluate_when("${a.output > 5}", low) is False


def test_when_with_braces_on_a_ref_is_rejected():
    # Braces on a ref inside the condition are a `${...}` inside an expression.
    with pytest.raises(ExpressionError, match=r"write `a.output > 5`"):
        evaluate_when("${a.output} > 5", _pool_with(a=8))


def test_when_bare_arithmetic_condition():
    # Arithmetic inside a condition works on the bare-field path.
    pool = _pool_with(a=5)
    assert evaluate_when("a.output + 1 > 5", pool) is True
    assert evaluate_when("a.output + 1 > 10", pool) is False


def test_when_record_both_spellings_identical():
    from yian.expr.expressions import evaluate_when_record

    rec = {"a": 8}
    assert evaluate_when_record("a > 5", rec) is True
    assert evaluate_when_record("${a > 5}", rec) is True


def test_first_failing_assert_both_spellings():
    from yian.expr.expressions import first_failing_assert

    pool = _pool_with(a=8)
    # all hold -> None
    assert first_failing_assert(["a.output > 5", "${a.output > 5}"], pool) is None
    # the failing one is returned (verbatim), and both spellings agree
    assert first_failing_assert(["a.output < 5"], pool) == "a.output < 5"
    assert first_failing_assert(["${a.output < 5}"], pool) == "${a.output < 5}"


# --- arithmetic in when:/asserts: ------------------------------------------- #


def test_when_arithmetic_basic_and_precedence():
    pool = _pool_with(a=3, b=2)
    assert evaluate_when("a.output + b.output == 5", pool) is True
    # precedence: * before +
    assert evaluate_when("a.output + b.output * 2 == 7", pool) is True
    # parens override
    assert evaluate_when("(a.output + b.output) * 2 == 10", pool) is True


def test_when_arithmetic_div_mod_unary():
    pool = _pool_with(n=6, d=4)
    assert evaluate_when("n.output / 3 == 2", pool) is True
    assert evaluate_when("d.output % 2 == 0", pool) is True
    assert evaluate_when("-n.output < 0", pool) is True
    # arithmetic on both sides of the comparison
    assert evaluate_when("n.output - 1 >= d.output + 1", pool) is True


def test_when_arithmetic_non_numeric_raises():
    pool = _pool_with(s="x")
    with pytest.raises(ExpressionError):
        evaluate_when("s.output + 1 == 2", pool)


def test_when_arithmetic_in_record_path():
    # the same grammar serves the strict-CASE record path
    from yian.expr.expressions import evaluate_when_record

    assert evaluate_when_record("a * b >= 10", {"a": 4, "b": 3}) is True
    assert evaluate_when_record("a % 2 == 1", {"a": 5}) is True


def test_when_string_equality_and_membership():
    # String equality and membership share the expression grammar with arithmetic.
    pool = _pool_with(label="approve")
    assert evaluate_when("label.output == 'approve'", pool) is True
    assert evaluate_when("label.output in 'approve/reject'", pool) is True


# --- list literals in when:/asserts: (the `in [...]` / `!= []` RHS) ---------- #


def test_when_list_literal_membership_and_equality():
    # seeds 05/10 use list literals on the RHS of `in` / `!=` / `==`.
    pool = _pool_with(style="relevance", bundle=["ACME", "BETA"], empty=[])
    assert evaluate_when('style.output in ["relevance", "value"]', pool) is True
    assert evaluate_when('style.output not in ["value", "quality"]', pool) is True
    assert evaluate_when("bundle.output != []", pool) is True
    assert evaluate_when("empty.output == []", pool) is True
    # numeric + nested-arithmetic elements also resolve as values
    assert evaluate_when("2 in [1, 2, 3]", pool) is True
    assert evaluate_when("5 in [2 + 3, 10]", pool) is True


def test_condition_context_treats_missing_reference_as_falsy():
    # Template rendering is strict, while condition evaluation treats a miss as falsy.
    pool = VariablePool()
    assert evaluate_when("ghost.output > 5", pool) is False


def test_record_condition_compares_a_miss_as_null():
    # A miss is None compared as a plain value: `==` is false and `!=` is true.
    assert evaluate_when_record('x == "done"', {}) is False
    assert evaluate_when_record('x != "done"', {}) is True
    assert evaluate_when_record('x != "done"', {"x": None}) is True
    assert evaluate_when_record("${x}", {}) is False
    # A None left-hand side of `in` is just a value that isn't in the container; only a
    # None right-hand side (the container itself) raises.
    assert evaluate_when_record('x in ["a"]', {}) is False
    with pytest.raises(ExpressionError):
        evaluate_when_record('"a" in ${x}', {})


def test_builtins_evaluate_in_conditions_and_bindings():
    from yian.expr.template import eval_binding

    assert evaluate_when_record('${upper(s) == "AB"}', {"s": "ab"}) is True
    assert eval_binding('${join(items, ",")}', {"items": ["a", "b"]}.get) == "a,b"


# --- record-scoped strict prompt rendering ---------------------------------- #


def test_render_record_substitutes_declared_inputs():
    from yian.expr.template import render_template_record

    rec = {"topic": "ZETA", "rating": {"value": 0.8}}
    assert render_template_record("Assess ${topic} v=${rating.value}", rec) == "Assess ZETA v=0.8"


def test_render_record_raises_on_unknown_or_none():
    from yian.expr.template import render_template_record

    with pytest.raises(ExpressionError):
        render_template_record("hi ${ghost}", {"topic": "X"})
    with pytest.raises(ExpressionError):
        render_template_record("hi ${t}", {"t": None})  # present-but-None -> raise (strict floor)


def test_render_record_no_refs_identity():
    from yian.expr.template import render_template_record

    assert render_template_record("plain", {}) == "plain"


def test_render_record_falsy_present_values_render():
    # 0 / "" / False are present (not missing) and must render, not raise.
    from yian.expr.template import render_template_record

    rec = {"w": 0, "s": {"on": False}, "name": ""}
    assert render_template_record("w=${w} flag=${s.on} n='${name}'", rec) == "w=0 flag=False n=''"


# --- record-scoped CASE condition evaluation -------------------------------- #


def test_evaluate_when_record_resolves_declared_inputs():
    from yian.expr.expressions import evaluate_when_record

    rec = {"label": "POSITIVE", "rating": {"value": 0.8}}
    assert evaluate_when_record("label == 'POSITIVE'", rec) is True
    assert evaluate_when_record("rating.value > 0.7", rec) is True
    assert evaluate_when_record("label == 'NEGATIVE'", rec) is False


def test_evaluate_when_record_missing_is_falsy_not_raise():
    # A declared-but-None input or dotted miss is falsy rather than an error.
    from yian.expr.expressions import evaluate_when_record

    assert evaluate_when_record("label == 'POSITIVE'", {"label": None}) is False
    assert evaluate_when_record("rating.value > 0.7", {"rating": None}) is False


def test_evaluate_when_record_in_with_none_operand_raises():
    # Boundary: `in`/`not in` with a None operand on EITHER side RAISES (not falsy) —
    # `_eval_comparison` shares this behavior with the pool path. Membership with `None`
    # fails the node, while equality and ordered comparisons remain falsy.
    from yian.expr.expressions import evaluate_when_record

    with pytest.raises(ExpressionError):
        evaluate_when_record("'x' in ${opts}", {"opts": None})   # None RHS
    with pytest.raises(ExpressionError):
        evaluate_when_record("x in 'abc'", {})                # None LHS (missing input)
    with pytest.raises(ExpressionError):
        evaluate_when_record("${x} not in 'abc'", {})


# --- CASE node end-to-end through the engine ----------------------------- #


def _branch_graph(cases):
    # Strict CASE: the router declares `score` (bound from the upstream node) and
    # routes on the bare `${score}`. CaseNode gets its inputs via the base attribute
    # the compiler threads — here set post-construction (no compiler in this unit test).
    # `.output` is a SKIP token; the upstream FuncNode emits the scalar value
    # directly, and `${score.output}` reads `pool.store["score"]` unwrapped.
    log: list = []
    cond = CaseNode("cond", cases)
    stamp_reads(cond, {"score": "${score.output}"})
    nodes = [
        FuncNode("score", lambda p: 0.8),
        cond,
        RecordNode("high", log),
        RecordNode("low", log),
    ]
    edges = [
        Edge("e0", START_ID, "score"),
        Edge("e1", "score", "cond"),
        Edge("e2", "cond", "high", "is_high"),
        Edge("e3", "cond", "low", ELSE_HANDLE),
        Edge("e4", "high", END_ID),
        Edge("e5", "low", END_ID),
    ]
    node_map = {n.id: n for n in nodes}
    return CompiledFlow.from_parts(_with_boundary(node_map), edges, wiring=derive_wiring(node_map)), log


def test_case_when_routes_high():
    g, log = _branch_graph([Case(handle="is_high", when="score >= 0.7")])
    list(_engine(g).run())
    assert log == ["high"]


def test_case_when_routes_default():
    g, log = _branch_graph([Case(handle="is_high", when="score >= 0.9")])
    list(_engine(g).run())
    assert log == ["low"]


def test_case_string_equality_routes():
    # The canonical pattern: an upstream node writes a label; the router declares it as
    # an input and does a pure string comparison on the bare `${label}`.
    log: list = []
    cond = CaseNode("cond", [Case(handle="is_high", when="label == 'POSITIVE'")])
    stamp_reads(cond, {"label": "${label.output}"})
    nodes = [
        FuncNode("label", lambda p: "POSITIVE"),
        cond,
        RecordNode("high", log),
        RecordNode("low", log),
    ]
    edges = [
        Edge("e0", START_ID, "label"),
        Edge("e1", "label", "cond"),
        Edge("e2", "cond", "high", "is_high"),
        Edge("e3", "cond", "low", ELSE_HANDLE),
        Edge("e4", "high", END_ID),
        Edge("e5", "low", END_ID),
    ]
    node_map = {n.id: n for n in nodes}
    g = CompiledFlow.from_parts(_with_boundary(node_map), edges, wiring=derive_wiring(node_map))
    list(_engine(g).run())
    assert log == ["high"]


def test_case_case_without_when_errors():
    g = _branch_graph([Case(handle="is_high")])[0]  # no `when`
    # A validated compiled CASE always has a string predicate. Missing one is
    # malformed setup, not expected node execution rejection.
    with pytest.raises(ValueError, match="has no `when` expression"):
        list(_engine(g).run())
