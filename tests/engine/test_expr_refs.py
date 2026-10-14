"""Tests for the ONE unified compile-time ref-walk (`expr.expressions.expr_refs`).

`expr_refs(parse_expr(text))` collects every reference-leaf PATH a unified
`${...}` expression reads, in source order (no dedupe). This is the single walk
behind the binding ref-collection (`expr_refs_of`), the prompt ref-collection
(`prompt_refs`), and the condition ref-collection.

A `refcall` WITHOUT a `call_suffix` is a reference (its dotted path is collected).
A `refcall` WITH a `call_suffix` is a builtin call: the callee name contributes
NOTHING, but each arg VALUE's refs ARE collected. Literals contribute nothing.
`item`-headed refs ARE collected here — the caller (compose `_ref_producer`) is
responsible for skipping `item` when minting edges.
"""

from yian.expr.expressions import expr_refs
from yian.expr.grammar import parse_expr


def _refs(text: str) -> list[str]:
    return expr_refs(parse_expr(text))


def test_two_heads_in_arithmetic():
    assert _refs("node_a.x + node_b.y") == ["node_a.x", "node_b.y"]


def test_item_headed_ref_is_collected():
    # `item`-headed refs ARE collected here (the `call(...)` `${item}` guard needs
    # to see them); the CALLER skips `item` when minting edges.
    assert _refs("item.x") == ["item.x"]


def test_hash_and_slash_segments_preserved():
    assert _refs("node#0.output") == ["node#0.output"]
    assert _refs("def/child.output") == ["def/child.output"]


def test_builtin_call_collects_arg_refs_not_callee_or_literal():
    # `join(items, ", ")`: the ref arg `items` is collected; the string literal and
    # the callee `join` contribute nothing.
    assert _refs('join(items, ", ")') == ["items"]


def test_default_collects_head_and_default_ref():
    # `a :- b`: both the head `a` and the default ref `b`.
    assert _refs("a :- b") == ["a", "b"]


def test_coalesce_collects_all_operands():
    assert _refs("x | y | z") == ["x", "y", "z"]


def test_required_collects_head():
    # `p :? "msg"`: the head `p` (the string message contributes nothing).
    assert _refs('p :? "msg"') == ["p"]


def test_call_with_only_literal_args_has_no_refs():
    # `upper("hi")`: callee + literal arg contribute nothing (no zero-arg builtin exists).
    assert _refs('upper("hi")') == []


def test_ref_inside_list_literal():
    assert _refs("[a, b]") == ["a", "b"]


def test_ref_inside_parens():
    assert _refs("(a + b)") == ["a", "b"]


def test_nested_call_collects_inner_ref():
    assert _refs("join(upper(x))") == ["x"]
