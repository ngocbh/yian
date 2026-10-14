"""Tests for the one spelling every message uses to print a `Literal` type."""

import pytest

from yian import LoadError, load_flow
from yian.typesys.types import EnumExpr, format_type_expr, parse_type, read_typedefs, type_for
from yian.typesys.values import Type, TypeCheckError, ValueKind, _labels, canonical_type_label


def _lit(*tags):
    return Type(kind=ValueKind.STRING, tags=frozenset(tags))


@pytest.mark.parametrize(
    "tags,label",
    [
        (("no_go", "go"), "Literal(go, no_go)"),
        (("a b", "z", "é"), 'Literal("a b", z, é)'),
        (("True", "x-y", "1a", 'say "hi"'), 'Literal("1a", "True", "say \\"hi\\"", "x-y")'),
        (("a,b",), 'Literal("a,b")'),
    ],
)
def test_canonical_label_spells_literal_in_the_parenthesized_form(tags, label):
    assert canonical_type_label(_lit(*tags)) == label
    assert type_for(label, {}) == _lit(*tags)  # the label reads back


def test_both_renderers_share_one_spelling():
    from yian.typesys.values import literal_label

    assert (
        format_type_expr(EnumExpr(("go", "no_go")))
        == literal_label(["go", "no_go"])
        == canonical_type_label(_lit("go", "no_go"))
        == "Literal(go, no_go)"
    )
    assert format_type_expr(parse_type("Literal(b, a)")) == "Literal(b, a)"  # authored order kept


def test_nested_literal_labels():
    assert (
        canonical_type_label(type_for("Optional(list(Literal(go, no_go)))", {}))
        == "Optional(list(Literal(go, no_go)))"
    )


def test_tag_only_sequence_hint_spells_a_parseable_literal():
    with pytest.raises(TypeCheckError) as caught:
        read_typedefs({"Mode": ["a b", "True"]})
    assert 'e.g. Mode: Literal("a b", "True")' in str(caught.value)
    assert type_for('Literal("a b", "True")', {}) == _lit("a b", "True")


def test_union_overlap_hint_spells_a_parseable_literal():
    with pytest.raises(TypeCheckError) as caught:
        read_typedefs({"Buy": {"qty": "int"}, "Sell": {"qty": "int"}, "Order": "Union(Buy, Sell)"})
    assert str(caught.value).endswith("e.g. kind: Literal(buy)")
    assert type_for("Literal(buy)", {}) == _lit("buy")


def test_order_key_keeps_the_bracket_spelling():
    assert _labels(_lit("no_go", "go"))[1] == 'Literal["go","no_go"]'


def test_case_message_prints_the_parenthesized_literal():
    flow = """id: f
name: f
input:
  s: !Verdict
typedefs:
  Verdict: Literal(go, no_go)
nodes:
  route:
    kind: case
    on: ${input.s}
    cases:
      - when: nogo
        then: a
    else: a
  a:
    kind: code
    code: return 1
output: ${a.output}
"""
    with pytest.raises(LoadError) as caught:
        load_flow(flow)
    assert "(`Literal(go, no_go)`); use one of: go, no_go" in str(caught.value)
