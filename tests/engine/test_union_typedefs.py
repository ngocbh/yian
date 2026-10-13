"""`Union[...]` of records in the type grammar and in `typedefs:`."""

import pytest

from yian.compose.helpers.errors import LoadError
from yian.compose.loader import load_flow
from yian.compose.source import SourceDocument
from yian.typesys.types import (
    OptionalExpr,
    RefExpr,
    UnionExpr,
    parse_type,
    read_typedefs,
    resolve_type,
    type_for,
)
from yian.typesys.values import (
    Type,
    TypeCheckError,
    ValueKind,
    canonical_type_label,
    canonical_union_order,
    union_tag_fields,
)


TRADES = {
    "Buy": {"action": "Literal[buy]", "qty": "int"},
    "Sell": {"action": "Literal[sell]", "price": "float"},
    "Hold": {"action": "Literal[hold]"},
}


def _registry(**extra):
    return read_typedefs({**TRADES, **extra})


def _tags(members):
    return [next(iter(member.fields["action"].tags)) for member in members]


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #


def test_parse_union():
    assert parse_type("Union[Buy, Sell]") == UnionExpr((RefExpr("Buy"), RefExpr("Sell")))


def test_parse_union_inside_optional():
    assert parse_type("Optional[Union[Buy, Sell]]") == OptionalExpr(
        UnionExpr((RefExpr("Buy"), RefExpr("Sell")))
    )


@pytest.mark.parametrize("text", ["Buy | Sell", "Optional[Buy | Sell]"])
def test_bar_operator_hints_union(text):
    with pytest.raises(TypeCheckError) as caught:
        parse_type(text)
    assert "use Union[Buy, Sell] (| is not a type operator)" in str(caught.value)


def test_bar_operator_chain_hints_all_members():
    with pytest.raises(TypeCheckError) as caught:
        parse_type("Buy | Sell | Hold")
    assert "use Union[Buy, Sell, Hold]" in str(caught.value)


def test_none_member_hints_optional():
    with pytest.raises(TypeCheckError) as caught:
        parse_type("Union[Buy, None]")
    assert "use Optional[Union[...]]" in str(caught.value)


@pytest.mark.parametrize("text", ["Union[]", "Union[()]"])
def test_empty_union_is_malformed(text):
    with pytest.raises(TypeCheckError) as caught:
        parse_type(text)
    assert f"malformed type expression {text!r}" in str(caught.value)


# --------------------------------------------------------------------------- #
# Resolution
# --------------------------------------------------------------------------- #


def test_typedef_union_orders_members_by_tag():
    registry = _registry(Decision="Union[Buy, Sell, Hold]")
    decision = type_for("Decision", registry)
    assert decision.kind is ValueKind.OBJECT
    assert decision.nullable is False
    assert _tags(decision.variants) == ["buy", "hold", "sell"]
    assert decision.variants[0] == type_for("Buy", registry)
    assert union_tag_fields(decision) == ("action",)


def test_union_resolves_in_every_position():
    registry = _registry(Decision="Union[Buy, Sell, Hold]", Order={"decision": "Decision"})
    decision = type_for("Decision", registry)

    optional = type_for("Optional[Decision]", registry)
    assert optional.nullable is True
    assert optional.variants == decision.variants

    listed = type_for("list[Decision]", registry)
    assert listed.kind is ValueKind.LIST_OBJECT
    assert listed.element == decision

    order = type_for("Order", registry)
    assert order.fields["decision"] == decision
    assert order.required == frozenset({"decision"})

    optional_field = type_for("Optional[Union[Buy, Sell]]", registry)
    assert optional_field.nullable is True

    inline = type_for("Union[Buy, Sell]", registry)
    assert _tags(inline.variants) == ["buy", "sell"]


def test_optional_union_field_is_not_required():
    registry = _registry(Order={"decision": "Optional[Union[Buy, Sell]]", "id": "str"})
    assert type_for("Order", registry).required == frozenset({"id"})


def test_nested_unions_and_aliases_flatten_without_duplicates():
    registry = _registry(
        Purchase="Buy",
        Trade="Union[Buy, Sell]",
        Decision="Union[Trade, Purchase, Hold, Union[Sell, Hold]]",
    )
    decision = type_for("Decision", registry)
    assert _tags(decision.variants) == ["buy", "hold", "sell"]
    assert decision == type_for("Union[Buy, Sell, Hold]", registry)


def test_single_member_union_is_the_member():
    registry = _registry()
    assert type_for("Union[Buy]", registry) == type_for("Buy", registry)
    assert type_for("Union[Buy, Buy]", registry) == type_for("Buy", registry)


def test_authored_order_does_not_affect_equality():
    registry = _registry()
    assert type_for("Union[Buy, Sell, Hold]", registry) == type_for(
        "Union[Hold, Sell, Buy]", registry
    )


def test_untagged_disjoint_union_resolves():
    registry = read_typedefs({
        "Text": {"text": "str"},
        "Call": {"name": "str", "input": "object"},
        "Reply": "Union[Text, Call]",
    })
    reply = type_for("Reply", registry)
    assert union_tag_fields(reply) == ()
    members = (type_for("Text", registry), type_for("Call", registry))
    assert reply.variants == canonical_union_order(members)
    assert set(map(canonical_type_label, reply.variants)) == set(map(canonical_type_label, members))


def test_resolve_type_accepts_union_expr_directly():
    registry = _registry()
    union = resolve_type(UnionExpr((RefExpr("Sell"), RefExpr("Buy"))), registry)
    assert isinstance(union, Type)
    assert _tags(union.variants) == ["buy", "sell"]


def test_canonical_label_uses_canonical_order():
    registry = _registry()
    union = type_for("Union[Sell, Buy]", registry)
    buy = canonical_type_label(type_for("Buy", registry))
    sell = canonical_type_label(type_for("Sell", registry))
    assert canonical_type_label(union) == f"Union[{buy},{sell}]"
    assert canonical_type_label(type_for("Optional[Union[Sell, Buy]]", registry)) == (
        f"Optional[Union[{buy},{sell}]]"
    )


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


def _resolve_error(text, **typedefs):
    registry = read_typedefs(typedefs) if typedefs else _registry()
    with pytest.raises(TypeCheckError) as caught:
        type_for(text, registry)
    return str(caught.value)


def test_member_must_be_a_record():
    registry = _registry(Ticker="str")
    with pytest.raises(TypeCheckError) as caught:
        type_for("Union[Buy, Ticker]", registry)
    assert "Union[Buy, Ticker]: member 'Ticker' is not a record" in str(caught.value)


@pytest.mark.parametrize("member", ["str", "list[Buy]", "object", "Literal[a, b]"])
def test_non_record_members_rejected(member):
    message = _resolve_error(f"Union[Buy, {member}]")
    assert f"member '{member}' is not a record" in message


def test_nullable_member_rejected():
    registry = _registry(MaybeSell="Optional[Sell]")
    with pytest.raises(TypeCheckError) as caught:
        type_for("Union[Buy, MaybeSell]", registry)
    assert "member 'MaybeSell' is nullable" in str(caught.value)
    assert "Optional[Union[...]]" in str(caught.value)


def test_overlapping_members_rejected():
    registry = read_typedefs({"Bull": {"conviction": "float"}, "Bear": {"conviction": "float"}})
    with pytest.raises(TypeCheckError) as caught:
        type_for("Union[Bull, Bear]", registry)
    assert (
        "Union[Bull, Bear]: members 'Bull' and 'Bear' can match the same value; add a tag "
        "field every member declares as a single-value Literal, with a different value in "
        "each member, e.g. kind: Literal[bull]"
    ) in str(caught.value)


def test_overlapping_shapes_that_differ_rejected():
    registry = read_typedefs({
        "Bull": {"conviction": "float"},
        "Bear": {"conviction": "float", "note": "Optional[str]"},
    })
    with pytest.raises(TypeCheckError) as caught:
        type_for("Union[Bull, Bear]", registry)
    assert "members 'Bull' and 'Bear' can match the same value" in str(caught.value)


def test_optional_literal_is_not_a_tag():
    registry = read_typedefs({
        "Bull": {"side": "Optional[Literal[bull]]", "conviction": "float"},
        "Bear": {"side": "Optional[Literal[bear]]", "conviction": "float"},
    })
    with pytest.raises(TypeCheckError) as caught:
        type_for("Union[Bull, Bear]", registry)
    assert "members 'Bull' and 'Bear' can match the same value" in str(caught.value)


def test_unknown_member():
    assert "unknown type 'Nope'" in _resolve_error("Union[Buy, Nope]")


def test_recursive_member_rejected():
    with pytest.raises(TypeCheckError) as caught:
        read_typedefs({
            "Leaf": {"kind": "Literal[leaf]"},
            "Node": {"kind": "Literal[node]", "child": "Union[Node, Leaf]"},
        })
    assert "typedef 'Node' field 'child': recursive type reference" in str(caught.value)


# --------------------------------------------------------------------------- #
# Eager checks in typedefs
# --------------------------------------------------------------------------- #


def test_unused_overlapping_union_fails_in_read_typedefs():
    with pytest.raises(TypeCheckError) as caught:
        read_typedefs({
            "Bull": {"conviction": "float"},
            "Bear": {"conviction": "float"},
            "Mood": "Union[Bull, Bear]",
        })
    assert "typedef 'Mood'" in str(caught.value)
    assert "can match the same value" in str(caught.value)


def test_unused_record_with_bad_union_field_fails_in_read_typedefs():
    with pytest.raises(TypeCheckError) as caught:
        read_typedefs({**TRADES, "Ticker": "str", "Order": {"pick": "Union[Buy, Ticker]"}})
    assert "typedef 'Order' field 'pick': Union[Buy, Ticker]" in str(caught.value)
    assert "member 'Ticker' is not a record" in str(caught.value)


def test_malformed_union_field_is_prefixed():
    with pytest.raises(TypeCheckError) as caught:
        read_typedefs({**TRADES, "Order": {"pick": "Union[Buy, Sell"}})
    assert "typedef 'Order' field 'pick': malformed type expression" in str(caught.value)




def test_alias_cycle_through_union_rejected():
    with pytest.raises(TypeCheckError) as caught:
        read_typedefs({**TRADES, "Decision": "Union[Buy, Again]", "Again": "Decision"})
    assert "cycle" in str(caught.value)


def test_self_referencing_union_alias_rejected():
    with pytest.raises(TypeCheckError) as caught:
        read_typedefs({**TRADES, "Decision": "Union[Buy, Decision]"})
    assert "cycle" in str(caught.value)


_BAD_UNION_FLOW = """\
id: f
name: f
input: {x: str}
typedefs:
  Bull: {conviction: float}
  Bear: {conviction: float}
  Mood: Union[Bull, Bear]
nodes:
  a: {kind: code, code: tests.seeds.fns:one_line_summary, input: {x: "${input.x}"}}
output: {y: "${a.output}"}
"""


def test_load_flow_locates_bad_union_at_its_typedef():
    typedefs = SourceDocument.parse(_BAD_UNION_FLOW).root.child("typedefs")
    expected = typedefs.key_range_of("Mood")
    with pytest.raises(LoadError) as caught:
        load_flow(_BAD_UNION_FLOW)
    assert "bad typedefs: typedef 'Mood'" in str(caught.value)
    assert "Union[Bull, Bear]: members 'Bull' and 'Bear' can match the same value" in str(
        caught.value
    )
    frames = caught.value.diagnostic.frames
    assert len(frames) == 1
    assert frames[0].range == expected


def test_load_flow_accepts_union_typedef():
    text = (
        "id: f\nname: f\ninput: {x: str}\n"
        "typedefs:\n"
        "  Buy: {action: 'Literal[buy]', qty: int}\n"
        "  Sell: {action: 'Literal[sell]', price: float}\n"
        "  Decision: Union[Buy, Sell]\n"
        "nodes:\n"
        "  a: {kind: code, code: tests.seeds.fns:one_line_summary, input: {x: \"${input.x}\"}}\n"
        "output: {y: \"${a.output}\"}\n"
    )
    load_flow(text)
