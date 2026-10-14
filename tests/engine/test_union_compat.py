"""Load-time compatibility of `Union(...)` types in `types_compatible`."""

from yian.compile.validation import types_compatible
from yian.typesys.types import read_typedefs, type_for
from yian.typesys.values import Type, ValueKind


REGISTRY = read_typedefs(
    {
        "Buy": {"action": "Literal(buy)", "qty": "int"},
        "Sell": {"action": "Literal(sell)", "price": "float"},
        "Hold": {"action": "Literal(hold)"},
        "BuyText": {"action": "Literal(buy)", "qty": "str"},
        "Either": {"action": "Literal(buy, sell)", "qty": "int", "price": "float"},
        "Note": {"text": "str"},
        "Count": {"n": "int"},
        "Score": {"n": "float"},
        "Mark": {"flag": "bool"},
        "SmallBuy": {"action": "Literal(buy)", "qty": "int(0..10)"},
        "LooseBuy": {"action": "str", "qty": "int"},
        "Order": {"trade": "Union(Buy, Sell)"},
        "WideOrder": {"trade": "Union(Buy, Sell, Hold)"},
    }
)


def _t(text):
    return type_for(text, REGISTRY)


def test_union_fits_wider_union():
    assert types_compatible(_t("Union(Buy, Sell)"), _t("Union(Buy, Sell, Hold)"))


def test_union_does_not_fit_narrower_union():
    assert not types_compatible(_t("Union(Buy, Sell, Hold)"), _t("Union(Buy, Sell)"))


def test_same_tags_with_incompatible_member_does_not_fit():
    assert not types_compatible(_t("Union(BuyText, Sell)"), _t("Union(Buy, Sell)"))


def test_member_record_fits_union():
    assert types_compatible(_t("Buy"), _t("Union(Buy, Sell)"))


def test_record_with_multi_tag_literal_does_not_fit_union():
    assert not types_compatible(_t("Either"), _t("Union(Buy, Sell)"))


def test_union_does_not_fit_member_record():
    assert not types_compatible(_t("Union(Buy, Sell)"), _t("Buy"))


def test_union_fits_object():
    assert types_compatible(_t("Union(Buy, Sell)"), Type.scalar(ValueKind.OBJECT))


def test_object_does_not_fit_union():
    assert not types_compatible(Type.scalar(ValueKind.OBJECT), _t("Union(Buy, Sell)"))


def test_nullable_union_does_not_fit_strict_union():
    assert not types_compatible(_t("Optional(Union(Buy, Sell))"), _t("Union(Buy, Sell)"))


def test_nullable_union_fits_nullable_union():
    assert types_compatible(
        _t("Optional(Union(Buy, Sell))"), _t("Optional(Union(Buy, Sell, Hold))")
    )


def test_nullable_member_record_fits_nullable_union():
    assert types_compatible(_t("Optional(Buy)"), _t("Optional(Union(Buy, Sell))"))


def test_nullable_member_record_does_not_fit_strict_union():
    assert not types_compatible(_t("Optional(Buy)"), _t("Union(Buy, Sell)"))


def test_none_fits_optional_union():
    assert types_compatible(Type.scalar(ValueKind.NONE), _t("Optional(Union(Buy, Sell))"))


def test_none_does_not_fit_strict_union():
    assert not types_compatible(Type.scalar(ValueKind.NONE), _t("Union(Buy, Sell)"))


def test_scalar_does_not_fit_union():
    assert not types_compatible(Type.scalar(ValueKind.STRING), _t("Union(Buy, Sell)"))


def test_list_of_unions_follows_members():
    assert types_compatible(_t("list(Union(Buy, Sell))"), _t("list(Union(Buy, Sell, Hold))"))
    assert not types_compatible(_t("list(Union(Buy, Sell, Hold))"), _t("list(Union(Buy, Sell))"))


def test_list_of_members_fits_list_of_unions():
    assert types_compatible(_t("list(Buy)"), _t("list(Union(Buy, Sell))"))


def test_untagged_union_fits_wider_untagged_union():
    assert types_compatible(_t("Union(Note, Count)"), _t("Union(Note, Count, Mark)"))
    assert not types_compatible(_t("Union(Note, Count, Mark)"), _t("Union(Note, Count)"))


def test_untagged_union_member_fits_by_widening():
    assert types_compatible(_t("Union(Note, Count)"), _t("Union(Note, Score)"))
    assert not types_compatible(_t("Union(Note, Score)"), _t("Union(Note, Count)"))


def test_untagged_member_record_fits_union():
    assert types_compatible(_t("Count"), _t("Union(Note, Count)"))
    assert not types_compatible(_t("Mark"), _t("Union(Note, Count)"))


def test_member_record_does_not_fit_bounded_member():
    assert not types_compatible(_t("Buy"), _t("Union(SmallBuy, Sell)"))


def test_strict_union_fits_wider_optional_union():
    assert types_compatible(_t("Union(Buy, Sell)"), _t("Optional(Union(Buy, Sell, Hold))"))


def test_list_of_optional_unions_does_not_fit_list_of_unions():
    assert not types_compatible(
        _t("list(Optional(Union(Buy, Sell)))"), _t("list(Union(Buy, Sell))")
    )


def test_union_field_follows_members():
    assert types_compatible(_t("Order"), _t("WideOrder"))
    assert not types_compatible(_t("WideOrder"), _t("Order"))


def test_plain_str_tag_does_not_fit_union():
    assert not types_compatible(_t("LooseBuy"), _t("Union(Buy, Sell)"))
