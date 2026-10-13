"""Union refinement on `Type`: shape, tag fields, overlap, value checking, helpers."""

from __future__ import annotations

import time
from dataclasses import replace

import pytest

from yian.typesys import (
    FieldAbsent,
    FieldPartial,
    FieldRead,
    TypeCheckError,
    canonical_type_label,
    canonical_union_order,
    restrict_union,
    types_overlap,
    union_field,
    union_member_for_tag,
    union_tag_fields,
)
from yian.typesys.values import Type, ValueKind, build_value_as

STR = Type.scalar(ValueKind.STRING)
INT = Type.scalar(ValueKind.INTEGER)
FLOAT = Type.scalar(ValueKind.NUMBER)
BOOL = Type.scalar(ValueKind.BOOLEAN)
DATE = Type.scalar(ValueKind.DATE)
DATETIME = Type.scalar(ValueKind.DATETIME)
OBJECT = Type.scalar(ValueKind.OBJECT)
NONE = Type.scalar(ValueKind.NONE)


def lit(*tags: str, nullable: bool = False) -> Type:
    return Type(kind=ValueKind.STRING, tags=frozenset(tags), nullable=nullable)


def rec(**fields: Type) -> Type:
    """A closed record whose non-nullable fields are required."""
    return Type(
        kind=ValueKind.OBJECT,
        fields=dict(fields),
        required=frozenset(name for name, typ in fields.items() if not typ.nullable),
    )


def union(*members: Type, nullable: bool = False) -> Type:
    return Type(kind=ValueKind.OBJECT, variants=tuple(members), nullable=nullable)


buy = rec(action=lit("buy"), qty=INT)
sell = rec(action=lit("sell"), price=FLOAT)
hold = rec(action=lit("hold"))
U = union(buy, sell)

text = rec(text=STR)
call = rec(name=STR, input=OBJECT)
UNTAGGED = union(call, text)


# --------------------------------------------------------------------------- #
# validate_shape
# --------------------------------------------------------------------------- #


def test_union_shape_is_valid():
    U.validate_shape()
    UNTAGGED.validate_shape()
    union(buy, sell, nullable=True).validate_shape()


@pytest.mark.parametrize(
    ("typ", "error"),
    [
        (Type(kind=ValueKind.STRING, variants=(buy, sell)), ValueError),
        (
            Type(kind=ValueKind.OBJECT, fields={"a": INT}, variants=(buy, sell)),
            ValueError,
        ),
        (union(buy), ValueError),
        (union(buy, STR), ValueError),
        (union(buy, OBJECT), ValueError),
        (union(buy, replace(sell, nullable=True)), ValueError),
        (Type(kind=ValueKind.OBJECT, variants=[buy, sell]), TypeError),  # type: ignore[arg-type]
    ],
    ids=[
        "kind",
        "other-refinement",
        "one-member",
        "scalar-member",
        "object-member",
        "nullable-member",
        "not-a-tuple",
    ],
)
def test_union_shape_rejects(typ, error):
    with pytest.raises(error):
        typ.validate_shape()


def test_list_of_union_stores_as_list_object():
    Type(kind=ValueKind.LIST_OBJECT, element=U).validate_shape()
    with pytest.raises(ValueError, match="expected 'list\\[object\\]'"):
        Type(kind=ValueKind.LIST_STRING, element=U).validate_shape()


# --------------------------------------------------------------------------- #
# union_tag_fields
# --------------------------------------------------------------------------- #


def test_union_tag_fields_tagged():
    assert union_tag_fields(U) == ("action",)
    assert union_tag_fields(union(buy, sell, hold)) == ("action",)


def test_union_tag_fields_untagged_disjoint():
    assert union_tag_fields(UNTAGGED) == ()


def test_union_tag_fields_of_a_record_is_empty():
    assert union_tag_fields(buy) == ()


def test_union_tag_fields_excludes_optional_literal():
    first = rec(action=lit("buy", nullable=True), qty=INT)
    second = rec(action=lit("sell"), price=FLOAT)
    assert union_tag_fields(union(first, second)) == ()


def test_union_tag_fields_excludes_not_required_literal():
    first = Type(
        kind=ValueKind.OBJECT,
        fields={"action": lit("buy"), "qty": INT},
        required=frozenset({"qty"}),
    )
    assert union_tag_fields(union(first, sell)) == ()


def test_union_tag_fields_excludes_multi_value_literal():
    first = rec(action=lit("buy", "add"), qty=INT)
    assert union_tag_fields(union(first, sell)) == ()


def test_union_tag_fields_returns_every_qualifying_field_sorted():
    first = rec(side=lit("b"), action=lit("buy"), qty=INT)
    second = rec(side=lit("s"), action=lit("sell"), price=FLOAT)
    assert union_tag_fields(union(first, second)) == ("action", "side")


def test_union_tag_fields_excludes_equal_values():
    first = rec(kind=lit("trade"), action=lit("buy"), qty=INT)
    second = rec(kind=lit("trade"), action=lit("sell"), price=FLOAT)
    assert union_tag_fields(union(first, second)) == ("action",)


# --------------------------------------------------------------------------- #
# types_overlap
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        # nullable on both sides, with NONE counting as nullable
        (replace(STR, nullable=True), replace(INT, nullable=True), True),
        (NONE, replace(INT, nullable=True), True),
        (NONE, INT, False),
        # same scalar kind, or int/float
        (INT, INT, True),
        (BOOL, BOOL, True),
        (INT, FLOAT, True),
        (BOOL, INT, False),
        # str with date/datetime/Literal
        (STR, DATE, True),
        (DATETIME, STR, True),
        (STR, lit("buy"), True),
        # Literal with date/datetime
        (lit("2026-01-02"), DATE, True),
        (DATE, lit("buy"), False),
        (lit("2026-01-02T10:00:00"), DATETIME, True),
        (lit("2026-01-02"), DATETIME, False),
        # two Literals
        (lit("a", "b"), lit("b", "c"), True),
        (lit("a"), lit("b"), False),
        # numeric bounds
        (Type(ValueKind.INTEGER, bounds=(0, 5)), Type(ValueKind.INTEGER, bounds=(3, 9)), True),
        (Type(ValueKind.INTEGER, bounds=(0, 2)), Type(ValueKind.INTEGER, bounds=(3, 9)), False),
        (Type(ValueKind.INTEGER, bounds=(0, 2)), Type(ValueKind.NUMBER, bounds=(2.5, 3.0)), False),
        (Type(ValueKind.INTEGER, bounds=(0, 2)), FLOAT, True),
        # two lists
        (Type(ValueKind.LIST_STRING, element=STR), Type(ValueKind.LIST_INTEGER, element=INT), True),
        (Type.scalar(ValueKind.LIST_ANY), Type.scalar(ValueKind.LIST_OBJECT), True),
        # object with a record or union
        (OBJECT, buy, True),
        (U, OBJECT, True),
        # two records
        (buy, sell, False),
        (rec(a=INT), rec(a=INT, b=replace(STR, nullable=True)), True),
        (rec(a=INT), rec(b=INT), False),
        (rec(a=INT), rec(a=STR), False),
        (rec(a=INT), rec(a=FLOAT), True),
        (
            rec(a=INT),
            Type(kind=ValueKind.OBJECT, fields={"a": STR}, required=frozenset()),
            False,
        ),
        # a union against anything
        (U, buy, True),
        (hold, U, False),
        (U, union(hold, sell), True),
        # different kinds otherwise
        (STR, INT, False),
        (DATE, DATETIME, False),
        (buy, STR, False),
        (Type.scalar(ValueKind.LIST_ANY), OBJECT, False),
    ],
)
def test_types_overlap(a, b, expected):
    assert types_overlap(a, b) is expected
    assert types_overlap(b, a) is expected


# --------------------------------------------------------------------------- #
# build_value_as
# --------------------------------------------------------------------------- #


def test_tagged_union_accepts_each_member():
    assert build_value_as(U, {"action": "buy", "qty": 2}).to_object() == {
        "action": "buy",
        "qty": 2,
    }
    stored = build_value_as(U, {"action": "sell", "price": 1.5})
    assert stored.kind is ValueKind.OBJECT


def test_tagged_union_missing_tag():
    with pytest.raises(TypeCheckError) as caught:
        build_value_as(U, {"qty": 2})
    assert str(caught.value) == "missing tag field 'action' (expected one of ['buy', 'sell'])"
    assert caught.value.path == ("action",)


def test_tagged_union_unknown_tag():
    with pytest.raises(TypeCheckError) as caught:
        build_value_as(U, {"action": "hold"})
    assert str(caught.value) == (
        "'hold' is not a tag of the union (expected one of ['buy', 'sell'])"
    )
    assert caught.value.path == ("action",)


def test_tagged_union_non_string_tag_value_is_not_a_tag():
    with pytest.raises(TypeCheckError) as caught:
        build_value_as(U, {"action": ["buy"]})
    assert caught.value.path == ("action",)


def test_tagged_union_member_record_check():
    with pytest.raises(TypeCheckError) as caught:
        build_value_as(U, {"action": "sell", "price": 1.0, "qty": 2})
    assert "unknown fields: ['qty']" in str(caught.value)
    assert caught.value.path == ("qty",)

    with pytest.raises(TypeCheckError) as caught:
        build_value_as(U, {"action": "sell", "qty": 2})
    assert caught.value.path == ("price",)


def test_union_rejects_non_dict_at_root():
    with pytest.raises(TypeCheckError) as caught:
        build_value_as(U, "buy")
    assert caught.value.path == ()
    with pytest.raises(TypeCheckError) as caught:
        build_value_as(UNTAGGED, 3)
    assert caught.value.path == ()


def test_optional_union_accepts_none():
    assert build_value_as(replace(U, nullable=True), None).to_object() is None
    with pytest.raises(TypeCheckError):
        build_value_as(U, None)


def test_untagged_union_picks_the_matching_member():
    assert build_value_as(UNTAGGED, {"text": "hi"}).to_object() == {"text": "hi"}
    assert build_value_as(UNTAGGED, {"name": "f", "input": {}}).to_object() == {
        "name": "f",
        "input": {},
    }


def test_untagged_union_matching_no_member():
    with pytest.raises(TypeCheckError) as caught:
        build_value_as(UNTAGGED, {"text": 3})
    call_label = canonical_type_label(call)
    text_label = canonical_type_label(text)
    assert str(caught.value) == (
        f"value matches no member of {canonical_type_label(UNTAGGED)}: "
        f"{call_label}: record missing required fields: ['input', 'name']; "
        f"{text_label}: 3 does not match declared type str"
    )
    assert caught.value.path == ()


def test_untagged_no_match_message_does_not_nest_member_messages():
    inner = union(rec(a=STR), rec(b=INT))
    outer = union(rec(wrap=inner), rec(c=BOOL))
    with pytest.raises(TypeCheckError) as caught:
        build_value_as(outer, {"wrap": {"a": 1}})
    wrap_label = canonical_type_label(rec(wrap=inner))
    c_label = canonical_type_label(rec(c=BOOL))
    assert str(caught.value) == (
        f"value matches no member of {canonical_type_label(outer)}: "
        f"{wrap_label}: value matches no member of {canonical_type_label(inner)}; "
        f"{c_label}: record missing required fields: ['c']"
    )
    assert caught.value.path == ()


def _deep_untagged_union(depth: int) -> Type:
    typ = union(rec(a=STR), rec(b=INT))
    for level in range(depth):
        typ = union(rec(inner=typ), rec(**{f"leaf{level}": INT}))
    return typ


def test_deep_untagged_union_labels_and_checks_in_linear_work():
    typ = _deep_untagged_union(30)
    started = time.perf_counter()
    label = canonical_type_label(typ)
    with pytest.raises(TypeCheckError, match="^value matches no member of "):
        build_value_as(typ, {"inner": 3})
    assert time.perf_counter() - started < 2.0
    assert label.count("Union[") == 31


def test_list_of_union():
    typ = Type(kind=ValueKind.LIST_OBJECT, element=U)
    stored = build_value_as(
        typ, [{"action": "buy", "qty": 1}, {"action": "sell", "price": 2.0}]
    )
    assert stored.kind is ValueKind.LIST_OBJECT
    with pytest.raises(TypeCheckError) as caught:
        build_value_as(typ, [{"action": "buy", "qty": 1}, {"action": "buy", "qty": "x"}])
    assert caught.value.path == (1, "qty")


def test_union_label_lists_members_in_canonical_order():
    assert canonical_type_label(union(sell, buy)) == (
        f"Union[{canonical_type_label(buy)},{canonical_type_label(sell)}]"
    )
    assert canonical_type_label(replace(U, nullable=True)).startswith("Optional[Union[")


# --------------------------------------------------------------------------- #
# union_member_for_tag / restrict_union / canonical_union_order
# --------------------------------------------------------------------------- #


def test_union_member_for_tag():
    assert union_member_for_tag(U, "buy") == buy
    assert union_member_for_tag(U, "sell") == sell
    assert union_member_for_tag(U, "hold") is None


def test_union_member_for_tag_on_another_tag_field():
    first = rec(side=lit("b"), action=lit("buy"), qty=INT)
    second = rec(side=lit("s"), action=lit("sell"), price=FLOAT)
    both = union(first, second)
    assert union_member_for_tag(both, "s", field="side") == second
    with pytest.raises(ValueError):
        union_member_for_tag(both, "buy", field="qty")


def test_restrict_union_on_another_tag_field():
    first = rec(side=lit("b"), action=lit("buy"), qty=INT)
    second = rec(side=lit("s"), action=lit("sell"), price=FLOAT)
    third = rec(side=lit("h"), action=lit("hold"))
    three = union(first, second, third)
    assert restrict_union(three, {"s"}, field="side") == second
    assert restrict_union(three, {"s", "b"}, field="side") == union(first, second)
    with pytest.raises(ValueError):
        restrict_union(three, {"sell"}, field="side")


def test_restrict_union_returns_members_in_canonical_order():
    assert restrict_union(union(sell, hold, buy), {"sell", "buy"}) == union(buy, sell)


def test_restrict_union_rejects_a_bare_string():
    with pytest.raises(TypeError):
        restrict_union(U, "buy")


def test_union_member_for_tag_needs_a_tagged_union():
    with pytest.raises(ValueError):
        union_member_for_tag(UNTAGGED, "text")


def test_restrict_union_to_one_tag_gives_the_member():
    assert restrict_union(union(buy, sell, hold), {"buy"}) == buy
    assert restrict_union(union(buy, sell, nullable=True), {"buy"}) == replace(
        buy, nullable=True
    )


def test_restrict_union_to_two_tags_keeps_nullable():
    three = union(buy, sell, hold, nullable=True)
    assert restrict_union(three, {"sell", "buy"}) == union(buy, sell, nullable=True)
    assert restrict_union(union(buy, sell, hold), {"buy", "sell"}) == U


def test_restrict_union_by_member():
    assert restrict_union(UNTAGGED, [text]) == text


def test_restrict_union_rejects_empty_and_unknown():
    with pytest.raises(ValueError):
        restrict_union(U, set())
    with pytest.raises(ValueError):
        restrict_union(U, {"hold"})
    with pytest.raises(ValueError):
        restrict_union(UNTAGGED, [hold])


def test_canonical_union_order():
    assert canonical_union_order((sell, hold, buy)) == (buy, hold, sell)
    labels = sorted([canonical_type_label(text), canonical_type_label(call)])
    ordered = canonical_union_order((text, call))
    assert [canonical_type_label(member) for member in ordered] == labels


# --------------------------------------------------------------------------- #
# union_field
# --------------------------------------------------------------------------- #


def test_union_field_shared_tag_reads_merged_literal():
    assert union_field(U, "action") == FieldRead(lit("buy", "sell"))


def test_union_field_equal_types_read_that_type():
    first = rec(action=lit("buy"), note=STR)
    second = rec(action=lit("sell"), note=STR)
    assert union_field(union(first, second), "note") == FieldRead(STR)


def test_union_field_equality_ignores_nullable_and_nullability_spreads():
    first = rec(action=lit("buy"), note=STR)
    second = rec(action=lit("sell"), note=replace(STR, nullable=True))
    assert union_field(union(first, second), "note") == FieldRead(
        replace(STR, nullable=True)
    )


def test_union_field_not_required_is_nullable():
    first = Type(
        kind=ValueKind.OBJECT,
        fields={"action": lit("buy"), "note": STR},
        required=frozenset({"action"}),
    )
    second = rec(action=lit("sell"), note=STR)
    assert union_field(union(first, second), "note") == FieldRead(
        replace(STR, nullable=True)
    )


def test_union_field_merged_literal_is_nullable_if_any_member_is():
    first = rec(action=lit("buy"), mode=lit("a"))
    second = rec(action=lit("sell"), mode=lit("b", nullable=True))
    assert union_field(union(first, second), "mode") == FieldRead(
        lit("a", "b", nullable=True)
    )


def test_union_field_differing_types_read_untyped():
    first = rec(action=lit("buy"), size=INT)
    second = rec(action=lit("sell"), size=STR)
    assert union_field(union(first, second), "size") == FieldRead(None)


def test_union_field_partial():
    result = union_field(union(buy, sell, hold), "qty")
    assert result == FieldPartial((buy,))
    assert union_field(UNTAGGED, "text") == FieldPartial((text,))


def test_union_field_absent():
    assert union_field(U, "zzz") == FieldAbsent()
