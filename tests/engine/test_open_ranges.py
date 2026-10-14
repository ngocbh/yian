"""One-sided numeric ranges: `int(0..)`, `int(..100)`, `float(0.0..)`."""

import math
import re

import pytest

from yian.typesys.types import format_type_expr, parse_type, read_typedefs, resolve_type, type_for
from yian.typesys.values import (
    Type,
    TypeCheckError,
    ValueKind,
    build_value_as,
    canonical_type_label,
    numeric_interval,
    types_overlap,
)

INT, NUM = ValueKind.INTEGER, ValueKind.NUMBER


@pytest.mark.parametrize("kind,bounds", [
    (INT, (0, None)), (INT, (None, 100)),
    (NUM, (0.0, None)), (NUM, (None, 1)), (NUM, (-1.5, None)),
])
def test_one_open_end_is_a_valid_shape(kind, bounds):
    Type(kind, bounds=bounds).validate_shape()


@pytest.mark.parametrize("kind,bounds,error", [
    (INT, (None, None), ValueError),
    (INT, (None, 1.0), TypeError),
    (NUM, (None, float("inf")), ValueError),
    (NUM, (10**400, None), ValueError),
])
def test_open_end_shape_still_rejects_bad_metadata(kind, bounds, error):
    typ = Type(kind)
    object.__setattr__(typ, "bounds", bounds)
    with pytest.raises(error) as caught:
        typ.validate_shape()
    assert type(caught.value) is error


def test_numeric_interval_maps_open_ends_to_infinity():
    assert numeric_interval(Type(INT, bounds=(0, None))) == (0, math.inf)
    assert numeric_interval(Type(INT, bounds=(None, 9))) == (-math.inf, 9)
    assert numeric_interval(Type(INT)) == (-math.inf, math.inf)
    assert numeric_interval(Type(INT, bounds=(1, 2))) == (1, 2)


@pytest.mark.parametrize("kind,bounds,label", [
    (INT, (0, None), "int(0..)"), (INT, (None, 100), "int(..100)"), (INT, (-5, None), "int(-5..)"),
    (NUM, (0.0, None), "float(0.0..)"),
])
def test_label_spells_the_open_end_empty(kind, bounds, label):
    assert canonical_type_label(Type(kind, bounds=bounds)) == label
    assert canonical_type_label(Type(kind, bounds=bounds, nullable=True)) == f"Optional({label})"


def test_value_check_one_sided():
    assert build_value_as(Type(INT, bounds=(0, None)), 10**50).to_object() == 10**50
    assert build_value_as(Type(INT, bounds=(None, 100)), -10**50).to_object() == -10**50
    with pytest.raises(TypeCheckError, match=r"^-1 is below the inclusive int minimum 0$"):
        build_value_as(Type(INT, bounds=(0, None)), -1)
    with pytest.raises(TypeCheckError, match=r"^101 is above the inclusive int maximum 100$"):
        build_value_as(Type(INT, bounds=(None, 100)), 101)
    with pytest.raises(TypeCheckError, match=r"^-0\.5 is below the inclusive float minimum 0\.0$"):
        build_value_as(Type(NUM, bounds=(0.0, None)), -0.5)
    # the two-sided message is unchanged
    with pytest.raises(TypeCheckError, match=r"^13 is outside inclusive int bounds \(1, 12\)$"):
        build_value_as(Type(INT, bounds=(1, 12)), 13)


def test_value_check_one_sided_rejects_nan():
    with pytest.raises(TypeCheckError, match=r"^nan is outside the inclusive range float\(0\.0\.\.\)$"):
        build_value_as(Type(NUM, bounds=(0.0, None)), math.nan)
    with pytest.raises(TypeCheckError, match=r"^nan is outside the inclusive range float\(\.\.1\)$"):
        build_value_as(Type(NUM, bounds=(None, 1), nullable=True), math.nan)


def test_overlap_with_open_ends():
    assert types_overlap(Type(INT, bounds=(0, None)), Type(INT, bounds=(None, 0)))
    assert not types_overlap(Type(INT, bounds=(1, None)), Type(INT, bounds=(None, 0)))
    assert types_overlap(Type(INT, bounds=(5, None)), Type(INT))


@pytest.mark.parametrize("text", [
    "int(0..)", "int(..100)", "float(0.0..)", "float(..-1.5)",
    "Optional(int(0..))", "list(float(..1.0))",
])
def test_open_ranges_round_trip(text):
    assert format_type_expr(parse_type(text)) == text
    assert canonical_type_label(type_for(text, {})) == text


def test_open_ranges_resolve_in_records_aliases_and_lists():
    reg = read_typedefs({
        "Count": "int(0..)",
        "Order": {"qty": "int(1..)", "price": "float(..100.0)"},
    })
    assert resolve_type(parse_type("Count"), reg).bounds == (0, None)
    order = resolve_type(parse_type("Order"), reg)
    assert order.fields["qty"].bounds == (1, None)
    assert order.fields["price"].bounds == (None, 100.0)
    assert type_for("list(Optional(int(..3)))", {}).element.bounds == (None, 3)


def test_neither_end_says_write_the_plain_type():
    with pytest.raises(TypeCheckError, match=re.escape("write int")):
        parse_type("int(..)")
