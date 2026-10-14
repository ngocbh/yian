"""One-sided numeric ranges: `int(0..)`, `int(..100)`, `float(0.0..)`."""

import math

import pytest

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


@pytest.mark.parametrize("kind,bounds", [
    (INT, (None, None)), (NUM, (None, None)),
    (INT, (None, 1.0)), (NUM, (None, float("inf"))),
    (NUM, (10**400, None)),
])
def test_open_end_shape_still_rejects_bad_metadata(kind, bounds):
    typ = Type(kind)
    object.__setattr__(typ, "bounds", bounds)
    with pytest.raises((TypeError, ValueError)):
        typ.validate_shape()


def test_numeric_interval_maps_open_ends_to_infinity():
    assert numeric_interval(Type(INT, bounds=(0, None))) == (0, math.inf)
    assert numeric_interval(Type(INT, bounds=(None, 9))) == (-math.inf, 9)
    assert numeric_interval(Type(INT)) == (-math.inf, math.inf)
    assert numeric_interval(Type(INT, bounds=(1, 2))) == (1, 2)


@pytest.mark.parametrize("bounds,label", [
    ((0, None), "int(0..)"), ((None, 100), "int(..100)"), ((-5, None), "int(-5..)"),
])
def test_label_spells_the_open_end_empty(bounds, label):
    assert canonical_type_label(Type(INT, bounds=bounds)) == label
    assert canonical_type_label(Type(INT, bounds=bounds, nullable=True)) == f"Optional({label})"


def test_float_label():
    assert canonical_type_label(Type(NUM, bounds=(0.0, None))) == "float(0.0..)"


def test_value_check_one_sided():
    assert build_value_as(Type(INT, bounds=(0, None)), 10**50).to_object() == 10**50
    assert build_value_as(Type(INT, bounds=(None, 100)), -10**50).to_object() == -10**50
    with pytest.raises(TypeCheckError, match=r"^-1 is below the inclusive int minimum 0$"):
        build_value_as(Type(INT, bounds=(0, None)), -1)
    with pytest.raises(TypeCheckError, match=r"^101 is above the inclusive int maximum 100$"):
        build_value_as(Type(INT, bounds=(None, 100)), 101)
    with pytest.raises(TypeCheckError, match=r"below the inclusive float minimum 0\.0"):
        build_value_as(Type(NUM, bounds=(0.0, None)), -0.5)
    # the two-sided message is unchanged
    with pytest.raises(TypeCheckError, match=r"^13 is outside inclusive int bounds \(1, 12\)$"):
        build_value_as(Type(INT, bounds=(1, 12)), 13)


def test_value_check_one_sided_rejects_nan():
    with pytest.raises(TypeCheckError):
        build_value_as(Type(NUM, bounds=(0.0, None)), math.nan)


def test_overlap_with_open_ends():
    assert types_overlap(Type(INT, bounds=(0, None)), Type(INT, bounds=(None, 0)))
    assert not types_overlap(Type(INT, bounds=(1, None)), Type(INT, bounds=(None, 0)))
    assert types_overlap(Type(INT, bounds=(5, None)), Type(INT))
