"""Tests for how `read_typedefs` walks typedef references: the walks are iterative, so a
long alias chain or a long record cycle never raises `RecursionError`, and a cycle is
reported at the typedef and field where the walk closes it."""

import time

import pytest

from yian.typesys.types import (
    AliasDef,
    RefExpr,
    ScalarExpr,
    _check_alias_cycles,
    read_typedefs,
)
from yian.typesys.values import TypeCheckError


def test_alias_cycle_check_walks_a_long_chain():
    registry = {"A0": AliasDef(target=ScalarExpr("int"))}
    for i in range(1, 5000):
        registry[f"A{i}"] = AliasDef(target=RefExpr(f"A{i - 1}"))
    start = time.monotonic()
    _check_alias_cycles(registry)
    assert time.monotonic() - start < 5


@pytest.mark.parametrize("n", [10, 200, 1000])
def test_record_cycle_reports_at_the_last_typedef(n):
    raw = {f"T{i}": {"a": f"T{(i + 1) % n}"} for i in range(n)}
    last = f"T{n - 1}"
    with pytest.raises(TypeCheckError) as info:
        read_typedefs(raw)
    assert str(info.value) == f"typedef {last!r} field 'a': recursive type reference: {last!r}"
    assert info.value.path == (last, "a")


def test_self_recursive_record():
    with pytest.raises(TypeCheckError) as info:
        read_typedefs({"R": {"a": "R"}})
    assert str(info.value) == "typedef 'R' field 'a': recursive type reference: 'R'"
    assert info.value.path == ("R", "a")


def test_two_record_cycle():
    with pytest.raises(TypeCheckError) as info:
        read_typedefs({"A": {"x": "B"}, "B": {"y": "A"}})
    assert str(info.value) == "typedef 'B' field 'y': recursive type reference: 'B'"
    assert info.value.path == ("B", "y")


def test_cycle_through_an_alias_and_a_record():
    with pytest.raises(TypeCheckError) as info:
        read_typedefs({"A": "B", "B": {"x": "A"}})
    assert str(info.value) == "typedef 'B' field 'x': recursive type reference: 'B'"
    assert info.value.path == ("B", "x")


def test_alias_cycle():
    with pytest.raises(TypeCheckError) as info:
        read_typedefs({"A": "B", "B": "A"})
    assert "alias cycle through" in str(info.value)
    assert info.value.path == ("A",)


def test_field_order_decides_between_a_union_error_and_a_cycle():
    with pytest.raises(TypeCheckError) as info:
        read_typedefs({
            "U1": {"x": "int"},
            "U2": {"x": "int"},
            "R": {"a": "Union(U1, U2)", "b": "R"},
        })
    assert str(info.value) == (
        "typedef 'R' field 'a': Union(U1, U2): members 'U1' and 'U2' can match the same "
        "value; add a tag field every member declares as a single-value Literal, with a "
        "different value in each member, e.g. kind: Literal(u1)"
    )
    assert info.value.path == ("R", "a")
