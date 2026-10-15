"""Tests for how `read_typedefs` walks typedef references: the walks are iterative, so a
long alias chain or a long record cycle never raises `RecursionError`, and a cycle is
reported at the typedef and field where the walk closes it."""

import time

import pytest

from yian.compose import load_flow
from yian.defaults import Defaults
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


def test_aliases_wrapped_in_optional_resolve():
    raw = {"A0": {"a": "int"}}
    for i in range(1, 20):
        raw[f"A{i}"] = "Optional(" * 99 + f"A{i - 1}" + ")" * 99
    read_typedefs(raw)


def test_aliases_wrapped_in_union_resolve():
    raw = {"A0": {"a": "int"}}
    for i in range(1, 100):
        raw[f"A{i}"] = "Union(" * 99 + f"A{i - 1}" + ")" * 99
    read_typedefs(raw)


def _flow(typedefs: dict[str, str], output: str) -> str:
    """Return a flow declaring `typedefs` whose CODE node `use` has `output: <output>`."""
    lines = "".join(f"  {name}: {defn}\n" for name, defn in typedefs.items())
    return (
        f"id: f\nname: f\ntypedefs:\n{lines}nodes:\n  use:\n    kind: code\n"
        f"    code: tests.engine._compose_codefns:echo\n    input:\n      topic: x\n"
        f"    output: {output}\noutput: ${{use.output}}\n"
    )


def _union_levels(limit: int, tagged: bool) -> str:
    """Return a flow of `limit` record levels, each above the bottom one holding an
    optional union of the level below and `S`; the bottom field nests `limit` Optionals."""
    tag = "k: Literal(r), " if tagged else ""
    deep = "Optional(" * limit + "int" + ")" * limit
    typedefs = {"S": "{k: Literal(s)}" if tagged else "{x: int}", "T0": f"{{{tag}a: '{deep}'}}"}
    for i in range(1, limit):
        typedefs[f"T{i}"] = f"{{{tag}a: 'Optional(Union(T{i - 1}, S))'}}"
    return _flow(typedefs, f"T{limit - 1}")


def _unions_over_a_deep_union(limit: int) -> str:
    """Return a flow like `_union_levels` with `limit - 1` records over a bottom field of
    `limit` nested unions of `S`, so `S` is the last of `limit` levels."""
    deep = "Union(" * limit + "S" + ")" * limit
    typedefs = {"S": "{k: Literal(s)}", "T0": f"{{k: Literal(r), a: '{deep}'}}"}
    for i in range(1, limit - 1):
        typedefs[f"T{i}"] = f"{{k: Literal(r), a: 'Optional(Union(T{i - 1}, S))'}}"
    return _flow(typedefs, f"T{limit - 2}")


def _records_over_a_deep_union(limit: int) -> str:
    """Return a flow of `limit - 1` records, each the only field of the next, over a
    bottom field of `limit` nested unions of `S`."""
    deep = "Union(" * limit + "S" + ")" * limit
    typedefs = {"S": "{x: int}", "T0": f"{{a: '{deep}'}}"}
    for i in range(1, limit - 1):
        typedefs[f"T{i}"] = f"{{a: T{i - 1}}}"
    return _flow(typedefs, f"T{limit - 2}")


@pytest.mark.parametrize("limit", [None, 125])
@pytest.mark.parametrize(
    "build",
    [
        lambda limit: _union_levels(limit, tagged=True),
        lambda limit: _union_levels(limit, tagged=False),
        _unions_over_a_deep_union,
        _records_over_a_deep_union,
    ],
    ids=["tagged-unions", "untagged-unions", "unions-over-union", "records-over-union"],
)
def test_types_at_the_limit_load(build, limit, set_defaults):
    if limit is not None:
        set_defaults(MAX_TYPE_NESTING=limit)
    load_flow(build(limit or Defaults().max_type_nesting))


def test_field_parse_follows_the_int_digit_limit(set_defaults):
    raw = {"R": {"a": "int(0..123456)"}}
    read_typedefs(raw)
    set_defaults(MAX_INT_DIGITS=3)
    with pytest.raises(TypeCheckError):
        read_typedefs(raw)


def test_field_parse_follows_the_type_nesting_limit(set_defaults):
    raw = {"R": {"a": "list(" * 6 + "int" + ")" * 6}}
    read_typedefs(raw)
    set_defaults(MAX_TYPE_NESTING=5)
    with pytest.raises(TypeCheckError):
        read_typedefs(raw)
