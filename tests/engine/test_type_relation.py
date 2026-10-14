"""Classification of an inferred type against a declared type."""

import pytest

from yian.compile.validation import type_relation
from yian.typesys.types import read_typedefs, type_for
from yian.typesys.values import Type, ValueKind

_REGISTRY = read_typedefs({
    "A": {"a": "int"},
    "AB": {"a": "int", "b": "Optional(str)"},
    "ABStr": {"a": "int", "b": "str"},
    "AX": {"a": "int", "x": "int"},
    "AStr": {"a": "str"},
    "AOpt": {"a": "Optional(int)"},
    "AOptStr": {"a": "Optional(str)"},
    "ALit": {"a": "Literal(x, y)"},
    "Nested": {"inner": "AB"},
    "NestedA": {"inner": "A"},
    "Buy": {"action": "Literal(buy)", "qty": "int"},
    "Sell": {"action": "Literal(sell)", "qty": "int"},
    "Hold": {"action": "Literal(hold)"},
    "Decision": "Union(Buy, Sell, Hold)",
    "X": {"x": "int"},
})


def _t(text: str) -> Type:
    """Resolve a type expression against the test typedefs."""
    return type_for(text, _REGISTRY)


@pytest.mark.parametrize(
    ("inferred", "declared"),
    [
        ("int", "int"),
        ("int", "float"),
        ("int", "Optional(int)"),
        ("Literal(a)", "str"),
        ("Literal(a)", "Literal(a, b)"),
        ("A", "object"),
        ("date", "str"),
        ("datetime", "str"),
        ("A", "AB"),
        ("None", "Optional(int)"),
        ("None", "None"),
        ("int(0..5)", "int(0..9)"),
        ("int(0..5)", "float"),
        ("list(int)", "list(float)"),
        ("list(int)", "list(Optional(int))"),
        ("NestedA", "Nested"),
        ("file", "file"),
    ],
)
def test_fits(inferred: str, declared: str) -> None:
    assert type_relation(_t(inferred), _t(declared)) == "fits"


@pytest.mark.parametrize(
    ("inferred", "declared"),
    [
        ("Optional(int)", "int"),
        ("int", "int(0..5)"),
        ("int(0..5)", "int(3..9)"),
        ("str", "Literal(a, b)"),
        ("Literal(a, b)", "Literal(b, c)"),
        ("object", "A"),
        ("str", "date"),
        ("str", "datetime"),
        ("AB", "A"),
        ("AOpt", "A"),
        ("list(Optional(int))", "list(int)"),
        # a tag-less source into a Literal stays narrower at any depth
        ("list(str)", "list(Literal(x, y))"),
        ("AStr", "ALit"),
        ("Literal(a, \"2026-01-01\")", "date"),
        ("Literal(\"2026-01-01T09:30:00\")", "datetime"),
        ("date", "Literal(a, \"2026-01-01\")"),
        # records are closed at every level
        ("Nested", "NestedA"),
        # both sides may omit the field, so the empty record is shared
        ("AOptStr", "AOpt"),
        ("float", "float(0..1)"),
        # a float-typed value may hold an int, since a typed binding never converts
        ("float", "int"),
    ],
)
def test_narrower(inferred: str, declared: str) -> None:
    assert type_relation(_t(inferred), _t(declared)) == "narrower"


def test_refined_list_into_unrefined_list_fits() -> None:
    assert type_relation(_t("list(int)"), Type.scalar(ValueKind.LIST_ANY)) == "fits"


def test_unrefined_list_into_refined_list_is_narrower() -> None:
    assert type_relation(Type.scalar(ValueKind.LIST_ANY), _t("list(int)")) == "narrower"


@pytest.mark.parametrize(
    ("inferred", "declared"),
    [
        ("str", "int"),
        ("bool", "int"),
        ("int", "list(int)"),
        ("list(str)", "list(int)"),
        ("list(int)", "list(object)"),
        ("object", "int"),
        ("int", "object"),
        ("file", "str"),
        ("date", "datetime"),
        ("datetime", "date"),
        ("Literal(a)", "Literal(b)"),
        ("int(0..5)", "int(10..20)"),
        ("None", "int"),
        ("ABStr", "A"),
        ("A", "AX"),
        ("AStr", "A"),
        ("Optional(str)", "Optional(int)"),
        ("A", "list(int)"),
        ("list(A)", "list(AStr)"),
        ("AOptStr", "A"),
        ("int", "float(0.5..0.7)"),
        ("Optional(int)", "None"),
        # the required field is a str; the declared field is an optional int
        ("AStr", "AOpt"),
        ("Literal(a)", "date"),
        ("Literal(\"2026-01-01\")", "datetime"),
        ("date", "Literal(a)"),
        ("datetime", "Literal(\"2026-01-01\")"),
    ],
)
def test_disjoint(inferred: str, declared: str) -> None:
    assert type_relation(_t(inferred), _t(declared)) == "disjoint"


def test_record_field_both_sides_may_hold_none_is_narrower() -> None:
    # An inferred literal record can require a field it holds as `None`, so neither
    # side may omit `a`, yet both admit `{"a": None}`.
    inferred = Type(
        kind=ValueKind.OBJECT,
        fields={"a": _t("Optional(str)")},
        required=frozenset({"a"}),
    )
    assert type_relation(inferred, _t("AOpt")) == "narrower"


@pytest.mark.parametrize(
    ("inferred", "declared", "expected"),
    [
        # a union member over the union is explicit narrowing
        ("Decision", "Buy", "narrower"),
        ("Buy", "Decision", "fits"),
        ("Decision", "Decision", "fits"),
        ("Union(Buy, Sell)", "Decision", "fits"),
        ("Decision", "Union(Buy, Sell)", "narrower"),
        ("Decision", "int", "disjoint"),
        ("Decision", "list(Buy)", "disjoint"),
        ("Decision", "object", "fits"),
        ("object", "Decision", "narrower"),
        ("Optional(Decision)", "Decision", "narrower"),
        ("Decision", "Optional(Decision)", "fits"),
        # records are closed, so a record matching no member shares no value
        ("X", "Decision", "disjoint"),
        ("Union(Buy, Sell)", "Hold", "disjoint"),
        ("Union(Buy, Sell)", "Union(Sell, Hold)", "narrower"),
        ("list(Buy)", "list(Decision)", "fits"),
        ("list(Decision)", "list(Buy)", "narrower"),
    ],
)
def test_union(inferred: str, declared: str, expected: str) -> None:
    assert type_relation(_t(inferred), _t(declared)) == expected
