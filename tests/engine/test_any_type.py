import datetime
from typing import Any

import pytest
from yian.typesys.values import (
    ANY_VALUE_ADAPTER,
    BooleanValue,
    FileRef,
    IntegerValue,
    ListAnyValue,
    NoneValue,
    NumberValue,
    ObjectValue,
    StringValue,
    Type,
    TypeCheckError,
    ValueKind,
    build_value_as,
)

ANY = Type.scalar(ValueKind.ANY)


def _nest(n: int, leaf: Any = 1) -> Any:
    """Return `leaf` wrapped in `n` lists."""
    value = leaf
    for _ in range(n):
        value = [value]
    return value


@pytest.mark.parametrize(
    ("value", "carrier", "stored"),
    [
        ("hello", StringValue, "hello"),
        (3, IntegerValue, 3),
        (2**80, IntegerValue, 2**80),
        (2.5, NumberValue, 2.5),
        (True, BooleanValue, True),
        ({"a": [1, None]}, ObjectValue, {"a": [1, None]}),
        ([1, "two", None], ListAnyValue, [1, "two", None]),
        ([1, 2.5], ListAnyValue, [1, 2.5]),
        ((1, 2), ListAnyValue, [1, 2]),
        ([True, 1], ListAnyValue, [True, 1]),
    ],
)
def test_any_stores_a_json_value_in_its_natural_carrier(value, carrier, stored):
    typed = build_value_as(ANY, value)

    assert type(typed) is carrier
    assert typed.value == stored


def test_any_keeps_list_items_as_given():
    ints = build_value_as(ANY, [1, 2.5]).value
    bools = build_value_as(ANY, [True, 1]).value

    assert type(ints[0]) is int
    assert type(bools[0]) is bool
    assert type(bools[1]) is int


def test_any_accepts_a_bare_value_kind():
    assert build_value_as(ValueKind.ANY, "hello") == StringValue(value="hello")


@pytest.mark.parametrize(
    ("value", "message", "path"),
    [
        (None, "None does not match declared type Any", ()),
        ({1, 2}, "{1, 2} is not a JSON value", ()),
        (b"x", "b'x' is not a JSON value", ()),
        (datetime.date(2026, 1, 1), "datetime.date(2026, 1, 1) is not a JSON value", ()),
        (FileRef(uri="f"), "is not a JSON value", ()),
        (object(), "is not a JSON value", ()),
        ({1: "a"}, "object keys must be text, got 1", ()),
        ({"a": {1, 2}}, "{1, 2} is not a JSON value", ("a",)),
        ({"a": [1, {2}]}, "{2} is not a JSON value", ("a", 1)),
    ],
)
def test_any_rejects_a_non_json_value_at_its_path(value, message, path):
    with pytest.raises(TypeCheckError) as caught:
        build_value_as(ANY, value)

    assert message in str(caught.value)
    assert caught.value.path == path


def test_any_rejects_an_over_deep_value():
    with pytest.raises(TypeCheckError, match="nests deeper than 100 levels"):
        build_value_as(ANY, _nest(150))


def test_any_accepts_a_value_at_the_nesting_limit():
    assert type(build_value_as(ANY, _nest(100))) is ListAnyValue


def test_any_ends_a_cyclic_value():
    cyclic: dict = {}
    cyclic["self"] = cyclic

    with pytest.raises(TypeCheckError):
        build_value_as(ANY, cyclic)


def test_optional_any_accepts_none():
    assert build_value_as(Type(kind=ValueKind.ANY, nullable=True), None) == NoneValue()


def test_object_rejects_a_non_text_key():
    with pytest.raises(TypeCheckError) as caught:
        build_value_as(ValueKind.OBJECT, {1: "a"})

    assert str(caught.value) == "object keys must be text, got 1"
    assert caught.value.path == ()


@pytest.mark.parametrize(
    ("value", "path"),
    [
        ({"a": {1, 2}}, ("a",)),
        ({"a": object()}, ("a",)),
        ({"d": datetime.date(2026, 1, 1)}, ("d",)),
    ],
)
def test_object_rejects_a_non_json_member(value, path):
    with pytest.raises(TypeCheckError, match="is not a JSON value") as caught:
        build_value_as(ValueKind.OBJECT, value)

    assert caught.value.path == path


def test_object_accepts_json_members():
    value = {"a": None, "b": [1, {"c": "d"}], "t": (1, 2)}

    assert build_value_as(ValueKind.OBJECT, value) == ObjectValue(value=value)


def test_flat_list_of_objects_rejects_a_non_text_key_at_the_item():
    with pytest.raises(TypeCheckError, match="object keys must be text, got 1") as caught:
        build_value_as(ValueKind.LIST_OBJECT, [{"a": 1}, {1: "b"}])

    assert caught.value.path == (1,)


def test_elementless_list_any_checks_items_are_json_values():
    assert build_value_as(ValueKind.LIST_ANY, [1, None, "x"]).value == [1, None, "x"]

    with pytest.raises(TypeCheckError, match="is not a JSON value") as caught:
        build_value_as(ValueKind.LIST_ANY, [{1, 2}])

    assert caught.value.path == (0,)


def test_record_with_required_any_field_rejects_null():
    record = Type(
        kind=ValueKind.OBJECT,
        fields={"v": ANY},
        required=frozenset({"v"}),
    )

    with pytest.raises(TypeCheckError) as caught:
        build_value_as(record, {"v": None})

    assert caught.value.path == ("v",)


def test_record_with_optional_any_field_accepts_absent_and_null():
    record = Type(
        kind=ValueKind.OBJECT,
        fields={"v": Type(kind=ValueKind.ANY, nullable=True)},
        required=frozenset(),
    )

    assert build_value_as(record, {}).value == {}
    assert build_value_as(record, {"v": None}).value == {"v": None}


@pytest.mark.parametrize(
    "value",
    ["hello", 3, 2**80, 2.5, True, {"a": [1, None]}, [1, "two", None], [1, 2.5], (1, 2)],
)
def test_any_value_round_trips_through_the_wire(value):
    typed = build_value_as(ANY, value)

    decoded = ANY_VALUE_ADAPTER.validate_python(
        ANY_VALUE_ADAPTER.dump_python(typed, mode="json")
    )

    assert type(decoded) is type(typed)
    assert decoded.value == typed.value
    assert [type(item) for item in _items(decoded.value)] == [
        type(item) for item in _items(typed.value)
    ]


def _items(value: Any) -> list:
    return value if isinstance(value, list) else [value]


def test_any_type_shape_is_valid():
    Type(kind=ValueKind.ANY).validate_shape()


@pytest.mark.parametrize(
    "refinement",
    [
        {"fields": {}},
        {"tags": frozenset({"a"})},
        {"bounds": (0, 1)},
        {"element": Type.scalar(ValueKind.STRING)},
    ],
)
def test_any_type_rejects_refinements(refinement):
    with pytest.raises(ValueError):
        Type(kind=ValueKind.ANY, **refinement).validate_shape()
