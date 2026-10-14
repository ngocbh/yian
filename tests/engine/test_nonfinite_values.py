import enum
import math
from dataclasses import dataclass

import pytest
from pydantic import BaseModel, ConfigDict

from yian.runtime.log import canonical_typed
from yian.typesys.values import (
    ObjectValue,
    Type,
    TypeCheckError,
    ValueKind,
    build_value,
    build_value_as,
    reject_nonfinite,
)

NONFINITE = [math.inf, -math.inf, math.nan]


class _Extras(BaseModel):
    model_config = ConfigDict(extra="allow")
    declared: float = 1.0


@dataclass
class _Box:
    value: float


class _Key(str, enum.Enum):
    A = "a"


@pytest.mark.parametrize("number", NONFINITE)
@pytest.mark.parametrize("wrap, path", [
    (lambda n: n, ()),
    (lambda n: {"a": [1.0, n]}, ("a", 1)),
    (lambda n: (0, {"b": n}), (1, "b")),
    (lambda n: ObjectValue(value={"k": n}), ("k",)),
    (lambda n: _Extras(lost=n), ("lost",)),
    (lambda n: _Box(n), ("value",)),
])
def test_reject_nonfinite_reports_the_first_path(number, wrap, path):
    with pytest.raises(TypeCheckError, match=f"non-finite number {number!r} ") as caught:
        reject_nonfinite(wrap(number), "output")
    assert caught.value.path == path


def test_reject_nonfinite_accepts_finite_values_and_cycles():
    cyclic = {"x": 1.5}
    cyclic["self"] = cyclic
    reject_nonfinite({"a": [1, 2.5, "inf", None, True]}, "output")
    reject_nonfinite(cyclic, "output")


@pytest.mark.parametrize("value, path, message", [
    (math.inf, (), "input 'y': non-finite number inf is not a yian value"),
    (
        {"a": [0, math.nan]},
        ("a", 1),
        "input 'y' at a[1]: non-finite number nan is not a yian value",
    ),
    (
        {"a": [0, {"b": math.inf}]},
        ("a", 1, "b"),
        "input 'y' at a[1].b: non-finite number inf is not a yian value",
    ),
    (
        [{"my key": -math.inf}],
        (0, "my key"),
        "input 'y' at [0]['my key']: non-finite number -inf is not a yian value",
    ),
])
def test_reject_nonfinite_names_where_and_path(value, path, message):
    with pytest.raises(TypeCheckError) as caught:
        reject_nonfinite(value, "input 'y'")
    assert str(caught.value) == message
    assert caught.value.path == path


def test_a_str_subclass_key_becomes_an_exact_str_segment():
    with pytest.raises(TypeCheckError) as caught:
        reject_nonfinite({_Key.A: math.inf}, "output")
    assert caught.value.path == ("a",) and type(caught.value.path[0]) is str


def test_reject_nonfinite_passes_finite_values():
    reject_nonfinite({"a": [1.0]}, "input 'y'")


def test_an_int_too_large_for_a_float_slot_fails_at_the_root():
    with pytest.raises(TypeCheckError, match="too large for float") as caught:
        build_value_as(Type.scalar(ValueKind.NUMBER), 10**400)
    assert caught.value.path == ()


def test_an_int_too_large_for_a_float_list_item_fails_at_its_index():
    with pytest.raises(TypeCheckError, match="too large for float") as caught:
        build_value_as(Type.scalar(ValueKind.LIST_NUMBER), [1, 10**400])
    assert caught.value.path == (1,)


def test_an_inferred_float_list_with_a_huge_int_fails_at_its_index():
    with pytest.raises(TypeCheckError, match="too large for float") as caught:
        build_value([0.5, 10**400])
    assert caught.value.path == (1,)
    with pytest.raises(TypeCheckError, match="too large for float"):
        canonical_typed([0.5, 10**400], None)


@pytest.mark.parametrize("value, path, message", [
    (1j, (), "input 'x': complex number 1j is not a yian value"),
    ({"a": [1, 2j]}, ("a", 1), "input 'x' at a[1]: complex number 2j is not a yian value"),
    (ObjectValue(value={"k": 3j}), ("k",), "input 'x' at k: complex number 3j is not a yian value"),
    ((0, {"b": 4j}), (1, "b"), "input 'x' at [1].b: complex number 4j is not a yian value"),
    (_Box(5j), ("value",), "input 'x' at value: complex number 5j is not a yian value"),
    (
        [1j, math.nan],
        (0,),
        "input 'x' at [0]: complex number 1j is not a yian value",
    ),
    (
        [math.nan, 1j],
        (0,),
        "input 'x' at [0]: non-finite number nan is not a yian value",
    ),
])
def test_reject_nonfinite_rejects_a_complex_at_its_path(value, path, message):
    with pytest.raises(TypeCheckError) as caught:
        reject_nonfinite(value, "input 'x'")
    assert str(caught.value) == message
    assert caught.value.path == path
