import math
from dataclasses import dataclass

import pytest
from pydantic import BaseModel, ConfigDict, RootModel

from yian.typesys.values import ObjectValue, TypeCheckError, find_nonfinite, reject_nonfinite

NONFINITE = [math.inf, -math.inf, math.nan]


@pytest.mark.parametrize("number", NONFINITE)
@pytest.mark.parametrize("wrap, path", [
    (lambda n: n, ()),
    (lambda n: {"a": [1.0, n]}, ("a", 1)),
    (lambda n: (0, {"b": n}), (1, "b")),
    (lambda n: ObjectValue(value={"k": n}), ("k",)),
])
def test_find_nonfinite_reports_the_first_path(number, wrap, path):
    found_path, found = find_nonfinite(wrap(number))
    assert found_path == path
    assert repr(found) == repr(number)


def test_find_nonfinite_walks_models_extras_and_dataclasses():
    class Extras(BaseModel):
        model_config = ConfigDict(extra="allow")
        declared: float = 1.0

    @dataclass
    class Box:
        value: float

    assert find_nonfinite(Extras(lost=math.inf))[0] == ("lost",)
    assert find_nonfinite(Box(math.nan))[0] == ("value",)


def test_find_nonfinite_descends_a_root_model_without_a_root_segment():
    assert find_nonfinite(RootModel[list[float]]([math.inf]))[0] == (0,)


def test_find_nonfinite_accepts_finite_values_and_cycles():
    cyclic = {"x": 1.5}
    cyclic["self"] = cyclic
    assert find_nonfinite({"a": [1, 2.5, "inf", None, True]}) is None
    assert find_nonfinite(cyclic) is None


@pytest.mark.parametrize("value, message", [
    (math.inf, "input 'y': non-finite number inf is not a yian value"),
    ({"a": [0, math.nan]}, "input 'y' at a[1]: non-finite number nan is not a yian value"),
    ([{"my key": -math.inf}], "input 'y' at [0]['my key']: non-finite number -inf is not a yian value"),
])
def test_reject_nonfinite_names_where_and_path(value, message):
    with pytest.raises(TypeCheckError) as caught:
        reject_nonfinite(value, "input 'y'")
    assert str(caught.value) == message
    assert caught.value.path == find_nonfinite(value)[0]


def test_a_str_subclass_key_becomes_an_exact_str_segment():
    import enum

    class Key(str, enum.Enum):
        A = "a"

    with pytest.raises(TypeCheckError) as caught:
        reject_nonfinite({Key.A: math.inf}, "output")
    assert caught.value.path == ("a",) and type(caught.value.path[0]) is str


def test_reject_nonfinite_passes_finite_values():
    reject_nonfinite({"a": [1.0]}, "input 'y'")
