from dataclasses import dataclass
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict

from yian.defaults import current
from yian.typesys.values import (
    FileRef,
    ObjectValue,
    TypeCheckError,
    build_value,
    recursive_entry_path,
    reject_deep_nesting,
    reject_non_text_keys,
)


def _as_dict(value: Any) -> dict:
    return {"a": value}


def _as_list(value: Any) -> list:
    return [value]


def _nest(n: int, wrap=_as_dict, leaf: Any = 1) -> Any:
    """Return `leaf` wrapped in `n` levels of `wrap`."""
    value = leaf
    for _ in range(n):
        value = wrap(value)
    return value


def _message(path: str, limit: int = 100) -> str:
    return (
        f"output at {path}: value nests deeper than {limit} levels, "
        "the limit for a yian value (each record and list is a level)"
    )


@dataclass
class _Box:
    value: Any


class _Model(BaseModel):
    value: Any


class _Extras(BaseModel):
    model_config = ConfigDict(extra="allow")


def test_limit_is_100():
    assert current().max_value_nesting == 100


@pytest.mark.parametrize("value", [1, "s", None, {}, [], _nest(100), _nest(100, _as_list)])
def test_values_within_the_limit_pass(value):
    reject_deep_nesting(value, "output")


def test_dict_chain_past_the_limit_names_the_path():
    with pytest.raises(TypeCheckError) as exc_info:
        reject_deep_nesting(_nest(101), "output")
    assert exc_info.value.path == ("a",) * 100
    assert str(exc_info.value) == _message(".".join(["a"] * 100))


def test_list_chain_past_the_limit_names_the_path():
    with pytest.raises(TypeCheckError) as exc_info:
        reject_deep_nesting(_nest(101, _as_list), "output")
    assert exc_info.value.path == (0,) * 100
    assert str(exc_info.value) == _message("[0]" * 100)


def test_enclosing_levels_count_toward_the_limit():
    with pytest.raises(TypeCheckError) as exc_info:
        reject_deep_nesting(_nest(100), "output", enclosing_levels=1)
    assert exc_info.value.path == ("a",) * 99


def test_negative_enclosing_levels_let_the_value_go_one_over():
    reject_deep_nesting(_nest(101), "output", enclosing_levels=-1)
    with pytest.raises(TypeCheckError) as exc_info:
        reject_deep_nesting(_nest(102), "output", enclosing_levels=-1)
    assert exc_info.value.path == ("a",) * 101


def test_mixed_path_segments():
    value = {"xs": _nest(100, _as_list)}
    with pytest.raises(TypeCheckError) as exc_info:
        reject_deep_nesting(value, "output")
    assert exc_info.value.path == ("xs",) + (0,) * 99
    assert str(exc_info.value) == _message("xs" + "[0]" * 99)


def test_non_identifier_key_renders_quoted():
    value = {"my key": _nest(100)}
    with pytest.raises(TypeCheckError) as exc_info:
        reject_deep_nesting(value, "output")
    assert exc_info.value.path == ("my key",) + ("a",) * 99
    assert str(exc_info.value) == _message("['my key']." + ".".join(["a"] * 99))


@pytest.mark.parametrize("wrap, segment", [
    (lambda v: (v,), 0),
    (_Box, "value"),
    (lambda v: _Model(value=v), "value"),
])
def test_tuple_dataclass_and_model_each_count_as_a_level(wrap, segment):
    reject_deep_nesting(wrap(_nest(99)), "output")
    with pytest.raises(TypeCheckError) as exc_info:
        reject_deep_nesting(wrap(_nest(100)), "output")
    assert exc_info.value.path == (segment,) + ("a",) * 99


def test_model_extras_are_direct_children():
    reject_deep_nesting(_Extras(extra=_nest(99)), "output")
    with pytest.raises(TypeCheckError) as exc_info:
        reject_deep_nesting(_Extras(extra=_nest(100)), "output")
    assert exc_info.value.path == ("extra",) + ("a",) * 99


def test_file_ref_at_the_last_level_passes():
    reject_deep_nesting(_nest(99, leaf=FileRef(uri="file:///x")), "output")
    with pytest.raises(TypeCheckError):
        reject_deep_nesting(_nest(100, leaf=FileRef(uri="file:///x")), "output")


def test_typed_value_adds_no_level():
    reject_deep_nesting(build_value(_nest(100)), "output")
    with pytest.raises(TypeCheckError) as exc_info:
        reject_deep_nesting(build_value(_nest(101)), "output")
    assert exc_info.value.path == ("a",) * 100


def test_first_offender_depth_first_in_order():
    value = {"ok": 1, "deep": _nest(101), "deeper": _nest(150)}
    with pytest.raises(TypeCheckError) as exc_info:
        reject_deep_nesting(value, "output")
    assert exc_info.value.path == ("deep",) + ("a",) * 99


def test_very_deep_value_does_not_recurse(set_defaults):
    deep = _nest(100_000, _as_list)
    with pytest.raises(TypeCheckError):
        reject_deep_nesting(deep, "output")
    set_defaults.unchecked(max_value_nesting=200_000)
    reject_deep_nesting(deep, "output")


def test_cycles_are_not_reported_as_too_deep():
    d: dict = {}
    d["self"] = d
    reject_deep_nesting(d, "output")
    items: list = []
    items.append(items)
    reject_deep_nesting(items, "output")


def test_recursive_entry_path_names_the_entry_that_refers_back():
    items: list = [1]
    items.append(items)
    record: dict = {"a": {}}
    record["a"]["b"] = record
    pair: list = []
    pair.append(("x", pair))

    assert recursive_entry_path(items) == (1,)
    assert recursive_entry_path(record) == ("a", "b")
    assert recursive_entry_path(pair) == (0, 1)


def test_recursive_entry_path_is_none_without_a_cycle():
    shared = [1, 2]

    assert recursive_entry_path({"a": shared, "b": [shared]}) is None
    assert recursive_entry_path(1) is None


def test_limit_is_read_at_call_time(set_defaults):
    set_defaults(MAX_VALUE_NESTING=3)
    reject_deep_nesting(_nest(3), "output")
    with pytest.raises(TypeCheckError) as exc_info:
        reject_deep_nesting(_nest(4), "output")
    assert exc_info.value.path == ("a",) * 3
    assert str(exc_info.value) == _message("a.a.a", limit=3)


def test_empty_path_message_has_no_location():
    with pytest.raises(TypeCheckError) as exc_info:
        reject_deep_nesting({}, "output", enclosing_levels=100)
    assert exc_info.value.path == ()
    assert str(exc_info.value) == (
        "output: value nests deeper than 100 levels, "
        "the limit for a yian value (each record and list is a level)"
    )


def _key_error(value: Any) -> TypeCheckError:
    """Return the error `reject_non_text_keys` raises for `value` at `input 'v'`."""
    with pytest.raises(TypeCheckError) as exc_info:
        reject_non_text_keys(value, "input 'v'")
    return exc_info.value


def test_text_keys_in_a_cyclic_value_pass():
    d: dict = {"a": [1, "b"]}
    d["self"] = d
    reject_non_text_keys(d, "input 'v'")


def test_a_top_level_non_text_key_has_no_location():
    error = _key_error({1: "a"})
    assert error.path == ()
    assert str(error) == "input 'v': object keys must be text, got 1"


@pytest.mark.parametrize("wrap", [_Box, lambda v: _Model(value=v)])
def test_a_non_text_key_inside_a_dataclass_or_model_names_its_path(wrap):
    error = _key_error({"m": wrap({2: "b"})})
    assert error.path == ("m", "value")
    assert str(error) == "input 'v' at m.value: object keys must be text, got 2"


def test_a_typed_value_is_unwrapped_for_the_key_check():
    # `model_construct` skips validation, which would reject the key.
    bad = ObjectValue.model_construct(value={1: "a"})
    assert _key_error(bad).path == ()
    error = _key_error({"x": bad})
    assert error.path == ("x",)
    assert str(error) == "input 'v' at x: object keys must be text, got 1"


def test_a_non_identifier_parent_key_renders_quoted():
    error = _key_error({"a b": {True: 1}})
    assert error.path == ("a b",)
    assert str(error) == "input 'v' at ['a b']: object keys must be text, got True"
