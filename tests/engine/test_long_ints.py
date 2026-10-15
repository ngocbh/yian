"""An int with more than 4300 decimal digits is not a yian value."""

from __future__ import annotations

import json
import math
import re
import sys

import pytest

from yian import load_flow, run_flow
from yian.typesys.values import (
    IntTooLongError,
    ListAnyValue,
    OverLimitInt,
    TypeCheckError,
    int_digits,
    long_int_digits,
    parse_int_text,
    parse_json_int,
    reject_long_ints,
    reject_nonfinite,
    rewrite_digit_limit_text,
    value_repr,
)


BIG = 10**4999 + 12345  # 5000 digits
D = "1" * 5000
LIMIT_MSG = "integer with 5000 digits is larger than yian allows (4300 digits)"

_get_limit = getattr(sys, "get_int_max_str_digits", None)
needs_limit = pytest.mark.skipif(_get_limit is None, reason="this Python has no digit limit")


@pytest.fixture
def python_limit():
    """Yield a setter for Python's int/str digit limit, restoring it afterwards."""
    before = _get_limit()
    try:
        yield sys.set_int_max_str_digits
    finally:
        sys.set_int_max_str_digits(before)


def _assert_clean(message: str) -> None:
    """Check a message shows no digits of the long int and no Python limit hint."""
    assert "set_int_max_str_digits" not in message
    assert re.search(r"[0-9]{600}", message) is None


@pytest.mark.parametrize("number, digits", [
    (0, 1),
    (9, 1),
    (10, 2),
    (-10**4299, 4300),
    (10**4300 - 1, 4300),
    (10**4300, 4301),
    (BIG, 5000),
    (10**100000, 100001),
], ids=["0", "9", "10", "-1e4299", "1e4300-1", "1e4300", "big", "1e100000"])
def test_int_digits_counts_decimal_digits(number, digits):
    assert int_digits(number) == digits


def test_parse_int_text_skips_leading_zeros():
    assert parse_int_text("0" * 4999 + "1") == 1


def test_parse_int_text_rejects_a_long_int():
    with pytest.raises(IntTooLongError) as caught:
        parse_int_text("-" + D)
    assert isinstance(caught.value, ValueError)
    assert str(caught.value) == LIMIT_MSG
    assert caught.value.digits == 5000
    _assert_clean(str(caught.value))


@pytest.mark.parametrize("text, expected", [
    ("9" * 4300, int("9" * 4300)),
    ("+12", 12),
    (" 1_000 ", 1000),
    ("0_" * 3000 + "1", 1),
], ids=["4300-nines", "plus", "underscores", "zero-underscores"])
def test_parse_int_text_accepts_int_grammar(text, expected):
    assert parse_int_text(text) == expected


def test_parse_int_text_skips_leading_zeros_of_any_script():
    assert parse_int_text("\u0660" * 4400 + "1") == 1


def test_parse_int_text_raises_int_own_error_for_invalid_text():
    with pytest.raises(ValueError) as caught:
        parse_int_text("abc")
    assert not isinstance(caught.value, IntTooLongError)
    assert "invalid literal" in str(caught.value)


@needs_limit
def test_parse_int_text_rejects_a_long_int_with_python_limit_off(python_limit):
    python_limit(0)
    with pytest.raises(IntTooLongError) as caught:
        parse_int_text("-" + D)
    assert str(caught.value) == LIMIT_MSG


def test_parse_json_int_marks_a_long_token():
    parsed = json.loads("[1, " + D + "]", parse_int=parse_json_int)
    assert parsed[0] == 1
    marker = parsed[1]
    assert isinstance(marker, OverLimitInt)
    assert marker.digits == 5000


@pytest.mark.parametrize("value, message, path", [
    (BIG, "input 'n': " + LIMIT_MSG, ()),
    (-BIG, "input 'n': " + LIMIT_MSG, ()),
    ({"a": [1, BIG]}, "input 'n' at a[1]: " + LIMIT_MSG, ("a", 1)),
    ({BIG: 1}, "input 'n': " + LIMIT_MSG, ()),
    ([parse_json_int(D)], "input 'n' at [0]: " + LIMIT_MSG, (0,)),
    (ListAnyValue(value=[BIG]), "input 'n' at [0]: " + LIMIT_MSG, (0,)),
], ids=["root", "negative", "nested", "key", "marker", "typed-value"])
def test_reject_nonfinite_rejects_a_long_int(value, message, path):
    with pytest.raises(TypeCheckError) as caught:
        reject_nonfinite(value, "input 'n'")
    assert str(caught.value) == message
    assert caught.value.path == path
    _assert_clean(str(caught.value))


def test_reject_nonfinite_passes_ints_within_the_limit_and_bools():
    reject_nonfinite(10**4300 - 1, "input 'n'")
    reject_nonfinite(True, "input 'n'")


def test_reject_nonfinite_still_reports_nan():
    with pytest.raises(TypeCheckError) as caught:
        reject_nonfinite(math.nan, "input 'n'")
    assert str(caught.value) == "input 'n': non-finite number nan is not a yian value"


def test_reject_long_ints_reports_the_path():
    with pytest.raises(TypeCheckError) as caught:
        reject_long_ints({"a": BIG}, "feedback")
    assert str(caught.value) == "feedback at a: " + LIMIT_MSG
    assert caught.value.path == ("a",)


def test_reject_long_ints_ignores_nan():
    reject_long_ints(math.nan, "x")


def test_long_int_digits_finds_the_first_long_int():
    assert long_int_digits(["x", {"k": BIG}]) == 5000
    assert long_int_digits([1, "a"]) is None


@needs_limit
def test_value_repr_shows_a_digit_count_for_a_long_int(python_limit):
    python_limit(4300)
    assert value_repr(BIG) == "<int with 5000 digits>"
    assert value_repr([BIG]) == "<list holding an int with 5000 digits>"
    assert value_repr({BIG}) == "<set too large to display>"


@needs_limit
def test_value_repr_shows_a_digit_count_with_python_limit_off(python_limit):
    python_limit(0)
    assert value_repr(BIG) == "<int with 5000 digits>"
    assert value_repr([BIG]) == "<list holding an int with 5000 digits>"


def test_value_repr_of_a_small_int():
    assert value_repr(5) == "5"


def test_rewrite_digit_limit_text_replaces_python_message():
    text = (
        "bad: Exceeds the limit (4300 digits) for integer string conversion: "
        "value has 5000 digits; use sys.set_int_max_str_digits() to increase the limit"
    )
    assert rewrite_digit_limit_text(text) == "bad: " + LIMIT_MSG


def test_rewrite_digit_limit_text_without_a_count():
    text = (
        "Exceeds the limit (4300 digits) for integer string conversion; "
        "use sys.set_int_max_str_digits() to increase the limit"
    )
    assert rewrite_digit_limit_text(text) == "integer longer than yian allows (4300 digits)"


def test_rewrite_digit_limit_text_matches_any_host_limit():
    text = (
        "Exceeds the limit (9000 digits) for integer string conversion: "
        "value has 10000 digits; use sys.set_int_max_str_digits() to increase the limit"
    )
    assert rewrite_digit_limit_text(text) == (
        "integer with 10000 digits is larger than yian allows (4300 digits)"
    )


def test_rewrite_digit_limit_text_leaves_other_text():
    assert rewrite_digit_limit_text("some other error") == "some other error"


def _input_flow(declared: str) -> str:
    return f"""
id: long-input
name: long_input
input:
  {declared}
nodes:
  c:
    kind: code
    input:
      y: ${{input.{declared.split(":")[0]}}}
    code: |
      return 1
output: ${{c.output}}
"""


@pytest.mark.parametrize("declared, inputs, message", [
    ("n: !int", {"n": BIG}, "input 'n': " + LIMIT_MSG),
    ("n: !int", {"n": D}, "input 'n': " + LIMIT_MSG),
    ("xs: !list(int)", {"xs": [1, BIG]}, "input 'xs' at [1]: " + LIMIT_MSG),
    ("n: !str", {"n": BIG}, "input 'n': " + LIMIT_MSG),
], ids=["int", "int-text", "list", "str"])
def test_a_long_int_run_input_is_rejected_before_the_run_starts(
    tmp_path, declared, inputs, message
):
    flow = load_flow(_input_flow(declared))
    log = tmp_path / "run.jsonl"

    with pytest.raises(TypeCheckError) as caught:
        run_flow(flow, inputs, log=log)

    assert str(caught.value) == message
    _assert_clean(str(caught.value))
    assert not log.exists()


def test_yian_max_int_digits_sets_the_limit(set_defaults):
    set_defaults(MAX_INT_DIGITS=100)
    message = "integer with 101 digits is larger than yian allows (100 digits)"

    with pytest.raises(TypeCheckError) as caught:
        reject_nonfinite({"a": 10**100}, "input 'n'")
    assert str(caught.value) == "input 'n' at a: " + message
    reject_nonfinite(10**100 - 1, "input 'n'")
    with pytest.raises(IntTooLongError, match=re.escape(message)):
        parse_int_text("1" * 101)
    assert isinstance(parse_json_int("1" * 101), OverLimitInt)
    assert long_int_digits([10**100]) == 101
