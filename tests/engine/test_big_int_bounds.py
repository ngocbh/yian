"""Int range ends of any size parse, label exactly, check values and show in messages."""

from __future__ import annotations

import re
import sys

import pytest

import yian.json_codec as json_codec
from yian import load_flow, run_flow
from yian.compile.manifest import flow_fingerprint
from yian.nodes.agent import structured
from yian.typesys.types import parse_type, resolve_type
from yian.typesys.values import (
    Type,
    TypeCheckError,
    ValueKind,
    _labels,
    build_value_as,
    canonical_type_label,
)


BIG = 10**4999 + 12345  # 5000 digits, past Python's default 4300-digit limit
DIGITS = json_codec.int_text(BIG)
D = "1" * 5000
ONES = (10**5000 - 1) // 9  # the int spelled by D
PLACEHOLDER = "<int too large to display>"


@pytest.fixture(autouse=True)
def _default_int_digit_limit():
    """Run each test under Python's default int/str digit limit."""
    get = getattr(sys, "get_int_max_str_digits", None)
    if get is None:
        yield  # this Python has no limit
        return
    before = get()
    sys.set_int_max_str_digits(4300)
    try:
        yield
        after = get()
    finally:
        sys.set_int_max_str_digits(before)
    assert after == 4300, "yian code changed the global digit limit"


def _assert_clean(msg: str) -> None:
    assert "set_int_max_str_digits" not in msg


def _int_type(text: str) -> Type:
    return resolve_type(parse_type(text), {})


@pytest.mark.parametrize(
    ("text", "bounds"),
    [
        (f"int(0..{D})", (0, ONES)),
        (f"int(-{D}..0)", (-ONES, 0)),
        (f"int({D}..)", (ONES, None)),
    ],
    ids=["upper", "negative-lower", "open-upper"],
)
def test_a_long_int_range_end_parses_exactly(text, bounds):
    assert parse_type(text).bounds == bounds


def test_a_long_float_range_end_is_rejected_as_a_type_error():
    with pytest.raises(TypeCheckError) as caught:
        parse_type(f"float(0.0..{D})")
    assert "float bounds require" in str(caught.value)
    _assert_clean(str(caught.value))


def test_a_union_bar_with_a_long_range_end_hints_the_union_spelling():
    with pytest.raises(TypeCheckError) as caught:
        parse_type(f"int(0..{D}) | str")
    msg = str(caught.value)
    assert f"Union(int(0..{D}), str)" in msg
    _assert_clean(msg)


def test_a_comma_range_with_a_long_end_shows_the_range_spelling():
    with pytest.raises(TypeCheckError) as caught:
        parse_type(f"int(0, {D})")
    msg = str(caught.value)
    assert f"write a range: int(0..{D})" in msg
    _assert_clean(msg)


def test_a_bracket_range_with_a_long_end_says_types_use_parentheses():
    with pytest.raises(TypeCheckError) as caught:
        parse_type(f"int[0, {D}]")
    msg = str(caught.value)
    assert msg.startswith("types use parentheses, not brackets")
    _assert_clean(msg)


def test_the_canonical_label_of_a_long_range_end_writes_every_digit():
    assert _labels(_int_type("int(0..5)")) == ("int(0..5)", "int[0,5]")
    typ = _int_type(f"int(0..{D})")
    assert canonical_type_label(typ) == f"int(0..{D})"
    assert _labels(typ) == (f"int(0..{D})", f"int[0,{D}]")


def test_an_open_range_end_still_labels_empty():
    assert canonical_type_label(_int_type("int(0..)")) == "int(0..)"


def test_an_out_of_bounds_message_shows_a_long_end_as_a_placeholder():
    with pytest.raises(TypeCheckError) as caught:
        build_value_as(_int_type(f"int(0..{D})"), ONES + 1)
    msg = str(caught.value)
    assert msg == f"{PLACEHOLDER} is outside inclusive int bounds (0, {PLACEHOLDER})"
    assert re.search(r"[0-9]{600}", msg) is None


def test_a_short_out_of_bounds_message_is_unchanged():
    with pytest.raises(TypeCheckError) as caught:
        build_value_as(_int_type("int(1..12)"), 13)
    assert str(caught.value) == "13 is outside inclusive int bounds (1, 12)"


def _range_flow(high: str) -> str:
    return f"""
id: big-range
name: big_range
input:
  x: !int(0..{high})
nodes:
  c:
    kind: code
    input:
      n: ${{input.x}}
    code: |
      return inputs["n"]
output: ${{c.output}}
"""


def test_a_flow_input_with_a_long_range_end_checks_values():
    flow = load_flow(_range_flow(D))

    ok = run_flow(flow, {"x": BIG})
    assert ok.status == "succeeded"
    assert ok.output == BIG

    bad = run_flow(flow, {"x": ONES + 1})
    assert bad.status == "failed"
    assert bad.failure.code == "contract.input"
    msg = bad.failure.message
    assert msg == f"input 'x': {PLACEHOLDER} is outside inclusive int bounds (0, {PLACEHOLDER})"
    _assert_clean(msg)
    assert re.search(r"[0-9]{600}", msg) is None


def test_the_fingerprint_sees_every_digit_of_a_range_end():
    other = D[:-1] + "2"
    first = flow_fingerprint(load_flow(_range_flow(D)).compiled)
    second = flow_fingerprint(load_flow(_range_flow(other)).compiled)
    assert first != second


def test_the_fallback_instruction_writes_a_long_range_end():
    schema = structured.type_to_schema(Type(kind=ValueKind.INTEGER, bounds=(0, ONES)))
    assert schema is not None
    assert D in structured._fallback_instruction(schema)
