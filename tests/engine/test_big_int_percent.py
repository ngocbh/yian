"""`%` formatting writes ints of any size as Python would without its digit limit."""

from __future__ import annotations

import re
import sys

import pytest

import yian.json_codec as json_codec
from yian import load_flow, run_flow
from yian.expr.expressions import ExpressionError, eval_expr
from yian.expr.grammar import parse_expr
from yian.expr.percent_format import format_text


BIG = 10**4999 + 12345  # 5000 digits, past Python's default 4300-digit limit
DIGITS = json_codec.int_text(BIG)
SMALL = 10**12 + 7


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


def _expected(fmt: str, x: int) -> str:
    """Return the text Python's `%` writes for `x` under `fmt`, built from `DIGITS`."""
    sign = "-" if x < 0 else ""
    text = sign + DIGITS
    plus = "-" if x < 0 else "+"
    return {
        "%s": text,
        "%r": text,
        "%a": text,
        "%d": text,
        "%i": text,
        "%u": text,
        "%+d": plus + DIGITS,
        "% d": ("-" if x < 0 else " ") + DIGITS,
        "%5010d": text.rjust(5010),
        "%-5010d|": text.ljust(5010) + "|",
        "%05010d": sign + "0" * (5010 - len(text)) + DIGITS,
        "%.5005d": sign + "00000" + DIGITS,
        "%+.5005d": plus + "00000" + DIGITS,
        "%.d": text,
        "%10.3s": text[:3].rjust(10),
        "%(k)d": text,
        "%s and %d": text + " and 3",
        "%*d+": text.rjust(5010),
        "%*d-": text.ljust(5010),
        "%%%d": "%" + text,
        "%x": sign + format(abs(x), "x"),
    }[fmt]


# (format, label, args builder); the label tells the two `%*d` rows apart.
_FORMATS = [
    *[(fmt, fmt, lambda x: x) for fmt in (
        "%s", "%r", "%a", "%d", "%i", "%u", "%+d", "% d", "%5010d", "%-5010d|",
        "%05010d", "%.5005d", "%+.5005d", "%.d", "%10.3s", "%%%d", "%x",
    )],
    ("%(k)d", "%(k)d", lambda x: {"k": x}),
    ("%s and %d", "%s and %d", lambda x: (x, 3)),
    ("%*d", "%*d+", lambda x: (5010, x)),
    ("%*d", "%*d-", lambda x: (-5010, x)),
]

_ERRORS = [
    ("%5%", lambda x: (x,)),
    ("%(k)s %(j)d", lambda x: {"k": x}),
    ("%s %s", lambda x: ("a", x, "c")),
    ("%d %d", lambda x: (x,)),
    ("%(k)s", lambda x: (x,)),
]


@pytest.mark.parametrize("x", [BIG, -BIG], ids=["big", "negative-big"])
@pytest.mark.parametrize(("fmt", "label", "args"), _FORMATS, ids=[f[1] for f in _FORMATS])
def test_format_text_writes_a_long_int_in_full(fmt, label, args, x):
    assert format_text(fmt, args(x)) == _expected(label, x)


@pytest.mark.parametrize("x", [SMALL, -SMALL], ids=["small", "negative-small"])
@pytest.mark.parametrize(("fmt", "label", "args"), _FORMATS, ids=[f[1] for f in _FORMATS])
def test_format_text_matches_python_for_short_ints(fmt, label, args, x):
    assert format_text(fmt, args(x)) == fmt % args(x)


def _python_error(fmt, args):
    with pytest.raises(Exception) as info:
        fmt % args
    return type(info.value), str(info.value)


@pytest.mark.parametrize(("fmt", "args"), _ERRORS, ids=[e[0] for e in _ERRORS])
def test_format_text_raises_pythons_error(fmt, args):
    expected_type, expected_text = _python_error(fmt, args(SMALL))

    for x in (SMALL, BIG):
        with pytest.raises(Exception) as info:
            format_text(fmt, args(x))
        assert (type(info.value), str(info.value)) == (expected_type, expected_text)


def test_format_text_keeps_a_list_as_the_mapping_and_the_single_argument():
    with pytest.raises(TypeError, match="^not enough arguments for format string$"):
        format_text("%s %s", [BIG])
    assert format_text("%s", [BIG]) == "[" + DIGITS + "]"
    assert format_text("abc", [BIG]) == "abc"


def test_a_long_int_that_still_cannot_be_written_is_reported_without_pythons_hint():
    with pytest.raises(ExpressionError) as info:
        eval_expr(parse_expr('"%s" % [s]'), lambda path: {BIG} if path == "s" else None)

    message = str(info.value)
    assert "an int too long to format" in message
    assert "set_int_max_str_digits" not in message


_FLOW = """
id: big-percent
name: big_percent
input:
  x: !int
  r: !object
nodes:
  show:
    kind: code
    input:
      text: '${EXPR}'
    code: |
      return inputs["text"]
output: ${show.output}
"""


def _run(expression: str):
    flow = load_flow(_FLOW.replace("EXPR", expression))
    return run_flow(flow, {"x": BIG, "r": {"k": BIG}})


@pytest.mark.parametrize(
    ("expression", "output"),
    [
        ('"%d" % input.x', DIGITS),
        ('"v=%s" % input.x', "v=" + DIGITS),
        ('"%s" % [input.x]', "[" + DIGITS + "]"),
        ('"%(k)d" % input.r', DIGITS),
    ],
    ids=["d", "s", "list", "key"],
)
def test_a_flow_formats_a_long_int_with_percent(expression, output):
    result = _run(expression)

    assert result.status == "succeeded", result.failure
    assert result.output == output


@pytest.mark.parametrize(
    ("expression", "detail"),
    [
        ('"%s and %d" % [input.x, 3]', "not enough arguments for format string"),
        ('"%f" % input.x', "int too large to convert to float"),
    ],
    ids=["list-is-a-mapping", "float-overflow"],
)
def test_a_flow_reports_pythons_percent_error(expression, detail):
    result = _run(expression)

    assert result.status == "failed"
    assert result.failure.code == "binding.evaluate"
    message = result.failure.message
    assert detail in message
    assert "set_int_max_str_digits" not in message
    assert re.search(r"[0-9]{600}", message) is None
