"""An int literal with more than 4300 digits in a flow file is a located load error."""

from __future__ import annotations

import re
import sys

import pytest

from yian.compose import LoadError, load_flow
from yian.run import run_flow
from yian.typesys.types import parse_type
from yian.typesys.values import TypeCheckError, int_digits, long_int_message

D = "1" * 5000
_ONES = (10**5000 - 1) // 9  # int(D), built without str->int conversion
LIMIT_MSG = "integer with 5000 digits is larger than yian allows (4300 digits)"

_HEAD = "id: f\nname: f\n"


def _node(binding: str) -> str:
    """A `use` node binding `topic` to `binding`; the binding is on line 8 of `_flow`."""
    return (
        "  use:\n    kind: code\n    code: tests.engine._compose_codefns:echo\n"
        f"    input:\n      topic: {binding}\n    output: str\n"
    )


def _flow(binding: str = "hi", *, inputs: str = "") -> str:
    """A flow whose `use` node binds `binding`, after an optional `input:` block."""
    return _HEAD + inputs + "nodes:\n" + _node(binding) + "output: ${use.output}\n"


def _load_error(text: str) -> LoadError:
    """Load `text`, expecting a load error whose message shows no long digits."""
    with pytest.raises(LoadError) as caught:
        load_flow(text)
    message = str(caught.value)
    assert "set_int_max_str_digits" not in message
    assert re.search(r"[0-9]{600}", message) is None
    return caught.value


def _line(error: LoadError) -> int:
    """The 1-based line the error points at."""
    return error.diagnostic.frames[-1].range.start_line


# --- YAML int literals ------------------------------------------------------ #


@pytest.mark.parametrize(
    ("text", "line", "value"),
    [
        (_flow(inputs=f"input:\n  n: !int {D}\n"), 4, _ONES),
        (_flow(inputs=f"input:\n  n:\n    type: int\n    default: {D}\n"), 6, _ONES),
        (_flow(D), 8, _ONES),
        (_flow(f"!int {D}"), 8, _ONES),
        (_HEAD + f"nodes:\n  v: !int {D}\noutput: ${{v.output}}\n", 4, _ONES),
        (_flow(f"-{D}"), 8, _ONES),
        (_flow("0x" + "f" * 4000), 8, int("f" * 4000, 16)),
        (_flow("0o" + "7" * 5000), 8, int("7" * 5000, 8)),
    ],
    ids=[
        "input-default",
        "input-default-long-form",
        "bare-binding",
        "tagged-binding",
        "value-entry",
        "negative",
        "hex",
        "octal",
    ],
)
def test_a_long_int_literal_is_a_located_load_error(text, line, value):
    error = _load_error(text)

    assert long_int_message(int_digits(value)) in str(error)
    assert _line(error) == line


def test_leading_zeros_do_not_count():
    text = _HEAD + "nodes:\n  v: !int " + "0" * 4999 + "1\noutput: ${v.output}\n"
    result = run_flow(load_flow(text), {})

    assert result.status == "succeeded", result.failure
    assert result.output == 1


@pytest.mark.skipif(
    getattr(sys, "get_int_max_str_digits", None) is None, reason="this Python has no digit limit"
)
@pytest.mark.parametrize("binding", [D, "${" + D + "}"], ids=["yaml", "expression"])
def test_the_limit_holds_with_pythons_own_limit_off(binding):
    before = sys.get_int_max_str_digits()
    sys.set_int_max_str_digits(0)
    try:
        error = _load_error(_flow(binding))
    finally:
        sys.set_int_max_str_digits(before)

    assert LIMIT_MSG in str(error)


# --- range ends ------------------------------------------------------------- #


@pytest.mark.parametrize(
    "declared", [f"!int(0..{D})", f"!int(..-{D})", f"!float(0..{D})"], ids=["int", "neg", "float"]
)
def test_a_long_range_end_is_a_load_error(declared):
    error = _load_error(_flow(inputs=f"input:\n  n: {declared}\n"))

    assert LIMIT_MSG in str(error)


@pytest.mark.parametrize(
    ("end", "digits"),
    [(D, 5000), ("0x" + "f" * 4000, int_digits(int("f" * 4000, 16)))],
    ids=["decimal", "hex"],
)
def test_a_long_end_in_the_bracket_spelling_names_the_limit(end, digits):
    with pytest.raises(TypeCheckError) as caught:
        parse_type(f"int[0, {end}]")

    message = str(caught.value)
    assert long_int_message(digits) in message
    assert "set_int_max_str_digits" not in message
    assert re.search(r"[0-9]{600}", message) is None


# --- expression number tokens ----------------------------------------------- #


_PT = "typedefs:\n  Pt:\n    x: int\n"


@pytest.mark.parametrize(
    "text",
    [
        _flow("${" + D + " + 1}"),
        _flow("${" + D + "}"),
        _HEAD
        + "nodes:\n  gate:\n    kind: case\n    cases:\n"
        + "      - when: ${" + D + " > 0}\n        then: use\n"
        + _node("hi")
        + "output: ${use.output}\n",
        _HEAD
        + "nodes:\n  a:\n    kind: agent\n"
        + '    prompt: "${' + D + '}"\n    output: str\noutput: ${a.output}\n',
        _HEAD + _PT + "nodes:\n" + _node("${Pt(x=" + D + ")}") + "output: ${use.output}\n",
        _flow() + 'asserts:\n  - "' + D + ' > 0"\n',
        _flow().replace(
            "    output: str\n", "    output: str\n    asserts:\n      - " + D + " > 0\n"
        ),
    ],
    ids=[
        "sum",
        "lone",
        "case-when",
        "agent-prompt",
        "constructor-argument",
        "flow-assert",
        "node-assert",
    ],
)
def test_a_long_int_in_an_expression_is_a_load_error(text):
    error = _load_error(text)

    assert LIMIT_MSG in str(error)


def test_a_long_int_literal_in_inline_code_names_the_limit():
    # Python's compiler refuses it; the load error carries yian's message, not Python's hint.
    error = _load_error(f"""
id: f
name: f
nodes:
  c:
    kind: code
    code: |
      return {D}
output: ${{c.output}}
""")
    assert LIMIT_MSG in str(error)
