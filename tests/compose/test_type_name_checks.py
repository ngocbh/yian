"""Empty or non-UTF-8 Literal tags and record field names fail the load the same way
wherever the type is used: as an output, as an input, or through a typedef."""

from __future__ import annotations

import pytest

from yian.compose.helpers.errors import LoadError
from yian.compose.loader import load_flow


def _flow(*, typedefs: str = "", input: str = "  n: !int", output: str = "str") -> str:
    """Return a one-node code flow with the given typedefs, input, and output blocks."""
    block = f"typedefs:\n{typedefs}\n" if typedefs else ""
    return f"id: e\nname: e\n{block}input:\n{input}\noutput: {output}\nkind: code\ncode: |\n  return 1\n"


_EMPTY_TAG = "empty Literal member"
_EMPTY_FIELD = "field names must be non-empty strings"
_SURROGATE = "Unicode surrogate"

CASES = {
    "literal-output": (_flow(output="'Literal(\"\", a)'"), _EMPTY_TAG),
    "literal-input": (_flow(input="  m: !Literal('')"), _EMPTY_TAG),
    "literal-typedef-input": (
        _flow(typedefs="  Mode: 'Literal(\"\", fast)'", input="  m: !Mode"),
        _EMPTY_TAG,
    ),
    "literal-surrogate-output": (_flow(output="'Literal(\"\\uD800\", a)'"), _SURROGATE),
    # A tag can't spell a surrogate escape, so the input names a typedef.
    "literal-surrogate-input": (
        _flow(typedefs="  Mode: 'Literal(\"\\uD800\", a)'", input="  m: !Mode"),
        _SURROGATE,
    ),
    "typedef-field-output": (_flow(typedefs='  R:\n    "": int', output="R"), _EMPTY_FIELD),
    "typedef-field-input": (
        _flow(typedefs='  R:\n    "": int', input="  r: !R"),
        _EMPTY_FIELD,
    ),
    "typedef-field-surrogate": (
        _flow(typedefs='  R:\n    "\\uD800": int', input="  r: !R"),
        _SURROGATE,
    ),
    "inline-field-surrogate-output": (
        _flow(output='\n  "\\uD800": int'),
        _SURROGATE,
    ),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_bad_tag_or_field_name_is_a_load_error(case):
    """Each bad name raises a `LoadError` naming the problem, never a raw TypeError."""
    text, message = CASES[case]
    with pytest.raises(LoadError, match=message):
        load_flow(text)
