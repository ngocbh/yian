"""A text or expression field YAML reads as non-text is a load error at that value."""

from __future__ import annotations

import pytest

from yian.compose import load_flow
from yian.compose.helpers.errors import LoadError
from yian.compose.nodes.base import not_text_error

NON_TEXT = [
    ("5", "an integer"),
    ("0", "an integer"),
    ("true", "a boolean"),
    ("false", "a boolean"),
    ("1.5", "a float"),
    ("2026-10-14", "a date"),
    ("[a]", "a list"),
    ("[]", "a list"),
    ("{x: 1}", "a mapping"),
    ("{}", "a mapping"),
]


def _error(text: str) -> LoadError:
    """Load `text`, expecting a `LoadError` (not a raw `TypeError`/`KeyError`)."""
    with pytest.raises(LoadError) as caught:
        load_flow(text)
    return caught.value


def _line(text: str, marker: str) -> int:
    """The 1-based line of `text` holding `marker` (the first match)."""
    for number, line in enumerate(text.splitlines(), start=1):
        if marker in line:
            return number
    raise AssertionError(f"{marker!r} not in text")


def _start_line(error: LoadError) -> int:
    return error.diagnostic.frames[-1].range.start_line


def test_not_text_error_names_the_field():
    assert not_text_error("prompt", 5) == (
        "`prompt:` must be text, but YAML read it as an integer; put it in quotes"
    )
    assert not_text_error("questions[0].question", [1]) == (
        "`questions[0].question:` must be text, but YAML read it as a list; put it in quotes"
    )
    assert not_text_error("questions[0].header", None) == (
        "`questions[0].header:` must be text, but it is empty; write the text"
    )


def test_not_text_error_takes_a_fix():
    assert not_text_error("kind", [1], fix="write one of a, b") == (
        "`kind:` must be text, but YAML read it as a list; write one of a, b"
    )
    assert not_text_error("kind", None, fix="write one of a, b") == (
        "`kind:` must be text, but it is empty; write one of a, b"
    )


def _code(v: str) -> str:
    """A flow whose CODE node `a` has `code: <v>`."""
    return """\
id: f
name: f
nodes:
  a:
    kind: code
    output: str
    code: @V
output: ${a.output}
""".replace("@V", v)


def _tool(v: str) -> str:
    """A flow whose TOOL node `a` has `tool_id: <v>`."""
    return """\
id: f
name: f
nodes:
  a:
    kind: tool
    tool_id: @V
    input: {symbol: x}
output: ${a.output}
""".replace("@V", v)


@pytest.mark.parametrize(("value", "phrase"), NON_TEXT)
def test_non_text_code_is_a_load_error(value, phrase):
    text = _code(value)
    error = _error(text)
    assert (
        f"node 'a' (kind=code): `code:` must be text, but YAML read it as {phrase}; "
        "put it in quotes"
    ) in str(error)
    assert _start_line(error) == _line(text, "code: ")


@pytest.mark.parametrize(("value", "phrase"), NON_TEXT)
def test_non_text_tool_id_is_a_load_error(value, phrase):
    text = _tool(value)
    error = _error(text)
    assert (
        f"node 'a' (kind=tool): `tool_id:` must be text, but YAML read it as {phrase}; "
        "put it in quotes"
    ) in str(error)
    assert _start_line(error) == _line(text, "tool_id: ")


@pytest.mark.parametrize(
    ("text", "kind", "key"),
    [(_code("null"), "code", "code"), (_tool("null"), "tool", "tool_id")],
    ids=["code", "tool_id"],
)
def test_null_code_and_tool_id_are_empty(text, kind, key):
    error = _error(text)
    assert (
        f"node 'a' (kind={kind}): `{key}:` must be text, but it is empty; write the text"
    ) in str(error)
    assert _start_line(error) == _line(text, f"{key}: ")


def test_tagged_non_text_tool_id():
    # `!str 5` still reads as an integer here; the tag's own error comes once it is quoted.
    error = _error(_tool("!str 5"))
    assert "`tool_id:` must be text, but YAML read it as an integer; put it in quotes" in str(
        error
    )


_CODE_IN_DEF = """\
id: f
name: f
defs:
  body:
    nodes:
      one:
        kind: code
        output: str
        code: tests.engine._compose_codefns:echo
      two:
        kind: code
        output: str
        code: 5
    output: ${two.output}
nodes:
  a:
    kind: call
    call: body
output: ${a.output}
"""

_CODE_COMPACT = """\
id: f
name: f
output: str
kind: code
code: 5
"""

_CODE_MERGED = """\
id: f
name: f
x-base: &base {code: 5}
nodes:
  a:
    <<: *base
    kind: code
    output: str
output: ${a.output}
"""

_CODE_INLINE_BODY = """\
id: f
name: f
nodes:
  a:
    kind: call
    nodes:
      echo:
        kind: code
        output: str
        code: 5
    output: ${echo.output}
output: ${a.output}
"""


@pytest.mark.parametrize(
    ("text", "node", "line"),
    [
        (_CODE_IN_DEF, "two", _line(_CODE_IN_DEF, "code: 5")),
        (_CODE_COMPACT, "f", _line(_CODE_COMPACT, "code: 5")),
        (_CODE_MERGED, "a", _line(_CODE_MERGED, "x-base:")),
        (_CODE_INLINE_BODY, "echo", _line(_CODE_INLINE_BODY, "code: 5")),
    ],
    ids=["def", "compact", "merge", "inline-body"],
)
def test_code_in_def_compact_merge_and_inline_body(text, node, line):
    error = _error(text)
    assert f"node {node!r} (kind=code): `code:` must be text" in str(error)
    assert _start_line(error) == line


@pytest.mark.parametrize("text", [_code("\"return 'x'\""), _tool("'5'")])
def test_text_code_and_tool_id_still_load(text):
    load_flow(text)
