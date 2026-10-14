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
