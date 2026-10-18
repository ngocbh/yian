"""YAML 1.1 spellings that the flow loader's YAML 1.2 core rules read as plain text, and
names for what YAML read a non-text value as."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from yian.typesys import Type, TypeCheckError, build_value_as

_BOOLEAN_WORDS = {"yes": True, "y": True, "on": True, "no": False, "n": False, "off": False}


def former_boolean_fix(value: Any, declared: Type | None = None) -> tuple[bool, str] | None:
    """Return the boolean a YAML 1.1 boolean spelling meant and the fix text, else `None`.

    Matching ignores case and surrounding whitespace; the fix shows the stripped text as
    written. Only a bare text value gets a fix; a list or record gets `None`, even when
    it holds such a word.

    Args:
        value (`Any`):
            A value the loader read, typically one that failed a `bool` slot.
        declared (`Type | None`, *optional*, defaults to `None`):
            The slot's type. When given, the fix is offered only if the boolean fits it.

    Returns:
        `tuple[bool, str] | None`:
            The boolean the spelling meant and the fix to show, or `None` when the
            value is not one `yes`/`no`/`on`/`off`/`y`/`n` or its boolean does not fit
            `declared`.

    Example:
        `former_boolean_fix("yes")` is `(True, "`yes` is text; write `true`")`;
        `former_boolean_fix("true")` is `None`.
    """
    fix = _word_fix(value)
    if fix is None or declared is None or _rejection(declared, fix[0]) is None:
        return fix
    return None


def yaml_read_as(value: Any) -> str:
    """Name what YAML read a value as, for an error message about a value that must be text.

    Args:
        value (`Any`):
            The constructed value, such as a mapping key or a field's value.

    Returns:
        `str`:
            A phrase such as ``a boolean``, ``a date``, or ``a list``.

    Example:
        `yaml_read_as(5)` is `"an integer"`; `yaml_read_as(None)` is `"null"`.
    """
    # A bool is an int and a datetime is a date, so the subclasses go first.
    if isinstance(value, bool):
        return "a boolean"
    if isinstance(value, int):
        return "an integer"
    if isinstance(value, float):
        return "a float"
    if isinstance(value, datetime):
        return "a date and time"
    if isinstance(value, date):
        return "a date"
    if value is None:
        return "null"
    if isinstance(value, bytes):
        return "binary data"
    if isinstance(value, list):
        return "a list"
    if isinstance(value, dict):
        return "a mapping"
    return f"a `{type(value).__name__}`"


def _word_fix(value: Any) -> tuple[bool, str] | None:
    """Return the boolean and fix text for one YAML 1.1 boolean word, else `None`."""
    if not isinstance(value, str):
        return None
    text = value.strip()
    flag = _BOOLEAN_WORDS.get(text.lower())
    if flag is None:
        return None
    return flag, f"`{text}` is text; write `{'true' if flag else 'false'}`"


def _rejection(declared: Type, value: Any) -> TypeCheckError | None:
    """Return the write-boundary check's error for `value`, or `None` when it fits."""
    try:
        build_value_as(declared, value)
    except TypeCheckError as exc:
        return exc
    return None
