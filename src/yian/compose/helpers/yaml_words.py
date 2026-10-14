"""YAML 1.1 spellings that the flow loader's YAML 1.2 core rules read as plain text, and
names for what YAML read a non-text value as."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from yian.typesys import Type, TypeCheckError, build_value_as

_BOOLEAN_WORDS = {"yes": True, "y": True, "on": True, "no": False, "n": False, "off": False}
_MISSING = object()


def former_boolean_fix(value: Any, declared: Type | None = None) -> tuple[bool, str] | None:
    """Return the boolean a YAML 1.1 boolean spelling meant and the fix text, else `None`.

    Matching ignores case and surrounding whitespace; the fix shows the stripped text as
    written. With a declared type, a list or record is searched too: the word is the
    value the type check rejects, and the fix is offered when its boolean clears that
    rejection.

    Args:
        value (`Any`):
            A value the loader read, typically one that failed a `bool` slot.
        declared (`Type | None`, *optional*, defaults to `None`):
            The slot's type. When given, the fix is offered only if the boolean fits it.

    Returns:
        `tuple[bool, str] | None`:
            The boolean the spelling meant and the fix to show, or `None` when no
            `yes`/`no`/`on`/`off`/`y`/`n` is at fault or its boolean does not fit
            `declared`.

    Example:
        `former_boolean_fix("yes")` is `(True, "`yes` is text; write `true`")`;
        `former_boolean_fix("true")` is `None`.
    """
    if isinstance(value, str):
        fix = _word_fix(value)
        if fix is None or declared is None or _rejection(declared, fix[0]) is None:
            return fix
        return None
    if declared is None or not isinstance(value, (list, dict)):
        return None
    rejected = _rejection(declared, value)
    if rejected is None:
        return None
    leaf = _at(value, rejected.path)
    fix = _word_fix(leaf) if rejected.path else None
    if fix is None:
        return None
    again = _rejection(declared, _replaced(value, rejected.path, fix[0]))
    if again is not None and again.path == rejected.path:
        return None
    return fix


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


def _at(value: Any, path: tuple[str | int, ...]) -> Any:
    """Return the item a type-check path names inside `value`, or a sentinel."""
    for segment in path:
        if isinstance(value, dict) and segment in value:
            value = value[segment]
        elif isinstance(value, list) and type(segment) is int and 0 <= segment < len(value):
            value = value[segment]
        else:
            return _MISSING
    return value


def _replaced(value: Any, path: tuple[str | int, ...], leaf: Any) -> Any:
    """Return a copy of `value` with the item at `path` replaced by `leaf`."""
    if not path:
        return leaf
    head, rest = path[0], path[1:]
    if isinstance(value, dict):
        return {**value, head: _replaced(value[head], rest, leaf)}
    copy = list(value)
    copy[head] = _replaced(value[head], rest, leaf)
    return copy
