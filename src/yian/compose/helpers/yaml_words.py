"""YAML 1.1 spellings that the flow loader's YAML 1.2 core rules read as plain text."""

from __future__ import annotations

from typing import Any

_BOOLEAN_WORDS = {"yes": True, "y": True, "on": True, "no": False, "n": False, "off": False}


def former_boolean_fix(value: Any) -> tuple[bool, str] | None:
    """Return the boolean a YAML 1.1 boolean spelling meant and the fix text, else `None`.

    Matching ignores case and surrounding whitespace; the fix shows the stripped text as
    written.

    Args:
        value (`Any`):
            A value the loader read, typically text that failed a `bool` slot.

    Returns:
        `tuple[bool, str] | None`:
            The boolean the spelling meant and the fix to show, or `None` when `value` is
            not text or not one of `yes`/`no`/`on`/`off`/`y`/`n`.

    Example:
        `former_boolean_fix("yes")` is `(True, "`yes` is text; write `true`")`;
        `former_boolean_fix("true")` is `None`.
    """
    if not isinstance(value, str):
        return None
    text = value.strip()
    flag = _BOOLEAN_WORDS.get(text.lower())
    if flag is None:
        return None
    return flag, f"`{text}` is text; write `{'true' if flag else 'false'}`"
