"""JSON text for values that hold ints of any size.

Python limits how many digits an int may have when it is converted to or from
text (`sys.set_int_max_str_digits`). That limit is process-wide, so yian never
changes it; this module converts long ints itself and otherwise defers to the
standard `json` module. Converting a long int either way takes time quadratic
in its digit count (about 0.2 s each way at 100,000 digits), and yian sets no
limit on that cost. This module imports only the standard library and sits
below every yian package.
"""

from __future__ import annotations

from decimal import Decimal
import json
import re
from typing import Any, Callable


# 2000 bits is about 603 decimal digits, below the smallest digit limit Python
# allows (640), so `int.__repr__` can never hit the limit under this size.
_SHORT_INT_BITS = 2000
_SHORT_INT_TEXT = 600
_INT_LITERAL = re.compile(r"[+-]?[0-9]+(?:_[0-9]+)*")


def int_text(value: int) -> str:
    """Return the decimal text of an int of any size.

    Args:
        value (`int`): The int to write; an int subclass is written as its int value.

    Returns:
        `str`: The value's decimal digits, with a leading `-` when negative.
    """
    if value.bit_length() <= _SHORT_INT_BITS:
        return int.__repr__(value)
    # Decimal converts ints and writes its text without the digit limit;
    # its exponent is 0, so the text never uses scientific notation.
    return str(Decimal(value))


def parse_int(text: str) -> int:
    """Return the int that decimal text spells, at any length.

    Args:
        text (`str`): Text in `int()`'s base-10 grammar: an optional sign, digits
            in `_`-separated groups, and surrounding whitespace.

    Returns:
        `int`: The parsed value.

    Raises:
        `ValueError`: The text is not a base-10 int literal.
    """
    if len(text) <= _SHORT_INT_TEXT:
        return int(text)
    stripped = text.strip()
    if _INT_LITERAL.fullmatch(stripped) is None:
        # Unlike int(), don't echo text that may be megabytes long.
        raise ValueError("invalid literal for int() with base 10")
    return int(Decimal(stripped.replace("_", "")))


def dumps(
    value: Any,
    *,
    ensure_ascii: bool = True,
    allow_nan: bool = True,
    sort_keys: bool = False,
    separators: tuple[str, str] | None = None,
    indent: int | str | None = None,
    default: Callable[[Any], Any] | None = None,
) -> str:
    """Return the same JSON text as `json.dumps`, writing ints of any size in full.

    Values without long ints take `json.dumps` directly. Otherwise a slower
    encoder writes the same text with one Python frame per nesting level, so a
    value nested near the recursion limit that also holds a long int raises
    `RecursionError`; callers that bound nesting map that to their depth error.
    When the fast path raises `ValueError`, `default` runs again for the same
    objects, so it should have no side effects.

    Args:
        value (`Any`): The value to encode.
        ensure_ascii (`bool`, *optional*, defaults to `True`): Escape non-ASCII text.
        allow_nan (`bool`, *optional*, defaults to `True`): Write NaN and infinities
            instead of raising `ValueError`.
        sort_keys (`bool`, *optional*, defaults to `False`): Sort object keys.
        separators (`tuple[str, str] | None`, *optional*, defaults to `None`): Item
            and key separators; `None` uses json's defaults for `indent`.
        indent (`int | str | None`, *optional*, defaults to `None`): Pretty-print
            with this many spaces, or this string, per level.
        default (`Callable[[Any], Any] | None`, *optional*, defaults to `None`):
            Converts objects json can't encode.

    Returns:
        `str`: The JSON text.

    Raises:
        `ValueError`: The value holds a disallowed NaN or infinity, or a cycle.
        `TypeError`: The value holds an object or key json can't encode.
    """
    options = dict(
        ensure_ascii=ensure_ascii,
        allow_nan=allow_nan,
        sort_keys=sort_keys,
        separators=separators,
        indent=indent,
        default=default,
    )
    try:
        return json.dumps(value, **options)
    except ValueError:
        # Most likely the digit limit; any other ValueError is raised again below.
        return _encode(value, **options)


def loads(data: str | bytes, **kwargs: Any) -> Any:
    """Parse JSON text as `json.loads` does, reading ints of any size.

    Args:
        data (`str | bytes`): The JSON document.
        **kwargs: Further `json.loads` options, such as `object_pairs_hook`.

    Returns:
        `Any`: The parsed value.
    """
    return json.loads(data, parse_int=parse_int, **kwargs)


def _encode(
    value: Any,
    *,
    ensure_ascii: bool = True,
    allow_nan: bool = True,
    sort_keys: bool = False,
    separators: tuple[str, str] | None = None,
    indent: int | str | None = None,
    default: Callable[[Any], Any] | None = None,
) -> str:
    """Return `json.dumps(value, ...)`'s exact text, writing ints with `int_text`."""
    if separators is None:
        separators = (", ", ": ") if indent is None else (",", ": ")
    item_separator, key_separator = separators
    if isinstance(indent, int):
        indent = " " * indent
    markers: set[int] = set()
    parts: list[str] = []

    def key_text(key: Any) -> str:
        if isinstance(key, str):
            return key
        if isinstance(key, float) or key is True or key is False or key is None:
            return json.dumps(key, allow_nan=allow_nan)
        if isinstance(key, int):
            return int_text(key)
        raise TypeError(
            f"keys must be str, int, float, bool or None, not {key.__class__.__name__}"
        )

    def enter(obj: Any) -> None:
        if id(obj) in markers:
            raise ValueError("Circular reference detected")
        markers.add(id(obj))

    def write(obj: Any, level: int) -> None:
        if isinstance(obj, (str, float)) or obj is None or obj is True or obj is False:
            parts.append(json.dumps(obj, ensure_ascii=ensure_ascii, allow_nan=allow_nan))
        elif isinstance(obj, int):
            parts.append(int_text(obj))
        elif isinstance(obj, (list, tuple, dict)):
            write_container(obj, level)
        elif default is None:
            raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")
        else:
            enter(obj)
            write(default(obj), level)
            markers.discard(id(obj))

    def write_container(obj: list | tuple | dict, level: int) -> None:
        is_dict = isinstance(obj, dict)
        if not obj:
            parts.append("{}" if is_dict else "[]")
            return
        enter(obj)
        if indent is None:
            opening, between, closing = "", item_separator, ""
        else:
            newline = "\n" + indent * (level + 1)
            opening, between, closing = newline, item_separator + newline, "\n" + indent * level
        parts.append(("{" if is_dict else "[") + opening)
        if is_dict:
            items = sorted(obj.items()) if sort_keys else obj.items()
            for index, (key, item) in enumerate(items):
                if index:
                    parts.append(between)
                parts.append(json.dumps(key_text(key), ensure_ascii=ensure_ascii) + key_separator)
                write(item, level + 1)
        else:
            for index, item in enumerate(obj):
                if index:
                    parts.append(between)
                write(item, level + 1)
        parts.append(closing + ("}" if is_dict else "]"))
        markers.discard(id(obj))

    write(value, 0)
    return "".join(parts)
