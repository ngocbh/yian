"""JSON text and plain text for values that hold ints of any size.

Python limits how many digits an int may have when it is converted to or from
text (`sys.set_int_max_str_digits`). That limit is process-wide, so yian never
changes it; this module converts long ints itself and otherwise defers to the
standard `json` module for JSON text and to `str` for plain text. Converting a
long int either way takes time quadratic in its digit count (about 0.2 s each
way at 100,000 digits), and yian sets no limit on that cost. This module
imports only the standard library and sits below every yian package.
"""

from __future__ import annotations

from decimal import Decimal
import json
import re
from typing import Any, Callable


# 2000 bits is about 603 decimal digits and 600 characters hold at most 600
# digits, both below the smallest digit limit Python allows (640), so
# `int.__repr__` and `int()` can never hit the limit under these sizes.
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
        text (`str`): Text of up to 600 characters is parsed by `int()`; longer
            text must be ASCII digits with an optional sign and `_` groups,
            with optional surrounding whitespace.

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
    encoder, which recurses in Python for each nesting level, writes the same text.
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
        `RecursionError`: The value holds a long int and is nested near the
            recursion limit; callers that bound nesting map this to their depth
            error.
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

    Documents without long ints take `json.loads` directly. When that raises a
    `ValueError` other than `json.JSONDecodeError`, the document is parsed
    again with `parse_int`, so hooks such as `object_pairs_hook` may run twice
    for the same objects and should have no side effects.

    Args:
        data (`str | bytes`): The JSON document.
        **kwargs: Further `json.loads` options, such as `object_pairs_hook`; an
            explicit `parse_int` replaces this module's and disables the retry.

    Returns:
        `Any`: The parsed value.

    Raises:
        `json.JSONDecodeError`: The document is not valid JSON.
        `ValueError`: A hook rejected a value.
    """
    if "parse_int" in kwargs:
        return json.loads(data, **kwargs)
    try:
        return json.loads(data, **kwargs)
    except json.JSONDecodeError:
        # Malformed text fails at the same place on a second pass.
        raise
    except ValueError:
        # Most likely the digit limit; a hook's own error is raised again below.
        return json.loads(data, parse_int=parse_int, **kwargs)


class _IntDigits:
    """Stands in for a long int when printing; its `repr` is the int's digits."""

    def __init__(self, value: int) -> None:
        self._text = int_text(value)

    def __repr__(self) -> str:
        return self._text


def displayable(value: Any) -> Any:
    """Return `value` with every long int swapped for an object whose `repr` is its digits.

    `repr` and `str` of an int past Python's process-wide digit limit raise
    `ValueError`; the returned value prints instead. Dicts (keys and values),
    lists and tuples are walked and rebuilt, as their base type, only when they
    hold a long int; any other value is returned unchanged.

    Args:
        value (`Any`): The value to print.

    Returns:
        `Any`: `value` itself, or a copy whose long ints print as their digits.
    """
    if isinstance(value, int) and not isinstance(value, bool):
        return _IntDigits(value) if value.bit_length() > _SHORT_INT_BITS else value
    if isinstance(value, dict):
        pairs = [(displayable(key), displayable(item)) for key, item in value.items()]
        if all(
            new_key is key and new_item is item
            for (new_key, new_item), (key, item) in zip(pairs, value.items())
        ):
            return value
        return dict(pairs)
    if isinstance(value, (list, tuple)):
        items = [displayable(item) for item in value]
        if all(new is old for new, old in zip(items, value)):
            return value
        return items if isinstance(value, list) else tuple(items)
    return value


def as_text(value: Any) -> str:
    """Return `str(value)`, writing ints of any size in full.

    Values that `str` converts take it directly. When `str` raises
    `ValueError`, the value is converted again through `displayable`, so a long
    int, alone or inside dicts (keys included), lists and tuples, is written with
    every digit.

    Args:
        value (`Any`): The value to convert.

    Returns:
        `str`: The text `str(value)` would give without the digit limit; on the
            retry, a dict, list or tuple subclass is written as its base type.

    Raises:
        `ValueError`: `str` raised for a value that `displayable` can't walk,
            such as a long int inside a set or another object.

    Example:
        ```python
        >>> as_text({"n": 10**5000})[:8]
        "{'n': 10"
        ```
    """
    try:
        return str(value)
    except ValueError:
        # Most likely the digit limit; any other ValueError is raised again here.
        return str(displayable(value))


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
    scalar_text = json.JSONEncoder(ensure_ascii=ensure_ascii, allow_nan=allow_nan).encode
    markers: set[int] = set()
    parts: list[str] = []

    def key_text(key: Any) -> str:
        if isinstance(key, str):
            return key
        if isinstance(key, float) or key is True or key is False or key is None:
            return scalar_text(key)
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
            parts.append(scalar_text(obj))
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
                parts.append(scalar_text(key_text(key)) + key_separator)
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
