"""Text formatting with `%` that writes ints of any size.

Python's `%` honors the process-wide int/str digit limit
(`sys.set_int_max_str_digits`), which yian never changes, so `"%d" % n` raises
`ValueError` once `n` has more digits than the limit. `format_text` writes the
text Python's `%` would write without that limit, and gives Python's own
result or error everywhere else. This module imports only the standard library
and `yian.json_codec`.
"""

from __future__ import annotations

import re
from typing import Any

import yian.json_codec as json_codec


# The rest of CPython's conversion spec after `%` and any `(key)`: flags, width,
# precision, an ignored length modifier, then the conversion character. CPython
# reads only ASCII digits in a width or precision.
_SPEC = re.compile(
    r"(?P<flags>[-+ #0]*)(?P<width>\*|[0-9]+)?(?:\.(?P<prec>\*|[0-9]*))?[hlL]?(?P<conv>.)",
    re.DOTALL,
)
_CONVERSIONS = frozenset("sradiuoxXeEfFgGc")
# A short int that still overflows every C-sized field `%` reads (a `*` width or
# precision, `%c`), so Python raises the same error for it as for a long int.
_STAND_IN = 1 << 64


class _Mismatch(Exception):
    """The arguments don't fit the format; Python's `%` reports how."""


class _Arguments:
    """Hands out the right side of `%` in the order CPython binds it.

    A tuple supplies positional arguments. Any other object is the single
    positional argument, and is also the mapping for `%(key)` specs when it
    is not a `str` and has `__getitem__` (so a list is a mapping, as in CPython).
    """

    def __init__(self, args: Any) -> None:
        self._mapping = (
            args
            if not isinstance(args, (tuple, str)) and hasattr(type(args), "__getitem__")
            else None
        )
        if isinstance(args, tuple):
            self._args, self._length, self._index = args, len(args), 0
        else:
            self._bind_one(args)

    def next(self) -> Any:
        """Return the next positional argument.

        Raises:
            `_Mismatch`: No argument is left.
        """
        if self._index >= self._length:
            raise _Mismatch
        self._index += 1
        return self._args if self._length < 0 else self._args[self._index - 1]

    def select(self, key: str) -> None:
        """Make the mapping's value at `key` the only argument left.

        Raises:
            `_Mismatch`: The right side is not a mapping.
            `KeyError`: The mapping has no such key (or the mapping's own error).
        """
        if self._mapping is None:
            raise _Mismatch
        self._bind_one(self._mapping[key])

    def _bind_one(self, value: Any) -> None:
        """Make `value` the only argument left."""
        # CPython's arglen/argidx for a lone argument: one `next` takes it
        # (-2 < -1), a second finds none (-1 is not < -1).
        self._args, self._length, self._index = value, -1, -2

    def all_used(self) -> bool:
        """Return whether formatting may end here without leaving arguments unused."""
        return self._index >= self._length or self._mapping is not None


def format_text(fmt: str, args: Any) -> str:
    """Return `fmt % args` as Python writes it without the int/str digit limit.

    `%s`, `%r` and `%a` write long ints, alone or inside dicts, lists and
    tuples, with every digit; `%d`, `%i` and `%u` write a long int with every
    digit and honor flags, width and precision. Every other conversion, and
    every error (a missing key, too few or too many arguments, a wrong argument
    type), is Python's own.

    Args:
        fmt (`str`): The format string.
        args (`Any`): The right side of `%`.

    Returns:
        `str`: The formatted text.

    Raises:
        `TypeError`, `ValueError`, `KeyError`, `OverflowError`: The error
            Python's `%` raises for this format and these arguments.

    Example:
        ```python
        >>> format_text("%d", 10**5000)[:4]
        '1000'
        ```
    """
    arguments = _Arguments(args)
    parts: list[str] = []
    position = 0
    try:
        while (start := fmt.find("%", position)) != -1:
            parts.append(fmt[position:start])
            if fmt.startswith("%%", start):
                parts.append("%")
                position = start + 2
                continue
            key, spec_start = _read_key(fmt, start + 1)
            spec = _SPEC.match(fmt, spec_start)
            if spec is None or spec["conv"] not in _CONVERSIONS:
                raise _Mismatch
            parts.append(_convert(key, spec, arguments))
            position = spec.end()
        parts.append(fmt[position:])
        if not arguments.all_used():
            raise _Mismatch
    except _Mismatch:
        return _python_format(fmt, args)
    return "".join(parts)


def _read_key(fmt: str, start: int) -> tuple[str | None, int]:
    """Return the `(key)` at `start`, read as CPython does, and the index after it.

    Parentheses nest: the key ends at the `)` that balances the opening one.
    With no `(` at `start`, the key is `None` and the index is `start`.

    Raises:
        `_Mismatch`: The key's parentheses never balance.
    """
    if not fmt.startswith("(", start):
        return None, start
    depth = 0
    for index in range(start, len(fmt)):
        if fmt[index] == "(":
            depth += 1
        elif fmt[index] == ")":
            depth -= 1
            if depth == 0:
                return fmt[start + 1 : index], index + 1
    raise _Mismatch


def _convert(key: str | None, spec: re.Match[str], arguments: _Arguments) -> str:
    """Return the text of one conversion spec, binding its arguments."""
    if key is not None:
        arguments.select(key)
    stars = [arguments.next() for part in (spec["width"], spec["prec"]) if part == "*"]
    value = arguments.next()
    precision = "" if spec["prec"] is None else "." + spec["prec"]
    python_spec = "%" + spec["flags"] + (spec["width"] or "") + precision + spec["conv"]
    if spec["conv"] in "sra":
        return python_spec % (*stars, json_codec.displayable(value))
    if spec["conv"] in "diu" and _is_long_int(value):
        # Python checks the `*` arguments, width and precision on a short int.
        python_spec % (*stars, 0)
        return _long_int_text(value, spec, stars)
    return python_spec % (*stars, value)


def _long_int_text(value: int, spec: re.Match[str], stars: list[int]) -> str:
    """Return `%d` text for a long int, as Python writes it without the digit limit."""
    flags = spec["flags"]
    star_values = iter(stars)
    width = next(star_values) if spec["width"] == "*" else int(spec["width"] or 0)
    left = "-" in flags or width < 0
    width = abs(width)
    if spec["prec"] is None:
        precision = 0
    elif spec["prec"] == "*":
        precision = next(star_values)
    else:
        precision = int(spec["prec"] or 0)
    digits = json_codec.int_text(abs(value))
    digits = "0" * (precision - len(digits)) + digits
    sign = "-" if value < 0 else "+" if "+" in flags else " " if " " in flags else ""
    padding = width - len(sign) - len(digits)
    if padding <= 0:
        return sign + digits
    if left:
        return sign + digits + " " * padding
    if "0" in flags:
        return sign + "0" * padding + digits
    return " " * padding + sign + digits


def _is_long_int(value: Any) -> bool:
    """Return whether `value` is an int too long for Python to write as text."""
    return isinstance(value, int) and json_codec.displayable(value) is not value


def _python_format(fmt: str, args: Any) -> str:
    """Return `fmt % args`, raising the error Python raises without the digit limit."""
    # With long ints swapped for short ones, every conversion this module already
    # wrote still succeeds, so Python raises the error that follows them rather
    # than failing on the digit limit first.
    fmt % _masked(args)
    return fmt % args


def _masked(value: Any) -> Any:
    """Return `value` with every long int, in dicts (keys too), lists and tuples, made short."""
    if _is_long_int(value):
        return _STAND_IN if value > 0 else -_STAND_IN
    if isinstance(value, dict):
        return {_masked(key): _masked(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_masked(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_masked(item) for item in value)
    return value
