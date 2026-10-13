"""The one accepted string form of `date` and `datetime` values.

The grammar is spelled out here instead of delegated to `fromisoformat`, whose accepted
set changes between Python versions, so a string parses the same way on every interpreter.
"""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta, timezone

DATE_FORMAT = "2026-05-21"
DATETIME_FORMAT = "2026-05-21T14:30[:00][Z|+07:00]"

# `[0-9]` rather than `\d`, which also matches non-ASCII digits.
_DATE_RE = re.compile(r"([0-9]{4})-([0-9]{2})-([0-9]{2})")
_DATETIME_RE = re.compile(
    r"([0-9]{4})-([0-9]{2})-([0-9]{2})[T ]([0-9]{2}):([0-9]{2})"
    r"(?::([0-9]{2})(?:\.([0-9]{1,6}))?)?"
    r"(Z|[+-][0-9]{2}:[0-9]{2})?"
)


def parse_date(text: str) -> date:
    """Parse a `date` string of the form `YYYY-MM-DD`.

    Example: `parse_date("2026-05-21")` returns `date(2026, 5, 21)`.

    Args:
        text (`str`):
            The string to parse. Surrounding whitespace is not allowed.

    Returns:
        `date`:
            The parsed date.

    Raises:
        `ValueError`:
            Raised when `text` does not match the grammar, or names an impossible date
            such as `2026-02-30`.
    """
    match = _DATE_RE.fullmatch(text)
    if match is None:
        raise ValueError(f"date must look like {DATE_FORMAT}")
    return _date_from(match)


def parse_datetime(text: str) -> datetime:
    """Parse a `datetime` string of the form `YYYY-MM-DDTHH:MM[:SS[.ffffff]][Z|±HH:MM]`.

    The separator may be `T` or a space. A fraction of one to six digits is read as
    the leading digits of the microseconds, so `.5` is 500000 µs. `Z` is UTC. Without
    an offset the result is naive.

    Example: `parse_datetime("2026-05-21T14:30Z")` returns
    `datetime(2026, 5, 21, 14, 30, tzinfo=timezone.utc)`.

    Args:
        text (`str`):
            The string to parse. Surrounding whitespace is not allowed.

    Returns:
        `datetime`:
            The parsed datetime, aware when `text` carries an offset.

    Raises:
        `ValueError`:
            Raised when `text` does not match the grammar, or names an impossible value
            such as hour 24 or offset `+07:60`.
    """
    match = _DATETIME_RE.fullmatch(text)
    if match is None:
        raise ValueError(f"datetime must look like {DATETIME_FORMAT}")
    return _datetime_from(match)


def parse_date_or_datetime(text: str) -> date | datetime:
    """Parse `text` as a `date` when it has the date shape, otherwise as a `datetime`.

    Example: `parse_date_or_datetime("2026-05-21")` returns a `date`, and
    `parse_date_or_datetime("2026-05-21T14:30")` returns a `datetime`.

    Args:
        text (`str`):
            The string to parse. Surrounding whitespace is not allowed.

    Returns:
        `date | datetime`:
            A `date` for `YYYY-MM-DD`, otherwise a `datetime`.

    Raises:
        `ValueError`:
            Raised when `text` matches neither grammar, or names an impossible value.
    """
    if match := _DATE_RE.fullmatch(text):
        return _date_from(match)
    if match := _DATETIME_RE.fullmatch(text):
        return _datetime_from(match)
    raise ValueError(f"must be a date ({DATE_FORMAT}) or datetime ({DATETIME_FORMAT})")


def _date_from(match: re.Match[str]) -> date:
    year, month, day = (int(g) for g in match.groups())
    try:
        return date(year, month, day)
    except ValueError as exc:
        raise ValueError(f"not a valid date: {exc}") from None


def _datetime_from(match: re.Match[str]) -> datetime:
    year, month, day, hour, minute, second, fraction, offset = match.groups()
    tzinfo = None
    if offset == "Z":
        tzinfo = timezone.utc
    elif offset is not None:
        off_hours, off_minutes = int(offset[1:3]), int(offset[4:6])
        if off_hours >= 24:
            raise ValueError("not a valid datetime: UTC offset must be under 24:00")
        if off_minutes >= 60:
            raise ValueError("not a valid datetime: UTC offset minutes must be under 60")
        delta = timedelta(hours=off_hours, minutes=off_minutes)
        tzinfo = timezone(-delta if offset[0] == "-" else delta)
    try:
        return datetime(
            int(year),
            int(month),
            int(day),
            int(hour),
            int(minute),
            int(second or 0),
            int((fraction or "0").ljust(6, "0")),
            tzinfo=tzinfo,
        )
    except ValueError as exc:
        raise ValueError(f"not a valid datetime: {exc}") from None
