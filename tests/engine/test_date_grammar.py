"""The `date`/`datetime` string grammar.

The grammar is explicit rather than delegated to `fromisoformat`, so these tests pass
identically on any interpreter.
"""

from datetime import date, datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from yian.compose import LoadError, load_flow
from yian.typesys.seeding import now_utc, today_utc
from yian.typesys.temporal import parse_date, parse_date_or_datetime, parse_datetime
from yian.typesys.types import type_for
from yian.typesys.values import (
    DateTimeValue,
    DateValue,
    TypeCheckError,
    ValueKind,
    _scalar_matches,
    build_value_as,
)

DATE_MSG = "date must look like 2026-05-21"
DATETIME_MSG = "datetime must look like 2026-05-21T14:30[:00][Z|+07:00]"


def _tz(hours: int, minutes: int = 0) -> timezone:
    sign = -1 if hours < 0 else 1
    return timezone(sign * timedelta(hours=abs(hours), minutes=minutes))


def test_parse_date_accepts_the_grammar():
    assert parse_date("2026-05-21") == date(2026, 5, 21)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("2026-05-21T14:30", datetime(2026, 5, 21, 14, 30)),
        ("2026-05-21 14:30", datetime(2026, 5, 21, 14, 30)),
        ("2026-05-21T14:30:05", datetime(2026, 5, 21, 14, 30, 5)),
        ("2026-05-21T14:30:05.5", datetime(2026, 5, 21, 14, 30, 5, 500000)),
        ("2026-05-21T14:30:05.123456", datetime(2026, 5, 21, 14, 30, 5, 123456)),
        ("2026-05-21T14:30:05Z", datetime(2026, 5, 21, 14, 30, 5, tzinfo=timezone.utc)),
        ("2026-05-21T14:30:05.123Z", datetime(2026, 5, 21, 14, 30, 5, 123000, tzinfo=timezone.utc)),
        ("2026-05-21T14:30:05+07:00", datetime(2026, 5, 21, 14, 30, 5, tzinfo=_tz(7))),
        ("2026-05-21T14:30:05-03:30", datetime(2026, 5, 21, 14, 30, 5, tzinfo=_tz(-3, 30))),
        ("2026-05-21T14:30:05-00:00", datetime(2026, 5, 21, 14, 30, 5, tzinfo=timezone.utc)),
        ("2026-05-21T14:30Z", datetime(2026, 5, 21, 14, 30, tzinfo=timezone.utc)),
        ("2026-05-21 14:30+07:00", datetime(2026, 5, 21, 14, 30, tzinfo=_tz(7))),
        ("2026-05-21T14:30:05.25+05:30", datetime(2026, 5, 21, 14, 30, 5, 250000, tzinfo=_tz(5, 30))),
    ],
)
def test_parse_datetime_accepts_the_grammar(text, expected):
    got = parse_datetime(text)
    assert got == expected
    assert got.utcoffset() == expected.utcoffset()


@pytest.mark.parametrize(
    "text",
    [
        "20200101",
        "2020-W01-3",
        "2020-001",
        "2026-5-21",
        " 2026-05-21",
        "2026-05-21\n",
        "2026-05-21T00:00",
        "２０２６-05-21",
    ],
)
def test_parse_date_rejects_other_shapes(text):
    with pytest.raises(ValueError, match=DATE_MSG):
        parse_date(text)


@pytest.mark.parametrize(
    "text",
    [
        "2026-05-21",
        "20200101T000000",
        "2020-W01-3T00:00",
        "2026-05-21T14",
        "2026-05-21t14:30",
        "2026-05-21T14:30:05.1234567",
        "2026-05-21T14:30+0700",
        "2026-05-21T14:30+07",
        "2026-05-21T14:30:05z",
        "2026-05-21T14:30 ",
        "２０２６-05-21T14:30",
    ],
)
def test_parse_datetime_rejects_other_shapes(text):
    with pytest.raises(ValueError) as exc:
        parse_datetime(text)
    assert DATETIME_MSG in str(exc.value)


def test_parse_date_or_datetime_picks_by_shape():
    got = parse_date_or_datetime("2026-05-21")
    assert got == date(2026, 5, 21)
    assert not isinstance(got, datetime)
    assert parse_date_or_datetime("2026-05-21T14:30Z") == datetime(2026, 5, 21, 14, 30, tzinfo=timezone.utc)


def test_parse_date_or_datetime_names_both_formats_on_a_miss():
    with pytest.raises(ValueError) as exc:
        parse_date_or_datetime("20260521")
    message = str(exc.value)
    assert "2026-05-21" in message
    assert "2026-05-21T14:30[:00][Z|+07:00]" in message


def test_parse_date_or_datetime_keeps_the_specific_reason():
    with pytest.raises(ValueError, match="not a valid date"):
        parse_date_or_datetime("2026-02-30")


@pytest.mark.parametrize("text", ["2026-13-01", "2026-02-30", "0000-01-01"])
def test_parse_date_rejects_impossible_values(text):
    with pytest.raises(ValueError) as exc:
        parse_date(text)
    assert str(exc.value).startswith("not a valid date")


@pytest.mark.parametrize(
    "text",
    [
        "2026-05-21T24:00",
        "2026-05-21T14:60",
        "2026-05-21T14:30:60",
        "2026-05-21T14:30+24:00",
        "2026-05-21T14:30+07:60",
    ],
)
def test_parse_datetime_rejects_impossible_values(text):
    with pytest.raises(ValueError) as exc:
        parse_datetime(text)
    assert str(exc.value).startswith("not a valid datetime")


def test_parse_date_or_datetime_keeps_the_datetime_reason():
    with pytest.raises(ValueError, match="not a valid datetime"):
        parse_date_or_datetime("2026-05-21T24:00")


def test_parse_datetime_rejects_a_fraction_without_seconds():
    with pytest.raises(ValueError) as exc:
        parse_datetime("2026-05-21T14:30.5")
    assert DATETIME_MSG in str(exc.value)


# --- typed values ----------------------------------------------------------------------------- #


def test_date_value_keeps_the_text_and_rejects_other_shapes():
    assert DateValue(value="2026-05-21").value == "2026-05-21"
    with pytest.raises(ValidationError, match=DATE_MSG):
        DateValue(value="20200101")


@pytest.mark.parametrize("text", ["2020-01-01T00:00:00Z", "2020-01-01T00:00:00.5"])
def test_datetime_value_keeps_the_text(text):
    assert DateTimeValue(value=text).value == text


@pytest.mark.parametrize("text", ["2026-05-21", "20200101T000000"])
def test_datetime_value_rejects_other_shapes(text):
    with pytest.raises(ValidationError) as exc:
        DateTimeValue(value=text)
    assert DATETIME_MSG in str(exc.value)


def test_seeded_clock_values_fit_the_grammar():
    assert DateTimeValue(value=now_utc())
    assert DateValue(value=today_utc())


def test_build_value_as_accepts_the_grammar():
    assert build_value_as(type_for("datetime", {}), "2020-01-01T00:00:00Z").value == "2020-01-01T00:00:00Z"


@pytest.mark.parametrize(
    "type_str, value, message",
    [
        (
            "datetime",
            "20200101T000000",
            f"'20200101T000000' does not match declared type datetime: {DATETIME_MSG}",
        ),
        ("date", "2020-W01-3", f"'2020-W01-3' does not match declared type date: {DATE_MSG}"),
        (
            "date",
            "2026-02-30",
            "'2026-02-30' does not match declared type date: "
            "not a valid date: day is out of range for month",
        ),
        ("date", 5, "5 does not match declared type date"),
    ],
)
def test_build_value_as_names_the_format(type_str, value, message):
    with pytest.raises(TypeCheckError) as exc:
        build_value_as(type_for(type_str, {}), value)
    assert str(exc.value) == message


@pytest.mark.parametrize(
    "kind, value, expected",
    [
        (ValueKind.DATETIME, "20200101T000000", False),
        (ValueKind.DATE, "2020-W01-3", False),
        (ValueKind.DATETIME, "2020-01-01T00:00:00Z", True),
        (ValueKind.DATETIME, "2026-05-21", False),
    ],
)
def test_scalar_matches_uses_the_grammar(kind, value, expected):
    assert _scalar_matches(kind, value) is expected


def test_flow_default_outside_the_grammar_fails_at_load():
    with pytest.raises(LoadError) as exc:
        load_flow('id: f\nname: f\ninput:\n  at: datetime = "20200101T000000"\nnodes: {}\n')
    assert DATETIME_MSG in str(exc.value)
