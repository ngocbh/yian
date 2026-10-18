/**
 * The `date`/`datetime` string grammar, a port of the parse tests and the typed-value tests of
 * `tests/engine/test_date_grammar.py`. Python builds a `DateValue` to check its text; TS has one
 * `TypedValue` class with an unchecked constructor, so those rows decode with
 * `TypedValue.fromWire`, the checked path.
 *
 * The grammar is explicit rather than delegated to `Date.parse`, so these tests pass
 * identically on any runtime.
 *
 * Later: `test_seeded_clock_values_fit_the_grammar` (with seeding), the `build_value_as` and
 * `_scalar_matches` rows (with the write boundary) and the load rows (with the loader).
 */

import { describe, expect, test } from "bun:test";

import {
  type ParsedDate,
  type ParsedDatetime,
  parseDate,
  parseDateOrDatetime,
  parseDatetime,
} from "../../src/typesys/temporal.ts";
import { TypedValue } from "../../src/typesys/values.ts";

const DATE_MSG = "date must look like 2026-05-21";
const DATETIME_MSG = "datetime must look like 2026-05-21T14:30[:00][Z|+07:00]";
const EITHER_MSG = "must be a date (2026-05-21) or datetime (2026-05-21T14:30[:00][Z|+07:00])";

/** Return the error `run` throws, failing unless it is a `RangeError`. */
function rangeError(run: () => unknown): RangeError {
  try {
    run();
  } catch (error) {
    expect(error).toBeInstanceOf(RangeError);
    return error as RangeError;
  }
  throw new Error("expected a RangeError");
}

function date(year: number, month: number, day: number): ParsedDate {
  return { kind: "date", year, month, day };
}

function datetime(
  [year, month, day, hour, minute, second = 0, microsecond = 0]: [
    number,
    number,
    number,
    number,
    number,
    number?,
    number?,
  ],
  offsetMinutes: number | null = null,
): ParsedDatetime {
  return { kind: "datetime", year, month, day, hour, minute, second, microsecond, offsetMinutes };
}

describe("parseDate", () => {
  test("accepts the grammar", () => {
    expect(parseDate("2026-05-21")).toEqual(date(2026, 5, 21));
  });

  test.each(["2024-02-29", "2000-02-29"])("accepts a leap day: %p", (text) => {
    expect(parseDate(text).day).toBe(29);
  });

  test.each([
    "20200101",
    "2020-W01-3",
    "2020-001",
    "2026-5-21",
    " 2026-05-21",
    "2026-05-21\n",
    "2026-05-21T00:00",
    "２０２６-05-21",
  ])("rejects other shapes: %p", (text) => {
    expect(rangeError(() => parseDate(text)).message).toBe(DATE_MSG);
  });

  test.each([
    ["2026-13-01", "not a valid date: month must be in 1..12"],
    ["2026-00-01", "not a valid date: month must be in 1..12"],
    ["2026-02-30", "not a valid date: day is out of range for month"],
    ["2026-01-00", "not a valid date: day is out of range for month"],
    ["2026-02-29", "not a valid date: day is out of range for month"],
    ["1900-02-29", "not a valid date: day is out of range for month"],
    ["0000-01-01", "not a valid date: year 0 is out of range"],
    // The year is checked before the month and day.
    ["0000-13-40", "not a valid date: year 0 is out of range"],
  ])("rejects impossible values: %p", (text, message) => {
    expect(rangeError(() => parseDate(text)).message).toBe(message);
  });
});

describe("parseDatetime", () => {
  test.each<[string, ParsedDatetime]>([
    ["2026-05-21T14:30", datetime([2026, 5, 21, 14, 30])],
    ["2026-05-21 14:30", datetime([2026, 5, 21, 14, 30])],
    ["2026-05-21T14:30:05", datetime([2026, 5, 21, 14, 30, 5])],
    ["2026-05-21T14:30:05.5", datetime([2026, 5, 21, 14, 30, 5, 500000])],
    ["2026-05-21T14:30:05.123456", datetime([2026, 5, 21, 14, 30, 5, 123456])],
    ["2026-05-21T14:30:05Z", datetime([2026, 5, 21, 14, 30, 5], 0)],
    ["2026-05-21T14:30:05.123Z", datetime([2026, 5, 21, 14, 30, 5, 123000], 0)],
    ["2026-05-21T14:30:05+07:00", datetime([2026, 5, 21, 14, 30, 5], 420)],
    ["2026-05-21T14:30:05-03:30", datetime([2026, 5, 21, 14, 30, 5], -210)],
    ["2026-05-21T14:30:05-00:00", datetime([2026, 5, 21, 14, 30, 5], 0)],
    ["2026-05-21T14:30Z", datetime([2026, 5, 21, 14, 30], 0)],
    ["2026-05-21 14:30+07:00", datetime([2026, 5, 21, 14, 30], 420)],
    ["2026-05-21T14:30:05.25+05:30", datetime([2026, 5, 21, 14, 30, 5, 250000], 330)],
  ])("accepts the grammar: %p", (text, expected) => {
    expect(parseDatetime(text)).toEqual(expected);
  });

  test.each([
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
    // A fraction needs seconds.
    "2026-05-21T14:30.5",
  ])("rejects other shapes: %p", (text) => {
    expect(rangeError(() => parseDatetime(text)).message).toBe(DATETIME_MSG);
  });

  test.each([
    ["2026-05-21T24:00", "not a valid datetime: hour must be in 0..23"],
    ["2026-05-21T14:60", "not a valid datetime: minute must be in 0..59"],
    ["2026-05-21T14:30:60", "not a valid datetime: second must be in 0..59"],
    ["2026-05-21T14:30+24:00", "not a valid datetime: UTC offset must be under 24:00"],
    ["2026-05-21T14:30+07:60", "not a valid datetime: UTC offset minutes must be under 60"],
    ["2026-02-30T10:00", "not a valid datetime: day is out of range for month"],
    // The offset is checked first, then the date, then the time.
    ["0000-13-01T25:00+24:60", "not a valid datetime: UTC offset must be under 24:00"],
    ["2026-13-01T25:00+07:60", "not a valid datetime: UTC offset minutes must be under 60"],
    ["0000-13-01T25:00", "not a valid datetime: year 0 is out of range"],
    ["2026-13-01T25:00", "not a valid datetime: month must be in 1..12"],
    ["2026-12-01T25:61:61", "not a valid datetime: hour must be in 0..23"],
    ["2026-12-01T23:61:61", "not a valid datetime: minute must be in 0..59"],
  ])("rejects impossible values: %p", (text, message) => {
    expect(rangeError(() => parseDatetime(text)).message).toBe(message);
  });
});

describe("parseDateOrDatetime", () => {
  test("picks by shape", () => {
    expect(parseDateOrDatetime("2026-05-21")).toEqual(date(2026, 5, 21));
    expect(parseDateOrDatetime("2026-05-21T14:30Z")).toEqual(datetime([2026, 5, 21, 14, 30], 0));
  });

  test("names both formats on a miss", () => {
    expect(rangeError(() => parseDateOrDatetime("20260521")).message).toBe(EITHER_MSG);
  });

  test.each([
    ["2026-02-30", "not a valid date: day is out of range for month"],
    ["2026-05-21T24:00", "not a valid datetime: hour must be in 0..23"],
  ])("keeps the specific reason: %p", (text, message) => {
    expect(rangeError(() => parseDateOrDatetime(text)).message).toBe(message);
  });
});

describe("date and datetime typed values", () => {
  test("a date value keeps the text and rejects other shapes", () => {
    expect(TypedValue.fromWire({ kind: "date", value: "2026-05-21" }).value).toBe("2026-05-21");
    const error = rangeError(() => TypedValue.fromWire({ kind: "date", value: "20200101" }));
    expect(error.message).toContain(DATE_MSG);
  });

  test.each(["2020-01-01T00:00:00Z", "2020-01-01T00:00:00.5"])(
    "a datetime value keeps the text: %p",
    (text) => {
      expect(TypedValue.fromWire({ kind: "datetime", value: text }).value).toBe(text);
    },
  );

  test.each(["2026-05-21", "20200101T000000"])(
    "a datetime value rejects other shapes: %p",
    (text) => {
      const error = rangeError(() => TypedValue.fromWire({ kind: "datetime", value: text }));
      expect(error.message).toContain(DATETIME_MSG);
    },
  );
});
