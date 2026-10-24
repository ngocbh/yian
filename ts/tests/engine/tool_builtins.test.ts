/**
 * The TOOL built-ins at run time: `take` and `sleep` return the wait the engine decides, a host
 * tool or `ask` that returns `pending()` records its call and waits, a held call is reused, and
 * `waitResultStep` shapes a host's answer. TS-only: Python yian has no inboxes or pending calls.
 */

import { describe, expect, test } from "bun:test";

import { deadlineUtc, formatRecordTime } from "../../src/typesys/temporal.ts";

describe("temporal", () => {
  test("deadlineUtc reads a date at midnight UTC and normalises an offset", () => {
    expect(deadlineUtc("2026-10-24")).toBe(Date.UTC(2026, 9, 24));
    expect(deadlineUtc("2026-10-24T09:30+07:00")).toBe(Date.UTC(2026, 9, 24, 2, 30));
  });

  test("deadlineUtc refuses an instant before year 1", () => {
    expect(() => deadlineUtc("0001-01-01T00:00+01:00")).toThrow(new RangeError("date value out of range"));
  });

  test("formatRecordTime writes a record time and refuses one out of range", () => {
    expect(formatRecordTime(Date.UTC(2026, 9, 24, 9))).toBe("2026-10-24T09:00:00.000Z");
    expect(() => formatRecordTime(Number.NaN)).toThrow(
      new RangeError("run-log clock time NaN is outside the years 0000 to 9999"),
    );
    expect(() => formatRecordTime(Number.POSITIVE_INFINITY)).toThrow(RangeError);
  });
});
