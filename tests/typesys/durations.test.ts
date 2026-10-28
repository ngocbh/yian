/** Durations: a positive integer then s, m, h or d, read as seconds. */
import { expect, test } from "bun:test";

import { DURATION_FORMAT, parseDuration } from "../../src/typesys/temporal.ts";

test.each([
  ["90s", 90],
  ["30m", 1800],
  ["24h", 86400],
  ["2d", 172800],
  ["1s", 1],
])("parseDuration(%p) is %p seconds", (text, seconds) => {
  expect(parseDuration(text)).toBe(seconds);
});

test.each(["86400", "0s", "-5m", "1.5h", "5 m", "5M", "5w", " 5m", "5m\n", "", "m"])(
  "parseDuration refuses %p",
  (text) => {
    expect(() => parseDuration(text)).toThrow(RangeError);
  },
);

test("a plain number gets the seconds spelling as its hint", () => {
  expect(() => parseDuration("86400")).toThrow(
    `a duration is a positive integer and a unit (${DURATION_FORMAT}); write 86400s`,
  );
});

test("a duration past the safe integer range is refused", () => {
  expect(() => parseDuration(`${Number.MAX_SAFE_INTEGER}d`)).toThrow(RangeError);
});

test("a plain number too large to spell in seconds gets no hint", () => {
  const text = "99999999999999999999";
  expect(() => parseDuration(text)).toThrow(`a duration is a positive integer and a unit (${DURATION_FORMAT})`);
  expect(() => parseDuration(text)).not.toThrow("write");
});

test.each([[5], [5n], [null], [["5m"]]])("parseDuration refuses the non-text %p, with no hint", (value) => {
  expect(() => parseDuration(value as unknown as string)).toThrow(TypeError);
  expect(() => parseDuration(value as unknown as string)).toThrow(
    `a duration is text: a positive integer and a unit (${DURATION_FORMAT})`,
  );
  expect(() => parseDuration(value as unknown as string)).not.toThrow("write");
});
