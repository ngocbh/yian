/**
 * Run ids and the UTC clock seeded into a pool's `system` namespace, a port of
 * `tests/engine/test_seeding.py`.
 *
 * TS-only: the injected-clock rows, which pin the `isoformat()` spelling (the fraction only when
 * non-zero, microseconds from the fraction of a millisecond, a negative epoch, the year bounds
 * and exact arithmetic far from 1970), `todayUtc` of the same instants, one clock reading for
 * both `today` and `now`, seeding from the real clock (kinds only), and the `RangeError` for a
 * clock outside years 1 to 9999.
 */

import { describe, expect, test } from "bun:test";

import { VariablePool } from "../../src/typesys/pool.ts";
import { defaultRunId, nowUtc, seedSystemClock, todayUtc } from "../../src/typesys/seeding.ts";
import { parseDatetime } from "../../src/typesys/temporal.ts";
import { ValueKind } from "../../src/typesys/values.ts";

const AFTERNOON = Date.UTC(2026, 9, 18, 14, 30, 5);

describe("the UTC clock", () => {
  test("nowUtc is an ISO datetime in UTC (test_now_utc_is_iso_datetime)", () => {
    const now = nowUtc();
    parseDatetime(now);
    expect(now).toContain("T");
    expect(now.endsWith("+00:00")).toBe(true);
  });

  test.each<[number, string]>([
    [AFTERNOON, "2026-10-18T14:30:05+00:00"],
    [AFTERNOON + 123, "2026-10-18T14:30:05.123000+00:00"],
    [AFTERNOON + 123.456, "2026-10-18T14:30:05.123456+00:00"],
    [-0.5, "1969-12-31T23:59:59.999500+00:00"],
    [-62135596800000, "0001-01-01T00:00:00+00:00"],
    [253402300799999, "9999-12-31T23:59:59.999000+00:00"],
    [Date.UTC(9999, 0, 1, 0, 0, 0, 1), "9999-01-01T00:00:00.001000+00:00"],
  ])("nowUtc spells an injected clock like isoformat(): %p", (epochMs, text) => {
    expect(nowUtc(epochMs)).toBe(text);
    expect(todayUtc(epochMs)).toBe(text.slice(0, 10));
  });

  test.each([NaN, 253402300800000, -62135596800001, 1e16])(
    "nowUtc refuses a clock outside years 1 to 9999: %p",
    (epochMs) => {
      const message = `clock time must be within years 1 to 9999, got ${epochMs}`;
      expect(() => nowUtc(epochMs)).toThrow(RangeError);
      expect(() => nowUtc(epochMs)).toThrow(message);
      expect(() => todayUtc(epochMs)).toThrow(message);
    },
  );
});

describe("seeding the system namespace", () => {
  test(
    "seedSystemClock seeds typed today and now (test_seed_system_clock_seeds_typed_today_and_now)",
    () => {
      // An injected clock: Python compares with a second clock read, which races midnight.
      const pool = new VariablePool();
      seedSystemClock(pool, AFTERNOON);
      expect(pool.system.get("today")!.kind).toBe(ValueKind.DATE);
      expect(pool.system.get("now")!.kind).toBe(ValueKind.DATETIME);
      expect(pool.system.get("today")!.value).toBe("2026-10-18");
      expect(pool.resolve("system", ["today"])).toBe("2026-10-18");
      expect(pool.resolve("system", ["now"])).toBe("2026-10-18T14:30:05+00:00");
    },
  );

  test("seedSystemClock reads the real clock by default", () => {
    const pool = new VariablePool();
    seedSystemClock(pool);
    expect(pool.system.get("today")!.kind).toBe(ValueKind.DATE);
    expect(pool.system.get("now")!.kind).toBe(ValueKind.DATETIME);
  });

  test("seedSystemClock reads the clock once, so today is now's date", () => {
    const pool = new VariablePool();
    seedSystemClock(pool, Date.UTC(2026, 9, 18, 23, 59, 59, 999));
    expect(pool.resolve("system", ["today"])).toBe("2026-10-18");
    expect(pool.resolve("system", ["now"])).toBe("2026-10-18T23:59:59.999000+00:00");
  });
});

describe("run ids", () => {
  test("defaultRunId is a fresh hex id (test_default_run_id_is_a_fresh_nonempty_string)", () => {
    const [a, b] = [defaultRunId(), defaultRunId()];
    expect(a).toMatch(/^[0-9a-f]{32}$/);
    expect(a).not.toBe(b);
  });
});
