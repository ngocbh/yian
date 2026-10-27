/** Run identity and ambient UTC clock seeding for the typed variable pool. */

import { randomUUID } from "node:crypto";

import type { VariablePool } from "./pool.ts";
import { buildValueAs, ValueKind } from "./values.ts";

/**
 * Return the clock time `epochMs` (epoch milliseconds, whose fraction carries microseconds) as
 * an ISO-8601 datetime in UTC: a `T` separator, a fraction
 * of 6 digits only when the microseconds are non-zero, and the offset `+00:00`. The real clock
 * (`Date.now()`) has millisecond resolution, so its fraction ends in `000`.
 *
 * @throws RangeError when `epochMs` isn't a finite time within years 1 to 9999.
 */
export function nowUtc(epochMs: number = Date.now()): string {
  const refusal = `clock time must be within years 1 to 9999, got ${epochMs}`;
  if (!Number.isFinite(epochMs)) throw new RangeError(refusal);
  // Whole milliseconds and microseconds apart: `epochMs * 1000` passes 2^53 after mid-2255,
  // so a product would lose microseconds far from 1970.
  const whole = Math.floor(epochMs);
  const sub = Math.round((epochMs - whole) * 1000);
  const date = new Date(whole + (sub === 1000 ? 1 : 0));
  const year = date.getUTCFullYear();
  // Written so a NaN year (a finite time past the Date range) is refused too.
  if (!(year >= 1 && year <= 9999)) throw new RangeError(refusal);
  const fraction = date.getUTCMilliseconds() * 1000 + (sub % 1000);
  const micros = fraction ? "." + String(fraction).padStart(6, "0") : "";
  return date.toISOString().slice(0, 19) + micros + "+00:00";
}

/**
 * Return the UTC date (`YYYY-MM-DD`) of the clock time `epochMs`.
 *
 * @throws RangeError when `epochMs` isn't a finite time within years 1 to 9999.
 */
export function todayUtc(epochMs: number = Date.now()): string {
  return nowUtc(epochMs).slice(0, 10);
}

/**
 * Return a fresh run identifier for when the host doesn't supply one.
 *
 * A run id is inherently unique per run, so the boundary mints one (a UUID v4 as 32 hex
 * digits) unless the host injects a meaningful value, mirroring how the clock ambients are
 * auto-seeded, so `${system.run_id}` always resolves.
 */
export function defaultRunId(): string {
  return randomUUID().replaceAll("-", "");
}

/**
 * Seed the host-ambient clock into the pool's `system` namespace, once per run.
 *
 * `${system.today}` is a `date`; `${system.now}` a `datetime`. Both come from one reading of
 * the clock `epochMs`, so `today` is always `now`'s date. Computed once here and inherited by
 * call/map children (they copy `pool.system`), so every node, child flow, and MAP element
 * observes one consistent clock. The clock is the only seam through which a flow may read
 * "now" (the engine never resolves date words).
 *
 * @throws RangeError when `epochMs` isn't a finite time within years 1 to 9999.
 */
export function seedSystemClock(pool: VariablePool, epochMs: number = Date.now()): void {
  pool.system.set("today", buildValueAs(ValueKind.DATE, todayUtc(epochMs)));
  pool.system.set("now", buildValueAs(ValueKind.DATETIME, nowUtc(epochMs)));
}
