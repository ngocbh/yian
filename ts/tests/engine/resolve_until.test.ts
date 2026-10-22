/**
 * Resolving a WAIT `until` source at the node-owned binding boundary: a port of
 * `tests/engine/test_resolve_until.py`.
 *
 * Ported: `test_literal_iso_passthrough`, `test_ref_iso_passthrough`,
 * `test_non_date_error_is_origin_neutral_for_raw_and_wrapped_sources`,
 * `test_literal_with_z_suffix_passes_through_unchanged`,
 * `test_literal_outside_the_grammar_names_the_accepted_forms`,
 * `test_deadline_that_overflows_in_utc_names_the_value` and
 * `test_impossible_literal_names_the_reason`.
 *
 * Changed: `RangeError` stands for Python's `ValueError` (a `BindingError` is one). A JS `Date`
 * holds year 0, so the overflow row keeps Python's verdict through an explicit UTC year range
 * check (1 to 9999), refused with Python's `OverflowError` text, `date value out of range`. TS
 * dates are text, so Python's `isoformat()` branch for date objects has nothing to read here.
 * A failure message writes the resolved deadline in the text form (`"…"`), where Python writes
 * its `repr` (`'…'`).
 *
 * TS-only: `9999-12-31T23:30-01:00` is out of range in UTC and `9999-12-31T23:59:59.999999Z` is
 * in range; a `Binding`-wrapped literal passes through.
 *
 * Not ported: `test_retired_runtime_surface.py`'s `test_effect_nodes_drop_scratch_keys` and
 * `test_wait_event_mode_and_the_watchdog_are_gone`: they check that Python-era fields, a keyword
 * and modules are gone, and TS never had them.
 */

import { expect, test } from "bun:test";
import { inspect } from "node:util";

import { OriginRef } from "../../src/diagnostics.ts";
import { BindingError } from "../../src/nodes/binding.ts";
import { Binding } from "../../src/nodes/contracts.ts";
import { resolveUntil } from "../../src/nodes/wait/node.ts";
import { VariablePool } from "../../src/typesys/pool.ts";

/** Return what `resolveUntil(source, pool)` throws, or fail when it returns. */
function rejection(source: unknown, pool: VariablePool = new VariablePool()): BindingError {
  try {
    resolveUntil(source, pool);
  } catch (error) {
    expect(error).toBeInstanceOf(BindingError);
    return error as BindingError;
  }
  throw new Error(`resolveUntil accepted ${inspect(source)}`);
}

test("literal iso passthrough", () => {
  expect(resolveUntil("2026-01-01T00:00:00", new VariablePool())).toBe("2026-01-01T00:00:00");
});

test("ref iso passthrough", () => {
  const pool = new VariablePool();
  pool.set("__start__", { t: "2026-06-20T12:00:00" });
  expect(resolveUntil("${input.t}", pool)).toBe("2026-06-20T12:00:00");
});

test("non date error is origin neutral for raw and wrapped sources", () => {
  // WAIT reports semantic source text without leaking the origin carrier repr.
  const pool = new VariablePool();
  pool.set("__start__", { n: 5n });
  const rawSource = "${input.n}";
  const expected = "wait `until` ('${input.n}') did not resolve to a date/datetime: 5";

  for (const source of [rawSource, new Binding(rawSource, { origin: new OriginRef("wait-until") })]) {
    const caught = rejection(source, pool);
    expect(caught).toBeInstanceOf(RangeError);
    expect(caught.message).toBe(expected);
  }
});

test("literal with z suffix passes through unchanged", () => {
  expect(resolveUntil("2026-01-02T00:00:00Z", new VariablePool())).toBe("2026-01-02T00:00:00Z");
});

for (const [id, literal] of [
  ["basic-format", "20260102"],
  ["week-date", "2026-W01-5"],
  ["seven-digit-fraction", "2026-01-02T00:00:00.1234567"],
] as const) {
  test(`literal outside the grammar names the accepted forms (${id})`, () => {
    const caught = rejection(literal);

    const message = caught.message;
    expect(caught.failureCode).toBe("contract.input");
    expect(message).toContain("2026-05-21");
    expect(message).toContain("2026-05-21T14:30[:00][Z|+07:00]");
    // The value is named once; the parser's message shows the format, not the value.
    expect(message.split(literal).length - 1).toBe(1);
  });
}

test("deadline that overflows in utc names the value", () => {
  const literal = "0001-01-01T00:00+01:00";
  const caught = rejection(literal);

  expect(caught.failureCode).toBe("contract.input");
  expect(caught.message).toBe('wait `until` "0001-01-01T00:00+01:00": date value out of range');
});

for (const [id, literal, reason] of [
  ["impossible-date", "2026-02-30", "not a valid date"],
  ["impossible-datetime", "2026-05-21T24:00", "not a valid datetime"],
] as const) {
  test(`impossible literal names the reason (${id})`, () => {
    const caught = rejection(literal);

    expect(caught.failureCode).toBe("contract.input");
    expect(caught.message).toContain(reason);
  });
}

test("a deadline past year 9999 in UTC is out of range", () => {
  const caught = rejection("9999-12-31T23:30-01:00");

  expect(caught.failureCode).toBe("contract.input");
  expect(caught.message).toBe('wait `until` "9999-12-31T23:30-01:00": date value out of range');
  expect(caught.cause).toBeInstanceOf(RangeError);
});

test("the last microsecond of year 9999 in UTC is in range", () => {
  expect(resolveUntil("9999-12-31T23:59:59.999999Z", new VariablePool())).toBe("9999-12-31T23:59:59.999999Z");
});

test("a binding-wrapped literal passes through", () => {
  const source = new Binding("2026-01-02", { origin: new OriginRef("wait-until") });
  expect(resolveUntil(source, new VariablePool())).toBe("2026-01-02");
});
