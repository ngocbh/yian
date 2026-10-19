/**
 * The memory JSON snapshot, from `tests/engine/test_memory_contracts.py`, through
 * `checkMemoryDeclaration` on a custom type: `test_json_contracts_reject_non_exact_or_non_finite_values`,
 * `test_json_contracts_reject_cycles`, the declaration rows of
 * `test_memory_strings_must_have_a_canonical_utf8_representation`, and the declaration half of
 * `test_json_scalar_identity_preserves_exact_types_at_every_depth`; plus JavaScript's own values (a
 * `"__proto__"` key, a null-prototype object, a repeated subtree, an array hole) and the value
 * nesting limit.
 *
 * Changed: the non-exact rows are JavaScript's (a `Map`, a `Date`, an own symbol key, a class
 * instance, an array subclass for `_ListSubclass`, a top-level array for `_DictSubclass`), each
 * with its exact class and message. The scalar identity test checks that `true`, `1n` and `1` all pass
 * at every depth and that a built-in rule tells them apart. A memory config nests at most
 * `maxValueNesting` levels (memory JSON nesting doesn't exist here), with that limit's message.
 * Python's `ValueError` is `RangeError`.
 *
 * Ported by the next commits: the rest of the declaration, identity and host-object tests.
 *
 * Not ported: the tuple and bytes rows (TS has only arrays and no bytes in a config); `test_memory_json_rejects_nesting_beyond_the_memory_limit` (its
 * carriers aren't ported; the nesting test below replaces it for declarations).
 *
 * Left for later: the recall, learning and conversation tests, with their carriers.
 */

import { afterEach, expect, test } from "bun:test";

import { BUILTIN, forget, install } from "../../src/defaults.ts";
import { checkMemoryDeclaration } from "../../src/memory/contracts.ts";

/** Check a custom-type declaration, so the built-in key allowlist does not apply. */
function check(config: unknown): void {
  checkMemoryDeclaration("notes", "custom", config);
}

/** Assert that `fn` throws exactly `cls` with exactly `message`. */
function expectError(fn: () => unknown, cls: typeof TypeError | typeof RangeError, message: string): void {
  let caught: unknown;
  try {
    fn();
  } catch (error) {
    caught = error;
  }
  expect(caught).toBeInstanceOf(cls);
  expect((caught as Error).message).toBe(message);
}

afterEach(() => forget());

class Point {
  x = 1n;
}

class Items extends Array<unknown> {}

// test_json_contracts_reject_non_exact_or_non_finite_values
test.each<[string, unknown, typeof TypeError | typeof RangeError, string]>([
  ["a Map in an array", { array: [new Map()] }, TypeError, "config.array[0] must contain exact JSON values"],
  ["a Date", { date: new Date(0) }, TypeError, "config.date must contain exact JSON values"],
  ["NaN", { nan: Number.NaN }, RangeError, "config.nan JSON number must be finite"],
  ["Infinity", { infinity: Number.POSITIVE_INFINITY }, RangeError, "config.infinity JSON number must be finite"],
  ["a symbol key", { [Symbol("key")]: 1n }, TypeError, "config JSON object keys must be strings"],
  ["a class instance", { point: new Point() }, TypeError, "config.point must contain exact JSON values"],
  ["an array subclass", { a: Items.from([1n]) }, TypeError, "config.a must contain exact JSON values"],
  ["a top-level array", [1n], TypeError, "config must be an exact JSON object"],
])("JSON contracts reject %s", (_name, config, cls, message) => {
  expectError(() => check(config), cls, message);
});

test("a null-prototype object is a JSON object", () => {
  check(Object.assign(Object.create(null), { a: 1n }));
  check({ a: Object.assign(Object.create(null), { b: 1n }) });
});

// test_json_contracts_reject_cycles
test("JSON contracts reject cycles", () => {
  const config: Record<string, unknown> = {};
  config.self = config;
  expectError(() => check(config), RangeError, "config.self contains a circular JSON value");
});

test("a subtree reached twice without a cycle passes", () => {
  const s = { x: 1n };
  check({ a: s, b: s });
  check({ a: [s, s] });
});

test("an array hole is refused", () => {
  expectError(() => check({ a: [1n, , 2n] }), TypeError, "config.a[1] must contain exact JSON values");
});

// test_memory_strings_must_have_a_canonical_utf8_representation
test.each<[string, unknown]>([
  ["a value", { text: "\ud800" }],
  ["a key", { "\ud800": "text" }],
])("memory strings reject a lone surrogate in %s", (_name, config) => {
  expectError(() => check(config), RangeError, "Unicode surrogate is not valid canonical JSON");
});

// test_json_scalar_identity_preserves_exact_types_at_every_depth (the declaration half)
test("JSON scalars keep their exact kinds at every depth", () => {
  for (const value of [true, 1n, 1]) check({ nested: [value, { value }] });
  checkMemoryDeclaration("notes", "reflective", { max_items: 1n });
  expect(() => checkMemoryDeclaration("notes", "reflective", { max_items: true })).toThrow("got true");
  expect(() => checkMemoryDeclaration("notes", "reflective", { max_items: 1 })).toThrow("got 1");
});

test("a __proto__ key stays a data key", () => {
  const config = { ["__proto__"]: 1n };
  expect(Object.hasOwn(config, "__proto__")).toBe(true);
  check(config);
  // The snapshot's key list holds it, so a built-in type reports it as unknown.
  expect(() => checkMemoryDeclaration("notes", "episodic", config)).toThrow("unknown key(s) ['__proto__']");
});

test("a memory config nests at most maxValueNesting levels", () => {
  install({ ...BUILTIN, maxValueNesting: 3 });
  check({ a: { a: {} } });
  check({ a: [[]] });
  expectError(
    () => check({ a: { a: { a: {} } } }),
    RangeError,
    "config.a.a.a nests deeper than 3 levels, the limit for a yian value (each record and list is a level)",
  );
});
