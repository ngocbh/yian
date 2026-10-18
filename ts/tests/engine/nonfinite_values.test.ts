/**
 * Which numbers are yian values, a port of `tests/engine/test_nonfinite_values.py`. In TS yian
 * `±Infinity` is a float and NaN is the only number that isn't a value; host intake
 * (`fromHost`) refuses it with Python's located message form, so the `reject_nonfinite` rows
 * run through `fromHost` with NaN as the number (the rule: infinity and minus infinity are
 * float values, NaN is not a yian value). Also here: how intake reads a `number` as an int or
 * a float, and the `int is too large for float` rows of the write boundary.
 *
 * Also here: `test_long_ints.py::test_reject_nonfinite_still_reports_nan`.
 *
 * Later: the `canonical_typed` half of
 * `test_an_inferred_float_list_with_a_huge_int_fails_at_its_index` (with the run log).
 *
 * Not ported: the complex-number rows (no complex numbers in TS); the `ObjectValue`, pydantic
 * extras and dataclass rows (a `TypedValue`, a class instance and a dataclass are refused by
 * intake as non-plain objects, see `plain_containers.test.ts`); the cycle half of
 * `test_reject_nonfinite_accepts_finite_values_and_cycles` (intake refuses a cycle); and
 * `test_a_str_subclass_key_becomes_an_exact_str_segment` (a record key is always a string).
 */

import { afterEach, beforeEach, describe, expect, test } from "bun:test";

import { BUILTIN, forget, install } from "../../src/defaults.ts";
import {
  buildValue,
  buildValueAs,
  fromHost,
  Type,
  TypeCheckError,
  type Value,
  ValueKind,
} from "../../src/typesys/values.ts";

// Start each test from the built-in values, whatever `YIAN_*` variables the shell sets.
beforeEach(() => install(BUILTIN));
afterEach(forget);

/** Return the `TypeCheckError` `call()` throws. */
function typeCheckError(call: () => unknown): TypeCheckError {
  try {
    call();
  } catch (error) {
    expect(error).toBeInstanceOf(TypeCheckError);
    return error as TypeCheckError;
  }
  throw new Error("expected a TypeCheckError");
}

describe("NaN at intake", () => {
  // Infinity is a float value; NaN is not a yian value.
  test.each<[string, (n: number) => unknown, (string | number)[]]>([
    ["the root", (n) => n, []],
    ["a list item", (n) => ({ a: [1.5, n] }), ["a", 1]],
    ["a record in a list", (n) => [0, { b: n }], [1, "b"]],
  ])("reports the first path: %s", (_id, wrap, path) => {
    const error = typeCheckError(() => fromHost(wrap(NaN), "output"));
    expect(error.message).toContain(": NaN is not a yian value");
    expect(error.path).toEqual(path);
    expect(() => fromHost(wrap(Infinity), "output")).not.toThrow();
    expect(() => fromHost(wrap(-Infinity), "output")).not.toThrow();
  });

  test("finite values pass", () => {
    expect(fromHost({ a: [1, 2.5, "inf", null, true] }, "output")).toStrictEqual({
      a: [1n, 2.5, "inf", null, true],
    });
  });

  test.each<[unknown, (string | number)[], string]>([
    [{ a: [0, NaN] }, ["a", 1], "input 'y' at a[1]: NaN is not a yian value"],
    [{ a: [0, { b: NaN }] }, ["a", 1, "b"], "input 'y' at a[1].b: NaN is not a yian value"],
    [[{ "my key": NaN }], [0, "my key"], "input 'y' at [0]['my key']: NaN is not a yian value"],
  ])("names where and the path: %p", (value, path, message) => {
    const error = typeCheckError(() => fromHost(value, "input 'y'"));
    expect(error.message).toBe(message);
    expect(error.path).toEqual(path);
  });

  // From test_long_ints.py::test_reject_nonfinite_still_reports_nan.
  test("still reports NaN", () => {
    expect(typeCheckError(() => fromHost(NaN, "input 'n'")).message).toBe(
      "input 'n': NaN is not a yian value",
    );
  });
});

describe("numbers at intake", () => {
  test.each<[string, number, Value]>([
    ["Infinity", Infinity, Infinity],
    ["-Infinity", -Infinity, -Infinity],
    ["a fraction", 2.5, 2.5],
    ["1e20", 1e20, 1e20],
    ["the largest safe integer", 9007199254740991, 9007199254740991n],
    ["the smallest safe integer", -9007199254740991, -9007199254740991n],
    ["2^53", 9007199254740992, 9007199254740992],
    ["-0", -0, 0n],
  ])("%s", (_id, number, expected) => {
    expect(fromHost(number, "output")).toBe(expected);
  });
});

describe("an int too large for a float", () => {
  test("fails at the root of a float slot", () => {
    const error = typeCheckError(() => buildValueAs(Type.scalar(ValueKind.NUMBER), 10n ** 400n));
    expect(error.message).toBe("int is too large for float");
    expect(error.path).toEqual([]);
  });

  test("fails at the index of a float list item", () => {
    const error = typeCheckError(() =>
      buildValueAs(Type.scalar(ValueKind.LIST_NUMBER), [1n, 10n ** 400n]),
    );
    expect(error.message).toBe("int is too large for float");
    expect(error.path).toEqual([1]);
  });

  test("fails at its index in an inferred float list", () => {
    const error = typeCheckError(() => buildValue([0.5, 10n ** 400n]));
    expect(error.message).toBe("int is too large for float");
    expect(error.path).toEqual([1]);
  });
});
