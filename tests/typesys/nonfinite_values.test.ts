/**
 * Which numbers are yian values: `±Infinity` is a float and NaN is the only number that isn't a
 * value, refused by host intake (`fromHost`) with a located message. Also how intake reads a
 * `number` as an int or a float, and a huge int in a float slot at the write boundary.
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
    [[{ "my key": NaN }], [0, "my key"], `input 'y' at [0]["my key"]: NaN is not a yian value`],
  ])("names where and the path: %p", (value, path, message) => {
    const error = typeCheckError(() => fromHost(value, "input 'y'"));
    expect(error.message).toBe(message);
    expect(error.path).toEqual(path);
  });

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

describe("a huge int in a float slot stays an int", () => {
  test("at the root of a float slot", () => {
    const typed = buildValueAs(Type.scalar(ValueKind.NUMBER), 10n ** 400n);
    expect(typed.kind).toBe(ValueKind.INTEGER);
    expect(typed.value).toBe(10n ** 400n);
  });

  test("in a float list item", () => {
    const typed = buildValueAs(Type.scalar(ValueKind.LIST_NUMBER), [1n, 10n ** 400n]);
    expect(typed.kind).toBe(ValueKind.LIST_NUMBER);
    expect(typed.value).toEqual([1n, 10n ** 400n]);
  });

  test("in an inferred float list", () => {
    const typed = buildValue([0.5, 10n ** 400n]);
    expect(typed.kind).toBe(ValueKind.LIST_NUMBER);
    expect(typed.value).toEqual([0.5, 10n ** 400n]);
  });
});
