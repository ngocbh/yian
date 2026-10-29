/**
 * How write-check and wire-decode messages show the value they refuse: a float keeps its
 * fraction digit (`2.0`), so an int and a float with the same digits read apart; numeric
 * bounds are spelled as the type label spells them; and a refused `Literal` tag names the
 * `Literal` type.
 */

import { afterEach, beforeEach, describe, expect, test } from "bun:test";

import { BUILTIN, forget, install } from "../../src/defaults.ts";
import { buildValueAs, Type, TypeCheckError, TypedValue, ValueKind } from "../../src/typesys/values.ts";

// Start each test from the built-in values, whatever `YIAN_*` variables the shell sets.
beforeEach(() => install(BUILTIN));
afterEach(forget);

/** Return the message of the `TypeCheckError` `call()` throws. */
function typeCheckMessage(call: () => unknown): string {
  try {
    call();
  } catch (error) {
    expect(error).toBeInstanceOf(TypeCheckError);
    return (error as TypeCheckError).message;
  }
  throw new Error("expected a TypeCheckError");
}

/** Return the message of the `RangeError` `call()` throws. */
function rangeMessage(call: () => unknown): string {
  try {
    call();
  } catch (error) {
    expect(error).toBeInstanceOf(RangeError);
    return (error as RangeError).message;
  }
  throw new Error("expected a RangeError");
}

const INT = Type.scalar(ValueKind.INTEGER);

describe("a float in a write-check message", () => {
  test("a whole float refused by an int slot keeps its fraction digit", () => {
    expect(typeCheckMessage(() => buildValueAs(ValueKind.INTEGER, 2.0))).toBe(
      "2.0 does not match declared type int",
    );
  });

  test("a whole float refused by a list(int) item keeps its fraction digit", () => {
    expect(typeCheckMessage(() => buildValueAs(new Type({ kind: ValueKind.LIST_INTEGER, element: INT }), [1n, 3.0]))).toBe(
      "3.0 does not match declared type int",
    );
  });

  test("a whole float refused by a flat list[int] keeps its fraction digit", () => {
    expect(typeCheckMessage(() => buildValueAs(ValueKind.LIST_INTEGER, [1n, 3.0]))).toBe(
      "list element 3.0 is not int (declared list(int))",
    );
  });

  test("floats inside a refused container keep their fraction digits, ints don't", () => {
    expect(typeCheckMessage(() => buildValueAs(ValueKind.STRING, [1n, 2.0, 0.5]))).toBe(
      "[1,2.0,0.5] does not match declared type str",
    );
  });

  test("an int keeps its digits", () => {
    expect(typeCheckMessage(() => buildValueAs(ValueKind.STRING, 2n))).toBe(
      "2 does not match declared type str",
    );
  });
});

describe("a numeric bound in a write-check message", () => {
  test("a float minimum is spelled as the type label spells it", () => {
    const typ = new Type({ kind: ValueKind.NUMBER, bounds: [1.0, null] });
    expect(typeCheckMessage(() => buildValueAs(typ, 0.5))).toBe(
      "0.5 is below the inclusive float minimum 1.0",
    );
  });

  test("a float maximum is spelled as the type label spells it", () => {
    const typ = new Type({ kind: ValueKind.NUMBER, bounds: [null, 2.0] });
    expect(typeCheckMessage(() => buildValueAs(typ, 3n))).toBe(
      "3 is above the inclusive float maximum 2.0",
    );
  });

  test("an int minimum keeps its digits", () => {
    const typ = new Type({ kind: ValueKind.INTEGER, bounds: [1n, null] });
    expect(typeCheckMessage(() => buildValueAs(typ, 0n))).toBe(
      "0 is below the inclusive int minimum 1",
    );
  });
});

describe("a refused Literal tag", () => {
  test("names the Literal type", () => {
    const typ = new Type({ kind: ValueKind.STRING, tags: new Set(["go", "no go"]) });
    expect(typeCheckMessage(() => buildValueAs(typ, "maybe"))).toBe(
      '"maybe" is not one of Literal(go, "no go")',
    );
  });
});

describe("a number in a wire-decode message", () => {
  test("an int in a float wire form is named as an int", () => {
    expect(rangeMessage(() => TypedValue.fromWire({ kind: "float", value: 2n }))).toBe(
      "invalid float typed value: expected a number other than NaN, got int 2",
    );
  });

  test("a whole float in an int wire form is named as a float", () => {
    expect(rangeMessage(() => TypedValue.fromWire({ kind: "int", value: 2 }))).toBe(
      "invalid int typed value: expected a bigint, got float 2.0",
    );
  });
});
