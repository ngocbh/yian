/**
 * Supplied top-level inputs at the run boundary, a port of the binder half of
 * `tests/engine/test_flow_inputs.py`: `coerceInputs` converts text under each parameter's
 * `coerceStrings` policy and `applyDefaults` fills omitted inputs. The parameters are built by
 * hand with `coerceStrings: true`, as compose's flow-input reader builds them.
 *
 * Ported: `test_int_with_default_and_coercion` (the `coerceInputs` half),
 * `test_apply_defaults_fills_omitted`, the three bool spelling tests and
 * `test_bad_number_passes_through_unchanged`.
 *
 * TS-only: the text rules for `float` and `int` inputs (the rules `int-in-float` and `text`).
 * Text for a `float` input reads an int spelling as that int at any size (`"2"` is `2n`, kept
 * an int in the float slot), a float spelling as a float, `inf`/`-inf` as infinity, and refuses
 * a spelling beyond the float range; Python's `float()` reads `"2"` as `2.0` and also takes
 * `nan`, `Infinity` and `+inf`, which stay text here. An `int` input's text has no digit limit
 * and its digits are ASCII. Also TS-only: a parameter without `coerceStrings`, an `Any` input
 * and a non-record `raw`; and a 5000-digit `bigint` default, the counterpart of
 * `test_long_int_boundaries.py`'s two `apply_defaults` tests (no digit limit, so it is
 * accepted rather than rejected).
 *
 * Later: the 15 declaration tests (`test_required_scalar`, `test_float_default`, ...) read
 * inputs with compose's `read_flow_inputs` and land with it.
 */

import { describe, expect, test } from "bun:test";

import { applyDefaults, checkInputNames, coerceInputs } from "../../src/nodes/binding.ts";
import { Parameter, ValueContract } from "../../src/nodes/contracts.ts";
import { buildValueAs, Type, TypeCheckError, ValueKind } from "../../src/typesys/index.ts";

/** Return a flow input `name` of scalar `kind`, with a default when `fallback` is given. */
function input(name: string, kind: ValueKind, fallback?: unknown): Parameter {
  const contract = new ValueContract(Type.scalar(kind));
  if (fallback === undefined) return new Parameter(name, { contract, required: true, coerceStrings: true });
  return new Parameter(name, { contract, hasDefault: true, default: fallback, coerceStrings: true });
}

/** Return what `coerceInputs` makes of `text` supplied for an input `x` of `kind`. */
function coerced(kind: ValueKind, text: string): unknown {
  return coerceInputs([input("x", kind)], { x: text })["x"];
}

test("an int input coerces its text", () => {
  const params = [input("window", ValueKind.INTEGER, 30n)];
  expect(coerceInputs(params, { window: "5" })).toEqual({ window: 5n });
});

test("applyDefaults fills omitted inputs", () => {
  const params = [input("topic", ValueKind.STRING), input("window", ValueKind.INTEGER, 30n)];
  expect(applyDefaults(params, coerceInputs(params, { topic: "ACME" }))).toEqual({ topic: "ACME", window: 30n });
});

describe("bool spellings", () => {
  test.each(["1", "true", "TRUE", "yes", "y", "on", " True "])("%p is true", (raw) => {
    expect(coerced(ValueKind.BOOLEAN, raw)).toBe(true);
  });

  test.each(["0", "false", "FALSE", "no", "n", "off", " False "])("%p is false", (raw) => {
    expect(coerced(ValueKind.BOOLEAN, raw)).toBe(false);
  });

  // Like a bad number, an unrecognized spelling stays text so the contract rejects it.
  test.each(["maybe", "ture", "approve", "No thanks", ""])("%p stays text", (raw) => {
    expect(coerced(ValueKind.BOOLEAN, raw)).toBe(raw);
  });
});

test("a bad number passes through unchanged", () => {
  expect(coerced(ValueKind.INTEGER, "not-a-number")).toBe("not-a-number");
});

describe("float input text", () => {
  test.each([
    ["2", 2n],
    ["2.0", 2],
    [" 1_000 ", 1000n],
    ["1e3", 1000],
    [".5", 0.5],
    ["inf", Infinity],
    ["-inf", -Infinity],
    ["-0.0", 0],
  ] as const)("%p reads as %p", (text, value) => {
    expect(coerced(ValueKind.NUMBER, text)).toBe(value);
  });

  test("an int spelling is an int at any size", () => {
    const text = "1" + "0".repeat(400);
    const value = coerced(ValueKind.NUMBER, text);
    expect(value).toBe(10n ** 400n);
    // The float contract keeps it as given.
    expect(buildValueAs(Type.scalar(ValueKind.NUMBER), value).value).toBe(10n ** 400n);
  });

  test("a spelling beyond the float range is refused", () => {
    const call = () => coerced(ValueKind.NUMBER, "1e400");
    expect(call).toThrow(TypeCheckError);
    expect(call).toThrow(new TypeCheckError("input 'x': number is too large for a float"));
  });

  test.each(["nan", "Infinity", "+inf", "0x10", "1__0", "", " abc "])("%p stays text as given", (text) => {
    expect(coerced(ValueKind.NUMBER, text)).toBe(text);
  });
});

describe("int input text", () => {
  test("leading zeros are dropped", () => {
    expect(coerced(ValueKind.INTEGER, "007")).toBe(7n);
  });

  test("an int has no digit limit", () => {
    const text = "9".repeat(5000);
    expect(coerced(ValueKind.INTEGER, text)).toBe(BigInt(text));
  });

  test("a float spelling stays text", () => {
    expect(coerced(ValueKind.INTEGER, "2.0")).toBe("2.0");
  });
});

test("text stays text without coerceStrings or for an Any input", () => {
  const plain = new Parameter("n", { contract: new ValueContract(Type.scalar(ValueKind.INTEGER)) });
  expect(coerceInputs([plain], { n: "5" })).toEqual({ n: "5" });
  expect(coerced(ValueKind.ANY, "5")).toBe("5");
});

test("inputs must be a record", () => {
  const params = [input("x", ValueKind.INTEGER)];
  expect(() => checkInputNames(params, [["x", "1"]])).toThrow(
    new TypeError("inputs must be a mapping of input names to values, got Array"),
  );
  const fromMap = () => coerceInputs(params, new Map([["x", "1"]]) as never);
  // Bun's toThrow(error) compares only the message, so the class is checked on its own.
  expect(fromMap).toThrow(TypeError);
  expect(fromMap).toThrow(new TypeError("inputs must be a mapping of input names to values, got Map"));
});

test("unknown input names are listed in the text form", () => {
  const params = [input("x", ValueKind.INTEGER)];
  expect(() => checkInputNames(params, { zz: 1n })).toThrow(new RangeError('unknown input "zz"; declared inputs: x'));
  expect(() => checkInputNames(params, { b: 1n, a: 2n })).toThrow(
    new RangeError('unknown inputs "a", "b"; declared inputs: x'),
  );
});

test("applyDefaults accepts a long int default", () => {
  const big = 10n ** 5000n - 1n;
  expect(applyDefaults([new Parameter("x", { hasDefault: true, default: big })], {})).toEqual({ x: big });
});
