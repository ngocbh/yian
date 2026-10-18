/**
 * The `Type` vocabulary's shape checks, levels and labels, a port of the hand-built half of
 * `tests/engine/test_types.py`; the parser and resolver tests come with `types.ts`.
 */

import { describe, expect, test } from "bun:test";

import {
  type Bounds,
  boundText,
  canonicalTypeLabel,
  listKindForElement,
  Type,
  TypeCheckError,
  typeEquals,
  typeLevels,
  ValueKind,
} from "../../src/typesys/values.ts";

const INT = Type.scalar(ValueKind.INTEGER);
const FLOAT = Type.scalar(ValueKind.NUMBER);
const STR = Type.scalar(ValueKind.STRING);

/** Return the error `typ.validateShape()` throws, failing unless it is exactly `cls`. */
function shapeError(typ: Type, cls: typeof TypeError | typeof RangeError): Error {
  try {
    typ.validateShape();
  } catch (error) {
    expect((error as Error).constructor).toBe(cls);
    return error as Error;
  }
  throw new Error("validateShape() did not throw");
}

/** Return a record type whose fields are all required. */
function record(fields: Record<string, Type>): Type {
  return new Type({
    kind: ValueKind.OBJECT,
    fields: new Map(Object.entries(fields)),
    required: new Set(Object.keys(fields)),
  });
}

describe("numeric bounds shape", () => {
  const INTEGER = ValueKind.INTEGER;
  const NUMBER = ValueKind.NUMBER;
  const TWO_ITEM = "type bounds must be a two-item array";
  const INT_EXACT = "int bounds require exact numeric endpoints, excluding bool";
  const FLOAT_EXACT = "float bounds require exact numeric endpoints, excluding bool";
  const FINITE = "float bounds require finite endpoints";
  const NO_BOUNDS = "only int and float types may declare bounds";
  test.each([
    ["int '1,2'", INTEGER, "1,2", TypeError, TWO_ITEM],
    ["int [1n]", INTEGER, [1n], TypeError, TWO_ITEM],
    ["int [1n, 2n, 3n]", INTEGER, [1n, 2n, 3n], TypeError, TWO_ITEM],
    ["int [true, 2n]", INTEGER, [true, 2n], TypeError, INT_EXACT],
    ["int [1n, 2.0]", INTEGER, [1n, 2.0], TypeError, INT_EXACT],
    [
      "int [2n, 1n]",
      INTEGER,
      [2n, 1n],
      RangeError,
      "type bounds must be ordered from minimum to maximum",
    ],
    ["float [0, false]", NUMBER, [0, false], TypeError, FLOAT_EXACT],
    [
      "float [null, null]",
      NUMBER,
      [null, null],
      RangeError,
      "type bounds need at least one endpoint",
    ],
    ["float [NaN, 1]", NUMBER, [NaN, 1], RangeError, FINITE],
    ["float [0, Infinity]", NUMBER, [0, Infinity], RangeError, FINITE],
    [
      "float [2n ** 53n + 1n, null]",
      NUMBER,
      [2n ** 53n + 1n, null],
      RangeError,
      "float bounds require lossless float endpoints",
    ],
    ["str", ValueKind.STRING, [1n, 2n], RangeError, NO_BOUNDS],
    ["list[int]", ValueKind.LIST_INTEGER, [1n, 2n], RangeError, NO_BOUNDS],
    ["object", ValueKind.OBJECT, [1n, 2n], RangeError, NO_BOUNDS],
  ] as const)("rejects forged metadata: %s", (_label, kind, bounds, cls, message) => {
    const typ = new Type({ kind, bounds } as never);
    expect(shapeError(typ, cls).message).toBe(message);
  });

  test("bounds do not combine with incompatible refinements", () => {
    for (const refinement of [
      { fields: new Map<string, Type>() },
      { tags: new Set(["a"]) },
      { element: INT },
    ]) {
      const typ = new Type({ kind: ValueKind.INTEGER, bounds: [1n, 2n], ...refinement });
      shapeError(typ, RangeError);
    }
  });

  test.each([10n ** 400n, -(10n ** 400n), 2n ** 53n + 1n, -(2n ** 53n + 1n)])(
    "float bounds reject %# without a lossless finite float",
    (endpoint) => {
      const typ = new Type({ kind: ValueKind.NUMBER, bounds: [endpoint, endpoint] });
      expect(shapeError(typ, RangeError).message).toMatch(/finite|lossless/);
    },
  );

  test("ends compare exactly across bigint and number", () => {
    const ok: Bounds[] = [[2n ** 53n, 2 ** 53], [0n, 0.5], [-1, 0n]];
    for (const bounds of ok) new Type({ kind: ValueKind.NUMBER, bounds }).validateShape();
    const typ = new Type({ kind: ValueKind.NUMBER, bounds: [1n, 0.5] });
    expect(shapeError(typ, RangeError).message).toMatch(/ordered/);
  });
});

describe("runtime type shape", () => {
  test("accepts resolved and shared graphs", () => {
    const shared = record({ value: FLOAT, confidence: FLOAT });
    record({ first: shared, second: shared }).validateShape();
    new Type({ kind: ValueKind.LIST_ANY, element: STR }).validateShape();
  });

  test.each([
    [/kind/, { kind: "string" }],
    [/nullable/, { kind: ValueKind.STRING, nullable: 1 }],
    [/nullable/, { kind: ValueKind.STRING, nullable: null }],
    [/fields/, { kind: ValueKind.OBJECT, fields: [] }],
    [/required fields/, { kind: ValueKind.OBJECT, fields: new Map(), required: [] }],
    [/tags/, { kind: ValueKind.STRING, tags: ["yes"] }],
    [/nested type/, { kind: ValueKind.LIST_STRING, element: "str" }],
  ])("rejects a malformed carrier with a TypeError matching %p", (message, init) => {
    expect(shapeError(new Type(init as never), TypeError).message).toMatch(message);
  });

  test.each([
    [/only object types/, { kind: ValueKind.STRING, fields: new Map() }],
    [/need a record field mapping/, { kind: ValueKind.OBJECT, required: new Set() }],
    [
      /unknown record fields/,
      { kind: ValueKind.OBJECT, fields: new Map(), required: new Set(["missing"]) },
    ],
    [/only string types/, { kind: ValueKind.OBJECT, tags: new Set(["yes"]) }],
    [/only list types/, { kind: ValueKind.STRING, element: STR }],
    [
      /structural refinement|only string types/,
      { kind: ValueKind.OBJECT, fields: new Map(), tags: new Set(["yes"]) },
    ],
  ])("rejects conflicting refinements with a RangeError matching %p", (message, init) => {
    expect(shapeError(new Type(init as never), RangeError).message).toMatch(message);
  });

  test("names unknown required fields in the text form, sorted", () => {
    const typ = new Type({
      kind: ValueKind.OBJECT,
      fields: new Map(),
      required: new Set(["b", "a"]),
    });
    expect(() => typ.validateShape()).toThrow('type requires unknown record fields: ["a","b"]');
  });

  test.each([
    new Type({ kind: ValueKind.LIST_INTEGER, element: STR }),
    new Type({ kind: ValueKind.LIST_STRING, element: Type.scalar(ValueKind.DATE) }),
    new Type({
      kind: ValueKind.LIST_OBJECT,
      element: new Type({ kind: ValueKind.STRING, tags: new Set(["yes"]) }),
    }),
  ])("rejects mismatched list storage %#", (typ) => {
    expect(shapeError(typ, RangeError).message).toMatch(/list storage kind/);
  });

  test("names the list storage kinds in Python's quoting", () => {
    const typ = new Type({ kind: ValueKind.LIST_INTEGER, element: STR });
    expect(() => typ.validateShape()).toThrow(
      "list storage kind 'list[int]' does not match element type; expected 'list[str]'",
    );
  });

  test("rejects noncanonical metadata", () => {
    const metadata = "bad\ud800";
    for (const typ of [
      new Type({ kind: ValueKind.OBJECT, fields: new Map([[metadata, STR]]) }),
      new Type({
        kind: ValueKind.OBJECT,
        fields: new Map([["ok", STR]]),
        required: new Set([metadata]),
      }),
      new Type({ kind: ValueKind.STRING, tags: new Set([metadata]) }),
    ]) {
      expect(shapeError(typ, RangeError).message).toMatch(/Unicode surrogate/);
    }
  });

  test("rejects cycles", () => {
    const fields = new Map<string, Type>();
    const cyclic = new Type({ kind: ValueKind.OBJECT, fields });
    fields.set("self", cyclic);
    expect(shapeError(cyclic, RangeError).message).toMatch(/acyclic/);
  });
});

describe("Type", () => {
  test("scalar carries only a storage tag", () => {
    const typ = Type.scalar(ValueKind.DATE);
    expect(typ.kind).toBe(ValueKind.DATE);
    expect(typ.nullable).toBe(false);
    expect(typ.fields).toBeUndefined();
    expect(typ.bounds).toBeUndefined();
  });

  test("replace changes the given properties and leaves the original unchanged", () => {
    const original = new Type({ kind: ValueKind.INTEGER, bounds: [0n, null] });
    const optional = original.replace({ nullable: true });
    expect(optional).not.toBe(original);
    expect(optional.nullable).toBe(true);
    expect(optional.bounds).toBe(original.bounds);
    expect(optional.kind).toBe(ValueKind.INTEGER);
    expect(original.nullable).toBe(false);
  });

  test("list of a scalar, Literal, record or other element", () => {
    expect(listKindForElement(INT)).toBe(ValueKind.LIST_INTEGER);
    expect(listKindForElement(Type.scalar(ValueKind.DATE))).toBe(ValueKind.LIST_ANY);
    expect(listKindForElement(new Type({ kind: ValueKind.STRING, tags: new Set(["a"]) }))).toBe(
      ValueKind.LIST_STRING,
    );
    expect(listKindForElement(record({ a: INT }))).toBe(ValueKind.LIST_OBJECT);
    expect(listKindForElement(Type.scalar(ValueKind.ANY))).toBe(ValueKind.LIST_ANY);
  });
});

describe("typeEquals", () => {
  test("compares records by content, whatever the field order", () => {
    const a = new Type({ kind: ValueKind.OBJECT, fields: new Map([["x", INT], ["y", STR]]) });
    const b = new Type({
      kind: ValueKind.OBJECT,
      fields: new Map([["y", Type.scalar(ValueKind.STRING)], ["x", INT]]),
    });
    expect(typeEquals(a, b)).toBe(true);
    expect(typeEquals(a, record({ x: INT, y: STR }))).toBe(false);
  });

  test("compares bounds by numeric value", () => {
    const a = new Type({ kind: ValueKind.NUMBER, bounds: [1n, null] });
    expect(typeEquals(a, new Type({ kind: ValueKind.NUMBER, bounds: [1, null] }))).toBe(true);
    expect(typeEquals(a, new Type({ kind: ValueKind.NUMBER, bounds: [null, 1] }))).toBe(false);
    expect(typeEquals(a, FLOAT)).toBe(false);
  });

  test("tells nullable, tags, required sets and variant order apart", () => {
    expect(typeEquals(INT, INT.replace({ nullable: true }))).toBe(false);
    const lit = (...tags: string[]) => new Type({ kind: ValueKind.STRING, tags: new Set(tags) });
    expect(typeEquals(lit("a", "b"), lit("b", "a"))).toBe(true);
    expect(typeEquals(lit("a", "b"), lit("a", "c"))).toBe(false);
    const fields = new Map([["a", INT], ["b", INT]]);
    const req = (...names: string[]) =>
      new Type({ kind: ValueKind.OBJECT, fields, required: new Set(names) });
    expect(typeEquals(req("a"), req("a"))).toBe(true);
    expect(typeEquals(req("a"), req("b"))).toBe(false);
    const buy = record({ qty: INT });
    const sell = record({ price: FLOAT });
    const union = (...variants: Type[]) => new Type({ kind: ValueKind.OBJECT, variants });
    expect(typeEquals(union(buy, sell), union(record({ qty: INT }), sell))).toBe(true);
    expect(typeEquals(union(buy, sell), union(sell, buy))).toBe(false);
  });

  test("compares list elements", () => {
    const list = (element: Type) => new Type({ kind: ValueKind.LIST_INTEGER, element });
    expect(typeEquals(list(INT), list(Type.scalar(ValueKind.INTEGER)))).toBe(true);
    expect(typeEquals(list(INT), list(INT.replace({ nullable: true })))).toBe(false);
    expect(typeEquals(list(INT), Type.scalar(ValueKind.LIST_INTEGER))).toBe(false);
  });
});

describe("TypeCheckError", () => {
  test("is a named RangeError with an empty default path", () => {
    const error = new TypeCheckError("bad");
    expect(error).toBeInstanceOf(RangeError);
    expect(error.name).toBe("TypeCheckError");
    expect(error.message).toBe("bad");
    expect(error.path).toEqual([]);
  });

  test("keeps a given path, frozen", () => {
    const path = ["a", 0];
    const error = new TypeCheckError("bad", { path });
    path.push("b");
    expect(error.path).toEqual(["a", 0]);
    expect(Object.isFrozen(error.path)).toBe(true);
  });
});

describe("typeLevels", () => {
  test("counts a scalar as zero", () => {
    expect(typeLevels(STR)).toBe(0);
  });

  test("counts a list as one", () => {
    expect(typeLevels(new Type({ kind: ValueKind.LIST_INTEGER, element: INT }))).toBe(1);
  });

  test("counts each record", () => {
    expect(typeLevels(record({ a: record({ b: INT }) }))).toBe(2);
  });

  test("counts a union as its deepest member", () => {
    const shallow = record({ a: INT });
    const deep = record({ b: record({ c: record({ d: INT }) }) });
    expect(typeLevels(new Type({ kind: ValueKind.OBJECT, variants: [shallow, deep] }))).toBe(3);
  });

  test("measures a shared type once", () => {
    // Each level's two fields hold the same Type, so an unmemoized walk visits 2**60 leaves.
    let typ = INT;
    for (let i = 0; i < 60; i++) typ = record({ a: typ, b: typ });
    expect(typeLevels(typ)).toBe(60);
  });
});

describe("canonicalTypeLabel", () => {
  test("label of a list without an element type", () => {
    expect(canonicalTypeLabel(Type.scalar(ValueKind.LIST_STRING))).toBe("list(str)");
    expect(canonicalTypeLabel(Type.scalar(ValueKind.LIST_ANY))).toBe("list(Optional(Any))");
  });

  // Each label is the one Python's `canonical_type_label` gives.
  test.each([
    [[0.0, 1.5], "float(0.0..1.5)"],
    [[-0.5, 1n], "float(-0.5..1)"],
    [[0.001, 2n], "float(0.001..2)"],
    [[1e-5, 1e16], "float(1e-05..1e+16)"],
    [[0.0001, 1234567890123456.0], "float(0.0001..1234567890123456.0)"],
    [[1.5e-7, 1e21], "float(1.5e-07..1e+21)"],
    [[0.1, 12.5], "float(0.1..12.5)"],
  ] as const)("spells float ends %p as Python does", (bounds, label) => {
    expect(canonicalTypeLabel(new Type({ kind: ValueKind.NUMBER, bounds }))).toBe(label);
  });

  // An untagged union orders its members by the order key, the bracket spelling of each
  // member (`{"a":float[0.0,1]}`), so these orders are the ones Python gives.
  test.each([
    // `null` (the key's open end) sorts after `0.0`; the label's empty end would sort first.
    [[0.0, 1n], [null, 1n], 'Union({"a":float(0.0..1)},{"a":float(..1)})'],
    [[null, 1n], [0.0, 1n], 'Union({"a":float(0.0..1)},{"a":float(..1)})'],
    // `1e-05` sorts after `1.5`, which `0.00001` would not.
    [[1e-5, 2n], [1.5, 2n], 'Union({"a":float(1.5..2)},{"a":float(1e-05..2)})'],
    // An int end stays an int: `0` sorts before `0.0`.
    [[0.0, 1.5], [0n, 1.5], 'Union({"a":float(0..1.5)},{"a":float(0.0..1.5)})'],
  ] as const)("orders members by the key's float ends: %p, %p", (first, second, label) => {
    const member = (bounds: Bounds) => record({ a: new Type({ kind: ValueKind.NUMBER, bounds }) });
    const union = new Type({ kind: ValueKind.OBJECT, variants: [member(first), member(second)] });
    expect(canonicalTypeLabel(union)).toBe(label);
  });
});

describe("boundText", () => {
  // Each float spelling is Python's `repr` of the same float.
  test.each([
    [1n, "1"],
    [-(10n ** 30n), "-1000000000000000000000000000000"],
    [0, "0.0"],
    [-1.5e-7, "-1.5e-07"],
    [-0.0001, "-0.0001"],
    [1e15, "1000000000000000.0"],
    [123456789012345680, "1.2345678901234568e+17"],
    [0.30000000000000004, "0.30000000000000004"],
    [123.456, "123.456"],
    [5e-324, "5e-324"],
    [1.7976931348623157e308, "1.7976931348623157e+308"],
    // A hand-built Type can hold -0.
    [-0, "-0.0"],
    [null, ""],
  ] as const)("spells %p as %p", (end, text) => {
    expect(boundText(end)).toBe(text);
  });

  test("spells a non-finite end as JavaScript does", () => {
    // A non-finite end is never printed in a type (the bounds check refuses it); only the
    // bracket-spelling unparse can meet one, and it spells JavaScript's `Infinity`.
    expect(boundText(Infinity)).toBe("Infinity");
    expect(boundText(-Infinity)).toBe("-Infinity");
  });
});
