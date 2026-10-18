/**
 * The `Type` vocabulary's shape checks, levels and labels, the type-text parser,
 * `formatTypeExpr`, the bracket-spelling error, typedef names, and `resolveType`/`typeFor`, a
 * port of those tests of `tests/engine/test_types.py`. Later: the value halves and value
 * tests (with `buildValueAs`, including the widening half of
 * `test_union_member_value_widens_its_float_fields`), the pool write test, `test_public_exports`,
 * and `test_bounded_type_defaults_preserve_compatible_integer_payloads` (with the loader).
 * `test_type_for_preserves_its_public_keyword_name` is not ported (TypeScript has no keyword
 * arguments); `test_is_shadow_guard` checks the private guard through `readTypedefs`.
 */

import { describe, expect, test } from "bun:test";

import { reprText } from "../../src/diagnostics.ts";
import {
  formatTypeExpr,
  legacyRespelling,
  parseType,
  readTypedefs,
  resolveType,
  type ScalarName,
  type TypeDef,
  type TypeExpr,
  type TypeRegistry,
  typeFor,
} from "../../src/typesys/types.ts";
import {
  type Bounds,
  boundText,
  canonicalTypeLabel,
  canonicalUnionOrder,
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
    expect(typeLevels(typeFor("list(int)", new Map()))).toBe(1);
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

// --------------------------------------------------------------------------------------- //
// The type-text parser
// --------------------------------------------------------------------------------------- //

const scalar = (name: ScalarName, bounds?: Bounds): TypeExpr =>
  bounds === undefined ? { kind: "scalar", name } : { kind: "scalar", name, bounds };
const ref = (name: string): TypeExpr => ({ kind: "ref", name });
const list = (element: TypeExpr): TypeExpr => ({ kind: "list", element });
const optional = (inner: TypeExpr): TypeExpr => ({ kind: "optional", inner });
const literal = (...tags: string[]): TypeExpr => ({ kind: "enum", tags });
const union = (...members: TypeExpr[]): TypeExpr => ({ kind: "union", members });
const DICT: TypeExpr = { kind: "dict" };

const UNBALANCED =
  "unbalanced parentheses — if this type sits inside a `{...}` mapping, its comma split it; " +
  "quote it or use block style";

/** Return the message of the `TypeCheckError` `parseType(text)` throws. */
function parseError(text: string): string {
  try {
    parseType(text);
  } catch (error) {
    expect(error).toBeInstanceOf(TypeCheckError);
    return (error as TypeCheckError).message;
  }
  throw new Error(`parseType(${JSON.stringify(text)}) did not throw`);
}

/** Return Python's `malformed type expression` message for `src` (no `'` or `\` in it). */
function malformed(src: string, detail?: string): string {
  return `malformed type expression '${src}'${detail === undefined ? "" : `: ${detail}`}`;
}

describe("parseType", () => {
  test("scalars", () => {
    for (const name of ["str", "int", "float", "date", "datetime", "bool", "object"] as const) {
      expect(parseType(name)).toEqual(scalar(name));
    }
    expect(parseType("Any")).toEqual(scalar("Any"));
  });

  test("lists and topics", () => {
    expect(parseType("List(float)")).toEqual(list(scalar("float")));
    expect(parseType("list(str)")).toEqual(list(scalar("str")));
    expect(parseType("topics")).toEqual(list(scalar("str")));
    expect(parseType("List(Rating)")).toEqual(list(ref("Rating")));
    expect(parseType("List(int)")).toEqual(list(scalar("int")));
    expect(parseType("list(Rating)")).toEqual(list(ref("Rating")));
  });

  test("references", () => {
    expect(parseType("Rating")).toEqual(ref("Rating"));
    expect(parseType("Action")).toEqual(ref("Action"));
  });

  test("Optional", () => {
    expect(parseType("Optional(str)")).toEqual(optional(scalar("str")));
    expect(parseType("Optional(Rating)")).toEqual(optional(ref("Rating")));
    expect(parseType("Optional(date)")).toEqual(optional(scalar("date")));
  });

  test("Literal, quoted and unquoted", () => {
    expect(parseType("Literal(pro, con, mixed)")).toEqual(literal("pro", "con", "mixed"));
    expect(parseType('Literal("pro", "con")')).toEqual(literal("pro", "con"));
    expect(parseType("Literal(defer)")).toEqual(literal("defer"));
  });

  test("the old engine names are registry references", () => {
    for (const legacy of ["string", "integer", "number", "boolean"]) {
      expect(parseType(legacy)).toEqual(ref(legacy));
    }
  });

  test.each([
    ["Optional(str)", optional(scalar("str"))],
    ["list(Buy)", list(ref("Buy"))],
    ["List(str)", list(scalar("str"))],
    ["Optional(list(Buy))", optional(list(ref("Buy")))],
    ["Union(Buy, Sell)", union(ref("Buy"), ref("Sell"))],
    ["Union(Buy)", union(ref("Buy"))],
    ["Literal(approve, reject)", literal("approve", "reject")],
    ['Literal("a b")', literal("a b")],
    ["Literal('x', y)", literal("x", "y")],
    ["int(1..12)", scalar("int", [1n, 12n])],
    ["int(0..1)", scalar("int", [0n, 1n])],
    ["int(-5..5)", scalar("int", [-5n, 5n])],
    ["int( 1 .. 12 )", scalar("int", [1n, 12n])],
    ["float(0.0..1.0)", scalar("float", [0, 1])],
    ["float(1e-3..2)", scalar("float", [0.001, 2n])],
    ["int(0..)", scalar("int", [0n, null])],
    ["int(..100)", scalar("int", [null, 100n])],
    ["int(-5..)", scalar("int", [-5n, null])],
    ["float(0.0..)", scalar("float", [0, null])],
    ["float(..-1.5)", scalar("float", [null, -1.5])],
    ["int( 0 .. )", scalar("int", [0n, null])],
    ["int(+4..4)", scalar("int", [4n, 4n])],
    ["dict(str, int)", DICT],
    ["Dict(str, Buy)", DICT],
    ["Any", scalar("Any")],
    ["None", scalar("None")],
    ["none", scalar("None")],
    ["Optional(None)", optional(scalar("None"))],
    ["topics", list(scalar("str"))],
    ["Literal('a'\n)", literal("a")],
  ] as const)("parenthesized syntax %p", (text, expr) => {
    expect(parseType(text)).toEqual(expr);
  });

  test("an int end keeps every digit", () => {
    const large = 10n ** 400n;
    expect(parseType(`int(${large}..)`)).toEqual(scalar("int", [large, null]));
    expect(parseType(`int(${-large}..${large})`)).toEqual(scalar("int", [-large, large]));
  });

  test("a -0 float end is stored as 0", () => {
    // Python keeps -0.0; a -0 end never reaches a manifest or fingerprint here.
    expect(parseType("float(-0.0..1)")).toEqual(scalar("float", [0, 1n]));
    expect(formatTypeExpr(parseType("float(-0.0..1)"))).toBe("float(0.0..1)");
  });

  test.each([
    ["int(..)", malformed("int(..)", "a range needs at least one end; write int for any int")],
    [
      "float(..)",
      malformed("float(..)", "a range needs at least one end; write float for any float"),
    ],
    [
      "int()",
      malformed("int()", "a numeric range needs `..`, e.g. int(1..12), int(0..), or int(..100)"),
    ],
    [
      "int(1)",
      malformed("int(1)", "a numeric range needs `..`, e.g. int(1..12), int(0..), or int(..100)"),
    ],
    [
      "float(0)",
      malformed(
        "float(0)",
        "a numeric range needs `..`, e.g. float(0.0..1.0), float(0.0..), or float(..1.0)",
      ),
    ],
    ["int(1..2..)", malformed("int(1..2..)", "a range takes one lo..hi, e.g. int(1..12)")],
    ["float(..1e999)", "invalid bounds in 'float(..1e999)': float bounds require finite endpoints"],
    [
      "float(1e999..2)",
      "invalid bounds in 'float(1e999..2)': float bounds require finite endpoints",
    ],
    [
      "int(1.5..)",
      "invalid bounds in 'int(1.5..)': int bounds require exact numeric endpoints, excluding bool",
    ],
    [
      "int(..1.0)",
      "invalid bounds in 'int(..1.0)': int bounds require exact numeric endpoints, excluding bool",
    ],
    [
      "int(1.5..2)",
      "invalid bounds in 'int(1.5..2)': int bounds require exact numeric endpoints, excluding bool",
    ],
    ["int(1, 12)", malformed("int(1, 12)", "write a range: int(1..12)")],
    [
      "int(3..1)",
      "invalid bounds in 'int(3..1)': type bounds must be ordered from minimum to maximum",
    ],
    ["Buy | Sell", malformed("Buy | Sell", "use Union(Buy, Sell) (| is not a type operator)")],
    [
      "Union(Buy, None)",
      "None is not a union member in 'Union(Buy, None)'; use Optional(Union(...))",
    ],
    ["Optional(str", malformed("Optional(str", UNBALANCED)],
    ["Optional(str))", malformed("Optional(str))", UNBALANCED)],
    [
      "Literal(True)",
      "invalid Literal member in 'Literal(True)' (use names or quoted strings)",
    ],
    [
      "Literal(None)",
      "invalid Literal member in 'Literal(None)' (use names or quoted strings)",
    ],
    ["Literal(1)", "invalid Literal member in 'Literal(1)' (use names or quoted strings)"],
    [
      "Literal(a(b))",
      "invalid Literal member in 'Literal(a(b))' (use names or quoted strings)",
    ],
    ["Literal(a b)", malformed("Literal(a b)")],
    [
      'Literal("")',
      "empty Literal member in 'Literal(\"\")' (tags must be non-empty names or quoted strings)",
    ],
    ["Literal()", malformed("Literal()", "Literal(...) needs a member")],
    ["Union()", malformed("Union()", "Union(...) needs a type")],
    ["list()", malformed("list()", "list(...) needs a type")],
    ["dict()", malformed("dict()", "dict(...) needs a type")],
    ["Optional(str, int)", malformed("Optional(str, int)", "Optional(...) takes one type")],
    ["dict(str)", malformed("dict(str)", "dict(...) takes 2 types")],
    ["Foo(int)", "unknown type constructor 'Foo' in 'Foo(int)'"],
    ["bool(0..1)", "unknown type constructor 'bool' in 'bool(0..1)'"],
    ["", malformed("")],
    ["   ", malformed("")],
    ["True", malformed("True")],
    ["int(1..2) str", malformed("int(1..2) str")],
    ["float(0..None)", "numeric bounds require literal endpoints in 'float(0..None)'"],
    ["int(lo..hi)", "numeric bounds require literal endpoints in 'int(lo..hi)'"],
    ["int(1 + 1..3)", malformed("int(1 + 1..3)")],
  ] as const)("error for %p", (text, message) => {
    expect(parseError(text)).toBe(message);
  });

  test("rejects a raw Unicode surrogate", () => {
    expect(parseError('Literal("\ud800")')).toBe(
      String.raw`type expression 'Literal("\ud800")' contains a Unicode surrogate ` +
        "(types must be valid UTF-8 text)",
    );
  });

  test("malformed text", () => {
    expect(parseError("list(")).toBe(malformed("list(", UNBALANCED));
    // a number literal is not a type
    expect(parseError("123")).toBe(malformed("123"));
  });

  test.each([
    "int()", "int(1)", "int(1..2..3)", "float(0)", "float(0..1..2)",
    "int(2..1)", "float(2.0..1.0)", "int(True..2)", "int(0..False)",
    "float(False..1)", "float(0..True)", "int(1.0..2)", "int(1..2.0)",
    "float(0..1e999)", "float(-1e999..0)", "float(nan..1)", "float(0..inf)",
    "int('1'..2)", "float(0..None)", "int(1 + 1..3)", "int(-(1 + 1)..3)",
    "float(0..float('inf'))", "int(lo..hi)", "int(1:2)", "int((1..2))",
    "int(--1..2)", "float(0j..1)", "bool(0..1)", "str(1..2)",
  ])("numeric bounds reject a nonliteral or malformed end: %p", (text) => {
    parseError(text);
  });

  test("a bracket inside a quoted Literal member is text", () => {
    expect(parseType('Literal("a[0]")')).toEqual(literal("a[0]"));
  });

  test.each([
    ['Literal("a[b")', ["a[b"]],
    ['Literal("x", "[")', ["x", "["]],
    ["Literal('a]b', c)", ["a]b", "c"]],
  ] as const)("brackets inside quoted members stay parenthesized: %p", (text, tags) => {
    expect(parseType(text)).toEqual(literal(...tags));
  });

  test("deep nesting is a TypeCheckError", () => {
    expect(parseError("Optional(".repeat(300) + "str" + ")".repeat(300))).toBe(
      "type nested too deeply in 'Optional(Optional(Optional(Optional(Opti'... " +
        "(at most 100 levels)",
    );
  });

  test("moderate nesting parses", () => {
    const text = "Optional(".repeat(30) + "str" + ")".repeat(30);
    expect(formatTypeExpr(parseType(text))).toBe(text);
  });

  test.each([
    ["Optional(", "'Optional('"],
    ["list(", "'list('"],
    // the text is trimmed first
    ["Union(A, ", "'Union(A,'"],
    ["int(1", "'int(1'"],
    ["int(1..", "'int(1..'"],
    ["int(1..2", "'int(1..2'"],
    ["Literal(", "'Literal('"],
    ['Literal("a', `'Literal("a'`],
    ['Union(A, Literal("x', `'Union(A, Literal("x'`],
    ["Optional(Literal('x", `"Optional(Literal('x"`],
    // the closing `)` is quoted, so `_expect` finds the end
    ['Literal("a)"', `'Literal("a)"'`],
  ])("input ending inside parentheses hints unbalanced: %p", (text, quoted) => {
    expect(parseError(text)).toBe(`malformed type expression ${quoted}: ${UNBALANCED}`);
  });

  test.each([
    // ASCII digits only
    ["int(١..٣)", malformed("int(١..٣)")],
    ["int(1..12, 3)", malformed("int(1..12, 3)", "a range takes one lo..hi, e.g. int(1..12)")],
    ["int(1..2..3)", malformed("int(1..2..3)", "a range takes one lo..hi, e.g. int(1..12)")],
    ["Literal(a,)", malformed("Literal(a,)", "remove the trailing comma in Literal(...)")],
    ["Union(A,)", malformed("Union(A,)", "remove the trailing comma in Union(...)")],
    ["list(str,)", malformed("list(str,)", "remove the trailing comma in list(...)")],
    ['"abc', malformed('"abc')],
  ])("more syntax errors: %p", (text, message) => {
    expect(parseError(text)).toBe(message);
  });

  test("error precedence", () => {
    // The tokenizer fails at `$` before the `|` is read.
    expect(parseError("Union(A | B, $)")).toBe(malformed("Union(A | B, $)"));
    // A quoted `)` still counts, so the parentheses balance and the error is bare.
    expect(parseError('Literal("a)",')).toBe(malformed('Literal("a)",'));
    // The None member is checked before the trailing token.
    expect(parseError("Union(A, None) x")).toBe(
      "None is not a union member in 'Union(A, None) x'; use Optional(Union(...))",
    );
  });

  test.each([
    ["int(1.0, 2)", "int(1..12)"],
    ["float(0.5, 2)", "float(0.5..2)"],
    ["int(0,)", "int(0..)"],
    ["int(,100)", "int(..100)"],
    ["int(,)", "int(1..12)"],
    // float ends in Python's spelling
    ["float(1.0, 2)", "float(1.0..2)"],
    ["float(1e-5, 0.5)", "float(1e-05..0.5)"],
  ])("the two-argument range hint suggests only a valid range: %p", (text, range) => {
    expect(parseError(text)).toBe(malformed(text, `write a range: ${range}`));
  });
});

describe("parseType: quoted Literal members", () => {
  test.each([
    [String.raw`Literal("a\tb", 'it\'s', "é")`, ["a\tb", "it's", "é"]],
    [String.raw`Literal("\"q\"", "\\")`, ['"q"', "\\"]],
    [String.raw`Literal("\b\f\n\r\té")`, ["\b\f\n\r\té"]],
    // Python keeps the backslash: `a\/b`.
    [String.raw`Literal("a\/b")`, ["a/b"]],
    // A JSON high-low pair is one character; Python refuses it as two surrogates.
    [String.raw`Literal("\ud83d\ude00")`, ["😀"]],
    ['Literal("😀")', ["😀"]],
    // A raw TAB is part of the tag.
    ['Literal("a\tb")', ["a\tb"]],
  ] as const)("decodes %p", (text, tags) => {
    expect(parseType(text)).toEqual(literal(...tags));
  });

  test.each([
    [String.raw`Literal("\ud800")`, String.raw`'\ud800' in 'Literal("\\ud800")'`],
    [String.raw`Literal("\ude00")`, String.raw`'\ude00' in 'Literal("\\ude00")'`],
    [String.raw`Literal("\ud83dx")`, String.raw`'\ud83dx' in 'Literal("\\ud83dx")'`],
  ])("an escaped lone surrogate is refused: %p", (text, quoted) => {
    expect(parseError(text)).toBe(
      `Literal member ${quoted} contains a Unicode surrogate (tags must be valid UTF-8 text)`,
    );
  });

  const invalid = (c: string) =>
    `invalid escape '\\${c}' in a quoted Literal member (use JSON escapes)`;

  test.each([
    // Python decodes these three: `A`, `•`, and `\d` kept as written.
    [String.raw`Literal("\x41")`, String.raw`'Literal("\\x41")'`, invalid("x")],
    [String.raw`Literal("\N{BULLET}")`, String.raw`'Literal("\\N{BULLET}")'`, invalid("N")],
    [String.raw`Literal("\d")`, String.raw`'Literal("\\d")'`, invalid("d")],
    // A backslash before U+2028 is an escape (Python keeps it as text).
    ['Literal("a\\\u2028b")', String.raw`'Literal("a\\\u2028b")'`, invalid("\u2028")],
    [
      String.raw`Literal("\u12")`,
      String.raw`'Literal("\\u12")'`,
      String.raw`'\u' needs four hex digits in a quoted Literal member`,
    ],
    [
      'Literal("a\nb")',
      String.raw`'Literal("a\nb")'`,
      "a quoted Literal member can't hold a line break",
    ],
    [
      'Literal("a\rb")',
      String.raw`'Literal("a\rb")'`,
      "a quoted Literal member can't hold a line break",
    ],
    // The line break is checked before escapes, so this ordering is TS-only; Python reports
    // an empty Literal member.
    [
      'Literal("\\\r")',
      String.raw`'Literal("\\\r")'`,
      "a quoted Literal member can't hold a line break",
    ],
  ])("rejects %p", (text, quoted, detail) => {
    expect(parseError(text)).toBe(`malformed type expression ${quoted}: ${detail}`);
  });

  test("every printed member reads back", () => {
    const tags = ["a\tb", 'say "hi"', "\\", "it's", "\u2028", "😀", "True", "a b"];
    expect(parseType(formatTypeExpr(literal(...tags)))).toEqual(literal(...tags));
  });
});

describe("formatTypeExpr", () => {
  test.each([
    "Optional(list(Buy))",
    "Union(Buy, Sell)",
    "Literal(a, b)",
    "int(1..12)",
    "float(0.0..1.0)",
    "int(-5..5)",
  ])("round-trips %p", (text) => {
    expect(formatTypeExpr(parseType(text))).toBe(text);
  });

  test("spells float ends as Python does and keeps an int end an int", () => {
    expect(formatTypeExpr(parseType("float(1e-3..2)"))).toBe("float(0.001..2)");
    expect(formatTypeExpr(parseType("float(1e-5..1e16)"))).toBe("float(1e-05..1e+16)");
  });

  test("normalizes spacing and signs", () => {
    expect(formatTypeExpr(parseType("Optional(Union(Buy,Sell))"))).toBe(
      "Optional(Union(Buy, Sell))",
    );
    expect(formatTypeExpr(parseType("int( -2 .. +4 )"))).toBe("int(-2..4)");
  });

  test("prints a dict as dict", () => {
    expect(formatTypeExpr(parseType("dict(str, int)"))).toBe("dict");
  });
});

describe("bracket spelling", () => {
  /** The error for bracket text that has no parenthesized spelling. */
  const bracketError = (detail: string) => `types use parentheses, not brackets (${detail})`;
  const RANGE_NEEDS_DOTS = "a numeric range needs `..`, e.g. int(1..12), int(0..), or int(..100)";

  test.each([
    ["Optional[list[Buy]]", "Optional(list(Buy))"],
    ["int[1, 12]", "int(1..12)"],
    ["Optional[Union[Buy, Sell]]", "Optional(Union(Buy, Sell))"],
    ["dict[str, int]", "object"],
    ['Literal["a b", c]', 'Literal("a b", c)'],
    ["int[-1, +2]", "int(-1..2)"],
    ["Optional[None]", "Optional(None)"],
    ["list[dict[str, Buy]]", "list(object)"],
    // A trailing comma makes a one-member tuple.
    ["Union[A,]", "Union(A)"],
    ["float[1., .5e1]", "float(1.0..5.0)"],
    ["int[00, 2]", "int(0..2)"],
    // Python keeps the sign: `float(-0.0..1)`.
    ["float[-0.0, 1]", "float(0.0..1)"],
    // Python reads identifiers in NFKC form.
    ["Optional[ｓｔｒ]", "Optional(str)"],
    ["Optional[℘]", "Optional(℘)"],
    // Keywords are matched before NFKC, so `Ｔｒｕｅ` is a name.
    ["Literal[Ｔｒｕｅ]", 'Literal("True")'],
    ["Optional[Ｔｒｕｅ]", "Optional(True)"],
    // Python refuses a keyword as a name: `malformed type expression 'Literal[if, b]'`.
    ["Literal[if, b]", "Literal(if, b)"],
  ])("is an error showing the parenthesized spelling: %p", (text, spelled) => {
    expect(parseError(text)).toBe(`types use parentheses: \`${spelled}\``);
    expect(legacyRespelling(text)).toBe(spelled);
  });

  // The bracket half of test_deep_nesting_is_a_type_error.
  test("bracket nesting is capped at 100 levels", () => {
    const deep = "Optional[".repeat(300) + "str" + "]".repeat(300);
    // Python reports the `SyntaxError` of its parser's own limit.
    expect(parseError(deep)).toBe(
      bracketError(
        "type nested too deeply in 'Optional[Optional[Optional[Optional[Opti'... " +
          "(at most 100 levels)",
      ),
    );
    // Python respells 150 levels.
    expect(parseError("Optional[".repeat(150) + "str" + "]".repeat(150))).toStartWith(
      "types use parentheses, not brackets (type nested too deeply in ",
    );
    const moderate = "Optional[".repeat(100) + "str" + "]".repeat(100);
    const spelled = "Optional(".repeat(100) + "str" + ")".repeat(100);
    expect(parseError(moderate)).toBe(`types use parentheses: \`${spelled}\``);
  });

  test.each([
    [
      "Optional[Buy | Sell]",
      malformed("Optional[Buy | Sell]", "use Union(Buy, Sell) (| is not a type operator)"),
    ],
    [
      "Buy | Foo[ 'x' ,1.50]",
      `malformed type expression "Buy | Foo[ 'x' ,1.50]": ` +
        "use Union(Buy, Foo['x', 1.5]) (| is not a type operator)",
    ],
    [
      "Optional[x]|y",
      malformed("Optional[x]|y", "use Union(Optional(x), y) (| is not a type operator)"),
    ],
    ["Buy | []", malformed("Buy | []", "use Union(Buy, []) (| is not a type operator)")],
    ["Foo[int]", "unknown type constructor 'Foo' in 'Foo[int]'"],
    ["list[any]", "unknown type 'any'; write Any"],
    ["int[1, x]", "numeric bounds require literal endpoints in 'int[1, x]'"],
    ["int[--1, 2]", "numeric bounds require literal endpoints in 'int[--1, 2]'"],
    ["int[-x, 2]", "numeric bounds require literal endpoints in 'int[-x, 2]'"],
    ["int[+None, 2]", "numeric bounds require literal endpoints in 'int[+None, 2]'"],
    [
      "int[3, 1]",
      "invalid bounds in 'int[3, 1]': type bounds must be ordered from minimum to maximum",
    ],
    [
      "int[1.5, 2]",
      "invalid bounds in 'int[1.5, 2]': int bounds require exact numeric endpoints, excluding bool",
    ],
    [
      "float[1e400, 2]",
      "invalid bounds in 'float[1e400, 2]': float bounds require finite endpoints",
    ],
    [
      "Union[Buy, None]",
      "None is not a union member in 'Union[Buy, None]'; use Optional(Union(...))",
    ],
    ["Literal[1]", "invalid Literal member in 'Literal[1]' (use names or quoted strings)"],
    ["Literal[True]", "invalid Literal member in 'Literal[True]' (use names or quoted strings)"],
    [
      'Literal[""]',
      `empty Literal member in 'Literal[""]' (tags must be non-empty names or quoted strings)`,
    ],
    [
      String.raw`Literal['\ud800']`,
      String.raw`Literal member '\ud800' in "Literal['\\ud800']" contains a Unicode surrogate ` +
        "(tags must be valid UTF-8 text)",
    ],
    ["Optional[True]", malformed("Optional[True]")],
    ["[str]", malformed("[str]")],
    ["[]", malformed("[]")],
    ["list[A,]", malformed("list[A,]")],
    ["A[1][2]", malformed("A[1][2]")],
    ["A[1], B", malformed("A[1], B")],
    ["int[1, 2, 3]", malformed("int[1, 2, 3]", RANGE_NEEDS_DOTS)],
    // test_bracket_single_endpoint_hint_uses_the_range_spelling
    ["int[1]", malformed("int[1]", RANGE_NEEDS_DOTS)],
    ["int[1,]", malformed("int[1,]", RANGE_NEEDS_DOTS)],
    // Python crashes here with an uncaught `UnicodeEncodeError` from `tokenize`.
    [
      "Optional[\ud800]",
      String.raw`type expression 'Optional[\ud800]' contains a Unicode surrogate ` +
        "(types must be valid UTF-8 text)",
    ],
  ])("explains why %p has no parenthesized spelling", (text, detail) => {
    expect(parseError(text)).toBe(bracketError(detail));
  });

  test.each([
    // test_bracket_text_that_is_no_type_still_says_types_use_parentheses. Python appends
    // the `SyntaxError` detail to this row and the next three.
    "list[",
    "A[]",
    "int[01, 2]",
    "Optional[²]",
    // Python reads `0x10` as 16 and respells `int(16..20)`.
    "int[0x10, 20]",
    // Python: `invalid bounds in 'int[1_000, 2]': type bounds must be ordered ...`.
    "int[1_000, 2]",
    // Python: `numeric bounds require literal endpoints in 'int[1j, 2]'`.
    "int[1j, 2]",
    // Python respells `Literal(a)`.
    "Literal[r'a']",
    // Python respells `Literal(ab)`.
    'Literal["a" "b"]',
    // Python respells `object`.
    "dict[1:2]",
    // Python respells `int(1..2)`.
    "int[(1), 2]",
  ])("text that fails the syntax pass is malformed: %p", (text) => {
    expect(parseError(text)).toBe(bracketError(`malformed type expression ${reprText(text)}`));
  });

  test("an operand that is no type is printed as written", () => {
    const text = "Buy | A[b,] | -x | [a, 'q'] | --1 | -A[1] | None | True | 1e400";
    // Python prints the overflowing float as `1e309`.
    expect(parseError(text)).toBe(
      bracketError(
        `malformed type expression "${text}": use Union(Buy, A[b,], -x, [a, 'q'], --1, ` +
          "-A[1], None, True, Infinity) (| is not a type operator)",
      ),
    );
  });
});

// --------------------------------------------------------------------------------------- //
// Registry and resolution
// --------------------------------------------------------------------------------------- //

/** A registry of record typedefs built directly, without checks, from each one's fields. */
function records(raw: Record<string, Record<string, string>>): TypeRegistry {
  return new Map(
    Object.entries(raw).map(([name, fields]): [string, TypeDef] => [
      name,
      { kind: "record", fields: new Map(Object.entries(fields)) },
    ]),
  );
}

const EMPTY: TypeRegistry = new Map();

const REG: TypeRegistry = new Map([
  ...records({
    Rating: { value: "float", confidence: "float" },
    Prices: { closes: "List(float)", last: "float" },
  }),
  ["Action", { kind: "variant", tags: ["Approve", "Reject", "Defer"] }],
]);

/** Return the message of the `TypeCheckError` `typeFor(text, reg)` throws. */
function typeForError(text: string, reg: TypeRegistry): string {
  try {
    typeFor(text, reg);
  } catch (error) {
    expect(error).toBeInstanceOf(TypeCheckError);
    return (error as TypeCheckError).message;
  }
  throw new Error(`typeFor(${JSON.stringify(text)}) did not throw`);
}

describe("resolveType", () => {
  test("Optional is nullable", () => {
    const sh = resolveType(optional(scalar("str")), EMPTY);
    expect(sh.kind).toBe(ValueKind.STRING);
    expect(sh.nullable).toBe(true);
  });

  test("an inline Literal is an enum", () => {
    const sh = resolveType(literal("pro", "con"), EMPTY);
    expect(sh.kind).toBe(ValueKind.STRING);
    expect(sh.tags).toEqual(new Set(["pro", "con"]));
  });
});

describe("typeFor", () => {
  test("an Optional record field is not required", () => {
    const sh = typeFor("Sig", records({ Sig: { score: "float", note: "Optional(str)" } }));
    expect(sh.required).toEqual(new Set(["score"]));
    expect(sh.fields!.get("note")!.nullable).toBe(true);
    expect(sh.fields!.get("score")!.nullable).toBe(false);
  });

  test("the old engine names no longer resolve", () => {
    for (const legacy of ["string", "integer", "number", "boolean"]) {
      expect(typeForError(legacy, EMPTY)).toBe(`unknown type '${legacy}'`);
    }
    expect(typeForError("list(string)", EMPTY)).toBe("unknown type 'string'");
  });

  test("a union of scalars is rejected", () => {
    expect(typeForError("Union(int, str)", EMPTY)).toBe(
      "Union(int, str): member 'int' is not a record",
    );
  });

  test("scalars and lists", () => {
    expect(typeFor("str", REG).kind).toBe(ValueKind.STRING);
    expect(typeFor("date", REG).kind).toBe(ValueKind.DATE);
    expect(typeFor("datetime", REG).kind).toBe(ValueKind.DATETIME);
    expect(typeFor("List(float)", REG).kind).toBe(ValueKind.LIST_NUMBER);
  });

  test.each([
    ["int(1..12)", [1n, 12n]],
    ["float(-0.5..1.0)", [-0.5, 1.0]],
    ["int(-12..-1)", [-12n, -1n]],
    ["int(+4..4)", [4n, 4n]],
    ["float(-0.25..+0.25)", [-0.25, 0.25]],
    ["float(2..2)", [2n, 2n]],
  ] as const)("keeps inclusive bounds: %p", (text, bounds) => {
    const typ = typeFor(text, EMPTY);
    expect(typ.bounds).toEqual(bounds);
    typ.validateShape();
  });

  test("large integer bounds stay exact", () => {
    const large = 10n ** 400n;
    const typ = typeFor(`int(${-large}..${large})`, EMPTY);
    typ.validateShape();
    expect(typ.bounds).toEqual([-large, large]);
  });

  test("bounded labels include nested domains", () => {
    expect(canonicalTypeLabel(typeFor("int( -2 .. +4 )", EMPTY))).toBe("int(-2..4)");
    expect(canonicalTypeLabel(typeFor("list(Optional(float(0.0..1.5)))", EMPTY))).toBe(
      "list(Optional(float(0.0..1.5)))",
    );
  });

  test.each([
    [10n ** 400n, "finite"],
    [-(10n ** 400n), "finite"],
    [2n ** 53n + 1n, "lossless float"],
    [-(2n ** 53n + 1n), "lossless float"],
  ])("a float range rejects %# without a lossless finite float", (endpoint, needs) => {
    const text = `float(${endpoint}..${endpoint})`;
    expect(typeForError(text, EMPTY)).toBe(
      `invalid bounds in '${text}': float bounds require ${needs} endpoints`,
    );
  });

  test("dict is a lenient object placeholder", () => {
    expect(typeFor("dict(str, int)", REG).kind).toBe(ValueKind.OBJECT);
  });

  test("records and variants", () => {
    const s = typeFor("Rating", REG);
    expect(s.kind).toBe(ValueKind.OBJECT);
    expect(s.required).toEqual(new Set(["value", "confidence"]));
    const a = typeFor("Action", REG);
    expect(a.kind).toBe(ValueKind.STRING);
    expect(a.tags).toEqual(new Set(["Approve", "Reject", "Defer"]));
  });

  test("a list of records", () => {
    const s = typeFor("List(Rating)", REG);
    expect(s.kind).toBe(ValueKind.LIST_OBJECT);
    expect(s.element!.fields).toBeDefined();
  });

  test("an unknown type throws", () => {
    expect(typeForError("Nope", REG)).toBe("unknown type 'Nope'");
  });

  test("a recursive record is rejected", () => {
    expect(typeForError("Node", records({ Node: { child: "Node" } }))).toBe(
      "recursive type reference: 'Node'",
    );
  });

  test("resolved and shared graphs have a valid shape", () => {
    const shared = typeFor("Rating", REG);
    record({ first: shared, second: shared }).validateShape();
    for (const text of [
      "str",
      "Optional(str)",
      "Action",
      "Rating",
      "List(Action)",
      "List(Rating)",
      "List(date)",
      "List(List(int))",
    ]) {
      typeFor(text, REG).validateShape();
    }
  });

  test("labels use parentheses", () => {
    const label = (text: string) => canonicalTypeLabel(typeFor(text, EMPTY));
    expect(label("Optional(int(1..12))")).toBe("Optional(int(1..12))");
    expect(label("list(str)")).toBe("list(str)");
    expect(label("Literal(b, a)")).toBe("Literal(a, b)");
    expect(label("float(0.5..1.5)")).toBe("float(0.5..1.5)");
  });

  test("union member order follows the bracket labels", () => {
    // By the parenthesized labels `{"a":int(1..2)}` sorts first; by the bracket labels
    // `{"a":int,"b":str}` does. Keeping the bracket order keeps manifests stable.
    const reg = readTypedefs({ A: { a: "int(1..2)" }, B: { a: "int", b: "str" } });
    const union = typeFor("Union(A, B)", reg);
    const [first, second] = union.variants!;
    expect(new Set(first!.fields!.keys())).toEqual(new Set(["a", "b"]));
    const order = canonicalUnionOrder([second!, first!]);
    expect(order[0]).toBe(first!);
    expect(order[1]).toBe(second!);
    expect(canonicalTypeLabel(union)).toBe('Union({"a":int,"b":str},{"a":int(1..2)})');
  });

  test("a registry of union members with float fields resolves", () => {
    const reg = readTypedefs({
      Buy: { action: "Literal(buy)", qty: "int" },
      Sell: { action: "Literal(sell)", price: "float" },
      Count: { n: "int" },
      Score: { n: "float", label: "str" },
    });
    expect(typeFor("Union(Buy, Sell)", reg).variants).toHaveLength(2);
    expect(typeFor("Union(Count, Score)", reg).variants).toHaveLength(2);
  });
});

/** Return the `TypeCheckError` `readTypedefs(raw)` throws. */
function typedefsError(raw: Readonly<Record<string, unknown>>): TypeCheckError {
  try {
    readTypedefs(raw);
  } catch (error) {
    expect(error).toBeInstanceOf(TypeCheckError);
    return error as TypeCheckError;
  }
  throw new Error("readTypedefs did not throw");
}

describe("typedef names", () => {
  test.each(["str", "list"])("%p fails PascalCase before the shadow check", (name) => {
    const error = typedefsError({ [name]: "int" });
    expect(error.message).toBe(
      `typedef name '${name}' must be PascalCase (an uppercase identifier)`,
    );
    expect(error.path).toEqual([name]);
  });

  test.each([
    "Optional",
    "Literal",
    "Any",
    "List",
    "Dict",
    "Union",
    "None",
    "NONE",
    "Str",
    "INT",
    "DateTime",
  ])("%p shadows a builtin", (name) => {
    const error = typedefsError({ [name]: "int" });
    expect(error.message).toBe(
      `typedef name '${name}' shadows a builtin scalar or typing constructor`,
    );
    expect(error.path).toEqual([name]);
  });

  // `String` was the old engine's name, no longer a scalar keyword.
  test.each(["ANY", "String", "Rating"])("%p loads", (name) => {
    expect(readTypedefs({ [name]: "int" }).has(name)).toBe(true);
  });
});
