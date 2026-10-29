/**
 * The `Any` type's shape, labels, overlap, parsing, typedefs and resolution; the values `Any`,
 * `object` and the list kinds take at the write boundary (`buildValueAs`) and through the wire;
 * its compatibility and proof checks; and the load-time binding checks and hints around it.
 */

import { afterEach, describe, expect, test } from "bun:test";

import { typeRelation, typesCompatible } from "../../src/compile/validation.ts";
import { requireProvenType } from "../../src/compose/build/inline_inputs.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import { forget } from "../../src/defaults.ts";
import { outputContractFieldError, ValueContract } from "../../src/nodes/contracts.ts";
import { logJson, parseJson } from "../../src/typesys/json.ts";
import { parseType, readTypedefs, typeFor } from "../../src/typesys/types.ts";
import {
  buildValue,
  buildValueAs,
  canonicalTypeLabel,
  canonicalUnionOrder,
  FileRef,
  hasFile,
  requiresFile,
  type StorageKind,
  Type,
  TypeCheckError,
  TypedValue,
  typeEquals,
  typesOverlap,
  type Value,
  ValueKind,
} from "../../src/typesys/values.ts";
import { assembled, assembleText, loadError, loadErrorAsync, runCase } from "../helpers/declarations.ts";
import { typeCheckError } from "../helpers/type_errors.ts";

const ANY = Type.scalar(ValueKind.ANY);
const OPTIONAL_ANY = new Type({ kind: ValueKind.ANY, nullable: true });
const LIST_ANY = new Type({ kind: ValueKind.LIST_ANY, element: ANY });
const LIST_OPTIONAL_ANY = new Type({ kind: ValueKind.LIST_ANY, element: OPTIONAL_ANY });
const UNPROVEN_LIST = Type.scalar(ValueKind.LIST_ANY);
const FILE = Type.scalar(ValueKind.FILE);

/** A closed record whose fields are all required. */
function rec(fields: Record<string, Type>): Type {
  return new Type({
    kind: ValueKind.OBJECT,
    fields: new Map(Object.entries(fields)),
    required: new Set(Object.keys(fields)),
  });
}

const FILE_RECORD = rec({ f: FILE });

const REGISTRY = readTypedefs({
  Buy: { action: "Literal(buy)", qty: "int" },
  Sell: { action: "Literal(sell)", qty: "int" },
  Rec: { a: "int" },
});

/** Resolve a type expression against the compatibility typedefs. */
const t = (text: string): Type => typeFor(text, REGISTRY);

describe("Any type shape", () => {
  test("is valid", () => {
    new Type({ kind: ValueKind.ANY }).validateShape();
  });

  test.each([
    ["fields", { fields: new Map<string, Type>() }, /only object types may declare record fields/],
    ["tags", { tags: new Set(["a"]) }, /only string types may declare variant tags/],
    ["bounds", { bounds: [0n, 1n] }, /only int and float types may declare bounds/],
    [
      "element",
      { element: Type.scalar(ValueKind.STRING) },
      /only list types may declare an element type/,
    ],
  ] as const)("rejects %s", (_name, refinement, message) => {
    const typ = new Type({ kind: ValueKind.ANY, ...refinement });
    expect(() => typ.validateShape()).toThrow(RangeError);
    expect(() => typ.validateShape()).toThrow(message);
  });
});

describe("Any labels", () => {
  test.each([
    ["Any", ANY],
    ["Optional(Any)", OPTIONAL_ANY],
    ["list(Any)", LIST_ANY],
    ["list(Optional(Any))", LIST_OPTIONAL_ANY],
  ] as const)("authored %s resolves and labels back", (text, typ) => {
    expect(typeEquals(typeFor(text, new Map()), typ)).toBe(true);
    expect(canonicalTypeLabel(typ)).toBe(text);
  });

  test("a list with unproven items is labelled list of Optional(Any)", () => {
    expect(canonicalTypeLabel(UNPROVEN_LIST)).toBe("list(Optional(Any))");
  });
});

describe("Any overlap", () => {
  test.each([
    ["str", t("str"), true],
    ["Rec", t("Rec"), true],
    ["list(int)", t("list(int)"), true],
    ["Any", ANY, true],
    ["None", t("None"), false],
    // A file slot reads a file's JSON form, which is an `Any` value.
    ["file", FILE, true],
    ["{f: file}", FILE_RECORD, true],
  ] as const)("Any overlaps every type but None: %s", (_id, other, expected) => {
    expect(typesOverlap(ANY, other)).toBe(expected);
    expect(typesOverlap(other, ANY)).toBe(expected);
  });

  test("a record with an Any field overlaps one with an int field", () => {
    const reg = readTypedefs({ A: { v: "Any" }, B: { v: "int" } });
    expect(typesOverlap(typeFor("A", reg), typeFor("B", reg))).toBe(true);
  });

  test("Any overlaps a list of files through the empty list", () => {
    const listOfFiles = new Type({ kind: ValueKind.LIST_ANY, element: FILE });
    expect(typesOverlap(ANY, listOfFiles)).toBe(true);
    expect(
      typedefsError({ A: { v: "Any" }, B: { v: "list(file)" }, W: "Union(A, B)" }),
    ).toBe(`typedef 'W': ${overlap("Union(A, B)", "A", "B")}`);
  });
});

describe("Any parsing", () => {
  test("parses as its own scalar", () => {
    expect(parseType("Any")).toEqual({ kind: "scalar", name: "Any" });
  });

  test("lowercase any is not a type", () => {
    expect(() => parseType("any")).toThrow(TypeCheckError);
    expect(() => parseType("any")).toThrow(/^unknown type 'any'; write Any$/);
  });
});

describe("Any in typedefs", () => {
  test("records that differ only in list item nullability sort deterministically", () => {
    const reg = readTypedefs({ A: { xs: "list(Any)" }, B: { xs: "list(Optional(Any))" } });
    const a = typeFor("A", reg);
    const b = typeFor("B", reg);
    for (const order of [canonicalUnionOrder([a, b]), canonicalUnionOrder([b, a])]) {
      expect(order).toHaveLength(2);
      expect(order[0]).toBe(a);
      expect(order[1]).toBe(b);
    }
  });

  test("a typedef cannot be named Any", () => {
    expect(typedefsError({ Any: "str" })).toBe(
      "typedef name 'Any' shadows a builtin scalar or typing constructor",
    );
  });

  test("Any resolves through an alias and a record field", () => {
    const reg = readTypedefs({ Payload: "Any", Rec: { data: "Any" } });
    expect(typeEquals(typeFor("Payload", reg), ANY)).toBe(true);
    expect(typeEquals(typeFor("Rec", reg).fields!.get("data")!, ANY)).toBe(true);
  });
});

/** Return the message of the `TypeCheckError` `readTypedefs(raw)` throws. */
function typedefsError(raw: Readonly<Record<string, unknown>>): string {
  try {
    readTypedefs(raw);
  } catch (error) {
    expect(error).toBeInstanceOf(TypeCheckError);
    return (error as TypeCheckError).message;
  }
  throw new Error("readTypedefs did not throw");
}

/** The overlap error for `label`, members `a` and `b`, with the `file` note if `files`. */
function overlap(label: string, a: string, b: string, files = false): string {
  const why = files
    ? " (a `file` slot reads `{uri, mime, name}`, so `file` overlaps `object`, `Any` and a " +
      "record of that shape)"
    : "";
  return (
    `${label}: members '${a}' and '${b}' can match the same value${why}; add a tag field ` +
    "every member declares as a single-value Literal, with a different value in each " +
    `member, e.g. kind: Literal(${a.toLowerCase()})`
  );
}

describe("Any in untagged unions", () => {
  test("a union of Any and int fields does not resolve", () => {
    expect(typedefsError({ A: { v: "Any" }, B: { v: "int" }, U: "Union(A, B)" })).toBe(
      `typedef 'U': ${overlap("Union(A, B)", "A", "B")}`,
    );
  });

  test("a union of Any and a union with a file member does not resolve", () => {
    const typedefs = {
      F: { f: "file" },
      N: { n: "int" },
      U: "Union(F, N)",
      OF: { f: "Optional(file)", n: "int" },
      A: { v: "Any" },
      B: { v: "U" },
      W: "Union(A, B)",
    };
    expect(typedefsError(typedefs)).toBe(
      `typedef 'W': ${overlap("Union(A, B)", "A", "B", true)}`,
    );
  });
});

/** Return `leaf` wrapped in `n` lists. */
function nest(n: number, leaf: Value = 1n): Value {
  let value = leaf;
  for (let i = 0; i < n; i++) value = [value];
  return value;
}

/** Return `value` with each scalar replaced by its JS type, keeping lists and records. */
function jsTypes(value: Value): unknown {
  if (Array.isArray(value)) return value.map(jsTypes);
  if (typeof value === "object" && value !== null) {
    return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, jsTypes(item)]));
  }
  return value === null ? "null" : typeof value;
}

class Foo {}

describe("Any values", () => {
  test.each<[string, Value, StorageKind]>([
    ["text", "hello", ValueKind.STRING],
    ["an int", 3n, ValueKind.INTEGER],
    ["a long int", 2n ** 80n, ValueKind.INTEGER],
    ["a float", 2.5, ValueKind.NUMBER],
    ["a bool", true, ValueKind.BOOLEAN],
    ["a record", { a: [1n, null] }, ValueKind.OBJECT],
    ["a mixed list", [1n, "two", null], ValueKind.LIST_ANY],
    ["ints and floats", [1n, 2.5], ValueKind.LIST_ANY],
    ["a bool and an int", [true, 1n], ValueKind.LIST_ANY],
  ])("Any stores a JSON value in its natural kind: %s", (_id, value, kind) => {
    const typed = buildValueAs(ANY, value);
    expect(typed.kind).toBe(kind);
    expect(typed.value).toEqual(value);
  });

  test("Any keeps list items as given", () => {
    const ints = buildValueAs(ANY, [1n, 2.5]).value as Value[];
    const bools = buildValueAs(ANY, [true, 1n]).value as Value[];
    expect(typeof ints[0]).toBe("bigint");
    expect(typeof bools[0]).toBe("boolean");
    expect(typeof bools[1]).toBe("bigint");
  });

  test("Any accepts a bare value kind", () => {
    const typed = buildValueAs(ValueKind.ANY, "hello");
    expect([typed.kind, typed.value]).toEqual([ValueKind.STRING, "hello"]);
  });

  test.each<[string, unknown, string, (string | number)[]]>([
    ["null", null, "null does not match declared type Any", []],
    ["a Set", new Set([1n, 2n]), "<Set object> is not a JSON value", []],
    ["bytes", new Uint8Array([120]), "<Uint8Array object> is not a JSON value", []],
    ["a Date", new Date(0), "<Date object> is not a JSON value", []],
    ["a Map", new Map(), "<Map object> is not a JSON value", []],
    ["undefined", undefined, "undefined is not a JSON value", []],
    [
      "a file",
      new FileRef({ uri: "f" }),
      '{"uri":"f","mime":null,"name":null} is not a JSON value',
      [],
    ],
    ["a class instance", new Foo(), "<Foo object> is not a JSON value", []],
    ["a nested Set", { a: new Set([1n]) }, "<Set object> is not a JSON value", ["a"]],
    ["a Set in a list", { a: [1n, new Set([2n])] }, "<Set object> is not a JSON value", ["a", 1]],
  ])("Any rejects a non-JSON value at its path: %s", (_id, value, message, path) => {
    const error = typeCheckError(() => buildValueAs(ANY, value));
    expect(error.message).toBe(message);
    expect(error.path).toEqual(path);
  });

  test("Any leaves the nesting limit to the boundaries", () => {
    expect(buildValueAs(ANY, nest(150)).value).toEqual(nest(150));
  });

  test("Any ends a cyclic value", () => {
    const cyclic: Record<string, Value> = {};
    cyclic["self"] = { items: [1n, cyclic] };
    const error = typeCheckError(() => buildValueAs(ANY, cyclic));
    expect(error.message).toBe("value contains itself");
    expect(error.path).toEqual(["self", "items", 1]);
  });

  test("Any accepts a shared value that is not a cycle", () => {
    const shared = [1n, 2n];
    expect(buildValueAs(ANY, { a: shared, b: shared }).value).toEqual({
      a: [1n, 2n],
      b: [1n, 2n],
    });
  });

  test("Any reports the first bad part in order", () => {
    const error = typeCheckError(() =>
      buildValueAs(ANY, { a: [1n, new Set([2n])], b: new Map() }),
    );
    expect(error.message).toBe("<Set object> is not a JSON value");
    expect(error.path).toEqual(["a", 1]);
  });

  test("Optional(Any) accepts null", () => {
    const typed = buildValueAs(OPTIONAL_ANY, null);
    expect([typed.kind, typed.value]).toEqual([ValueKind.NONE, null]);
  });

  test.each<[string, unknown, (string | number)[]]>([
    ["a Set", { a: new Set([1n, 2n]) }, ["a"]],
    ["a class instance", { a: new Foo() }, ["a"]],
    ["a Date", { d: new Date(0) }, ["d"]],
  ])("object rejects a non-JSON member: %s", (_id, value, path) => {
    const error = typeCheckError(() => buildValueAs(ValueKind.OBJECT, value));
    expect(error.message).toContain("is not a JSON value");
    expect(error.path).toEqual(path);
  });

  test("object accepts JSON members", () => {
    const value = { a: null, b: [1n, { c: "d" }] };
    const typed = buildValueAs(ValueKind.OBJECT, value);
    expect(typed.kind).toBe(ValueKind.OBJECT);
    expect(typed.value).toBe(value);
  });

  test("an elementless list(Any) checks its items are JSON values", () => {
    expect(buildValueAs(ValueKind.LIST_ANY, [1n, null, "x"]).value).toEqual([1n, null, "x"]);
    const error = typeCheckError(() => buildValueAs(ValueKind.LIST_ANY, [new Set([1n])]));
    expect(error.message).toContain("is not a JSON value");
    expect(error.path).toEqual([0]);
  });

  test("a record with a required Any field rejects null", () => {
    expect(typeCheckError(() => buildValueAs(rec({ v: ANY }), { v: null })).path).toEqual(["v"]);
  });

  test("a record with an Optional(Any) field accepts absence and null", () => {
    const record = new Type({
      kind: ValueKind.OBJECT,
      fields: new Map([["v", OPTIONAL_ANY]]),
      required: new Set(),
    });
    expect(buildValueAs(record, {}).value).toEqual({});
    expect(buildValueAs(record, { v: null }).value).toEqual({ v: null });
  });

  test("list(Any) rejects a null item and list(Optional(Any)) accepts it", () => {
    const error = typeCheckError(() => buildValueAs(typeFor("list(Any)", new Map()), [1n, null]));
    expect(error.path).toEqual([1]);
    const typed = buildValueAs(typeFor("list(Optional(Any))", new Map()), [1n, null]);
    expect([typed.kind, typed.value]).toEqual([ValueKind.LIST_ANY, [1n, null]]);
  });

  test("the issue examples pass", () => {
    const text = buildValueAs(typeFor("Any", new Map()), "hello");
    expect([text.kind, text.value]).toEqual([ValueKind.STRING, "hello"]);
    const list = buildValueAs(typeFor("list(Any)", new Map()), [1n, "two"]);
    expect([list.kind, list.value]).toEqual([ValueKind.LIST_ANY, [1n, "two"]]);
  });

  test.each(["list(list(int))", "list(Any)"])(
    "a refined list names its declared type: %s",
    (text) => {
      const error = typeCheckError(() => buildValueAs(typeFor(text, new Map()), 3n));
      expect(error.message).toBe(`3 is not a list for declared ${text}`);
    },
  );

  test.each<[string, Value]>([
    ["text", "hello"],
    ["an int", 3n],
    ["a long int", 2n ** 80n],
    ["a float", 2.5],
    ["a bool", true],
    ["a record", { a: [1n, null, 2.5, true] }],
    ["a mixed list", [1n, "two", null]],
    ["ints and floats", [1n, 2.5]],
    ["a bool and a record", [true, { b: [1n, 2.0] }]],
  ])("an Any value round-trips through the wire: %s", (_id, value) => {
    const typed = buildValueAs(ANY, value);
    const decoded = TypedValue.fromWire(parseJson(logJson(typed.toWire()), "codec"));
    expect(decoded.kind).toBe(typed.kind);
    expect(decoded.value).toEqual(typed.value);
    expect(jsTypes(decoded.value)).toEqual(jsTypes(typed.value));
  });
});

describe("field reads on Any", () => {
  test("a field read on Any is not checked", () => {
    const contract = new ValueContract(ANY);
    expect(outputContractFieldError(contract, ["anything"], "xs.anything")).toBe(null);
  });

  test("an assert reading a field of an Any output loads", () => {
    assembleText(
      "id: p\nname: p\n" +
        "input:\n  s: !str\n" +
        "nodes:\n" +
        "  step:\n" +
        "    kind: code\n" +
        "    input:\n" +
        "      x: ${input.s}\n" +
        "    output: Any\n" +
        "    code: return null\n" +
        "    asserts:\n" +
        "      - output.a.b == 1\n" +
        "output: ${step.output}\n",
    );
  });

  test("assembled: load-types/engine_any_type__an_assert_reading_a_field_of_an_any_output_loads", () => {
    runCase({ id: "load-types/engine_any_type__an_assert_reading_a_field_of_an_any_output_loads", call: assembled });
  });
});

const LIST_FILE = new Type({ kind: ValueKind.LIST_ANY, element: FILE });

const FILE_TYPEDEFS = {
  F: { f: "file" },
  N: { n: "int" },
  U: "Union(F, N)",
  OF: { f: "Optional(file)", n: "int" },
};

describe("compatibility with Any", () => {
  test("unproven list items compare as Optional(Any)", () => {
    const listStr = new Type({
      kind: ValueKind.LIST_STRING,
      element: Type.scalar(ValueKind.STRING),
    });
    const listFile = new Type({ kind: ValueKind.LIST_ANY, element: FILE });

    expect(typesCompatible(UNPROVEN_LIST, typeFor("list(Optional(Any))", new Map()))).toBe(true);
    expect(typesCompatible(listStr, UNPROVEN_LIST)).toBe(true);
    expect(typesCompatible(listFile, UNPROVEN_LIST)).toBe(false);
    expect(typeRelation(UNPROVEN_LIST, listStr)).toBe("narrower");
  });

  test("proof rejects an unproven list and accepts list(Optional(Any))", () => {
    const error = loadError(() => requireProvenType(UNPROVEN_LIST, { bindingLabel: "call.input.xs" }));
    expect(error.message).toBe(
      "cannot infer type for inline binding 'call.input.xs': list type 'list(Optional(Any))' has no " +
        "proven element type; declare its type with a tag (`xs: !<type> ...`) or use an explicitly " +
        "typed named `defs:` body",
    );

    const proven = typeFor("list(Optional(Any))", new Map());
    expect(requireProvenType(proven, { bindingLabel: "call.input.xs" })).toBe(proven);
  });

  test.each([
    "str",
    "int",
    "float",
    "bool",
    "date",
    "datetime",
    "object",
    "Rec",
    "list(int)",
    "list(Any)",
    "list(Optional(Any))",
    "Literal(a)",
    "int(1..3)",
    "int(0..)",
    "Union(Buy, Sell)",
    "Any",
  ])("a non-null JSON type fits Any: %s", (text) => {
    expect(typesCompatible(t(text), ANY)).toBe(true);
    expect(typeRelation(t(text), ANY)).toBe("fits");
  });

  test("an unproven list fits Any", () => {
    expect(typesCompatible(UNPROVEN_LIST, ANY)).toBe(true);
    expect(typeRelation(UNPROVEN_LIST, ANY)).toBe("fits");
  });

  test.each<[string, Type]>([
    ["Optional(int)", t("Optional(int)")],
    ["None", t("None")],
    ["Optional(Any)", OPTIONAL_ANY],
  ])("a nullable source does not fit Any: %s", (_id, source) => {
    expect(typesCompatible(source, ANY)).toBe(false);
  });

  test.each<[string, Type]>([
    ["Optional(int)", t("Optional(int)")],
    ["None", t("None")],
    ["Optional(Any)", OPTIONAL_ANY],
    ["Any", ANY],
  ])("a nullable source fits Optional(Any): %s", (_id, source) => {
    expect(typesCompatible(source, OPTIONAL_ANY)).toBe(true);
  });

  test.each([
    "str",
    "int",
    "float",
    "object",
    "Rec",
    "Union(Buy, Sell)",
    "list(Any)",
    "list(str)",
    "Literal(a)",
    "int(1..3)",
    "Optional(str)",
    "None",
  ])("Any fits nothing but Any: %s", (sink) => {
    expect(typesCompatible(ANY, t(sink))).toBe(false);
  });

  test.each<[string, Type]>([
    ["file", FILE],
    ["list(file)", LIST_FILE],
    ["{f: file}", FILE_RECORD],
  ])("a type holding a file does not fit Any: %s", (_id, source) => {
    expect(typesCompatible(source, ANY)).toBe(false);
    expect(typeRelation(source, ANY)).toBe("disjoint");
  });

  test("a list fits list(Any) when its items fit Any", () => {
    expect(typesCompatible(t("list(int)"), LIST_ANY)).toBe(true);
    expect(typesCompatible(t("list(Rec)"), LIST_ANY)).toBe(true);
    expect(typesCompatible(t("list(Optional(int))"), LIST_ANY)).toBe(false);
    expect(typesCompatible(LIST_FILE, LIST_ANY)).toBe(false);
  });

  test.each<[string, Type]>([
    ["list(int)", t("list(int)")],
    ["list(Optional(str))", t("list(Optional(str))")],
    ["list(Any)", LIST_ANY],
    ["list(Optional(Any))", LIST_OPTIONAL_ANY],
    ["an unproven list", UNPROVEN_LIST],
  ])("a list of JSON items fits list(Optional(Any)): %s", (_id, source) => {
    expect(typesCompatible(source, LIST_OPTIONAL_ANY)).toBe(true);
  });

  test.each<[string, Type, string, Type]>([
    ["list(Optional(Any))", LIST_OPTIONAL_ANY, "list(Any)", LIST_ANY],
    ["list(Optional(Any))", LIST_OPTIONAL_ANY, "list(str)", t("list(str)")],
    ["an unproven list", UNPROVEN_LIST, "list(Any)", LIST_ANY],
    ["an unproven list", UNPROVEN_LIST, "list(str)", t("list(str)")],
  ])(
    "a list of Optional(Any) does not fit a list of non-null items: %s into %s",
    (_source, source, _sink, sink) => {
      expect(typesCompatible(source, sink)).toBe(false);
    },
  );

  test.each<[string, Type]>([
    ["Optional(int)", t("Optional(int)")],
    ["Optional(Any)", OPTIONAL_ANY],
    ["Optional(Rec)", t("Optional(Rec)")],
  ])("a nullable JSON type is narrower than Any: %s", (_id, inferred) => {
    expect(typeRelation(inferred, ANY)).toBe("narrower");
  });

  test.each([
    "str",
    "int",
    "float(..2.5)",
    "int(1..3)",
    "int(0..)",
    "object",
    "Rec",
    "Buy",
    "Union(Buy, Sell)",
    "list(str)",
    "list(Any)",
    "date",
    "Literal(a)",
    "Optional(str)",
  ])("Any is narrower than any other type: %s", (declared) => {
    expect(typeRelation(ANY, t(declared))).toBe("narrower");
  });

  test.each<[string, Type]>([
    ["Any", ANY],
    ["Optional(Any)", OPTIONAL_ANY],
  ])("Any fits Any and Optional(Any): %s", (_id, declared) => {
    expect(typesCompatible(ANY, declared)).toBe(true);
    expect(typeRelation(ANY, declared)).toBe("fits");
  });

  test.each<[string, Type]>([
    ["file", FILE],
    ["None", t("None")],
  ])("Any is disjoint from file and None: %s", (_id, declared) => {
    expect(typeRelation(ANY, declared)).toBe("disjoint");
  });

  test("Any is narrower than a union with one file member, and back", () => {
    const union = typeFor("U", readTypedefs(FILE_TYPEDEFS));
    expect(typeRelation(ANY, union)).toBe("narrower");
    expect(typeRelation(union, ANY)).toBe("narrower");
    expect(typesOverlap(ANY, union)).toBe(true);
    expect(typesOverlap(union, ANY)).toBe(true);
  });

  test("Any is narrower than a record whose file field may be null", () => {
    const record = typeFor("OF", readTypedefs(FILE_TYPEDEFS));
    expect(typeRelation(ANY, record)).toBe("narrower");
    expect(typeRelation(record, ANY)).toBe("narrower");
  });
});

describe("requiresFile and hasFile", () => {
  test.each<[string, Type]>([
    ["file", FILE],
    ["Optional(file)", new Type({ kind: ValueKind.FILE, nullable: true })],
    ["list(file)", LIST_FILE],
    ["a record with a file field", FILE_RECORD],
    [
      "a union of records with a file field",
      new Type({ kind: ValueKind.OBJECT, members: [FILE_RECORD, FILE_RECORD] }),
    ],
  ])("requires a file when every non-null value holds one: %s", (_id, typ) => {
    expect(requiresFile(typ)).toBe(true);
    expect(hasFile(typ)).toBe(true);
  });

  test.each(["U", "OF"])("a type that may hold a file does not require one: %s", (text) => {
    const typ = typeFor(text, readTypedefs(FILE_TYPEDEFS));
    expect(hasFile(typ)).toBe(true);
    expect(requiresFile(typ)).toBe(false);
  });

  test("requires a non-null list element and a required field", () => {
    const optionalFile = new Type({ kind: ValueKind.FILE, nullable: true });
    const optionalField = new Type({
      kind: ValueKind.OBJECT,
      fields: new Map([["f", FILE]]),
      required: new Set(),
    });
    expect(requiresFile(new Type({ kind: ValueKind.LIST_ANY, element: optionalFile }))).toBe(
      false,
    );
    expect(requiresFile(optionalField)).toBe(false);
    expect(requiresFile(ANY)).toBe(false);
  });
});

describe("loading flows that wire Any", () => {
  afterEach(() => forget());

  /**
   * A flow whose `c` node calls `child`, taking `t: !<childType>`, with `binding`; its producer
   * `step` is typed `stepOutput` (untyped when `null`), and `c` maps over `${step.output}` with
   * `over`.
   */
  function callFlow(
    childType: string,
    binding: string,
    options: { stepOutput?: string | null; over?: boolean } = {},
  ): string {
    const { stepOutput = "Any", over = false } = options;
    const output = stepOutput === null ? "" : `    output: ${stepOutput}\n`;
    const node = over ? "    kind: map\n    over: ${step.output}\n" : "    kind: call\n";
    return (
      "id: p\nname: p\n" +
      "typedefs:\n  Rec:\n    a: int\n  F:\n    f: file\n" +
      "defs:\n  child:\n" +
      `    input:\n      t: !${childType}\n` +
      "    nodes:\n      n:\n        kind: code\n        input:\n          t: ${input.t}\n" +
      "        output: str\n        code: return JSON.stringify(inputs['t'])\n" +
      "    output: ${n.output}\n" +
      `nodes:\n  step:\n    kind: code\n${output}    code: return 'hello'\n` +
      `  c:\n${node}    call: child\n    input:\n      t: ${binding}\n` +
      "output: ${c.output}\n"
    );
  }

  /** The message of the load error loading `callFlow(childType, binding, options)` raises. */
  const message = async (
    childType: string,
    binding: string,
    options: { stepOutput?: string | null; over?: boolean } = {},
  ) => (await loadErrorAsync(loadFlow(callFlow(childType, binding, options)))).message;

  test("tagging an Any output as a union with a file member loads", () => {
    assembleText(
      "id: p\nname: p\n" +
        "typedefs:\n  F:\n    f: file\n  N:\n    n: int\n  U: Union(F, N)\n" +
        "nodes:\n" +
        "  step:\n    kind: code\n    output: Any\n    code: \"return {n: 1n}\"\n" +
        "  use:\n    kind: code\n    input:\n      x: !U ${step.output}\n" +
        "    output: str\n    code: return 'ok'\n" +
        "output: ${use.output}\n",
    );
  });

  test("Any into a child str input suggests a tag or isinstance", async () => {
    expect(await message("str", "${step.output}")).toBe(
      "call node 'c': binding 't' — child expects 'str', source is 'Any'; " +
        "tag the slot (`t: !str ${...}`) to check it at runtime, or narrow it with `isinstance` in a `case`",
    );
  });

  test("the Any hint names the typedef of a child record input", async () => {
    expect(await message("Rec", "${step.output}")).toBe(
      "call node 'c': binding 't' — child expects '{\"a\":int}', source is 'Any'; " +
        "tag the slot (`t: !Rec ${...}`) to check it at runtime, or narrow it with `isinstance` in a `case`",
    );
  });

  test("optional Any into a child Any input is a plain nullability error", async () => {
    expect(await message("Any", "${step.output}", { stepOutput: "Optional(Any)" })).toBe(
      "call node 'c': binding 't' — child expects 'Any', source is 'Optional(Any)'",
    );
  });

  test("Any into a child input that requires a file gets no hint", async () => {
    expect(await message("F", "${step.output}")).toBe(
      "call node 'c': binding 't' — child expects '{\"f\":file}', source is 'Any'",
    );
  });

  test("a tagged Any binding into a child str input loads", async () => {
    await loadFlow(callFlow("str", "!str ${step.output}"));
  });

  test("an untyped source into a child Any input is left to runtime", async () => {
    await loadFlow(callFlow("Any", "${step.output}", { stepOutput: null }));
  });

  test("a map over a list of Any into a child str input suggests a tag", async () => {
    expect(await message("str", "${item}", { stepOutput: "list(Any)", over: true })).toBe(
      "map node 'c': binding 't' — child expects 'str', source is 'Any'; " +
        "tag the slot (`t: !str ${...}`) to check it at runtime, or narrow it with `isinstance` in a `case`",
    );
  });

  test("a map over a list of Any into a child Any input loads", async () => {
    await loadFlow(callFlow("Any", "${item}", { stepOutput: "list(Any)", over: true }));
  });

  const ANY_INTO_LOOP = `id: p
name: p
defs:
  bump:
    input:
      n: !int
    nodes:
      step:
        kind: code
        input:
          n: \${input.n}
        output: int
        code: return inputs["n"] + 1n
    output:
      n: \${step.output}
nodes:
  seed:
    kind: code
    output: Any
    code: return 1n
  loop:
    kind: loop
    times: 2
    input:
      n: SEED
    call: bump
output: \${loop.output}
`;

  /** Load the loop flow with `seed` as its `n:` seed. */
  const loadLoop = (seed: string) => loadFlow(ANY_INTO_LOOP.replace("SEED", seed));

  test("Any into a loop carried int suggests a tag or isinstance", async () => {
    expect((await loadErrorAsync(loadLoop("${seed.output}"))).message).toBe(
      "loop node 'loop': binding 'n' — child expects 'int', source is 'Any'; " +
        "tag the slot (`n: !int ${...}`) to check it at runtime, or narrow it with `isinstance` in a `case`",
    );
  });

  test("a tagged loop seed over an Any output loads", async () => {
    await loadLoop("!int ${seed.output}");
  });

  test("assembled: load-types/engine_any_type__any_into_a_loop_carried_int_suggests_a_tag_or_isinstance", () => {
    runCase({
      id: "load-types/engine_any_type__any_into_a_loop_carried_int_suggests_a_tag_or_isinstance",
      call: assembled,
    });
  });

  test("assembled: load-types/engine_any_type__a_tagged_loop_seed_over_an_any_output_loads", () => {
    runCase({
      id: "load-types/engine_any_type__a_tagged_loop_seed_over_an_any_output_loads",
      call: assembled,
    });
  });
});

describe("-0, NaN and lone surrogates in object and Any values", () => {
  test.each([ValueKind.OBJECT, ValueKind.ANY])(
    "%s turns -0 into 0, copying only what changes",
    (kind) => {
      const keep = { x: 1n };
      const value = { a: { n: -0 }, keep };
      const built = buildValueAs(kind, value).value as Record<string, Record<string, Value>>;
      expect(Object.is(built["a"]!["n"], 0)).toBe(true);
      expect(built["keep"]).toBe(keep);
      expect(Object.is(value.a.n, -0)).toBe(true);
    },
  );

  test.each<[string, ValueKind, unknown, string, (string | number)[]]>([
    ["NaN inside Any", ValueKind.ANY, { a: [NaN] }, "NaN is not a yian value", ["a", 0]],
    ["NaN inside object", ValueKind.OBJECT, { a: NaN }, "NaN is not a yian value", ["a"]],
    [
      "a lone surrogate inside Any",
      ValueKind.ANY,
      ["\ud800"],
      "text holding a lone surrogate is not a yian value",
      [0],
    ],
    [
      "a lone surrogate key inside object",
      ValueKind.OBJECT,
      { a: { "\udc00": 1n } },
      "text holding a lone surrogate is not a yian value",
      ["a"],
    ],
    [
      "a lone surrogate key at Any's root",
      ValueKind.ANY,
      { "\ud800": 1n },
      "text holding a lone surrogate is not a yian value",
      [],
    ],
  ])("refuses %s", (_id, kind, value, message, path) => {
    const error = typeCheckError(() => buildValueAs(kind, value));
    expect(error.message).toBe(message);
    expect(error.path).toEqual(path);
  });
});

describe("an array whose prototype was swapped", () => {
  // An array is never a record, whatever its prototype, and it is a list only with
  // `Array.prototype`, so its log line can always be read back.
  const swapped = () => Object.setPrototypeOf([1n, 2n], Object.prototype) as unknown;

  test("buildValue can't wrap it", () => {
    const error = typeCheckError(() => buildValue(swapped()));
    expect(error.message).toBe("cannot wrap value of type 'Object'");
    expect(error.path).toEqual([]);
  });

  test("Any refuses it as not a JSON value", () => {
    const error = typeCheckError(() => buildValueAs(ValueKind.ANY, swapped()));
    expect(error.message).toBe("<Object object> is not a JSON value");
    expect(error.path).toEqual([]);
  });
});
