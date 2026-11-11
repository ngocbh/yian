/**
 * The typed value system: inference (`buildValue`), the declared-type write boundary
 * (`buildValueAs`) and the wire form. A round trip goes through `toWire`, the run log's encoder and
 * lossless reader, and `TypedValue.fromWire`, and checks the kind and the JS type of the value.
 */

import { describe, expect, test } from "bun:test";

import { logJson, parseJson } from "../../src/typesys/json.ts";
import {
  buildValue,
  buildValueAs,
  FileRef,
  fromHost,
  type StorageKind,
  Type,
  TypeCheckError,
  TypedValue,
  type Value,
  ValueKind,
} from "../../src/typesys/values.ts";
import { typeCheckError } from "../helpers/type_errors.ts";

/** Return `typed` after a run-log round trip: `toWire`, `logJson`, `parseJson` and `fromWire`. */
function roundTrip(typed: TypedValue): TypedValue {
  return TypedValue.fromWire(parseJson(logJson(typed.toWire()), "codec"));
}

/** Return the JS type of `value`: `typeof`, or `array` / `record` for a container. */
function jsType(value: Value): string {
  if (Array.isArray(value)) return "array";
  if (typeof value === "object" && value !== null) {
    return Object.getPrototypeOf(value) === Object.prototype ? "record" : "other object";
  }
  return typeof value;
}

/** Expect `run` to throw a `TypeCheckError` with exactly `message` at `path`. */
function expectTypeCheckError(
  run: () => unknown,
  message: string,
  path: readonly (string | number)[],
): void {
  const error = typeCheckError(run);
  expect(error.message).toBe(message);
  expect(error.path).toEqual(path);
}

/** Return a record type whose fields are `fields`, requiring `required` (default: all). */
function record(fields: Record<string, Type>, required = Object.keys(fields)): Type {
  return new Type({
    kind: ValueKind.OBJECT,
    fields: new Map(Object.entries(fields)),
    required: new Set(required),
  });
}

class Foo {}
class Sub extends Array<unknown> {}

describe("inference", () => {
  test("infers scalars", () => {
    expect(buildValue("hi").kind).toBe(ValueKind.STRING);
    expect(buildValue(3n).kind).toBe(ValueKind.INTEGER);
    expect(buildValue(3.5).kind).toBe(ValueKind.NUMBER);
    expect(buildValue({ a: 1n }).kind).toBe(ValueKind.OBJECT);
    expect(buildValue(null).kind).toBe(ValueKind.NONE);
  });

  test("a bool is not an int", () => {
    expect(buildValue(true).kind).toBe(ValueKind.BOOLEAN);
    expect(buildValue(1n).kind).toBe(ValueKind.INTEGER);
  });

  test("infers lists", () => {
    expect(buildValue(["a", "b"]).kind).toBe(ValueKind.LIST_STRING);
    // A list mixing ints and floats is a list of floats; its ints stay ints.
    const mixed = buildValue([1n, 2.0]);
    expect(mixed.kind).toBe(ValueKind.LIST_NUMBER);
    expect(mixed.value).toEqual([1n, 2]);
    const halves = buildValue([1n, 2.5]);
    expect(halves.kind).toBe(ValueKind.LIST_NUMBER);
    expect(halves.value).toEqual([1n, 2.5]);
    expect(buildValue([]).kind).toBe(ValueKind.LIST_ANY);
    expect(buildValue([1n, "a"]).kind).toBe(ValueKind.LIST_ANY);
  });

  test("a file is never inferred", () => {
    // A record that looks like a file stays an object; only a FileRef is a file.
    expect(buildValue({ uri: "s3://x" }).kind).toBe(ValueKind.OBJECT);
    expect(buildValue(new FileRef({ uri: "s3://x" })).kind).toBe(ValueKind.FILE);
  });

  test("is idempotent", () => {
    const typed = buildValue(5n);
    expect(buildValue(typed)).toBe(typed);
  });

  test("an unwrappable value throws", () => {
    expectTypeCheckError(() => buildValue(new Foo()), "cannot wrap value of type 'Foo'", []);
    expectTypeCheckError(() => buildValue(new Date(0)), "cannot wrap value of type 'Date'", []);
  });

  test.each(["2026-06-08", "2026-06-12T14:30:00+00:00"])(
    "a plain string is not inferred as a date or datetime: %p",
    (text) => {
      expect(buildValue(text).kind).toBe(ValueKind.STRING);
    },
  );

  // TS-only: buildValue walks its value, keeping files and refusing what isn't JSON.
  test.each<[string, Value]>([
    ["at the root", new FileRef({ uri: "f" })],
    ["as an item", [new FileRef({ uri: "f" })]],
    ["as a field", { f: new FileRef({ uri: "f" }) }],
  ])("keeps a file %s, which an Any slot refuses", (_id, value) => {
    expect(buildValue(value).value).toBe(value);
    expect(typeCheckError(() => buildValueAs(ValueKind.ANY, value)).message).toContain(
      "is not a JSON value",
    );
  });

  test.each<[string, Value]>([
    ["null", null],
    ["true", true],
    ["a long int", 2n ** 80n],
    ["a float", 1.5],
    ["text", "x"],
    ["nested containers", [null, { a: [] }]],
  ])("accepts a JSON value: %s", (_id, value) => {
    expect(buildValue(value).value).toBe(value);
  });

  test("turns -0 into 0 at any depth, copying only what changes", () => {
    const zero = buildValue(-0);
    expect(zero.kind).toBe(ValueKind.NUMBER);
    expect(Object.is(zero.value, 0)).toBe(true);

    const list = buildValue([0.5, -0]);
    expect(list.kind).toBe(ValueKind.LIST_NUMBER);
    expect(Object.is((list.value as Value[])[1], 0)).toBe(true);

    const keep = { a: 1n };
    const value = { n: [-0], keep };
    const built = buildValue(value).value as Record<string, Value>;
    expect(Object.is((built["n"] as Value[])[0], 0)).toBe(true);
    expect(built["keep"]).toBe(keep);
    expect(Object.is(value.n[0], -0)).toBe(true);
  });

  test.each<[string, unknown, string, (string | number)[]]>([
    ["NaN at the root", NaN, "NaN is not a yian value", []],
    ["NaN in a record", { a: [1, NaN] }, "NaN is not a yian value", ["a", 1]],
    [
      "a lone surrogate at the root",
      "\ud800",
      "text holding a lone surrogate is not a yian value",
      [],
    ],
    [
      "a lone surrogate in a record",
      { a: { b: "x\udc00" } },
      "text holding a lone surrogate is not a yian value",
      ["a", "b"],
    ],
    [
      "a lone surrogate in a key",
      { a: { "\ud800": 1n } },
      "text holding a lone surrogate is not a yian value",
      ["a"],
    ],
    ["a Date in a record", { a: new Date(0) }, "Date is not a yian value", ["a"]],
    ["a Map in a list", [new Map()], "Map is not a yian value", [0]],
    ["a class instance in a record", { a: new Foo() }, "Foo is not a yian value", ["a"]],
    ["undefined in a record", { a: undefined }, "undefined is not a yian value; use null", ["a"]],
    ["a hole in a list", [1n, , 2n], "undefined is not a yian value; use null", [1]],
    ["a symbol in a list", [Symbol("s")], "symbol is not a yian value", [0]],
  ])("refuses %s", (_id, value, message, path) => {
    expectTypeCheckError(() => buildValue(value), message, path);
  });

  test.each<[string, unknown]>([
    ["a Date", new Date(0)],
    ["a class instance", new Foo()],
    ["undefined", undefined],
    ["a symbol", Symbol("s")],
    ["NaN", NaN],
    ["a lone surrogate", "x\udc00"],
  ])("refuses %s in the words host intake uses", (_id, part) => {
    const intake = typeCheckError(() => fromHost({ a: part }, "output")).message;
    expect(intake).toBe(`output at a: ${typeCheckError(() => buildValue({ a: part })).message}`);
  });

  test("refuses a value that contains itself", () => {
    const cyclic: Value[] = [];
    cyclic.push(cyclic);
    expectTypeCheckError(() => buildValue(cyclic), "value contains itself", [0]);
  });
});

describe("declared-type write boundary", () => {
  test("an int in a float slot stays an int", () => {
    const typed = buildValueAs(ValueKind.NUMBER, 7n);
    expect(typed.kind).toBe(ValueKind.INTEGER);
    expect(typed.value).toBe(7n);
    const list = buildValueAs(ValueKind.LIST_NUMBER, [1n, 2.5]);
    expect(list.kind).toBe(ValueKind.LIST_NUMBER);
    expect(list.value).toEqual([1n, 2.5]);
  });

  test("rejects a mismatch", () => {
    expectTypeCheckError(
      () => buildValueAs(ValueKind.INTEGER, "not-an-int"),
      '"not-an-int" does not match declared type int',
      [],
    );
    // A bool is not an int here.
    expectTypeCheckError(
      () => buildValueAs(ValueKind.INTEGER, true),
      "true does not match declared type int",
      [],
    );
  });

  test("a typed list validates its elements", () => {
    expect(buildValueAs(ValueKind.LIST_STRING, ["ACME", "BETA"]).kind).toBe(
      ValueKind.LIST_STRING,
    );
    expect(() => buildValueAs(ValueKind.LIST_STRING, ["ACME", 3n])).toThrow(TypeCheckError);
  });

  test.each<[StorageKind, ValueKind, Value]>([
    [ValueKind.LIST_STRING, ValueKind.STRING, "alpha"],
    [ValueKind.LIST_INTEGER, ValueKind.INTEGER, 7n],
    [ValueKind.LIST_NUMBER, ValueKind.NUMBER, 1.5],
    [ValueKind.LIST_BOOLEAN, ValueKind.BOOLEAN, true],
    [ValueKind.LIST_OBJECT, ValueKind.OBJECT, { name: "alpha" }],
  ])("a specialized %s rejects null without a nullable element", (listKind, element, item) => {
    const refined = new Type({ kind: listKind, element: Type.scalar(element) });
    expect(typeCheckError(() => buildValueAs(refined, [item, null])).path).toEqual([1]);
    expect(typeCheckError(() => buildValueAs(listKind, [item, null])).path).toEqual([1]);
  });

  test("a date rejects other values", () => {
    expect(() => buildValueAs(ValueKind.DATE, "not-a-date")).toThrow(TypeCheckError);
    expect(() => buildValueAs(ValueKind.DATE, 20260608n)).toThrow(TypeCheckError);
  });

  test("a datetime rejects other values", () => {
    expect(() => buildValueAs(ValueKind.DATETIME, "not-a-datetime")).toThrow(TypeCheckError);
    expect(() => buildValueAs(ValueKind.DATETIME, 20260612n)).toThrow(TypeCheckError);
  });

  test("a date is not a datetime, nor the reverse", () => {
    expect(() => buildValueAs(ValueKind.DATETIME, "2026-06-12")).toThrow(TypeCheckError);
    expect(() => buildValueAs(ValueKind.DATE, "2026-06-12T14:30:00+00:00")).toThrow(
      TypeCheckError,
    );
  });

  test("takes a Type or a bare kind", () => {
    expect(buildValueAs(Type.scalar(ValueKind.STRING), "x").kind).toBe(ValueKind.STRING);
    const typed = buildValueAs(ValueKind.NUMBER, 3n);
    expect(typed.kind).toBe(ValueKind.INTEGER);
    expect(typed.value).toBe(3n);
  });

  test("unwraps a TypedValue", () => {
    const typed = buildValueAs(ValueKind.NUMBER, buildValue(3n));
    expect(typed.kind).toBe(ValueKind.INTEGER);
    expect(typed.value).toBe(3n);
  });

  test("checks variant membership", () => {
    const action = new Type({
      kind: ValueKind.STRING,
      tags: new Set(["Approve", "Reject", "Defer"]),
    });
    expect(buildValueAs(action, "Approve").value).toBe("Approve");
    expectTypeCheckError(
      () => buildValueAs(action, "approve"),
      '"approve" is not one of Literal(Approve, Defer, Reject)',
      [],
    );
  });

  test("checks record fields", () => {
    const rating = record({
      value: Type.scalar(ValueKind.NUMBER),
      confidence: Type.scalar(ValueKind.NUMBER),
    });
    expect(buildValueAs(rating, { value: 0.8, confidence: 0.9 }).kind).toBe(ValueKind.OBJECT);
    expect(() => buildValueAs(rating, { value: 0.8 })).toThrow(TypeCheckError);
    expect(() => buildValueAs(rating, { value: 0.8, confidence: 0.9, x: 1n })).toThrow(
      TypeCheckError,
    );
    expect(() => buildValueAs(rating, { value: "hi", confidence: 0.9 })).toThrow(TypeCheckError);
    expectTypeCheckError(
      () => buildValueAs(rating, [0.8]),
      "[0.8] is not an object for a record type",
      [],
    );
  });

  test("a nullable field accepts null and absence", () => {
    const sig = record(
      {
        score: Type.scalar(ValueKind.NUMBER),
        note: new Type({ kind: ValueKind.STRING, nullable: true }),
      },
      ["score"],
    );
    expect(buildValueAs(sig, { score: 0.5, note: null }).kind).toBe(ValueKind.OBJECT);
    expect(buildValueAs(sig, { score: 0.5 }).kind).toBe(ValueKind.OBJECT);
    expect(buildValueAs(sig, { score: 0.5, note: "hi" }).kind).toBe(ValueKind.OBJECT);
    expect(() => buildValueAs(sig, { score: null, note: "x" })).toThrow(TypeCheckError);
  });

  test("a record keeps the value's key order", () => {
    const typ = record({ a: Type.scalar(ValueKind.INTEGER), b: Type.scalar(ValueKind.INTEGER) });
    expect(Object.keys(buildValueAs(typ, { b: 1n, a: 2n }).value as object)).toEqual(["b", "a"]);
  });

  test("a list of records", () => {
    const rating = record({ value: Type.scalar(ValueKind.NUMBER) });
    const list = new Type({ kind: ValueKind.LIST_OBJECT, element: rating });
    const typed = buildValueAs(list, [{ value: 1.0 }, { value: 2.0 }]);
    expect(typed.kind).toBe(ValueKind.LIST_OBJECT);
    expect(typed.value).toHaveLength(2);
    expect(() => buildValueAs(list, [{ value: "bad" }])).toThrow(TypeCheckError);
  });

  test("a nested failure carries the contract-relative path", () => {
    const item = record({ score: Type.scalar(ValueKind.INTEGER) });
    const payload = record({ items: new Type({ kind: ValueKind.LIST_OBJECT, element: item }) });
    const root = record({ payload });
    // Values in messages use the text form: text is JSON-quoted.
    expectTypeCheckError(
      () => buildValueAs(root, { payload: { items: [{ score: 1n }, { score: "bad" }] } }),
      '"bad" does not match declared type int',
      ["payload", "items", 1, "score"],
    );
    const container = typeCheckError(() =>
      buildValueAs(root, { payload: { items: "not-a-list" } }),
    );
    expect(container.path).toEqual(["payload", "items"]);
  });

  test("a flat list failure carries the actual element index", () => {
    // Values in messages use the text form: text is JSON-quoted.
    expectTypeCheckError(
      () => buildValueAs(ValueKind.LIST_INTEGER, [1n, "bad", 3n]),
      'list element "bad" is not int (declared list(int))',
      [1],
    );
  });

  test("record shape failures choose one deterministic field path", () => {
    const typ = record({
      alpha: Type.scalar(ValueKind.INTEGER),
      beta: Type.scalar(ValueKind.INTEGER),
    });
    expectTypeCheckError(
      () => buildValueAs(typ, {}),
      'record missing required fields: ["alpha","beta"]',
      ["alpha"],
    );
    expectTypeCheckError(
      () => buildValueAs(typ, { alpha: 1n, beta: 2n, zeta: 3n, extra: 4n }),
      'record has unknown fields: ["extra","zeta"]',
      ["extra"],
    );
  });

  test("nullable and variant failures use the current field path", () => {
    const typ = record(
      {
        note: new Type({ kind: ValueKind.STRING, nullable: true }),
        decision: new Type({ kind: ValueKind.STRING, tags: new Set(["accept", "reject"]) }),
      },
      ["decision"],
    );
    expect(buildValueAs(typ, { note: null, decision: "accept" }).value).toEqual({
      note: null,
      decision: "accept",
    });
    expect(typeCheckError(() => buildValueAs(typ, { note: 3n, decision: "accept" })).path).toEqual(
      ["note"],
    );
    expect(
      typeCheckError(() => buildValueAs(typ, { note: "ok", decision: "maybe" })).path,
    ).toEqual(["decision"]);
  });

  test("a record stores a __proto__ field as data", () => {
    const typ = new Type({
      kind: ValueKind.OBJECT,
      fields: new Map([["__proto__", Type.scalar(ValueKind.INTEGER)]]),
      required: new Set(["__proto__"]),
    });
    const value = {};
    Object.defineProperty(value, "__proto__", { value: 1n, enumerable: true });
    const built = buildValueAs(typ, value).value as object;
    expect(Object.getPrototypeOf(built)).toBe(Object.prototype);
    expect(Object.keys(built)).toEqual(["__proto__"]);
  });
});

describe("TS-only: -0, NaN and lone surrogates at typed positions", () => {
  test("-0 becomes 0 at a float slot and a flat list(float) item", () => {
    expect(Object.is(buildValueAs(ValueKind.NUMBER, -0).value, 0)).toBe(true);
    const items = buildValueAs(ValueKind.LIST_NUMBER, [1.5, -0]).value as Value[];
    expect(Object.is(items[1], 0)).toBe(true);
  });

  test.each<[string, ValueKind | Type, unknown, string, (string | number)[]]>([
    ["NaN at a float slot", ValueKind.NUMBER, NaN, "NaN is not a yian value", []],
    ["NaN in a flat list(float)", ValueKind.LIST_NUMBER, [1, NaN], "NaN is not a yian value", [1]],
    [
      "NaN in a record field",
      record({ x: Type.scalar(ValueKind.NUMBER) }),
      { x: NaN },
      "NaN is not a yian value",
      ["x"],
    ],
    [
      "a lone surrogate at a str slot",
      ValueKind.STRING,
      "\ud800",
      "text holding a lone surrogate is not a yian value",
      [],
    ],
    [
      "a lone surrogate in a flat list(str)",
      ValueKind.LIST_STRING,
      ["a", "b\udc00"],
      "text holding a lone surrogate is not a yian value",
      [1],
    ],
    [
      "a lone surrogate at a variant slot",
      new Type({ kind: ValueKind.STRING, tags: new Set(["a"]) }),
      "\ud800",
      "text holding a lone surrogate is not a yian value",
      [],
    ],
    [
      "a lone surrogate in a file's JSON form",
      ValueKind.FILE,
      { uri: "\ud800", mime: null, name: null },
      "text holding a lone surrogate is not a yian value",
      ["uri"],
    ],
    // A typed record's own keys are field names: a surrogate key is an unknown field.
    [
      "a lone surrogate key in a typed record",
      record({ a: Type.scalar(ValueKind.INTEGER) }),
      { a: 1n, "\ud800": 1n },
      'record has unknown fields: ["\\ud800"]',
      ["\ud800"],
    ],
  ])("refuses %s", (_id, declared, value, message, path) => {
    expectTypeCheckError(() => buildValueAs(declared, value), message, path);
  });
});

describe("lossless serialization", () => {
  test.each<[Value, StorageKind, string]>([
    [3n, ValueKind.INTEGER, "bigint"],
    [3, ValueKind.NUMBER, "number"],
    [true, ValueKind.BOOLEAN, "boolean"],
    ["x", ValueKind.STRING, "string"],
    [{ k: [1n, 2n] }, ValueKind.OBJECT, "record"],
    [["a", "b"], ValueKind.LIST_STRING, "array"],
  ])("a JSON round trip keeps the kind and JS type: %p", (value, kind, type) => {
    const back = roundTrip(buildValue(value));
    expect(back.kind).toBe(kind);
    // int, float and bool survive the JSON number ambiguity through the kind tag.
    expect(jsType(back.value)).toBe(type);
    expect(back.value).toEqual(value);
  });

  test("a file round-trips as a FileRef", () => {
    const typed = buildValue(new FileRef({ uri: "s3://b/k", mime: "text/csv", name: "d.csv" }));
    expect(typed.toWire()).toEqual({
      kind: "file",
      value: { uri: "s3://b/k", mime: "text/csv", name: "d.csv" },
    });
    const back = roundTrip(typed);
    expect(back.kind).toBe(ValueKind.FILE);
    expect(back.value).toBeInstanceOf(FileRef);
    expect(back.value).toEqual(new FileRef({ uri: "s3://b/k", mime: "text/csv", name: "d.csv" }));
  });

  test("toWire writes a nested FileRef as its mapping", () => {
    const file = new FileRef({ uri: "u" });
    const typed = new TypedValue(ValueKind.LIST_ANY, [{ f: file }, [file]]);
    const mapping = { uri: "u", mime: null, name: null };
    expect(typed.toWire()).toEqual({ kind: "list[Any]", value: [{ f: mapping }, [mapping]] });
    expect(typed.value).toEqual([{ f: file }, [file]]);
  });

  test.each<[StorageKind, string]>([
    [ValueKind.DATE, "2026-06-08"],
    [ValueKind.DATETIME, "2026-06-12T14:30:00+00:00"],
  ])("a %s value builds and round-trips", (kind, text) => {
    const typed = buildValueAs(kind, text);
    expect(typed.kind).toBe(kind);
    expect(typed.value).toBe(text);
    const back = roundTrip(typed);
    expect(back.kind).toBe(kind);
    expect(back.value).toBe(text);
  });

  test.each<[StorageKind, ValueKind, Value]>([
    [ValueKind.LIST_STRING, ValueKind.STRING, "alpha"],
    [ValueKind.LIST_INTEGER, ValueKind.INTEGER, 7n],
    [ValueKind.LIST_NUMBER, ValueKind.NUMBER, 1.5],
    [ValueKind.LIST_BOOLEAN, ValueKind.BOOLEAN, true],
    [ValueKind.LIST_OBJECT, ValueKind.OBJECT, { name: "alpha" }],
  ])("a nullable refined %s keeps its carrier", (kind, element, item) => {
    const declared = new Type({ kind, element: new Type({ kind: element, nullable: true }) });
    const typed = buildValueAs(declared, [item, null]);
    expect(typed.kind).toBe(kind);
    expect(typed.value).toEqual([item, null]);
    const back = roundTrip(typed);
    expect(back.kind).toBe(kind);
    expect(back.value).toEqual([item, null]);
  });

  test("a recorded object past the nesting limit still decodes", () => {
    let value: Value = "leaf";
    for (let i = 0; i < 150; i++) value = { a: value };
    const back = TypedValue.fromWire(parseJson(logJson({ kind: "object", value }), "codec"));
    expect(back.kind).toBe(ValueKind.OBJECT);
    expect(back.value).toEqual(value);
    expect(back.toWire()).toEqual({ kind: "object", value });
  });

  test("fromWire accepts a list[float] holding an int", () => {
    // A `list(float)` slot keeps its int items, so the log writes such lists.
    const back = TypedValue.fromWire({ kind: "list[float]", value: [1n, 2.5, null] });
    expect(back.kind).toBe(ValueKind.LIST_NUMBER);
    expect(back.value).toEqual([1n, 2.5, null]);
  });

  test("a TypedValue is frozen and gives its value", () => {
    const typed = new TypedValue(ValueKind.INTEGER, 1n);
    expect(Object.isFrozen(typed)).toBe(true);
    expect(typed.toObject()).toBe(1n);
  });
});

describe("TypedValue.fromWire refuses", () => {
  test.each<[string, unknown]>([
    ["a non-record", ["int", 1n]],
    ["a missing key", { kind: "int" }],
    ["an extra key", { kind: "int", value: 1n, extra: 1n }],
    ["kind Any", { kind: "Any", value: 1n }],
    ["an unknown kind", { kind: "string", value: "x" }],
    ["a non-text kind", { kind: 1n, value: 1n }],
    ["None holding a value", { kind: "None", value: 0n }],
    ["int holding a float", { kind: "int", value: 1 }],
    // Nothing writes a scalar `float` holding an int (such a slot stores kind `int`), so it is
    // refused.
    ["float holding an int", { kind: "float", value: 1n }],
    ["float holding NaN", { kind: "float", value: NaN }],
    ["bool holding an int", { kind: "bool", value: 1n }],
    ["object holding a list", { kind: "object", value: [] }],
    ["list[Any] holding a record", { kind: "list[Any]", value: {} }],
    ["list[str] holding an int", { kind: "list[str]", value: ["a", 1n] }],
    ["list[str] with a hole", { kind: "list[str]", value: [, "a"] }],
    ["list[str] as an Array subclass", { kind: "list[str]", value: Sub.from(["a"]) }],
    ["list[object] holding a list", { kind: "list[object]", value: [[]] }],
    ["date holding other text", { kind: "date", value: "x" }],
    ["date holding an int", { kind: "date", value: 1n }],
    ["file holding text", { kind: "file", value: "s3://b/k" }],
    ["file missing fields", { kind: "file", value: { uri: "s3://b/k" } }],
    [
      "file with an extra field",
      { kind: "file", value: { uri: "u", mime: null, name: null, size: 1n } },
    ],
    ["file with a non-text uri", { kind: "file", value: { uri: 1n, mime: null, name: null } }],
  ])("%s", (_id, data) => {
    expect(() => TypedValue.fromWire(data)).toThrow(RangeError);
  });

  test("with a message naming the kind", () => {
    expect(() => TypedValue.fromWire({ kind: "list[str]", value: [1n] })).toThrow(
      "invalid list[str] typed value: expected an array of strings or null, got [1]",
    );
    expect(() => TypedValue.fromWire({ kind: "Any", value: 1n })).toThrow(
      'typed-value kind must be a storage kind, got "Any"',
    );
    expect(() => TypedValue.fromWire({ kind: "int", value: 1n, extra: 1n })).toThrow(
      'invalid typed value fields; missing=[], extra=["extra"]',
    );
    expect(() => TypedValue.fromWire({ kind: "file", value: { uri: "u" } })).toThrow(
      'invalid file reference fields; missing=["mime","name"], extra=[]',
    );
    expect(() =>
      TypedValue.fromWire({ kind: "file", value: { uri: 1n, mime: null, name: null } }),
    ).toThrow("invalid file typed value: file uri must be text, got 1");
  });
});
