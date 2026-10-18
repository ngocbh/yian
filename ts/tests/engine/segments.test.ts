/**
 * The typed value system's wire form, a port of the round-trip tests of
 * `tests/engine/test_segments.py`. Carriers are built with the `TypedValue` constructor; a
 * round trip goes through `toWire`, the run log's encoder and lossless reader, and
 * `TypedValue.fromWire`.
 *
 * Also here: the wire half of
 * `test_run_records.py::test_recorded_object_past_the_nesting_limit_still_replays`.
 *
 * Later: inference (`buildValue`), declared-type checks (`buildValueAs`), the build halves of
 * the date, datetime and nullable-list round trips, and the structural `Type` rows, with the
 * write boundary.
 *
 * Not ported: checks of Python's carrier subclasses (`type(back) is IntegerValue`); TS has one
 * `TypedValue` class, so a round trip checks the kind and the JS type of the value.
 */

import { describe, expect, test } from "bun:test";

import { logJson, parseJson } from "../../src/typesys/json.ts";
import {
  FileRef,
  type StorageKind,
  TypedValue,
  type Value,
  ValueKind,
} from "../../src/typesys/values.ts";

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

describe("lossless serialization", () => {
  test.each<[StorageKind, Value, string]>([
    [ValueKind.INTEGER, 3n, "bigint"],
    [ValueKind.NUMBER, 3, "number"],
    [ValueKind.BOOLEAN, true, "boolean"],
    [ValueKind.STRING, "x", "string"],
    [ValueKind.OBJECT, { k: [1n, 2n] }, "record"],
    [ValueKind.LIST_STRING, ["a", "b"], "array"],
  ])("a JSON round trip keeps the kind and JS type: %s", (kind, value, type) => {
    const back = roundTrip(new TypedValue(kind, value));
    expect(back.kind).toBe(kind);
    // int, float and bool survive the JSON number ambiguity through the kind tag.
    expect(jsType(back.value)).toBe(type);
    expect(back.value).toEqual(value);
  });

  test("a file round-trips as a FileRef", () => {
    const typed = new TypedValue(
      ValueKind.FILE,
      new FileRef({ uri: "s3://b/k", mime: "text/csv", name: "d.csv" }),
    );
    expect(typed.toWire()).toEqual({
      kind: "file",
      value: { uri: "s3://b/k", mime: "text/csv", name: "d.csv" },
    });
    const back = roundTrip(typed);
    expect(back.kind).toBe(ValueKind.FILE);
    expect(back.value).toBeInstanceOf(FileRef);
    expect((back.value as FileRef).uri).toBe("s3://b/k");
    expect((back.value as FileRef).name).toBe("d.csv");
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
  ])("a %s value round-trips", (kind, text) => {
    const back = TypedValue.fromWire(new TypedValue(kind, text).toWire());
    expect(back.kind).toBe(kind);
    expect(back.value).toBe(text);
  });

  test.each<[StorageKind, Value]>([
    [ValueKind.LIST_STRING, "alpha"],
    [ValueKind.LIST_INTEGER, 7n],
    [ValueKind.LIST_NUMBER, 1.5],
    [ValueKind.LIST_BOOLEAN, true],
    [ValueKind.LIST_OBJECT, { name: "alpha" }],
  ])("a nullable refined %s keeps its carrier", (kind, item) => {
    const back = roundTrip(new TypedValue(kind, [item, null]));
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
    // A deliberate difference: Python's strict adapter widens the int 1 for float; the log
    // writer always writes a float with `.0`, so TS refuses it.
    ["float holding an int", { kind: "float", value: 1n }],
    ["float holding NaN", { kind: "float", value: NaN }],
    ["bool holding an int", { kind: "bool", value: 1n }],
    ["object holding a list", { kind: "object", value: [] }],
    ["list[Any] holding a record", { kind: "list[Any]", value: {} }],
    ["list[str] holding an int", { kind: "list[str]", value: ["a", 1n] }],
    ["list[object] holding a list", { kind: "list[object]", value: [[]] }],
    ["date holding other text", { kind: "date", value: "x" }],
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
      "invalid typed value fields; missing=[], extra=['extra']",
    );
    expect(() => TypedValue.fromWire({ kind: "file", value: { uri: "u" } })).toThrow(
      "invalid file reference fields; missing=['mime', 'name'], extra=[]",
    );
  });
});
