/**
 * Files inside typed containers, a port of the value checks (`buildValueAs`), file
 * normalization (`normalizeFiles`), the overlap tests, the two typed `bindParams` reads, the
 * `field_of` table and the two traced item reads of `tests/engine/test_files_in_containers.py`,
 * with `FileRef` constructor refusals
 * (`RangeError`, as pydantic's `ValidationError` is a `ValueError`). Where Python checks that
 * a value is kept with `is`, these tests use `toBe`.
 *
 * TS-only: the `normalizeTypedFiles` tests, and the `normalizeFiles` rows keeping an `object`
 * value that holds NaN, turning a `-0` at an `object` position into `0`, and keeping a file
 * form whose text holds a lone surrogate.
 *
 * Later: flow runs (with the runner).
 *
 * Not ported: the `field_of` rows on dict subclasses, `defaultdict` and a `FileRef` subclass; a
 * record is a plain object and a dotted read gives its own keys. The tuple in
 * `test_an_object_without_files_is_stored_as_given` is a list here, and
 * `test_normalize_files_reads_a_tuple_as_a_list_only_when_it_changes` is not ported (no tuples
 * in TS).
 */

import { describe, expect, test } from "bun:test";

import { evalExpr } from "../../src/expr/expressions.ts";
import { parseExpr } from "../../src/expr/grammar.ts";
import { ResolvedValue, SourceRef } from "../../src/expr/lineage.ts";
import { bindParams } from "../../src/nodes/binding.ts";
import { Binding, Parameter, ValueContract } from "../../src/nodes/contracts.ts";
import { VariablePool } from "../../src/typesys/pool.ts";
import { readTypedefs, typeFor } from "../../src/typesys/types.ts";
import {
  buildValueAs,
  FileRef,
  fieldOf,
  normalizeFiles,
  normalizeTypedFiles,
  Type,
  TypeCheckError,
  TypedValue,
  typesOverlap,
  type Value,
  type ValueRecord,
  ValueKind,
} from "../../src/typesys/values.ts";

const REGISTRY = readTypedefs({
  Rec: { f: "file", n: "int" },
  Form: { uri: "str", mime: "Optional(str)", name: "Optional(str)" },
  IntUri: { uri: "int" },
  Other: { uri: "str", other: "str" },
  XFile: { x: "file" },
  XObject: { x: "object" },
  Box: { x: "object" },
  Num: { n: "float" },
  TFile: { kind: "Literal(file)", x: "file" },
  TObject: { kind: "Literal(object)", x: "object" },
});

/** Resolve a type expression against this module's typedefs. */
const t = (text: string): Type => typeFor(text, REGISTRY);

const FILE = Type.scalar(ValueKind.FILE);
const OBJECT = Type.scalar(ValueKind.OBJECT);
const ANY = Type.scalar(ValueKind.ANY);
const REF = new FileRef({ uri: "f" });

/** A fresh copy of a file's JSON form, `{uri: "f", mime: null, name: null}`. */
function form(): Value {
  return { uri: "f", mime: null, name: null };
}

/** Return the `TypeCheckError` `run` throws. */
function typeCheckError(run: () => unknown): TypeCheckError {
  try {
    run();
  } catch (error) {
    expect(error).toBeInstanceOf(TypeCheckError);
    return error as TypeCheckError;
  }
  throw new Error("expected a TypeCheckError");
}

describe("a file slot", () => {
  test("a file slot reads the JSON form", () => {
    const typed = buildValueAs(FILE, form());
    expect(typed.kind).toBe(ValueKind.FILE);
    expect(typed.value).toEqual(REF);
    expect(typed.value).toBeInstanceOf(FileRef);
  });

  test("a file slot keeps a FileRef", () => {
    const typed = buildValueAs(FILE, REF);
    expect(typed.kind).toBe(ValueKind.FILE);
    expect(typed.value).toBe(REF);
  });

  test("the JSON form keeps its mime and name", () => {
    const value = { uri: "f", mime: "text/plain", name: "a.txt" };
    expect(buildValueAs(FILE, value).value).toEqual(new FileRef(value));
  });

  test.each<[string, Value, Value]>([
    ["list(file)", [form(), REF], [REF, REF]],
    ["Rec", { f: form(), n: 1n }, { f: REF, n: 1n }],
    ["Optional(file)", null, null],
    ["list(Optional(file))", [form(), null], [REF, null]],
    ["list(Rec)", [{ f: form(), n: 1n }], [{ f: REF, n: 1n }]],
  ])("a nested file slot reads the JSON form: %s", (declared, value, expected) => {
    expect(buildValueAs(t(declared), value).toObject()).toEqual(expected);
  });

  test.each<[string, Value]>([
    ["a missing mime", { uri: "f", name: null }],
    ["an extra size", { uri: "f", mime: null, name: null, size: 1n }],
    ["an int uri", { uri: 1n, mime: null, name: null }],
    ["an int mime", { uri: "f", mime: 2n, name: null }],
  ])("a file slot rejects a record that is not the JSON form: %s", (_id, value) => {
    expect(typeCheckError(() => buildValueAs(FILE, value)).message).toContain(
      "does not match declared type file",
    );
    const error = typeCheckError(() => buildValueAs(t("list(file)"), [value]));
    expect(error.message).toContain("does not match declared type file");
    expect(error.path).toEqual([0]);
  });
});

describe("an object slot reads a nested file as its mapping", () => {
  test("an object reads a nested file as its mapping", () => {
    expect(buildValueAs(OBJECT, { f: REF }).value).toEqual({ f: form() });
  });

  test("a list of objects reads a nested file as its mapping", () => {
    const flat = buildValueAs(ValueKind.LIST_OBJECT, [{ f: REF }, { g: [REF] }]);
    const refined = buildValueAs(t("list(object)"), [{ f: REF }]);
    expect(flat.value).toEqual([{ f: form() }, { g: [form()] }]);
    expect(refined.value).toEqual([{ f: form() }]);
  });

  test("an object field reads a nested file as its mapping", () => {
    expect(buildValueAs(t("Box"), { x: { f: REF } }).value).toEqual({ x: { f: form() } });
  });

  test("an object without files is stored as given", () => {
    const value = { items: [1n, 2n], nested: { a: [3n] } };
    const stored = buildValueAs(OBJECT, value).value as Record<string, Value>;
    expect(stored).toBe(value);
    expect(stored["items"]).toBe(value.items);
  });

  test("reading nested files leaves the given object unchanged", () => {
    const value = { f: REF, keep: { a: 1n } };
    const stored = buildValueAs(OBJECT, value).value as Record<string, Value>;
    expect(value).toEqual({ f: REF, keep: { a: 1n } });
    expect(value.f).toBe(REF);
    expect(stored["keep"]).toBe(value.keep);
  });

  test.each<[string, Type, Value]>([
    ["object holding a file", OBJECT, REF],
    ["list(object) holding a file", t("list(object)"), [REF]],
    ["Any holding a file", ANY, REF],
    ["Any holding a nested file", ANY, { f: REF }],
    ["list(Any) holding a file", t("list(Any)"), [REF]],
    ["list(Any) holding a nested file", t("list(Any)"), [{ f: REF }]],
  ])("a file where no file position reads it still fails: %s", (_id, declared, value) => {
    typeCheckError(() => buildValueAs(declared, value));
  });

  test("an object holding itself still fails", () => {
    const cyclic: Record<string, Value> = {};
    cyclic["self"] = cyclic;
    cyclic["f"] = REF;
    expect(typeCheckError(() => buildValueAs(OBJECT, cyclic)).message).toBe(
      "value contains itself",
    );
  });
});

describe("normalizeFiles", () => {
  test("reads the JSON form at a file position", () => {
    expect(normalizeFiles(t("Rec"), { f: form(), n: 1n })).toStrictEqual({ f: REF, n: 1n });
  });

  test("reads a nested file as a mapping at an object position", () => {
    expect(normalizeFiles(t("Box"), { x: { f: REF } })).toStrictEqual({ x: { f: form() } });
  });

  test("keeps an int at a float position", () => {
    const normalized = normalizeFiles(t("Num"), { n: 3n }) as ValueRecord;
    expect(typeof normalized["n"]).toBe("bigint");
  });

  test.each<[string, Value]>([
    ["int", 3n],
    ["Num", { n: 3n }],
    ["list(str)", ["a"]],
    ["Rec", { f: REF, n: 1n }],
    ["list(file)", [REF]],
    ["object", { a: [1n] }],
    ["Any", { uri: "f", mime: null, name: null }],
  ])("returns the same value when nothing changes: %s", (declared, value) => {
    expect(normalizeFiles(t(declared), value)).toBe(value);
  });

  test.each<[string, string[], Value]>([
    ["XFile first", ["XFile", "XObject"], { x: REF }],
    ["XObject first", ["XObject", "XFile"], { x: form() }],
  ])("picks an untagged union member like the write check: %s", (_id, members, expected) => {
    // Both members admit `{x: form}`, so the first one decides how `x` reads.
    const union = new Type({ kind: ValueKind.OBJECT, variants: members.map(t) });
    const value = { x: form() };
    expect(normalizeFiles(union, value)).toStrictEqual(expected);
    expect(buildValueAs(union, value).toObject()).toStrictEqual(expected);
  });

  test.each<[string, Value, Value]>([
    ["file", { kind: "file", x: form() }, { kind: "file", x: REF }],
    ["object", { kind: "object", x: { f: REF } }, { kind: "object", x: { f: form() } }],
  ])("picks a tagged union member by its tag: %s", (_id, value, expected) => {
    expect(normalizeFiles(t("Union(TFile, TObject)"), value)).toStrictEqual(expected);
  });

  test("reads an Optional union", () => {
    const declared = t("Optional(Union(TFile, TObject))");
    expect(normalizeFiles(declared, null)).toBeNull();
    expect(normalizeFiles(declared, { kind: "file", x: form() })).toStrictEqual({
      kind: "file",
      x: REF,
    });
  });

  test.each<[string, Value]>([
    ["Rec", "text"],
    ["list(file)", { f: form() }],
    ["Union(TFile, TObject)", { kind: "other", x: form() }],
    ["Union(TFile, TObject)", { x: form() }],
    ["Union(XFile, IntUri)", { y: form() }],
    ["Union(XFile, IntUri)", [form()]],
    // An object position reads with the write check's JSON walk, which refuses NaN, so the
    // whole value is kept; Python still maps the file.
    ["object", { f: REF, bad: NaN }],
  ])("keeps a value that fails the type: %s", (declared, value) => {
    expect(normalizeFiles(t(declared), value)).toBe(value);
  });

  test("turns a -0 at an object position into 0", () => {
    // Negative zero becomes 0 when a float becomes a value; Python keeps -0.0 and the value.
    const value = { x: { a: -0 } };
    const normalized = normalizeFiles(t("Box"), value) as ValueRecord;
    expect(normalized).not.toBe(value);
    expect(Object.is((normalized["x"] as ValueRecord)["a"], 0)).toBe(true);
  });

  test("keeps a file form whose text holds a lone surrogate", () => {
    // Text holding a lone surrogate is not a yian value, and normalizeFiles never throws for
    // a value that doesn't fit; Python builds the FileRef.
    const value = { f: { uri: "\ud800", mime: null, name: null }, n: 1n };
    expect(normalizeFiles(t("Rec"), value)).toBe(value);
  });

  test.each<[string, Value]>([
    ["file", form()],
    ["Rec", { f: form(), n: 1n }],
    ["list(Rec)", [{ f: REF, n: 1n }, { f: form(), n: 2n }]],
    ["Optional(list(Optional(file)))", [form(), null]],
    ["object", { f: REF, g: [REF, 1n] }],
    ["list(object)", [{ f: REF }, { a: 1n }]],
    ["Box", { x: { f: [REF] } }],
    ["Union(TFile, TObject)", { kind: "file", x: form() }],
    ["Union(XFile, IntUri)", { x: form() }],
  ])("agrees with the write check: %s", (declared, value) => {
    const typ = t(declared);
    expect(normalizeFiles(typ, value)).toStrictEqual(buildValueAs(typ, value).toObject());
  });
});

describe("normalizeTypedFiles", () => {
  test("a list of files decoded from the wire holds FileRefs", () => {
    const typ = t("list(file)");
    const written = buildValueAs(typ, [REF, form()]);
    const decoded = TypedValue.fromWire(written.toWire());
    expect(decoded.value).toStrictEqual([form(), form()]);
    const normalized = normalizeTypedFiles(typ, decoded);
    expect(normalized.kind).toBe(decoded.kind);
    expect(normalized.value).toStrictEqual([REF, REF]);
  });

  test("returns the same TypedValue when nothing changes", () => {
    const typed = buildValueAs(t("list(file)"), [REF]);
    expect(normalizeTypedFiles(t("list(file)"), typed)).toBe(typed);
  });

  test("a null type returns the TypedValue", () => {
    const typed = TypedValue.fromWire({ kind: "list[Any]", value: [form()] });
    expect(normalizeTypedFiles(null, typed)).toBe(typed);
  });
});

describe("untagged unions whose members can share a file", () => {
  test.each([
    ["file, object", FILE, OBJECT, true],
    ["file, Any", FILE, ANY, true],
    ["file, Form", FILE, t("Form"), true],
    ["XFile, XObject", t("XFile"), t("XObject"), true],
    ["file, int", FILE, t("int"), false],
    ["file, IntUri", FILE, t("IntUri"), false],
    ["file, Other", FILE, t("Other"), false],
  ] as const)("a file overlaps the types that admit its JSON form: %s", (_id, a, b, expected) => {
    expect(typesOverlap(a, b)).toBe(expected);
    expect(typesOverlap(b, a)).toBe(expected);
  });
});

// --- typed bindings deliver the slot's file representation --- //

/** Bind one parameter to `!declared ${a.output}` over a pool holding `output`. */
function bind(declared: Type, output: Value): unknown {
  const pool = new VariablePool();
  pool.set("a", output);
  const record = bindParams(
    [new Parameter("x")],
    new Map([["x", new Binding("${a.output}", { contract: new ValueContract(declared) })]]),
    pool,
  );
  return record["x"];
}

test("an object binding reads a nested file as its mapping", () => {
  expect(bind(OBJECT, { f: REF })).toEqual({ f: form() });
});

test("a file binding reads the JSON form as a file", () => {
  expect(bind(t("Rec"), { f: form(), n: 1n })).toEqual({ f: REF, n: 1n });
});

/** A record holding an own `__proto__` key, as yian builds one. */
function protoKeyRecord(): Value {
  const record = {};
  Object.defineProperty(record, "__proto__", {
    value: 1n,
    writable: true,
    enumerable: true,
    configurable: true,
  });
  return record as Value;
}

describe("fieldOf", () => {
  const file = new FileRef({ uri: "f", mime: "m", name: "n" });
  test.each<[string, Value, string, [boolean, Value]]>([
    ["a key", { a: 1n }, "a", [true, 1n]],
    ["a key holding null", { a: null }, "a", [true, null]],
    ["a missing key", { a: 1n }, "b", [false, null]],
    ["a file's uri", file, "uri", [true, "f"]],
    ["a file's mime", file, "mime", [true, "m"]],
    ["a file's name", file, "name", [true, "n"]],
    ["an unset file field", new FileRef({ uri: "f" }), "name", [true, null]],
    ["another name on a file", new FileRef({ uri: "f" }), "other", [false, null]],
    ["__class__ on a file", new FileRef({ uri: "f" }), "__class__", [false, null]],
    ["model_dump on a file", new FileRef({ uri: "f" }), "model_dump", [false, null]],
    ["a name on text", "text", "upper", [false, null]],
    ["an index on a list", [1n], "0", [false, null]],
    ["a name on null", null, "a", [false, null]],
    // TS-only: a dotted read gives own keys only, never an inherited JS property.
    ["constructor on a record", { a: 1n }, "constructor", [false, null]],
    ["toString on a record", { a: 1n }, "toString", [false, null]],
    ["__proto__ on a record", { a: 1n }, "__proto__", [false, null]],
    ["constructor on a file", new FileRef({ uri: "f" }), "constructor", [false, null]],
    ["length on text", "text", "length", [false, null]],
    ["an own __proto__ key", protoKeyRecord(), "__proto__", [true, 1n]],
    ["a null-prototype record's key", Object.assign(Object.create(null), { a: 1n }), "a", [true, 1n]],
  ])("reads a record key or a file field: %s", (_id, value, name, expected) => {
    expect(fieldOf(value, name)).toEqual(expected);
  });
});

test("a traced item read of a file field routes to its source", () => {
  const item = new ResolvedValue(REF, true, new SourceRef("run", 1, ["xs", 0]));

  const traced = evalExpr(parseExpr("item.uri"), () => new ResolvedValue(null, false), { item, trace: true });

  expect(traced.value).toBe("f");
  expect(traced.route.source).toEqual(new SourceRef("run", 1, ["xs", 0, "uri"]));
});

test("a traced item read of another file field has no source route", () => {
  const item = new ResolvedValue(REF, true, new SourceRef("run", 1, ["xs", 0]));

  const traced = evalExpr(parseExpr("item.other"), () => new ResolvedValue(null, false), { item, trace: true });

  expect(traced.value).toBeNull();
  expect(traced.route.source).toBeNull();
});

describe("FileRef", () => {
  test("defaults mime and name to null and is frozen", () => {
    const file = new FileRef({ uri: "s3://b/k" });
    expect([file.uri, file.mime, file.name]).toEqual(["s3://b/k", null, null]);
    expect(Object.isFrozen(file)).toBe(true);
  });

  test.each<[string, unknown]>([
    ["a missing uri", {}],
    ["a non-text uri", { uri: 1n }],
    ["a null uri", { uri: null }],
    ["a non-text mime", { uri: "u", mime: 1n }],
    ["a non-text name", { uri: "u", name: false }],
    ["a uri with a lone surrogate", { uri: "\ud800" }],
    ["a name with a lone surrogate", { uri: "u", name: "a\udc00" }],
  ])("refuses %s", (_id, fields) => {
    expect(() => new FileRef(fields as never)).toThrow(RangeError);
  });

  test("names the field in its message", () => {
    expect(() => new FileRef({ uri: 1n } as never)).toThrow("file uri must be text, got 1");
    expect(() => new FileRef({ uri: "u", name: "a\udc00" })).toThrow(
      "file name holds a lone surrogate",
    );
  });
});
