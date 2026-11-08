/** The memory value: `MemoryRef`, its wire form, and where a memory slot reads it. */
import { describe, expect, test } from "bun:test";

import { kindName } from "../../src/expr/builtins.ts";
import { canonicalSeed, canonicalTyped } from "../../src/runtime/log.ts";
import { logJson, parseJson } from "../../src/typesys/json.ts";
import {
  buildValue,
  buildValueAs,
  InboxRef,
  jsonPart,
  fieldOf,
  fromHost,
  hasFile,
  memoryType,
  MemoryRef,
  type MemoryRefInit,
  normalizeFiles,
  normalizeTypedFiles,
  requiresFile,
  toHost,
  Type,
  TypeCheckError,
  TypedValue,
  typesOverlap,
  ValueKind,
  valueRepr,
  writeAsJson,
} from "../../src/typesys/values.ts";

const FIELDS: MemoryRefInit = {
  identity: "proj-1a2b3c4d/answer/lessons",
  type: "FileMemory",
  settings: { index: "MEMORY.md", max_file_chars: 100n },
  version: "sha256:ab",
  recall: { text: "notes", cut: null, hint: null },
  tools: [
    { name: "read", write: false, description: "Read a file.", parameters: { type: "object", properties: { path: { type: "string" } } } },
    { name: "append", write: true, description: "Append to a file.", parameters: { type: "object" } },
  ],
};
const ref = new MemoryRef(FIELDS);
const FILE_MEMORY = memoryType("FileMemory", null);
const STR = Type.scalar(ValueKind.STRING);
const INPUTS = new Type({
  kind: ValueKind.OBJECT,
  fields: new Map([
    ["q", STR],
    ["lessons", FILE_MEMORY],
  ]),
  required: new Set(["q", "lessons"]),
});

/** Write `wire` as a run-log line and read it back, as a resume does. */
function throughLog(wire: unknown): unknown {
  return parseJson(logJson(wire), "codec");
}

describe("MemoryRef", () => {
  test("holds the recorded fields, frozen to the leaves", () => {
    expect([ref.identity, ref.type, ref.version]).toEqual([FIELDS.identity, "FileMemory", "sha256:ab"]);
    expect(ref.recall).toEqual(FIELDS.recall);
    expect(ref.tools.map((tool) => tool.name)).toEqual(["read", "append"]);
    expect(Object.isFrozen(ref)).toBe(true);
    expect(Object.isFrozen(ref.settings)).toBe(true);
    expect(Object.isFrozen(ref.tools[0]!.parameters["properties"])).toBe(true);
    expect(MemoryRef.isMemoryRef(ref)).toBe(true);
  });

  test("a change to the fields it was built from doesn't reach it", () => {
    const settings: Record<string, unknown> = { index: "MEMORY.md" };
    const built = new MemoryRef({ ...FIELDS, settings: settings as never });
    settings["index"] = "OTHER.md";
    expect(built.settings).toEqual({ index: "MEMORY.md" });
  });

  test("an object given its prototype or its mapping is not one", () => {
    expect(MemoryRef.isMemoryRef(Object.create(MemoryRef.prototype))).toBe(false);
    expect(MemoryRef.isMemoryRef({ memory: FIELDS })).toBe(false);
  });

  test.each([
    ["an empty identity", { identity: "" }],
    ["a lone surrogate", { identity: "\uD800" }],
    ["a type that isn't text", { type: 5 }],
    ["settings that aren't a record", { settings: [] }],
    ["a recall without a cut", { recall: { text: "x", hint: null } }],
    ["a hint without a tool", { recall: { text: "x", cut: null, hint: { text: "Read with {tool}." } } }],
    ["a tool listed twice", { tools: [FIELDS.tools[0], FIELDS.tools[0]] }],
    ["a tool with an extra field", { tools: [{ ...FIELDS.tools[0], resumable: true }] }],
    ["a tool whose write isn't a bool", { tools: [{ ...FIELDS.tools[0], write: "no" }] }],
    ["parameters that aren't JSON", { tools: [{ ...FIELDS.tools[0], parameters: { at: new Date(0) } }] }],
  ])("refuses %s", (_, change) => {
    expect(() => new MemoryRef({ ...FIELDS, ...change } as never)).toThrow(RangeError);
  });
});

describe("memory wire", () => {
  test("a memory value round-trips through its wire form", () => {
    const wire = buildValue(ref).toWire();
    expect(wire.kind).toBe(ValueKind.MEMORY);
    expect(TypedValue.fromWire(throughLog(wire)).value).toEqual(ref);
    expect(MemoryRef.isMemoryRef(TypedValue.fromWire(throughLog(wire)).value)).toBe(true);
  });

  test("the wire form is the mapping {memory}, with tools kept in spec order", () => {
    const wire = buildValue(ref).toWire() as unknown as { value: { memory: { tools: { name: string }[] } } };
    expect(Object.keys(wire.value)).toEqual(["memory"]);
    expect((throughLog(wire) as typeof wire).value.memory.tools.map((tool) => tool.name)).toEqual(["read", "append"]);
  });

  test("a memory value inside START's input record round-trips through the input types", () => {
    const [wire] = canonicalTyped({ q: "x", lessons: ref }, INPUTS);
    const back = normalizeTypedFiles(INPUTS, TypedValue.fromWire(throughLog(wire)));
    expect(back.value).toEqual({ q: "x", lessons: ref });
    expect(MemoryRef.isMemoryRef((back.value as Record<string, unknown>)["lessons"])).toBe(true);
  });

  test("a seed writes a nested memory value as its mapping", () => {
    const [wire] = canonicalSeed({ lessons: ref }) as [{ lessons: Record<string, unknown> }, unknown];
    expect(Object.keys(wire.lessons)).toEqual(["memory"]);
  });

  test.each([
    [{ kind: "memory", value: "lessons" }],
    [{ kind: "memory", value: { memory: { ...FIELDS, extra: 1n } } }],
    [{ kind: "memory", value: { memory: FIELDS, extra: null } }],
    [{ kind: "memory", value: { memory: { ...FIELDS, identity: "" } } }],
  ])("fromWire refuses %p", (wire) => {
    expect(() => TypedValue.fromWire(wire)).toThrow(RangeError);
  });
});

describe("memory at the host boundary", () => {
  test("host intake and hand-over keep the value", () => {
    expect((fromHost({ a: ref }, "x") as { a: unknown }).a).toBe(ref);
    expect((toHost({ a: ref }) as { a: unknown }).a).toBe(ref);
  });

  test("a forged memory form from a host stays a plain record", () => {
    const forged = fromHost({ memory: FIELDS }, "input") as Record<string, unknown>;
    expect(MemoryRef.isMemoryRef(forged)).toBe(false);
    expect(buildValue(forged).kind).toBe(ValueKind.OBJECT);
  });

  test("a message shows a memory value by its identity", () => {
    expect(valueRepr(ref)).toBe("<memory proj-1a2b3c4d/answer/lessons>");
    expect(valueRepr({ a: ref })).toBe('{"a":<memory proj-1a2b3c4d/answer/lessons>}');
  });

  test("JSON text writes a memory value as its mapping", () => {
    expect(JSON.parse(writeAsJson({ a: ref }))).toEqual(JSON.parse(JSON.stringify({ a: { memory: { ...FIELDS, settings: { index: "MEMORY.md", max_file_chars: 100 } } } })));
  });

  test("a dotted read finds no field on a memory value", () => {
    expect(fieldOf(ref, "identity")).toEqual([false, null]);
  });

  test("an expression error names the kind memory", () => {
    expect(kindName(ref)).toBe("memory");
  });
});

describe("memory at the write boundary", () => {
  test("a bare memory value infers kind memory", () => {
    expect(buildValue(ref).kind).toBe(ValueKind.MEMORY);
  });

  test("a memory slot stores the MemoryRef it is given", () => {
    const stored = buildValueAs(FILE_MEMORY, ref);
    expect(stored.kind).toBe(ValueKind.MEMORY);
    expect(stored.value).toBe(ref);
  });

  test("a memory slot refuses a memory value's JSON form: only a read of yian's own log builds one", () => {
    expect(() => buildValueAs(FILE_MEMORY, { memory: FIELDS })).toThrow(TypeCheckError);
    expect(() => buildValueAs(FILE_MEMORY, jsonPart(ref, [], "mapping"))).toThrow("does not match declared type FileMemory");
  });

  test("a memory slot with no memory type named is refused as malformed", () => {
    expect(() => buildValueAs(ValueKind.MEMORY, ref)).toThrow("a memory type needs its name");
  });

  test("an Optional memory slot takes null", () => {
    expect(buildValueAs(FILE_MEMORY.replace({ nullable: true }), null).kind).toBe(ValueKind.NONE);
  });

  test.each([["lessons"], [{ memory: { ...FIELDS, identity: "" } }], [null], [[ref]]])(
    "a memory slot refuses %p",
    (value) => {
      expect(() => buildValueAs(FILE_MEMORY, value)).toThrow(TypeCheckError);
    },
  );

  test("a memory slot refuses a memory of another type, naming both", () => {
    const vector = new MemoryRef({ ...FIELDS, type: "VectorDBMemory" });
    expect(() => buildValueAs(FILE_MEMORY, vector)).toThrow(
      "<memory proj-1a2b3c4d/answer/lessons> does not match declared type FileMemory: it is a VectorDBMemory",
    );
  });

  test("the refusal names the declared memory type", () => {
    expect(() => buildValueAs(FILE_MEMORY, "x")).toThrow('"x" does not match declared type FileMemory');
  });

  test.each([[ValueKind.ANY], [ValueKind.OBJECT]])("%p refuses a memory value", (kind) => {
    expect(() => buildValueAs(kind, ref)).toThrow(TypeCheckError);
  });

  // An object slot reads every field of its value, so a memory nested in one would show its recall.
  test.each([
    ["an object field", Type.scalar(ValueKind.OBJECT), { a: ref }],
    ["a deeper object field", Type.scalar(ValueKind.OBJECT), { a: [{ b: ref }] }],
    ["a list(object) item's field", Type.scalar(ValueKind.LIST_OBJECT), [{ a: ref }]],
    ["an Any field", Type.scalar(ValueKind.ANY), { a: ref }],
  ])("%s refuses a nested memory value", (_, typ, value) => {
    expect(() => buildValueAs(typ, value as never)).toThrow("is not a JSON value");
  });

  test("an object slot still reads a nested file or inbox handle as its mapping", () => {
    expect(buildValueAs(Type.scalar(ValueKind.OBJECT), { a: new InboxRef("m") } as never).value).toEqual({ a: { inbox: "m" } });
  });
});

describe("memory reading type", () => {
  test("an object position keeps a nested memory value as it is, never its readable mapping", () => {
    const read = normalizeFiles(Type.scalar(ValueKind.OBJECT), { a: ref } as never) as { a: unknown };
    expect(read.a).toBe(ref);
  });

  test("a record's memory field reads its mapping back as a MemoryRef, as a read of the run log does", () => {
    const read = normalizeFiles(INPUTS, { q: "x", lessons: { memory: FIELDS } } as never) as { lessons: unknown };
    expect(MemoryRef.isMemoryRef(read.lessons)).toBe(true);
  });

  test("a memory type holds a value that isn't plain JSON", () => {
    expect(hasFile(FILE_MEMORY)).toBe(true);
    expect(requiresFile(FILE_MEMORY)).toBe(true);
  });

  test("a memory overlaps only a memory", () => {
    expect(typesOverlap(FILE_MEMORY, FILE_MEMORY)).toBe(true);
    expect(typesOverlap(FILE_MEMORY, Type.scalar(ValueKind.ANY))).toBe(false);
    expect(typesOverlap(FILE_MEMORY, Type.scalar(ValueKind.OBJECT))).toBe(false);
    expect(typesOverlap(Type.scalar(ValueKind.NONE), FILE_MEMORY.replace({ nullable: true }))).toBe(true);
  });
});
