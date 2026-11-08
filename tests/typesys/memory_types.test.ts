/** Memory types in the type language: `FileMemory`, `VectorDBMemory(T)` and a host's own. */
import { describe, expect, test } from "bun:test";

import { formatTypeExpr, parseType, readTypedefs, typeFor, withMemoryTypes } from "../../src/typesys/types.ts";
import {
  canonicalTypeLabel,
  isMemoryType,
  memoryType,
  Type,
  typeEquals,
  ValueKind,
} from "../../src/typesys/values.ts";

const CASE = readTypedefs({ Case: { title: "str" } });

describe("the type language", () => {
  test.each(["FileMemory", "VectorDBMemory", "VectorDBMemory(Case)", "Optional(FileMemory)"])(
    "%s parses, resolves to a memory and prints back",
    (text) => {
      expect(isMemoryType(typeFor(text, CASE))).toBe(true);
      expect(formatTypeExpr(parseType(text))).toBe(text);
    },
  );

  test("a memory type carries its name and item type; Optional makes it nullable", () => {
    const file = typeFor("FileMemory", CASE);
    expect([file.kind, file.memory, file.element, file.nullable]).toEqual([ValueKind.MEMORY, "FileMemory", undefined, false]);
    const cases = typeFor("VectorDBMemory(Case)", CASE);
    expect([cases.memory, cases.element?.fields?.has("title")]).toEqual(["VectorDBMemory", true]);
    expect(typeFor("Optional(FileMemory)", CASE).nullable).toBe(true);
  });

  test.each([
    ["list(FileMemory)", "a memory is never a list item in 'list(FileMemory)'"],
    ["Union(FileMemory, str)", "a memory is never a union member in 'Union(FileMemory, str)'"],
    ["Union(Case, FileMemory)", "a memory is never a union member in 'Union(Case, FileMemory)'"],
    ["FileMemory(Case)", "FileMemory takes no item type in 'FileMemory(Case)'"],
    ["Dict(str, FileMemory)", "a memory is never a dict key or value in 'dict(str, FileMemory)'"],
    ["Inbox(FileMemory)", "an inbox holds values, not memories, in 'Inbox(FileMemory)'"],
    ["VectorDBMemory(str)", "VectorDBMemory's item type must be a record typedef in 'VectorDBMemory(str)'"],
    ["VectorDBMemory(FileMemory)", "VectorDBMemory's item type must be a record typedef in 'VectorDBMemory(FileMemory)'"],
    ["VectorDBMemory(Message)", "VectorDBMemory's item type must be a record typedef in 'VectorDBMemory(Message)'"],
    ["VectorDBMemory(Optional(Case))", "VectorDBMemory's item type must be a record typedef in 'VectorDBMemory(Optional(Case))'"],
    ["VectorDBMemory(Case, Case)", "VectorDBMemory(...) takes one type"],
  ])("%s is refused", (text, message) => {
    expect(() => typeFor(text, CASE)).toThrow(message);
  });

  test("a record field can't be a memory", () => {
    expect(() => readTypedefs({ R: { m: "FileMemory" } })).toThrow(
      "typedef 'R' field 'm': a memory is never a record field",
    );
    expect(() => readTypedefs({ R: { m: "Optional(FileMemory)" } })).toThrow(
      "typedef 'R' field 'm': a memory is never a record field",
    );
  });

  // A memory is written only as a flow input's tag, so no name can build one from a value.
  test.each(["FileMemory", "Optional(FileMemory)", "VectorDBMemory(Case)"])("a typedef can't name the memory type %s", (text) => {
    let thrown: unknown = null;
    try {
      readTypedefs({ Case: { title: "str" }, Notes: text });
    } catch (error) {
      thrown = error;
    }
    expect((thrown as Error).message).toBe(
      "typedef 'Notes' names a memory type; declare memories as flow inputs (`lessons: !FileMemory`)",
    );
    expect((thrown as { path: unknown }).path).toEqual(["Notes"]);
  });

  test("the typedef walkers look inside an item type", () => {
    expect(() => readTypedefs({ Cases: "VectorDBMemory(Unknown)" })).toThrow("unknown type 'Unknown'");
  });
});

describe("host memory types", () => {
  test("a host memory type parses when the registry knows it", () => {
    const registry = withMemoryTypes(new Map(), ["TradeJournal"]);
    const typ = typeFor("TradeJournal", registry);
    expect([isMemoryType(typ), typ.memory, typ.element]).toEqual([true, "TradeJournal", undefined]);
    expect(typeFor("Optional(TradeJournal)", registry).nullable).toBe(true);
    expect(() => typeFor("list(TradeJournal)", registry)).toThrow("a memory is never a list item");
  });

  test("without the registry entry a host memory type is an unknown type", () => {
    expect(() => typeFor("TradeJournal", new Map())).toThrow("unknown type 'TradeJournal'");
  });

  test("a built-in contract type needs no entry and is left as it is", () => {
    const registry = withMemoryTypes(new Map(), ["FileMemory", "VectorDBMemory"]);
    expect(registry.size).toBe(0);
  });

  test("a host memory type named like a typedef leaves the typedef as it is", () => {
    const registry = withMemoryTypes(readTypedefs({ TradeJournal: "str" }), ["TradeJournal"]);
    expect(isMemoryType(typeFor("TradeJournal", registry))).toBe(false);
  });

  test.each([
    ["trade_journal", "memory type name 'trade_journal' must be PascalCase (an uppercase identifier)"],
    ["Message", "memory type name 'Message' shadows the built-in type 'Message'"],
    ["Inbox", "memory type name 'Inbox' shadows a builtin scalar or typing constructor"],
  ])("a host memory type named %s is refused", (name, message) => {
    expect(() => withMemoryTypes(new Map(), [name])).toThrow(message);
  });
});

describe("the resolved type", () => {
  test("label", () => {
    expect(canonicalTypeLabel(memoryType("FileMemory", null))).toBe("FileMemory");
    expect(canonicalTypeLabel(memoryType("FileMemory", null).replace({ nullable: true }))).toBe("Optional(FileMemory)");
    expect(canonicalTypeLabel(typeFor("VectorDBMemory(Case)", CASE))).toBe('VectorDBMemory({"title":str})');
  });

  test("equality follows the name and the item type", () => {
    expect(typeEquals(memoryType("FileMemory", null), typeFor("FileMemory", CASE))).toBe(true);
    expect(typeEquals(memoryType("FileMemory", null), memoryType("TradeJournal", null))).toBe(false);
    const other = readTypedefs({ Other: { name: "str" } });
    expect(typeEquals(typeFor("VectorDBMemory(Case)", CASE), typeFor("VectorDBMemory(Other)", other))).toBe(false);
    expect(typeEquals(typeFor("VectorDBMemory(Case)", CASE), memoryType("VectorDBMemory", null))).toBe(false);
  });

  test("shape", () => {
    expect(() => memoryType("FileMemory", null).validateShape()).not.toThrow();
    expect(() => typeFor("VectorDBMemory(Case)", CASE).validateShape()).not.toThrow();
    expect(() => new Type({ kind: ValueKind.MEMORY }).validateShape()).toThrow("a memory type needs its name");
    expect(() => new Type({ kind: ValueKind.STRING, memory: "FileMemory" }).validateShape()).toThrow(
      "only memory types may name a memory type",
    );
    expect(() => memoryType("VectorDBMemory", Type.scalar(ValueKind.STRING)).validateShape()).toThrow(
      "a memory's item type must be a record",
    );
  });
});
