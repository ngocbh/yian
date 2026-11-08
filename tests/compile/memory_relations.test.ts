/** How a memory type relates to other types: only to the same memory type, item record included. */
import { expect, test } from "bun:test";

import { type TypeRelation, typeRelation, typesCompatible } from "../../src/compile/validation.ts";
import { readTypedefs, typeFor, withMemoryTypes } from "../../src/typesys/types.ts";

const REGISTRY = withMemoryTypes(readTypedefs({ Case: { title: "str" }, Other: { name: "str" } }), ["TradeJournal"]);
const t = (text: string) => typeFor(text, REGISTRY);

test.each<[string, string, TypeRelation]>([
  ["FileMemory", "FileMemory", "fits"],
  ["FileMemory", "Optional(FileMemory)", "fits"],
  ["VectorDBMemory(Case)", "VectorDBMemory(Case)", "fits"],
  ["FileMemory", "VectorDBMemory", "disjoint"],
  ["FileMemory", "TradeJournal", "disjoint"],
  ["VectorDBMemory(Case)", "VectorDBMemory(Other)", "disjoint"],
  ["VectorDBMemory(Case)", "VectorDBMemory", "disjoint"],
  ["FileMemory", "Any", "disjoint"],
  ["FileMemory", "object", "disjoint"],
  ["Any", "FileMemory", "disjoint"],
  ["str", "FileMemory", "disjoint"],
])("typeRelation(%p, %p) is %p", (inferred, declared, relation) => {
  expect(typeRelation(t(inferred), t(declared))).toBe(relation);
});

test.each<[string, string, boolean]>([
  ["FileMemory", "FileMemory", true],
  ["FileMemory", "Optional(FileMemory)", true],
  ["Optional(FileMemory)", "FileMemory", false],
  ["VectorDBMemory(Case)", "VectorDBMemory(Other)", false],
  ["TradeJournal", "FileMemory", false],
  ["FileMemory", "Any", false],
  ["object", "FileMemory", false],
])("typesCompatible(%p, %p) is %p", (source, sink, compatible) => {
  expect(typesCompatible(t(source), t(sink))).toBe(compatible);
});
