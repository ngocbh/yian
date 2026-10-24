/** How `Inbox(T)` relates to other types: only to another inbox, by its element. */
import { expect, test } from "bun:test";

import { type TypeRelation, typeRelation, typesCompatible } from "../../src/compile/validation.ts";
import { readTypedefs, typeFor } from "../../src/typesys/types.ts";

const NONE = readTypedefs({});
const t = (text: string) => typeFor(text, NONE);

test.each<[string, string, TypeRelation]>([
  ["Inbox(str)", "Inbox(str)", "fits"],
  ["Inbox(int)", "Inbox(float)", "fits"],
  ["Inbox(str)", "Inbox(int)", "disjoint"],
  ["Inbox(str)", "Any", "disjoint"],
  ["Inbox(str)", "object", "disjoint"],
  ["Inbox(str)", "Optional(Any)", "disjoint"],
  ["Any", "Inbox(str)", "disjoint"],
  ["str", "Inbox(str)", "disjoint"],
])("typeRelation(%p, %p) is %p", (inferred, declared, relation) => {
  expect(typeRelation(t(inferred), t(declared))).toBe(relation);
});

test.each<[string, string, boolean]>([
  ["Inbox(int)", "Inbox(float)", true],
  ["Inbox(str)", "Any", false],
  ["object", "Inbox(str)", false],
])("typesCompatible(%p, %p) is %p", (source, sink, compatible) => {
  expect(typesCompatible(t(source), t(sink))).toBe(compatible);
});
