/**
 * How deep a type may nest once its typedef names are expanded, a port of the `typeFor`
 * tests of `tests/engine/test_typedef_nesting.py`, with the record chain built by hand.
 * Later: the `readTypedefs` walks, cycle reports and level and chain checks, and the flow
 * loading tests.
 */

import { describe, expect, test } from "bun:test";

import { type TypeDef, type TypeRegistry, typeFor } from "../../src/typesys/types.ts";
import { TypeCheckError, typeLevels } from "../../src/typesys/values.ts";

/** Return `n` records, `T0: {a: int}` and each next one `{a: <previous>}`. */
function records(n: number): TypeRegistry {
  const registry = new Map<string, TypeDef>();
  for (let i = 0; i < n; i++) {
    const field = i === 0 ? "int" : `T${i - 1}`;
    registry.set(`T${i}`, { kind: "record", fields: new Map([["a", field]]) });
  }
  return registry;
}

/** Return `name` wrapped in `lists` nested `list(...)`s. */
function wrapped(name: string, lists = 100): string {
  return "list(".repeat(lists) + name + ")".repeat(lists);
}

const EXPANDED =
  "type nests deeper than 100 levels once its typedef names are expanded, the limit for a " +
  "type (each record and list is a level)";

describe("typeFor nesting", () => {
  test("bounds a type with its typedefs expanded", () => {
    const registry = records(100);
    for (const text of [wrapped("T99"), wrapped("T99", 1)]) {
      expect(() => typeFor(text, registry)).toThrow(TypeCheckError);
      expect(() => typeFor(text, registry)).toThrow(EXPANDED);
    }
  });

  test("a type at the limit resolves", () => {
    expect(typeLevels(typeFor(wrapped("T98", 1), records(99)))).toBe(100);
    typeFor(wrapped("int"), new Map());
  });
});
