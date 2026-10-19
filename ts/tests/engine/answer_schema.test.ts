/**
 * `answerSchema` derives a light answer schema from a HUMAN_INPUT output type: a port of
 * `tests/engine/test_answer_schema.py`, where TS exports the helper Python keeps private.
 *
 * TS-only: an enum is sorted by code units, whatever order the `Literal` lists its values in.
 */

import { expect, test } from "bun:test";

import { readType } from "../../src/compose/helpers/types.ts";
import { answerSchema } from "../../src/compose/nodes/human_input.ts";
import { readTypedefs } from "../../src/typesys/types.ts";

const REGISTRY = readTypedefs({});

/** The answer schema of the type `spec` declares. */
function schema(spec: unknown): Readonly<Record<string, unknown>>[] {
  return answerSchema(readType(spec, REGISTRY));
}

test("none type is empty", () => {
  expect(answerSchema(null)).toEqual([]);
});

test("scalar", () => {
  expect(schema("str")).toEqual([{ name: "answer", type: "str", required: true }]);
});

test("nullable scalar is optional", () => {
  expect(schema("Optional(str)")).toEqual([{ name: "answer", type: "str", required: false }]);
});

test("literal enum carries values", () => {
  expect(schema("Literal(approve, reject)")).toEqual([
    { name: "answer", type: "str", required: true, enum: ["approve", "reject"] },
  ]);
});

test("list of enum keeps the element enum", () => {
  // A list output used to drop its element enum.
  const entries = schema("list(Literal(approve, reject))");
  expect(entries[0]!["name"]).toBe("answer");
  expect(entries[0]!["enum"]).toEqual(["approve", "reject"]);
});

test("record expands per field with required", () => {
  const byName = new Map(schema({ rating: "float", note: "Optional(str)" }).map((entry) => [entry["name"], entry]));
  expect(byName.get("rating")).toEqual({ name: "rating", type: "float", required: true });
  expect(byName.get("note")).toEqual({ name: "note", type: "str", required: false });
});

test("nested record recurses", () => {
  const inner = schema({ inner: { x: "int" } }).find((entry) => entry["name"] === "inner")!;
  expect(inner["fields"]).toEqual([{ name: "x", type: "int", required: true }]);
});

test("an enum is sorted by code units", () => {
  expect(schema("Literal(reject, approve, Zed)")[0]!["enum"]).toEqual(["Zed", "approve", "reject"]);
});
