/**
 * Untyped values that cross the run log's JSON boundary are checked as JSON values: the log half
 * of `tests/engine/test_untyped_output_json.py`.
 *
 * Ported: `test_canonical_typed_accepts_untyped_files` (field, item; the exact wire).
 *
 * Ported, the state module's output canonicalization: `test_a_partial_record_checks_its_opaque_field_as_json`,
 * with the value layer's wording (`<Set object> is not a JSON value`, no `output at g: ` prefix).
 *
 * Changed:
 * - `test_canonical_typed_rejects_untyped_non_json_values`: the `Date` and `Set` rows, with the
 *   value layer's wording (`<Date object> is not a JSON value`, no `output at <path>: ` prefix) at
 *   Python's paths.
 * - `test_canonical_typed_reports_the_type_check_for_typed_values`: the declared-`str` row (a
 *   record with key `"1"`) and the `Set` row, with the value layer's wording.
 * - `test_canonical_typed_still_rejects_lone_surrogates` (str, object, untyped): the value layer
 *   refuses the text before it reaches the writer, so each row throws its `TypeCheckError` `text
 *   holding a lone surrogate is not a yian value` (Python: the writer's `ValueError`).
 *
 * Not ported:
 * - the int-key rows of the two tests above (JS object keys are always text);
 * - `test_canonical_typed_records_a_text_subclass_key_as_plain_text` (a `str` subclass key);
 * - `test_canonical_typed_does_not_walk_a_built_value`: it spies on the module's
 *   `reject_non_json`, which an ESM export can't be replaced with;
 * - the flow and resume tests: they run flows, ported with the run module;
 * - the `reject_non_json`, `value_repr` and date-literal tests: they test the value layer and
 *   the flow loader, not the log.
 */

import { expect, test } from "bun:test";

import { PartialRecordContract, ValueContract } from "../../src/nodes/contracts.ts";
import { canonicalTyped } from "../../src/runtime/log.ts";
import { canonicalOutputValue } from "../../src/runtime/state.ts";
import { FileRef, type StorageKind, Type, TypeCheckError, type Value, ValueKind } from "../../src/typesys/index.ts";

/** Assert `fn` throws a `TypeCheckError` with exactly `message` at `path`. */
function expectTypeCheck(fn: () => unknown, message: string, path: readonly (string | number)[]): void {
  let caught: unknown;
  try {
    fn();
  } catch (error) {
    caught = error;
  }
  expect(caught).toBeInstanceOf(TypeCheckError);
  expect((caught as TypeCheckError).message).toBe(message);
  expect((caught as TypeCheckError).path).toEqual(path);
}

test.each<[string, unknown, string, (string | number)[]]>([
  ["a date", { d: new Date(Date.UTC(2026, 0, 1)) }, "<Date object> is not a JSON value", ["d"]],
  ["a set", { s: new Set([1n, 2n]) }, "<Set object> is not a JSON value", ["s"]],
])("canonicalTyped rejects an untyped non-JSON value: %s", (_name, value, message, path) => {
  expectTypeCheck(() => canonicalTyped(value, null), message, path);
});

const FILE_WIRE = { uri: "f", mime: null, name: null };

test.each<[string, unknown, { kind: StorageKind; value: Value }]>([
  ["field", { f: new FileRef({ uri: "f" }) }, { kind: "object", value: { f: FILE_WIRE } }],
  ["item", [new FileRef({ uri: "f" })], { kind: "list[Any]", value: [FILE_WIRE] }],
])("canonicalTyped accepts an untyped file: %s", (_name, value, expected) => {
  const [wire] = canonicalTyped(value, null);
  // A nested file is recorded as a plain object.
  expect(wire).toEqual(expected);
});

test.each<[string, unknown, ValueKind, string, (string | number)[]]>([
  ["a set in an object", { s: new Set([1n, 2n]) }, ValueKind.OBJECT, "<Set object> is not a JSON value", ["s"]],
  ["a record as str", { 1: "a" }, ValueKind.STRING, '{"1":"a"} does not match declared type str', []],
])("canonicalTyped reports the type check for a typed value: %s", (_name, value, declared, message, path) => {
  expectTypeCheck(() => canonicalTyped(value, declared), message, path);
});

test.each<[string, unknown, ValueKind | null, (string | number)[]]>([
  ["str", "\ud800", ValueKind.STRING, []],
  ["object", { k: "\ud800" }, ValueKind.OBJECT, ["k"]],
  ["untyped", "\ud800", null, []],
])("canonicalTyped still rejects a lone surrogate: %s", (_name, value, declared, path) => {
  expectTypeCheck(() => canonicalTyped(value, declared), "text holding a lone surrogate is not a yian value", path);
});

test("a partial record checks its opaque field as JSON", () => {
  // A typed `file` field `f` and an opaque field `g`.
  const contract = new PartialRecordContract([
    ["f", new ValueContract(Type.scalar(ValueKind.FILE))],
    ["g", null],
  ]);
  const [wire] = canonicalOutputValue({ f: new FileRef({ uri: "f" }), g: [1n] }, contract, null);
  expect(wire.value).toEqual({ f: FILE_WIRE, g: [1n] });

  expectTypeCheck(
    () => canonicalOutputValue({ f: new FileRef({ uri: "f" }), g: new Set([1n, 2n]) }, contract, null),
    "<Set object> is not a JSON value",
    ["g"],
  );
});
