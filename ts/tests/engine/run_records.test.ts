/**
 * Run records and their canonical wire codec: a port of `tests/engine/test_run_records.py` and of
 * `test_long_int_old_logs.py::test_invalid_record_json_keeps_its_message`.
 *
 * Ported, the canonical helper tests:
 * `test_typed_object_output_records_a_nested_file_as_its_mapping`,
 * `test_recorded_object_past_the_nesting_limit_still_replays`,
 * `test_seed_keeps_supported_values_and_independent_canonical_copies`,
 * `test_pool_decoder_rejects_unknown_fields_and_wrong_types` (3 rows). Python's `ValueError` is
 * a `RangeError` here.
 *
 * TS-only: a seed keeps `±Infinity` as a float (a TS yian value, unlike Python's); a seed's
 * record keys come out in code-unit order.
 *
 * Changed:
 * - `test_typed_object_output_rejects_a_non_json_member`: Python's date and datetime rows are one
 *   JS `Date` row, refused by the value layer as `<Date object> is not a JSON value`.
 * - `test_seed_canonicalizes_tuple_date_and_file_ref`: only the `FileRef` half (TS has no tuples
 *   or dates among its values).
 * - `test_seed_rejects_nonfinite_before_json_mode_normalization`: the NaN rows `raw`, `dict` and
 *   `typed`, refused with the value layer's `NaN is not a yian value` at Python's paths. A TS seed
 *   is a plain value, never a `TypedValue`, so the `typed` row checks an unchecked `TypedValue`
 *   holding NaN through `canonicalTyped`.
 * - `test_typed_output_rejects_nonfinite_at_its_field_path`: the TS wording, at `["a", 1]`.
 * - `test_canonical_helpers_reject_unordered_containers`: a JS `Set`, refused by the value layer
 *   as `<Set object> is not a JSON value` by both helpers.
 * - `test_canonical_helpers_reject_cyclic_node_data`: both helpers give the value layer's `value
 *   contains itself` (Python's seed helper says `cyclic container`).
 * - `test_pool_codec_is_canonical_and_lossless_for_typed_values`: no date value; the tuple is an
 *   array.
 *
 * Not ported: the `inf` and `-inf` rows (a TS float may be infinite) and the tuple, model and
 * dataclass rows of the non-finite seed test; the pydantic-extras tests; the dataclass row of the
 * unordered-container test; `test_canonical_seed_rejects_keys_before_json_stringification` (JS
 * object keys are always text); `test_typed_object_output_canonicalizes_a_tuple` (TS has no
 * tuples); the recursion-depth tests (the TS codec is iterative);
 * `test_pool_codec_rejects_nonfinite_values` (`inf` is a TS float, and a NaN fails at `pool.set`).
 */

import { expect, test } from "bun:test";

import { canonicalSeed, canonicalTyped, decodePool, encodePool } from "../../src/runtime/log.ts";
import {
  FileRef,
  type PoolWire,
  TypeCheckError,
  TypedValue,
  type Value,
  ValueKind,
  VariablePool,
} from "../../src/typesys/index.ts";

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

test("a typed object output rejects a non-JSON member", () => {
  expectTypeCheck(
    () => canonicalTyped({ items: [1n, 2n], nested: { at: new Date(Date.UTC(2026, 7, 13)) } }, ValueKind.OBJECT),
    "<Date object> is not a JSON value",
    ["nested", "at"],
  );
});

test("a typed object output records a nested file as its mapping", () => {
  const file = new FileRef({ uri: "s3://bucket/report" });

  const [wire, decoded] = canonicalTyped({ items: [1n, 2n], nested: { at: file } }, ValueKind.OBJECT);

  const mapping = { uri: "s3://bucket/report", mime: null, name: null };
  expect(wire).toEqual({ kind: "object", value: { items: [1n, 2n], nested: { at: mapping } } });
  expect(decoded.toObject()).toEqual(wire.value);
});

test("a recorded object past the nesting limit still replays", () => {
  let value: Value = "leaf";
  for (let i = 0; i < 150; i++) value = { a: value };
  const recorded = TypedValue.fromWire({ kind: "object", value });

  const [wire, decoded] = canonicalTyped(recorded, ValueKind.OBJECT);

  expect(wire).toEqual({ kind: "object", value });
  expect(decoded).toEqual(recorded);
});

test("a seed canonicalizes a file ref", () => {
  const seed = { carried: [{ day: "2026-08-13" }, 3n], file: new FileRef({ uri: "memory://answer" }) };

  const [wire, decoded] = canonicalSeed(seed);

  expect(wire).toEqual({
    carried: [{ day: "2026-08-13" }, 3n],
    file: { mime: null, name: null, uri: "memory://answer" },
  });
  expect(Object.keys((wire as Record<string, Record<string, unknown>>)["file"]!)).toEqual(["mime", "name", "uri"]);
  expect(decoded).toEqual(wire);
  expect(canonicalSeed(decoded)[0]).toEqual(wire);
});

test("a seed keeps supported values and independent canonical copies", () => {
  const value = { files: [new FileRef({ uri: "memory://answer" })], count: 4n, ratio: 0.5 };
  const [wire, decoded] = canonicalSeed(value);
  type Seed = { files: { uri: string }[]; count: unknown };
  expect(wire).toEqual(decoded);
  expect(typeof (decoded as Seed).count).toBe("bigint");
  (decoded as Seed).files[0]!.uri = "changed";
  expect((wire as Seed).files[0]!.uri).toBe("memory://answer");
  expect(value.files[0]!.uri).toBe("memory://answer");
});

test("a seed keeps an infinite float", () => {
  const [wire, decoded] = canonicalSeed({ high: Infinity, low: [-Infinity] });
  expect(wire).toEqual({ high: Infinity, low: [-Infinity] });
  expect(decoded).toEqual(wire);
});

test.each<[string, () => unknown, (string | number)[]]>([
  ["raw", () => canonicalSeed(NaN), []],
  ["dict", () => canonicalSeed({ nested: [NaN] }), ["nested", 0]],
  ["typed", () => canonicalTyped(new TypedValue(ValueKind.OBJECT, { nested: NaN }), null), ["nested"]],
])("a seed rejects NaN: %s", (_name, fn, path) => {
  expectTypeCheck(fn, "NaN is not a yian value", path);
});

test("a typed output rejects NaN at its field path", () => {
  expectTypeCheck(() => canonicalTyped({ a: [1.5, NaN] }, null), "NaN is not a yian value", ["a", 1]);
});

test("the canonical helpers reject unordered containers", () => {
  expectTypeCheck(() => canonicalSeed({ values: new Set(["alpha", "beta"]) }), "<Set object> is not a JSON value", [
    "values",
  ]);
  expectTypeCheck(
    () => canonicalTyped({ values: new Set(["alpha", "beta"]) }, ValueKind.OBJECT),
    "<Set object> is not a JSON value",
    ["values"],
  );
});

test("the canonical helpers reject cyclic node data", () => {
  const cyclic: unknown[] = [];
  cyclic.push(cyclic);

  expectTypeCheck(() => canonicalSeed(cyclic), "value contains itself", [0]);
  expectTypeCheck(() => canonicalTyped(cyclic, null), "value contains itself", [0]);
});

test("the pool codec is canonical and lossless for typed values", () => {
  const pool = new VariablePool("start");
  pool.set("start", { items: [1n, 2n], name: "x" });
  pool.set("file", new FileRef({ uri: "s3://bucket/input", name: "input.txt" }));
  pool.addSystem("run_id", "run-pool");

  const wire = encodePool(pool);
  const decoded = decodePool(wire);

  expect(wire.store["start"]!.value).toEqual({ items: [1n, 2n], name: "x" });
  expect(decoded.get("start")).toEqual(wire.store["start"]!.value);
  expect(decoded.get("file")).toEqual(new FileRef({ uri: "s3://bucket/input", mime: null, name: "input.txt" }));
  expect(encodePool(decoded)).toEqual(wire);
});

test.each<[string, unknown, string]>([
  [
    "an extra pool field",
    { store: {}, system: {}, start_id: "start", extra: true },
    'invalid variable pool fields; missing=[], extra=["extra"]',
  ],
  [
    "an int typed value holding text",
    { store: { node: { kind: "int", value: "1" } }, system: {}, start_id: "start" },
    'invalid int typed value: expected a bigint, got "1"',
  ],
  [
    "an extra typed-value field",
    { store: { node: { kind: "int", value: 1n, extra: true } }, system: {}, start_id: "start" },
    'invalid typed value fields; missing=[], extra=["extra"]',
  ],
])("the pool decoder rejects %s", (_name, wire, message) => {
  let caught: unknown;
  try {
    decodePool(wire as PoolWire);
  } catch (error) {
    caught = error;
  }
  expect((caught as Error).constructor).toBe(RangeError);
  expect((caught as Error).message).toBe(message);
});
