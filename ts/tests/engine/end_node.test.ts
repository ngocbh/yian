/**
 * The END boundary node of `nodes/end/node.ts`: a port of `tests/engine/test_end_node.py`.
 * Ported: `test_end_kind`, `test_end_record_zero_outputs_returns_empty_record`,
 * `test_end_record_one_output_returns_record`, `test_end_record_one_output_captures_a_keyed_path`,
 * `test_end_record_two_outputs_returns_keyed_object`, `test_end_record_params_are_output_names`,
 * `test_end_value_returns_the_bare_value`,
 * `test_end_value_without_a_name_has_no_params_and_returns_none`,
 * `test_end_value_captures_the_whole_output_path`, `test_end_record_returns_record`,
 * `test_end_list_joins_in_over_order`, `test_end_list_n_zero_returns_empty_list`,
 * `test_end_list_params_are_e0_to_en`.
 *
 * Changed: `test_end_value_rejects_more_than_one_name` passes the constructor an options object
 * and expects a `RangeError` (Python's `ValueError`); the factories take an options object, and
 * Python's `list_` is `EndNode.list`.
 *
 * TS-only: the constructor's "exactly one of outputNames or n" `RangeError`, with neither and with
 * both.
 *
 * Later (T4-6): `test_end_post_asserts_fire_record_scoped_through_eval_node`,
 * `test_end_post_asserts_resolve_namespaced_cross_node_ref_via_synth_param`.
 */

import { expect, test } from "bun:test";

import { NodeKind, Output } from "../../src/nodes/base.ts";
import { EndNode } from "../../src/nodes/end/node.ts";

test("end kind", () => {
  expect(EndNode.record("__end__", { outputNames: ["a"] }).kind).toBe(NodeKind.END);
});

// RECORD mode (a flow's `output: {...}`): always a record keyed by the declared names.
test("end record zero outputs returns empty record", async () => {
  const out = await EndNode.record("__end__", { outputNames: [] }).run({});
  expect(out).toBeInstanceOf(Output);
  expect((out as Output).value).toEqual({});
});

test("end record one output returns record", async () => {
  const out = await EndNode.record("__end__", { outputNames: ["only"] }).run({ only: 42n });
  expect((out as Output).value).toEqual({ only: 42n });
});

test("end record one output captures a keyed path", () => {
  const node = EndNode.record("__end__", { outputNames: ["only"] });
  expect(node.captureBackwardContext()).toEqual({ input_output_paths: { only: ["only"] } });
});

test("end record two outputs returns keyed object", async () => {
  const node = EndNode.record("__end__", { outputNames: ["report", "n"] });
  const out = await node.run({ report: "r", n: 4n });
  expect((out as Output).value).toEqual({ report: "r", n: 4n });
});

test("end record params are output names", () => {
  const node = EndNode.record("__end__", { outputNames: ["report", "n"] });
  expect(node.params!.map((p) => p.name)).toEqual(["report", "n"]);
});

// VALUE mode (a flow's `output: ${x}`, or no output): returns the one value or null.
test("end value returns the bare value", async () => {
  const node = EndNode.value("__end__", { outputName: "only" });
  expect(node.params!.map((p) => p.name)).toEqual(["only"]);
  expect(node.returnsRecord).toBe(false);
  expect(((await node.run({ only: 42n })) as Output).value).toBe(42n);
});

test("end value without a name has no params and returns none", async () => {
  const node = EndNode.value("__end__", { outputName: null });
  expect(node.params).toEqual([]);
  expect(((await node.run({})) as Output).value).toBeNull();
});

test("end value captures the whole output path", () => {
  const node = EndNode.value("__end__", { outputName: "only" });
  expect(node.backwardInputNames()).toEqual(["only"]);
  expect(node.captureBackwardContext()).toEqual({ input_output_paths: { only: [] } });
});

test("end record returns record", () => {
  expect(EndNode.record("__end__", { outputNames: ["a", "b"] }).returnsRecord).toBe(true);
});

test("end value rejects more than one name", () => {
  expect(() => new EndNode("__end__", { outputNames: ["a", "b"], value: true })).toThrow(RangeError);
  expect(() => new EndNode("__end__", { n: 1, value: true })).toThrow(RangeError);
});

test("end takes exactly one of outputNames or n", () => {
  const message = "EndNode takes exactly one of outputNames (value/record) or n (list)";
  expect(() => new EndNode("__end__")).toThrow(new RangeError(message));
  expect(() => new EndNode("__end__", { outputNames: ["a"], n: 1 })).toThrow(new RangeError(message));
});

// LIST mode (the MAP fan-in): joins in over order (index).
test("end list joins in over order", async () => {
  const node = EndNode.list("__end__", { n: 3 });
  const out = await node.run({ e0: "ACME", e1: "BETA", e2: "GOOG" });
  expect((out as Output).value).toEqual(["ACME", "BETA", "GOOG"]);
});

test("end list n zero returns empty list", async () => {
  const out = await EndNode.list("__end__", { n: 0 }).run({});
  expect((out as Output).value).toEqual([]);
});

test("end list params are e0 to en", () => {
  const node = EndNode.list("__end__", { n: 2 });
  expect(node.params!.map((p) => p.name)).toEqual(["e0", "e1"]);
});
