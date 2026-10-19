/**
 * Forward captures: a port of `tests/engine/test_forward_trace.py`. Ported:
 * `test_forward_capture_rejects_duplicate_record_names`, and the constructor halves of
 * `test_forward_capture_round_trips_invocation_names_and_structural_layout` (the caller's later
 * edit of `node_data` doesn't show),
 * `test_forward_capture_rejects_malformed_invocation_inventory_and_paths` (10 rows) and
 * `test_forward_capture_rejects_noncanonical_embedded_json` (5 rows), and
 * `test_scope_free_evidence_rejects_recalled_sections`. Python's `ValueError` is a `RangeError`
 * here.
 *
 * TS-only: one row per `ForwardCapture` message with its class; a capture with items, scope and
 * node data keeps frozen copies; `ForwardCapture`'s defaults; an unfinished capture may keep an
 * unfinished item; `BackwardCapture`'s defaults, one row per check its constructor wires in, and
 * its frozen copy of `nodeData`; a `Set` subclass whose iterator changes between passes stores the
 * keys that were checked.
 *
 * Changed:
 * - Captures hold frozen copies, with no copy on read: a reader's edit throws a `TypeError` where
 *   Python changes a private copy, and `evaluatedKeys` is a private `Set` copy, read-only by
 *   contract, where Python keeps the caller's `frozenset`.
 * - Inventories and items are arrays, not tuples; `evaluatedKeys` is a `Set`, not a `frozenset`.
 * - Each rejected row names its one error class, where Python accepts either `TypeError` or
 *   `ValueError`. A malformed `value_json` throws the reader's `JSONDecodeError`.
 *
 * Not ported: the forged-field rows of `test_codec_revalidates_forged_immutable_capture_fields`
 * (`children`, `path`, `scope`): a TS route, source and scope are frozen by their constructors,
 * so a capture holds them without rebuilding them.
 *
 * Later (T4-2): the log round trips: the record encode and decode halves of the round-trip,
 * malformed-inventory and noncanonical-JSON tests above, and the other log tests of the module.
 * Later (T4-3): the trace tests, which run flows.
 */

import { expect, test } from "bun:test";

import { ValueRoute } from "../../src/expr/lineage.ts";
import { EvaluationEvidence, MemorySection } from "../../src/memory/contracts.ts";
import { ExecutionPosition, ExecutionScope } from "../../src/nodes/contracts.ts";
import { BackwardCapture, ForwardCapture } from "../../src/runtime/contracts.ts";
import { JSONDecodeError } from "../../src/typesys/json.ts";

type ErrorClass = typeof TypeError | typeof RangeError;

/** A record route with one literal field `x`. */
const ROUTE_X = ValueRoute.record([["x", ValueRoute.literal(1)]]);

/** A finished capture of `x`. */
const FINISHED = new ForwardCapture({ x: 1 }, ROUTE_X);

/** An unfinished capture of `x`. */
const UNFINISHED = new ForwardCapture(null, ROUTE_X);

test("a forward capture keeps invocation names and a detached copy of node data", () => {
  const data = { input_output_paths: { result: [] as string[] } };
  const route = ValueRoute.record([["result", ValueRoute.literal({ items: [1] })]]);
  const capture = new ForwardCapture({ result: { items: [1] } }, route, {
    nodeData: data,
    inputNames: ["result"],
    variableNames: ["instruction"],
  });
  data.input_output_paths.result.push("changed");
  expect(capture.inputNames).toEqual(["result"]);
  expect(capture.variableNames).toEqual(["instruction"]);
  expect(capture.nodeData).toEqual({ input_output_paths: { result: [] } });
});

test.each<[string, string, unknown, ErrorClass, string]>([
  ["input names null", "inputNames", null, TypeError, "input_names must be an array"],
  ["duplicate input names", "inputNames", ["result", "result"], RangeError, "input_names must contain unique names"],
  ["a non-string input name", "inputNames", [1], TypeError, "input_names must contain only strings"],
  [
    "duplicate variable names",
    "variableNames",
    ["instruction", "instruction"],
    RangeError,
    "variable_names must contain unique names",
  ],
  ["a non-string variable name", "variableNames", [false], TypeError, "variable_names must contain only strings"],
  ["paths as an array", "nodeData", { input_output_paths: [] }, TypeError, "input_output_paths must be a record"],
  [
    "a path for an unknown input",
    "nodeData",
    { input_output_paths: { unknown: [] } },
    RangeError,
    "input_output_paths must name declared inputs",
  ],
  [
    "a boolean path part",
    "nodeData",
    { input_output_paths: { result: [true] } },
    RangeError,
    "output path components must be strings or non-negative integers",
  ],
  [
    "a negative path part",
    "nodeData",
    { input_output_paths: { result: [-1] } },
    RangeError,
    "output path components must be strings or non-negative integers",
  ],
  [
    "a path as a string",
    "nodeData",
    { input_output_paths: { result: "result" } },
    TypeError,
    "input_output_paths values must be arrays",
  ],
])("a forward capture rejects malformed invocation inventory and paths: %s", (_label, field, value, cls, message) => {
  const options = { inputNames: ["result"], variableNames: ["instruction"], [field]: value };
  const route = ValueRoute.record([["result", ValueRoute.literal(1)]]);
  const build = () => new ForwardCapture({ result: 1 }, route, options as never);
  expect(build).toThrow(cls);
  expect(build).toThrow(message);
});

test.each([
  ["duplicate key", '{"a":1,"a":2}'],
  ["nested duplicate key", '{"nested":{"a":1,"a":2}}'],
  ["surrogate string", '"\\ud800"'],
  ["surrogate value", '{"nested":"\\ud800"}'],
  ["surrogate key", '{"\\ud800":1}'],
])("a forward capture rejects noncanonical embedded JSON: %s", (_label, valueJson) => {
  const route = new ValueRoute("literal", { valueJson });
  expect(() => new ForwardCapture({ x: null }, ValueRoute.record([["x", route]]))).toThrow(JSONDecodeError);
});

test("a forward capture rejects duplicate record names", () => {
  const route = ValueRoute.record([
    ["x", ValueRoute.literal(1)],
    ["x", ValueRoute.literal(2)],
  ]);
  const build = () => new ForwardCapture({ x: 2 }, route);
  expect(build).toThrow(RangeError);
  expect(build).toThrow("unique");
});

test("a forward capture keeps frozen copies of its values", () => {
  const bound = { x: [1, 2] };
  const keys = new Set(["x"]);
  const items: [number, ForwardCapture][] = [[0, FINISHED], [2, FINISHED]];
  const data = { static: { label: "a" }, input_output_paths: { x: [0] } };
  const names = ["x"];
  const scope = new ExecutionScope(["work"]);
  const capture = new ForwardCapture(bound, ROUTE_X, {
    evaluatedKeys: keys,
    items,
    scope,
    nodeData: data,
    inputNames: names,
    variableNames: ["tone"],
  });
  bound.x.push(3);
  keys.add("y");
  items.push([3, FINISHED]);
  items[0]![0] = 9;
  data.static.label = "b";
  names.push("y");

  expect(capture.boundInputs).toEqual({ x: [1, 2] });
  expect(capture.route).toBe(ROUTE_X);
  expect([...capture.evaluatedKeys]).toEqual(["x"]);
  expect(capture.evaluatedKeys).not.toBe(keys);
  expect(capture.items).toEqual([[0, FINISHED], [2, FINISHED]]);
  expect(capture.items[0]![1]).toBe(FINISHED);
  expect(capture.scope).toBe(scope);
  expect(capture.nodeData).toEqual({ static: { label: "a" }, input_output_paths: { x: [0] } });
  expect(capture.inputNames).toEqual(["x"]);
  expect(capture.variableNames).toEqual(["tone"]);

  expect(() => (capture.boundInputs!.x as number[]).push(4)).toThrow(TypeError);
  expect(() => (capture.items as unknown[]).push([5, FINISHED])).toThrow(TypeError);
  expect(() => {
    (capture.items[0] as unknown as number[])[0] = 7;
  }).toThrow(TypeError);
  expect(() => {
    (capture.nodeData!.static as { label: string }).label = "c";
  }).toThrow(TypeError);
  expect(() => (capture.inputNames as string[]).push("z")).toThrow(TypeError);
  expect(() => (capture.variableNames as string[]).push("z")).toThrow(TypeError);
  expect(() => {
    (capture as { scope: ExecutionScope | null }).scope = null;
  }).toThrow(TypeError);
});

test("a forward capture stores the evaluated keys it checked", () => {
  // Yields `x` on the first pass and an uncaptured key on any later pass.
  class ShiftingSet extends Set<string> {
    passes = 0;
    override *[Symbol.iterator](): SetIterator<string> {
      yield this.passes++ === 0 ? "x" : "nope";
    }
  }
  const capture = new ForwardCapture({ x: 1 }, ROUTE_X, { evaluatedKeys: new ShiftingSet(["x"]) });
  expect([...capture.evaluatedKeys]).toEqual(["x"]);
});

test("a forward capture defaults to empty keys, items and inventories", () => {
  const capture = new ForwardCapture({}, ValueRoute.record([]));
  expect(capture.evaluatedKeys.size).toBe(0);
  expect(capture.items).toEqual([]);
  expect(capture.scope).toBeNull();
  expect(capture.nodeData).toBeNull();
  expect(capture.inputNames).toEqual([]);
  expect(capture.variableNames).toEqual([]);
});

test("an unfinished capture may keep unfinished items", () => {
  const capture = new ForwardCapture(null, ROUTE_X, { items: [[0, UNFINISHED]] });
  expect(capture.boundInputs).toBeNull();
  expect(capture.items[0]![1]).toBe(UNFINISHED);
});

test.each<[string, () => unknown, ErrorClass, string]>([
  [
    "a route that isn't a record",
    () => new ForwardCapture(null, ValueRoute.literal(1)),
    RangeError,
    "input route must be a record",
  ],
  [
    "evaluated keys that aren't a set",
    () => new ForwardCapture({ x: 1 }, ROUTE_X, { evaluatedKeys: ["x"] as never }),
    TypeError,
    "evaluated_keys must be a set of strings",
  ],
  [
    "a non-string evaluated key",
    () => new ForwardCapture({ x: 1 }, ROUTE_X, { evaluatedKeys: new Set([1]) as never }),
    TypeError,
    "evaluated_keys must be a set of strings",
  ],
  [
    "an evaluated key that isn't captured",
    () => new ForwardCapture({ x: 1 }, ROUTE_X, { evaluatedKeys: new Set(["y"]) }),
    RangeError,
    "evaluated_keys must name captured routes",
  ],
  [
    "bound inputs as an array",
    () => new ForwardCapture([1] as never, ROUTE_X),
    TypeError,
    "bound_inputs must be a JSON object or null",
  ],
  [
    "bound input keys that differ from the routes",
    () => new ForwardCapture({ y: 1 }, ROUTE_X),
    RangeError,
    "bound_inputs keys must match input routes",
  ],
  [
    "items that aren't an array",
    () => new ForwardCapture({ x: 1 }, ROUTE_X, { items: new Set() as never }),
    TypeError,
    "items must be an array",
  ],
  [
    "an item that isn't a pair",
    () => new ForwardCapture({ x: 1 }, ROUTE_X, { items: [[0]] as never }),
    TypeError,
    "item captures must be index/capture pairs",
  ],
  [
    "a negative item index",
    () => new ForwardCapture({ x: 1 }, ROUTE_X, { items: [[-1, FINISHED]] }),
    RangeError,
    "item indices must be non-negative and increasing",
  ],
  [
    "a repeated item index",
    () => new ForwardCapture({ x: 1 }, ROUTE_X, { items: [[1, FINISHED], [1, FINISHED]] }),
    RangeError,
    "item indices must be non-negative and increasing",
  ],
  [
    "an item that isn't a ForwardCapture",
    () => new ForwardCapture({ x: 1 }, ROUTE_X, { items: [[0, { ...FINISHED }]] as never }),
    TypeError,
    "item capture must be a ForwardCapture",
  ],
  [
    "an unfinished capture with a finished item",
    () => new ForwardCapture(null, ROUTE_X, { items: [[0, FINISHED]] }),
    RangeError,
    "unfinished binding cannot retain usable item inputs",
  ],
  [
    "a scope that isn't an ExecutionScope",
    () => new ForwardCapture({ x: 1 }, ROUTE_X, { scope: new ExecutionPosition("call", ["work"]) as never }),
    TypeError,
    "scope must be an ExecutionScope or null",
  ],
  [
    "node data as an array",
    () => new ForwardCapture({ x: 1 }, ROUTE_X, { nodeData: [] as never }),
    TypeError,
    "node_data must be a record or null",
  ],
])("a forward capture rejects %s", (_label, build, cls, message) => {
  expect(build).toThrow(cls);
  expect(build).toThrow(message);
});

test("a backward capture defaults to empty inventories, no scope and empty node data", () => {
  for (const capture of [new BackwardCapture(), new BackwardCapture({ nodeData: null })]) {
    expect(capture.inputNames).toEqual([]);
    expect(capture.variableNames).toEqual([]);
    expect(capture.scope).toBeNull();
    expect(capture.nodeData).toEqual({});
  }
});

test("a backward capture keeps a frozen copy of its node data", () => {
  const data = { static: { label: "a" }, input_output_paths: { result: [] as string[] } };
  const names = ["result"];
  const scope = new ExecutionScope(["work"]);
  const capture = new BackwardCapture({ inputNames: names, variableNames: ["tone"], scope, nodeData: data });
  data.static.label = "b";
  data.input_output_paths.result.push("changed");
  names.push("other");
  expect(capture.nodeData).toEqual({ static: { label: "a" }, input_output_paths: { result: [] } });
  expect(capture.inputNames).toEqual(["result"]);
  expect(capture.variableNames).toEqual(["tone"]);
  expect(capture.scope).toBe(scope);
  expect(() => {
    (capture.nodeData.static as { label: string }).label = "c";
  }).toThrow(TypeError);
  expect(() => (capture.inputNames as string[]).push("z")).toThrow(TypeError);
  expect(() => (capture.variableNames as string[]).push("z")).toThrow(TypeError);
  expect(() => {
    (capture as { scope: ExecutionScope | null }).scope = null;
  }).toThrow(TypeError);
});

test.each<[string, string, unknown, ErrorClass, string]>([
  ["duplicate input names", "inputNames", ["result", "result"], RangeError, "input_names must contain unique names"],
  [
    "duplicate variable names",
    "variableNames",
    ["instruction", "instruction"],
    RangeError,
    "variable_names must contain unique names",
  ],
  [
    "a path for an unknown input",
    "nodeData",
    { input_output_paths: { unknown: [] } },
    RangeError,
    "input_output_paths must name declared inputs",
  ],
  [
    "a scope that isn't an ExecutionScope",
    "scope",
    { staticPath: ["work"] },
    TypeError,
    "scope must be an ExecutionScope or null",
  ],
  ["node data as an array", "nodeData", [], TypeError, "node_data must be a record"],
])("a backward capture rejects %s", (_label, field, value, cls, message) => {
  const options = { inputNames: ["result"], variableNames: ["instruction"], [field]: value };
  const build = () => new BackwardCapture(options as never);
  expect(build).toThrow(cls);
  expect(build).toThrow(message);
});

test("scope-free evidence rejects recalled sections", () => {
  const build = () => new EvaluationEvidence(null, {}, null, null, [new MemorySection("outer", "recalled")]);
  expect(build).toThrow(RangeError);
  expect(build).toThrow("memory_sections memories must appear in scope.memories");
});
