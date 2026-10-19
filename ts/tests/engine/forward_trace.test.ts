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
 * Ported, the record halves on the run log: of
 * `test_forward_capture_round_trips_invocation_names_and_structural_layout`,
 * `test_forward_capture_rejects_malformed_invocation_inventory_and_paths` (10 rows) and
 * `test_forward_capture_rejects_noncanonical_embedded_json` (5 rows, the reader's
 * `JSONDecodeError`); `test_schema_six_requires_explicit_invocation_inventories`,
 * `test_forward_capture_round_trips_without_memory_scope`,
 * `test_forward_capture_rejects_malformed_nested_route_wire` (15 rows),
 * `test_forward_capture_keeps_the_existing_canonical_depth_range`,
 * `test_codec_preserves_evaluated_route_variants` (9 rows),
 * `test_codec_preserves_opaque_argument_labels` and
 * `test_codec_rejects_misaligned_or_nonstring_opaque_argument_labels`. Each rejected row names its
 * class and message (a `JSONDecodeError` its reason).
 *
 * TS-only (run log): repeated or unsorted `evaluated_keys` on the wire, and an evaluated key added
 * to a capture's mutable `Set` after construction, refused by the encoder.
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
 * - Schema 10 routes have no `count`, so the two `count` rows of the nested-route test are extra
 *   fields; schema 6 reads `unsupported run-record schema: 6`.
 * - A decoded capture is frozen, so editing its inputs or node data throws a `TypeError` where
 *   Python edits a private copy.
 * - `partial()` returns a record holding a file (the TS leaf JSON can't hold) for Python's
 *   `object()`; `collect_args` takes the named parameters `first`, `second` and `value`, each
 *   defaulting to 0, for Python's `*args, **kwargs`, so its call log holds the bound values.
 *
 * Not ported: the forged-field rows of `test_codec_revalidates_forged_immutable_capture_fields`
 * (`children`, `path`, `scope`): a TS route, source and scope are frozen by their constructors,
 * so a capture holds them without rebuilding them; the repetition rows (`${a * 2}`,
 * `${-2 * a}`) of `test_codec_preserves_evaluated_route_variants`, since repetition is a type
 * error in TS.
 *
 * Later (T4-3): the trace tests, which run flows.
 */

import { afterEach, expect, test } from "bun:test";

import { registerTemplateFn, TEMPLATE_FNS, type TemplateFn, type TemplateParam } from "../../src/expr/builtins.ts";
import { ResolvedValue, routeFeedback, SourceRef, ValueRoute } from "../../src/expr/lineage.ts";
import { evalBinding } from "../../src/expr/template.ts";
import { EvaluationEvidence, MemorySection } from "../../src/memory/contracts.ts";
import { ExecutionPosition, ExecutionScope } from "../../src/nodes/contracts.ts";
import { BackwardCapture, ForwardCapture } from "../../src/runtime/contracts.ts";
import {
  canonicalSeed,
  canonicalTyped,
  decodeRecord,
  encodeRecord,
  NodeSucceededRecord,
  RecordEnvelope,
  SCHEMA_VERSION,
} from "../../src/runtime/log.ts";
import { FileRef, type Value } from "../../src/typesys/index.ts";
import { JSONDecodeError, logJson, parseJson } from "../../src/typesys/json.ts";

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

/** Embedded `value_json` texts the reader refuses, each with its `JSONDecodeError` reason. */
const NONCANONICAL_JSON: [string, string, string][] = [
  ["duplicate key", '{"a":1,"a":2}', 'duplicate JSON object key: "a"'],
  ["nested duplicate key", '{"nested":{"a":1,"a":2}}', 'duplicate JSON object key: "a"'],
  ["surrogate string", '"\\ud800"', "string has an unpaired surrogate"],
  ["surrogate value", '{"nested":"\\ud800"}', "string has an unpaired surrogate"],
  ["surrogate key", '{"\\ud800":1}', "object key has an unpaired surrogate"],
];

/** Assert that `fn` throws exactly a `JSONDecodeError` with `reason`. */
function expectJsonDecodeError(fn: () => unknown, reason: string): void {
  let caught: unknown;
  try {
    fn();
  } catch (error) {
    caught = error;
  }
  expect((caught as Error)?.constructor).toBe(JSONDecodeError);
  expect((caught as JSONDecodeError).reason).toBe(reason);
}

test.each(NONCANONICAL_JSON)(
  "a forward capture rejects noncanonical embedded JSON: %s",
  (_label, valueJson, reason) => {
    const route = new ValueRoute("literal", { valueJson });
    expectJsonDecodeError(() => new ForwardCapture({ x: null }, ValueRoute.record([["x", route]])), reason);
  },
);

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

// --------------------------------------------------------------------------------------- //
// Captures on the run log
// --------------------------------------------------------------------------------------- //

/** A node-success envelope at seq 3 carrying `capture`, with a typed value of `value`. */
function capturedEnvelope(capture: ForwardCapture, value: unknown = null, evidence: EvaluationEvidence | null = null) {
  const record = new NodeSucceededRecord("work", "work", canonicalTyped(value, null)[0], {
    evidence,
    forwardCapture: capture,
  });
  return new RecordEnvelope(SCHEMA_VERSION, "run", 3, record);
}

/** The decoded JSON of `envelope`'s line, ints as `bigint`s, as a mutable object. */
function wireOf(envelope: RecordEnvelope): Record<string, any> {
  return parseJson(encodeRecord(envelope), "codec") as Record<string, any>;
}

/** Decode `wire` written back as a log line. */
function decodeWire(wire: unknown): RecordEnvelope {
  return decodeRecord(logJson(wire));
}

/** Assert that `fn` throws exactly `cls` with exactly `message`. */
function expectError(fn: () => unknown, cls: new (...args: never[]) => Error, message: string): void {
  let caught: unknown;
  try {
    fn();
  } catch (error) {
    caught = error;
  }
  expect((caught as Error)?.constructor).toBe(cls);
  expect((caught as Error).message).toBe(message);
}

test("a forward capture round trips invocation names and structural layout", () => {
  const data = { input_output_paths: { result: [] as string[] } };
  const capture = new ForwardCapture(
    { result: { items: [1n] } },
    ValueRoute.record([["result", ValueRoute.literal({ items: [1n] })]]),
    { nodeData: data, inputNames: ["result"], variableNames: ["instruction"] },
  );
  data.input_output_paths.result.push("changed");
  const envelope = capturedEnvelope(capture, { items: [1n] });

  const decoded = decodeRecord(encodeRecord(envelope));

  expect(decoded).toEqual(envelope);
  const decodedCapture = (decoded.record as NodeSucceededRecord).forwardCapture!;
  expect(decodedCapture.inputNames).toEqual(["result"]);
  expect(decodedCapture.variableNames).toEqual(["instruction"]);
  expect(decodedCapture.nodeData).toEqual({ input_output_paths: { result: [] } });
  expect(decoded.schema).toBe(SCHEMA_VERSION);
  const old = wireOf(envelope);
  old.schema = 6n;
  expectError(() => decodeWire(old), RangeError, "unsupported run-record schema: 6");
});

const CAPTURE_LABEL = "node_succeeded payload forward_capture";

test.each<[string, string, unknown, ErrorClass, string]>([
  ["input names null", "input_names", null, TypeError, `${CAPTURE_LABEL} input_names must be an array of strings`],
  ["duplicate input names", "input_names", ["result", "result"], RangeError, "input_names must contain unique names"],
  [
    "a non-string input name",
    "input_names",
    [1n],
    TypeError,
    `${CAPTURE_LABEL} input_names must be an array of strings`,
  ],
  [
    "duplicate variable names",
    "variable_names",
    ["instruction", "instruction"],
    RangeError,
    "variable_names must contain unique names",
  ],
  [
    "a non-string variable name",
    "variable_names",
    [false],
    TypeError,
    `${CAPTURE_LABEL} variable_names must be an array of strings`,
  ],
  ["paths as an array", "node_data", { input_output_paths: [] }, TypeError, "input_output_paths must be a record"],
  [
    "a path for an unknown input",
    "node_data",
    { input_output_paths: { unknown: [] } },
    RangeError,
    "input_output_paths must name declared inputs",
  ],
  [
    "a boolean path part",
    "node_data",
    { input_output_paths: { result: [true] } },
    RangeError,
    "output path components must be strings or non-negative integers",
  ],
  [
    "a negative path part",
    "node_data",
    { input_output_paths: { result: [-1n] } },
    RangeError,
    "output path components must be strings or non-negative integers",
  ],
  [
    "a path as a string",
    "node_data",
    { input_output_paths: { result: "result" } },
    TypeError,
    "input_output_paths values must be arrays",
  ],
])("a capture's record rejects malformed invocation inventory and paths: %s", (_label, field, value, cls, message) => {
  const capture = new ForwardCapture({ result: 1n }, ValueRoute.record([["result", ValueRoute.literal(1n)]]), {
    inputNames: ["result"],
    variableNames: ["instruction"],
  });
  const wire = wireOf(capturedEnvelope(capture, 1n));
  wire.payload.forward_capture[field] = value;
  expectError(() => decodeWire(wire), cls, message);
});

test.each(NONCANONICAL_JSON)(
  "a capture's record rejects noncanonical embedded JSON: %s",
  (_label, valueJson, reason) => {
    const capture = new ForwardCapture({ x: null }, ValueRoute.record([["x", ValueRoute.literal(null)]]));
    const wire = wireOf(capturedEnvelope(capture));
    wire.payload.forward_capture.route.children[0].value_json = valueJson;
    expectJsonDecodeError(() => decodeWire(wire), reason);
  },
);

// TS-only: the evaluated keys on the wire, and a key added to the capture's mutable `Set` after
// construction, which the encoder checks again.
test.each<[string, string[]]>([
  ["repeated", ["x", "x"]],
  ["unsorted", ["y", "x"]],
])("a capture's record rejects %s evaluated keys", (_label, keys) => {
  const route = ValueRoute.record([
    ["x", ValueRoute.literal(1n)],
    ["y", ValueRoute.literal(2n)],
  ]);
  const wire = wireOf(capturedEnvelope(new ForwardCapture({ x: 1n, y: 2n }, route)));
  wire.payload.forward_capture.evaluated_keys = keys;
  expectError(() => decodeWire(wire), RangeError, "evaluated_keys must be sorted and unique");
});

test("the encoder refuses an evaluated key added after construction", () => {
  const capture = new ForwardCapture({ x: 1n }, ValueRoute.record([["x", ValueRoute.literal(1n)]]));
  (capture.evaluatedKeys as Set<string>).add("y");
  const envelope = capturedEnvelope(capture);
  expectError(() => encodeRecord(envelope), RangeError, "evaluated_keys must name captured routes");
});

test.each(["input_names", "variable_names"])("a capture's record requires its explicit inventory %s", (field) => {
  const wire = wireOf(capturedEnvelope(new ForwardCapture({}, ValueRoute.record([]))));
  delete wire.payload.forward_capture[field];
  expectError(() => decodeWire(wire), RangeError, `invalid ${CAPTURE_LABEL} fields; missing=["${field}"], extra=[]`);
});

test("a forward capture round trips without a memory scope", () => {
  const inputs = { value: [1n, null] };
  const nested = new ValueRoute("source", { source: new SourceRef("run", 1, ["nested"]) });
  const route = ValueRoute.record([["value", nested]]);
  const item = new ForwardCapture({ x: null }, ValueRoute.record([["x", ValueRoute.literal(null)]]), {
    evaluatedKeys: new Set(["x"]),
  });
  const capture = new ForwardCapture(inputs, route, {
    evaluatedKeys: new Set(["value"]),
    items: [[0, item]],
    scope: new ExecutionScope(["work"]),
    nodeData: { saved: [2n] },
  });
  inputs.value.push(99n);
  const evidence = new EvaluationEvidence(null, { value: [1n, null] }, null, null);
  const envelope = capturedEnvelope(capture, null, evidence);

  const decoded = decodeRecord(encodeRecord(envelope));

  expect(decoded).toEqual(envelope);
  const decodedCapture = (decoded.record as NodeSucceededRecord).forwardCapture!;
  expect(decodedCapture.boundInputs).toEqual({ value: [1n, null] });
  expect(() => (decodedCapture.boundInputs!.value as unknown[]).push(8n)).toThrow(TypeError);
  expect(() => (decodedCapture.nodeData!.saved as unknown[]).push(9n)).toThrow(TypeError);
  expect(capture.boundInputs).toEqual({ value: [1n, null] });
  expect(capture.nodeData).toEqual({ saved: [2n] });
  expect(() => {
    (capture as { scope: ExecutionScope | null }).scope = null;
  }).toThrow(TypeError);
  expect(decoded.schema).toBe(SCHEMA_VERSION);
  const old = wireOf(envelope);
  old.schema = 6n;
  expectError(() => decodeWire(old), RangeError, "unsupported run-record schema: 6");
});

test.each<[string, unknown, ErrorClass, string]>([
  [
    "source",
    { run_id: "run", execution_seq: true, path: [] },
    TypeError,
    "route source execution_seq must be an integer",
  ],
  [
    "source",
    { run_id: "run", execution_seq: 1n, path: [-1n] },
    RangeError,
    "route path components must be strings or non-negative integers",
  ],
  [
    "source",
    { run_id: "run", execution_seq: 1n, path: [false] },
    RangeError,
    "route path components must be strings or non-negative integers",
  ],
  ["source", { run_id: "", execution_seq: 1n, path: [] }, RangeError, "source run_id must be non-empty"],
  [
    "source",
    { run_id: "run", execution_seq: 1n, path: [], extra: 1n },
    RangeError,
    'invalid route source fields; missing=[], extra=["extra"]',
  ],
  [
    "children",
    [{}],
    RangeError,
    "invalid route fields; " +
      'missing=["children","expression","kind","names","path","result_type","sizes","snapshot","source","spans",' +
      '"value_json"], extra=[]',
  ],
  // Schema 10 routes have no `count`, so Python's two `count` rows are extra fields here.
  ["count", true, RangeError, 'invalid route fields; missing=[], extra=["count"]'],
  ["count", 2n, RangeError, 'invalid route fields; missing=[], extra=["count"]'],
  ["kind", "unknown", RangeError, "unknown route kind"],
  ["names", ["unused"], RangeError, "route source cannot carry names"],
  ["sizes", [2n], RangeError, "route source cannot carry sizes"],
  ["value_json", "Infinity", RangeError, "route source cannot carry value_json"],
  ["expression", 2n, TypeError, "route expression must be a string"],
  ["path", [0n], RangeError, "route source cannot carry path"],
  ["spans", [[0n, 1n]], RangeError, "route source cannot carry spans"],
])("a capture's record rejects a malformed nested route wire: %s %p", (field, value, cls, message) => {
  const capture = new ForwardCapture(
    { x: 1n },
    ValueRoute.record([["x", new ValueRoute("source", { source: new SourceRef("run", 1) })]]),
  );
  const wire = wireOf(capturedEnvelope(capture));
  wire.payload.forward_capture.route.children[0][field] = value;
  expectError(() => decodeWire(wire), cls, message);
});

test("a forward capture keeps the existing canonical depth range", () => {
  let value: unknown = 1n;
  for (let i = 0; i < 140; i++) value = [value];
  const [, inputs] = canonicalSeed({ x: value });
  const source = new ValueRoute("source", { source: new SourceRef("run", 0, ["x"]) });
  for (const route of [ValueRoute.literal(value), source]) {
    const capture = new ForwardCapture(inputs as Record<string, unknown>, ValueRoute.record([["x", route]]));
    const envelope = capturedEnvelope(capture);
    expect(decodeRecord(encodeRecord(envelope))).toEqual(envelope);
    expect(capture.boundInputs).toEqual(inputs as Record<string, unknown>);
  }
});

// Each registration's previous entry (or its absence), put back after the test, as Python's
// `monkeypatch.setitem` does.
const saved: [string, TemplateFn | undefined][] = [];

/** Register a stand-in builtin `name` running `call` for this test only. */
function standIn(name: string, call: (...args: Value[]) => unknown, params: TemplateParam[] = []): void {
  saved.push([name, TEMPLATE_FNS.get(name)]);
  registerTemplateFn(name, { params, call });
}

afterEach(() => {
  for (const [name, fn] of saved.reverse()) {
    if (fn === undefined) TEMPLATE_FNS.delete(name);
    else TEMPLATE_FNS.set(name, fn);
  }
  saved.length = 0;
});

/** A traced resolver over `a`, `b` and `n`, each its own run input. */
function resolveVariants(path: string): ResolvedValue {
  const values: Record<string, Value> = { a: [1n], b: [2n], n: 3n };
  if (!Object.hasOwn(values, path)) return new ResolvedValue(null, false);
  return new ResolvedValue(values[path]!, true, new SourceRef("run", 0, [path]));
}

test.each([
  "${a}",
  "${a | b}",
  "${[a, b]}",
  "${a + b}",
  "${n + 1}",
  "n=${n}",
  "${absent | absent}",
  "partial: ${partial()}",
  "score: ${score()}",
])("the codec preserves evaluated route variants: %s", (expression) => {
  // A file is the TS leaf JSON can't hold, so `partial()` gives a partial snapshot.
  standIn("partial", () => ({ kept: [1n], lost: new FileRef({ uri: "file:///tmp/x" }) }));
  standIn("score", () => Infinity);
  const traced = evalBinding(expression, resolveVariants, { trace: true });
  const capture = new ForwardCapture({ x: traced.value }, ValueRoute.record([["x", traced.route]]));
  const envelope = capturedEnvelope(capture);
  expect(decodeRecord(encodeRecord(envelope))).toEqual(envelope);
});

// TS builtins take named parameters, so `collect_args` takes `first`, `second` and `value`, each
// defaulting to 0, where Python's takes `*args, **kwargs`; its call log holds the bound values.
const OPAQUE_ARGUMENTS: [string, string[], Value[], bigint][] = [
  ["1", [""], [1n, 0n, 0n], 1n],
  ["1, 2", ["", ""], [1n, 2n, 0n], 3n],
  ["value=1", ["value"], [0n, 0n, 1n], 1n],
  ["1, value=2", ["", "value"], [1n, 0n, 2n], 3n],
];

test.each(OPAQUE_ARGUMENTS)("the codec preserves opaque argument labels: %s", (args, names, bound, result) => {
  const calls: Value[][] = [];
  standIn(
    "collect_args",
    (...values) => {
      calls.push(values);
      return (values as bigint[]).reduce((sum, item) => sum + item, 0n);
    },
    [
      { name: "first", default: 0 },
      { name: "second", default: 0 },
      { name: "value", default: 0 },
    ],
  );
  const traced = evalBinding("${collect_args(" + args + ")}", () => new ResolvedValue(null, false), {
    trace: true,
  });
  const capture = new ForwardCapture({ x: traced.value }, ValueRoute.record([["x", traced.route]]));
  const envelope = capturedEnvelope(capture, result);

  const decoded = decodeRecord(encodeRecord(envelope));

  expect(decoded).toEqual(envelope);
  const route = (decoded.record as NodeSucceededRecord).forwardCapture!.route.children[0]!;
  expect(route.kind).toBe("opaque");
  expect(route.names).toEqual(names);
  const stopped = routeFeedback(route, "Explain this result.", () => null);
  expect(stopped.contributions).toEqual([]);
  expect(stopped.stopped).toHaveLength(1);
  expect(stopped.stopped[0]!.context.at(-1)!.result).toBe(result);
  expect(stopped.stopped[0]!.context.at(-1)!.expression).toBe(traced.route.expression);
  expect(calls).toEqual([bound]);
});

test.each<[unknown[], string]>([
  [[], "opaque route names must match its children"],
  [["", "", ""], "opaque route names must match its children"],
  [[1n, "value"], "route names must be strings"],
])("the codec rejects misaligned or non-string opaque argument labels: %p", (names, message) => {
  const route = new ValueRoute("opaque", {
    valueJson: "3",
    names: ["", "value"],
    children: [ValueRoute.literal(1n), ValueRoute.literal(2n)],
  });
  const wire = wireOf(capturedEnvelope(new ForwardCapture({ x: 3n }, ValueRoute.record([["x", route]]))));
  wire.payload.forward_capture.route.children[0].names = names;
  expectError(() => decodeWire(wire), RangeError, message);
});
