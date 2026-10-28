/**
 * Forward captures and the retained trace: `ForwardCapture` and `BackwardCapture`, their run-log
 * codec, trace deltas, the run state's trace checks, replay, and learning through the engine and
 * through CALL, MAP and LOOP growth.
 */

import { afterAll, afterEach, expect, spyOn, test } from "bun:test";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { buildGrow, callSubgraph, cloneChild } from "../../src/compile/expand.ts";
import { resolveLlmCascade } from "../../src/compile/llm_cascade.ts";
import { expansionDigest } from "../../src/compile/manifest.ts";
import { CompiledFlow, copyGraph, Edge, END_ID, Flow, FlowOutput, START_ID } from "../../src/compile/model.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import { registerTemplateFn, TEMPLATE_FNS, type TemplateFn, type TemplateParam } from "../../src/expr/builtins.ts";
import { Failure, type FailureCode } from "../../src/diagnostics.ts";
import { NodeExpanded, NodeSucceeded, RunFailed, RunSucceeded } from "../../src/events.ts";
import { type ChatModel, type Event, EventStream } from "../../src/llm/events.ts";
import { MISSING_ROOT } from "../../src/expr/expressions.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import type { ModelAccess } from "../../src/llm_clients/factory.ts";
import {
  type FeedbackValue,
  type RootRead,
  routeFeedback,
  SourceRef,
  sourceKey,
  ValueRoute,
} from "../../src/expr/lineage.ts";
import { evalBinding } from "../../src/expr/template.ts";
import { MemoryScope, ModelTextInteraction } from "../../src/memory/contracts.ts";
import { AgentNode } from "../../src/nodes/agent/node.ts";
import { DeclSite, MemorySnapshotEnvelope, MemorySpec, MemoryStack } from "../../src/memory/index.ts";
import { BindingCapture, BindingError } from "../../src/nodes/binding.ts";
import {
  BackwardContext,
  BackwardResult,
  ELSE_HANDLE,
  Grow,
  Node,
  type NodeCaps,
  NodeKind,
  type NodeResult,
  Output,
  Route,
  retainFailure,
} from "../../src/nodes/base.ts";
import { CallNode } from "../../src/nodes/call/node.ts";
import { CaseNode } from "../../src/nodes/case/node.ts";
import { CodeNode } from "../../src/nodes/code/node.ts";
import {
  ChildFlowBoundary,
  ExecutionPosition,
  ExecutionScope,
  Parameter,
  SeedConnection,
  ValueContract,
} from "../../src/nodes/contracts.ts";
import { EndNode } from "../../src/nodes/end/node.ts";
import { LoopNode } from "../../src/nodes/loop/node.ts";
import { MapNode } from "../../src/nodes/map/node.ts";
import { StartNode } from "../../src/nodes/start/node.ts";
import type { FlowEngine } from "../../src/runtime/engine.ts";
import { computeBackward } from "../../src/runtime/backward.ts";
import { EvaluatedOutcome, evalNode } from "../../src/runtime/eval_node.ts";
import { BackwardCapture, BackwardOutcome, ForwardCapture } from "../../src/runtime/contracts.ts";
import {
  BackwardFinishedRecord,
  BackwardReadyRecord,
  BackwardStartedRecord,
  canonicalSeed,
  canonicalTyped,
  decodeRecord,
  encodeRecord,
  JsonlRunLog,
  type MemoryRunLog,
  NodeExpandedRecord,
  NodeFailedRecord,
  NodeRoutedRecord,
  NodeStartedRecord,
  NodeSucceededRecord,
  PoolCommitRecord,
  RecordEnvelope,
  RunLogSchemaError,
  type RunRecord,
  RunResumedRecord,
  RunStartedRecord,
  RunStoppedRecord,
  SCHEMA_VERSION,
} from "../../src/runtime/log.ts";
import {
  captureBindings,
  checkMemoryScope,
  type OutcomeRecord,
  type PrepareTraceOptions,
  prepareTrace,
  type ReboundSource,
  resolvePoolPath,
  type Rebinder,
  RetainedTrace,
  SeedRoutes,
  SourceEntry,
  sameSourceEntry,
  selectSeedRoute,
  TraceDelta,
  type TraceDeltaFields,
  TraceOccurrence,
  TraceTerminal,
} from "../../src/runtime/trace.ts";
import {
  commitPrepared,
  type GrowBuilder,
  PreparedTransition,
  prepareNodeFailed,
  prepareNodeRouted,
  prepareNodeStarted,
  prepareNodeStep,
  prepareNodeSucceeded,
  preparePoolCommit,
  prepareRecord,
  prepareRunResumed,
  prepareRunStopped,
  replayRecords,
  RunState,
} from "../../src/runtime/state.ts";
import { FileRef, Type, type Value, ValueKind, VariablePool } from "../../src/typesys/index.ts";
import { JSONDecodeError, type JsonObject, logJson, parseJson } from "../../src/typesys/json.ts";
import {
  aborted,
  BlockingNode,
  BranchNode,
  childFlow,
  commitStartSuccess,
  engine,
  expectUnchanged,
  FailNode,
  flow,
  forwardCapture,
  fragmentBuilder,
  FuncNode,
  graph,
  initializedAndStarted,
  learnableStarted,
  NO_ENV,
  noGrowth,
  prepareOutput,
  rejected,
  replayTraceSnapshot,
  RUN_ID,
  snapshot,
  startTraceNode,
  terminalSkipRouteFlow,
  textReply,
  traceSnapshot,
  prepareGrowth,
  fixedRebinder,
  serving,
} from "../helpers/run_state_fixtures.ts";
import { replayRunLog, runFlow } from "../../src/run.ts";

type ErrorClass = typeof TypeError | typeof RangeError;

/** A record route with one literal field `x`. */
const ROUTE_X = ValueRoute.record([["x", ValueRoute.literal(1)]]);

/** A capture of `x`. */
const FINISHED = new ForwardCapture(ROUTE_X);

test("a forward capture keeps invocation names", () => {
  const route = ValueRoute.record([["result", ValueRoute.literal({ items: [1] })]]);
  const capture = new ForwardCapture(route, { inputNames: ["result"], variableNames: ["instruction"] });
  expect(capture.inputNames).toEqual(["result"]);
  expect(capture.variableNames).toEqual(["instruction"]);
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
])("a forward capture rejects malformed invocation inventory: %s", (_label, field, value, cls, message) => {
  const options = { inputNames: ["result"], variableNames: ["instruction"], [field]: value };
  const route = ValueRoute.record([["result", ValueRoute.literal(1)]]);
  const build = () => new ForwardCapture(route, options as never);
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
    expectJsonDecodeError(() => new ForwardCapture(ValueRoute.record([["x", route]])), reason);
  },
);

test("a forward capture rejects duplicate record names", () => {
  const route = ValueRoute.record([
    ["x", ValueRoute.literal(1)],
    ["x", ValueRoute.literal(2)],
  ]);
  const build = () => new ForwardCapture(route);
  expect(build).toThrow(RangeError);
  expect(build).toThrow("unique");
});

test("a forward capture keeps frozen copies of its values", () => {
  const bound = { x: [1, 2] };
  const keys = new Set(["x"]);
  const items: [number, ForwardCapture][] = [[0, FINISHED], [2, FINISHED]];
  const names = ["x"];
  const scope = new ExecutionScope(["work"]);
  const capture = new ForwardCapture(ROUTE_X, {
    evaluatedKeys: keys,
    items,
    scope,
    inputNames: names,
    variableNames: ["tone"],
  });
  bound.x.push(3);
  keys.add("y");
  items.push([3, FINISHED]);
  items[0]![0] = 9;
  names.push("y");

  expect(capture.route).toBe(ROUTE_X);
  expect([...capture.evaluatedKeys]).toEqual(["x"]);
  expect(capture.evaluatedKeys).not.toBe(keys);
  expect(capture.items).toEqual([[0, FINISHED], [2, FINISHED]]);
  expect(capture.items[0]![1]).toBe(FINISHED);
  expect(capture.scope).toBe(scope);
  expect(capture.inputNames).toEqual(["x"]);
  expect(capture.variableNames).toEqual(["tone"]);

  expect(() => (capture.items as unknown[]).push([5, FINISHED])).toThrow(TypeError);
  expect(() => {
    (capture.items[0] as unknown as number[])[0] = 7;
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
  const capture = new ForwardCapture(ROUTE_X, { evaluatedKeys: new ShiftingSet(["x"]) });
  expect([...capture.evaluatedKeys]).toEqual(["x"]);
});

test("a forward capture defaults to empty keys, items and inventories", () => {
  const capture = new ForwardCapture(ValueRoute.record([]));
  expect(capture.evaluatedKeys.size).toBe(0);
  expect(capture.items).toEqual([]);
  expect(capture.scope).toBeNull();
  expect(capture.inputNames).toEqual([]);
  expect(capture.variableNames).toEqual([]);
});

test.each<[string, () => unknown, ErrorClass, string]>([
  [
    "a route that isn't a record",
    () => new ForwardCapture(ValueRoute.literal(1)),
    RangeError,
    "input route must be a record",
  ],
  [
    "evaluated keys that aren't a set",
    () => new ForwardCapture(ROUTE_X, { evaluatedKeys: ["x"] as never }),
    TypeError,
    "evaluated_keys must be a set of strings",
  ],
  [
    "a non-string evaluated key",
    () => new ForwardCapture(ROUTE_X, { evaluatedKeys: new Set([1]) as never }),
    TypeError,
    "evaluated_keys must be a set of strings",
  ],
  [
    "an evaluated key that isn't captured",
    () => new ForwardCapture(ROUTE_X, { evaluatedKeys: new Set(["y"]) }),
    RangeError,
    "evaluated_keys must name captured routes",
  ],
  [
    "items that aren't an array",
    () => new ForwardCapture(ROUTE_X, { items: new Set() as never }),
    TypeError,
    "items must be an array",
  ],
  [
    "an item that isn't a pair",
    () => new ForwardCapture(ROUTE_X, { items: [[0]] as never }),
    TypeError,
    "item captures must be index/capture pairs",
  ],
  [
    "a negative item index",
    () => new ForwardCapture(ROUTE_X, { items: [[-1, FINISHED]] }),
    RangeError,
    "item indices must be non-negative and increasing",
  ],
  [
    "a repeated item index",
    () => new ForwardCapture(ROUTE_X, { items: [[1, FINISHED], [1, FINISHED]] }),
    RangeError,
    "item indices must be non-negative and increasing",
  ],
  [
    "an item that isn't a ForwardCapture",
    () => new ForwardCapture(ROUTE_X, { items: [[0, { ...FINISHED }]] as never }),
    TypeError,
    "item capture must be a ForwardCapture",
  ],
  [
    "a scope that isn't an ExecutionScope",
    () => new ForwardCapture(ROUTE_X, { scope: new ExecutionPosition("call", ["work"]) as never }),
    TypeError,
    "scope must be an ExecutionScope or null",
  ],
])("a forward capture rejects %s", (_label, build, cls, message) => {
  expect(build).toThrow(cls);
  expect(build).toThrow(message);
});

test("a backward capture defaults to empty inventories and no scope", () => {
  const capture = new BackwardCapture();
  expect(capture.inputNames).toEqual([]);
  expect(capture.variableNames).toEqual([]);
  expect(capture.scope).toBeNull();
});

test("a backward capture keeps frozen copies of its inventories", () => {
  const names = ["result"];
  const scope = new ExecutionScope(["work"]);
  const capture = new BackwardCapture({ inputNames: names, variableNames: ["tone"], scope });
  names.push("other");
  expect(capture.inputNames).toEqual(["result"]);
  expect(capture.variableNames).toEqual(["tone"]);
  expect(capture.scope).toBe(scope);
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
    "a scope that isn't an ExecutionScope",
    "scope",
    { staticPath: ["work"] },
    TypeError,
    "scope must be an ExecutionScope or null",
  ],
])("a backward capture rejects %s", (_label, field, value, cls, message) => {
  const options = { inputNames: ["result"], variableNames: ["instruction"], [field]: value };
  const build = () => new BackwardCapture(options as never);
  expect(build).toThrow(cls);
  expect(build).toThrow(message);
});

// --------------------------------------------------------------------------------------- //
// Captures on the run log
// --------------------------------------------------------------------------------------- //

/** A node-success envelope at seq 3 carrying `capture`, naming its commit at seq 2. */
function capturedEnvelope(capture: ForwardCapture) {
  const record = new NodeSucceededRecord("work", 2, { forwardCapture: capture });
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

test("a forward capture round trips invocation names", () => {
  const capture = new ForwardCapture(ValueRoute.record([["result", ValueRoute.literal({ items: [1n] })]]),
    { inputNames: ["result"], variableNames: ["instruction"] },
  );
  const envelope = capturedEnvelope(capture);

  const decoded = decodeRecord(encodeRecord(envelope));

  expect(decoded).toEqual(envelope);
  const decodedCapture = (decoded.record as NodeSucceededRecord).forwardCapture!;
  expect(decodedCapture.inputNames).toEqual(["result"]);
  expect(decodedCapture.variableNames).toEqual(["instruction"]);
  expect(decoded.schema).toBe(SCHEMA_VERSION);
  const old = wireOf(envelope);
  old.schema = 6n;
  expectError(() => decodeWire(old), RunLogSchemaError, "unsupported run-record schema: 6");
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
])("a capture's record rejects malformed invocation inventory: %s", (_label, field, value, cls, message) => {
  const capture = new ForwardCapture(ValueRoute.record([["result", ValueRoute.literal(1n)]]), {
    inputNames: ["result"],
    variableNames: ["instruction"],
  });
  const wire = wireOf(capturedEnvelope(capture));
  wire.payload.forward_capture[field] = value;
  expectError(() => decodeWire(wire), cls, message);
});

test.each(NONCANONICAL_JSON)(
  "a capture's record rejects noncanonical embedded JSON: %s",
  (_label, valueJson, reason) => {
    const capture = new ForwardCapture(ValueRoute.record([["x", ValueRoute.literal(null)]]));
    const wire = wireOf(capturedEnvelope(capture));
    wire.payload.forward_capture.route.children[0].value_json = valueJson;
    expectJsonDecodeError(() => decodeWire(wire), reason);
  },
);

// The evaluated keys on the wire, and a key added to the capture's mutable `Set` after
// construction, which the encoder checks again.
test.each<[string, string[]]>([
  ["repeated", ["x", "x"]],
  ["unsorted", ["y", "x"]],
])("a capture's record rejects %s evaluated keys", (_label, keys) => {
  const route = ValueRoute.record([
    ["x", ValueRoute.literal(1n)],
    ["y", ValueRoute.literal(2n)],
  ]);
  const wire = wireOf(capturedEnvelope(new ForwardCapture(route)));
  wire.payload.forward_capture.evaluated_keys = keys;
  expectError(() => decodeWire(wire), RangeError, "evaluated_keys must be sorted and unique");
});

test("the encoder refuses an evaluated key added after construction", () => {
  const capture = new ForwardCapture(ValueRoute.record([["x", ValueRoute.literal(1n)]]));
  (capture.evaluatedKeys as Set<string>).add("y");
  const envelope = capturedEnvelope(capture);
  expectError(() => encodeRecord(envelope), RangeError, "evaluated_keys must name captured routes");
});

test.each(["input_names", "variable_names"])("a capture's record requires its explicit inventory %s", (field) => {
  const wire = wireOf(capturedEnvelope(new ForwardCapture(ValueRoute.record([]))));
  delete wire.payload.forward_capture[field];
  expectError(() => decodeWire(wire), RangeError, `invalid ${CAPTURE_LABEL} fields; missing=["${field}"], extra=[]`);
});

test("a forward capture round trips without a memory scope", () => {
  const inputs = { value: [1n, null] };
  const nested = new ValueRoute("source", { source: new SourceRef("run", 1, ["nested"]) });
  const route = ValueRoute.record([["value", nested]]);
  const item = new ForwardCapture(ValueRoute.record([["x", ValueRoute.literal(null)]]), {
    evaluatedKeys: new Set(["x"]),
  });
  const capture = new ForwardCapture(route, {
    evaluatedKeys: new Set(["value"]),
    items: [[0, item]],
    scope: new ExecutionScope(["work"]),
  });
  inputs.value.push(99n);
  const envelope = capturedEnvelope(capture);

  const decoded = decodeRecord(encodeRecord(envelope));

  expect(decoded).toEqual(envelope);
  expect(() => {
    (capture as { scope: ExecutionScope | null }).scope = null;
  }).toThrow(TypeError);
  expect(decoded.schema).toBe(SCHEMA_VERSION);
  const old = wireOf(envelope);
  old.schema = 6n;
  expectError(() => decodeWire(old), RunLogSchemaError, "unsupported run-record schema: 6");
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
  // Schema 10 routes have no `count`, so a `count` is an extra field.
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
  const capture = new ForwardCapture(ValueRoute.record([["x", new ValueRoute("source", { source: new SourceRef("run", 1) })]]),
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
    const capture = new ForwardCapture(ValueRoute.record([["x", route]]));
    const envelope = capturedEnvelope(capture);
    expect(decodeRecord(encodeRecord(envelope))).toEqual(envelope);
  }
});

// Each registration's previous entry (or its absence), put back after the test.
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
function resolveVariants(names: readonly string[]): RootRead {
  const values: Record<string, Value> = { a: [1n], b: [2n], n: 3n };
  const head = names[0]!;
  if (!Object.hasOwn(values, head)) return { used: 1, present: false, value: null, why: null };
  return { used: 1, present: true, value: values[head]!, why: null, source: new SourceRef("run", 0, [head]) };
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
])("the codec preserves evaluated route variants, an interpolation's text left out: %s", (expression) => {
  // A file is the TS leaf JSON can't hold, so `partial()` gives a partial snapshot.
  standIn("partial", () => ({ kept: [1n], lost: new FileRef({ uri: "file:///tmp/x" }) }));
  standIn("score", () => Infinity);
  const traced = evalBinding(expression, resolveVariants, { trace: true });
  const withoutText = (route: ValueRoute): ValueRoute =>
    route.with({ children: route.children.map(withoutText), ...(route.kind === "interpolation" ? { valueJson: null } : {}) });
  const capture = new ForwardCapture(ValueRoute.record([["x", traced.route]]));
  const recorded = new ForwardCapture(ValueRoute.record([["x", withoutText(traced.route)]]));
  expect(decodeRecord(encodeRecord(capturedEnvelope(capture)))).toEqual(capturedEnvelope(recorded));
});

// TS builtins take named parameters, so `collect_args` takes `first`, `second` and `value`, each
// defaulting to 0; its call log holds the bound values.
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
  const traced = evalBinding("${collect_args(" + args + ")}", MISSING_ROOT, {
    trace: true,
  });
  const capture = new ForwardCapture(ValueRoute.record([["x", traced.route]]));
  const envelope = capturedEnvelope(capture);

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
  const wire = wireOf(capturedEnvelope(new ForwardCapture(ValueRoute.record([["x", route]]))));
  wire.payload.forward_capture.route.children[0].names = names;
  expectError(() => decodeWire(wire), RangeError, message);
});

// The trace entries and `RetainedTrace`'s reads. Entries are installed by setting
// the trace's internal maps directly.

const RUN = "run";

/** A source route reading `path` of execution `seq`. */
function sourceRoute(seq: number, path: (string | number)[] = []): ValueRoute {
  return new ValueRoute("source", { source: new SourceRef(RUN, seq, path), expression: `\${source_${seq}}` });
}

/** The canonical typed wire text of `value`, as a source entry holds it. */
function typedJson(value: unknown): string {
  return logJson(canonicalTyped(value, null)[0]);
}

/**
 * The commit each test success writes: its pool key and canonical typed value text. A trace reads
 * a success's value from its commit (`PrepareTraceOptions.committed`), never from the record, so
 * the seq a test success names is a stand-in.
 */
const COMMITTED = new WeakMap<RunRecord, { key: string; valueJson: string }>();

/** A success of `nodeId` writing the typed wire `value` under `target`, as `COMMITTED` records it. */
function committedSuccess(
  nodeId: string,
  target: string,
  value: { kind: string; value: unknown },
  options: { forwardCapture?: ForwardCapture | null } = {},
): NodeSucceededRecord {
  const record = new NodeSucceededRecord(nodeId, 1, options);
  COMMITTED.set(record, { key: target, valueJson: logJson(value) });
  return record;
}

/** `prepareTrace` with a test success's commit filled in from `COMMITTED`. */
function prepare(
  trace: RetainedTrace | null,
  envelope: RecordEnvelope,
  node: Node | null,
  options: PrepareTraceOptions = {},
): TraceDelta | null {
  const committed = COMMITTED.get(envelope.record) ?? null;
  return prepareTrace(trace, envelope, node, committed === null ? options : { committed, ...options });
}

/**
 * The envelopes of a learnable run: RunStarted, `NodeStarted("a")` (seq 1) and `NodeSucceeded("a")`
 * with `2n` (seq 2), whose capture routes input `v` to source 0 path `["x"]`. The START inputs
 * `{x: 1n}` (source 0) come from the run start's commit, which `startedTrace` installs without a
 * seq of its own.
 */
function learnableRun(): RecordEnvelope[] {
  const started = new RunStartedRecord({
    flowFingerprint: "a".repeat(64),
    startWiringOverride: {},
    effectiveLlmByStaticPath: {},
    numWorkers: 0,
    learnable: true,
  });
  const forwardCapture = new ForwardCapture(ValueRoute.record([["v", sourceRoute(0, ["x"])]]), {
    inputNames: ["v"],
  });
  return [
    new RecordEnvelope(SCHEMA_VERSION, RUN, 0, started),
    new RecordEnvelope(SCHEMA_VERSION, RUN, 1, new NodeStartedRecord("a")),
    new RecordEnvelope(SCHEMA_VERSION, RUN, 2, committedSuccess("a", "a", { kind: "int", value: 2n }, { forwardCapture })),
  ];
}

/** Install `entry` as an accepted source and as `target`'s current writer. */
function install(trace: RetainedTrace, target: string, entry: SourceEntry): void {
  trace._sources.set(sourceKey(entry.source!), entry);
  trace._sourceNames.set(sourceKey(entry.source!), target);
  trace._currentWriters.set(target, entry);
}

/**
 * Install the occurrence of `a` started at seq 1, accepted with `record` at seq 2, its bound inputs
 * rebuilding to `{v: 1}`.
 */
function installOccurrence(trace: RetainedTrace, record: OutcomeRecord, nodeData: JsonObject = {}): TraceOccurrence {
  const capture = record.forwardCapture!;
  const envelope = new RecordEnvelope(SCHEMA_VERSION, RUN, 2, record);
  const occurrence = new TraceOccurrence(new SourceRef(RUN, 1), "a", 2, envelope, capture.route, {
    rebind: () => ({ inputs: { v: 1n }, capture, inputRoute: capture.route }),
    nodeDataRebuilder: () => nodeData,
  });
  trace._occurrences.push(occurrence);
  trace.occurrenceBySource.set(sourceKey(occurrence.source), occurrence);
  return occurrence;
}

/**
 * A trace of `learnableRun()`: START's source 0, `a`'s output (source 1) and its occurrence, accepted
 * with `record` (by default the run's `NodeSucceeded`), whose node data rebuilds to `nodeData`.
 */
function learnableTrace(
  record: OutcomeRecord = learnableRun()[2]!.record as NodeSucceededRecord,
  nodeData: JsonObject = {},
): RetainedTrace {
  const trace = new RetainedTrace(RUN, "__start__", "end", ["x"]);
  install(trace, "__start__", new SourceEntry(new SourceRef(RUN, 0), 0, typedJson({ x: 1n })));
  install(trace, "a", new SourceEntry(new SourceRef(RUN, 1), 2, typedJson(2n)));
  installOccurrence(trace, record, nodeData);
  return trace;
}

test("captureBindings joins a binding capture and its backward capture", () => {
  const file = new FileRef({ uri: "file:///tmp/x" });
  const fileForm = { uri: "file:///tmp/x", mime: null, name: null };
  const item = new BindingCapture({ f: file }, ValueRoute.record([["f", sourceRoute(0, ["f"])]]), new Set());
  const bindings = new BindingCapture(
    { x: 1n, f: file },
    ValueRoute.record([
      ["x", sourceRoute(0, ["x"])],
      ["f", sourceRoute(0, ["f"])],
    ]),
    new Set(["x"]),
    [[0, item]],
  );
  const scope = new ExecutionScope(["a"]);
  const backward = new BackwardCapture({
    inputNames: ["x", "f"],
    variableNames: ["v"],
    scope,
  });

  const capture = captureBindings(bindings, backward);

  expect(capture.route).toBe(bindings.route);
  expect([...capture.evaluatedKeys]).toEqual(["x"]);
  expect(capture.scope).toBe(scope);
  expect(capture.inputNames).toEqual(["x", "f"]);
  expect(capture.variableNames).toEqual(["v"]);
  expect(capture.items).toHaveLength(1);
  const [index, itemCapture] = capture.items[0]!;
  expect(index).toBe(0);
  expect(itemCapture.inputNames).toEqual([]);
  expect(itemCapture.variableNames).toEqual([]);
  expect(itemCapture.scope).toBeNull();
});

test("captureBindings requires exact carriers", () => {
  const bindings = new BindingCapture({ x: 1n }, ROUTE_X, new Set());
  expectError(
    () => captureBindings({ ...bindings } as BindingCapture, new BackwardCapture()),
    TypeError,
    "retained evaluation requires a BindingCapture",
  );
  expectError(
    () => captureBindings(bindings, { inputNames: [] } as unknown as BackwardCapture),
    TypeError,
    "retained evaluation requires a BackwardCapture",
  );
});

test("a source entry's value restores files through its type, its stored value doesn't", () => {
  const fileForm = { uri: "f", mime: null, name: null };
  const files = new Type({
    kind: ValueKind.OBJECT,
    fields: new Map([["f", Type.scalar(ValueKind.FILE)]]),
    required: new Set(),
  });
  const entry = new SourceEntry(null, 0, typedJson({ f: new FileRef({ uri: "f" }) }), files);

  const value = entry.value as { f: FileRef };
  expect(FileRef.isFileRef(value.f)).toBe(true);
  expect(value.f.uri).toBe("f");
  expect(entry.storedValue).toEqual({ f: fileForm });
  expect(new SourceEntry(null, 0, entry.valueJson).value).toEqual({ f: fileForm });
  expect(entry.value).not.toBe(entry.value);
  expect(entry.storedValue).not.toBe(entry.storedValue);
  expect(Object.isFrozen(entry)).toBe(true);
});

test("sameSourceEntry compares source keys, outcome sequences, value texts and files", () => {
  const files = Type.scalar(ValueKind.FILE);
  const valueJson = logJson({ kind: "int", value: 1n });
  const entry = (
    source: SourceRef | null = new SourceRef(RUN, 1),
    seq = 2,
    text = valueJson,
    type: Type | null = files,
  ) => new SourceEntry(source, seq, text, type);

  expect(sameSourceEntry(entry(), entry())).toBe(true);
  expect(sameSourceEntry(entry(), entry(new SourceRef(RUN, 2)))).toBe(false);
  expect(sameSourceEntry(entry(null), entry())).toBe(false);
  expect(sameSourceEntry(entry(), entry(null))).toBe(false);
  expect(sameSourceEntry(entry(), entry(undefined, 3))).toBe(false);
  expect(sameSourceEntry(entry(), entry(undefined, 2, logJson({ kind: "int", value: 2n })))).toBe(false);
  expect(sameSourceEntry(entry(undefined, 2, valueJson, null), entry())).toBe(false);
});

const TERMINAL_FAILURE = new Failure({ code: "node.execute", message: "Failed.", nodeId: "a" });

test.each<[string, () => unknown, ErrorClass, string]>([
  [
    "succeeded not a boolean",
    () => new TraceTerminal(1 as unknown as boolean, null, null, TERMINAL_FAILURE),
    TypeError,
    "terminal succeeded must be a bool",
  ],
  [
    "a source of another class",
    () => new TraceTerminal(false, {} as SourceRef, null, TERMINAL_FAILURE),
    TypeError,
    "source must be an exact SourceRef",
  ],
  [
    "a source with a path",
    () => new TraceTerminal(false, new SourceRef(RUN, 1, ["x"]), null, TERMINAL_FAILURE),
    RangeError,
    "terminal source must identify a root value",
  ],
  [
    "a success without output",
    () => new TraceTerminal(true, null, null, null),
    RangeError,
    "successful terminal requires output text and no failure",
  ],
  [
    "a success with a failure",
    () => new TraceTerminal(true, null, typedJson(1n), TERMINAL_FAILURE),
    RangeError,
    "successful terminal requires output text and no failure",
  ],
  [
    "a failure without a failure",
    () => new TraceTerminal(false, null, null, null),
    RangeError,
    "failed terminal requires a failure and no output",
  ],
  [
    "a failure with output",
    () => new TraceTerminal(false, null, typedJson(1n), TERMINAL_FAILURE),
    RangeError,
    "failed terminal requires a failure and no output",
  ],
  [
    "a failure of another class",
    () => new TraceTerminal(false, null, null, { ...TERMINAL_FAILURE } as Failure),
    RangeError,
    "failed terminal requires a failure and no output",
  ],
  [
    "output that isn't a typed wire",
    () => new TraceTerminal(true, null, "1", null),
    RangeError,
    "typed-value wire must be a JSON object",
  ],
])("a trace terminal refuses %s", (_name, build, cls, message) => {
  expectError(build, cls, message);
});

test("a trace terminal's output is fresh canonical JSON, or null when it failed", () => {
  const output = typedJson({ f: new FileRef({ uri: "f" }) });
  const succeeded = new TraceTerminal(true, new SourceRef(RUN, 3), output, null);
  expect(succeeded.output).toEqual({ f: { uri: "f", mime: null, name: null } });
  expect(succeeded.output).not.toBe(succeeded.output);
  const failed = new TraceTerminal(false, new SourceRef(RUN, 1), null, TERMINAL_FAILURE);
  expect(failed.output).toBeNull();
  expect(failed.failure).not.toBe(TERMINAL_FAILURE);
  expect(failed.failure).toEqual(TERMINAL_FAILURE);
  expect(Object.isFrozen(failed)).toBe(true);
});

test("checkMemoryScope returns null or an exact MemoryScope", () => {
  const scope = new MemoryScope(["a"], { memories: ["notes"] });
  expect(checkMemoryScope(null)).toBeNull();
  expect(checkMemoryScope(scope)).toBe(scope);
  expectError(
    () => checkMemoryScope({ ...scope } as MemoryScope),
    TypeError,
    "historical memory scope must be a MemoryScope or null",
  );
});

test("a trace occurrence keeps a decoded copy of its envelope", () => {
  const keys = new Set<string>();
  const capture = new ForwardCapture(ValueRoute.record([["v", sourceRoute(0, ["x"])]]), {
    evaluatedKeys: keys,
  });
  const record = new NodeSucceededRecord("a", 1, { forwardCapture: capture });
  const envelope = new RecordEnvelope(SCHEMA_VERSION, RUN, 2, record);
  const hidden = new Set(["v"]);
  const scope = new MemoryScope(["a"], { memories: ["notes"] });

  const occurrence = new TraceOccurrence(new SourceRef(RUN, 1), "a", 2, envelope, capture.route, {
    memoryScope: scope,
    hiddenInputs: hidden,
  });
  const recordJson = encodeRecord(envelope);
  (capture.evaluatedKeys as Set<string>).add("v");
  hidden.add("w");

  expect(occurrence.recordJson).toBe(recordJson);
  expect(occurrence.envelope).not.toBe(envelope);
  expect(occurrence.envelope.record).toBe(occurrence.record);
  expect([...(occurrence.record as NodeSucceededRecord).forwardCapture!.evaluatedKeys]).toEqual([]);
  expect(occurrence.memoryScope).toBe(scope);
  expect([...occurrence.hiddenInputs]).toEqual(["v"]);
  expect(Object.isFrozen(occurrence)).toBe(true);
  expectError(
    () => new TraceOccurrence(new SourceRef(RUN, 1), "a", 2, envelope, capture.route, { memoryScope: {} as MemoryScope }),
    TypeError,
    "historical memory scope must be a MemoryScope or null",
  );
});

test("selectSeedRoute selects a seed field by name, or by item index and name", () => {
  const field = sourceRoute(0, ["x"]);
  const joined = sourceRoute(0, ["y"]);
  const itemField = sourceRoute(0, ["z"]);
  const item = new ForwardCapture(ValueRoute.record([["n", itemField]]));
  const capture = new ForwardCapture(ValueRoute.record([["v", field]]), { items: [[2, item]] });

  expect(selectSeedRoute(capture, new SeedConnection("child", "v", ["v"]))).toBe(field);
  const inputRoute = ValueRoute.record([["v", joined]]);
  expect(selectSeedRoute(capture, new SeedConnection("child", "v", ["v"]), inputRoute)).toBe(joined);
  expect(selectSeedRoute(capture, new SeedConnection("child", "n", [2, "n"]), inputRoute)).toBe(itemField);
  expect(selectSeedRoute(capture, new SeedConnection("child", "n", [3, "n"]))).toBeNull();
  expect(selectSeedRoute(capture, new SeedConnection("child", "w", ["w"]))).toBeNull();
});

test("a retained trace starts empty and checks its root input names", () => {
  const trace = new RetainedTrace(RUN, "__start__");
  expect(trace.endId).toBeNull();
  expect(trace.rootInputNames).toEqual([]);
  expect(trace.rootStartSource).toBeNull();
  expect(trace.terminal).toBeNull();
  expect(trace.nextSequence).toBe(0);
  expect(trace.occurrences).toEqual([]);
  expect(trace.boundaries).toEqual([]);
  expect(trace.sources.size).toBe(0);
  expectError(
    () => new RetainedTrace(RUN, "__start__", null, ["x", "x"]),
    RangeError,
    "root_input_names must contain unique names",
  );
});

test("lookupSource reads a root value or a path inside it as fresh canonical JSON", () => {
  const trace = new RetainedTrace(RUN, "__start__");
  const file = new FileRef({ uri: "f" });
  install(trace, "b", new SourceEntry(new SourceRef(RUN, 3), 4, typedJson({ items: [10n, { f: file }] })));

  expect(trace.lookupSource(new SourceRef(RUN, 3))).toEqual({
    items: [10n, { f: { uri: "f", mime: null, name: null } }],
  });
  expect(trace.lookupSource(new SourceRef(RUN, 3))).not.toBe(trace.lookupSource(new SourceRef(RUN, 3)));
  expect(trace.lookupSource(new SourceRef(RUN, 3, ["items", 1, "f", "uri"]))).toBe("f");
  expect(trace.lookupSource(new SourceRef(RUN, 3, ["items", 0]))).toBe(10n);
  const absent = "source path is absent from its accepted value";
  expectError(() => trace.lookupSource(new SourceRef(RUN, 3, ["missing"])), RangeError, absent);
  expectError(() => trace.lookupSource(new SourceRef(RUN, 3, ["items", 2])), RangeError, absent);
  expectError(() => trace.lookupSource(new SourceRef(RUN, 3, ["items", -1])), RangeError, absent);
  expectError(() => trace.lookupSource(new SourceRef(RUN, 3, ["items", "0"])), RangeError, absent);
  expectError(() => trace.lookupSource(new SourceRef(RUN, 3, [0])), RangeError, absent);
  expectError(() => trace.lookupSource(new SourceRef(RUN, 3, ["items", 0, "x"])), RangeError, absent);
  expectError(
    () => trace.lookupSource(new SourceRef(RUN, 5, ["x"])),
    RangeError,
    `no accepted source for ${sourceKey(new SourceRef(RUN, 5))}`,
  );
});

test("resolve reads each root's value, presence and producer", () => {
  const trace = learnableTrace();
  install(trace, "b", new SourceEntry(new SourceRef(RUN, 3), 4, typedJson({ f: "text" })));
  trace.system.set("k", typedJson("ambient"));
  const present = (value: Value, source: SourceRef | null) => ({ used: 2, present: true, value, why: null, source });
  const absent = (why: string) => ({ used: 2, present: false, value: null, why });

  expect(trace.resolve(["input", "x"])).toEqual(present(1n, new SourceRef(RUN, 0, ["x"])));
  expect(trace.resolve(["b", "output", "f"])).toEqual(present({ f: "text" }, new SourceRef(RUN, 3)));
  expect(trace.resolve(["b", "output"])).toEqual(present({ f: "text" }, new SourceRef(RUN, 3)));
  expect(trace.resolve(["system", "k"])).toEqual(present("ambient", null));
  expect(trace.resolve(["system", "absent"])).toEqual(absent("system value 'absent' is not set"));
  expect(trace.resolve(["unknown", "output"])).toEqual(absent("node 'unknown' was skipped"));
  expect(trace.resolve(["input", "y"])).toEqual(absent("input 'y' was not given"));
  expect(trace.resolve(["x", "k"])).toEqual({ used: 1, present: false, value: null, why: null });
});

test("a traced pool read returns the root, with its source projected to nothing", () => {
  const pool = new VariablePool();
  pool.set("a", { k: { m: 1n } });
  const read = resolvePoolPath(["a", "output", "k", "m"], pool, () => new SourceRef("r", 3));
  expect(read.used).toBe(2);
  expect(read.present).toBe(true);
  expect(read.value).toEqual({ k: { m: 1n } });
  expect(read.source).toEqual(new SourceRef("r", 3, []));
});

test("occurrence finds an accepted occurrence by its start", () => {
  const trace = learnableTrace();
  expect(trace.occurrence(new SourceRef(RUN, 1)).nodeId).toBe("a");
  expect(trace.occurrences.map((item) => item.nodeId)).toEqual(["a"]);
  expectError(
    () => trace.occurrence(new SourceRef(RUN, 0)),
    RangeError,
    `no accepted occurrence for ${sourceKey(new SourceRef(RUN, 0))}`,
  );
});

test("backwardContext adds the evaluation evidence from the material and reads a success's output", () => {
  const { forwardCapture } = learnableRun()[2]!.record as NodeSucceededRecord;
  const trace = learnableTrace(new NodeSucceededRecord("a", 1, { forwardCapture }));
  const material = {
    interactions: [new ModelTextInteraction("Done.")],
    baseSystemPrompt: "System.",
    prompt: "Prompt.",
    memorySections: [],
    digestOk: true,
    requests: new Map(),
  };
  const source = new SourceRef(RUN, 1);
  const context = trace.backwardContext(source, { downstreamContext: [{ index: 0n }], material });

  expect(context.inputs).toEqual({ v: 1n });
  expect(context.inputNames).toEqual(["v"]);
  expect(context.output).toBe(2n);
  expect(context.failure).toBeNull();
  expect(context.runId).toBe(RUN);
  expect(context.nodeId).toBe("a");
  expect(context.nodeStartedSeq).toBe(1);
  expect(context.nodeData).toEqual({
    _evaluation_evidence: {
      scope: null,
      bound_inputs: { v: 1n },
      base_system_prompt: "System.",
      prompt: "Prompt.",
      memory_sections: [],
      interactions: [{ kind: "model_text", text: "Done." }],
    },
  });
  expect(context.downstreamContext).toEqual([{ index: 0n }]);
  // Without material, AGENT backward reads no evidence.
  expect(trace.backwardContext(source).nodeData).toEqual({ _evaluation_evidence: null });
});

test("backwardContext refuses node data using the reserved evidence key", () => {
  const capture = new ForwardCapture(ValueRoute.record([["v", ValueRoute.literal(1n)]]));
  const trace = learnableTrace(new NodeRoutedRecord("a", "next", { forwardCapture: capture }), {
    _evaluation_evidence: "forged",
  });
  expectError(
    () => trace.backwardContext(new SourceRef(RUN, 1)),
    RangeError,
    "node data uses reserved _evaluation_evidence",
  );
});

test("backwardContext takes a routed terminal failure and reads output only for a success", () => {
  const capture = new ForwardCapture(ValueRoute.record([["v", ValueRoute.literal(1n)]]));
  const failure = new Failure({ code: "graph.terminal_unreachable", message: "Unreachable.", nodeId: "a" });
  // Source 1 holds a value, so the null output comes from the record's class.
  const record = new NodeRoutedRecord("a", "next", { terminalFailure: failure, forwardCapture: capture });
  const trace = learnableTrace(record, { k: "kept" });

  const context = trace.backwardContext(new SourceRef(RUN, 1));

  expect(context.failure).toEqual(failure);
  expect(context.output).toBeNull();
  expect(context.nodeData).toEqual({ k: "kept", _evaluation_evidence: null });
});

// Trace deltas and `prepareTrace`, driven over `learnableRun()`.

/** A CODE node with a settable scope, memory scope and hidden inputs, recording its backward calls. */
class TraceNode extends Node {
  readonly kind = NodeKind.CODE;
  /** The inputs `modelHiddenInputs` reports. */
  hidden: ReadonlySet<string> = new Set();
  /** Each backward call's context and feedback, in call order. */
  readonly calls: [BackwardContext, readonly FeedbackValue[]][] = [];

  override modelHiddenInputs(): ReadonlySet<string> {
    return this.hidden;
  }

  override backward(context: BackwardContext, feedback: readonly FeedbackValue[]): BackwardResult {
    this.calls.push([context, feedback]);
    return new BackwardResult();
  }
}

/** The root metadata a learnable RunStarted is prepared with. */
const ROOT: PrepareTraceOptions = { rootStartId: "__start__", rootEndId: "end", rootInputNames: ["x"] };

/** `learnableRun()`'s capture: input `v` routed to source 0 path `["x"]`. */
function captureV(options: ConstructorParameters<typeof ForwardCapture>[1] = {}): ForwardCapture {
  return new ForwardCapture(ValueRoute.record([["v", sourceRoute(0, ["x"])]]), {
    inputNames: ["v"],
    ...options,
  });
}

/** An envelope of `record` at `seq`. */
function at(seq: number, record: RunRecord): RecordEnvelope {
  return new RecordEnvelope(SCHEMA_VERSION, RUN, seq, record);
}

/** A success of `nodeId` writing the int `value` under `target`, captured by `capture`. */
function success(nodeId: string, target = nodeId, value = 2n, capture: ForwardCapture | null = captureV()) {
  return committedSuccess(nodeId, target, { kind: "int", value }, { forwardCapture: capture });
}

/** Prepare `envelope`, check the delta's origin and apply it; the trace unchanged when there is no delta. */
function step(
  trace: RetainedTrace | null,
  envelope: RecordEnvelope,
  node: Node | null = null,
  options: PrepareTraceOptions = {},
): RetainedTrace | null {
  const delta = prepare(trace, envelope, node, options);
  if (delta === null) return trace;
  delta.validateOrigin(trace);
  return delta.apply();
}

/**
 * The trace after `learnableRun()`'s RunStarted with the START inputs `{x: 1n}` installed as the
 * run start's commit does (source 0, outcome seq 0), and after `NodeStarted("a")` too when
 * `started`.
 */
function startedTrace(started = false): RetainedTrace {
  const [run, start] = learnableRun();
  const trace = step(null, run!, null, ROOT)!;
  install(trace, "__start__", new SourceEntry(new SourceRef(RUN, 0), 0, typedJson({ x: 1n })));
  return started ? step(trace, start!)! : trace;
}

/** The trace after all of `learnableRun()`, with `node` as `a`. */
function drivenTrace(node: TraceNode = new TraceNode("a"), options: PrepareTraceOptions = {}): RetainedTrace {
  return step(startedTrace(true), learnableRun()[2]!, node, options)!;
}

/** Start `node` at `seq` and accept `record` for it at `seq + 1`. */
function runNode(
  trace: RetainedTrace,
  seq: number,
  node: TraceNode,
  record: RunRecord,
  options: PrepareTraceOptions = {},
): RetainedTrace {
  return step(step(trace, at(seq, new NodeStartedRecord(node.id)))!, at(seq + 1, record), node, options)!;
}

test("a binding that finished must rebind; only one that didn't rebuilds to null", () => {
  const unfinished = fixedRebinder(null);
  // A success's binding finished, so a rebind that fails is a fault, not missing inputs.
  const succeeded = drivenTrace(new TraceNode("a"), { rebinder: unfinished });
  expectError(
    () => succeeded.occurrences[0]!.boundInputs,
    RangeError,
    "the bound inputs of 'a' don't rebuild: the binding didn't finish",
  );
  // A binding that didn't finish, as its capture records, bound none, and nothing is rebound.
  let rebinds = 0;
  const counted: Rebinder = (sources, resolve, route) => {
    rebinds += 1;
    return unfinished(sources, resolve, route);
  };
  const failedWith = (code: FailureCode, capture: ForwardCapture, rebinder: Rebinder) => {
    const failure = new Failure({ code, message: "Bad input.", nodeId: "a" });
    return runNode(startedTrace(), 1, new TraceNode("a"), new NodeFailedRecord(failure, { forwardCapture: capture }), {
      rebinder,
      terminalSucceeded: false,
      terminalFailure: failure,
    }).occurrences[0]!;
  };
  expect(failedWith("binding.evaluate", captureV({ bound: false }), counted).boundInputs).toBeNull();
  expect(rebinds).toBe(0);
  // A body failing with a binding code after its binding finished rebuilds its inputs.
  expect(failedWith("contract.input", captureV(), fixedRebinder({ v: 1n })).boundInputs).toEqual({ v: 1n });
});

test("only a failure's capture may hold a binding that didn't finish", () => {
  const record = new NodeSucceededRecord("a", 2, { forwardCapture: captureV({ bound: false }) });
  expectError(
    () => runNode(startedTrace(), 1, new TraceNode("a"), record, { rebinder: fixedRebinder({ v: 1n }) }),
    RangeError,
    "only a failure's capture may hold a binding that didn't finish",
  );
});

test("prepareTrace and apply build the trace record by record", () => {
  const [run, , learned] = learnableRun();
  const node = new TraceNode("a");
  node.hidden = new Set(["v"]);
  node.memoryScope = new MemoryScope(["a"], { memories: ["notes"] });

  const trace = step(null, run!, null, ROOT)!;
  expect(trace.runId).toBe(RUN);
  expect(trace.startId).toBe("__start__");
  expect(trace.endId).toBe("end");
  expect(trace.rootInputNames).toEqual(["x"]);
  expect(trace.nextSequence).toBe(1);
  expect(trace.currentWriters.size).toBe(0);

  // The run start's START inputs commit installs source 0; a node's commit only advances the trace.
  const inputs = at(1, new PoolCommitRecord("__start__", "__start__", { kind: "object", value: { x: 1n } }));
  expect(step(trace, inputs, null, { initialCommit: true })).toBe(trace);
  const startEntry = new SourceEntry(new SourceRef(RUN, 0), 0, typedJson({ x: 1n }));
  expect(sameSourceEntry(trace.currentWriters.get("__start__")!, startEntry)).toBe(true);
  expect(trace.sources.get(sourceKey(new SourceRef(RUN, 0)))).toBe(trace.currentWriters.get("__start__"));
  expect(trace.rootWriter).toBeNull();
  expect(trace.started.size).toBe(0);

  expect(step(trace, at(2, new NodeStartedRecord("a")))).toBe(trace);
  expect([...trace.started]).toEqual([["a", 2]]);
  expect(trace.nextSequence).toBe(3);
  expect(step(trace, at(3, new PoolCommitRecord("a", "a", { kind: "int", value: 2n })))).toBe(trace);
  expect(trace.currentWriters.has("a")).toBe(false);
  expect(trace.nextSequence).toBe(4);

  const files = Type.scalar(ValueKind.FILE);
  const succeeded = at(4, committedSuccess("a", "a", { kind: "int", value: 2n }, {
    forwardCapture: (learned!.record as NodeSucceededRecord).forwardCapture,
  }));
  // The rebinder gets each pool name the route read, with the producer's start, and reads the
  // value it wrote from the trace's history.
  let given: [ReadonlyMap<string, number>, ReboundSource] | null = null;
  const rebinder: Rebinder = (sources, resolve) => {
    given = [new Map(sources), resolve("__start__", 0)];
    return { capture: new BindingCapture({ v: 1n }, ValueRoute.record([]), new Set()), failure: null };
  };
  const rebuiltFrom: unknown[] = [];
  const nodeDataRebuilder = (inputs: Readonly<Record<string, unknown>> | null) => {
    rebuiltFrom.push(inputs);
    return { k: "rebuilt" };
  };
  // Node data needs the bound inputs, so a finished binding with no rebinder can't rebuild it.
  expectError(
    () => step(trace, succeeded, node, { writerFiles: files, nodeDataRebuilder }),
    RangeError,
    "rebuilding the node data of a finished binding requires a rebinder",
  );
  expect(trace.occurrences).toHaveLength(0);
  expect(step(trace, succeeded, node, { writerFiles: files, rebinder, nodeDataRebuilder })).toBe(trace);
  const source = new SourceRef(RUN, 2);
  const occurrence = trace.occurrences[0]!;
  expect(occurrence.source).toEqual(source);
  expect(occurrence.nodeId).toBe("a");
  expect(occurrence.outcomeSeq).toBe(4);
  expect(occurrence.recordJson).toBe(encodeRecord(succeeded));
  expect(occurrence.inputRoute).toBe((succeeded.record as NodeSucceededRecord).forwardCapture!.route);
  expect(occurrence.memoryScope).toBe(node.memoryScope);
  expect([...occurrence.hiddenInputs]).toEqual(["v"]);
  expect(trace.occurrence(source)).toBe(occurrence);
  expect(trace.started.size).toBe(0);
  expect(trace.currentWriters.get("a")!.files).toBe(files);
  const writer = new SourceEntry(source, 4, logJson({ kind: "int", value: 2n }), files);
  expect(sameSourceEntry(trace.currentWriters.get("a")!, writer)).toBe(true);
  expect(trace.sources.get(sourceKey(source))).toBe(trace.currentWriters.get("a"));
  expect(trace.rootStartSource).toBeNull();
  expect(trace.rootWriter).toBeNull();
  expect(trace.terminal).toBeNull();
  expect(trace.nextSequence).toBe(5);

  const handler = trace.backwardHandlers.get(sourceKey(source))!;
  const context = trace.backwardContext(source);
  handler(context, ["Review."]);
  expect(node.calls).toEqual([[context, ["Review."]]]);
  expect(context.inputs).toEqual({ v: 1n });
  // The node data is rebuilt once, from the bound inputs binding the node again gave.
  expect(context.nodeData).toEqual({ k: "rebuilt", _evaluation_evidence: null });
  expect(rebuiltFrom).toEqual([{ v: 1n }]);
  const [names, read] = given!;
  expect([...names]).toEqual([["__start__", 0]]);
  expect(read.value.toObject()).toEqual({ x: 1n });
  expect(read.source).toEqual(new SourceRef(RUN, 0));
});

test("the run start's commits install the system values and the START inputs, the root writer when START ends", () => {
  const record = new RunStartedRecord({
    flowFingerprint: "a".repeat(64),
    startWiringOverride: {},
    effectiveLlmByStaticPath: {},
    numWorkers: 0,
    learnable: true,
  });

  const trace = step(null, at(0, record), null, { ...ROOT, rootEndId: "__start__" })!;
  step(trace, at(1, new PoolCommitRecord("__start__", "system.k", { kind: "str", value: "ambient" })), null, {
    initialCommit: true,
  });
  expect(trace.rootWriter).toBeNull();
  step(trace, at(2, new PoolCommitRecord("__start__", "__start__", { kind: "object", value: { x: 1n } })), null, {
    initialCommit: true,
  });

  const start = trace.currentWriters.get("__start__")!;
  expect(sameSourceEntry(start, new SourceEntry(new SourceRef(RUN, 0), 0, typedJson({ x: 1n })))).toBe(true);
  expect(trace.rootWriter).toBe(start);
  expect([...trace.sources.keys()]).toEqual([sourceKey(new SourceRef(RUN, 0))]);
  expect(trace.resolve(["system", "k"])).toEqual({ used: 2, present: true, value: "ambient", why: null, source: null });
  expect(trace.nextSequence).toBe(3);
});

test("prepareTrace returns null for a run that doesn't learn and for records before its RunStarted", () => {
  const [run, start, succeeded] = learnableRun();
  const { flowFingerprint } = run!.record as RunStartedRecord;
  const plain = new RunStartedRecord({
    flowFingerprint,
    startWiringOverride: {},
    effectiveLlmByStaticPath: {},
    numWorkers: 0,
  });
  expect(prepareTrace(null, at(0, plain), null, ROOT)).toBeNull();
  expect(prepareTrace(null, start!, null)).toBeNull();
  expect(prepareTrace(null, at(2, success("a", "a", 2n, null)), new TraceNode("a"))).toBeNull();
  expect(prepare(startedTrace(true), succeeded!, new TraceNode("a"))).not.toBeNull();
});

test("terminals take the root writer, the failed occurrence or the first causal route", () => {
  const output = typedJson(2n);
  const succeeded = drivenTrace();
  const root = runNode(succeeded, 3, new TraceNode("b"), success("b", "end", 2n), {
    terminalSucceeded: true,
    terminalOutputJson: output,
  });
  const writer = root.currentWriters.get("end")!;
  expect(root.rootWriter).toBe(writer);
  expect(root.terminal!.succeeded).toBe(true);
  expect(root.terminal!.source).toEqual(new SourceRef(RUN, 3));
  expect(root.terminal!.outputJson).toBe(output);

  const failure = new Failure({ code: "node.execute", message: "Failed.", nodeId: "b" });
  const failedRecord = new NodeFailedRecord(failure, { forwardCapture: captureV() });
  const failed = runNode(drivenTrace(), 3, new TraceNode("b"), failedRecord, {
    terminalSucceeded: false,
    terminalFailure: failure,
  });
  expect(failed.terminal!.source).toEqual(new SourceRef(RUN, 3));
  expect(failed.terminal!.failure).toEqual(failure);

  const unreachable = new Failure({ code: "graph.terminal_unreachable", message: "Unreachable.", nodeId: "r" });
  const routedRecord = new NodeRoutedRecord("r", "next", { terminalFailure: unreachable, forwardCapture: captureV() });
  const routed = runNode(drivenTrace(), 3, new TraceNode("r"), routedRecord);
  expect(routed.causalRouteSource).toEqual(new SourceRef(RUN, 3));
  expect(routed.terminal).toBeNull();
  const second = new NodeRoutedRecord("q", "next", { terminalFailure: unreachable, forwardCapture: captureV() });
  const later = runNode(routed, 5, new TraceNode("q"), second, {
    terminalSucceeded: false,
    terminalFailure: unreachable,
  });
  expect(later.causalRouteSource).toEqual(new SourceRef(RUN, 3));
  expect(later.terminal!.source).toEqual(new SourceRef(RUN, 3));
});

test("the root START's first outcome sets the root start source", () => {
  const node = new TraceNode("__start__");
  const trace = runNode(startedTrace(), 1, node, success("__start__", "__start__", 2n));
  expect(trace.rootStartSource).toEqual(new SourceRef(RUN, 1));
  expect(trace.currentWriters.get("__start__")!.outcomeSeq).toBe(2);
});

/** A trace after `a` started, with `edit` applied to it, for one failing preparation. */
function editedTrace(edit: (trace: RetainedTrace) => void): RetainedTrace {
  const trace = startedTrace(true);
  edit(trace);
  return trace;
}

/** `a` with the execution scope `scope`. */
function scopedNode(scope: ExecutionScope | null): TraceNode {
  const node = new TraceNode("a");
  node.executionScope = scope;
  return node;
}

/** Prepare `a`'s success at seq 2 to `target` over `trace` with `capture`, `node` and `options`. */
function prepareA({
  capture = captureV(),
  options = {},
  node = new TraceNode("a"),
  trace = startedTrace(true),
  target = "a",
}: {
  capture?: ForwardCapture | null;
  options?: PrepareTraceOptions;
  node?: Node | null;
  trace?: RetainedTrace;
  target?: string;
} = {}): () => unknown {
  return () => prepare(trace, at(2, success("a", target, 2n, capture)), node, options);
}

const SCOPE_A = new ExecutionScope(["a"]);
const TYPED_TWO = typedJson(2n);

test.each<[string, () => unknown, ErrorClass, string]>([
  [
    "a learnable RunStarted without a root end",
    () => prepareTrace(null, learnableRun()[0]!, null, { rootInputNames: ["x"] }),
    RangeError,
    "retained RunStarted requires verified root metadata",
  ],
  [
    "a learnable RunStarted without root input names",
    () => prepareTrace(null, learnableRun()[0]!, null, { rootEndId: "end" }),
    RangeError,
    "retained RunStarted requires verified root metadata",
  ],
  [
    "a learnable RunStarted without a root start",
    () => prepareTrace(null, learnableRun()[0]!, null, { rootEndId: "end", rootInputNames: ["x"] }),
    RangeError,
    "retained RunStarted requires verified root metadata",
  ],
  [
    "an accepted success without its commit",
    () => prepareTrace(startedTrace(true), learnableRun()[2]!, new TraceNode("a")),
    RangeError,
    "an accepted success requires its commit",
  ],
  [
    "a capture with no trace",
    () => prepareTrace(null, learnableRun()[2]!, new TraceNode("a")),
    RangeError,
    "forward capture requires recorded learning",
  ],
  [
    "a record at another sequence",
    () => prepareTrace(startedTrace(), learnableRun()[2]!, new TraceNode("a")),
    RangeError,
    "trace disagrees with accepted record coordinates",
  ],
  [
    "a record of another run",
    () =>
      prepareTrace(
        startedTrace(),
        new RecordEnvelope(SCHEMA_VERSION, "other", 1, new NodeStartedRecord("a")),
        null,
      ),
    RangeError,
    "trace disagrees with accepted record coordinates",
  ],
  [
    "terminal output without a disposition",
    () => prepareTrace(startedTrace(), learnableRun()[1]!, null, { terminalOutputJson: TYPED_TWO }),
    RangeError,
    "terminal payload requires a terminal disposition",
  ],
  [
    "a terminal failure without a disposition",
    () => prepareTrace(startedTrace(), learnableRun()[1]!, null, { terminalFailure: TERMINAL_FAILURE }),
    RangeError,
    "terminal payload requires a terminal disposition",
  ],
  [
    "a resume naming other starts",
    () => prepareTrace(startedTrace(true), at(2, new RunResumedRecord(["b"], 0)), null),
    RangeError,
    "trace unfinished starts disagree with retry frontier",
  ],
  [
    "an outcome without a capture",
    prepareA({ capture: null }),
    RangeError,
    "accepted outcome requires forward capture while learning",
  ],
  [
    "an outcome without a node",
    prepareA({ node: null }),
    RangeError,
    "trace outcome requires its accepted start",
  ],
  [
    "an outcome of a node that didn't start",
    prepareA({ node: new TraceNode("b") }),
    RangeError,
    "trace outcome requires its accepted start",
  ],
  [
    "a scoped capture on a node without a scope",
    prepareA({ capture: captureV({ scope: SCOPE_A }), node: scopedNode(null) }),
    RangeError,
    "forward capture scope does not match the executing node",
  ],
  [
    "a capture without a scope on a scoped node",
    prepareA({ node: scopedNode(SCOPE_A) }),
    RangeError,
    "forward capture scope does not match the executing node",
  ],
  [
    "a capture and a node with different scopes",
    prepareA({ capture: captureV({ scope: SCOPE_A }), node: scopedNode(new ExecutionScope(["b"])) }),
    RangeError,
    "forward capture scope does not match the executing node",
  ],
  [
    "a route to an unaccepted source",
    prepareA({ capture: new ForwardCapture(ValueRoute.record([["v", sourceRoute(5)]])) }),
    RangeError,
    "route source must name an earlier accepted producer",
  ],
  [
    "a route to a source accepted at this record",
    prepareA({
      trace: editedTrace((trace) =>
        install(trace, "b", new SourceEntry(new SourceRef(RUN, 0), 2, typedJson({ x: 1n }))),
      ),
    }),
    RangeError,
    "route source must name an earlier accepted producer",
  ],
  [
    "an item route to an unaccepted source",
    prepareA({
      capture: captureV({ items: [[0, new ForwardCapture(ValueRoute.record([["n", sourceRoute(5)]]))]] }),
    }),
    RangeError,
    "route source must name an earlier accepted producer",
  ],
  [
    "a seed route to an unaccepted source",
    prepareA({
      trace: editedTrace((trace) =>
        trace._pendingConnections.set("a", new SeedRoutes(new SourceRef(RUN, 0), [["v", sourceRoute(5)]])),
      ),
    }),
    RangeError,
    "route source must name an earlier accepted producer",
  ],
  [
    "a seed field that wasn't captured",
    prepareA({ options: { inputConnections: [new SeedConnection("child", "w", ["w"])] } }),
    RangeError,
    "seed connection field or item was not captured",
  ],
  [
    "a seed item that wasn't captured",
    prepareA({ options: { inputConnections: [new SeedConnection("child", "n", [0, "n"])] } }),
    RangeError,
    "seed connection field or item was not captured",
  ],
  [
    "a seed target that already has an owner",
    prepareA({
      options: { inputConnections: [new SeedConnection("child", "p", ["v"])] },
      trace: editedTrace((trace) => trace._pendingConnections.set("child", new SeedRoutes(new SourceRef(RUN, 0), []))),
    }),
    RangeError,
    "seed connection target already has an accepted owner",
  ],
  [
    "a child boundary endpoint that already has an owner",
    prepareA({
      options: { childBoundaries: [new ChildFlowBoundary("cs", "ce")] },
      trace: editedTrace((trace) => trace.currentBoundaries.set("ce", [0, "endSource"])),
    }),
    RangeError,
    "child boundary target already has an accepted owner",
  ],
  [
    "a node whose backward isn't a function",
    prepareA({ node: Object.assign(new TraceNode("a"), { backward: "not callable" }) }),
    TypeError,
    "node backward handler must be callable",
  ],
  [
    "terminal output without a root writer",
    prepareA({ options: { terminalSucceeded: true, terminalOutputJson: TYPED_TWO } }),
    RangeError,
    "terminal output disagrees with the accepted root writer",
  ],
  [
    "terminal output other than the root writer's",
    prepareA({ options: { terminalSucceeded: true, terminalOutputJson: typedJson(3n) }, target: "end" }),
    RangeError,
    "terminal output disagrees with the accepted root writer",
  ],
  [
    "a successful terminal without output",
    prepareA({ options: { terminalSucceeded: true }, target: "end" }),
    RangeError,
    "terminal output disagrees with the accepted root writer",
  ],
  [
    "a terminal failure without a causal occurrence",
    prepareA({ options: { terminalSucceeded: false, terminalFailure: TERMINAL_FAILURE } }),
    RangeError,
    "accepted failure requires an accepted causal occurrence",
  ],
])("prepareTrace refuses %s", (_label, prepare, cls, message) => {
  expectError(prepare, cls, message);
});

test("prepareTrace keeps the lookup error as the cause of an absent source path", () => {
  const prepare = prepareA({
    capture: new ForwardCapture(ValueRoute.record([["v", sourceRoute(0, ["y"])]])),
  });
  let caught: Error | undefined;
  try {
    prepare();
  } catch (error) {
    caught = error as Error;
  }
  expect(caught!.constructor).toBe(RangeError);
  expect(caught!.message).toBe("route source path is absent from its accepted value");
  const cause = caught!.cause as Error;
  expect(cause.constructor).toBe(RangeError);
  expect(cause.message).toBe("source path is absent from its accepted value");
});

test("a trace delta's terminal compare reads the output text canonically", () => {
  // The same int written with other spacing is the same decoded typed value.
  const trace = step(startedTrace(true), at(2, success("a", "end")), new TraceNode("a"), {
    terminalSucceeded: true,
    terminalOutputJson: '{ "value": 2, "kind": "int" }',
  })!;
  expect(trace.terminal!.source).toEqual(new SourceRef(RUN, 1));
});

test.each<[string, (trace: RetainedTrace) => RetainedTrace | null, string]>([
  ["another trace", () => startedTrace(true), "trace transition requires its originating trace"],
  ["no trace", () => null, "trace transition requires its originating trace"],
  [
    "a stale sequence",
    (trace) => {
      trace.nextSequence += 1;
      return trace;
    },
    "trace transition has a stale sequence",
  ],
])("validateOrigin refuses %s", (_label, target, message) => {
  const trace = startedTrace(true);
  const delta = prepare(trace, learnableRun()[2]!, new TraceNode("a"))!;
  expectError(() => delta.validateOrigin(target(trace)), RangeError, message);
});

test("validateOrigin refuses an occurrence that isn't exactly a TraceOccurrence", () => {
  const trace = startedTrace(true);
  const { occurrence } = prepare(trace, learnableRun()[2]!, new TraceNode("a"))!;
  // Reads every field of the real occurrence through its prototype.
  const forged = new TraceDelta(trace, 2, { occurrence: Object.create(occurrence) as TraceOccurrence });
  expectError(() => forged.validateOrigin(trace), TypeError, "trace delta requires an exact occurrence");
});

test("validateOrigin refuses a consumed connection that is no longer pending", () => {
  const trace = drivenTrace(new TraceNode("a"), { inputConnections: [new SeedConnection("c", "p", ["v"])] });
  step(trace, at(3, new NodeStartedRecord("c")));
  const capture = new ForwardCapture(ValueRoute.record([["p", ValueRoute.literal(1n)]]));
  const delta = prepare(trace, at(4, success("c", "c", 1n, capture)), new TraceNode("c"))!;
  const { owner, fields } = trace.pendingConnections.get("c")!;
  trace._pendingConnections.set("c", new SeedRoutes(owner, fields));
  expectError(() => delta.validateOrigin(trace), RangeError, "seed connection no longer has its prepared owner");
  trace._pendingConnections.delete("c");
  expectError(() => delta.validateOrigin(trace), RangeError, "seed connection no longer has its prepared owner");
});

test("seed connections install sorted by target and input, and the child start consumes them", () => {
  const connections = [
    new SeedConnection("c", "q", ["v"]),
    new SeedConnection("b", "p", ["v"]),
    new SeedConnection("c", "p", ["v"]),
  ];
  const trace = drivenTrace(new TraceNode("a"), { inputConnections: connections });
  const seeded = (learnableRun()[2]!.record as NodeSucceededRecord).forwardCapture!.route.children[0]!;
  expect([...trace.pendingConnections.keys()]).toEqual(["b", "c"]);
  const c = trace.pendingConnections.get("c")!;
  expect(c.owner).toEqual(new SourceRef(RUN, 1));
  expect(c.fields.map(([name]) => name)).toEqual(["p", "q"]);
  expect(c.fields[0]![1].source).toEqual(seeded.source);

  const literal = ValueRoute.literal(1n);
  const evaluated = ValueRoute.literal(1n);
  const capture = new ForwardCapture(ValueRoute.record([
      ["p", literal],
      ["q", evaluated],
    ]),
    { evaluatedKeys: new Set(["q"]) },
  );
  runNode(trace, 3, new TraceNode("c"), success("c", "c", 1n, capture));

  const route = trace.occurrence(new SourceRef(RUN, 3)).inputRoute;
  expect(route.names).toEqual(["p", "q"]);
  expect(route.children[0]).toBe(c.fields[0]![1]);
  expect(route.children[1]).toBe(evaluated);
  expect([...trace.pendingConnections.keys()]).toEqual(["b"]);
});

test("child boundaries install sorted and take their endpoints' accepted starts", () => {
  const boundaries = [new ChildFlowBoundary("s2", "e2"), new ChildFlowBoundary("s1", "e1")];
  const trace = drivenTrace(new TraceNode("a"), { childBoundaries: boundaries });
  const installed = trace.boundaries;
  expect(installed.map((item) => [item.startId, item.endId])).toEqual([
    ["s1", "e1"],
    ["s2", "e2"],
  ]);
  expect(installed[0]!.owner).toEqual(new SourceRef(RUN, 1));
  expect(installed[0]!.startSource).toBeNull();

  runNode(trace, 3, new TraceNode("s1"), success("s1"));
  expect(trace.boundaries[0]!.startSource).toEqual(new SourceRef(RUN, 3));
  expect(trace.boundaries[0]!.endSource).toBeNull();
  expect(trace.currentBoundaries.has("s1")).toBe(false);
  runNode(trace, 5, new TraceNode("e1"), success("e1"));
  expect(trace.boundaries[0]!.startSource).toEqual(new SourceRef(RUN, 3));
  expect(trace.boundaries[0]!.endSource).toEqual(new SourceRef(RUN, 5));
  expect(trace.boundaries[0]!.owner).toEqual(new SourceRef(RUN, 1));
  expect(installed[0]!.startSource).toBeNull();
  expect(trace.boundaries[1]).toBe(installed[1]);
});

test("an expansion prunes current names only, and other outcomes ignore retired ids", () => {
  const trace = drivenTrace(new TraceNode("a"), {
    inputConnections: [new SeedConnection("b", "p", ["v"])],
    childBoundaries: [new ChildFlowBoundary("s1", "e1")],
    retiredIds: ["a"],
  });
  expect(trace.currentWriters.has("a")).toBe(true);

  const expanded = new NodeExpandedRecord("g", ["a", "b"], "b".repeat(64), { forwardCapture: captureV() });
  runNode(trace, 3, new TraceNode("g"), expanded, { retiredIds: ["a", "b", "s1"] });

  expect(trace.currentWriters.has("a")).toBe(false);
  expect(trace.pendingConnections.has("b")).toBe(false);
  expect(trace.currentBoundaries.has("s1")).toBe(false);
  expect(trace.currentBoundaries.has("e1")).toBe(true);
  expect(trace.sources.has(sourceKey(new SourceRef(RUN, 1)))).toBe(true);
  expect(trace.occurrences.map((item) => item.nodeId)).toEqual(["a", "g"]);
  expect(trace.boundaries).toHaveLength(1);
  expect(trace.resolve(["a", "output"]).present).toBe(false);
});

test("a resumed record clears the unfinished starts", () => {
  const trace = startedTrace(true);
  step(trace, at(2, new RunResumedRecord(["a"], 0)));
  expect(trace.started.size).toBe(0);
  expect(trace.nextSequence).toBe(3);
});

test("an applied delta never changes an earlier occurrence", () => {
  const trace = drivenTrace();
  const occurrences = trace.occurrences;
  const [occurrence] = occurrences;
  const { recordJson, inputRoute } = occurrence!;

  runNode(trace, 3, new TraceNode("b"), success("b", "a", 5n), { childBoundaries: [new ChildFlowBoundary("s", "e")] });

  expect(trace.occurrences[0]).toBe(occurrence);
  expect(occurrence!.recordJson).toBe(recordJson);
  expect(occurrence!.inputRoute).toBe(inputRoute);
  expect(Object.isFrozen(occurrence)).toBe(true);
  expect(occurrences).toHaveLength(1);
  expect(trace.lookupSource(new SourceRef(RUN, 1))).toBe(2n);
  const read = trace.resolve(["a", "output"]);
  expect(read).toEqual({ used: 2, present: true, value: 5n, why: null, source: new SourceRef(RUN, 3) });
});

test("a trace delta holds frozen copies of its arrays and defaults to an empty change", () => {
  const trace = startedTrace();
  const empty = new TraceDelta(trace, 1, {});
  expect(empty.originatingTrace).toBe(trace);
  expect(empty.sequence).toBe(1);
  for (const field of [
    "initialized",
    "started",
    "occurrence",
    "handler",
    "writer",
    "consumedConnection",
    "boundaryUpdate",
    "rootStartSource",
    "rootWriter",
    "causalRouteSource",
    "terminal",
  ] as const) {
    expect(empty[field]).toBeNull();
  }
  expect(empty.resumed).toBe(false);
  expect([empty.pruneIds, empty.connections, empty.boundaries]).toEqual([[], [], []]);

  const pruneIds = ["a"];
  const started: [string, number] = ["a", 1];
  const delta = new TraceDelta(trace, 1, { started, pruneIds });
  pruneIds.push("b");
  started[1] = 9;
  expect(delta.pruneIds).toEqual(["a"]);
  expect(delta.started).toEqual(["a", 1]);
  expect(Object.isFrozen(delta)).toBe(true);
  expect(Object.isFrozen(delta.pruneIds)).toBe(true);
  expect(Object.isFrozen(delta.started)).toBe(true);
});

// The frontier trace: the learnable stop.

/** A learnable started state with START in flight, its log and a prepared `RunStopped`. */
function preparedLearnableStop(): { state: RunState; log: MemoryRunLog; prepared: PreparedTransition } {
  const { state, log } = learnableStarted(flow());
  commitPrepared(log, state, prepareNodeStarted(state, START_ID));
  return { state, log, prepared: prepareRunStopped(state, "signal", "SIGTERM") };
}

/** Return `prepared` with its record or its plan's trace delta replaced. */
function forgedStop(
  prepared: PreparedTransition,
  changes: { record?: RunRecord; traceDelta?: TraceDelta },
): PreparedTransition {
  const plan = Object.freeze({ ...prepared._plan, traceDelta: changes.traceDelta ?? prepared._plan.traceDelta });
  return new PreparedTransition(
    changes.record ?? prepared.record,
    prepared.expectedRunId,
    prepared.expectedSequence,
    plan as never,
    [],
  );
}

test("a learnable RunStopped carries a sequence-only trace delta and commits", () => {
  const { state, log, prepared } = preparedLearnableStop();
  const trace = state.trace!;
  const delta = prepared._plan.traceDelta!;

  expect(delta).toBeInstanceOf(TraceDelta);
  expect(delta.originatingTrace).toBe(trace);
  expect(delta.sequence).toBe(4);
  expect(delta.started).toBeNull();
  expect(delta.resumed).toBe(false);

  expect(commitPrepared(log, state, prepared)).toEqual([]);

  expect(prepared.record).toStrictEqual(new RunStoppedRecord("signal", "SIGTERM", [START_ID]));
  // The trace applies in place: only its sequence moves.
  expect(state.trace).toBe(trace);
  expect(trace.nextSequence).toBe(5);
  expect([...trace._started]).toEqual([[START_ID, 3]]);
  expect(log.nextSequence).toBe(state.nextSequence);
});

test("a learnable stop whose trace delta starts a node is refused", () => {
  const { state, log, prepared } = preparedLearnableStop();
  const forged = forgedStop(prepared, {
    traceDelta: new TraceDelta(state.trace, state.nextSequence, { started: [START_ID, state.nextSequence] }),
  });

  expectError(
    () => commitPrepared(log, state, forged),
    RangeError,
    "frontier trace stop must only advance the accepted sequence",
  );

  expect(log.nextSequence).toBe(4);
  expect(state.trace!.nextSequence).toBe(4);
});

test("a learnable frontier plan holding an outcome record is refused", () => {
  const { state, log, prepared } = preparedLearnableStop();
  const failure = new Failure({ code: "node.execute", message: "done", nodeId: START_ID });
  const forged = forgedStop(prepared, { record: new NodeFailedRecord(failure) });

  expectError(
    () => commitPrepared(log, state, forged),
    RangeError,
    "frontier trace requires a dispatch, resume, step, stop, wait, wake, run-wait, close, withdraw or interrupt record",
  );

  expect(log.nextSequence).toBe(4);
});

// The run-state reducer's trace authority: outcome and terminal trace checks before append.

/** Return `prepared` with its plan's trace delta replaced by `traceDelta`. */
function withTrace(prepared: PreparedTransition, traceDelta: TraceDelta | null): PreparedTransition {
  return new PreparedTransition(
    prepared.record,
    prepared.expectedRunId,
    prepared.expectedSequence,
    Object.freeze({ ...prepared._plan, traceDelta }) as never,
    prepared.observers,
  );
}

/** Return a copy of `delta` with `changes`. */
function changedDelta(delta: TraceDelta, changes: TraceDeltaFields): TraceDelta {
  return new TraceDelta(delta.originatingTrace, delta.sequence, { ...delta, ...changes });
}

/** Return a copy of `occurrence` with `changes`, built through its constructor. */
function changedOccurrence(
  occurrence: TraceOccurrence,
  changes: {
    source?: SourceRef;
    nodeId?: string;
    outcomeSeq?: number;
    envelope?: RecordEnvelope;
    memoryScope?: MemoryScope | null;
  },
): TraceOccurrence {
  return new TraceOccurrence(
    changes.source ?? occurrence.source,
    changes.nodeId ?? occurrence.nodeId,
    changes.outcomeSeq ?? occurrence.outcomeSeq,
    changes.envelope ?? occurrence.envelope,
    occurrence.inputRoute,
    {
      memoryScope: changes.memoryScope === undefined ? occurrence.memoryScope : changes.memoryScope,
      hiddenInputs: occurrence.hiddenInputs,
      seed: occurrence.seed,
      steps: occurrence.steps,
      capture: occurrence.capture,
    },
  );
}

/** Return a copy of `entry` with `changes`. */
function changedEntry(entry: SourceEntry, changes: { source?: SourceRef; outcomeSeq?: number; valueJson?: string }) {
  return new SourceEntry(
    changes.source ?? entry.source,
    changes.outcomeSeq ?? entry.outcomeSeq,
    changes.valueJson ?? entry.valueJson,
    entry.files,
  );
}

/** Return a frozen copy of `value`'s own fields under `cls`'s prototype, skipping its constructor's checks. */
function forged<T extends object>(cls: abstract new (...args: never[]) => T, value: T, changes: object): T {
  return Object.freeze(Object.assign(Object.create(cls.prototype) as T, value, changes));
}

/**
 * Assert that committing `prepared` throws exactly `cls` with `message` and leaves the log, the
 * state and its trace unchanged.
 */
function expectRefused(
  state: RunState,
  log: MemoryRunLog,
  prepared: PreparedTransition,
  cls: ErrorClass,
  message: string,
): void {
  const records = log.records();
  const before = snapshot(state);
  const beforeTrace = traceSnapshot(state.trace!);

  expectError(() => commitPrepared(log, state, prepared), cls, message);

  expect(log.records()).toEqual(records);
  expectUnchanged(state, before);
  expect(traceSnapshot(state.trace!)).toEqual(beforeTrace);
  expect(state.nextSequence).toBe(log.nextSequence);
  expect(state.trace!.nextSequence).toBe(log.nextSequence);
}

/** The outcome kinds `preparedScopedOutcome` prepares. */
const OUTCOME_KINDS = ["success", "route", "failure", "growth"] as const;

/** An outcome kind `preparedScopedOutcome` prepares. */
type OutcomeKind = (typeof OUTCOME_KINDS)[number];

/** A CODE node that may grow the graph. */
class FuncSpawner extends FuncNode {
  override readonly isSpawner = true;
  // A spawner's seed is rebuilt from its binding on replay, so the double is derived and grows `{}`.
  override readonly derived = true;

  override forward(): Grow {
    return new Grow({});
  }
}

/**
 * A CODE node `physical` (scoped unless `scoped` is false),
 * the sole in-flight node of a learnable run, and its prepared outcome: a success committed to END,
 * which ends the run; a route along its `go` edge into END, which makes END ready; a failure, which
 * ends the run; or (a spawner) a growth into `fragment("physical")`. `original` is a copy of the
 * flow before the outcome, for a replay.
 */
function preparedScopedOutcome(
  kind: OutcomeKind = "success",
  options: { scoped?: boolean } = {},
): {
  node: FuncNode;
  state: RunState;
  log: MemoryRunLog;
  prepared: PreparedTransition;
  original: CompiledFlow;
} {
  const node = new (kind === "growth" ? FuncSpawner : FuncNode)("physical", () => "done");
  node.memoryScope = (options.scoped ?? true) ? new MemoryScope([node.id], { memories: ["memory"] }) : null;
  const { state, log } = startTraceNode(node, {
    keys: ["memory"],
    learnable: true,
    routeHandle: kind === "route" ? "go" : undefined,
  });
  const original = copyGraph(state.flow);
  const carriers = { forwardCapture: forwardCapture(node, {}) };
  let prepared: PreparedTransition;
  if (kind === "success") {
    prepared = prepareOutput(log, state, node.id, "done", { commitAs: END_ID, ...carriers });
  } else if (kind === "route") {
    prepared = prepareNodeRouted(state, node.id, "go", carriers);
  } else if (kind === "failure") {
    const failure = new Failure({ code: "node.execute", message: "failed", nodeId: node.id });
    prepared = prepareNodeFailed(state, failure, carriers);
  } else {
    prepared = prepareGrowth(log, state, node.id, new Grow({}), fragmentBuilder, carriers);
  }
  return { node, state, log, prepared, original };
}

/**
 * A learnable `terminalSkipRouteFlow({sibling})` after START's
 * success, with `router` (and with `sibling`, `unrelated`) in flight and `router`'s prepared `go`
 * route, which skips END and ends the run. `original` is a copy of the started flow, for a replay.
 */
function preparedTerminalRoute(options: { sibling?: boolean } = {}): {
  state: RunState;
  log: MemoryRunLog;
  prepared: PreparedTransition;
  original: CompiledFlow;
} {
  const { state, log } = learnableStarted(terminalSkipRouteFlow({ sibling: options.sibling }));
  const original = copyGraph(state.flow);
  commitPrepared(log, state, prepareNodeStarted(state, START_ID));
  commitStartSuccess(state, log, {});
  for (const nodeId of options.sibling ? ["router", "unrelated"] : ["router"]) {
    commitPrepared(log, state, prepareNodeStarted(state, nodeId));
  }
  const prepared = prepareNodeRouted(state, "router", "go", {
    forwardCapture: forwardCapture(state.flow.nodes.get("router")!, {}),
  });
  return { state, log, prepared, original };
}

/** Return a fresh copy of `source`. */
function copiedSource(source: SourceRef | null): SourceRef | null {
  return source === null ? null : new SourceRef(source.runId, source.executionSeq, [...source.path]);
}

/** Return a fresh copy of `terminal`. */
function copiedTerminal(terminal: TraceTerminal): TraceTerminal {
  const failure = terminal.failure === null ? null : retainFailure(terminal.failure);
  return new TraceTerminal(terminal.succeeded, copiedSource(terminal.source), terminal.outputJson, failure);
}

/** Replay `envelopes` over a fresh state of `original`. */
function replayed(original: CompiledFlow, envelopes: readonly RecordEnvelope[]): RunState {
  return replayRecords(new RunState(original), envelopes, { growBuilder: fragmentBuilder });
}

/** A START -> END flow for the learnable trace tests. */
function startEndFlow(): CompiledFlow {
  return graph([], [[START_ID, END_ID]]);
}

test("the trace is applied only after a successful append", () => {
  const { state, log } = learnableStarted(startEndFlow());
  commitPrepared(log, state, prepareNodeStarted(state, START_ID));
  const trace = state.trace!;
  const prepared = prepareOutput(log, state, START_ID, {}, {
    forwardCapture: forwardCapture(state.flow.nodes.get(START_ID)!, {}),
  });
  const prior = traceSnapshot(trace);
  expect(trace.occurrences).toEqual([]);
  const append = log.append;

  log.append = () => {
    throw new Error("append failed");
  };
  expectError(() => commitPrepared(log, state, prepared), Error, "append failed");
  expect(traceSnapshot(trace)).toEqual(prior);

  log.append = append;
  commitPrepared(log, state, prepared);
  expect(trace.occurrences).toHaveLength(1);
  expect(trace.currentWriters.get(START_ID)!.source!.executionSeq).toBe(3);
});

test("the trace rejects a forged memory scope before append and when copied: carrier", () => {
  const node = new FuncNode("code", () => "done");
  node.memoryScope = new MemoryScope(["code"], { memories: ["memory"] });
  const { state, log } = startTraceNode(node, { keys: ["memory"], learnable: true });
  const prepared = prepareOutput(log, state, node.id, "done", { forwardCapture: forwardCapture(node, {}) });
  const delta = prepared._plan.traceDelta!;
  const malformed = {} as MemoryScope;
  const message = "historical memory scope must be a MemoryScope or null";

  expectError(() => changedOccurrence(delta.occurrence!, { memoryScope: malformed }), TypeError, message);
  const occurrence = forged(TraceOccurrence, delta.occurrence!, { memoryScope: malformed });
  expectRefused(state, log, withTrace(prepared, changedDelta(delta, { occurrence })), TypeError, message);
});

test("a terminal trace output must match the prepared outcome before append", () => {
  const { state, log, prepared } = preparedScopedOutcome();
  const delta = prepared._plan.traceDelta!;
  const terminal = delta.terminal!;
  const changed = new TraceTerminal(terminal.succeeded, terminal.source, typedJson("forged"), terminal.failure);

  expectRefused(
    state,
    log,
    withTrace(prepared, changedDelta(delta, { terminal: changed })),
    RangeError,
    "terminal trace disagrees with the prepared terminal outcome",
  );
});

test("an outcome trace source must match the physical accepted start before append", () => {
  const { state, log, prepared } = preparedScopedOutcome("failure");
  const delta = prepared._plan.traceDelta!;
  const occurrence = changedOccurrence(delta.occurrence!, { source: new SourceRef(state.runId!, 0) });

  expectRefused(
    state,
    log,
    withTrace(prepared, changedDelta(delta, { occurrence })),
    RangeError,
    "physical trace occurrence disagrees with the accepted outcome",
  );
});

test.each(
  OUTCOME_KINDS.flatMap((kind) =>
    (["source", "node", "sequence", "record"] as const).map((change) => [kind, change] as const),
  ),
)(
  "an outcome trace occurrence must match the canonical record: %s, %s",
  (kind, change) => {
    const { state, log, prepared } = preparedScopedOutcome(kind);
    const delta = prepared._plan.traceDelta!;
    const occurrence = delta.occurrence!;
    const changes = {
      source: { source: new SourceRef(state.runId!, 0) },
      node: { nodeId: START_ID },
      sequence: { outcomeSeq: occurrence.outcomeSeq + 1 },
      record: { envelope: log.records().at(-1)! },
    }[change];

    expectRefused(
      state,
      log,
      withTrace(prepared, changedDelta(delta, { occurrence: changedOccurrence(occurrence, changes) })),
      RangeError,
      "physical trace occurrence disagrees with the accepted outcome",
    );
  },
);

test.each(
  OUTCOME_KINDS.flatMap((kind) =>
    (["initialized", "started", "resumed"] as const).map((field) => [kind, field] as const),
  ),
)(
  "an outcome trace cannot override dispatch authority: %s, %s",
  (kind, field) => {
    const { state, log, prepared } = preparedScopedOutcome(kind);
    const changes: TraceDeltaFields = {
      initialized: { initialized: state.trace },
      started: { started: ["physical", 0] as const },
      resumed: { resumed: true },
    }[field];

    expectRefused(
      state,
      log,
      withTrace(prepared, changedDelta(prepared._plan.traceDelta!, changes)),
      RangeError,
      "outcome trace cannot change initialization or dispatch authority",
    );
  },
);

test.each([
  ["target", "physical trace writer disagrees with the accepted output target"],
  ["source", "physical trace writer disagrees with the accepted outcome"],
  ["sequence", "physical trace writer disagrees with the accepted outcome"],
  ["value", "physical trace writer disagrees with the accepted outcome"],
  ["missing", "physical trace writer disagrees with the accepted output target"],
] as const)("an outcome trace writer must match the same accepted physical output: %s", (change, message) => {
  // A state with nothing committed yet: the failure's preparation writes nothing.
  const { node, state, log } = preparedScopedOutcome("failure");
  const prepared = prepareOutput(log, state, node.id, "done", { forwardCapture: forwardCapture(node, {}) });
  const delta = prepared._plan.traceDelta!;
  const [target, writer] = delta.writer!;
  const writers: Record<typeof change, readonly [string, SourceEntry] | null> = {
    target: [state.flow.startId, writer],
    source: [target, changedEntry(writer, { source: new SourceRef(state.runId!, 0) })],
    sequence: [target, changedEntry(writer, { outcomeSeq: writer.outcomeSeq + 1 })],
    value: [target, changedEntry(writer, { valueJson: [...state.trace!.sources.values()][0]!.valueJson })],
    missing: null,
  };

  const changed = changedDelta(delta, { writer: writers[change] });
  expectRefused(state, log, withTrace(prepared, changed), RangeError, message);
});

/** The forged-field rows of a terminal trace. */
const TERMINAL_CHANGES = [
  "none",
  "carrier",
  "source_none",
  "source_sequence",
  "source_run",
  "succeeded",
  "disposition",
  "output",
  "failure",
] as const;

test.each(
  (["success", "route", "failure"] as const).flatMap((kind) =>
    TERMINAL_CHANGES.map((change) => [kind, change] as const),
  ),
)("a terminal trace rejects forged fields before append: %s, %s", (kind, change) => {
  const { state, log, prepared } = kind === "route" ? preparedTerminalRoute() : preparedScopedOutcome(kind);
  const delta = prepared._plan.traceDelta!;
  const terminal = delta.terminal!;
  const { succeeded, outputJson } = terminal;
  const failure = new Failure({ code: "node.execute", message: "forged", nodeId: "router" });
  const source = terminal.source!;
  const withSource = (changed: SourceRef | null) => new TraceTerminal(succeeded, changed, outputJson, terminal.failure);
  const terminals: Record<(typeof TERMINAL_CHANGES)[number], () => TraceTerminal | null> = {
    none: () => null,
    carrier: () => ({}) as TraceTerminal,
    source_none: () => withSource(null),
    source_sequence: () => withSource(new SourceRef(state.runId!, 0)),
    source_run: () => withSource(new SourceRef("foreign-run", source.executionSeq)),
    succeeded: () => forged(TraceTerminal, terminal, { succeeded: Number(succeeded) }),
    disposition: () =>
      succeeded
        ? new TraceTerminal(false, source, null, failure)
        : new TraceTerminal(true, source, typedJson("forged"), null),
    output: () =>
      succeeded
        ? new TraceTerminal(true, source, typedJson("forged"), null)
        : forged(TraceTerminal, terminal, { outputJson: typedJson("forged") }),
    failure: () =>
      succeeded ? forged(TraceTerminal, terminal, { failure }) : new TraceTerminal(false, source, null, failure),
  };
  const message =
    change === "none" || change === "carrier"
      ? "terminal outcome requires an exact terminal trace"
      : "terminal trace disagrees with the prepared terminal outcome";

  expectRefused(
    state,
    log,
    withTrace(prepared, changedDelta(delta, { terminal: terminals[change]() })),
    RangeError,
    message,
  );
});

/** Each fabricated terminal field of a nonterminal outcome, and its message. */
const NONTERMINAL_FIELDS = [
  ["terminal", "nonterminal outcome cannot install a terminal trace"],
  ["rootWriter", "terminal trace root writer disagrees with the accepted outcome"],
  ["causalRouteSource", "terminal trace causal source disagrees with the accepted outcome"],
  ["rootStartSource", "terminal trace root START source disagrees with the accepted outcome"],
] as const;

test.each(
  (["success", "route", "growth"] as const).flatMap((kind) =>
    NONTERMINAL_FIELDS.map(([field, message]) => [kind, field, message] as const),
  ),
)("a nonterminal outcome rejects fabricated terminal trace authority: %s, %s", (kind, field, message) => {
  // The success is prepared on a state with nothing committed yet: the failure's preparation writes nothing.
  const scoped = preparedScopedOutcome(kind === "success" ? "failure" : kind);
  const { node, state, log } = scoped;
  const prepared =
    kind === "success"
      ? prepareOutput(log, state, node.id, "done", { forwardCapture: forwardCapture(node, {}) })
      : scoped.prepared;
  const delta = prepared._plan.traceDelta!;
  expect(delta.terminal).toBeNull();
  const source = delta.occurrence!.source;
  const changes: Record<string, TraceDeltaFields> = {
    terminal: {
      terminal: new TraceTerminal(
        false,
        source,
        null,
        new Failure({ code: "node.execute", message: "forged", nodeId: node.id }),
      ),
    },
    rootWriter: { rootWriter: [...state.trace!.currentWriters.values()][0]! },
    causalRouteSource: { causalRouteSource: source },
    rootStartSource: { rootStartSource: source },
  };

  expectRefused(state, log, withTrace(prepared, changedDelta(delta, changes[field]!)), RangeError, message);
});

test.each(["none", "source", "sequence", "value"] as const)(
  "a terminal trace root writer must be the physical accepted output: %s",
  (change) => {
    const { state, log, prepared } = preparedScopedOutcome();
    const delta = prepared._plan.traceDelta!;
    const writer = delta.rootWriter!;
    const writers = {
      none: null,
      source: changedEntry(writer, { source: new SourceRef(state.runId!, 0) }),
      sequence: changedEntry(writer, { outcomeSeq: writer.outcomeSeq + 1 }),
      value: changedEntry(writer, { valueJson: [...state.trace!.currentWriters.values()][0]!.valueJson }),
    };

    expectRefused(
      state,
      log,
      withTrace(prepared, changedDelta(delta, { rootWriter: writers[change] })),
      RangeError,
      "terminal trace root writer disagrees with the accepted outcome",
    );
  },
);

test.each(["none", "sequence", "run"] as const)(
  "a terminal trace root START latch must match the accepted start: %s",
  (change) => {
    const { state, log } = learnableStarted(startEndFlow());
    commitPrepared(log, state, prepareNodeStarted(state, START_ID));
    const prepared = prepareOutput(log, state, START_ID, {}, {
      forwardCapture: forwardCapture(state.flow.nodes.get(START_ID)!, {}),
    });
    const delta = prepared._plan.traceDelta!;
    const source = delta.rootStartSource!;
    const sources = {
      none: null,
      sequence: new SourceRef(state.runId!, 0),
      run: new SourceRef("foreign-run", source.executionSeq),
    };

    expectRefused(
      state,
      log,
      withTrace(prepared, changedDelta(delta, { rootStartSource: sources[change] })),
      RangeError,
      "terminal trace root START source disagrees with the accepted outcome",
    );
    commitPrepared(log, state, prepared);
    expect(state.trace!.rootStartSource).toEqual(source);
  },
);

test.each(
  OUTCOME_KINDS.flatMap((kind) =>
    (["none", "foreign_path", "foreign_memory"] as const).map((change) => [kind, change] as const),
  ),
)(
  "an outcome rejects a valid scope substitution before append: %s, %s",
  (kind, change) => {
    const { node, state, log, prepared } = preparedScopedOutcome(kind);
    const scope = node.memoryScope!;
    const scopes = {
      none: null,
      foreign_path: new MemoryScope(["other"], { memories: scope.memories }),
      foreign_memory: new MemoryScope(scope.staticPath, { memories: ["foreign"] }),
    };
    const delta = prepared._plan.traceDelta!;
    const occurrence = changedOccurrence(delta.occurrence!, { memoryScope: scopes[change] });

    expectRefused(
      state,
      log,
      withTrace(prepared, changedDelta(delta, { occurrence })),
      RangeError,
      "outcome memory scope disagrees with the physical executing node",
    );
  },
);

/** Each selected or omitted scope authority, and its message. */
const SCOPE_AUTHORITY_CHANGES = [
  ["display_node", "outcome memory scope disagrees with the physical executing node"],
  ["delta", "a retained outcome requires its physical trace occurrence"],
  ["occurrence", "a retained outcome requires its physical trace occurrence"],
] as const;

test.each(
  OUTCOME_KINDS.flatMap((kind) =>
    SCOPE_AUTHORITY_CHANGES.map(([change, message]) => [kind, change, message] as const),
  ),
)("outcome scope authority can't be selected or omitted by the trace delta: %s, %s", (kind, change, message) => {
  const { state, log, prepared } = preparedScopedOutcome(kind);
  const delta = prepared._plan.traceDelta!;
  const deltas: Record<typeof change, TraceDelta | null> = {
    display_node: changedDelta(delta, {
      occurrence: changedOccurrence(delta.occurrence!, { nodeId: START_ID, memoryScope: null }),
    }),
    delta: null,
    occurrence: changedDelta(delta, { occurrence: null }),
  };

  expectRefused(state, log, withTrace(prepared, deltas[change]), RangeError, message);
});

test.each(["success", "route", "failure"] as const)(
  "a terminal trace accepts detached authority and replays it: %s",
  (kind) => {
    const { state, log, prepared, original } = kind === "route" ? preparedTerminalRoute() : preparedScopedOutcome(kind);
    const delta = prepared._plan.traceDelta!;
    const writer = delta.rootWriter;
    const detached = changedDelta(delta, {
      terminal: copiedTerminal(delta.terminal!),
      rootWriter: writer === null ? null : changedEntry(writer, { source: copiedSource(writer.source) ?? undefined }),
      rootStartSource: copiedSource(delta.rootStartSource),
      causalRouteSource: copiedSource(delta.causalRouteSource),
    });

    commitPrepared(log, state, withTrace(prepared, detached));

    expect(state.trace!.terminal).toEqual(detached.terminal);
    const replay = replayed(original, log.records());
    expect(replayTraceSnapshot(replay.trace!)).toEqual(replayTraceSnapshot(state.trace!));
    expect(replay.result).toEqual(state.result);
  },
);

test.each([false, true])(
  "a terminal trace uses the accepted route latch until an explicit failure (explicit failure: %p)",
  (explicitFailure) => {
    const { state, log, prepared: route, original } = preparedTerminalRoute({ sibling: true });
    const delta = route._plan.traceDelta!;
    expect(delta.terminal).toBeNull();
    const causal = delta.causalRouteSource!;
    const sibling = new SourceRef(state.runId!, state.trace!.started.get("unrelated")!);
    for (const source of [null, sibling]) {
      expectRefused(
        state,
        log,
        withTrace(route, changedDelta(delta, { causalRouteSource: source })),
        RangeError,
        "terminal trace causal source disagrees with the accepted outcome",
      );
    }
    commitPrepared(log, state, route);
    const carriers = { forwardCapture: forwardCapture(state.flow.nodes.get("unrelated")!, {}) };
    const final = explicitFailure
      ? prepareNodeFailed(
          state,
          new Failure({ code: "node.execute", message: "later failure", nodeId: "unrelated" }),
          carriers,
        )
      : prepareOutput(log, state, "unrelated", "done", carriers);
    const [expectedSource, wrongSource] = explicitFailure ? [sibling, causal] : [causal, sibling];
    const finalDelta = final._plan.traceDelta!;
    const terminal = finalDelta.terminal!;
    const wrong = new TraceTerminal(terminal.succeeded, wrongSource, terminal.outputJson, terminal.failure);
    expectRefused(
      state,
      log,
      withTrace(final, changedDelta(finalDelta, { terminal: wrong })),
      RangeError,
      "terminal trace disagrees with the prepared terminal outcome",
    );

    commitPrepared(log, state, withTrace(final, changedDelta(finalDelta, { terminal: copiedTerminal(terminal) })));

    expect(state.trace!.terminal!.source).toEqual(expectedSource);
    expect(state.trace!.terminal!.failure).toEqual((state.result as RunFailed).failure);
    const records = log.records();
    for (let length = 1; length <= records.length; length++) {
      const replay = replayed(copyGraph(original), records.slice(0, length));
      expect(replay.trace!.terminal).toEqual(length === records.length ? state.trace!.terminal : null);
    }
  },
);

test.each(OUTCOME_KINDS.flatMap((kind) => [true, false].map((scoped) => [kind, scoped] as const)))(
  "an outcome accepts a detached physical scope and replays it: %s, scoped %p",
  (kind, scoped) => {
    const { node, state, log, prepared, original } = preparedScopedOutcome(kind, { scoped });
    const delta = prepared._plan.traceDelta!;
    const scope = node.memoryScope;
    const copied =
      scope === null
        ? null
        : new MemoryScope([...scope.staticPath], {
            memories: [...scope.memories],
            occurrencePath: [...scope.occurrencePath],
          });

    const detached = changedOccurrence(delta.occurrence!, { memoryScope: copied });
    commitPrepared(log, state, withTrace(prepared, changedDelta(delta, { occurrence: detached })));

    const occurrence = state.trace!.occurrences.at(-1)!;
    expect(occurrence.memoryScope).toEqual(node.memoryScope);
    const replay = replayed(original, log.records());
    expect(replay.trace!.occurrence(occurrence.source).memoryScope).toEqual(occurrence.memoryScope);
  },
);

test.each<[string, Value, boolean, (string | number)[] | null]>([
  ["input.x", null, true, ["x"]],
  ["input.absent", null, false, null],
  ["input.x.more", null, true, ["x"]],
  ["input.items.0", [1n], true, ["items"]],
  [`${START_ID}.output`, { x: null, items: [1n] }, true, []],
  ["system.run_id.ignored", RUN_ID, true, null],
  ["system", null, false, null],
  ["input", null, false, null],
  ["input. x", null, false, null],
])("the paired resolver preserves pool namespaces: %s", (path, value, present, sourcePath) => {
  const { state } = learnableStarted(startEndFlow(), { inputs: { x: null, items: [1n] } });
  const names = path.split(".");
  const resolved = state.trace!.resolve(names);

  expect(resolved.value).toEqual(value);
  expect(resolved.present).toBe(present);
  expect(resolved.source ?? null).toEqual(sourcePath === null ? null : new SourceRef(state.runId!, 0, sourcePath));
  const pooled = state.pool.readRoot(names);
  expect([resolved.value, resolved.present]).toEqual([pooled.value, pooled.present]);
});

test("the reducer requires a capture exactly when learning", () => {
  const { state: plain, log: plainLog } = initializedAndStarted(startEndFlow());
  const capture = forwardCapture(plain.flow.nodes.get(START_ID)!, {});
  commitPrepared(plainLog, plain, preparePoolCommit(plain, START_ID, {}));
  expectError(
    () => prepareNodeSucceeded(plain, START_ID, { forwardCapture: capture }),
    RangeError,
    "forward capture requires recorded learning",
  );

  const { state, log } = learnableStarted(startEndFlow());
  commitPrepared(log, state, prepareNodeStarted(state, START_ID));
  expectError(
    () => prepareOutput(log, state, START_ID, {}),
    RangeError,
    "accepted outcome requires forward capture while learning",
  );
});

// The frontier trace checks on a learnable run.

/** A frontier record a learnable run prepares after START's success. */
type FrontierKind = "started" | "resumed" | "step" | "stopped";

/**
 * A learnable `terminalSkipRouteFlow()` after START's success,
 * with `router`'s prepared dispatch or, once `router` is in flight, a prepared resume, step or
 * stop. `original` is a copy of the started flow, for a replay.
 */
function preparedTraceFrontier(kind: FrontierKind): {
  state: RunState;
  log: MemoryRunLog;
  prepared: PreparedTransition;
  original: CompiledFlow;
} {
  const { state, log } = learnableStarted(terminalSkipRouteFlow());
  const original = copyGraph(state.flow);
  commitPrepared(log, state, prepareNodeStarted(state, START_ID));
  commitStartSuccess(state, log, {});
  let prepared = prepareNodeStarted(state, "router");
  if (kind !== "started") {
    commitPrepared(log, state, prepared);
    if (kind === "resumed") prepared = prepareRunResumed(state, { numWorkers: 0 });
    else if (kind === "step") prepared = prepareNodeStep(state, "router", { kind: "test", event: "progress" });
    else prepared = prepareRunStopped(state, "signal", "SIGTERM");
  }
  return { state, log, prepared, original };
}

test("a frontier trace can't inject a causal terminal source before append", () => {
  const { state, log, prepared } = preparedTraceFrontier("started");
  const delta = changedDelta(prepared._plan.traceDelta!, { causalRouteSource: new SourceRef(state.runId!, 0) });

  expectRefused(
    state,
    log,
    withTrace(prepared, delta),
    RangeError,
    "frontier trace cannot change outcome or terminal history",
  );
});

/** The message each frontier kind's record check gives a forged `started` or `resumed`. */
const FRONTIER_RECORD_MESSAGES: Record<FrontierKind, string> = {
  started: "frontier trace start disagrees with its canonical record",
  resumed: "frontier trace resume disagrees with its canonical record",
  step: "frontier trace step must only advance the accepted sequence",
  stopped: "frontier trace stop must only advance the accepted sequence",
};

/** The fields that would change outcome or terminal history, plus the missing delta. */
const HISTORY_FIELDS = [
  "missing",
  "initialized",
  "occurrence",
  "handler",
  "writer",
  "rootStartSource",
  "rootWriter",
  "causalRouteSource",
  "terminal",
] as const;

/** A forged field of a frontier trace delta. */
type FrontierField = (typeof HISTORY_FIELDS)[number] | "started" | "resumed";

/**
 * Each kind's forged fields: a dispatch's history fields (its `started` and `resumed` rows aren't
 * portable), a resume's `started` too, and a step's and a stop's `started` and `resumed` too.
 */
const FRONTIER_ROWS: (readonly [FrontierKind, FrontierField])[] = [
  ...HISTORY_FIELDS.map((field) => ["started", field] as const),
  ...(["started", ...HISTORY_FIELDS] as const).map((field) => ["resumed", field] as const),
  ...(["step", "stopped"] as const).flatMap((kind) =>
    (["started", "resumed", ...HISTORY_FIELDS] as const).map((field) => [kind, field] as const),
  ),
];

test.each(FRONTIER_ROWS)("a frontier trace rejects an inappropriate field before append: %s, %s", (kind, field) => {
  const { state, log, prepared } = preparedTraceFrontier(kind);
  const trace = state.trace!;
  const delta = prepared._plan.traceDelta!;
  const writer = [...trace.currentWriters][0]!;
  const changes: Record<Exclude<FrontierField, "missing">, TraceDeltaFields> = {
    initialized: { initialized: trace },
    started: { started: ["router", delta.sequence] },
    resumed: { resumed: true },
    occurrence: { occurrence: trace.occurrences.at(-1)! },
    handler: {
      handler: () => {
        throw new Error("invoked a forged frontier handler");
      },
    },
    writer: { writer },
    rootStartSource: { rootStartSource: new SourceRef(state.runId!, 0) },
    rootWriter: { rootWriter: writer[1] },
    causalRouteSource: { causalRouteSource: new SourceRef(state.runId!, 0) },
    terminal: { terminal: new TraceTerminal(true, writer[1].source, writer[1].valueJson, null) },
  };
  const message =
    field === "missing"
      ? "frontier trace delta requires its exact accepted sequence"
      : field === "started" || field === "resumed"
        ? FRONTIER_RECORD_MESSAGES[kind]
        : "frontier trace cannot change outcome or terminal history";

  expectRefused(
    state,
    log,
    withTrace(prepared, field === "missing" ? null : changedDelta(delta, changes[field])),
    RangeError,
    message,
  );
});

test.each(["node", "sequence"] as const)("a frontier trace start's identity is exact: %s", (change) => {
  const { state, log, prepared } = preparedTraceFrontier("started");
  const delta = prepared._plan.traceDelta!;
  const started = { node: [START_ID, delta.sequence] as const, sequence: ["router", 0] as const }[change];

  expectRefused(
    state,
    log,
    withTrace(prepared, changedDelta(delta, { started })),
    RangeError,
    "frontier trace start disagrees with its canonical record",
  );
});

test.each(["started", "resumed"] as const)(
  "a frontier trace and terminal source match a live replay without repreparing: %s",
  async (kind) => {
    const { state, log, prepared, original } = preparedTraceFrontier(kind);
    commitPrepared(log, state, withTrace(prepared, changedDelta(prepared._plan.traceDelta!, {})));
    if (kind === "resumed") commitPrepared(log, state, prepareNodeStarted(state, "router"));
    const physical = new SourceRef(state.runId!, state.trace!.started.get("router")!);
    commitPrepared(log, state, prepareNodeRouted(state, "router", "go", {
      forwardCapture: forwardCapture(state.flow.nodes.get("router")!, {}),
    }));

    expect(state.trace!.terminal!.source).toEqual(physical);
    expect(physical.executionSeq).toBeGreaterThan(0);
    const [root, results] = await computeBackward(state.trace!, "Review.");
    expect(results.find((item) => sourceKey(item.source) === sourceKey(physical))!.invoked).toBe(true);
    const replay = replayRecords(new RunState(copyGraph(original)), log.records(), { growBuilder: noGrowth });
    expect(replayTraceSnapshot(replay.trace!)).toEqual(replayTraceSnapshot(state.trace!));
    expect(await computeBackward(replay.trace!, "Review.")).toEqual([root, results]);
  },
);

// The engine rows: captures and trace entries a run through `FlowEngine` accepts.

/** Return a learnable engine for `START -> node -> END`; the other options go to the fixture. */
function learnableSingle(node: Node, options: Parameters<typeof engine>[1] = {}): FlowEngine {
  return engine(graph([node], [
    [START_ID, node.id],
    [node.id, END_ID],
  ]), { learnable: true, ...options });
}

/** Return the accepted occurrences of `built`'s trace by node id. */
function occurrencesById(built: FlowEngine): Map<string, TraceOccurrence> {
  return new Map(built.state.trace!.occurrences.map((item) => [item.nodeId, item]));
}

/** Return the accepted success of `nodeId` in `built`'s log. */
function successEnvelope(built: FlowEngine, nodeId: string): RecordEnvelope {
  return built.log
    .records()
    .find((envelope) => envelope.record instanceof NodeSucceededRecord && envelope.record.nodeId === nodeId)!;
}

/** Return a record route's children by name. */
function routesByName(route: ValueRoute): Map<string, ValueRoute> {
  return new Map(route.names.map((name, index) => [name, route.children[index]!]));
}

test("accepted backward inventories and layout replay without recapturing", async () => {
  const loaded = (
    await loadFlow(`id: captured_layout
name: captured_layout
input:
  topic: !str reefs
nodes:
  work:
    kind: code
    input: {topic: '\${input.topic}'}
    code: return inputs['topic']
    output: str
output: \${work.output}
asserts: ["work.output != null"]
`)
  ).compiled;
  const built = engine(loaded, { learnable: true });
  const original = copyGraph(built.state.flow);
  await built.run();
  const occurrences = new Map(built.state.trace!.occurrences.map((item) => [item.nodeId, item]));
  const start = (occurrences.get(START_ID)!.record as NodeSucceededRecord).forwardCapture!;
  const terminal = (occurrences.get(END_ID)!.record as NodeSucceededRecord).forwardCapture!;
  expect(start.inputNames).toEqual(["topic"]);
  expect(occurrences.get(START_ID)!.boundInputs).toEqual({ topic: "reefs" });
  expect(terminal.inputNames).toEqual(["result"]);
  expect(Object.keys(occurrences.get(END_ID)!.boundInputs!).length).toBeGreaterThan(1);
  expect(occurrences.get(END_ID)!.nodeData).toEqual({ input_output_paths: { result: [] } });

  const forbidden = () => {
    throw new Error("replay executed or recaptured a node");
  };
  const inputNames = Node.prototype.backwardInputNames;
  const initializedRoots: string[] = [];
  const spies = [
    spyOn(CodeNode.prototype, "forward").mockImplementation(forbidden),
    spyOn(Node.prototype, "backwardInputNames").mockImplementation(function (this: Node) {
      expect(this.id).toBe(START_ID);
      initializedRoots.push(this.id);
      return inputNames.call(this);
    }),
    spyOn(Node.prototype, "declaredVariables").mockImplementation(forbidden),
    ...[Node.prototype, StartNode.prototype, EndNode.prototype].map((proto) =>
      spyOn(proto, "captureBackwardContext").mockImplementation(forbidden),
    ),
  ];
  let replay: RunState;
  try {
    replay = replayRecords(new RunState(original), built.log.records(), { growBuilder: fragmentBuilder });
  } finally {
    for (const spy of spies) spy.mockRestore();
  }
  expect(initializedRoots).toEqual([START_ID]);
  expect(replayTraceSnapshot(replay.trace!)).toEqual(replayTraceSnapshot(built.state.trace!));
  const occurrence = replay.trace!.occurrences.find((item) => item.nodeId === END_ID)!;
  const handler = replay.trace!.backwardHandlers.get(sourceKey(occurrence.source))!;
  const end = original.nodes.get(END_ID) as EndNode;
  // The bound inputs and node data are rebuilt from END, so they are read before it changes.
  const inputs = occurrence.boundInputs;
  const nodeData = occurrence.nodeData;
  end.names = ["changed", "afterward"];
  end.params = [];
  const capture = (occurrence.record as NodeSucceededRecord).forwardCapture!;
  const context = new BackwardContext(inputs, {
    inputNames: capture.inputNames,
    variableNames: capture.variableNames,
    output: replay.trace!.sources.get(sourceKey(occurrence.source))!.value,
    nodeData,
  });
  expect((await handler(context, ["Review the answer."])).inputFeedback).toEqual({ result: ["Review the answer."] });
});

test("an append failure does not install worker data", async () => {
  const expected = new Error("append unavailable");
  const node = new FuncNode("writer", () => "actual prompt");
  const built = learnableSingle(node);
  const append = built.log.append.bind(built.log);
  built.log.append = (record) => {
    if (record instanceof NodeSucceededRecord && record.nodeId === "writer") {
      expect(record.forwardCapture).not.toBeNull();
      throw expected;
    }
    return append(record);
  };

  expect(await rejected(built.run())).toBe(expected);
  // The append failure refuses the success; the output's commit before it stays unread.
  expect(built.log.records().at(-2)!.record).toEqual(new NodeStartedRecord("writer"));
  const trace = built.state.trace!;
  expect(trace.occurrences.every((item) => item.nodeId !== "writer")).toBe(true);
  expect(trace.currentWriters.has("writer")).toBe(false);
  expect(built.state.pool.getValue("writer")).toBeNull();
});

test("the trace source is the actual commit-as producer", async () => {
  /** Commits its value to `target`. */
  class Redirect extends TraceNode {
    override forward(): NodeResult {
      return new Output({ value: null }, { commitAs: "target" });
    }
  }
  const producer = new Redirect("physical");
  const consumer = new FuncNode("read", (inputs) => inputs, { reads: { x: "${target.output.value}" } });
  const built = engine(
    graph([producer, new FuncNode("target", () => null), consumer], [
      [START_ID, "physical"],
      ["target", "read"],
      ["read", END_ID],
    ]),
    { learnable: true },
  );

  await built.run();

  const trace = built.state.trace!;
  const occurrences = occurrencesById(built);
  const physical = occurrences.get("physical")!;
  const entry = trace.currentWriters.get("target")!;
  expect(entry.source).toEqual(physical.source);
  expect(entry.source!.executionSeq).toBeLessThan(physical.outcomeSeq);
  expect(trace.lookupSource(entry.source!)).toEqual({ value: null });
  const consumerCapture = (occurrences.get("read")!.record as NodeSucceededRecord).forwardCapture!;
  expect(occurrences.get("read")!.boundInputs).toEqual({ x: null });
  expect(consumerCapture.route.children[0]!.source).toEqual(
    new SourceRef(RUN_ID, physical.source.executionSeq, ["value"]),
  );
  // The handler at the producer's source is the producer's own backward.
  const handler = trace.backwardHandlers.get(sourceKey(physical.source))!;
  const context = trace.backwardContext(physical.source);
  handler(context, ["Review."]);
  expect(producer.calls).toEqual([[context, ["Review."]]]);
});

test.each(["binding", "body", "output", "route", "grow"] as const)(
  "failed binding and rejected outcomes retain canonical failures: %s",
  async (stage) => {
    /** Fails at `stage`. */
    class Failing extends Node {
      readonly kind = NodeKind.CALL;
      override readonly isSpawner = true;
      override readonly growDepthDelta = 1;

      override forward(): NodeResult {
        if (stage === "body") throw new Error("body failed");
        if (stage === "output") return new Output(1n, { commitAs: "absent" });
        if (stage === "route") return new Route("absent");
        return new Grow(new Set([1n]));
      }
    }
    const node = new Failing("work");
    node.params = [new Parameter("x")];
    const wiring = { work: { x: stage === "binding" ? "${input.x:?missing}" : 1n } };
    const built = engine(graph([node], [
      [START_ID, "work"],
      ["work", END_ID],
    ], { wiring }), { learnable: true });

    await built.run();

    const accepted = built.state.trace!.occurrences.at(-1)!;
    expect(accepted.nodeId).toBe("work");
    const record = accepted.record as NodeFailedRecord;
    expect(record.failure).toEqual((built.state.result as RunFailed).failure);
    // A binding that didn't finish rebuilds to none.
    expect(accepted.boundInputs).toEqual(stage === "binding" ? null : { x: 1n });
  },
);

test("a pooled lookup keeps value and producer identity together", async () => {
  // `peer` starts first and commits only after `consumer` did, so `consumer` binds while `peer` is
  // in flight and `late` binds after it committed.
  const consumerDone = Promise.withResolvers<void>();
  const peer = new BlockingNode("peer", consumerDone.promise.then(() => ({ value: 7n, null: null })));
  // `?.` reads null while `peer` has no output yet, and for its absent key.
  const consumer = new FuncNode("consumer", (inputs) => inputs, { reads: { before: "${peer.output?.value | 0}" } });
  const late = new FuncNode("late", (inputs) => inputs, {
    reads: { after: "${peer.output.value | 0}", null: "${peer.output.null}", missing: "${peer.output?.missing}" },
  });
  const built = engine(
    graph([peer, consumer, late], [
      [START_ID, "peer"],
      [START_ID, "consumer"],
      ["peer", "late"],
      ["consumer", END_ID],
      ["late", END_ID],
    ]),
    {
      learnable: true,
      numWorkers: 2,
      onEvent: (event) => {
        if (event instanceof NodeSucceeded && event.nodeId === "consumer") consumerDone.resolve();
      },
    },
  );

  await built.run();

  expect(built.state.result).toBeInstanceOf(RunSucceeded);
  const occurrences = occurrencesById(built);
  const producer = occurrences.get("peer")!;
  const read = occurrences.get("consumer")!;
  expect(producer.source.executionSeq).toBeLessThan(read.source.executionSeq);
  expect(read.outcomeSeq).toBeLessThan(producer.outcomeSeq);
  const early = (read.record as NodeSucceededRecord).forwardCapture!;
  expect(read.boundInputs).toEqual({ before: 0n });
  expect(routesByName(early.route).get("before")!.children[0]!.kind).toBe("literal");
  const after = (occurrences.get("late")!.record as NodeSucceededRecord).forwardCapture!;
  expect(occurrences.get("late")!.boundInputs).toEqual({ after: 7n, null: null, missing: null });
  const routes = routesByName(after.route);
  expect(routes.get("after")!.children[0]!.source).toEqual(
    new SourceRef(RUN_ID, producer.source.executionSeq, ["value"]),
  );
  expect(routes.get("null")!.source).toEqual(new SourceRef(RUN_ID, producer.source.executionSeq, ["null"]));
  expect(routes.get("missing")!.kind).toBe("literal");
});

test("discarded pooled outcomes are not trace occurrences", async () => {
  // Both start in one fill, so `slow` is in flight when `bad` fails the run.
  const slow = new BlockingNode("slow", new Promise(() => {}));
  const built = engine(
    graph([slow, new FailNode("bad", "failed")], [
      [START_ID, "slow"],
      [START_ID, "bad"],
      ["slow", END_ID],
      ["bad", END_ID],
    ]),
    { learnable: true, numWorkers: 2 },
  );

  await built.run();

  expect(slow.released).toBe(true);
  expect(built.state.result).toBeInstanceOf(RunFailed);
  const trace = built.state.trace!;
  expect(trace.started.has("slow")).toBe(true);
  expect(trace.occurrences.map((item) => item.nodeId)).toEqual([START_ID, "bad"]);
  expect(trace.currentWriters.has("slow")).toBe(false);
  expect(trace.backwardHandlers.has(sourceKey(new SourceRef(RUN_ID, trace.started.get("slow")!)))).toBe(false);
  expectError(
    () => prepareRecord(built.state, new BackwardStartedRecord(1, [null])),
    RangeError,
    "feedback requires a recorded list",
  );
  const [root, attributed] = await computeBackward(trace, null);
  for (const record of [
    new BackwardStartedRecord(1, null),
    new BackwardReadyRecord(1, root, attributed, []),
    new BackwardFinishedRecord(1, new BackwardOutcome("succeeded", { feedback: root })),
  ]) {
    expect(commitPrepared(built.log, built.state, prepareRecord(built.state, record))).toEqual([]);
  }
  expect(attributed.length).toBe(2);
  expect(attributed.every((item) => !item.invoked)).toBe(true);
  expect(built.state.backwardState.outcome!.feedback).toEqual(root);
});

test("the trace memory scope is detached and replayed", async () => {
  const scope = new MemoryScope(["code"], { memories: ["memory"] });
  const node = new FuncNode("code", () => ({ items: [1n] }));
  node.memoryScope = scope;
  const built = engine(graph([node], [[START_ID, "code"]], { outputs: [new FlowOutput("value", "${code.output}")] }), {
    learnable: true,
    memories: [new MemorySpec("memory", "memory", {}, [new DeclSite([], ["root"])])],
    memoryStacks: [new MemoryStack(["code"], ["memory"])],
    memorySnapshots: [new MemorySnapshotEnvelope("memory", {})],
  });
  const original = copyGraph(built.state.flow);

  await built.run();

  const occurrence = built.state.trace!.occurrences.find((item) => item.nodeId === "code")!;
  expect(occurrence.memoryScope).toEqual(scope);
  const payload = (parseJson(occurrence.recordJson, "codec") as Record<string, object>).payload!;
  expect(Object.hasOwn(payload, "memory_scope")).toBe(false);
  built.state.flow.nodes.get("code")!.memoryScope = null;
  expect(occurrence.memoryScope).toEqual(scope);
  const replay = replayRecords(new RunState(original), built.log.records(), { growBuilder: fragmentBuilder });
  expect(replay.trace!.occurrence(occurrence.source).memoryScope).toEqual(scope);
});

test("learn false omits backward capture", async () => {
  const built = engine(graph([], [[START_ID, END_ID]]), { learnable: false });

  await built.run();

  expect(built.state.trace).toBeNull();
  expect([...built.state.flow.nodes.values()].every((node) => node.executionScope === null)).toBe(true);
  for (const envelope of built.log.records()) {
    if (envelope.record instanceof NodeSucceededRecord) expect(envelope.record.forwardCapture).toBeNull();
  }
});

test("a retry clears the unfinished start identity and the primed input is replaced", async () => {
  const node = new FuncNode("read", (inputs) => inputs, { reads: { x: "${input.x}" } });
  const built = learnableSingle(node, { inputs: { x: 2n } });
  const { state, log } = built;
  expect(state.trace!.resolve(["input", "x"]).source).toEqual(new SourceRef(RUN_ID, 0, ["x"]));
  commitPrepared(log, state, prepareNodeStarted(state, START_ID));
  const abandoned = state.trace!.started.get(START_ID)!;
  commitPrepared(log, state, prepareRunResumed(state, { numWorkers: 0 }));
  expect(state.trace!.started.size).toBe(0);

  await built.run();

  const trace = state.trace!;
  const start = trace.occurrences.find((item) => item.nodeId === START_ID)!;
  expect(start.source.executionSeq).not.toBe(abandoned);
  expect(trace.backwardHandlers.has(sourceKey(new SourceRef(RUN_ID, abandoned)))).toBe(false);
  expect((start.record as NodeSucceededRecord).forwardCapture!.route.children[0]!.source).toEqual(
    new SourceRef(RUN_ID, 0, ["x"]),
  );
  expect(trace.resolve(["input", "x"]).source).toEqual(new SourceRef(RUN_ID, start.source.executionSeq, ["x"]));
});

test("the capture canonicalizes supported binding objects", async () => {
  const raw = { values: [1n, 2n], file: new FileRef({ uri: "file:///tmp/example.txt" }) };
  const built = learnableSingle(new FuncNode("work", () => "done", { reads: raw }));

  await built.run();

  // The bound inputs, rebuilt from the binding, are canonical.
  expect(occurrencesById(built).get("work")!.boundInputs).toEqual(canonicalSeed(raw)[1] as JsonObject);
});

test.each(OPAQUE_ARGUMENTS)("learning retains opaque calls with arguments: %s", async (args, names, bound, result) => {
  const calls: Value[][] = [];
  const delivered: unknown[] = [];
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
  const node = new FuncNode(
    "work",
    (inputs) => {
      delivered.push(inputs);
      return inputs;
    },
    { reads: { x: "${collect_args(" + args + ")}" } },
  );
  const built = learnableSingle(node);

  await built.run();

  expect(built.state.result).toBeInstanceOf(RunSucceeded);
  const capture = (occurrencesById(built).get("work")!.record as NodeSucceededRecord).forwardCapture!;
  expect(delivered).toEqual([{ x: result }]);
  expect(capture.route.children[0]!.names).toEqual(names);
  expect(calls).toEqual([bound]);
  expect(occurrencesById(built).get("work")!.boundInputs).toEqual({ x: result });
});

test.each<[string, unknown, string]>([
  ["string", "\ud800", "result"],
  ["list", ["kept", "\ud800"], "result at [1]"],
  ["record-value", { kept: 1n, lost: "\ud800" }, "result at lost"],
  ["record-key", { "\ud800": 1n }, "result"],
  ["nested-key", { kept: 1n, lost: { "\ud800": 1n } }, "result at lost"],
])("noncanonical unicode temporaries fail the binding alike with and without learning: %s", async (_, temporary, at) => {
  standIn("bad_text", () => temporary);
  const failures: unknown[] = [];
  for (const learnable of [false, true]) {
    const node = new FuncNode("work", (inputs) => inputs, { reads: { x: "${bad_text() == bad_text()}" } });
    const built = engine(graph([node], [
      [START_ID, "work"],
      ["work", END_ID],
    ]), { learnable });
    await built.run();
    const { code, message } = (built.state.result as RunFailed).failure;
    failures.push([code, message]);
  }
  const expected = [
    "binding.evaluate",
    `expression function 'bad_text' failed: ${at}: text holding a lone surrogate is not a yian value`,
  ];
  expect(failures).toEqual([expected, expected]);
});

test("learning retains noncanonical partial temporary comparisons", async () => {
  const calls: null[] = [];
  standIn("partial_text", () => {
    calls.push(null);
    return { safe: [1n], lost: new FileRef({ uri: "file:///tmp/x" }) };
  });
  const built = learnableSingle(new FuncNode("work", (inputs) => inputs, { reads: { x: "${partial_text() != null}" } }));

  await built.run();

  expect(built.state.result).toBeInstanceOf(RunSucceeded);
  const envelope = successEnvelope(built, "work");
  expect(decodeRecord(encodeRecord(envelope))).toEqual(envelope);
  const capture = (envelope.record as NodeSucceededRecord).forwardCapture!;
  const opaque = capture.route.children[0]!.children[0]!;
  expect(opaque.kind).toBe("opaque");
  expect(opaque.resultType).toBe("record");
  const fields = routesByName(opaque.snapshot!);
  expect(parseJson(fields.get("safe")!.valueJson!, "codec")).toEqual([1n]);
  expect(fields.get("lost")!.resultType).toBe("file");
  const routed = routeFeedback(capture.route, "Explain.", () => null);
  expect(routed.contributions).toEqual([]);
  expect(routed.stopped).toHaveLength(1);
  expect(routed.stopped[0]!.context.at(-1)!.result).toBeNull();
  expect(calls).toEqual([null]);
  expect(occurrencesById(built).get("work")!.boundInputs).toEqual({ x: true });
});

test("a discarded agent worker keeps its render capture out of the accepted trace", async () => {
  const config = { provider: "openai", model: "gpt-5.5" };
  const entered = Promise.withResolvers<void>();
  const prompts: string[] = [];
  const blocking: ChatModel = {
    ...config,
    stream: (request, options) => {
      prompts.push(request.messages.at(-1)!.text());
      entered.resolve();
      return new EventStream(
        (async function* (): AsyncGenerator<Event> {
          await aborted(options!.signal!);
        })(),
      );
    },
  };

  /** Fails once the agent's model call is in flight. */
  class FailAfterEntry extends Node {
    readonly kind = NodeKind.CODE;

    override async forward(): Promise<NodeResult> {
      await entered.promise;
      throw new Error("sibling failed");
    }
  }

  const slow = new AgentNode("slow", { prompt: "captured but discarded", llmConfig: config });
  const built = engine(
    graph([slow, new FailAfterEntry("bad")], [
      [START_ID, "slow"],
      [START_ID, "bad"],
      ["slow", END_ID],
      ["bad", END_ID],
    ]),
    {
      learnable: true,
      numWorkers: 2,
      effectiveLlmByStaticPath: { slow: config },
      models: { env: NO_ENV, createModel: () => blocking },
    },
  );

  await built.run();

  expect(built.state.result).toBeInstanceOf(RunFailed);
  expect(prompts).toEqual(["captured but discarded"]);
  const trace = built.state.trace!;
  expect(trace.started.has("slow")).toBe(true);
  expect(trace.occurrences.map((item) => item.nodeId)).toEqual([START_ID, "bad"]);
  expect(trace.currentWriters.has("slow")).toBe(false);
});

// The growth rows: captures and trace entries through CALL, MAP and LOOP growth.

/**
 * A `START -> spawn -> END` flow whose CALL `spawn` reads `sources` and calls a START -> END child
 * with one parameter per source; `extraNodes` hang off START, after
 * `spawn` or, with `extraFirst`, before it.
 */
function seedCallFlow(
  sources: Readonly<Record<string, string>> = { x: "${input.seed}" },
  extraNodes: readonly Node[] = [],
  options: { extraFirst?: boolean } = {},
): CompiledFlow {
  const parameters = Object.keys(sources).map((name) => new Parameter(name));
  const child = CompiledFlow.fromParts(
    new Map<string, Node>([
      [START_ID, new StartNode(START_ID, { params: parameters })],
      [END_ID, EndNode.value(END_ID, { outputName: null })],
    ]),
    [new Edge("order", START_ID, END_ID, { ordering: true })],
    {
      wiring: new Map([
        [START_ID, new Map(Object.keys(sources).map((name) => [name, `\${input.${name}}`]))],
        [END_ID, new Map()],
      ]),
    },
  );
  const node = new CallNode("spawn", { flowId: "child", child });
  node.params = parameters;
  const entries: [string, string][] = extraNodes.map((item) => [START_ID, item.id]);
  entries.splice(options.extraFirst ? entries.length : 0, 0, [START_ID, "spawn"]);
  return graph([node, ...extraNodes], [...entries, ["spawn", END_ID]], { wiring: { spawn: sources } });
}

// A child body that decrements the carried `n`.
const COUNTDOWN = `code: 'return {n: inputs["n"] - 1}'`;

/**
 * Load a flow whose `outer` CALL (or MAP over `[n, n]`) calls a LOOP of two iterations, each body
 * a CALL of a countdown leaf.
 */
async function nestedGrowthFlow(kind: "call" | "map"): Promise<CompiledFlow> {
  const over = kind === "map" ? "    over: '${[input.n, input.n]}'\n" : "";
  const seed = kind === "map" ? "${item}" : "${input.n}";
  const text = `
id: nested-seeds
name: nested-seeds
input:
  n: !int
defs:
  leaf:
    input:
      n: !int
    nodes:
      work:
        kind: code
        input: {n: '\${input.n}'}
        ${COUNTDOWN}
        output: {n: int}
    output: \${work.output}
  body:
    input:
      n: !int
    nodes:
      invoke:
        kind: call
        call: leaf
        input: {n: '\${input.n}'}
    output: \${invoke.output}
  repeated:
    input:
      n: !int
    nodes:
      repeat:
        kind: loop
        call: body
        times: 2
        input: {n: '\${input.n}'}
    output: \${repeat.output}
nodes:
  outer:
    kind: ${kind}
    call: repeated
${over}    input: {n: '${seed}'}
output: \${outer.output}
`;
  return (await loadFlow(text)).compiled;
}

/** Return a learnable engine over `flow` with real growth; the other options go to the fixture. */
function growing(flow: CompiledFlow, options: Parameters<typeof engine>[1] = {}): FlowEngine {
  return engine(flow, { learnable: true, growBuilder: buildGrow, ...options });
}

/** Return `fragment` with its growth recipes replaced. */
function withRecipes(
  fragment: Flow,
  recipes: { inputConnections?: readonly SeedConnection[]; childBoundaries?: readonly ChildFlowBoundary[] },
): Flow {
  return new Flow(fragment.nodes, fragment.edges, {
    wiring: fragment.wiring,
    startId: fragment.startId,
    endId: fragment.endId,
    inputConnections: recipes.inputConnections ?? fragment.inputConnections,
    childBoundaries: recipes.childBoundaries ?? fragment.childBoundaries,
  });
}

/** Return the capture of an accepted occurrence's record. */
function captureOf(occurrence: TraceOccurrence): ForwardCapture {
  return occurrence.capture;
}

/**
 * Return the `[source, feedback]` contributions of routing `feedback` through `occurrence`'s joined
 * input route over `trace`.
 */
function contributions(trace: RetainedTrace, occurrence: TraceOccurrence, feedback: FeedbackValue) {
  const routed = routeFeedback(occurrence.inputRoute, feedback, (source) => trace.lookupSource(source));
  return routed.contributions.map((item) => [item.source, item.feedback]);
}

test.each(["call", "map"] as const)(
  "nested growth preserves authored paths and local driver zero: %s",
  async (kind) => {
    // Every node the run has held, by id, so retired drivers can still be read.
    const seen = new Map<string, Node>();
    const built = growing(await nestedGrowthFlow(kind), {
      inputs: { n: 9n },
      onEvent: () => {
        for (const [id, node] of built.state.flow.nodes) seen.set(id, node);
      },
    });

    await built.run();

    const trace = built.state.trace!;
    const occurrences = occurrencesById(built);
    const loopPath = ["outer", "repeat"];
    const invokePath = [...loopPath, "invoke"];
    for (let item = 0; item < (kind === "map" ? 2 : 1); item++) {
      const prefix = kind === "map" ? `outer#${item}` : "outer";
      const outer =
        kind === "map"
          ? new ExecutionPosition("map_item", ["outer"], item)
          : new ExecutionPosition("call", ["outer"]);
      for (let iteration = 0; iteration < 3; iteration++) {
        const driver = occurrences.get(`${prefix}/repeat` + (iteration ? `~${iteration}` : ""))!;
        expect(captureOf(driver).scope!.staticPath).toEqual(loopPath);
        expect(captureOf(driver).scope!.occurrencePath).toEqual([
          outer,
          new ExecutionPosition("loop_driver", loopPath, iteration),
        ]);
      }
      for (let iteration = 0; iteration < 2; iteration++) {
        const bodyStart = occurrences.get(`${prefix}/repeat#${iteration}/${START_ID}`)!;
        const callId = `${prefix}/repeat#${iteration}/invoke`;
        const leafStart = occurrences.get(`${callId}/${START_ID}`)!;
        const position = [outer, new ExecutionPosition("loop_body", loopPath, iteration)];
        expect(captureOf(bodyStart).scope!.occurrencePath).toEqual(position);
        expect(captureOf(leafStart).scope!.occurrencePath).toEqual([
          ...position,
          new ExecutionPosition("call", invokePath),
        ]);
        expect(captureOf(leafStart).scope!.flowPath).toEqual(invokePath);
        expect(leafStart.inputRoute.children[0]!.source).toEqual(
          new SourceRef(bodyStart.source.runId, bodyStart.source.executionSeq, ["n"]),
        );
      }
      const driverNode = seen.get(`${prefix}/repeat`) as LoopNode;
      expect((driverNode.child as CompiledFlow).nodes.get("invoke")!.executionScope!.occurrencePath).toEqual([]);
    }
    expect(trace.boundaries.every((boundary) => boundary.startSource !== null && boundary.endSource !== null)).toBe(
      true,
    );
  },
);

test.each<[string, (string | number)[]]>([
  ["absent", ["absent"]],
  ["item", [9, "x"]],
])("a missing seed selector is an accepted growth failure: %s", async (_, path) => {
  const invalidSelector: GrowBuilder = (spawner, seed) => {
    const fragment = buildGrow(spawner, seed);
    return withRecipes(fragment, { inputConnections: [new SeedConnection(fragment.startId, "x", path)] });
  };
  const built = growing(seedCallFlow(), { inputs: { seed: 7n }, growBuilder: invalidSelector });

  await built.run();

  const record = built.state.trace!.occurrences.at(-1)!.record;
  expect(record).toBeInstanceOf(NodeFailedRecord);
  const failure = (record as NodeFailedRecord).failure;
  expect(failure.code).toBe("contract.growth");
  expect(failure).toEqual((built.state.result as RunFailed).failure);
  expect(built.state.trace!.occurrences.at(-1)!.boundInputs).toEqual({ x: 7n });
  expect(built.state.trace!.pendingConnections.size).toBe(0);
  expect(built.state.trace!.boundaries).toEqual([]);
});

// The start of each refusal; the digest row goes on to name both digests.
const TAMPERING_MESSAGES = {
  "selector-digest": "NodeExpandedRecord topology digest mismatch: recorded ",
  "missing-field": 'seed connection ["absent"] was not captured',
  "missing-item": 'seed connection [9,"x"] was not captured',
  "boundary-role": "child boundary role disagrees with its execution scope",
};

/** Return what `fn` throws, or fail when it returns. */
function thrown(fn: () => unknown): unknown {
  try {
    fn();
  } catch (error) {
    return error;
  }
  throw new Error("expected a throw");
}

test.each(["selector-digest", "missing-field", "missing-item", "boundary-role"] as const)(
  "growth recipe tampering fails replay before apply: %s",
  async (change) => {
    const built = growing(seedCallFlow(), { inputs: { seed: 7n } });
    const original = copyGraph(built.state.flow);
    await built.run();
    const envelopes = built.log.records();
    const at = envelopes.findIndex((envelope) => envelope.record instanceof NodeExpandedRecord);
    let record = envelopes[at]!.record as NodeExpandedRecord;

    const changedBuilder: GrowBuilder = (spawner, seed) => {
      const fragment = buildGrow(spawner, seed);
      const path = change === "missing-item" ? [9, "x"] : ["absent"];
      return withRecipes(fragment, {
        inputConnections: [new SeedConnection(fragment.startId, "x", path)],
        childBoundaries:
          change === "boundary-role" ? [new ChildFlowBoundary(fragment.endId, fragment.startId)] : undefined,
      });
    };

    const replay = replayRecords(new RunState(copyGraph(original)), envelopes.slice(0, at), {
      growBuilder: buildGrow,
    });
    if (change === "missing-field" || change === "missing-item") {
      const changed = changedBuilder(replay.flow.nodes.get("spawn")!, { x: 7n });
      record = new NodeExpandedRecord(record.spawnerId, record.pruneIds, expansionDigest(changed), {
        forwardCapture: record.forwardCapture,
      });
    }
    const before = [traceSnapshot(replay.trace!), [...replay.flow.nodes.keys()], replay.nextSequence];
    const refused = thrown(() => prepareRecord(replay, record, { growBuilder: changedBuilder }));
    expect(refused).toBeInstanceOf(RangeError);
    expect((refused as Error).message).toStartWith(TAMPERING_MESSAGES[change]);
    expect([traceSnapshot(replay.trace!), [...replay.flow.nodes.keys()], replay.nextSequence]).toEqual(before);
  },
);

test.each<[string, boolean]>([
  ["${peer.output | null}", false],
  ["${peer.output}", false],
  ["${peer.output}", true],
])("reference-looking seed strings reach the child as text: %s, early peer %s", async (seed, earlyPeer) => {
  // A seed is resolved data: the child START receives the exact text, its local route is a
  // literal, and no feedback reaches `peer`, whichever order `peer` runs in.
  const peer = new FuncNode("peer", () => 11n);
  const built = growing(seedCallFlow(undefined, [peer], { extraFirst: earlyPeer }), { inputs: { seed } });

  await built.run();

  expect(built.state.result).toBeInstanceOf(RunSucceeded);
  const trace = built.state.trace!;
  const occurrences = occurrencesById(built);
  const parent = occurrences.get("spawn")!;
  const child = occurrences.get(`spawn/${START_ID}`)!;
  expect(parent.boundInputs).toEqual({ x: seed });
  const capture = captureOf(child);
  expect(child.boundInputs).toEqual({ x: seed });
  expect([...capture.evaluatedKeys]).toEqual([]);
  expect(capture.route.children[0]!.kind).toBe("literal");
  // The joined route reaches the run input the seed came from, not `peer`.
  const root = occurrences.get(START_ID)!;
  expect(child.inputRoute.children[0]!.source).toEqual(
    new SourceRef(root.source.runId, root.source.executionSeq, ["seed"]),
  );
  const peerSource = occurrences.get("peer")!.source;
  expect(contributions(trace, child, { x: "Change." }).map(([source]) => source)).not.toContainEqual(peerSource);
});

test("a failed child binding joins the captured fields", async () => {
  const flow = seedCallFlow({ safe: "${input.safe}", x: "${input.seed}" });
  const child = (flow.nodes.get("spawn") as CallNode).child as CompiledFlow;
  child.nodes.get(START_ID)!.params = [
    new Parameter("safe"),
    new Parameter("x", { contract: new ValueContract(Type.scalar(ValueKind.INTEGER)), required: true }),
  ];
  const built = growing(flow, { inputs: { safe: 7n, seed: "not a number" } });

  await built.run();

  const trace = built.state.trace!;
  const occurrence = trace.occurrences.at(-1)!;
  expect(occurrence.nodeId).toBe(`spawn/${START_ID}`);
  const record = occurrence.record as NodeFailedRecord;
  expect(record).toBeInstanceOf(NodeFailedRecord);
  expect(record.failure).toEqual((built.state.result as RunFailed).failure);
  expect(record.failure.code).toBe("contract.input");
  expect(occurrence.boundInputs).toBeNull();
  expect(occurrence.inputRoute.names).toEqual(["safe", "x"]);
  expect(record.forwardCapture!.route.names).toEqual(["safe", "x"]);
  const root = trace.occurrences.find((item) => item.nodeId === START_ID)!;
  expect(occurrence.inputRoute.children[0]!.source).toEqual(
    new SourceRef(root.source.runId, root.source.executionSeq, ["safe"]),
  );
  expect(trace.boundaries).toHaveLength(1);
  const [boundary] = trace.boundaries;
  expect(boundary!.startSource).toEqual(occurrence.source);
  expect(boundary!.endSource).toBeNull();
  expect(trace.pendingConnections.size).toBe(0);
});

/** Evaluate started `nodeId` as the engine does for a learnable run. */
function evaluateStarted(state: RunState, nodeId: string): Promise<EvaluatedOutcome> {
  const trace = state.trace!;
  return evalNode(state.flow.nodes.get(nodeId)!, state.flow, state.pool, {
    wiringOverride: state.wiringOverrides.get(nodeId) ?? null,
    captureBindings: true,
    sourceResolver: (names) => trace.resolve(names),
  });
}

/** The joined forward capture of an evaluation. */
function joinedCapture(evaluated: EvaluatedOutcome): ForwardCapture {
  return captureBindings(evaluated.bindings!, evaluated.backwardCapture!);
}

test("seed connections wait for the append and survive an unfinished start's retry", async () => {
  const built = growing(seedCallFlow(), { inputs: { seed: 7n } });
  const { state, log } = built;
  commitPrepared(log, state, prepareNodeStarted(state, START_ID));
  const start = await evaluateStarted(state, START_ID);
  commitPrepared(
    log,
    state,
    prepareOutput(log, state, START_ID, (start.outcome as Output).value, { forwardCapture: joinedCapture(start) }),
  );
  commitPrepared(log, state, prepareNodeStarted(state, "spawn"));
  const evaluated = await evaluateStarted(state, "spawn");
  const prepared = prepareGrowth(log, state, "spawn", evaluated.outcome as Grow, built.growBuilder, {
    forwardCapture: joinedCapture(evaluated),
  });
  const before = [traceSnapshot(state.trace!), [...state.flow.nodes.keys()], state.nextSequence];
  const append = log.append;
  log.append = () => {
    throw new Error("append failed");
  };
  expect(() => commitPrepared(log, state, prepared)).toThrow("append failed");
  expect([traceSnapshot(state.trace!), [...state.flow.nodes.keys()], state.nextSequence]).toEqual(before);
  log.append = append;
  commitPrepared(log, state, prepared);
  const target = `spawn/${START_ID}`;
  const pending = state.trace!.pendingConnections.get(target)!;
  expect(pending).toBeDefined();
  commitPrepared(log, state, prepareNodeStarted(state, target));
  const abandoned = state.trace!.started.get(target)!;
  commitPrepared(log, state, prepareRunResumed(state, { numWorkers: 0 }));
  expect(state.trace!.pendingConnections.get(target)).toBe(pending);

  await built.run();

  const child = state.trace!.occurrences.find((item) => item.nodeId === target)!;
  expect(child.source.executionSeq).not.toBe(abandoned);
  expect(child.inputRoute.children[0]!.source).not.toBeNull();
  expect(state.trace!.pendingConnections.size).toBe(0);
});

test.each([false, true])(
  "pruned seed connections do not leak into reused node ids: skip first %s",
  async (skipFirst) => {
    const sharedStart = `shared/${START_ID}`;
    const sharedEnd = `shared/${END_ID}`;
    const child = (seedCallFlow().nodes.get("spawn") as CallNode).child as CompiledFlow;
    const emptyChild = (seedCallFlow({}).nodes.get("spawn") as CallNode).child as CompiledFlow;

    /** Grows an empty child and prunes the shared child's boundary. */
    class Prune extends Node {
      readonly kind = NodeKind.CODE;
      override readonly isSpawner = true;
      // A live growth is checked against its rebuild from `from`, so the spawner double is derived.
      override readonly derived = true;

      override forward(): NodeResult {
        return new Grow({}, [sharedStart, sharedEnd]);
      }
    }

    const first = new CallNode("first", { flowId: "child", child: copyGraph(child) });
    const again = new CallNode("again", { flowId: "child", child: copyGraph(child) });
    first.params = again.params = [new Parameter("x")];

    const builder: GrowBuilder = (spawner, seed) => {
      if (spawner.id === "prune") return callSubgraph(emptyChild, "prune", {});
      if (spawner.id === "first" && skipFirst) {
        const finish = EndNode.value("finish", { outputName: null });
        finish.commitAs = "first";
        return new Flow(
          new Map<string, Node>([
            ["entry", new StartNode("entry", { params: [] })],
            ["choice", new CaseNode("choice", [])],
            [sharedStart, new StartNode(sharedStart, { params: [new Parameter("x")] })],
            [sharedEnd, EndNode.value(sharedEnd, { outputName: null })],
            ["finish", finish],
          ]),
          [
            new Edge("enter", "entry", "choice", { ordering: true }),
            new Edge("chosen", "choice", "finish", { sourceHandle: ELSE_HANDLE }),
            new Edge("skipped", "choice", sharedStart, { sourceHandle: "other" }),
            new Edge("child", sharedStart, sharedEnd, { ordering: true }),
          ],
          {
            wiring: new Map<string, Map<string, unknown>>([
              ["entry", new Map()],
              ["choice", new Map()],
              [sharedStart, new Map([["x", (seed as Record<string, unknown>)["x"]]])],
              [sharedEnd, new Map()],
              ["finish", new Map()],
            ]),
            startId: "entry",
            endId: "finish",
            inputConnections: [new SeedConnection(sharedStart, "x", ["x"])],
            childBoundaries: [new ChildFlowBoundary(sharedStart, sharedEnd)],
          },
        );
      }
      const fragment = cloneChild((spawner as CallNode).child as Flow, "shared", seed as Record<string, unknown>);
      fragment.nodes.get(fragment.endId)!.commitAs = spawner.id;
      return fragment;
    };

    const seen = new Map<string, Map<string, SeedRoutes>>();
    const flow = graph([first, new Prune("prune"), again], [
      [START_ID, "first"],
      ["first", "prune"],
      ["prune", "again"],
      ["again", END_ID],
    ], { wiring: { first: { x: "${input.first}" }, prune: {}, again: { x: "${input.second}" } } });
    const built = growing(flow, {
      inputs: { first: 7n, second: 11n },
      growBuilder: builder,
      onEvent: (event) => {
        if (event instanceof NodeExpanded) seen.set(event.nodeId, new Map(built.state.trace!.pendingConnections));
      },
    });

    await built.run();

    expect(built.state.result).toBeInstanceOf(RunSucceeded);
    const trace = built.state.trace!;
    expect(seen.get("first")!.has(sharedStart)).toBe(true);
    expect(seen.get("prune")!.has(sharedStart)).toBe(false);
    expect(seen.get("again")!.get(sharedStart)!.owner).not.toEqual(seen.get("first")!.get(sharedStart)!.owner);
    const reused = trace.occurrences.filter((item) => item.nodeId === sharedStart);
    expect(reused).toHaveLength(skipFirst ? 1 : 2);
    const root = trace.occurrences.find((item) => item.nodeId === START_ID)!;
    expect(reused.at(-1)!.inputRoute.children[0]!.source).toEqual(
      new SourceRef(root.source.runId, root.source.executionSeq, ["second"]),
    );
    const boundaries = trace.boundaries.filter((item) => item.startId === sharedStart);
    expect(boundaries).toHaveLength(2);
    expect(boundaries[0]!.owner).not.toEqual(boundaries[1]!.owner);
    expect(boundaries[0]!.startSource).toEqual(skipFirst ? null : reused[0]!.source);
    expect(boundaries[1]!.startSource).toEqual(reused.at(-1)!.source);
    expect(trace.pendingConnections.size).toBe(0);
  },
);

test("CALL seed routes survive materialized values", async () => {
  const loaded = (
    await loadFlow(`
id: seed-call
name: seed-call
input:
  n: !int
defs:
  child:
    input:
      n: !int
    nodes:
      work:
        kind: code
        input: {n: '\${input.n}'}
        ${COUNTDOWN}
        output: {n: int}
    output: \${work.output}
nodes:
  spawn:
    kind: call
    call: child
    input: {n: '\${input.n}'}
output: \${spawn.output}
`)
  ).compiled;
  const built = growing(loaded, { inputs: { n: 9n } });

  await built.run();

  const trace = built.state.trace!;
  const occurrences = occurrencesById(built);
  const child = occurrences.get(`spawn/${START_ID}`)!;
  const rawCapture = captureOf(child);
  expect(child.boundInputs).toEqual({ n: 9n });
  expect(rawCapture.route.children[0]!.kind).toBe("literal");
  expect(contributions(trace, child, { n: "Use another seed." })).toEqual([
    [occurrences.get(START_ID)!.source, { n: "Use another seed." }],
  ]);
  expect(rawCapture.scope!.occurrencePath).toEqual([new ExecutionPosition("call", ["spawn"])]);
  expect(trace.boundaries).toHaveLength(1);
  const [boundary] = trace.boundaries;
  expect(boundary!.owner).toEqual(occurrences.get("spawn")!.source);
  expect([boundary!.startId, boundary!.endId]).toEqual([`spawn/${START_ID}`, `spawn/${END_ID}`]);
  expect([boundary!.startSource, boundary!.endSource]).toEqual([
    child.source,
    occurrences.get(`spawn/${END_ID}`)!.source,
  ]);
  expect(trace.pendingConnections.size).toBe(0);
});

test.each(["call", "map"] as const)("a null materialized seed keeps its selected source: %s", async (kind) => {
  let flow = seedCallFlow();
  if (kind === "map") {
    const node = new MapNode("spawn", { flowId: "child", child: (flow.nodes.get("spawn") as CallNode).child });
    node.params = [new Parameter("x")];
    flow = graph([node], [
      [START_ID, "spawn"],
      ["spawn", END_ID],
    ], { wiring: { spawn: { over: "${[input.seed]}", x: "${item}" } } });
  }
  const built = growing(flow, { inputs: { seed: null } });

  await built.run();

  const trace = built.state.trace!;
  const root = trace.occurrences.find((item) => item.nodeId === START_ID)!;
  const childId = kind === "map" ? `spawn#0/${START_ID}` : `spawn/${START_ID}`;
  const child = trace.occurrences.find((item) => item.nodeId === childId)!;
  expect(child.boundInputs).toEqual({ x: null });
  expect(captureOf(child).route.children[0]!.kind).toBe("literal");
  expect(contributions(trace, child, { x: "Fill this value." })).toEqual([
    [root.source, { seed: "Fill this value." }],
  ]);
});

test.each<[string, Record<string, unknown>, FeedbackValue[]]>([
  ["${input.left}", { left: [7n, 7n] }, [{ left: ["Change.", null] }, { left: [null, "Change."] }]],
  [
    "${input.left + input.right + input.right}",
    { left: [7n, 7n], right: [7n] },
    [{ left: ["Change.", null] }, { left: [null, "Change."] }, { right: ["Change."] }, { right: ["Change."] }],
  ],
  ["${input.left}", { left: [] }, []],
])("MAP seed routes preserve equal item positions: %s", async (over, inputs, expected) => {
  const node = new MapNode("spawn", { flowId: "child", child: childFlow() });
  node.params = [new Parameter("x")];
  const flow = graph([node], [
    [START_ID, "spawn"],
    ["spawn", END_ID],
  ], { wiring: { spawn: { over, x: "${item}" } } });
  const built = growing(flow, { inputs });

  await built.run();

  const trace = built.state.trace!;
  const occurrences = occurrencesById(built);
  for (const [index, feedback] of expected.entries()) {
    const child = occurrences.get(`spawn#${index}/${START_ID}`)!;
    expect(child.boundInputs).toEqual({ x: 7n });
    expect(captureOf(child).route.children[0]!.kind).toBe("literal");
    expect(contributions(trace, child, { x: "Change." })).toEqual([[occurrences.get(START_ID)!.source, feedback]]);
    expect(captureOf(child).scope!.occurrencePath).toEqual([new ExecutionPosition("map_item", ["spawn"], index)]);
  }
  expect(trace.boundaries).toHaveLength(expected.length);
  for (const nodeId of [`spawn/${START_ID}`, `spawn/${END_ID}`]) {
    expect(captureOf(occurrences.get(nodeId)!).scope).toEqual(new ExecutionScope(["spawn"]));
  }
  expect(trace.pendingConnections.size).toBe(0);
});

test.each([0, 3])("LOOP carried routes survive live pruning: times %d", async (times) => {
  const loopPolicy = times ? `times: ${times}` : "while: '${n < 0}'\n    max: 3";
  const loaded = (
    await loadFlow(`
id: seed-loop
name: seed-loop
input:
  n: !int
defs:
  child:
    input:
      n: !int
    nodes:
      work:
        kind: code
        input: {n: '\${input.n}'}
        ${COUNTDOWN}
        output: {n: int}
    output: \${work.output}
nodes:
  spawn:
    kind: loop
    call: child
    ${loopPolicy}
    input: {n: '\${input.n}'}
output: \${spawn.output}
`)
  ).compiled;
  const built = growing(loaded, { inputs: { n: 9n } });

  await built.run();

  const trace = built.state.trace!;
  const occurrences = occurrencesById(built);
  const drivers = [
    occurrences.get("spawn")!,
    ...Array.from({ length: times }, (_, i) => occurrences.get(`spawn~${i + 1}`)!),
  ];
  expect(new Set(drivers.map((item) => sourceKey(item.source))).size).toBe(times + 1);
  for (const [index, driver] of drivers.entries()) {
    expect(captureOf(driver).scope!.occurrencePath).toEqual([new ExecutionPosition("loop_driver", ["spawn"], index)]);
  }
  expect(trace.currentWriters.get("spawn")!.source).toEqual(drivers.at(-1)!.source);
  for (let index = 0; index < times; index++) {
    const child = occurrences.get(`spawn#${index}/${START_ID}`)!;
    const previous = occurrences.get(index === 0 ? START_ID : `spawn#${index - 1}/${END_ID}`)!;
    expect(child.boundInputs).toEqual({ n: BigInt(9 - index) });
    expect(contributions(trace, child, { n: "Change." })).toEqual([[previous.source, { n: "Change." }]]);
    expect(captureOf(child).scope!.occurrencePath).toEqual([new ExecutionPosition("loop_body", ["spawn"], index)]);
    const boundary = trace.boundaries[index]!;
    expect(boundary.owner).toEqual(drivers[index]!.source);
    expect(boundary.startSource).toEqual(child.source);
    expect(boundary.endSource).toEqual(occurrences.get(`spawn#${index}/${END_ID}`)!.source);
    if (index < times - 1) {
      expect(built.state.flow.nodes.has(child.nodeId)).toBe(false);
      expect(trace.sources.has(sourceKey(child.source))).toBe(true);
    }
  }
  expect(trace.pendingConnections.size).toBe(0);
});

/** Load the one-spawner flow whose CALL, MAP or LOOP `spawn` runs a countdown child. */
async function growthFlow(kind: "call" | "map" | "loop"): Promise<CompiledFlow> {
  const extra = { call: "", map: "    over: [1, 1]\n", loop: "    times: 3\n" }[kind];
  const binding = kind === "map" ? "'${item}'" : "1";
  const text = `
id: trace-growth
name: trace-growth
defs:
  child:
    input:
      n: !int
    nodes:
      work:
        kind: code
        input: {n: '\${input.n}'}
        ${COUNTDOWN}
        output: {n: int}
    output: \${work.output}
nodes:
  spawn:
    kind: ${kind}
    call: child
${extra}    input: {n: ${binding}}
output: \${spawn.output}
`;
  return (await loadFlow(text)).compiled;
}

test.each(["output", "route", "failure", "call", "map", "loop", "nested-call", "nested-map"] as const)(
  "trace replay matches live accepted prefixes: %s",
  async (kind) => {
    let replaying = false;
    const work = (): unknown => {
      if (replaying) throw new Error("replay evaluated a node body");
      if (kind === "failure") throw new Error("failed body");
      return { x: 1n };
    };
    let flow: CompiledFlow;
    if (kind === "nested-call" || kind === "nested-map") {
      flow = await nestedGrowthFlow(kind === "nested-call" ? "call" : "map");
    } else if (kind === "call" || kind === "map" || kind === "loop") {
      flow = await growthFlow(kind);
    } else if (kind === "route") {
      flow = graph([new BranchNode("choice", ELSE_HANDLE)], [
        [START_ID, "choice"],
        ["choice", END_ID, ELSE_HANDLE],
      ]);
    } else {
      flow = graph([new FuncNode("work", work)], [
        [START_ID, "work"],
        ["work", END_ID],
      ]);
    }
    // The trace after each accepted record, keyed by the log length it follows.
    const snapshots = new Map<number, Record<string, unknown>>();
    const observe = (): void => {
      snapshots.set(built.state.nextSequence, replayTraceSnapshot(built.state.trace!));
    };
    const built = growing(flow, { inputs: kind.startsWith("nested-") ? { n: 9n } : {}, onEvent: observe });
    const original = copyGraph(built.state.flow);
    observe();

    await built.run();

    expect(built.state.result).toBeInstanceOf(kind === "failure" ? RunFailed : RunSucceeded);
    const trace = built.state.trace!;
    if (kind === "loop") {
      const retired = trace.occurrences.filter((item) => !built.state.flow.nodes.has(item.nodeId));
      expect(retired.length).toBeGreaterThan(0);
      for (const item of retired) {
        expect(trace.backwardHandlers.has(sourceKey(item.source))).toBe(true);
        if (item.record instanceof NodeSucceededRecord) {
          expect(trace.sources.has(sourceKey(item.source))).toBe(true);
          expect(trace.currentWriters.has(item.nodeId)).toBe(false);
        }
      }
    }
    replaying = true;
    const records = built.log.records();
    const forward = spyOn(CodeNode.prototype, "forward").mockImplementation(() => {
      throw new Error("replay evaluated a node body");
    });
    try {
      for (const [length, expected] of [...snapshots].sort(([a], [b]) => a - b)) {
        const replay = replayRecords(new RunState(copyGraph(original)), records.slice(0, length), {
          growBuilder: buildGrow,
        });
        expect(replayTraceSnapshot(replay.trace!)).toEqual(expected);
      }
    } finally {
      forward.mockRestore();
    }
  },
);

test("a nested agent's render capture uses the authored model map and replays without execution", async () => {
  const loaded = (
    await loadFlow(`id: captured_models
name: captured_models
llm_config: {provider: ollama, model: root, ollama: {options: {temperature: 0.2}}}
defs:
  leaf:
    input:
      topic: !str
    nodes:
      writer:
        kind: agent
        input: {topic: '\${input.topic}'}
        prompt: '\${topic}: \${instruction}'
        variables:
          instruction: !str clearly
        output: {topic: str}
    output: \${writer.output}
  repeat_left:
    llm_config: {model: left}
    input:
      topic: !str
    nodes:
      twice:
        kind: loop
        call: leaf
        times: 2
        input: {topic: '\${input.topic}'}
    output: \${twice.output}
  repeat_right:
    llm_config: {model: right}
    input:
      topic: !str
    nodes:
      twice:
        kind: loop
        call: leaf
        times: 2
        input: {topic: '\${input.topic}'}
    output: \${twice.output}
nodes:
  left:
    kind: call
    call: repeat_left
    input: {topic: reefs}
  right:
    kind: map
    call: repeat_right
    over: [reefs, reefs]
    input: {topic: '\${item}'}
output: {left: '\${left.output}', right: '\${right.output}'}
`)
  ).compiled;
  // Each model call's prompt; the reply's topic is the prompt.
  const rendered: string[] = [];
  let replaying = false;
  const models: ModelAccess = {
    env: NO_ENV,
    createModel: (config) => {
      if (replaying) throw new Error("replay tried to invoke a model");
      return {
        provider: config["provider"] as string,
        model: config["model"] as string,
        stream: (request, options) => {
          const prompt = request.messages.at(-1)!.text();
          rendered.push(prompt);
          return new ScriptedModel([textReply(JSON.stringify({ topic: prompt }))]).stream(request, options);
        },
      };
    },
  };
  const built = growing(loaded, {
    effectiveLlmByStaticPath: resolveLlmCascade(loaded, {}, { defaults: { provider: "test", model: "test" } }),
    models,
  });
  const original = copyGraph(built.state.flow);

  await built.run();

  expect(built.state.result).toBeInstanceOf(RunSucceeded);
  const trace = built.state.trace!;
  const agents = trace.occurrences.filter((item) => captureOf(item).variableNames.join() === "instruction");
  expect(agents).toHaveLength(6);
  expect(rendered).toHaveLength(6);
  const block = { options: { temperature: 0.2 } };
  expect(built.state.effectiveLlmByStaticPath).toEqual({
    "left/twice/writer": { provider: "ollama", model: "left", ollama: block },
    "right/twice/writer": { provider: "ollama", model: "right", ollama: block },
  });
  const positions = new Set<string>();
  for (const occurrence of agents) {
    const capture = captureOf(occurrence);
    const scope = capture.scope!;
    const data = occurrence.nodeData;
    expect(data["effective_llm_config"]).toEqual(built.state.effectiveLlmByStaticPath[scope.staticPath.join("/")]);
    expect(data["prompt_template"]).toBe("${topic}: ${instruction}");
    expect(data["variables"]).toEqual({ instruction: "clearly" });
    expect(data["rendered_prompt"]).toBe(`${occurrence.boundInputs!["topic"]}: clearly`);
    expect([["left", "twice", "writer"], ["right", "twice", "writer"]]).toContainEqual([...scope.staticPath]);
    expect(scope.occurrencePath.map((item) => item.kind)).toEqual(
      scope.staticPath[0] === "left" ? ["call", "loop_body"] : ["map_item", "loop_body"],
    );
    positions.add(JSON.stringify([scope.staticPath, scope.occurrencePath]));
  }
  expect(positions.size).toBe(6);
  expect(agents.some((item) => !built.state.flow.nodes.has(item.nodeId))).toBe(true);

  replaying = true;
  const forbidden = () => {
    throw new Error("replay tried to resolve, render, capture, or invoke a model");
  };
  const capture = Node.prototype.captureBackwardContext;
  const spies = [
    spyOn(AgentNode.prototype, "forward").mockImplementation(forbidden),
    spyOn(AgentNode.prototype as unknown as { resolveVariables(): unknown }, "resolveVariables").mockImplementation(
      forbidden,
    ),
    // Replay rebuilds a derived record's capture from the flow, which asks the derived node for its
    // backward material; an AGENT's is never captured again.
    spyOn(Node.prototype, "captureBackwardContext").mockImplementation(function (this: Node) {
      if (this instanceof AgentNode) return forbidden();
      return capture.call(this);
    }),
  ];
  let replay: RunState;
  try {
    replay = replayRecords(new RunState(original), built.log.records(), { growBuilder: buildGrow });
    expect(replay.trace!.occurrences).toEqual(trace.occurrences);
  } finally {
    for (const spy of spies) spy.mockRestore();
  }
  // Learning renders each occurrence again with the node that ran, which the replay keeps.
  for (const occurrence of agents) {
    const { _evaluation_evidence: evidence, ...data } = replay.trace!.backwardContext(occurrence.source).nodeData;
    // The evidence is rebuilt from the steps by a backward pass; the replayed occurrence keeps
    // its one model call.
    expect(evidence).toBeNull();
    expect(replay.trace!.occurrence(occurrence.source).steps.map((step) => step["kind"])).toEqual(["model_call"]);
    expect(data).toEqual(occurrence.nodeData);
  }
});

/** A chat loop of three scripted turns: `reply` binds a template, `__end__` appends with `+`. */
const BOUND_CHAT = `
id: chat
name: chat
llm_config: {provider: anthropic, model: claude-test}
typedefs:
  Msg: {role: str, text: str}
defs:
  turn:
    input: {history: !list(Msg), log: !str}
    nodes:
      ask: {kind: tool, tool_id: ask, input: {prompt: "You:"}, output: str}
      reply:
        kind: agent
        input: {history: "\${input.history}", message: "\${ask.output}"}
        output: str
        prompt: |-
          Conversation so far: \${history}
          User: \${message}
    output:
      history: \${input.history + [Msg(role="user", text=ask.output), Msg(role="assistant", text=reply.output)]}
      log: "\${input.log}\\nUser: \${ask.output}\\nAssistant: \${reply.output}"
nodes:
  chat: {kind: loop, call: turn, input: {history: [], log: ""}, times: 3}
output: \${chat.output.log}
`;

const boundDir = mkdtempSync(join(tmpdir(), "yian-forward-trace-"));
afterAll(() => rmSync(boundDir, { recursive: true, force: true }));

test("an occurrence's bound inputs are rebuilt from its route: a template and a + binding", async () => {
  const log = join(boundDir, "chat.jsonl");
  const config = { provider: "anthropic", model: "claude-test" };
  const answers = ["a", "b", "c"];
  const chat = await loadFlow(BOUND_CHAT, { sourceLabel: "flow.yaml" });
  const result = await runFlow(chat, {}, {
    env: NO_ENV,
    log,
    numWorkers: 0,
    learnable: true,
    models: serving(new ScriptedModel([textReply("A"), textReply("B"), textReply("C")], config)),
    tools: { ask: () => answers.shift()! },
  });
  expect(result.status).toBe("succeeded");
  const reopened = JsonlRunLog.resume(log);
  let trace: RetainedTrace;
  try {
    trace = replayRunLog(chat, reopened).trace!;
  } finally {
    reopened[Symbol.dispose]();
  }
  const of = (nodeId: string) => trace.occurrences.find((occurrence) => occurrence.nodeId === nodeId)!;
  const first = [
    { role: "user", text: "a" },
    { role: "assistant", text: "A" },
  ];
  const second = [...first, { role: "user", text: "b" }, { role: "assistant", text: "B" }];
  expect(of("chat#1/reply").boundInputs).toEqual({ history: first, message: "b" });
  expect(of("chat#1/__end__").boundInputs).toEqual({
    history: second,
    log: "\nUser: a\nAssistant: A\nUser: b\nAssistant: B",
  });
  // The ask's prompt is one of its inputs.
  expect(of("chat#0/ask").boundInputs).toEqual({ prompt: "You:" });
});

/** A run-log line as plain JSON. */
type Wire = Record<string, unknown>;

test("a replayed derived record holding a forward capture is refused", async () => {
  const log = join(boundDir, "derived-capture.jsonl");
  const config = { provider: "anthropic", model: "claude-test" };
  const answers = ["a", "b", "c"];
  const chat = await loadFlow(BOUND_CHAT, { sourceLabel: "flow.yaml" });
  const result = await runFlow(chat, {}, {
    env: NO_ENV,
    log,
    numWorkers: 0,
    learnable: true,
    models: serving(new ScriptedModel([textReply("A"), textReply("B"), textReply("C")], config)),
    tools: { ask: () => answers.shift()! },
  });
  expect(result.status).toBe("succeeded");
  const lines = readFileSync(log, "utf8").trimEnd().split("\n").map((line) => parseJson(line, "codec") as Wire);
  const succeeded = (nodeId: string) =>
    lines.find((line) => line["kind"] === "node_succeeded" && (line["payload"] as Wire)["node_id"] === nodeId)!;
  const end = succeeded("chat#0/__end__")["payload"] as Wire;
  // A derived record holds none: it is rebuilt from the record's sources.
  expect(end["forward_capture"]).toBeNull();
  end["forward_capture"] = (succeeded("chat#0/ask")["payload"] as Wire)["forward_capture"];
  writeFileSync(log, `${lines.map((line) => logJson(line)).join("\n")}\n`);
  const reopened = JsonlRunLog.resume(log);
  try {
    expectError(() => replayRunLog(chat, reopened), RangeError, "a derived record holds no forward capture");
  } finally {
    reopened[Symbol.dispose]();
  }
});

/**
 * A greeting whose CODE node and top-level END each bind a template with literal text, which
 * records an interpolation route. With no `nick`, `shout`'s binding fails after `line`.
 */
const GREET = `
id: greet
name: greet
input: {name: !str, nick: !Optional(str) null}
nodes:
  shout:
    kind: code
    input: {line: "Hello, \${input.name}!", nick: '\${input.nick :? "no nick"}'}
    output: str
    code: return inputs['line']
output: "\${shout.output} (\${input.name})"
`;

/** The first interpolation route in `route`, depth first. */
function interpolationIn(route: ValueRoute): ValueRoute | null {
  if (route.kind === "interpolation") return route;
  for (const child of route.children) {
    const found = interpolationIn(child);
    if (found !== null) return found;
  }
  return null;
}

/** Run `GREET` learnable with `nick` into a fresh log; return the log's path. */
async function greetLog(name: string, nick: string | null): Promise<string> {
  const log = join(boundDir, `${name}.jsonl`);
  const greet = await loadFlow(GREET, { sourceLabel: "flow.yaml" });
  await runFlow(greet, { name: "Ann", nick }, { env: NO_ENV, log, numWorkers: 0, learnable: true });
  return log;
}

/** Replay the run log at `log` and return its trace's occurrences by node id. */
async function replayedOccurrences(log: string): Promise<(nodeId: string) => TraceOccurrence> {
  const greet = await loadFlow(GREET, { sourceLabel: "flow.yaml" });
  const reopened = JsonlRunLog.resume(log);
  let trace: RetainedTrace;
  try {
    trace = replayRunLog(greet, reopened).trace!;
  } finally {
    reopened[Symbol.dispose]();
  }
  return (nodeId) => trace.occurrences.find((occurrence) => occurrence.nodeId === nodeId)!;
}

/** The `value_json` of every interpolation route in the log at `log`, in line order. */
function interpolationTexts(log: string): unknown[] {
  const texts: unknown[] = [];
  const visit = (value: unknown): void => {
    if (Array.isArray(value)) value.forEach(visit);
    else if (value !== null && typeof value === "object") {
      const wire = value as Wire;
      if (wire["kind"] === "interpolation") texts.push(wire["value_json"]);
      Object.values(wire).forEach(visit);
    }
  };
  readFileSync(log, "utf8").trimEnd().split("\n").forEach((line) => visit(parseJson(line, "codec")));
  return texts;
}

test("an interpolation route's text isn't recorded: learning rebuilds it by binding its node again", async () => {
  const log = await greetLog("greet", "A");
  // `shout`'s `line` and END's output.
  expect(interpolationTexts(log)).toEqual([null, null]);
  const of = await replayedOccurrences(log);
  for (const [nodeId, text] of [
    ["shout", "Hello, Ann!"],
    ["__end__", "Hello, Ann! (Ann)"],
  ] as const) {
    const occurrence = of(nodeId);
    expect(interpolationIn(occurrence.capture.route)!.valueJson).toBe(JSON.stringify(text));
    expect(interpolationIn(occurrence.inputRoute)!.valueJson).toBe(JSON.stringify(text));
  }
});

test("a binding that didn't finish rebuilds the interpolation route it collected first", async () => {
  const log = await greetLog("greet-failed", null);
  expect(interpolationTexts(log)).toEqual([null]);
  const of = await replayedOccurrences(log);
  const shout = of("shout");
  expect(shout.record).toBeInstanceOf(NodeFailedRecord);
  expect(shout.boundInputs).toBeNull();
  expect(shout.capture.route.names).toEqual(["line"]);
  expect(interpolationIn(shout.capture.route)!.valueJson).toBe(JSON.stringify("Hello, Ann!"));
});

/** A MAP whose item binding renders a template, and fails on an item that is null. */
const LABELS = `
id: labels
name: labels
input: {xs: !list(Optional(str))}
defs:
  label:
    input: {line: !str, x: !str}
    nodes:
      echo: {kind: code, input: {line: "\${input.line}"}, output: str, code: "return inputs['line']"}
    output: \${echo.output}
nodes:
  each: {kind: map, over: "\${input.xs}", call: label, input: {line: "Item \${item}.", x: '\${item :? "no item"}'}}
output: \${each.output}
`;

test("a MAP whose item binding failed rebuilds each item's interpolation route", async () => {
  const log = join(boundDir, "labels.jsonl");
  const labels = await loadFlow(LABELS, { sourceLabel: "flow.yaml" });
  const result = await runFlow(labels, { xs: ["a", null] }, { env: NO_ENV, log, numWorkers: 0, learnable: true });
  expect(result.status).toBe("failed");
  expect(interpolationTexts(log)).toEqual([null, null]);
  const reopened = JsonlRunLog.resume(log);
  let trace: RetainedTrace;
  try {
    trace = replayRunLog(labels, reopened).trace!;
  } finally {
    reopened[Symbol.dispose]();
  }
  const each = trace.occurrences.find((occurrence) => occurrence.nodeId === "each")!;
  expect(each.record).toBeInstanceOf(NodeFailedRecord);
  expect(each.boundInputs).toBeNull();
  expect(each.capture.items.map(([index, item]) => [index, interpolationIn(item.route)!.valueJson])).toEqual([
    [0, JSON.stringify("Item a.")],
    [1, JSON.stringify("Item .")],
  ]);
});

test("a recorded interpolation route holding its text is refused", () => {
  const capture = new ForwardCapture(ValueRoute.record([["x", new ValueRoute("interpolation", {
    children: [new ValueRoute("source", { source: new SourceRef("run", 1) })],
    spans: [[2, 3]],
  })]]));
  const wire = wireOf(capturedEnvelope(capture));
  wire.payload.forward_capture.route.children[0].value_json = '"n=3"';
  expectError(() => decodeWire(wire), RangeError, "an interpolation route records no value_json");
});

test("a recorded capture holding node data is refused", () => {
  const wire = wireOf(capturedEnvelope(new ForwardCapture(ValueRoute.record([]))));
  wire.payload.forward_capture.node_data = {};
  expectError(
    () => decodeWire(wire),
    RangeError,
    `invalid ${CAPTURE_LABEL} fields; missing=[], extra=["node_data"]`,
  );
});

test("a MAP's forward failing otherwise when bound again is a fault, not kept", async () => {
  const log = join(boundDir, "labels-fault.jsonl");
  const labels = await loadFlow(LABELS, { sourceLabel: "flow.yaml" });
  await runFlow(labels, { xs: ["a", null] }, { env: NO_ENV, log, numWorkers: 0, learnable: true });
  const reopened = JsonlRunLog.resume(log);
  let trace: RetainedTrace;
  try {
    trace = replayRunLog(labels, reopened).trace!;
  } finally {
    reopened[Symbol.dispose]();
  }
  const each = trace.occurrences.find((occurrence) => occurrence.nodeId === "each")!;
  const spy = spyOn(MapNode.prototype, "forward").mockImplementation(() => {
    throw new TypeError("forward broke");
  });
  try {
    expectError(() => each.capture, TypeError, "forward broke");
  } finally {
    spy.mockRestore();
  }
});

/** A TOOL `sleep` whose `until` renders a template that isn't a date: its body fails after binding. */
const NAP = `
id: nap
name: nap
input: {x: !str}
nodes:
  nap: {kind: tool, tool_id: sleep, input: {until: "not a date \${input.x}"}}
output: \${nap.output}
`;

test("a tool body failing with a binding code after its binding finished rebuilds its inputs and text", async () => {
  const log = join(boundDir, "nap.jsonl");
  const nap = await loadFlow(NAP, { sourceLabel: "flow.yaml" });
  const result = await runFlow(nap, { x: "soon" }, { env: NO_ENV, log, numWorkers: 0, learnable: true });
  expect(result.status).toBe("failed");
  const reopened = JsonlRunLog.resume(log);
  let trace: RetainedTrace;
  try {
    trace = replayRunLog(nap, reopened).trace!;
  } finally {
    reopened[Symbol.dispose]();
  }
  const occurrence = trace.occurrences.find((each) => each.nodeId === "nap")!;
  expect(occurrence.record).toBeInstanceOf(NodeFailedRecord);
  expect(interpolationIn(occurrence.capture.route)!.valueJson).toBe(JSON.stringify("not a date soon"));
  expect(interpolationIn(occurrence.inputRoute)!.valueJson).toBe(JSON.stringify("not a date soon"));
  // `sleep` refuses `until` with `contract.input` after its binding finished.
  expect((occurrence.record as NodeFailedRecord).failure.code).toBe("contract.input");
  expect(occurrence.record.forwardCapture!.bound).toBe(true);
  expect(occurrence.boundInputs).toEqual({ until: "not a date soon" });
});

/** A TOOL `sleep` whose `until` falls back to a date when a deep read of `x` is null. */
const NAP_FALLBACK = `
id: nap-fallback
name: nap_fallback
input: {x: !object}
nodes:
  nap: {kind: tool, tool_id: sleep, input: {until: "\${input.x?.k?.deep?.more | 'tomorrow'}"}}
output: \${nap.output}
`;

test("a binding that failed rebuilds no inputs, even where binding again from its reads would finish", async () => {
  const log = join(boundDir, "nap-fallback.jsonl");
  const nap = await loadFlow(NAP_FALLBACK, { sourceLabel: "flow.yaml" });
  // `deep` is an int, so reading `.more` from it fails the binding before the fallback.
  const result = await runFlow(nap, { x: { k: { deep: 1n } } }, { env: NO_ENV, log, numWorkers: 0, learnable: true });
  expect(result.status).toBe("failed");
  const reopened = JsonlRunLog.resume(log);
  let trace: RetainedTrace;
  try {
    trace = replayRunLog(nap, reopened).trace!;
  } finally {
    reopened[Symbol.dispose]();
  }
  const occurrence = trace.occurrences.find((each) => each.nodeId === "nap")!;
  const record = occurrence.record as NodeFailedRecord;
  expect(record.failure.code).toBe("binding.evaluate");
  // The failed read recorded no route, so binding again without `x` would take the fallback.
  expect(record.forwardCapture!.route.names).toEqual([]);
  expect(record.forwardCapture!.bound).toBe(false);
  expect(occurrence.boundInputs).toBeNull();
});

/**
 * The occurrence of a TraceNode `a` whose binding failed (`binding.evaluate`), recorded with
 * `capture`, and whose node binds again as `rebinder`.
 */
function failedBindingOccurrence(capture: ForwardCapture, rebinder: Rebinder): TraceOccurrence {
  const failure = new Failure({ code: "binding.evaluate", message: "Missing.", nodeId: "a" });
  const unbound = new ForwardCapture(capture.route, {
    evaluatedKeys: capture.evaluatedKeys,
    items: capture.items,
    scope: capture.scope,
    inputNames: capture.inputNames,
    variableNames: capture.variableNames,
    bound: false,
  });
  const trace = runNode(startedTrace(), 1, new TraceNode("a"), new NodeFailedRecord(failure, { forwardCapture: unbound }), {
    rebinder,
    terminalSucceeded: false,
    terminalFailure: failure,
  });
  return trace.occurrences[0]!;
}

/** A rebinder whose binding fails after collecting `route` (and `items`); it counts its calls. */
function partialRebinder(route: ValueRoute, items: readonly (readonly [number, BindingCapture])[] = []) {
  const calls = { count: 0 };
  const rebinder: Rebinder = () => {
    calls.count += 1;
    return {
      capture: new BindingCapture(null, route, new Set(), items),
      failure: new BindingError("Missing.", { failureCode: "binding.evaluate" }),
    };
  };
  return { rebinder, calls };
}

/** An interpolation over source 0's `x` with `text`, or no text. */
function textRoute(text: string | null): ValueRoute {
  return new ValueRoute("interpolation", {
    children: [sourceRoute(0, ["x"])],
    spans: [[2, 3]],
    valueJson: text === null ? null : JSON.stringify(text),
  });
}

test("a capture without an interpolation route is read as held, without binding again", () => {
  let rebinds = 0;
  const counted: Rebinder = (sources, resolve, route) => {
    rebinds += 1;
    return fixedRebinder({ v: 1n })(sources, resolve, route);
  };
  const occurrence = drivenTrace(new TraceNode("a"), { rebinder: counted }).occurrences[0]!;
  expect(occurrence.capture.route).toEqual(captureV().route);
  expect(occurrence.inputRoute).toEqual(captureV().route);
  expect(rebinds).toBe(0);
  expect(occurrence.boundInputs).toEqual({ v: 1n });
  expect(rebinds).toBe(1);
});

test("an interpolation route's text rebuilds from the same binding; another shape is refused", () => {
  const recorded = new ForwardCapture(ValueRoute.record([["v", textRoute(null)]]), { inputNames: ["v"] });
  const rebound = partialRebinder(ValueRoute.record([["v", textRoute("x=1")]]));
  const filled = failedBindingOccurrence(recorded, rebound.rebinder);
  expect(filled.capture.route.children[0]!.valueJson).toBe('"x=1"');
  expect(filled.inputRoute.children[0]!.valueJson).toBe('"x=1"');
  expect(rebound.calls.count).toBe(1);

  const missing = failedBindingOccurrence(recorded, partialRebinder(ValueRoute.record([])).rebinder);
  expectError(() => missing.capture, RangeError, "binding again collects no input 'v'");
  const other = failedBindingOccurrence(recorded, partialRebinder(ValueRoute.record([["v", sourceRoute(0, ["x"])]])).rebinder);
  expectError(() => other.capture, RangeError, "binding again collects a source route where the record has a interpolation route");

  const item = new ForwardCapture(ValueRoute.record([["n", textRoute(null)]]));
  const withItem = new ForwardCapture(ValueRoute.record([["v", sourceRoute(0, ["x"])]]), { inputNames: ["v"], items: [[0, item]] });
  const noItem = failedBindingOccurrence(withItem, partialRebinder(ValueRoute.record([["v", sourceRoute(0, ["x"])]])).rebinder);
  expectError(() => noItem.inputRoute, RangeError, "binding again collects no item 0");
});

/**
 * A learnable flow whose AGENT renders a list prompt with typed variables and reads its model from
 * `pick` when it runs; its pre-assert holds only for a non-negative `n`.
 */
const PICKS = `
id: picks
name: picks
llm_config: {provider: openai}
typedefs: {Tone: 'Literal(concise, detailed)'}
input: {topic: !str, n: !int}
nodes:
  pick: {kind: code, output: str, code: "return 'gpt-picked'"}
  writer:
    kind: agent
    input: {topic: '\${input.topic}', n: '\${input.n}'}
    variables: {tone: !Tone concise, count: !int 3}
    llm_config: {model: '\${pick.output}'}
    asserts: ['n >= 0']
    system_prompt: 'Be \${tone}.'
    prompt:
      - user: '\${topic}, \${count} lines'
      - assistant: Which tone?
      - user: '\${tone}'
    output: str
output: \${writer.output}
`;

/** Run `PICKS` learnable with `n` into a fresh log; return the log's path and the run's status. */
async function picksLog(name: string, n: number): Promise<[string, string]> {
  const log = join(boundDir, `${name}.jsonl`);
  const flow = await loadFlow(PICKS, { sourceLabel: "flow.yaml" });
  const result = await runFlow(flow, { topic: "reefs", n }, {
    env: NO_ENV,
    log,
    numWorkers: 0,
    learnable: true,
    models: serving(new ScriptedModel([textReply("ok")], { provider: "openai", model: "gpt-picked" })),
  });
  return [log, result.status];
}

/** Whether any record of the log at `log` holds a `node_data` field. */
function recordsNodeData(log: string): boolean {
  return readFileSync(log, "utf8").includes('"node_data"');
}

/** Replay `PICKS`'s run log at `log` and return its trace's occurrences by node id. */
async function replayedPicks(log: string): Promise<(nodeId: string) => TraceOccurrence> {
  const flow = await loadFlow(PICKS, { sourceLabel: "flow.yaml" });
  const reopened = JsonlRunLog.resume(log);
  let trace: RetainedTrace;
  try {
    trace = replayRunLog(flow, reopened).trace!;
  } finally {
    reopened[Symbol.dispose]();
  }
  return (nodeId) => trace.occurrences.find((occurrence) => occurrence.nodeId === nodeId)!;
}

test("an AGENT's node data is rebuilt from its node and bound inputs, not read from the record", async () => {
  const [log, status] = await picksLog("picks", 2);
  expect(status).toBe("succeeded");
  expect(recordsNodeData(log)).toBe(false);
  const of = await replayedPicks(log);
  expect(of("__start__").nodeData).toEqual({ input_output_paths: { n: ["n"], topic: ["topic"] } });
  expect(of("pick").nodeData).toEqual({});
  expect(of("writer").nodeData).toEqual({
    effective_llm_config: {
      _connection: {
        credential_id: null,
        endpoint: "https://api.openai.com/v1",
        method: "api_key",
        source: "missing",
        store_path: null,
      },
      model: "gpt-picked",
      provider: "openai",
    },
    prompt_template: '[{"user":"${topic}, ${count} lines"},{"assistant":"Which tone?"},{"user":"${tone}"}]',
    rendered_prompt:
      '[{"role":"user","text":"reefs, 3 lines"},{"role":"assistant","text":"Which tone?"},{"role":"user","text":"concise"}]',
    variable_domains: { count: "int", tone: "Literal(concise, detailed)" },
    variables: { count: 3n, tone: "concise" },
  });
});

test("an AGENT whose pre-assert failed rebuilds no render, and one whose binding failed no node data", async () => {
  const [log, status] = await picksLog("picks-negative", -1);
  expect(status).toBe("failed");
  expect(recordsNodeData(log)).toBe(false);
  const of = await replayedPicks(log);
  expect((of("writer").record as NodeFailedRecord).failure.code).toBe("assertion.failed");
  expect(of("writer").nodeData).toEqual({});
  const greet = await greetLog("greet-node-data", null);
  const shout = (await replayedOccurrences(greet))("shout");
  expect(shout.boundInputs).toBeNull();
  expect(shout.nodeData).toEqual({});
});
