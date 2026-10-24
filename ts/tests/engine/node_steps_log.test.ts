/**
 * Node steps: a port of `tests/engine/test_node_steps_log.py`, the `NodeStepRecord` a running node
 * appends and how the run-state reducer folds it. Python's `ValueError` is a `RangeError` here.
 *
 * Ported: `test_node_step_record_detaches_the_callers_step`,
 * `test_node_step_record_rejects_invalid_values` (list, text, null, NaN, a class instance, a
 * `Set`, an empty and a non-text node id; each with its exact error).
 *
 * Ported, the wire tests: `test_node_step_round_trips_under_its_wire_kind` (schema 20),
 * `test_node_step_wire_payload_rejects_invalid_shapes` (list, text, empty node id, missing step,
 * extra field; each with its exact class and message),
 * `test_node_step_wire_payload_rejects_non_finite_numbers` (`invalid run-record JSON`, caused by
 * the reader's `NaN is not JSON`).
 *
 * Ported, the reducer tests: `test_prepare_node_step_requires_an_in_flight_node` (3 rows),
 * `test_prepare_node_step_rejects_an_uninitialized_state`,
 * `test_node_starts_count_started_records`, `test_resume_keeps_steps_and_a_resumed_start_counts_two`,
 * `test_success_deletes_the_nodes_steps_and_starts`, `test_prepare_node_step_rejects_a_terminal_state`,
 * `test_failure_deletes_the_nodes_steps_and_starts`, `test_route_deletes_the_nodes_steps_and_starts`,
 * `test_expansion_deletes_the_spawners_steps_and_starts` (on the shared `initializedSpawner` and
 * `fragment`),
 * `test_learnable_run_accepts_a_step_with_an_empty_trace_delta`,
 * `test_broken_step_state_is_rejected_before_append` (4 rows) and
 * `test_loop_body_steps_go_at_the_outcome_and_stay_gone_after_the_prune` (on the shared
 * `initializedSpawner`, `parentFlow` and `canonicalChild`; the LOOP seed's iteration is `0n`, and
 * the replay runs over a `copyGraph` copy of a fresh parent flow). Each rejected row names its one
 * class and full message, where Python matches a part of the message or accepts either class.
 *
 * Ported, the replay tests (through `replayRecords` on a fresh `fanoutFlow()`; the log never
 * grows, so the builder is never called): `test_replay_reproduces_steps_and_starts`,
 * `test_learnable_replay_reproduces_steps_starts_and_trace`,
 * `test_replay_without_steps_has_no_steps`.
 *
 * TS-only: a step writes `-0` as `0`; a `FileRef` in a step is refused; a wire step keeps
 * yian's infinity tag as `Infinity`. A step has a string `kind` and an int `n` from 1, the node's
 * step count across its starts plus one; a record with any other `n` is refused on append and on
 * replay ("node step n is not dense").
 *
 * Changed:
 * - the `infinity` row of `test_node_step_record_rejects_invalid_values`: a step keeps
 *   `±Infinity` (a TS yian value).
 * - `test_node_step_too_deep_for_python_is_a_value_error`: a 5000-deep step is accepted (the TS
 *   codec is iterative).
 * - `test_prepare_node_step_rejects_a_step_past_the_nesting_limit`: a live step past the limit is
 *   accepted. A step is node detail, not a yian value, and an AGENT's `model_call` step holds a
 *   reply whose tool call arguments may themselves sit at the limit.
 * - `test_prepare_node_step_appends_in_order_without_observers`: the plan's delta is checked by
 *   its `kind` and fields, not its class name; a node's steps are a frozen array, not a tuple.
 * - `test_committed_step_is_detached_from_the_caller_and_the_record`: the record's step is
 *   deep-frozen, so changing it is a `TypeError`.
 * - `test_prepare_node_step_rejects_invalid_steps`: the list and NaN rows, each with its class and
 *   message.
 * - `test_frontier_commit_rejects_a_step_record_with_a_start_delta`: the forged transition spreads
 *   the frozen plan with a start delta.
 *
 * Not ported (TS types): the `non-text-key` row of `test_node_step_record_rejects_invalid_values`
 * and the `{1: "x"}` row of `test_prepare_node_step_rejects_invalid_steps` (JS object keys are
 * always text); the `list-steps` row of `test_broken_step_state_is_rejected_before_append` (a tuple
 * and a list are both an array).
 */

import { expect, test } from "bun:test";

import { buildGrow } from "../../src/compile/expand.ts";
import { copyGraph, START_ID } from "../../src/compile/model.ts";
import { current } from "../../src/defaults.ts";
import { Grow } from "../../src/nodes/base.ts";
import { LoopNode } from "../../src/nodes/loop/node.ts";
import {
  decodeRecord,
  encodeRecord,
  MemoryRunLog,
  NodeExpandedRecord,
  NodeStepRecord,
  RecordEnvelope,
  SCHEMA_VERSION,
} from "../../src/runtime/log.ts";
import {
  commitPrepared,
  PreparedTransition,
  prepareNodeFailed,
  prepareNodeRouted,
  prepareNodeStarted,
  prepareNodeStep,
  prepareRecord,
  prepareRunResumed,
  replayRecords,
  RunState,
} from "../../src/runtime/state.ts";
import { FileRef, TypeCheckError } from "../../src/typesys/index.ts";
import { JSONDecodeError, parseJson } from "../../src/typesys/json.ts";
import {
  canonicalChild,
  commitStartSuccess,
  fanoutFlow,
  forwardCapture,
  fragmentBuilder,
  fragmentIds,
  initializedAndStarted,
  initializedSpawner,
  nodeFailure,
  parentFlow,
  prepareOutput,
  routeInFlight,
  prepareGrowth,
} from "./run_state_fixtures.ts";

/** Assert `fn` throws an error of exactly class `cls` with exactly `message`. */
function expectError(fn: () => unknown, cls: new (...args: never[]) => Error, message: string): unknown {
  let caught: unknown;
  try {
    fn();
  } catch (error) {
    caught = error;
  }
  expect((caught as Error | undefined)?.constructor).toBe(cls);
  expect((caught as Error).message).toBe(message);
  return caught;
}

/** Return a JSON object nested `levels` objects deep, itself included. */
function nested(levels: number): Record<string, unknown> {
  let value: Record<string, unknown> = {};
  for (let i = 0; i < levels - 1; i++) value = { inner: value };
  return value;
}

class Thing {}

const FAILED = {
  kind: "tool_attempt",
  event: "failed",
  attempt: 1n,
  error: { type: "TimeoutError", message: "slow" },
  retry_in: 1.5,
};

test("a node step round trips under its wire kind", () => {
  const envelope = new RecordEnvelope(SCHEMA_VERSION, "run-1", 3, new NodeStepRecord("a", 1, FAILED));

  const encoded = encodeRecord(envelope);
  const decoded = decodeRecord(encoded);

  expect(decoded).toEqual(envelope);
  expect(encodeRecord(decoded)).toBe(encoded);
  const wire = parseJson(encoded, "codec") as Record<string, unknown>;
  expect(wire["kind"]).toBe("node_step");
  expect(wire["schema"]).toBe(21n);
  expect(wire["payload"]).toEqual({ node_id: "a", n: 1n, step: FAILED });
});

test("a wire step keeps yian's infinity tag", () => {
  const text =
    '{"kind":"node_step","payload":{"n":1,"node_id":"a","step":{"high":{"$float":"inf"},"kind":"x"}},' +
    '"run_id":"run-1","schema":21,"seq":0}';
  const record = decodeRecord(text).record as NodeStepRecord;
  expect(record.step).toEqual({ high: Infinity, kind: "x" });
  expect(record.n).toBe(1);
  expect(encodeRecord(decodeRecord(text))).toBe(text);
});

test.each<[string, unknown, new (...args: never[]) => Error, string]>([
  ["list", { node_id: "a", n: 1, step: [1] }, TypeError, "node_step payload step must be an object"],
  ["text", { node_id: "a", n: 1, step: "text" }, TypeError, "node_step payload step must be an object"],
  ["empty-node-id", { node_id: "", n: 1, step: { kind: "x" } }, RangeError, "node_step payload node_id must be a non-empty string"],
  ["missing-step", { node_id: "a", n: 1 }, RangeError, 'invalid node_step payload fields; missing=["step"], extra=[]'],
  ["missing-n", { node_id: "a", step: { kind: "x" } }, RangeError, 'invalid node_step payload fields; missing=["n"], extra=[]'],
  ["zero-n", { node_id: "a", n: 0, step: { kind: "x" } }, RangeError, "node step n must be at least 1"],
  ["text-n", { node_id: "a", n: "1", step: { kind: "x" } }, TypeError, "node_step payload n must be an integer"],
  ["no-kind", { node_id: "a", n: 1, step: {} }, TypeError, "node step kind must be a non-empty string"],
  [
    "extra-field",
    { node_id: "a", n: 1, step: { kind: "x" }, extra: true },
    RangeError,
    'invalid node_step payload fields; missing=[], extra=["extra"]',
  ],
])("a node step wire payload rejects an invalid shape: %s", (_name, payload, cls, message) => {
  const text = JSON.stringify({ schema: SCHEMA_VERSION, run_id: "run-1", seq: 0, kind: "node_step", payload });
  expectError(() => decodeRecord(text), cls, message);
});

test("a node step wire payload rejects non-finite numbers", () => {
  const text =
    '{"schema":21,"run_id":"run-1","seq":0,"kind":"node_step","payload":{"node_id":"a","n":1,"step":{"retry_in":NaN}}}';
  const caught = expectError(() => decodeRecord(text), RangeError, "invalid run-record JSON") as Error;
  expect((caught.cause as Error).constructor).toBe(JSONDecodeError);
  expect((caught.cause as Error).message).toStartWith("NaN is not JSON");
});

test("a node step record detaches the caller's step", () => {
  const step = { kind: "tool_attempt", event: "started", attempt: 2n, tags: ["x", "y"] };
  const record = new NodeStepRecord("a", 1, step);
  step.attempt = 99n;
  step.tags.push("z");

  expect(record.step).toEqual({ attempt: 2n, event: "started", kind: "tool_attempt", tags: ["x", "y"] });
  expect(Object.keys(record.step)).toEqual(["attempt", "event", "kind", "tags"]);
});

test.each<[string, unknown, unknown, unknown, new (...args: never[]) => Error, string, (string | number)[] | null]>([
  ["list", "a", 1, ["not", "an", "object"], TypeError, "node step must be a JSON object", null],
  ["text", "a", 1, "text", TypeError, "node step must be a JSON object", null],
  ["null", "a", 1, null, TypeError, "node step must be a JSON object", null],
  ["NaN", "a", 1, { kind: "x", retry_in: NaN }, TypeCheckError, "NaN is not a yian value", ["retry_in"]],
  ["a class instance", "a", 1, { kind: "x", when: new Thing() }, TypeCheckError, "<Thing object> is not a JSON value", ["when"]],
  ["a Set", "a", 1, { kind: "x", items: new Set([1, 2]) }, TypeCheckError, "<Set object> is not a JSON value", ["items"]],
  [
    "a FileRef",
    "a",
    1,
    { kind: "x", file: new FileRef({ uri: "memory://answer" }) },
    TypeCheckError,
    '{"uri":"memory://answer","mime":null,"name":null} is not a JSON value',
    ["file"],
  ],
  ["an empty node id", "", 1, { kind: "x" }, RangeError, "node_id must be non-empty", null],
  ["a non-text node id", 1, 1, { kind: "x" }, TypeError, "node_id must be a string", null],
  ["no kind", "a", 1, { event: "started" }, TypeError, "node step kind must be a non-empty string", null],
  ["an empty kind", "a", 1, { kind: "" }, TypeError, "node step kind must be a non-empty string", null],
  ["a non-text kind", "a", 1, { kind: 1n }, TypeError, "node step kind must be a non-empty string", null],
  ["a float n", "a", 1.5, { kind: "x" }, TypeError, "node step n must be an int", null],
  ["a bigint n", "a", 1n, { kind: "x" }, TypeError, "node step n must be an int", null],
  ["a zero n", "a", 0, { kind: "x" }, RangeError, "node step n must be at least 1", null],
])("a node step record rejects %s", (_name, nodeId, n, step, cls, message, path) => {
  const caught = expectError(() => new NodeStepRecord(nodeId as never, n as never, step as never), cls, message);
  if (path !== null) expect((caught as TypeCheckError).path).toEqual(path);
});

test("a node step record keeps infinite floats and writes -0 as 0", () => {
  const record = new NodeStepRecord("a", 1, { kind: "x", high: Infinity, low: [-Infinity], zero: -0 });

  expect(record.step).toEqual({ high: Infinity, kind: "x", low: [-Infinity], zero: 0 });
  expect(Object.is(record.step["zero"], 0)).toBe(true);
});

test("a node step too deep for Python is accepted", () => {
  const record = new NodeStepRecord("a", 1, { kind: "x", ...nested(5000) });

  let value: unknown = record.step;
  let depth = 1;
  while (Object.keys(value as object).length > 0) {
    expect(Object.isFrozen(value)).toBe(true);
    value = (value as Record<string, unknown>)["inner"];
    depth++;
  }
  expect(depth).toBe(5000);
});

/** Return a state and log with START done and node `first` in flight. */
function firstInFlight(options: { learnable?: boolean } = {}): { state: RunState; log: MemoryRunLog } {
  const { state, log } = initializedAndStarted(fanoutFlow(), options);
  commitStartSuccess(state, log);
  commitPrepared(log, state, prepareNodeStarted(state, "first"));
  return { state, log };
}

test("prepareNodeStep appends in order without observers", () => {
  const { state, log } = firstInFlight();
  const started = { kind: "tool_attempt", event: "started", attempt: 2n };

  const first = prepareNodeStep(state, "first", FAILED);
  expect(first.observers).toEqual([]);
  expect(first.record).toStrictEqual(new NodeStepRecord("first", 1, FAILED));
  const plan = first._plan;
  if (plan.kind !== "frontier") throw new Error(`expected a frontier plan, found ${plan.kind}`);
  expect(plan.delta).toEqual({ kind: "step", nodeId: "first", step: FAILED });
  expect(state.nodeSteps.has("first")).toBe(false);
  expect(commitPrepared(log, state, first)).toEqual([]);
  const second = prepareNodeStep(state, "first", started);
  expect((second.record as NodeStepRecord).n).toBe(2);
  expect(commitPrepared(log, state, second)).toEqual([]);

  expect(state.nodeSteps).toEqual(new Map([["first", [FAILED, started]]]));
  expect(Object.isFrozen(state.nodeSteps.get("first"))).toBe(true);
  expect(state.nodeStarts.get("first")).toBe(1);
  const kinds = log.records().map((envelope) => envelope.record.constructor);
  expect(kinds.slice(-2)).toEqual([NodeStepRecord, NodeStepRecord]);
  expect(state.nextSequence).toBe(log.nextSequence);
  expect(state.ready).toEqual(["second"]);
});

test("a committed step is detached from the caller and the record", () => {
  const { state, log } = firstInFlight();
  const step = { kind: "tool_attempt", event: "failed", error: { type: "E", message: "m" } };

  const prepared = prepareNodeStep(state, "first", step);
  commitPrepared(log, state, prepared);
  step.error.type = "changed";
  expect(() => {
    ((prepared.record as NodeStepRecord).step["error"] as Record<string, unknown>)["type"] = "changed";
  }).toThrow(TypeError);

  expect(state.nodeSteps.get("first")).toEqual([
    { error: { message: "m", type: "E" }, event: "failed", kind: "tool_attempt" },
  ]);
});

test.each([
  ["second", "NodeStep node 'second' is not in-flight"],
  ["missing", "NodeStep node 'missing' does not exist in the flow"],
  [START_ID, `NodeStep node '${START_ID}' must have UNKNOWN disposition`],
] as const)("prepareNodeStep requires an in-flight node: %s", (nodeId, message) => {
  const { state, log } = firstInFlight();
  const before = log.nextSequence;

  expectError(() => prepareNodeStep(state, nodeId, { kind: "x", event: "started" }), RangeError, message);

  expect(log.nextSequence).toBe(before);
  expect(state.nodeSteps).toEqual(new Map());
});

test("prepareNodeStep refuses a terminal state", () => {
  const { state, log } = firstInFlight();
  commitPrepared(log, state, prepareNodeStarted(state, "second"));
  commitPrepared(log, state, prepareNodeFailed(state, nodeFailure("first")));
  expect(state.result).not.toBeNull();
  expect(state.inFlight.has("second")).toBe(true);

  expectError(
    () => prepareNodeStep(state, "second", { kind: "x", event: "started" }),
    RangeError,
    "terminal run state cannot prepare another graph transition",
  );
});

test("prepareNodeStep refuses an uninitialized state", () => {
  expectError(
    () => prepareNodeStep(new RunState(fanoutFlow()), START_ID, { kind: "x", event: "started" }),
    RangeError,
    "run state is not initialized",
  );
});

test("prepareNodeStep accepts a step past the value nesting limit", () => {
  const { state, log } = firstInFlight();
  const limit = Number(current().maxValueNesting);

  // An AGENT's model_call step wraps a reply whose tool call arguments may sit at the limit.
  commitPrepared(log, state, prepareNodeStep(state, "first", { kind: "x", ...nested(limit + 5) }));
  expect(state.nodeSteps.get("first")).toHaveLength(1);
});

test.each<[string, unknown, new (...args: never[]) => Error, string]>([
  ["a list", ["x"], TypeError, "node step must be a JSON object"],
  ["NaN", { n: NaN }, TypeCheckError, "NaN is not a yian value"],
])("prepareNodeStep refuses an invalid step: %s", (_name, step, cls, message) => {
  const { state } = firstInFlight();
  expectError(() => prepareNodeStep(state, "first", step), cls, message);
});

test("node starts count the started records", () => {
  const { state, log } = initializedAndStarted(fanoutFlow());
  expect(state.nodeStarts).toEqual(new Map([[START_ID, 1]]));
  commitStartSuccess(state, log);
  expect(state.nodeStarts).toEqual(new Map());
  commitPrepared(log, state, prepareNodeStarted(state, "first"));
  commitPrepared(log, state, prepareNodeStarted(state, "second"));
  expect(state.nodeStarts).toEqual(
    new Map([
      ["first", 1],
      ["second", 1],
    ]),
  );
});

test("a resume keeps the steps, and a resumed start counts two", () => {
  const { state, log } = firstInFlight();
  commitPrepared(log, state, prepareNodeStep(state, "first", FAILED));

  commitPrepared(log, state, prepareRunResumed(state, { numWorkers: 0 }));
  expect(state.nodeSteps).toEqual(new Map([["first", [FAILED]]]));
  expect(state.nodeStarts).toEqual(new Map([["first", 1]]));
  expect(state.inFlight.has("first")).toBe(false);
  expect(state.ready).toEqual(["first", "second"]);

  // A resumed node is ready again, not in flight, so it can't record a step yet.
  expectError(
    () => prepareNodeStep(state, "first", { kind: "x", event: "started" }),
    RangeError,
    "NodeStep node 'first' is not in-flight",
  );
  commitPrepared(log, state, prepareNodeStarted(state, "first"));
  expect(state.nodeStarts).toEqual(new Map([["first", 2]]));
  expect(state.nodeSteps).toEqual(new Map([["first", [FAILED]]]));
});

test("a success deletes the node's steps and starts", () => {
  const { state, log } = firstInFlight();
  commitPrepared(log, state, prepareNodeStep(state, "first", FAILED));
  commitPrepared(log, state, prepareNodeStarted(state, "second"));
  commitPrepared(log, state, prepareNodeStep(state, "second", FAILED));

  commitPrepared(log, state, prepareOutput(log, state, "first", { done: true }));

  expect(state.nodeSteps).toEqual(new Map([["second", [FAILED]]]));
  expect(state.nodeStarts).toEqual(new Map([["second", 1]]));
});

test("a failure deletes the node's steps and starts", () => {
  const { state, log } = firstInFlight();
  commitPrepared(log, state, prepareNodeStep(state, "first", FAILED));

  commitPrepared(log, state, prepareNodeFailed(state, nodeFailure("first")));

  expect(state.nodeSteps.has("first")).toBe(false);
  expect(state.nodeStarts.has("first")).toBe(false);
  expect(state.ready).toEqual(["second"]);
});

test("a route deletes the node's steps and starts", () => {
  const { state, log } = routeInFlight();
  commitPrepared(log, state, prepareNodeStep(state, "router", FAILED));
  expect(state.nodeStarts).toEqual(new Map([["router", 1]]));

  commitPrepared(log, state, prepareNodeRouted(state, "router", "yes"));

  expect(state.nodeSteps).toEqual(new Map());
  expect(state.nodeStarts).toEqual(new Map());
  expect(state.ready).toEqual(["chosen"]);
});

test("an expansion deletes the spawner's steps and starts", () => {
  const { state, log } = initializedSpawner();
  commitPrepared(log, state, prepareNodeStep(state, "spawn", FAILED));
  expect(state.nodeStarts).toEqual(new Map([["spawn", 1]]));

  commitPrepared(log, state, prepareGrowth(log, state, "spawn", new Grow({}), fragmentBuilder));

  expect(state.nodeSteps).toEqual(new Map());
  expect(state.nodeStarts).toEqual(new Map());
  expect(state.ready).toEqual(["spawn/child/start"]);
});

test("LOOP body steps go at the outcome and stay gone after the prune", async () => {
  const loop = new LoopNode("spawn", {
    flowId: "child",
    child: canonicalChild(),
    childInputs: [],
    predicateKind: "times",
    times: 2,
    maxIters: 2,
    originId: "spawn",
  });
  loop.params = [];
  const { state, log } = initializedSpawner(parentFlow({ spawner: loop }));
  const expansion = prepareGrowth(log, state, "spawn", new Grow([{}, 0n]), buildGrow);
  commitPrepared(log, state, expansion);
  const { fragmentStartId: bodyStart, fragmentEndId: bodyEnd } = fragmentIds(expansion);

  commitPrepared(log, state, prepareNodeStarted(state, bodyStart));
  commitPrepared(log, state, prepareNodeStep(state, bodyStart, FAILED));
  expect(state.nodeSteps).toEqual(new Map([[bodyStart, [FAILED]]]));
  commitPrepared(log, state, prepareOutput(log, state, bodyStart, {}));
  expect(state.nodeSteps).toEqual(new Map());
  expect(state.nodeStarts).toEqual(new Map());
  commitPrepared(log, state, prepareNodeStarted(state, bodyEnd));
  commitPrepared(log, state, prepareOutput(log, state, bodyEnd, null));

  commitPrepared(log, state, prepareNodeStarted(state, "spawn~1"));
  commitPrepared(log, state, prepareNodeStep(state, "spawn~1", FAILED));
  const grow = await state.flow.nodes.get("spawn~1")!.run({});
  const prepared = prepareGrowth(log, state, "spawn~1", grow as Grow, buildGrow);
  expect((prepared.record as NodeExpandedRecord).pruneIds).toContain(bodyStart);
  commitPrepared(log, state, prepared);

  expect(state.flow.nodes.has(bodyStart)).toBe(false);
  expect(state.nodeSteps).toEqual(new Map());
  expect(state.nodeStarts).toEqual(new Map());
  const replayed = replayRecords(new RunState(copyGraph(parentFlow({ spawner: loop }))), log.records(), {
    growBuilder: buildGrow,
  });
  expect(replayed.nodeSteps).toEqual(new Map());
  expect(replayed.nodeStarts).toEqual(new Map());
});

/** Return `firstInFlight`'s state after a step, a resume, a second start and a second step. */
function resumedWithSteps(options: { learnable?: boolean } = {}): { state: RunState; log: MemoryRunLog } {
  const { state, log } = firstInFlight(options);
  commitPrepared(log, state, prepareNodeStep(state, "first", FAILED));
  commitPrepared(log, state, prepareRunResumed(state, { numWorkers: 0 }));
  commitPrepared(log, state, prepareNodeStarted(state, "first"));
  commitPrepared(log, state, prepareNodeStep(state, "first", { kind: "tool_attempt", event: "started", attempt: 3n }));
  return { state, log };
}

test("a node's step n counts its steps across its starts", () => {
  const { log } = resumedWithSteps();
  const steps = log.records().flatMap((envelope) => (envelope.record instanceof NodeStepRecord ? [envelope.record.n] : []));
  expect(steps).toEqual([1, 2]);
});

test("a step whose n isn't the node's next is refused on append", () => {
  const { state, log } = firstInFlight();
  commitPrepared(log, state, prepareNodeStep(state, "first", FAILED));
  const before = log.nextSequence;

  for (const n of [1, 3]) {
    expectError(
      () => prepareRecord(state, new NodeStepRecord("first", n, FAILED)),
      RangeError,
      `node step n is not dense: node 'first' step ${n}, expected 2`,
    );
  }
  expect(log.nextSequence).toBe(before);
  expect(state.nodeSteps.get("first")).toEqual([FAILED]);
});

test("a step whose n isn't the node's next is refused on replay", () => {
  const { log } = resumedWithSteps();
  const records = log.records().map((envelope) =>
    envelope.record instanceof NodeStepRecord && envelope.record.n === 2
      ? new RecordEnvelope(envelope.schema, envelope.runId, envelope.seq, new NodeStepRecord("first", 3, envelope.record.step), envelope.at)
      : envelope,
  );

  expectError(
    () => replayRecords(new RunState(fanoutFlow()), records, { growBuilder: fragmentBuilder }),
    RangeError,
    "node step n is not dense: node 'first' step 3, expected 2",
  );
});

test("a replay reproduces the steps and starts", () => {
  const { state, log } = resumedWithSteps();

  const replayed = replayRecords(new RunState(fanoutFlow()), log.records(), { growBuilder: fragmentBuilder });

  expect(replayed.nodeSteps).toEqual(state.nodeSteps);
  expect(replayed.nodeStarts).toEqual(state.nodeStarts);
  expect(state.nodeStarts).toEqual(new Map([["first", 2]]));
  expect(replayed.nextSequence).toBe(state.nextSequence);
});

test("a learnable replay reproduces the steps, starts and trace", () => {
  const { state, log } = resumedWithSteps({ learnable: true });

  const replayed = replayRecords(new RunState(fanoutFlow()), log.records(), { growBuilder: fragmentBuilder });

  expect(replayed.nodeSteps).toEqual(state.nodeSteps);
  expect(replayed.nodeStarts).toEqual(state.nodeStarts);
  expect(state.nodeStarts).toEqual(new Map([["first", 2]]));
  expect(replayed.trace).not.toBeNull();
  expect(replayed.trace!.nextSequence).toBe(state.trace!.nextSequence);
  expect(state.trace!.nextSequence).toBe(state.nextSequence);
});

test("a replay without steps has no steps", () => {
  const { log } = firstInFlight();

  const replayed = replayRecords(new RunState(fanoutFlow()), log.records(), { growBuilder: fragmentBuilder });

  expect(replayed.nodeSteps).toEqual(new Map());
  expect(replayed.nodeStarts).toEqual(new Map([["first", 1]]));
});

test("a learnable run accepts a step with an empty trace delta", () => {
  const { state, log } = firstInFlight({ learnable: true });
  expect(state.trace).not.toBeNull();

  const prepared = prepareNodeStep(state, "first", FAILED);
  const delta = prepared._plan.traceDelta;
  expect(delta).not.toBeNull();
  expect(delta!.started).toBeNull();
  expect(delta!.resumed).toBe(false);
  expect(delta!.occurrence).toBeNull();
  commitPrepared(log, state, prepared);
  expect(state.nodeSteps).toEqual(new Map([["first", [FAILED]]]));
  expect(state.trace!.nextSequence).toBe(state.nextSequence);

  commitPrepared(
    log,
    state,
    prepareOutput(log, state, "first", { done: true }, {
      forwardCapture: forwardCapture(state.flow.nodes.get("first")!, {}),
    }),
  );
  expect(state.nodeSteps).toEqual(new Map());
});

test("a frontier commit refuses a step record with a start delta", () => {
  const { state, log } = firstInFlight();
  const prepared = prepareNodeStep(state, "first", FAILED);
  const forged = new PreparedTransition(
    prepared.record,
    prepared.expectedRunId,
    prepared.expectedSequence,
    Object.freeze({ ...prepared._plan, delta: Object.freeze({ kind: "started", nodeId: "first" }) }) as never,
    prepared.observers,
  );

  expectError(() => commitPrepared(log, state, forged), TypeError, "frontier record and prevalidated delta disagree");
  expect(state.nodeSteps).toEqual(new Map());
});

test.each(["missing-start", "orphan-steps", "zero-count", "non-object-step"] as const)(
  "broken step state is refused before append: %s",
  (corrupt) => {
    const { state, log } = firstInFlight();
    commitPrepared(log, state, prepareNodeStep(state, "first", FAILED));
    const prepared = prepareOutput(log, state, "first", { done: true });
    if (corrupt === "missing-start") state.nodeStarts.delete("first");
    else if (corrupt === "orphan-steps") state.nodeSteps.set("second", [FAILED]);
    else if (corrupt === "zero-count") state.nodeStarts.set("first", 0);
    else state.nodeSteps.set("first", ["text" as never]);
    const before = log.nextSequence;

    expectError(
      () => commitPrepared(log, state, prepared),
      RangeError,
      "node starts, steps and pending commits must match the unsettled invocations",
    );

    expect(log.nextSequence).toBe(before);
    expect(state.inFlight.has("first")).toBe(true);
  },
);
