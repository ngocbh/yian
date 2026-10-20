/**
 * Node steps: a port of `tests/engine/test_node_steps_log.py`, the `NodeStepRecord` a running node
 * appends and how the run-state reducer folds it. Python's `ValueError` is a `RangeError` here.
 *
 * Ported: `test_node_step_record_detaches_the_callers_step`,
 * `test_node_step_record_rejects_invalid_values` (list, text, null, NaN, a class instance, a
 * `Set`, an empty and a non-text node id; each with its exact error).
 *
 * Ported, the wire tests: `test_node_step_round_trips_under_its_wire_kind` (schema 10),
 * `test_node_step_wire_payload_rejects_invalid_shapes` (list, text, empty node id, missing step,
 * extra field; each with its exact class and message),
 * `test_node_step_wire_payload_rejects_non_finite_numbers` (`invalid run-record JSON`, caused by
 * the reader's `NaN is not JSON`).
 *
 * Ported, the reducer tests: `test_prepare_node_step_requires_an_in_flight_node` (3 rows),
 * `test_prepare_node_step_rejects_an_uninitialized_state`,
 * `test_prepare_node_step_rejects_a_step_past_the_nesting_limit`,
 * `test_node_starts_count_started_records`, `test_resume_keeps_steps_and_a_resumed_start_counts_two`,
 * `test_success_deletes_the_nodes_steps_and_starts`, `test_prepare_node_step_rejects_a_terminal_state`,
 * `test_failure_deletes_the_nodes_steps_and_starts`, `test_route_deletes_the_nodes_steps_and_starts`,
 * `test_expansion_deletes_the_spawners_steps_and_starts` (on the shared `initializedSpawner` and
 * `fragment`),
 * `test_learnable_run_accepts_a_step_with_an_empty_trace_delta` and
 * `test_broken_step_state_is_rejected_before_append` (4 rows). Each rejected row names its one
 * class and full message, where Python matches a part of the message or accepts either class.
 *
 * Ported, the replay tests (through `replayRecords` on a fresh `fanoutFlow()`; the log never
 * grows, so the builder is never called): `test_replay_reproduces_steps_and_starts`,
 * `test_learnable_replay_reproduces_steps_starts_and_trace`,
 * `test_replay_without_steps_has_no_steps`.
 *
 * TS-only: a step writes `-0` as `0`; a `FileRef` in a step is refused; a wire step keeps
 * yian's infinity tag as `Infinity`.
 *
 * Changed:
 * - the `infinity` row of `test_node_step_record_rejects_invalid_values`: a step keeps
 *   `±Infinity` (a TS yian value).
 * - `test_node_step_too_deep_for_python_is_a_value_error`: a 5000-deep step is accepted (the TS
 *   codec is iterative).
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
 *
 * Later (T3-22, it needs `build_grow` and `LoopNode.run`):
 * `test_loop_body_steps_go_at_the_outcome_and_stay_gone_after_the_prune`.
 */

import { expect, test } from "bun:test";

import { START_ID } from "../../src/compile/model.ts";
import { current } from "../../src/defaults.ts";
import { Grow } from "../../src/nodes/base.ts";
import {
  decodeRecord,
  encodeRecord,
  MemoryRunLog,
  NodeStepRecord,
  RecordEnvelope,
  SCHEMA_VERSION,
} from "../../src/runtime/log.ts";
import {
  commitPrepared,
  PreparedTransition,
  prepareExpanded,
  prepareNodeFailed,
  prepareNodeRouted,
  prepareNodeStarted,
  prepareNodeStep,
  prepareNodeSucceeded,
  prepareRunResumed,
  replayRecords,
  RunState,
} from "../../src/runtime/state.ts";
import { FileRef, TypeCheckError } from "../../src/typesys/index.ts";
import { JSONDecodeError, parseJson } from "../../src/typesys/json.ts";
import {
  commitStartSuccess,
  fanoutFlow,
  forwardCapture,
  fragmentBuilder,
  initializedAndStarted,
  initializedSpawner,
  nodeFailure,
  routeInFlight,
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
  event: "failed",
  attempt: 1n,
  error: { type: "TimeoutError", message: "slow" },
  retry_in: 1.5,
};

test("a node step round trips under its wire kind", () => {
  const envelope = new RecordEnvelope(SCHEMA_VERSION, "run-1", 3, new NodeStepRecord("a", FAILED));

  const encoded = encodeRecord(envelope);
  const decoded = decodeRecord(encoded);

  expect(decoded).toEqual(envelope);
  expect(encodeRecord(decoded)).toBe(encoded);
  const wire = parseJson(encoded, "codec") as Record<string, unknown>;
  expect(wire["kind"]).toBe("node_step");
  expect(wire["schema"]).toBe(10n);
  expect(wire["payload"]).toEqual({ node_id: "a", step: FAILED });
});

test("a wire step keeps yian's infinity tag", () => {
  const text =
    '{"kind":"node_step","payload":{"node_id":"a","step":{"high":{"$float":"inf"}}},' +
    '"run_id":"run-1","schema":10,"seq":0}';
  const record = decodeRecord(text).record as NodeStepRecord;
  expect(record.step).toEqual({ high: Infinity });
  expect(encodeRecord(decodeRecord(text))).toBe(text);
});

test.each<[string, unknown, new (...args: never[]) => Error, string]>([
  ["list", { node_id: "a", step: [1] }, TypeError, "node_step payload step must be an object"],
  ["text", { node_id: "a", step: "text" }, TypeError, "node_step payload step must be an object"],
  ["empty-node-id", { node_id: "", step: {} }, RangeError, "node_step payload node_id must be a non-empty string"],
  ["missing-step", { node_id: "a" }, RangeError, 'invalid node_step payload fields; missing=["step"], extra=[]'],
  [
    "extra-field",
    { node_id: "a", step: {}, extra: true },
    RangeError,
    'invalid node_step payload fields; missing=[], extra=["extra"]',
  ],
])("a node step wire payload rejects an invalid shape: %s", (_name, payload, cls, message) => {
  const text = JSON.stringify({ schema: SCHEMA_VERSION, run_id: "run-1", seq: 0, kind: "node_step", payload });
  expectError(() => decodeRecord(text), cls, message);
});

test("a node step wire payload rejects non-finite numbers", () => {
  const text =
    '{"schema":10,"run_id":"run-1","seq":0,"kind":"node_step","payload":{"node_id":"a","step":{"retry_in":NaN}}}';
  const caught = expectError(() => decodeRecord(text), RangeError, "invalid run-record JSON") as Error;
  expect((caught.cause as Error).constructor).toBe(JSONDecodeError);
  expect((caught.cause as Error).message).toStartWith("NaN is not JSON");
});

test("a node step record detaches the caller's step", () => {
  const step = { event: "started", attempt: 2n, tags: ["x", "y"] };
  const record = new NodeStepRecord("a", step);
  step.attempt = 99n;
  step.tags.push("z");

  expect(record.step).toEqual({ attempt: 2n, event: "started", tags: ["x", "y"] });
  expect(Object.keys(record.step)).toEqual(["attempt", "event", "tags"]);
});

test.each<[string, unknown, unknown, new (...args: never[]) => Error, string, (string | number)[] | null]>([
  ["list", "a", ["not", "an", "object"], TypeError, "node step must be a JSON object", null],
  ["text", "a", "text", TypeError, "node step must be a JSON object", null],
  ["null", "a", null, TypeError, "node step must be a JSON object", null],
  ["NaN", "a", { retry_in: NaN }, TypeCheckError, "NaN is not a yian value", ["retry_in"]],
  ["a class instance", "a", { when: new Thing() }, TypeCheckError, "<Thing object> is not a JSON value", ["when"]],
  ["a Set", "a", { items: new Set([1, 2]) }, TypeCheckError, "<Set object> is not a JSON value", ["items"]],
  [
    "a FileRef",
    "a",
    { file: new FileRef({ uri: "memory://answer" }) },
    TypeCheckError,
    '{"uri":"memory://answer","mime":null,"name":null} is not a JSON value',
    ["file"],
  ],
  ["an empty node id", "", { event: "started" }, RangeError, "node_id must be non-empty", null],
  ["a non-text node id", 1, { event: "started" }, TypeError, "node_id must be a string", null],
])("a node step record rejects %s", (_name, nodeId, step, cls, message, path) => {
  const caught = expectError(() => new NodeStepRecord(nodeId as never, step as never), cls, message);
  if (path !== null) expect((caught as TypeCheckError).path).toEqual(path);
});

test("a node step record keeps infinite floats and writes -0 as 0", () => {
  const record = new NodeStepRecord("a", { high: Infinity, low: [-Infinity], zero: -0 });

  expect(record.step).toEqual({ high: Infinity, low: [-Infinity], zero: 0 });
  expect(Object.is(record.step["zero"], 0)).toBe(true);
});

test("a node step too deep for Python is accepted", () => {
  const record = new NodeStepRecord("a", nested(5000));

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
  const started = { event: "started", attempt: 2n };

  const first = prepareNodeStep(state, "first", FAILED);
  expect(first.observers).toEqual([]);
  expect(first.record).toStrictEqual(new NodeStepRecord("first", FAILED));
  const plan = first._plan;
  if (plan.kind !== "frontier") throw new Error(`expected a frontier plan, found ${plan.kind}`);
  expect(plan.delta).toEqual({ kind: "step", nodeId: "first", step: FAILED });
  expect(state.nodeSteps.has("first")).toBe(false);
  expect(commitPrepared(log, state, first)).toEqual([]);
  expect(commitPrepared(log, state, prepareNodeStep(state, "first", started))).toEqual([]);

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
  const step = { event: "failed", error: { type: "E", message: "m" } };

  const prepared = prepareNodeStep(state, "first", step);
  commitPrepared(log, state, prepared);
  step.error.type = "changed";
  expect(() => {
    ((prepared.record as NodeStepRecord).step["error"] as Record<string, unknown>)["type"] = "changed";
  }).toThrow(TypeError);

  expect(state.nodeSteps.get("first")).toEqual([{ error: { message: "m", type: "E" }, event: "failed" }]);
});

test.each([
  ["second", "NodeStep node 'second' is not in-flight"],
  ["missing", "NodeStep node 'missing' does not exist in the flow"],
  [START_ID, `NodeStep node '${START_ID}' must have UNKNOWN disposition`],
] as const)("prepareNodeStep requires an in-flight node: %s", (nodeId, message) => {
  const { state, log } = firstInFlight();
  const before = log.nextSequence;

  expectError(() => prepareNodeStep(state, nodeId, { event: "started" }), RangeError, message);

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
    () => prepareNodeStep(state, "second", { event: "started" }),
    RangeError,
    "terminal run state cannot prepare another graph transition",
  );
});

test("prepareNodeStep refuses an uninitialized state", () => {
  expectError(
    () => prepareNodeStep(new RunState(fanoutFlow()), START_ID, { event: "started" }),
    RangeError,
    "run state is not initialized",
  );
});

test("prepareNodeStep refuses a step past the nesting limit", () => {
  const { state } = firstInFlight();
  const limit = Number(current().maxValueNesting);

  prepareNodeStep(state, "first", nested(limit));
  const caught = expectError(
    () => prepareNodeStep(state, "first", nested(limit + 1)),
    TypeCheckError,
    `node step at ${Array(limit).fill("inner").join(".")}: value nests deeper than ${limit} levels, ` +
      "the limit for a yian value (each record and list is a level)",
  );
  expect((caught as TypeCheckError).path).toEqual(Array(limit).fill("inner"));
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
    () => prepareNodeStep(state, "first", { event: "started" }),
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

  commitPrepared(log, state, prepareNodeSucceeded(state, "first", { done: true }));

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

  commitPrepared(log, state, prepareExpanded(state, "spawn", new Grow({}), fragmentBuilder));

  expect(state.nodeSteps).toEqual(new Map());
  expect(state.nodeStarts).toEqual(new Map());
  expect(state.ready).toEqual(["spawn/child/start"]);
});

/** Return `firstInFlight`'s state after a step, a resume, a second start and a second step. */
function resumedWithSteps(options: { learnable?: boolean } = {}): { state: RunState; log: MemoryRunLog } {
  const { state, log } = firstInFlight(options);
  commitPrepared(log, state, prepareNodeStep(state, "first", FAILED));
  commitPrepared(log, state, prepareRunResumed(state, { numWorkers: 0 }));
  commitPrepared(log, state, prepareNodeStarted(state, "first"));
  commitPrepared(log, state, prepareNodeStep(state, "first", { event: "started", attempt: 3n }));
  return { state, log };
}

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
    prepareNodeSucceeded(state, "first", { done: true }, {
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
    const prepared = prepareNodeSucceeded(state, "first", { done: true });
    if (corrupt === "missing-start") state.nodeStarts.delete("first");
    else if (corrupt === "orphan-steps") state.nodeSteps.set("second", [FAILED]);
    else if (corrupt === "zero-count") state.nodeStarts.set("first", 0);
    else state.nodeSteps.set("first", ["text" as never]);
    const before = log.nextSequence;

    expectError(
      () => commitPrepared(log, state, prepared),
      RangeError,
      "node starts and steps must match the unsettled invocations",
    );

    expect(log.nextSequence).toBe(before);
    expect(state.inFlight.has("first")).toBe(true);
  },
);
