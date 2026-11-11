/**
 * Node steps: the `NodeStepRecord` a running node appends and how the run-state reducer folds it.
 */

import { expect, spyOn, test } from "bun:test";

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
import { JSONDecodeError, type JsonObject, parseJson } from "../../src/typesys/json.ts";
import {
  canonicalChild,
  commitStartSuccess,
  fanoutFlow,
  fragmentBuilder,
  fragmentIds,
  initializedAndStarted,
  initializedSpawner,
  nodeFailure,
  parentFlow,
  prepareOutput,
  reducerWrite,
  routeInFlight,
  prepareGrowth,
} from "../helpers/run_state_fixtures.ts";

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
  expect(wire["schema"]).toBe(29n);
  expect(wire["payload"]).toEqual({ node_id: "a", n: 1n, step: FAILED });
});

test("a wire step keeps yian's infinity tag", () => {
  const text =
    '{"kind":"node_step","payload":{"n":1,"node_id":"a","step":{"high":{"$float":"inf"},"kind":"x"}},' +
    '"run_id":"run-1","schema":29,"seq":0}';
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
    '{"schema":29,"run_id":"run-1","seq":0,"kind":"node_step","payload":{"node_id":"a","n":1,"step":{"retry_in":NaN}}}';
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
  ["a class instance", "a", 1, { kind: "x", when: new Thing() }, TypeCheckError, "Thing is not a yian value", ["when"]],
  ["a Set", "a", 1, { kind: "x", items: new Set([1, 2]) }, TypeCheckError, "Set is not a yian value", ["items"]],
  [
    "a FileRef",
    "a",
    1,
    { kind: "x", file: new FileRef({ uri: "memory://answer" }) },
    TypeCheckError,
    '{"uri":"memory://answer","mime":null,"name":null} is not a JSON value',
    ["file"],
  ],
  ["an empty node id", "", 1, { kind: "x" }, RangeError, "node_id must be a non-empty string", null],
  ["a non-text node id", 1, 1, { kind: "x" }, TypeError, "node_id must be a non-empty string", null],
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

test("a node step 5000 levels deep is accepted", () => {
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
function firstInFlight(): { state: RunState; log: MemoryRunLog } {
  const { state, log } = initializedAndStarted(fanoutFlow());
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
  expect(plan.delta).toEqual({ kind: "step", nodeId: "first", step: FAILED, closedCalls: [] });
  expect(state.nodeSteps.has("first")).toBe(false);
  expect(commitPrepared(log, state, first)).toEqual([]);
  const second = prepareNodeStep(state, "first", started);
  expect((second.record as NodeStepRecord).n).toBe(2);
  expect(commitPrepared(log, state, second)).toEqual([]);

  expect(state.nodeSteps).toEqual(new Map([["first", [FAILED, started]]]));
  expect(state.nodeStarts.get("first")).toBe(1);
  const kinds = log.records().map((envelope) => envelope.record.constructor);
  expect(kinds.slice(-2)).toEqual([NodeStepRecord, NodeStepRecord]);
  expect(state.nextSequence).toBe(log.nextSequence);
  expect(state.ready).toEqual(["second"]);
});

test("recording a step appends to the node's step and seq lists in place", () => {
  const { state, log } = firstInFlight();
  const steps: JsonObject[] = [FAILED, { kind: "tool_attempt", event: "started", attempt: 2n }, { kind: "tool_attempt", event: "done" }];

  commitPrepared(log, state, prepareNodeStep(state, "first", steps[0]!));
  const recorded = state.nodeSteps.get("first");
  const seqs = state.nodeStepSeqs.get("first");
  for (const step of steps.slice(1)) commitPrepared(log, state, prepareNodeStep(state, "first", step));

  expect(state.nodeSteps.get("first")).toBe(recorded!);
  expect(state.nodeStepSeqs.get("first")).toBe(seqs!);
  expect(recorded).toEqual(steps);
  const stepSeqs = log.records().filter((envelope) => envelope.record instanceof NodeStepRecord).map((envelope) => envelope.seq);
  expect(seqs).toEqual(stepSeqs);
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

test("an outcome in a run that doesn't learn copies none of its node's steps", () => {
  const { state, log } = firstInFlight();
  commitPrepared(log, state, prepareNodeStep(state, "first", FAILED));
  const clone = spyOn(globalThis, "structuredClone");
  try {
    prepareNodeFailed(state, nodeFailure("first"));
    expect(clone).not.toHaveBeenCalled();
  } finally {
    clone.mockRestore();
  }
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
function resumedWithSteps(): { state: RunState; log: MemoryRunLog } {
  const { state, log } = firstInFlight();
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

test("a replay without steps has no steps", () => {
  const { log } = firstInFlight();

  const replayed = replayRecords(new RunState(fanoutFlow()), log.records(), { growBuilder: fragmentBuilder });

  expect(replayed.nodeSteps).toEqual(new Map());
  expect(replayed.nodeStarts).toEqual(new Map([["first", 1]]));
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

test.each(["missing-start", "orphan-steps", "zero-count"] as const)(
  "broken step state is refused before the next append: %s",
  (corrupt) => {
    const { state, log } = firstInFlight();
    commitPrepared(log, state, prepareNodeStep(state, "first", FAILED));
    const prepared = prepareOutput(log, state, "first", { done: true });
    if (corrupt === "missing-start") reducerWrite(state, "first", () => state.nodeStarts.delete("first"));
    else if (corrupt === "orphan-steps") reducerWrite(state, "second", () => state.nodeSteps.set("second", [FAILED]));
    else reducerWrite(state, "first", () => state.nodeStarts.set("first", 0));
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
