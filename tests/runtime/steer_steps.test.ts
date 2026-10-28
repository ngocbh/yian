/**
 * The `steer` node step: the reducer's refusals, the waits fold, the inbox store's lifetime, and
 * replay's attached values.
 */
import { expect, test } from "bun:test";

import { END_ID, START_ID } from "../../src/compile/model.ts";
import { RunSucceeded } from "../../src/events.ts";
import { Node, NodeKind, type NodeCaps, type NodeResult, Output, type SteerInbox, type SteerRead } from "../../src/nodes/base.ts";
import { Parameter, ValueContract } from "../../src/nodes/contracts.ts";
import type { FlowEngine } from "../../src/runtime/engine.ts";
import { NodeStepRecord, NodeWaitingRecord, type RunLog } from "../../src/runtime/log.ts";
import {
  commitPrepared,
  prepareInboxClosed,
  prepareNodeFailed,
  prepareNodeStarted,
  prepareNodeStep,
  prepareNodeSucceeded,
  prepareNodeWaiting,
  prepareSend,
  prepareTakeCommit,
  type PreparedTransition,
  type RunState,
  stepsWithSteerValues,
} from "../../src/runtime/state.ts";
import { availableUnread } from "../../src/runtime/waits.ts";
import type { JsonObject } from "../../src/typesys/json.ts";
import { inboxType, Type, ValueKind } from "../../src/typesys/values.ts";
import { expectUnchanged, FINGERPRINT, FuncNode, nodeFailure, snapshot } from "./run_state_fixtures.ts";
import { FakeHost, FakeTake, inboxEngine, inboxFlow } from "./inbox_fixtures.ts";
import { call, MSGS, paramFlow, replayed, started, succeed } from "./waits_fixtures.ts";

/** A second inbox param, `other: Inbox(str)`. */
const OTHER = new Parameter("other", { contract: new ValueContract(inboxType(Type.scalar(ValueKind.STRING))) });

/** A node that reads a steer inbox and records steps; the reducer only sees its traits. */
class SteerNode extends FuncNode {
  override readsSteerInbox = true;
  override readonly needsSteps = true;
}

/** Prepare and commit `prepared`, returning its seq. */
function commit(log: RunLog, state: RunState, prepared: PreparedTransition): number {
  commitPrepared(log, state, prepared);
  return prepared.expectedSequence;
}

/** A run with `chat` (a steer reader) and `f` (a plain node) started, and "a", "b" sent to `msgs`. */
function steering(params: readonly Parameter[] = [MSGS]) {
  const flow = paramFlow(params, new SteerNode("chat", () => "ok"), new FuncNode("f", () => 1));
  const run = started(flow, { fingerprint: FINGERPRINT });
  const { log, state } = run;
  commitPrepared(log, state, prepareNodeStarted(state, "chat"));
  commitPrepared(log, state, prepareNodeStarted(state, "f"));
  const s1 = commit(log, state, prepareSend(state, "msgs", "a"));
  const s2 = commit(log, state, prepareSend(state, "msgs", "b"));
  return { ...run, s1, s2 };
}

/** Return the RangeError message `prepare` throws, after checking it changed nothing. */
function refusal(state: RunState, prepare: () => unknown): string {
  const before = snapshot(state);
  let caught: unknown;
  try {
    prepare();
  } catch (error) {
    caught = error;
  }
  expect(caught).toBeInstanceOf(RangeError);
  expectUnchanged(state, before);
  return (caught as Error).message;
}

const steer = (from: readonly number[], inbox = "msgs"): JsonObject => ({ kind: "steer", inbox, from: from.map(BigInt) });

test("a steer step consumes the whole unread queue; the values stay in the store", () => {
  const { log, state, s1, s2 } = steering();
  commit(log, state, prepareNodeStep(state, "chat", steer([s1, s2])));
  expect(availableUnread(state.waits, "msgs")).toEqual([]);
  expect([state.inboxEvents.has(s1), state.inboxEvents.has(s2)]).toEqual([true, true]);
});

test.each<[string, (s1: number, s2: number) => [string, JsonObject], string]>([
  ["an empty from", () => ["chat", steer([])], "a steer step consumes at least one event"],
  [
    "seqs that aren't ints",
    (s1, s2) => ["chat", { kind: "steer", inbox: "msgs", from: [s1, s2] }],
    "a steer step's from must be a list of send seqs (ints)",
  ],
  [
    "part of the queue",
    (s1, s2) => ["chat", steer([s1])],
    "a steer step must consume the whole unread queue [S1, S2], found [S1]",
  ],
  [
    "the queue out of order",
    (s1, s2) => ["chat", steer([s2, s1])],
    "a steer step must consume the whole unread queue [S1, S2], found [S2, S1]",
  ],
  ["an undeclared inbox", (s1, s2) => ["chat", steer([s1, s2], "nope")], "steer inbox 'nope' is no inbox the START declares"],
  [
    "an extra key",
    (s1, s2) => ["chat", { ...steer([s1, s2]), values: [] }],
    "a steer step holds only kind, inbox and from",
  ],
  ["a node that reads no steer inbox", (s1, s2) => ["f", steer([s1, s2])], "a steer step on node 'f', which reads no steer inbox"],
])("a steer step with %s is refused", (_label, build, message) => {
  const { state, s1, s2 } = steering();
  const [node, step] = build(s1, s2);
  const expected = message.replaceAll("S1", String(s1)).replaceAll("S2", String(s2));
  expect(refusal(state, () => prepareNodeStep(state, node, step))).toBe(expected);
});

test("a node reads one steer inbox", () => {
  const { log, state, s1, s2 } = steering([MSGS, OTHER]);
  commit(log, state, prepareNodeStep(state, "chat", steer([s1, s2])));
  const s3 = commit(log, state, prepareSend(state, "other", "c"));
  expect(refusal(state, () => prepareNodeStep(state, "chat", steer([s3], "other")))).toBe(
    "node 'chat' reads steer inbox 'msgs', not 'other'",
  );
});

/** A run with `chat` (a steer reader on `msgs`) and `take` (a take on `msgs`) started, and "a", "b" sent. */
function competing() {
  const flow = paramFlow([MSGS], new SteerNode("chat", () => "ok"), new FakeTake("take", "msgs"));
  const run = started(flow, { fingerprint: FINGERPRINT });
  const { log, state } = run;
  commitPrepared(log, state, prepareNodeStarted(state, "chat"));
  commitPrepared(log, state, prepareNodeStarted(state, "take"));
  const s1 = commit(log, state, prepareSend(state, "msgs", "a"));
  const s2 = commit(log, state, prepareSend(state, "msgs", "b"));
  return { ...run, s1, s2 };
}

test("a take that commits first consumes its event; the steer read then gets only the rest", () => {
  const { log, state, s1, s2 } = competing();
  commit(log, state, prepareTakeCommit(state, "take", { inbox: "msgs", events: [s1], closed: false }));
  expect(refusal(state, () => prepareNodeStep(state, "chat", steer([s1, s2])))).toBe(
    `a steer step must consume the whole unread queue [${s2}], found [${s1}, ${s2}]`,
  );
  commit(log, state, prepareNodeStep(state, "chat", steer([s2])));
  expect(stepsWithSteerValues(state, "chat").map((step) => step["values"])).toEqual([["b"]]);
  commitPrepared(log, state, prepareNodeSucceeded(state, "take"));
  expect(state.pool.get("take")).toEqual({ events: ["a"], closed: false });
  expect(availableUnread(state.waits, "msgs")).toEqual([]);
});

test("a steer read that lands first consumes every event; a take can't commit one of them", () => {
  const { log, state, s1, s2 } = competing();
  commit(log, state, prepareNodeStep(state, "chat", steer([s1, s2])));
  refusal(state, () => prepareTakeCommit(state, "take", { inbox: "msgs", events: [s1], closed: false }));
  expect(availableUnread(state.waits, "msgs")).toEqual([]);
  const s3 = commit(log, state, prepareSend(state, "msgs", "c"));
  commit(log, state, prepareTakeCommit(state, "take", { inbox: "msgs", events: [s3], closed: false }));
  commitPrepared(log, state, prepareNodeSucceeded(state, "take"));
  expect(state.pool.get("take")).toEqual({ events: ["c"], closed: false });
  expect(stepsWithSteerValues(state, "chat").map((step) => step["values"])).toEqual([["a", "b"]]);
});

test("a steer read's values stay until its node succeeds, then go", () => {
  const { log, state, s1, s2 } = steering();
  commit(log, state, prepareNodeStep(state, "chat", steer([s1, s2])));
  succeed(log, state, "chat", "ok");
  expect([state.inboxEvents.has(s1), state.inboxEvents.has(s2)]).toEqual([false, false]);
});

test("a steer read's values go when its node fails", () => {
  const { log, state, s1, s2 } = steering();
  commit(log, state, prepareNodeStep(state, "chat", steer([s1, s2])));
  commitPrepared(log, state, prepareNodeFailed(state, nodeFailure("chat")));
  expect([state.inboxEvents.has(s1), state.inboxEvents.has(s2)]).toEqual([false, false]);
});

test("stepsWithSteerValues attaches each steer step's values to a copy", () => {
  const { log, state, s1, s2 } = steering();
  commit(log, state, prepareNodeStep(state, "chat", steer([s1, s2])));
  expect(stepsWithSteerValues(state, "chat")).toEqual([
    { kind: "steer", inbox: "msgs", from: [BigInt(s1), BigInt(s2)], values: ["a", "b"] },
  ]);
  expect(Object.hasOwn(state.nodeSteps.get("chat")![0]!, "values")).toBe(false);
  expect(stepsWithSteerValues(state, "f")).toEqual([]);
});

test("replay rebuilds the same store and the same step copies", () => {
  const { log, state, path, s1, s2 } = steering();
  commit(log, state, prepareNodeStep(state, "chat", steer([s1, s2])));
  const back = replayed(state.flow, path);
  expect([...back.inboxEvents.keys()]).toEqual([...state.inboxEvents.keys()]);
  expect(stepsWithSteerValues(back, "chat")).toEqual([
    { kind: "steer", inbox: "msgs", from: [BigInt(s1), BigInt(s2)], values: ["a", "b"] },
  ]);
});

test("a learnable occurrence's steps carry the steer values", () => {
  const flow = paramFlow([MSGS], new SteerNode("chat", () => "ok"));
  const { log, state } = started(flow, { learnable: true, fingerprint: FINGERPRINT });
  commitPrepared(log, state, prepareNodeStarted(state, "chat"));
  const s1 = commit(log, state, prepareSend(state, "msgs", "a"));
  commit(log, state, prepareNodeStep(state, "chat", steer([s1])));
  succeed(log, state, "chat", "ok");
  const occurrence = state.trace!.occurrences.find((item) => item.nodeId === "chat")!;
  expect(occurrence.steps).toEqual([{ kind: "steer", inbox: "msgs", from: [BigInt(s1)], values: ["a"] }]);
});

/** What one `SteerProbe` evaluation saw. */
interface ProbeSeen {
  queuedBefore: boolean;
  first: SteerRead | null;
  second: SteerRead | null;
  queuedAfter: boolean;
}

/** A node that opens steer inbox `inbox`, reads it twice and returns; or, with `pend`, waits on a host call after opening it. */
class SteerProbe extends Node {
  readonly kind = NodeKind.CODE;
  override readsSteerInbox = true;
  override readonly needsSteps = true;
  readonly inbox: string;
  seen: ProbeSeen | null = null;
  opened: SteerInbox | null = null;

  constructor(id: string, inbox = "m") {
    super(id);
    this.inbox = inbox;
  }

  override forward(_inputs: Record<string, unknown>, caps: NodeCaps): NodeResult {
    const steer = caps.openSteerInbox!(this.inbox);
    this.opened = steer;
    const queuedBefore = steer.queued();
    const first = steer.read();
    const second = steer.read();
    this.seen = { queuedBefore, first, second, queuedAfter: steer.queued() };
    return new Output("done");
  }
}

/** A host call node that reads a steer inbox: it opens `m` and waits on its call. */
class SteerHost extends FakeHost {
  override readsSteerInbox = true;

  override forward(inputs: Record<string, unknown>, caps: NodeCaps): NodeResult {
    caps.openSteerInbox!("m");
    return super.forward(inputs, caps);
  }
}

/** Commit a send of `event` to `inbox` on `run`'s state before it runs, returning its seq. */
function sendBefore(run: FlowEngine, event: string, inbox = "m"): number {
  return commit(run.log, run.state, prepareSend(run.state, inbox, event));
}

/** The `node_step` bodies of `log`, in order. */
const stepBodies = (log: RunLog) =>
  log.records().flatMap((entry) => (entry.record instanceof NodeStepRecord ? [entry.record.step] : []));

test("a read records one steer step for everything queued and returns its values", async () => {
  const probe = new SteerProbe("chat");
  const run = inboxEngine(inboxFlow(["m"], [probe], [[START_ID, "chat"], ["chat", END_ID]]));
  const s1 = sendBefore(run, "a");
  const s2 = sendBefore(run, "b");
  await run.run();
  expect(probe.seen).toEqual({
    queuedBefore: true,
    first: { n: 1, from: [s1, s2], values: ["a", "b"] },
    second: null,
    queuedAfter: false,
  });
  expect(stepBodies(run.log)).toEqual([{ kind: "steer", inbox: "m", from: [BigInt(s1), BigInt(s2)] }]);
  expect(run.state.result).toBeInstanceOf(RunSucceeded);
});

test("nothing queued: no read, no step", async () => {
  const probe = new SteerProbe("chat");
  const run = inboxEngine(inboxFlow(["m"], [probe], [[START_ID, "chat"], ["chat", END_ID]]));
  await run.run();
  expect(probe.seen!.first).toBeNull();
  expect(stepBodies(run.log)).toEqual([]);
});

test("a closed inbox still gives what is queued", async () => {
  const probe = new SteerProbe("chat");
  const run = inboxEngine(inboxFlow(["m"], [probe], [[START_ID, "chat"], ["chat", END_ID]]));
  sendBefore(run, "a");
  commit(run.log, run.state, prepareInboxClosed(run.state, "m")!);
  await run.run();
  expect(probe.seen!.first!.values).toEqual(["a"]);
});

test("opening an inbox the START doesn't declare is a host fault", async () => {
  const probe = new SteerProbe("chat", "nope");
  const run = inboxEngine(inboxFlow(["m"], [probe], [[START_ID, "chat"], ["chat", END_ID]]));
  const error = await run.run().then(
    () => null,
    (caught: unknown) => caught,
  );
  expect(error).toBeInstanceOf(RangeError);
  expect((error as Error).message).toBe("steer inbox 'nope' is no inbox the START declares");
});

test("a waiting node that opened its steer inbox waits with it", async () => {
  const host = new SteerHost("h");
  const run = inboxEngine(inboxFlow(["m"], [host], [[START_ID, "h"], ["h", END_ID]]));
  await run.run();
  const waiting = run.log.records().find((entry) => entry.record instanceof NodeWaitingRecord)!.record as NodeWaitingRecord;
  expect(waiting.steerInbox).toBe("m");
  expect(run.state.waits.waiting.get("h")!.steerInbox).toBe("m");
});

test("node_waiting steer_inbox is refused on a node that reads none, and for an undeclared inbox", () => {
  const flow = paramFlow([MSGS], new FakeHost("h"), new SteerHost("s"));
  const { log, state } = started(flow, { fingerprint: FINGERPRINT });
  for (const node of ["h", "s"]) {
    commitPrepared(log, state, prepareNodeStarted(state, node));
    const pending = { kind: "pending", call_id: call(node), tool: "lookup", arguments: {} };
    commitPrepared(log, state, prepareNodeStep(state, node, pending));
  }
  expect(refusal(state, () => prepareNodeWaiting(state, "h", { callIds: [call("h")], steerInbox: "msgs" }))).toBe(
    "NodeWaiting steer_inbox on node 'h', which reads no steer inbox",
  );
  expect(refusal(state, () => prepareNodeWaiting(state, "s", { callIds: [call("s")], steerInbox: "nope" }))).toBe(
    "NodeWaiting steer_inbox 'nope' is no inbox the START declares",
  );
});
