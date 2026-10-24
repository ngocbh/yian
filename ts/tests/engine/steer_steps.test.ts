/**
 * The `steer` node step: the reducer's refusals, the waits fold, the inbox store's lifetime, and
 * replay's attached values. TS-only: Python yian has no inboxes.
 */
import { expect, test } from "bun:test";

import { Parameter, ValueContract } from "../../src/nodes/contracts.ts";
import type { RunLog } from "../../src/runtime/log.ts";
import {
  commitPrepared,
  prepareNodeStarted,
  prepareNodeStep,
  prepareSend,
  type PreparedTransition,
  type RunState,
} from "../../src/runtime/state.ts";
import { availableUnread } from "../../src/runtime/waits.ts";
import type { JsonObject } from "../../src/typesys/json.ts";
import { inboxType, Type, ValueKind } from "../../src/typesys/values.ts";
import { expectUnchanged, FINGERPRINT, FuncNode, snapshot } from "./run_state_fixtures.ts";
import { MSGS, paramFlow, started } from "./waits_fixtures.ts";

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

const steer = (from: readonly number[], inbox = "msgs"): JsonObject => ({ kind: "steer", inbox, from: [...from] });

test("a steer step consumes the whole unread queue; the values stay in the store", () => {
  const { log, state, s1, s2 } = steering();
  commit(log, state, prepareNodeStep(state, "chat", steer([s1, s2])));
  expect(availableUnread(state.waits, "msgs")).toEqual([]);
  expect([state.inboxEvents.has(s1), state.inboxEvents.has(s2)]).toEqual([true, true]);
});

test.each<[string, (s1: number, s2: number) => [string, JsonObject], string]>([
  ["an empty from", () => ["chat", steer([])], "a steer step consumes at least one event"],
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
