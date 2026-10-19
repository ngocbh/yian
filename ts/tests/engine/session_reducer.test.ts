/**
 * Resume of unfinished agent invocations through the append-before-apply reducer: a port of
 * `tests/engine/test_session_reducer.py`.
 *
 * Ported: `test_forged_resume_plan_cannot_change_the_retry_frontier` (the forged transition
 * spreads the frozen plan and its delta; the error is the `RangeError` `frontier resume delta
 * disagrees with its record`, and the snapshot is the shared reducer snapshot).
 *
 * Changed: `test_active_conversation_preparation_is_not_exported`: the module namespace has no
 * `prepareConversationRecord`; Python's `__all__` half has no counterpart.
 *
 * Later (T4-5): `test_resume_allows_memory_free_unfinished_invocations` and
 * `test_resume_retries_every_unfinished_invocation`, which replay through `replayRecords`.
 */

import { expect, test } from "bun:test";

import { CompiledFlow, Edge, END_ID, START_ID } from "../../src/compile/model.ts";
import { DeclSite, MemoryScope, MemorySnapshotEnvelope, MemorySpec, MemoryStack } from "../../src/memory/index.ts";
import type { Node } from "../../src/nodes/base.ts";
import { EndNode } from "../../src/nodes/end/node.ts";
import { StartNode } from "../../src/nodes/start/node.ts";
import { MemoryRunLog } from "../../src/runtime/log.ts";
import * as reducer from "../../src/runtime/state.ts";
import { VariablePool } from "../../src/typesys/pool.ts";
import { expectUnchanged, FuncNode, snapshot } from "./run_state_fixtures.ts";

/** Return `START -> agent -> END` where `agent` has the memory scope `session`. */
function sessionFlow(): CompiledFlow {
  const agent = new FuncNode("agent", () => {
    throw new Error("replay invoked a node");
  });
  agent.memoryScope = new MemoryScope(["agent"], { memories: ["session"] });
  return CompiledFlow.fromParts(
    new Map<string, Node>([
      [START_ID, new StartNode(START_ID)],
      ["agent", agent],
      [END_ID, EndNode.value(END_ID, { outputName: null })],
    ]),
    [new Edge("start-agent", START_ID, "agent"), new Edge("agent-end", "agent", END_ID)],
  );
}

/** Return a state and log with the memory `session` recorded and `agent` in flight. */
function started(): { state: reducer.RunState; log: MemoryRunLog } {
  const flow = sessionFlow();
  const state = new reducer.RunState(flow);
  const pool = new VariablePool(flow.startId);
  pool.set(flow.startId, {});
  pool.addSystem("run_id", "run");
  const log = new MemoryRunLog("run");
  reducer.commitPrepared(
    log,
    state,
    reducer.prepareRunStarted(state, {
      runId: "run",
      flowFingerprint: "a".repeat(64),
      initialPool: pool,
      startWiringOverride: {},
      effectiveLlmByStaticPath: {},
      numWorkers: 0,
      memories: [new MemorySpec("session", "session", {}, [new DeclSite([], ["root"], "agent")], true)],
      memoryStacks: [new MemoryStack(["agent"], ["session"])],
      memorySnapshots: [new MemorySnapshotEnvelope("session", {})],
    }),
  );
  reducer.commitPrepared(log, state, reducer.prepareNodeStarted(state, START_ID));
  reducer.commitPrepared(log, state, reducer.prepareNodeSucceeded(state, START_ID, {}));
  reducer.commitPrepared(log, state, reducer.prepareNodeStarted(state, "agent"));
  return { state, log };
}

test("a forged resume plan can't change the retry frontier", () => {
  const { state, log } = started();
  const prepared = reducer.prepareRunResumed(state, { numWorkers: 0 });
  const plan = prepared._plan;
  if (plan.kind !== "frontier") throw new Error(`expected a frontier plan, found ${plan.kind}`);
  const forged = new reducer.PreparedTransition(
    prepared.record,
    prepared.expectedRunId,
    prepared.expectedSequence,
    Object.freeze({ ...plan, delta: Object.freeze({ ...plan.delta, retryNodeIds: Object.freeze([]) }) }),
    prepared.observers,
  );
  const before = snapshot(state);

  let caught: unknown;
  try {
    reducer.commitPrepared(log, state, forged);
  } catch (error) {
    caught = error;
  }

  expect((caught as Error).constructor).toBe(RangeError);
  expect((caught as Error).message).toBe("frontier resume delta disagrees with its record");
  expectUnchanged(state, before);
  expect([...state.inFlight]).toEqual(["agent"]);
});

test("active conversation preparation isn't exported", () => {
  expect("prepareConversationRecord" in reducer).toBe(false);
});
