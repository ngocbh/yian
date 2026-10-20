/**
 * Resume of unfinished agent invocations through the append-before-apply reducer: a port of
 * `tests/engine/test_session_reducer.py`.
 *
 * Ported: `test_forged_resume_plan_cannot_change_the_retry_frontier` (the forged transition
 * spreads the frozen plan and its delta; the error is the `RangeError` `frontier resume delta
 * disagrees with its record`, and the state is checked with the shared `snapshot`
 * and `expectUnchanged`),
 * `test_resume_allows_memory_free_unfinished_invocations` and
 * `test_resume_retries_every_unfinished_invocation` (2 rows). Python's local `_replay` is the
 * local `expectReplays`: `replayRecords` over a fresh flow, compared through the shared
 * `reducerSnapshot` plus the memory table, stacks, snapshots, `learnable` and backward state, the
 * topology, each node's class and scopes, and the trace.
 *
 * Changed:
 * - `test_active_conversation_preparation_is_not_exported`: the module namespace has no
 *   `prepareConversationRecord`; Python's `__all__` half has no counterpart.
 * - The replayed trace compares through the shared `replayTraceSnapshot`, the backward handlers
 *   by key: a TS bound handler can't name its node, so Python's `(type, id, function)` triple of
 *   each handler has no counterpart.
 */

import { expect, test } from "bun:test";

import { CompiledFlow, Edge, END_ID, START_ID } from "../../src/compile/model.ts";
import { DeclSite, MemoryScope, MemorySnapshotEnvelope, MemorySpec, MemoryStack } from "../../src/memory/index.ts";
import type { Node } from "../../src/nodes/base.ts";
import { EndNode } from "../../src/nodes/end/node.ts";
import { StartNode } from "../../src/nodes/start/node.ts";
import { ValueRoute } from "../../src/expr/lineage.ts";
import { ForwardCapture } from "../../src/runtime/contracts.ts";
import { MemoryRunLog, RunResumedRecord } from "../../src/runtime/log.ts";
import * as reducer from "../../src/runtime/state.ts";
import { VariablePool } from "../../src/typesys/pool.ts";
import {
  expectUnchanged,
  FuncNode,
  reducerSnapshot,
  replayTraceSnapshot,
  snapshot,
} from "./run_state_fixtures.ts";

/**
 * Return `START -> agent -> END` where `agent` has the memory scope `session`; with `sibling`,
 * also `START -> sibling`.
 */
function sessionFlow(options: { sibling?: boolean } = {}): CompiledFlow {
  const agent = new FuncNode("agent", () => {
    throw new Error("replay invoked a node");
  });
  agent.memoryScope = new MemoryScope(["agent"], { memories: ["session"] });
  const built = CompiledFlow.fromParts(
    new Map<string, Node>([
      [START_ID, new StartNode(START_ID)],
      ["agent", agent],
      [END_ID, EndNode.value(END_ID, { outputName: null })],
    ]),
    [new Edge("start-agent", START_ID, "agent"), new Edge("agent-end", "agent", END_ID)],
  );
  if (options.sibling) {
    const sibling = new FuncNode("sibling", () => {
      throw new Error("node called");
    });
    built.addSubgraph(new Map([["sibling", sibling]]), [new Edge("start-sibling", START_ID, "sibling")], new Map([
      ["sibling", new Map()],
    ]));
  }
  return built;
}

/**
 * Return a state and log with START succeeded and `agent` in flight. The memory `session` is
 * recorded unless `enabled` is false; `learnable` and `sibling` shape the run and the flow.
 */
function started(
  options: { learnable?: boolean; sibling?: boolean; enabled?: boolean } = {},
): { state: reducer.RunState; log: MemoryRunLog } {
  const enabled = options.enabled ?? true;
  const flow = sessionFlow(options);
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
      learnable: options.learnable ?? false,
      memories: enabled
        ? [new MemorySpec("session", "session", {}, [new DeclSite([], ["root"], "agent")], true)]
        : null,
      memoryStacks: enabled ? [new MemoryStack(["agent"], ["session"])] : null,
      memorySnapshots: enabled ? [new MemorySnapshotEnvelope("session", {})] : null,
    }),
  );
  reducer.commitPrepared(log, state, reducer.prepareNodeStarted(state, START_ID));
  const forwardCapture = state.learnable ? new ForwardCapture({}, ValueRoute.record([])) : null;
  reducer.commitPrepared(log, state, reducer.prepareNodeSucceeded(state, START_ID, {}, { forwardCapture }));
  reducer.commitPrepared(log, state, reducer.prepareNodeStarted(state, "agent"));
  return { state, log };
}

/** Return `reducerSnapshot(state)` with the memory, learning and backward state. */
function replaySnapshot(state: reducer.RunState): Record<string, unknown> {
  return {
    ...reducerSnapshot(state),
    memories: state.memories,
    memoryStacks: state.memoryStacks,
    memorySnapshots: state.memorySnapshots,
    learnable: state.learnable,
    backwardState: state.backwardState,
  };
}

/** Return `state`'s topology and each node's class, scopes and `commitAs`. */
function flowSnapshot(state: reducer.RunState): Record<string, unknown> {
  const flow = state.flow;
  return {
    initialStaticIds: [...state.initialStaticIds],
    startId: flow.startId,
    endId: flow.endId,
    edges: [...flow.edges],
    wiring: [...flow.wiring],
    incoming: [...flow.incomingEdges],
    outgoing: [...flow.outgoingEdges],
    nodes: [...flow.nodes].map(([nodeId, node]) => [
      nodeId,
      node.constructor,
      node.memoryScope,
      node.executionScope,
      node.commitAs,
    ]),
  };
}

/** Replay `log` over a fresh flow and expect the replay to equal the live `state`; return it. */
function expectReplays(
  state: reducer.RunState,
  log: MemoryRunLog,
  options: { sibling?: boolean } = {},
): reducer.RunState {
  const replayed = reducer.replayRecords(new reducer.RunState(sessionFlow(options)), log.records(), {
    growBuilder: () => {
      throw new Error("builder called");
    },
  });
  expect(replaySnapshot(replayed)).toEqual(replaySnapshot(state));
  expect(flowSnapshot(replayed)).toEqual(flowSnapshot(state));
  const trace = (s: reducer.RunState) => (s.trace === null ? null : replayTraceSnapshot(s.trace));
  expect(trace(replayed)).toEqual(trace(state));
  return replayed;
}

test("a resume allows memory-free unfinished invocations", () => {
  const { state, log } = started({ enabled: false });
  const firstSequence = state.acceptedStarts.get("agent")!;
  reducer.commitPrepared(log, state, reducer.prepareRunResumed(state, { numWorkers: 2 }));
  expect(state.acceptedStarts).toEqual(new Map());
  reducer.commitPrepared(log, state, reducer.prepareNodeStarted(state, "agent"));
  expect(state.acceptedStarts.get("agent")!).toBeGreaterThan(firstSequence);
  expect("conversationState" in state).toBe(false);
  expectReplays(state, log);
});

test.each([false, true])("a resume retries every unfinished invocation (learnable: %p)", (learnable) => {
  const { state, log } = started({ learnable, sibling: true });
  reducer.commitPrepared(log, state, reducer.prepareNodeStarted(state, "sibling"));
  const before = [...state.inFlight];
  const prepared = reducer.prepareRunResumed(state, { numWorkers: 0 });
  expect((prepared.record as RunResumedRecord).retryNodeIds).toEqual(before);
  expect(before).toEqual(["agent", "sibling"]);
  reducer.commitPrepared(log, state, prepared);
  expect(state.ready).toEqual(before);
  expect(state.inFlight).toEqual(new Set());
  expect(state.acceptedStarts).toEqual(new Map());
  expectReplays(state, log, { sibling: true });
});

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
