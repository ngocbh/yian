/**
 * Transactional run-state initialization: a port of `tests/engine/test_run_state_initialization.py`.
 *
 * Ported: `test_run_state_starts_pristine_around_the_owned_graph`.
 */

import { expect, test } from "bun:test";

import { NodeState } from "../../src/runtime/common.ts";
import { RunState } from "../../src/runtime/state.ts";
import { flow } from "./run_state_fixtures.ts";

test("run state starts pristine around the owned graph", () => {
  const owned = flow();

  const state = new RunState(owned);

  expect(state.flow).toBe(owned);
  expect(state.pool.startId).toBe(owned.startId);
  expect(state.pool.store.size).toBe(0);
  expect(state.pool.system.size).toBe(0);
  expect(state.nodeState).toEqual(new Map([...owned.nodes.keys()].map((id) => [id, NodeState.UNKNOWN])));
  expect(state.edgeState).toEqual(new Map(owned.edges.map((edge) => [edge.id, NodeState.UNKNOWN])));
  expect(state.ready).toEqual([]);
  expect(state.inFlight).toEqual(new Set());
  expect(state.acceptedStarts).toEqual(new Map());
  expect("conversationState" in state).toBe(false);
  expect(state.result).toBeNull();
  expect(state.depth).toEqual(new Map());
  expect(state.wiringOverrides).toEqual(new Map());
  expect(state.runId).toBeNull();
  expect(state.flowFingerprint).toBeNull();
  expect(state.nextSequence).toBe(0);
  expect(state.numWorkers).toBe(0);
  expect(state.input).toEqual({});
  expect(state.effectiveLlmByStaticPath).toEqual({});
});
