/**
 * Memory learning projections: a port of `tests/engine/test_runtime_memory_learning.py`.
 * Ported: `test_commit_id_has_one_documented_canonical_encoding` (both pinned digests),
 * `test_causal_route_episode_retains_the_accepted_failure` (with the `_scope` and `_evidence`
 * helpers) and `test_route_and_growth_projection_use_closed_source_neutral_envelopes`.
 *
 * Ported, the run-state tests (on the shared `startTraceNode`, Python's `_start_trace_node`; each
 * rejected row names its `RangeError` and full message, Python's `ValueError` matched in part):
 * `test_evidence_is_rejected_when_learning_is_not_enabled` (2 rows),
 * `test_learning_requires_evidence_and_rejects_ineligible_or_unrecorded_scope`,
 * `test_learning_rejects_a_scope_without_a_recorded_stack`,
 * `test_empty_enabled_table_retains_scope_free_evidence_without_episodes`,
 * `test_learning_rejects_trace_scope_that_omits_a_recorded_memory` (2 rows), and the `success`,
 * `route` and `growth` rows of `test_engine_episodes_use_the_physical_start_sequence_for_every_outcome`
 * (the growth into the shared `fragment`).
 *
 * Ported, after rewrite: `test_owner_context_projects_synthetic_source_free_failure_only_to_root_flow`
 * and `test_root_fallback_reaches_only_keys_owned_by_the_root_flow_site`, with the root result
 * (`BackwardResult(self_feedback=("original text",))`) and the empty results built by hand instead
 * of computed; the export half of `test_owner_contexts_are_the_only_episode_projection` (the
 * removed names stay absent from `runtime/learning.ts`).
 *
 * Changed:
 * - `test_commit_id_requires_a_positive_attempt_and_separates_attempts`: the invalid attempt ids
 *   are `null, 0, -1, true, "2", 2.5, 2n` (`2.0` is the valid `2` in TS), each a `RangeError`
 *   naming `attempt_id` (Python's `ValueError`); the rejected mutation container is a `Set`
 *   (Python passes a list), since a batch is an array.
 * - Payload and seed ints are `bigint`s (`1n`), and the growth seed's tuple is an array.
 * - `_episode_outcome` is the module export `episodeOutcome`; its `is` on the status becomes
 *   `toBe` on the status string.
 * - The root entry's `message_index` is `0n`; Python's tuples are arrays.
 * - The export half checks the removed names in camelCase (`projectOutcomeEpisodes`,
 *   `occurrenceKey`); the half that runs a flow comes with the engine (below).
 * - `test_graph_outcome_uses_constant_memory_roots_and_indexed_scope`: Python counts the calls of
 *   the patched `_recorded_stack`, which an ES module can't patch; the test commits the outcome over
 *   1024 keys and stacks and asserts the memory roots and the occurrence's scope keep their
 *   identity.
 *
 * TS-only: the run id, memory and mutation-item messages of `memoryCommitId`; a float payload
 * digests its `logJson` text, and a `-0` payload digests as a float `0` payload (the run log
 * writes both as `0.0`); `failureJson` equals `toJSON()`, refuses a value that isn't exactly a
 * `Failure` and revalidates the failure's fields;
 * `episodeOutcome` projects a success's canonical value and a failed record's failure JSON, and
 * refuses a record of another class and a forged success without a canonical typed value.
 *
 * Later (T4-5, replay): `test_replay_rejects_an_eligible_node_scope_that_differs_from_its_recorded_stack`
 * (it replays through `replayRecords`).
 *
 * Later (T4-7, with `computeBackward` and the engine): the owner-context and episode tests
 * (`test_owner_contexts_include_general_occurrences_and_empty_memories` through
 * `test_owner_variable_batches_follow_captured_inventory_across_wire_round_trip`, except
 * `test_owner_context_projects_synthetic_source_free_failure_only_to_root_flow`, ported above;
 * `test_runstarted_retains_defensive_memory_authority_and_projects_episodes`, the `failure` row of
 * `test_engine_episodes_use_the_physical_start_sequence_for_every_outcome` (it reads the owner
 * contexts),
 * `test_failure_episode_uses_full_canonical_durable_failure_json`,
 * `test_expanded_map_occurrence_projects_and_replays_from_live_node_scope` and the rest of
 * `test_owner_contexts_are_the_only_episode_projection`).
 */

import { createHash } from "node:crypto";

import { expect, test } from "bun:test";

import { END_ID, START_ID } from "../../src/compile/model.ts";
import { Failure, OriginRef } from "../../src/diagnostics.ts";
import {
  DeclSite,
  EvaluationEvidence,
  MemoryEpisodeStatus,
  MemoryMutation,
  MemoryOccurrenceKind,
  MemoryOccurrenceSegment,
  MemoryScope,
  MemorySpec,
  MemoryStack,
  ModelTextInteraction,
} from "../../src/memory/index.ts";
import { BackwardResult, Grow, Node, NodeKind, type NodeResult, Output } from "../../src/nodes/base.ts";
import {
  NodeExpandedRecord,
  NodeFailedRecord,
  NodeRoutedRecord,
  NodeStartedRecord,
  NodeSucceededRecord,
} from "../../src/runtime/log.ts";
import * as learning from "../../src/runtime/learning.ts";
import {
  commitPrepared,
  prepareExpanded,
  prepareNodeRouted,
  prepareNodeSucceeded,
} from "../../src/runtime/state.ts";
import { type OutcomeRecord, RetainedTrace, TraceTerminal } from "../../src/runtime/trace.ts";
import { FuncNode, forwardCapture, fragmentBuilder, stacks, startTraceNode } from "./run_state_fixtures.ts";

/** Assert that `fn` throws exactly `cls` with exactly `message`. */
function expectError(fn: () => unknown, cls: typeof TypeError | typeof RangeError, message: string): void {
  let caught: unknown;
  try {
    fn();
  } catch (error) {
    caught = error;
  }
  expect((caught as Error | undefined)?.constructor).toBe(cls);
  expect((caught as Error).message).toBe(message);
}

/** One outer and one local memory key, in stack order. */
function keys(): string[] {
  return ["outer", "local"];
}

/** One canonical dynamic MAP occurrence. */
function occurrence(index = 2): MemoryOccurrenceSegment[] {
  return [new MemoryOccurrenceSegment(MemoryOccurrenceKind.MAP_ITEM, ["flow"], index)];
}

/** One exact trace scope for a test node: by default `keys()` in `occurrence()`. */
function scope(
  nodeId = "agent",
  options: { memories?: readonly string[]; occurrence?: readonly MemoryOccurrenceSegment[] } = {},
): MemoryScope {
  return new MemoryScope(["flow", nodeId], {
    memories: options.memories ?? keys(),
    occurrencePath: options.occurrence ?? occurrence(),
  });
}

/** A trait-capable AGENT-kind node that evaluates without calling a model. */
class TraceNode extends Node {
  readonly kind = NodeKind.AGENT;
  override readonly needsEvaluationTrace = true;

  override forward(inputs: Record<string, unknown>): NodeResult {
    return new Output(inputs);
  }
}

/** A `TraceNode` that may grow the graph, Python's `_TraceNode` subclass `Spawner`. */
class TraceSpawner extends TraceNode {
  override readonly isSpawner = true;
}

/** Deterministic visible evidence for one accepted outcome. */
function evidence(memoryScope: MemoryScope | null): EvaluationEvidence {
  return new EvaluationEvidence(memoryScope, { question: "q" }, "system", "prompt", [], [
    new ModelTextInteraction("answer"),
  ]);
}

/** The root flow's declaration site. */
const ROOT = new DeclSite([], ["root"]);

/** A synthetic source-free failed terminal trace, with no occurrences. */
function syntheticTrace(): RetainedTrace {
  const trace = new RetainedTrace("synthetic", START_ID, END_ID);
  trace.terminal = new TraceTerminal(
    false,
    null,
    null,
    new Failure({ code: "graph.terminal_unreachable", message: "no producer", nodeId: null }),
  );
  return trace;
}

/** The root result a source-free failed terminal keeps for text feedback: the text as self feedback. */
const ORIGINAL_ROOT = new BackwardResult({ selfFeedback: ["original text"] });

/** One mutation whose payload holds text, null and an int. */
const MUTATIONS = [new MemoryMutation("m", { z: 1n, a: ["é", null] })];

test("commit id has one documented canonical encoding", () => {
  expect(learning.memoryCommitId("run", "notes", MUTATIONS, 1)).toBe(
    "ac-memory-v3:250e0a4c357acfd6ef4d37a84a5207bffdb3df2e343816df5e7219401a7bf47b",
  );
  expect(learning.memoryCommitId("run", "notes", MUTATIONS, 2)).toBe(
    "ac-memory-v3:1c87e547bc72a30942158a6c0c0bc3dc4fa21245e45564ac7571175e48cdf64a",
  );
});

test("commit id requires a positive attempt and separates attempts", () => {
  expect(learning.memoryCommitId("run", "notes", [], 2)).not.toBe(learning.memoryCommitId("run", "notes", [], 3));
  for (const invalid of [null, 0, -1, true, "2", 2.5, 2n]) {
    expectError(
      () => learning.memoryCommitId("run", "notes", MUTATIONS, invalid as number),
      RangeError,
      "attempt_id must be a positive integer",
    );
  }
  expectError(
    () => learning.memoryCommitId("run", "notes", new Set(MUTATIONS) as unknown as MemoryMutation[], 1),
    TypeError,
    "mutations must be an array",
  );
});

test.each<[string, () => unknown, typeof TypeError | typeof RangeError, string]>([
  [
    "an empty run id",
    () => learning.memoryCommitId("", "notes", [], 1),
    RangeError,
    "run_id must be a non-empty string",
  ],
  [
    "a run id that isn't text",
    () => learning.memoryCommitId(null as unknown as string, "notes", [], 1),
    RangeError,
    "run_id must be a non-empty string",
  ],
  ["an empty memory", () => learning.memoryCommitId("run", "", [], 1), RangeError, "memory must be a non-empty string"],
  [
    "a memory that isn't text",
    () => learning.memoryCommitId("run", 1 as unknown as string, [], 1),
    RangeError,
    "memory must be a non-empty string",
  ],
  [
    "a mutation that isn't exactly a MemoryMutation",
    () => learning.memoryCommitId("run", "notes", [{ mutationId: "m", payload: 1n } as unknown as MemoryMutation], 1),
    TypeError,
    "mutations items must be MemoryMutation values",
  ],
])("commit id refuses %s", (_label, fn, cls, message) => {
  expectError(fn, cls, message);
});

test("a float payload digests its log text", () => {
  const mutations = [new MemoryMutation("m", { x: 1.5, n: 2 })];
  const text = '["run",1,"notes",[["m",{"n":2.0,"x":1.5}]]]';
  const digest = createHash("sha256").update(text, "utf8").digest("hex");
  expect(learning.memoryCommitId("run", "notes", mutations, 1)).toBe(`ac-memory-v3:${digest}`);
});

test("a negative zero payload digests as zero", () => {
  expect(learning.memoryCommitId("run", "notes", [new MemoryMutation("m", { x: -0 })], 1)).toBe(
    learning.memoryCommitId("run", "notes", [new MemoryMutation("m", { x: 0 })], 1),
  );
});

test("failure JSON equals the failure's JSON form", () => {
  const failure = new Failure({
    code: "node.execute",
    message: "Failed.",
    nodeId: "agent",
    trace: [new OriginRef("writer")],
    causeType: "Error",
    traceback: "stack",
  });
  expect(learning.failureJson(failure)).toEqual(failure.toJSON());
});

test("failure JSON refuses a value that isn't exactly a Failure and revalidates its fields", () => {
  const failure = new Failure({ code: "node.execute", message: "Failed.", nodeId: "agent" });
  class SubFailure extends Failure {}
  for (const invalid of [failure.toJSON(), { ...failure }, new SubFailure(failure), null]) {
    expectError(() => learning.failureJson(invalid as Failure), TypeError, "failure projection requires a Failure");
  }
  const edited = new Failure({ code: "node.execute", message: "Failed.", nodeId: "agent" });
  (edited as { code: string }).code = "bogus";
  expectError(() => learning.failureJson(edited), RangeError, "unknown failure code 'bogus'");
});

test("causal route episode retains the accepted failure", () => {
  const failure = new Failure({ code: "graph.terminal_unreachable", message: "No terminal route.", nodeId: "agent" });
  const record = new NodeRoutedRecord("agent", "go", { evidence: evidence(scope()), terminalFailure: failure });
  const [status, output, failurePayload] = learning.episodeOutcome(record);
  expect(status).toBe(MemoryEpisodeStatus.FAILED);
  expect(output).toBeNull();
  expect(failurePayload!.code).toBe(failure.code);
});

test("route and growth projection use closed source-neutral envelopes", () => {
  const [, route] = learning.episodeOutcome(new NodeRoutedRecord("agent", "yes"));
  const [, growth] = learning.episodeOutcome(
    new NodeExpandedRecord("agent", { value: [1n, 2n] }, ["a", "z"], "b".repeat(64)),
  );

  expect(route).toEqual({ kind: "route", handle: "yes" });
  expect(growth).toEqual({ kind: "grow", seed: { value: [1n, 2n] }, prune_ids: ["a", "z"] });
});

test("an episode projects a success's canonical value and a failed record's failure JSON", () => {
  const record = new NodeSucceededRecord("agent", "agent", { kind: "object", value: { b: 1n, a: 2.5 } });
  const succeeded = learning.episodeOutcome(record);
  expect(succeeded).toEqual([MemoryEpisodeStatus.SUCCEEDED, { a: 2.5, b: 1n }, null]);
  expect(Object.keys(succeeded[1] as object)).toEqual(["a", "b"]);

  const failure = new Failure({ code: "node.execute", message: "Failed.", nodeId: "agent" });
  expect(learning.episodeOutcome(new NodeFailedRecord(failure))).toEqual([
    MemoryEpisodeStatus.FAILED,
    null,
    failure.toJSON(),
  ]);

  expectError(
    () => learning.episodeOutcome(new NodeStartedRecord("agent") as unknown as OutcomeRecord),
    TypeError,
    "unsupported graph outcome for episode projection",
  );
});

test("an episode refuses a success without a canonical typed value", () => {
  // A constructed record always holds a typed wire, so the record is forged through its prototype.
  const forged = Object.assign(Object.create(NodeSucceededRecord.prototype) as NodeSucceededRecord, {
    nodeId: "agent",
    value: { kind: "int" },
  });
  expectError(() => learning.episodeOutcome(forged), RangeError, "successful record has no canonical typed value");
});

test("owner context projects a synthetic source-free failure only to the root flow", () => {
  const trace = syntheticTrace();
  const memories = [
    new MemorySpec("inner", "inner", {}, [new DeclSite([], ["def", "child"])], false),
    new MemorySpec("outer", "outer", {}, [ROOT], false),
  ];
  const [inner, outer] = learning.projectOwnerContexts(trace, memories, [], "original text", ORIGINAL_ROOT, []);
  expect(outer!.episodes).toEqual([]);
  expect(inner!.episodes).toEqual([]);
  expect(inner!.feedback).toEqual([]);
  expect(outer!.feedback).toEqual([
    {
      execution_seq: null,
      owner_static_path: null,
      occurrence_path: null,
      execution_scope: null,
      root: true,
      channel: "self",
      target_name: null,
      value: "original text",
      message_index: 0n,
      contribution_context: {},
    },
  ]);
});

test("root fallback reaches only keys owned by the root flow site", () => {
  const trace = syntheticTrace();
  const memories = [
    new MemorySpec("agent_notes", "agent_notes", {}, [new DeclSite([], ["root"], "agent")], true),
    new MemorySpec("def_notes", "def_notes", {}, [new DeclSite([], ["def", "child"])], false),
    // An imported file's root flow is not the run's root flow.
    new MemorySpec("lib.notes", "lib.notes", {}, [new DeclSite(["lib"], ["root"])], false),
    new MemorySpec("shared", "shared", {}, [new DeclSite([], ["def", "child"]), ROOT], false),
  ];
  const [agentNotes, defNotes, imported, shared] = learning.projectOwnerContexts(
    trace,
    memories,
    [],
    "original text",
    ORIGINAL_ROOT,
    [],
  );
  expect(agentNotes!.feedback).toEqual([]);
  expect(defNotes!.feedback).toEqual([]);
  expect(imported!.feedback).toEqual([]);
  expect(shared!.feedback.map((entry) => (entry as { value: unknown }).value)).toEqual(["original text"]);
  expect(shared!.feedback.every((entry) => (entry as { root: unknown }).root === true)).toBe(true);
});

test("owner contexts are the only episode projection", () => {
  for (const name of ["RuntimeLearningState", "projectOutcomeEpisodes", "occurrenceKey"]) {
    expect(Object.hasOwn(learning, name)).toBe(false);
  }
  expect(typeof learning.projectOwnerContexts).toBe("function");
});

// The run-state reducer's learning authority, on `startTraceNode`.

test.each([
  ["success", [END_ID]],
  ["route", [END_ID]],
  ["growth", ["agent/child/start"]],
] as const)("an engine episode uses the physical start sequence: %s", (kind, ready) => {
  const node = new (kind === "growth" ? TraceSpawner : TraceNode)("agent");
  node.memoryScope = scope(node.id);
  const { state, log } = startTraceNode(node, {
    keys: node.memoryScope.memories,
    learnable: true,
    routeHandle: kind === "route" ? "go" : undefined,
  });
  const physicalSequence = state.trace!.started.get(node.id);
  const outcomeSequence = state.nextSequence;
  const carriers = { evidence: evidence(node.memoryScope), forwardCapture: forwardCapture(node, { question: "q" }) };
  const prepared = {
    success: () => prepareNodeSucceeded(state, node.id, { answer: 7n }, carriers),
    route: () => prepareNodeRouted(state, node.id, "go", carriers),
    growth: () => prepareExpanded(state, node.id, new Grow({}), fragmentBuilder, carriers),
  }[kind]();

  commitPrepared(log, state, prepared);

  expect(physicalSequence).not.toBe(outcomeSequence);
  expect(state.trace!.occurrences.at(-1)!.source.executionSeq).toBe(physicalSequence!);
  expect(state.ready).toEqual([...ready]);
});

test.each([
  ["memory disabled", false],
  ["memory enabled", true],
])("evidence is rejected when learning is not enabled: %s", (_label, enabled) => {
  const node = new TraceNode("agent");
  node.memoryScope = scope(node.id);
  const { state } = startTraceNode(node, { keys: enabled ? node.memoryScope.memories : null, learnable: false });

  expectError(
    () => prepareNodeSucceeded(state, node.id, "done", { evidence: evidence(node.memoryScope) }),
    RangeError,
    "node 'agent' is ineligible for evaluation evidence",
  );
  expect((prepareNodeSucceeded(state, node.id, "done").record as NodeSucceededRecord).evidence).toBeNull();
});

test("learning requires evidence and rejects an ineligible or unrecorded scope", () => {
  const node = new TraceNode("agent");
  node.memoryScope = scope(node.id);
  const { state } = startTraceNode(node, { keys: keys(), learnable: true });
  expectError(
    () => prepareNodeSucceeded(state, node.id, "done", { forwardCapture: forwardCapture(node, {}) }),
    RangeError,
    "eligible node 'agent' requires evaluation evidence while learning",
  );

  const ordinary = new FuncNode("ordinary", () => "done");
  ordinary.memoryScope = scope(ordinary.id);
  const { state: ordinaryState } = startTraceNode(ordinary, { keys: keys(), learnable: true });
  expectError(
    () =>
      prepareNodeSucceeded(ordinaryState, ordinary.id, "done", {
        evidence: evidence(ordinary.memoryScope),
        forwardCapture: forwardCapture(ordinary, { question: "q" }),
      }),
    RangeError,
    "node 'ordinary' is ineligible for evaluation evidence",
  );
  const accepted = prepareNodeSucceeded(ordinaryState, ordinary.id, "done", {
    forwardCapture: forwardCapture(ordinary, {}),
  });
  expect((accepted.record as NodeSucceededRecord).evidence).toBeNull();

  const mismatched = new TraceNode("agent");
  mismatched.memoryScope = scope(mismatched.id);
  const { state: mismatchedState } = startTraceNode(mismatched, {
    keys: keys(),
    learnable: true,
    stacks: [new MemoryStack(["flow", "agent"], ["outer"])],
  });
  expectError(
    () =>
      prepareNodeSucceeded(mismatchedState, mismatched.id, "done", {
        evidence: evidence(mismatched.memoryScope),
        forwardCapture: forwardCapture(mismatched, { question: "q" }),
      }),
    RangeError,
    "node 'agent' memory scope must carry its exact recorded memory stack",
  );
});

test("learning rejects a scope without a recorded stack", () => {
  const node = new TraceNode("agent");
  node.memoryScope = scope(node.id);
  const { state } = startTraceNode(node, { keys: keys(), learnable: true, stacks: [] });

  expectError(
    () =>
      prepareNodeSucceeded(state, node.id, "done", {
        evidence: evidence(node.memoryScope),
        forwardCapture: forwardCapture(node, { question: "q" }),
      }),
    RangeError,
    "node 'agent' has no recorded memory stack",
  );
});

test("an empty enabled table retains scope-free evidence without episodes", () => {
  const node = new TraceNode("agent");
  const { state } = startTraceNode(node, { keys: [], learnable: true });

  const prepared = prepareNodeSucceeded(state, node.id, "done", {
    evidence: evidence(null),
    forwardCapture: forwardCapture(node, { question: "q" }),
  });
  expect((prepared.record as NodeSucceededRecord).evidence!.scope).toBeNull();
  expectError(
    () =>
      prepareNodeSucceeded(state, node.id, "done", {
        evidence: evidence(scope(node.id)),
        forwardCapture: forwardCapture(node, { question: "q" }),
      }),
    RangeError,
    "node 'agent' evidence scope does not match its compiled scope",
  );
});

test.each(["outer", "local"])("learning rejects a trace scope that omits a recorded memory: keeps %s", (retained) => {
  const node = new TraceNode("agent");
  node.memoryScope = scope(node.id, { memories: [retained] });
  const { state } = startTraceNode(node, { keys: keys(), learnable: true, stacks: stacks(scope(node.id)) });

  expectError(
    () => prepareNodeSucceeded(state, node.id, "done", { forwardCapture: forwardCapture(node, {}) }),
    RangeError,
    "node 'agent' memory scope must carry its exact recorded memory stack",
  );
});

test("a graph outcome uses constant memory roots and an indexed scope", () => {
  const memoryKeys = Array.from({ length: 1024 }, (_, index) => `agent-${String(index).padStart(4, "0")}`);
  const node = new TraceNode("agent");
  node.memoryScope = new MemoryScope(["agents", "agent-1023"], { memories: [memoryKeys.at(-1)!] });
  const recorded = memoryKeys.map((key) => new MemoryStack(["agents", key], [key]));
  const { state, log } = startTraceNode(node, { keys: memoryKeys, learnable: true, stacks: recorded });
  const roots = [state.memories, state.memoryStacks, state.memorySnapshots];

  const prepared = prepareNodeSucceeded(state, node.id, "done", {
    evidence: evidence(node.memoryScope),
    forwardCapture: forwardCapture(node, { question: "q" }),
  });
  commitPrepared(log, state, prepared);

  const after = [state.memories, state.memoryStacks, state.memorySnapshots];
  after.forEach((root, index) => expect(root).toBe(roots[index]!));
  expect(state.trace!.occurrences.at(-1)!.memoryScope).toBe(node.memoryScope);
});
