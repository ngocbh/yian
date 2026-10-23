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
 * `test_learning_rejects_trace_scope_that_omits_a_recorded_memory` (2 rows),
 * `test_engine_episodes_use_the_physical_start_sequence_for_every_outcome` (4 rows; the growth
 * into the shared `fragment`; the `failure` row reads the owner contexts through a local
 * `contexts`, Python's `_contexts`), `test_failure_episode_uses_full_canonical_durable_failure_json`,
 * `test_owner_feedback_uses_recorded_execution_coordinates_independently_of_memory` (2 rows;
 * `backward` and `executionScope` assigned on the node; the run's recorded table and stacks stand
 * in for Python's equal `_table` and `_stacks`) and
 * `test_replay_rejects_an_eligible_node_scope_that_differs_from_its_recorded_stack` (through
 * `replayRecords` over `copyGraph` copies of the flow taken before the outcome).
 *
 * Ported: `test_owner_context_projects_synthetic_source_free_failure_only_to_root_flow` and
 * `test_root_fallback_reaches_only_keys_owned_by_the_root_flow_site` (both on `computeBackward`).
 *
 * Ported, after rewrite: `test_owner_contexts_are_the_only_episode_projection` (the removed names
 * stay absent from `runtime/learning.ts`, and `learningState` from the state), on a local
 * `successfulTerminal`, Python's `_successful_terminal`.
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
 * - `test_owner_contexts_are_the_only_episode_projection` checks the removed names in camelCase
 *   (`projectOutcomeEpisodes`, `occurrenceKey`, `learningState`).
 * - `test_runstarted_retains_defensive_memory_authority_and_projects_episodes`: the table's config
 *   and the snapshot are frozen, so their edits throw a `TypeError` (asserted) where Python's land
 *   on copies; `3` and `4` are `3n` and `4n`.
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
 * Changed, owner contexts over engine runs (`computeBackward`, then `projectOwnerContexts`; the
 * feedback entries' ints, `message_index` and `execution_seq` included, are `bigint`s):
 * - `test_owner_contexts_include_general_occurrences_and_empty_memories`: `engine` with the memory
 *   options; `declaredVariables` and `callback` are replaced on the node; the entries and episodes
 *   are frozen, so the tail's edits throw a `TypeError` (asserted) where Python's land on copies.
 * - On `ownerEngine` (the resolved layout of a root-owned memory, as Python's):
 *   `test_owner_feedback_retains_literals_and_opaque_local_stops` (2 rows; the template function
 *   registered with `registerTemplateFn` and restored in `afterEach`),
 *   `test_owner_contexts_keep_unavailable_inputs_and_causal_failures` (3 rows; the binding row
 *   empties `work`'s wiring, the route row patches `terminalSkipRouteFlow()`'s router on the
 *   instance, and the body's error is an `Error`, Python's `ValueError`),
 *   `test_owner_context_root_default_is_owned_once_by_accepted_start`,
 *   `test_owner_context_projection_from_replay_does_not_reinvoke_handlers` (the node methods
 *   assigned on a `copyGraph` copy, the replayed trace's `backwardHandlers` shadowed on the
 *   instance) and `test_owner_variable_batches_follow_captured_inventory_across_wire_round_trip`
 *   (2 rows; `declaredVariables` replaced on the node).
 * - `test_owner_context_root_fallback_uses_only_actual_source_zero_fields`: only the row where START
 *   is also the end, on a local `startOnlyRun`, its flow's readonly `endId` assigned through a
 *   cast. A run start commits only the system values and the START inputs, so END can't hold a
 *   preseeded result.
 * - `test_episode_outcome_projects_a_success_value`: the committed value is passed in, as the
 *   success names its `pool_commit`; `test_episode_refuses_a_success_without_a_typed_value` isn't
 *   ported, since the value no longer comes from the success record.
 * - `test_owner_context_authority_errors_are_not_context_construction_failures`: each refusal pins
 *   its class and message; a `Set` stands in for Python's list table and stacks; the trace's
 *   `lookupSource` is patched on the instance, and its `Error` (Python's `KeyError`) escapes
 *   unchanged. Not ported: the two forged-scope halves, since an occurrence and its scope are
 *   frozen.
 * - `test_owner_context_error_identifies_receiver_without_changing_forward_acceptance` (one test
 *   for Python's 2 rows): TS has no memory JSON nesting bound, so a spy on `LearnContext.fromFrozen`
 *   throws a `RangeError` on the call for `work`'s occurrence (the second) and passes the others
 *   to the original; the `OwnerContextError` names `memory`, `original` (the spy's error, also its
 *   `cause`) and `work`'s backward context, and the log and the terminal output stay unchanged. The
 *   spy is restored in `finally`.
 * - `test_root_feedback_context_failure_names_no_node` and
 *   `test_root_fallback_failure_names_no_node_end_to_end` (one test): `_freeze_feedback` can't be
 *   patched and TS has no memory JSON nesting bound, so on `startOnlyRun()` a spy on
 *   `LearnContext.fromFrozen` throws a `RangeError` on the root-fallback call (the second, after
 *   START's occurrence) and passes the others to the original; the driven failure names no node.
 *   The spy is restored in `finally`.
 * - `test_expanded_map_occurrence_projects_and_replays_from_live_node_scope`: the expansion and
 *   the replay grow through `buildGrow`; the dynamic scope's class is checked by its constructor.
 * - `test_owner_context_empty_map_retains_growth_and_unexecuted_child_memory`: `ownerEngine` with
 *   `growBuilder: buildGrow` (Python's `_engine` default).
 */

import { createHash } from "node:crypto";
import { isDeepStrictEqual } from "node:util";

import { afterEach, expect, spyOn, test } from "bun:test";

import { buildGrow } from "../../src/compile/expand.ts";
import { CompiledFlow, copyGraph, END_ID, FlowOutput, START_ID } from "../../src/compile/model.ts";
import { registerTemplateFn, TEMPLATE_FNS, type TemplateFn, type TemplateParam } from "../../src/expr/builtins.ts";
import { Failure, OriginRef } from "../../src/diagnostics.ts";
import { RunFailed, RunSucceeded } from "../../src/events.ts";
import {
  DeclSite,
  EvaluationEvidence,
  LearnContext,
  MemoryDecl,
  MemoryEpisodeStatus,
  MemoryMutation,
  MemoryOccurrenceKind,
  MemoryOccurrenceSegment,
  MemoryScope,
  MemorySnapshotEnvelope,
  MemorySpec,
  MemoryStack,
  ModelTextInteraction,
} from "../../src/memory/index.ts";
import {
  BackwardResult,
  ELSE_HANDLE,
  Grow,
  Node,
  NodeKind,
  type NodeResult,
  Output,
  Route,
} from "../../src/nodes/base.ts";
import { AgentNode } from "../../src/nodes/agent/node.ts";
import { Binding, comparePaths, ExecutionPosition, ExecutionScope, Parameter } from "../../src/nodes/contracts.ts";
import { EndNode } from "../../src/nodes/end/node.ts";
import { MapNode } from "../../src/nodes/map/node.ts";
import { StartNode } from "../../src/nodes/start/node.ts";
import { computeBackward, driveBackward } from "../../src/runtime/backward.ts";
import type { AttributedBackwardResult, BackwardOutcome } from "../../src/runtime/contracts.ts";
import { FlowEngine } from "../../src/runtime/engine.ts";
import {
  BackwardReadyRecord,
  decodeRecord,
  encodeRecord,
  MemoryRunLog,
  NodeExpandedRecord,
  NodeFailedRecord,
  NodeRoutedRecord,
  NodeStartedRecord,
  NodeSucceededRecord,
  RecordEnvelope,
  SCHEMA_VERSION,
} from "../../src/runtime/log.ts";
import * as learning from "../../src/runtime/learning.ts";
import { MemoryRuntime } from "../../src/runtime/memory_runtime.ts";
import {
  commitInitialPool,
  commitPrepared,
  prepareExpanded,
  prepareNodeFailed,
  prepareNodeRouted,
  prepareNodeStarted,
  prepareNodeSucceeded,
  preparePoolCommit,
  prepareRunStarted,
  replayRecords,
  RunState,
} from "../../src/runtime/state.ts";
import { type OutcomeRecord, RetainedTrace, TraceTerminal } from "../../src/runtime/trace.ts";
import { type Value, VariablePool } from "../../src/typesys/index.ts";
import {
  engine,
  feedbackEntries,
  FINGERPRINT,
  forwardCapture,
  fragmentBuilder,
  FuncNode,
  graph,
  noGrowth,
  ownerEngine,
  POLICY,
  prepareOutput,
  RecordingBackwardNode,
  RUN_ID,
  stacks,
  startTraceNode,
  terminalSkipRouteFlow,
} from "./run_state_fixtures.ts";
import { FakeMemory } from "./memory_fakes.ts";

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
  const [status, output, failurePayload] = learning.episodeOutcome(record, null);
  expect(status).toBe(MemoryEpisodeStatus.FAILED);
  expect(output).toBeNull();
  expect(failurePayload!.code).toBe(failure.code);
});

test("route and growth projection use closed source-neutral envelopes", () => {
  const [, route] = learning.episodeOutcome(new NodeRoutedRecord("agent", "yes"), null);
  const [, growth] = learning.episodeOutcome(
    new NodeExpandedRecord("agent", { value: [1n, 2n] }, ["a", "z"], "b".repeat(64)),
    null,
  );

  expect(route).toEqual({ kind: "route", handle: "yes" });
  expect(growth).toEqual({ kind: "grow", seed: { value: [1n, 2n] }, prune_ids: ["a", "z"] });
});

test("an episode projects a success's canonical value and a failed record's failure JSON", () => {
  const record = new NodeSucceededRecord("agent", 1);
  const succeeded = learning.episodeOutcome(record, { b: 1n, a: 2.5 });
  expect(succeeded).toEqual([MemoryEpisodeStatus.SUCCEEDED, { a: 2.5, b: 1n }, null]);
  expect(Object.keys(succeeded[1] as object)).toEqual(["a", "b"]);

  const failure = new Failure({ code: "node.execute", message: "Failed.", nodeId: "agent" });
  expect(learning.episodeOutcome(new NodeFailedRecord(failure), null)).toEqual([
    MemoryEpisodeStatus.FAILED,
    null,
    failure.toJSON(),
  ]);

  expectError(
    () => learning.episodeOutcome(new NodeStartedRecord("agent") as unknown as OutcomeRecord, null),
    TypeError,
    "unsupported graph outcome for episode projection",
  );
});

test("owner context projects a synthetic source-free failure only to the root flow", async () => {
  const trace = syntheticTrace();
  const memories = [
    new MemorySpec("inner", "inner", {}, [new DeclSite([], ["def", "child"])]),
    new MemorySpec("outer", "outer", {}, [ROOT]),
  ];
  const [root, results] = await computeBackward(trace, "original text");
  const [inner, outer] = learning.projectOwnerContexts(trace, memories, [], "original text", root, results);
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

test("root fallback reaches only keys owned by the root flow site", async () => {
  const trace = syntheticTrace();
  const memories = [
    new MemorySpec("agent_notes", "agent_notes", {}, [new DeclSite([], ["root"], "agent")]),
    new MemorySpec("def_notes", "def_notes", {}, [new DeclSite([], ["def", "child"])]),
    // An imported file's root flow is not the run's root flow.
    new MemorySpec("lib.notes", "lib.notes", {}, [new DeclSite(["lib"], ["root"])]),
    new MemorySpec("shared", "shared", {}, [new DeclSite([], ["def", "child"]), ROOT]),
  ];
  const [root, results] = await computeBackward(trace, "original text");

  const [agentNotes, defNotes, imported, shared] = learning.projectOwnerContexts(
    trace,
    memories,
    [],
    "original text",
    root,
    results,
  );
  expect(agentNotes!.feedback).toEqual([]);
  expect(defNotes!.feedback).toEqual([]);
  expect(imported!.feedback).toEqual([]);
  expect(shared!.feedback.map((entry) => (entry as { value: unknown }).value)).toEqual(["original text"]);
  expect(shared!.feedback.every((entry) => (entry as { root: unknown }).root === true)).toBe(true);
});

test("owner contexts are the only episode projection", async () => {
  const { state, table } = successfulTerminal();

  expect("learningState" in state).toBe(false);
  for (const name of ["RuntimeLearningState", "projectOutcomeEpisodes", "occurrenceKey"]) {
    expect(Object.hasOwn(learning, name)).toBe(false);
  }
  const episodes = (await contexts(state)).get("outer")!.episodes;
  expect(episodes).toHaveLength(1);
  expect(episodes[0]!.ownerStaticPath).toEqual(["flow", "agent"]);
  expect(state.backwardState.memoryOrder).toEqual(table.map((spec) => spec.key));
});

// The run-state reducer's learning authority, on `startTraceNode`.

/**
 * Return a terminal-success run of `START -> agent -> END` with one accepted trace episode, and
 * its expected table: `keys()`, sorted, each owned by the root flow with config `{limit: 3}`.
 */
function successfulTerminal(): { state: RunState; log: MemoryRunLog; table: MemorySpec[] } {
  const node = new TraceNode("agent");
  const memoryScope = scope(node.id);
  node.memoryScope = memoryScope;
  const { state, log } = startTraceNode(node, { keys: memoryScope.memories, learnable: true });
  commitPrepared(
    log,
    state,
    prepareOutput(log, state, node.id, { answer: 7n }, {
      evidence: evidence(memoryScope),
      forwardCapture: forwardCapture(node, { question: "q" }),
    }),
  );
  commitPrepared(log, state, prepareNodeStarted(state, END_ID));
  commitPrepared(
    log,
    state,
    prepareOutput(log, state, END_ID, null, { forwardCapture: forwardCapture(state.flow.nodes.get(END_ID)!, {}) }),
  );
  expect(state.result).toBeInstanceOf(RunSucceeded);
  const table = [...memoryScope.memories].sort().map((key) => new MemorySpec(key, key, { limit: 3n }, [ROOT]));
  return { state, log, table };
}

/** Return the learner contexts of a terminal run's retained trace, keyed by memory. */
async function contexts(state: RunState): Promise<Map<string, LearnContext>> {
  const [root, results] = await computeBackward(state.trace!, null);
  const projected = learning.projectOwnerContexts(state.trace!, state.memories!, state.memoryStacks!, null, root, results);
  return new Map(projected.map((context) => [context.memory, context]));
}

test("RunStarted retains a defensive memory authority and projects episodes", async () => {
  const { state, table } = successfulTerminal();

  expect(state.memories).toEqual(table);
  expect(state.memorySnapshots).toEqual([
    new MemorySnapshotEnvelope("local", { epoch: 4n }),
    new MemorySnapshotEnvelope("outer", { epoch: 4n }),
  ]);
  expect(state.learnable).toBe(true);
  expect(state.backwardState.memoryOrder).toEqual(["local", "outer"]);

  const byKey = await contexts(state);
  const outerEpisode = byKey.get("outer")!.episodes;
  const localEpisode = byKey.get("local")!.episodes;
  expect(outerEpisode).toHaveLength(1);
  expect(localEpisode).toHaveLength(1);
  expect(outerEpisode[0]!.ownerStaticPath).toEqual(["flow", "agent"]);
  expect(outerEpisode[0]!.occurrencePath).toEqual(occurrence());
  expect(localEpisode[0]!.occurrencePath).toEqual(occurrence());
  expect(outerEpisode[0]!.output).toEqual({ answer: 7n });
  expect(outerEpisode[0]!.inputs).toEqual({ question: "q" });
  expect(outerEpisode[0]!.interactions).toEqual([new ModelTextInteraction("answer")]);

  // The config and snapshot are frozen, so the edits throw where Python's land on copies.
  expect(() => {
    (state.memories![0]!.config as Record<string, unknown>).limit = 999n;
  }).toThrow(TypeError);
  expect(() => {
    (state.memorySnapshots![0]!.snapshot as Record<string, unknown>).epoch = 999n;
  }).toThrow(TypeError);
  expect(state.memories![0]!.config).toEqual({ limit: 3n });
  expect(state.memorySnapshots![0]!.snapshot).toEqual({ epoch: 4n });
});

test.each([
  ["success", [END_ID]],
  ["route", [END_ID]],
  ["failure", []],
  ["growth", ["agent/child/start"]],
] as const)("an engine episode uses the physical start sequence: %s", async (kind, ready) => {
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
    success: () => prepareOutput(log, state, node.id, { answer: 7n }, carriers),
    route: () => prepareNodeRouted(state, node.id, "go", carriers),
    failure: () =>
      prepareNodeFailed(state, new Failure({ code: "node.execute", message: "Failed.", nodeId: node.id }), carriers),
    growth: () => prepareExpanded(state, node.id, new Grow({}), fragmentBuilder, carriers),
  }[kind]();

  commitPrepared(log, state, prepared);

  expect(physicalSequence).not.toBe(outcomeSequence);
  expect(state.trace!.occurrences.at(-1)!.source.executionSeq).toBe(physicalSequence!);
  expect(state.ready).toEqual([...ready]);
  if (kind === "failure") {
    for (const context of (await contexts(state)).values()) {
      expect(context.episodes.at(-1)!.executionSeq).toBe(physicalSequence!);
    }
  }
});

test("a failure episode uses the full canonical durable failure JSON", async () => {
  const node = new TraceNode("agent");
  const memoryScope = scope(node.id, { memories: ["local"] });
  node.memoryScope = memoryScope;
  const { state, log } = startTraceNode(node, { keys: ["local"], learnable: true });
  const failure = new Failure({
    code: "node.execute",
    message: "visible",
    nodeId: node.id,
    trace: [new OriginRef("outer"), new OriginRef("inner")],
    causeType: "ValueError",
    traceback: "debug trace",
  });
  const prepared = prepareNodeFailed(state, failure, {
    evidence: evidence(memoryScope),
    forwardCapture: forwardCapture(node, { question: "q" }),
  });
  commitPrepared(log, state, prepared);

  expect(state.result).toBeInstanceOf(RunFailed);
  const episodes = (await contexts(state)).get("local")!.episodes;
  expect(episodes).toHaveLength(1);
  expect(episodes[0]!.failure).toEqual({
    code: "node.execute",
    message: "visible",
    node_id: "agent",
    trace: [{ key: "outer" }, { key: "inner" }],
    cause_type: "ValueError",
    traceback: "debug trace",
  });
  expect(episodes[0]!.output).toBeNull();
});

/** Return `node`'s run as `startTraceNode` builds it, with `"done"` committed as its pending output. */
function committedTraceNode(node: Node, options: Parameters<typeof startTraceNode>[1]): ReturnType<typeof startTraceNode> {
  const built = startTraceNode(node, options);
  commitPrepared(built.log, built.state, preparePoolCommit(built.state, node.id, "done"));
  return built;
}

test.each([
  ["memory disabled", false],
  ["memory enabled", true],
])("evidence is rejected when learning is not enabled: %s", (_label, enabled) => {
  const node = new TraceNode("agent");
  node.memoryScope = scope(node.id);
  const { state } = committedTraceNode(node, { keys: enabled ? node.memoryScope.memories : null, learnable: false });

  expectError(
    () => prepareNodeSucceeded(state, node.id, { evidence: evidence(node.memoryScope) }),
    RangeError,
    "node 'agent' is ineligible for evaluation evidence",
  );
  expect((prepareNodeSucceeded(state, node.id).record as NodeSucceededRecord).evidence).toBeNull();
});

test("learning requires evidence and rejects an ineligible or unrecorded scope", () => {
  const node = new TraceNode("agent");
  node.memoryScope = scope(node.id);
  const { state } = committedTraceNode(node, { keys: keys(), learnable: true });
  expectError(
    () => prepareNodeSucceeded(state, node.id, { forwardCapture: forwardCapture(node, {}) }),
    RangeError,
    "eligible node 'agent' requires evaluation evidence while learning",
  );

  const ordinary = new FuncNode("ordinary", () => "done");
  ordinary.memoryScope = scope(ordinary.id);
  const { state: ordinaryState } = committedTraceNode(ordinary, { keys: keys(), learnable: true });
  expectError(
    () =>
      prepareNodeSucceeded(ordinaryState, ordinary.id, {
        evidence: evidence(ordinary.memoryScope),
        forwardCapture: forwardCapture(ordinary, { question: "q" }),
      }),
    RangeError,
    "node 'ordinary' is ineligible for evaluation evidence",
  );
  const accepted = prepareNodeSucceeded(ordinaryState, ordinary.id, {
    forwardCapture: forwardCapture(ordinary, {}),
  });
  expect((accepted.record as NodeSucceededRecord).evidence).toBeNull();

  const mismatched = new TraceNode("agent");
  mismatched.memoryScope = scope(mismatched.id);
  const { state: mismatchedState } = committedTraceNode(mismatched, {
    keys: keys(),
    learnable: true,
    stacks: [new MemoryStack(["flow", "agent"], ["outer"])],
  });
  expectError(
    () =>
      prepareNodeSucceeded(mismatchedState, mismatched.id, {
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
  const { state } = committedTraceNode(node, { keys: keys(), learnable: true, stacks: [] });

  expectError(
    () =>
      prepareNodeSucceeded(state, node.id, {
        evidence: evidence(node.memoryScope),
        forwardCapture: forwardCapture(node, { question: "q" }),
      }),
    RangeError,
    "node 'agent' has no recorded memory stack",
  );
});

test("an empty enabled table retains scope-free evidence without episodes", () => {
  const node = new TraceNode("agent");
  const { state } = committedTraceNode(node, { keys: [], learnable: true });

  const prepared = prepareNodeSucceeded(state, node.id, {
    evidence: evidence(null),
    forwardCapture: forwardCapture(node, { question: "q" }),
  });
  expect((prepared.record as NodeSucceededRecord).evidence!.scope).toBeNull();
  expectError(
    () =>
      prepareNodeSucceeded(state, node.id, {
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
  const { state } = committedTraceNode(node, { keys: keys(), learnable: true, stacks: stacks(scope(node.id)) });

  expectError(
    () => prepareNodeSucceeded(state, node.id, { forwardCapture: forwardCapture(node, {}) }),
    RangeError,
    "node 'agent' memory scope must carry its exact recorded memory stack",
  );
});

test("a replay rejects an eligible node scope that differs from its recorded stack", () => {
  const node = new TraceNode("agent");
  node.memoryScope = scope(node.id);
  const { state, log } = startTraceNode(node, { keys: keys(), learnable: true });
  const original = copyGraph(state.flow);
  commitPrepared(
    log,
    state,
    prepareOutput(log, state, node.id, "done", {
      evidence: evidence(node.memoryScope),
      forwardCapture: forwardCapture(node, { question: "q" }),
    }),
  );
  const replayed = replayRecords(new RunState(copyGraph(original)), log.records(), { growBuilder: fragmentBuilder });
  expect(replayed.nextSequence).toBe(state.nextSequence);

  original.nodes.get("agent")!.memoryScope = scope(node.id, { memories: ["outer"] });
  expectError(
    () => replayRecords(new RunState(original), log.records(), { growBuilder: fragmentBuilder }),
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

  const prepared = prepareOutput(log, state, node.id, "done", {
    evidence: evidence(node.memoryScope),
    forwardCapture: forwardCapture(node, { question: "q" }),
  });
  commitPrepared(log, state, prepared);

  const after = [state.memories, state.memoryStacks, state.memorySnapshots];
  after.forEach((root, index) => expect(root).toBe(roots[index]!));
  expect(state.trace!.occurrences.at(-1)!.memoryScope).toBe(node.memoryScope);
});

test("an expanded MAP occurrence projects and replays from the live node scope", async () => {
  const childAgent = new AgentNode("agent", { prompt: "visible" });
  const child = graph([childAgent], [
    [START_ID, childAgent.id],
    [childAgent.id, END_ID],
  ]);
  for (const [nodeId, node] of child.nodes) {
    node.memoryScope = new MemoryScope(["flow", "map", "child", nodeId], { memories: ["outer"] });
  }
  const spawner = new MapNode("map", { flowId: "child", child, childInputs: [] });
  spawner.memoryScope = new MemoryScope(["flow", "map"], { memories: ["outer"] });
  const { state, log } = startTraceNode(spawner, {
    keys: ["outer"],
    learnable: true,
    stacks: stacks(spawner.memoryScope, ...[...child.nodes.values()].map((node) => node.memoryScope!)),
  });
  const pristineFlow = copyGraph(state.flow);

  commitPrepared(
    log,
    state,
    prepareExpanded(state, spawner.id, new Grow([{}]), buildGrow, {
      forwardCapture: forwardCapture(spawner, { over: [{}] }),
    }),
  );

  const succeed = (nodeId: string, value: unknown, nodeEvidence: EvaluationEvidence | null = null) => {
    expect(state.ready[0]).toBe(nodeId);
    commitPrepared(log, state, prepareNodeStarted(state, nodeId));
    commitPrepared(
      log,
      state,
      prepareOutput(log, state, nodeId, value, {
        evidence: nodeEvidence,
        forwardCapture: forwardCapture(state.flow.nodes.get(nodeId)!, nodeEvidence?.boundInputs ?? {}),
      }),
    );
  };

  succeed("map/__start__", {});
  succeed("map#0/__start__", {});
  const dynamicScope = state.flow.nodes.get("map#0/agent")!.memoryScope!;
  expect(dynamicScope.constructor).toBe(MemoryScope);
  expect(dynamicScope.staticPath).toEqual(["flow", "map", "child", "agent"]);
  succeed("map#0/agent", "visible", evidence(dynamicScope));
  succeed("map#0/__end__", null);
  succeed("map/__end__", [null]);
  succeed(END_ID, null);
  expect(state.result).toBeInstanceOf(RunSucceeded);

  const episodes = (await contexts(state)).get("outer")!.episodes;
  const owned = episodes.filter((item) => comparePaths(item.ownerStaticPath, dynamicScope.staticPath) === 0);
  expect(owned).toHaveLength(1);
  expect(owned[0]!.occurrencePath).toEqual([
    new MemoryOccurrenceSegment(MemoryOccurrenceKind.MAP_ITEM, ["flow", "map"], 0),
  ]);
  expect(owned[0]!.interactions).toEqual([new ModelTextInteraction("answer")]);

  const replayed = replayRecords(new RunState(pristineFlow), log.records(), { growBuilder: buildGrow });
  expect((await contexts(replayed)).get("outer")!.episodes).toEqual(episodes);
});

// Owner contexts over engine runs.

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

/** A `RecordingBackwardNode` `work` that outputs `x` and sends its feedback back to `x`. */
function passthroughWork(reads: Readonly<Record<string, unknown>>): RecordingBackwardNode {
  return new RecordingBackwardNode("work", (inputs) => inputs.x, (_context, feedback) => new BackwardResult({
    inputFeedback: { x: feedback },
  }), { reads });
}

/** `START -> work -> END`, END reading `answer` from `work`. */
function workFlow(node: Node): CompiledFlow {
  return graph([node], [[START_ID, "work"]], { outputs: [new FlowOutput("answer", "${work.output}")] });
}

test("owner contexts include general occurrences and empty memories", async () => {
  const memories = [
    new MemorySpec("empty", "empty", {}, [new DeclSite([], ["def", "unused"])]),
    new MemorySpec("local", "local", {}, [new DeclSite([], ["root"], "work")]),
    new MemorySpec("outer", "outer", {}, [ROOT]),
  ];
  const node = new RecordingBackwardNode("work", (inputs) => inputs.topic, (_context, feedback) => new BackwardResult({
    inputFeedback: { topic: feedback, literal: ["keep literal"] },
    selfFeedback: ["first", "second"],
    variableFeedback: { tone: ["one", "two"] },
  }), { reads: { topic: "${input.topic}", literal: { default: [1n] } } });
  node.declaredVariables = () => [new Parameter("tone")];
  const flow = workFlow(node);
  for (const current of flow.nodes.values()) {
    current.memoryScope = new MemoryScope([current.id], {
      memories: current === node ? ["outer", "local"] : ["outer"],
    });
  }
  const recorded = stacks(...[...flow.nodes.values()].map((current) => current.memoryScope!));
  const built = engine(flow, {
    inputs: { topic: "reefs" },
    learnable: true,
    memories,
    memoryStacks: recorded,
    memorySnapshots: memories.map((spec) => new MemorySnapshotEnvelope(spec.key, {})),
  });
  await built.run();
  const trace = built.state.trace!;
  const [root, results] = await computeBackward(trace, "review");
  node.callback = () => {
    throw new Error("projection invoked backward");
  };

  const projected = learning.projectOwnerContexts(trace, memories, recorded, "review", root, results);

  expect(projected.map((context) => context.memory)).toEqual(["empty", "local", "outer"]);
  const byKey = new Map(projected.map((context) => [context.memory, context]));
  expect(byKey.get("outer")!.episodes).toHaveLength(3);
  expect(trace.occurrences).toHaveLength(3);
  expect(byKey.get("empty")!.episodes).toEqual([]);
  expect(byKey.get("empty")!.feedback).toEqual([]);
  expect(byKey.get("local")!.episodes).toHaveLength(1);
  const localEpisode = byKey.get("local")!.episodes[0]!;
  expect(localEpisode.inputs).toEqual({ topic: "reefs", literal: { default: [1n] } });
  expect(localEpisode.interactions).toEqual([]);
  const inherited = byKey.get("outer")!.episodes.find((episode) => episode.executionSeq === localEpisode.executionSeq)!;
  expect(inherited.ownerStaticPath).toEqual(["work"]);
  expect(localEpisode.ownerStaticPath).toEqual(["work"]);
  expect(inherited.occurrencePath).toEqual([]);
  expect(localEpisode.occurrencePath).toEqual([]);
  const entries = feedbackEntries(byKey.get("local")!);
  expect(entries.map((entry) => [entry.channel, entry.target_name, entry.value])).toEqual([
    ["output", null, "review"],
    ["input", "topic", "review"],
    ["input", "literal", "keep literal"],
    ["self", null, "first"],
    ["self", null, "second"],
    ["variable", "tone", "one"],
    ["variable", "tone", "two"],
  ]);
  const occurrence = trace.occurrences.find((item) => item.nodeId === "work")!;
  for (const entry of entries) {
    expect(entry.execution_seq).toBe(BigInt(occurrence.source.executionSeq));
    expect(entry.owner_static_path).toEqual(["work"]);
    expect(entry.occurrence_path).toEqual([]);
    expect(entry.execution_scope).toEqual({ static_path: ["work"], flow_path: [], occurrence_path: [], boundary: null });
  }
  expect(entries.slice(3).map((entry) => [entry.message_index, entry.contribution_context])).toEqual([
    [0n, {}],
    [1n, {}],
    [0n, {}],
    [1n, {}],
  ]);
  expect((entries[2]!.contribution_context as { contributions: unknown }).contributions).toEqual([]);
  expect(entries.filter((entry) => entry.channel === "stopped")).toEqual([]);
  // The entries and episodes are frozen, so the edits throw where Python's land on copies.
  expect(() => {
    (entries[0] as Record<string, unknown>).value = "mutated";
  }).toThrow(TypeError);
  expect(() => (localEpisode.inputs!.literal as { default: unknown[] }).default.push(2n)).toThrow(TypeError);
  expect(feedbackEntries(byKey.get("local")!)[0]!.value).toBe("review");
  expect(byKey.get("local")!.episodes[0]!.inputs!.literal).toEqual({ default: [1n] });
});

test.each([false, true])("owner feedback retains literals and opaque local stops (opaque: %p)", async (opaque) => {
  const calls: unknown[] = [];
  standIn("owner_identity", (value) => {
    calls.push(value);
    return value;
  }, [{ name: "value" }]);
  const node = passthroughWork({ x: opaque ? "${owner_identity(input.topic)}" : "literal" });
  const { engine: built, memories, stacks: recorded } = ownerEngine(workFlow(node), { inputs: { topic: "reef" } });
  await built.run();
  const trace = built.state.trace!;
  const [root, results] = await computeBackward(trace, "unchanged");
  const projected = learning.projectOwnerContexts(trace, memories, recorded, "unchanged", root, results);
  const work = feedbackEntries(projected[0]!).filter(
    (entry) => JSON.stringify(entry.owner_static_path) === JSON.stringify(["work"]),
  );
  const expected = [["output", "unchanged"], ["input", "unchanged"]];
  if (opaque) expected.push(["stopped", "unchanged"]);
  expect(work.map((entry) => [entry.channel, entry.value])).toEqual(expected);
  expect(work[1]!.target_name).toBe("x");
  expect((work[1]!.contribution_context as { contributions: unknown }).contributions).toEqual([]);
  if (opaque) {
    const stopped = work[2]!.contribution_context as { contexts: Record<string, unknown>[] };
    expect(Object.hasOwn(stopped, "contribution_index")).toBe(false);
    expect(stopped.contexts.some((detail) =>
      detail.role === "opaque" &&
      detail.result === "reef" &&
      (detail.sources as { value: unknown }[])[0]!.value === "reef"
    )).toBe(true);
  }
  expect(calls).toEqual(opaque ? ["reef"] : []);
  expect(root.inputFeedback).toEqual({});
});

test.each(["binding", "body", "route"] as const)(
  "owner contexts keep unavailable inputs and causal failures: %s",
  async (failureKind) => {
    let node: Node = new FuncNode("work", () => {
      throw new Error("expected body failure");
    }, { reads: { x: [1n] } });
    let flow = workFlow(node);
    if (failureKind === "binding") {
      // A binding failure stops before the body runs.
      node.params = [new Parameter("x", { required: true })];
      flow.wiring.set("work", new Map());
    } else if (failureKind === "route") {
      flow = terminalSkipRouteFlow();
      node = flow.nodes.get("router")!;
      node.forward = () => new Route("go");
      node.params = [new Parameter("x")];
      flow.wiring.set(node.id, new Map([["x", new Binding([1n])]]));
    }
    const { engine: built, memories, stacks: recorded } = ownerEngine(flow);
    await built.run();
    const trace = built.state.trace!;
    const [root, results] = await computeBackward(trace, "review failure");
    const [context, ...rest] = learning.projectOwnerContexts(trace, memories, recorded, "review failure", root, results);
    expect(rest).toEqual([]);
    const episode = context!.episodes.find(
      (item) => JSON.stringify(item.ownerStaticPath) === JSON.stringify([node.id]),
    )!;
    expect(episode.status).toBe(MemoryEpisodeStatus.FAILED);
    expect(episode.output).toBeNull();
    expect(episode.inputs).toEqual(failureKind === "binding" ? null : { x: [1n] });
    expect(episode.failure).toEqual(learning.failureJson(trace.terminal!.failure!));
    const prefix = { binding: "binding.", body: "node.", route: "graph." }[failureKind];
    expect((episode.failure!.code as string).startsWith(prefix)).toBe(true);
    const entries = feedbackEntries(context!).filter((entry) => entry.execution_seq === BigInt(episode.executionSeq!));
    expect(entries.map((entry) => [entry.channel, entry.value])).toEqual([["output", "review failure"]]);
    expect((entries[0]!.contribution_context as { contexts: unknown }).contexts).toEqual([]);
  },
);

test("owner context root default is owned once by the accepted start", async () => {
  const start = new StartNode(START_ID, {
    params: [new Parameter("topic", { hasDefault: true, default: "reef" })],
  });
  const end = EndNode.value(END_ID, { outputName: "answer" });
  const { engine: built, memories, stacks: recorded } = ownerEngine(
    graph([start, end], [[START_ID, END_ID]], { wiring: { [END_ID]: { answer: "${input.topic}" } } }),
  );
  await built.run();
  const trace = built.state.trace!;
  const [root, results] = await computeBackward(trace, "keep default");
  const [context] = learning.projectOwnerContexts(trace, memories, recorded, "keep default", root, results);
  expect(root.inputFeedback).toEqual({ topic: ["keep default"] });
  expect(feedbackEntries(context!).some((entry) => entry.root)).toBe(false);
  const startInputs = feedbackEntries(context!).filter(
    (entry) => entry.channel === "input" && entry.execution_seq === BigInt(trace.rootStartSource!.executionSeq),
  );
  expect(startInputs.map((entry) => [entry.target_name, entry.value])).toEqual([["topic", "keep default"]]);
});

/**
 * Run a flow of only START, memory key `POLICY` on its scope, START also the end, so root input
 * feedback falls back to the root.
 */
async function startOnlyRun(): Promise<{
  state: RunState;
  log: MemoryRunLog;
  memories: MemorySpec[];
  stacks: MemoryStack[];
}> {
  const start = new StartNode(START_ID, {
    params: [new Parameter("topic"), new Parameter("omitted", { hasDefault: true, default: "unused" })],
  });
  (start as Node).forward = () => new Route(ELSE_HANDLE);
  start.memoryScope = new MemoryScope([START_ID], { memories: [POLICY] });
  const flow = CompiledFlow.fromParts(new Map([[START_ID, start]]), []);
  // `endId` is readonly and fixed by the constructor.
  (flow as { endId: string }).endId = START_ID;
  const initialPool = new VariablePool(START_ID);
  initialPool.set(START_ID, { topic: "reef", undeclared: "extra" });
  initialPool.addSystem("run_id", RUN_ID);
  const state = new RunState(flow);
  const log = new MemoryRunLog(RUN_ID);
  const memories = [new MemorySpec(POLICY, POLICY, { limit: 3n }, [ROOT])];
  const recorded = stacks(start.memoryScope);
  commitPrepared(log, state, prepareRunStarted(state, {
    runId: RUN_ID,
    flowFingerprint: FINGERPRINT,
    startWiringOverride: {},
    effectiveLlmByStaticPath: {},
    numWorkers: 0,
    learnable: true,
    memories,
    memoryStacks: recorded,
    memorySnapshots: [new MemorySnapshotEnvelope(POLICY, {})],
  }));
  commitInitialPool(log, state, initialPool);
  await new FlowEngine(state, log, { growBuilder: noGrowth, numWorkers: 0 }).run();
  return { state, log, memories, stacks: recorded };
}

test("owner context root fallback uses only actual source-zero fields", async () => {
  const { state, memories, stacks: recorded } = await startOnlyRun();
  const [root, results] = await computeBackward(state.trace!, "review");
  const [context] = learning.projectOwnerContexts(state.trace!, memories, recorded, "review", root, results);
  expect(results.every((item) => !item.invoked)).toBe(true);
  expect(context!.episodes).toHaveLength(1);
  expect(feedbackEntries(context!).map((entry) => [entry.target_name, entry.value])).toEqual([["topic", "review"]]);
  expect(feedbackEntries(context!).every((entry) => entry.root && entry.execution_seq === null)).toBe(true);
});

test("an owner context error identifies the receiver without changing forward acceptance", async () => {
  const node = passthroughWork({ x: "leaf" });
  const { engine: built, memories, stacks: recorded } = ownerEngine(workFlow(node));
  await built.run();
  const trace = built.state.trace!;
  expect(trace.terminal!.succeeded).toBe(true);
  expect(trace.terminal!.output).toBe("leaf");
  const [root, results] = await computeBackward(trace, "review");
  expect(learning.projectOwnerContexts(trace, memories, recorded, "review", root, results)).toHaveLength(1);
  const accepted = built.log.records();
  // TS has no memory JSON nesting bound, so the context check of `work`'s occurrence (the second
  // call, after START's) refuses instead.
  const original = new RangeError("work material breaks the memory contract");
  const fromFrozen = LearnContext.fromFrozen;
  let calls = 0;
  const spy = spyOn(LearnContext, "fromFrozen").mockImplementation((...args) => {
    calls += 1;
    if (calls === 2) throw original;
    return fromFrozen.apply(LearnContext, args);
  });
  let caught: unknown;
  try {
    learning.projectOwnerContexts(trace, memories, recorded, "review", root, results);
  } catch (error) {
    caught = error;
  } finally {
    spy.mockRestore();
  }
  expect(caught).toBeInstanceOf(learning.OwnerContextError);
  const failure = caught as learning.OwnerContextError;
  expect(failure.memory).toBe(memories[0]!.key);
  expect(failure.original).toBe(original);
  expect(failure.cause).toBe(original);
  expect(failure.message).toBe(original.message);
  const work = trace.occurrences.find((item) => item.nodeId === "work")!;
  expect(failure.context).toEqual(trace.backwardContext(work.source));
  expect(built.log.records()).toEqual(accepted);
  expect(trace.terminal!.output).toBe("leaf");
});

test("a root fallback context failure names no node", async () => {
  const { state, log, memories } = await startOnlyRun();
  // The second call checks the root fallback: the first checks START's occurrence.
  const original = new RangeError("boom");
  const fromFrozen = LearnContext.fromFrozen;
  let calls = 0;
  const spy = spyOn(LearnContext, "fromFrozen").mockImplementation((...args) => {
    calls += 1;
    if (calls === 2) throw original;
    return fromFrozen.apply(LearnContext, args);
  });
  let outcome: BackwardOutcome;
  try {
    outcome = await driveBackward(state, log, "review", {
      openMemory: () =>
        MemoryRuntime.open({
          memories,
          origins: new Map(memories.map((spec) => [spec.key, null])),
          memory: new Map([[POLICY, new FakeMemory()]]),
          snapshots: memories.map((spec) => new MemorySnapshotEnvelope(spec.key, {})),
        }),
    });
  } finally {
    spy.mockRestore();
  }
  const failure = outcome.failure!;
  expect(failure.code).toBe("memory.invalid_context");
  expect(failure.memory).toBe(POLICY);
  expect(failure.message).toBe("memory learn context could not be constructed: boom");
  expect([failure.scope, failure.runId, failure.nodeId, failure.nodeStartedSeq]).toEqual([null, null, null, null]);
  expect(failure.cause!.message).toBe(failure.message);
});

test("owner context authority errors are not context construction failures", async () => {
  const node = passthroughWork({ x: "${input.topic}" });
  const { engine: built, memories, stacks: recorded } = ownerEngine(workFlow(node), { inputs: { topic: "reef" } });
  await built.run();
  const trace = built.state.trace!;
  const [root, results] = await computeBackward(trace, "review");
  const project = (
    table: readonly MemorySpec[],
    stackList: readonly MemoryStack[],
    attributed: readonly AttributedBackwardResult[] = results,
  ) => learning.projectOwnerContexts(trace, table, stackList, "review", root, attributed);

  expectError(
    () => project(memories, recorded, results.slice(0, -1)),
    RangeError,
    "backward results must match the complete accepted occurrence order",
  );
  expectError(
    () => project([], [], results),
    RangeError,
    "historical memory scope disagrees with its recorded memory stack",
  );
  for (const invalid of [null, new Set(memories)]) {
    expectError(
      () => project(invalid as unknown as MemorySpec[], recorded),
      TypeError,
      "owner contexts require the complete recorded memory table",
    );
  }
  expectError(
    () => project([...memories, ...memories].reverse(), recorded),
    RangeError,
    "owner contexts require a memory table sorted by unique key",
  );
  expectError(
    () => project(memories, new Set(recorded) as unknown as MemoryStack[]),
    TypeError,
    "owner contexts require the recorded memory stacks",
  );
  expectError(
    () => project(memories, [...recorded, recorded[0]!]),
    RangeError,
    "recorded memory stacks must have unique locators",
  );
  const missing = new Error("missing source authority");
  const lookupSource = trace.lookupSource.bind(trace);
  trace.lookupSource = (source) => {
    if (source.path.length > 0) throw missing;
    return lookupSource(source);
  };
  expect(() => project(memories, recorded)).toThrow(missing);
});

test("owner context projection from replay doesn't reinvoke handlers", async () => {
  const node = passthroughWork({ x: "${input.topic}" });
  const { engine: built, memories, stacks: recorded } = ownerEngine(workFlow(node), { inputs: { topic: "reef" } });
  const original = copyGraph(built.state.flow);
  await built.run();
  const [root, results] = await computeBackward(built.state.trace!, "review");
  const expected = learning.projectOwnerContexts(built.state.trace!, memories, recorded, "review", root, results);
  for (const current of original.nodes.values()) {
    current.forward = () => {
      throw new Error("replayed forward");
    };
    current.backward = () => {
      throw new Error("replayed backward");
    };
    current.captureBackwardContext = () => {
      throw new Error("recaptured backward");
    };
  }
  const replayed = replayRecords(new RunState(original), built.log.records(), { growBuilder: noGrowth });
  Object.defineProperty(replayed.trace!, "backwardHandlers", {
    get() {
      throw new Error("read callback");
    },
  });
  expect(learning.projectOwnerContexts(replayed.trace!, memories, recorded, "review", root, results)).toEqual(expected);
});

test("an owner context of an empty MAP retains the growth and the unexecuted child memory", async () => {
  const child = graph([new FuncNode("work", (inputs) => inputs["x"], { reads: { x: "${input.x}" } })], [
    [START_ID, "work"],
  ], { outputs: [new FlowOutput("answer", "${work.output}")] });
  child.memories = [new MemoryDecl("child_notes", "child_notes", {}, new DeclSite([], ["def", "child"], null))];
  const mapper = new MapNode("map", { flowId: "child", child });
  mapper.params = [new Parameter("x")];
  const { engine: built, memories, stacks: recorded } = ownerEngine(
    graph([mapper], [[START_ID, "map"]], {
      wiring: { map: { over: [], x: "${item}" } },
      outputs: [new FlowOutput("answer", "${map.output}")],
    }),
    { growBuilder: buildGrow },
  );
  await built.run();
  const trace = built.state.trace!;
  const [root, results] = await computeBackward(trace, null);
  const [childContext, outer] = learning.projectOwnerContexts(trace, memories, recorded, null, root, results);

  expect(outer!.episodes).toHaveLength(5);
  expect(trace.occurrences).toHaveLength(5);
  expect(outer!.feedback).toEqual([]);
  expect(childContext!.episodes).toEqual([]);
  expect(childContext!.feedback).toEqual([]);
  const grown = outer!.episodes.filter((episode) =>
    isDeepStrictEqual(episode.output, { kind: "grow", seed: [], prune_ids: [] }),
  );
  expect(grown).toHaveLength(1);
  expect(grown[0]!.ownerStaticPath).toEqual(["map"]);
});

test.each([false, true])(
  "owner feedback uses recorded execution coordinates independently of memory (scope available: %p)",
  async (scopeAvailable) => {
    const node = new TraceNode("agent");
    node.memoryScope = scope(node.id);
    node.executionScope = scopeAvailable
      ? new ExecutionScope(["authored", "worker"], {
        flowPath: ["authored"],
        occurrencePath: [new ExecutionPosition("map_item", ["authored"], 3)],
      })
      : null;
    node.backward = () => new BackwardResult({ selfFeedback: ["local"] });
    const { state, log } = startTraceNode(node, { keys: node.memoryScope.memories, learnable: true });
    commitPrepared(
      log,
      state,
      prepareNodeFailed(state, new Failure({ code: "node.execute", message: "failed", nodeId: node.id }), {
        evidence: evidence(node.memoryScope),
        forwardCapture: forwardCapture(node, { question: "q" }),
      }),
    );
    const occurrenceOf = state.trace!.occurrences.find((item) => item.nodeId === "agent")!;
    expect(occurrenceOf.record.forwardCapture!.scope).toEqual(node.executionScope);
    node.executionScope = new ExecutionScope(["changed"]);
    const [root, results] = await computeBackward(state.trace!, "review");
    const projected = learning.projectOwnerContexts(
      state.trace!,
      state.memories!,
      state.memoryStacks!,
      "review",
      root,
      results,
    );
    for (const context of projected) {
      for (const entry of feedbackEntries(context)) {
        expect(entry.owner_static_path).toEqual(scopeAvailable ? ["authored", "worker"] : null);
        expect(entry.occurrence_path).toEqual(
          scopeAvailable ? [{ kind: "map_item", static_path: ["authored"], index: 3n }] : null,
        );
        if (scopeAvailable) {
          const executionScope = entry.execution_scope as Record<string, unknown>;
          expect(entry.owner_static_path).toEqual(executionScope.static_path);
          expect(entry.occurrence_path).toEqual(executionScope.occurrence_path);
        } else {
          expect(entry.execution_scope).toBeNull();
        }
      }
    }
    for (const context of projected) {
      expect(context.episodes).toHaveLength(1);
      const episode = context.episodes[0]!;
      expect(episode.inputs).toEqual({ question: "q" });
      expect(episode.ownerStaticPath).toEqual(["flow", "agent"]);
      expect(episode.occurrencePath).toEqual(occurrence());
      expect(episode.interactions).toEqual([new ModelTextInteraction("answer")]);
      expect(episode.baseSystemPrompt).toBe("system");
      expect(episode.prompt).toBe("prompt");
    }
  },
);

test.each([false, true])(
  "owner variable batches follow the captured inventory across a wire round trip (reverse targets: %p)",
  async (reverseTargets) => {
    let batches: Record<string, string[]> = { zeta: ["z2", "z1"], alpha: ["a2", "a1"] };
    if (reverseTargets) batches = Object.fromEntries(Object.entries(batches).reverse());
    const node = new RecordingBackwardNode("work", () => "answer", () => new BackwardResult({
      selfFeedback: ["second", "first"],
      variableFeedback: batches,
    }));
    node.declaredVariables = () => [new Parameter("zeta"), new Parameter("alpha")];
    const { engine: built, memories, stacks: recorded } = ownerEngine(workFlow(node));
    await built.run();
    const trace = built.state.trace!;
    const [root, results] = await computeBackward(trace, "review");
    node.declaredVariables = () => {
      throw new Error("read mutable declarations");
    };
    const projected = learning.projectOwnerContexts(trace, memories, recorded, "review", root, results);
    const entries = feedbackEntries(projected[0]!);
    expect(
      entries.filter((entry) => entry.channel === "variable").map((entry) => [
        entry.target_name,
        entry.value,
        entry.message_index,
      ]),
    ).toEqual([["zeta", "z2", 0n], ["zeta", "z1", 1n], ["alpha", "a2", 0n], ["alpha", "a1", 1n]]);
    expect(entries.filter((entry) => entry.channel === "self").map((entry) => [entry.value, entry.message_index]))
      .toEqual([["second", 0n], ["first", 1n]]);
    for (const entry of entries.filter((item) => item.channel === "self" || item.channel === "variable")) {
      expect(entry.contribution_context).toEqual({});
    }
    const ready = new BackwardReadyRecord(1, root, results, []);
    const decoded = decodeRecord(encodeRecord(new RecordEnvelope(SCHEMA_VERSION, built.state.runId!, 0, ready)))
      .record as BackwardReadyRecord;
    expect(
      learning.projectOwnerContexts(trace, memories, recorded, "review", decoded.rootResult, decoded.attributedResults),
    ).toEqual(projected);
  },
);
