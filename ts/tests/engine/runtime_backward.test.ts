/**
 * The accepted reverse walk: a port of `tests/engine/test_runtime_backward.py`. Ported:
 * `test_explicitly_synthetic_source_free_failure_keeps_direct_root_text` (it also refuses another
 * root result).
 *
 * Changed: Python's `ValueError` is a `RangeError`; the results are an array, not a tuple.
 *
 * TS-only, over `chainTrace()` (START, `p` and END driven through `prepareTrace` and `apply`):
 * `validateAttributedBackward` accepts the exact transfers END → p → START → source 0, and merges
 * the root inputs when START is reached and `p` also routes to source 0; one refusal per rule
 * (inventory order, feedback, attribution, a reached occurrence's result that isn't invoked (the
 * feedback check refuses it, since a result that isn't invoked has no feedback), a local contract
 * violation, the root, feedback to an unavailable occurrence, a non-terminal trace); refusals pin
 * the error's exact class and message; empty feedback seeds nothing; a node-data
 * path absent from the accepted output; an int-headed path's output context and its materialized
 * form; `materializeContext`'s shape and `bigint` integers; `ContextCache.lookup` looking a source
 * up once and not caching a failure; `takeMessages` order.
 *
 * TS-only, backward attempt state: one refusal per `BackwardAttempt` and `RuntimeBackwardState`
 * rule; the getters on empty, unfinished and finished histories; the unfinished outcome an
 * unfinished attempt derives, with `prefixLearning`'s statuses and failure order;
 * `proposalBatches`' table order; and `sameRuntimeBackwardState`, true for separately built equal
 * states and false for a different memory order, a different last record and a `null` versus
 * non-`null` `ready`.
 *
 * TS-only, the reducer over `chainTrace()` with memory key `notes` and ids from `memoryCommitId`:
 * a full attempt Started → Ready → Requested → Succeeded → Finished, then a second start; the
 * accepted Finished of each branch (unfinished before and after Ready, commit failure, learner
 * failure with and without an attributed context failure, computation failure); and one refusal
 * per reducer, Finished and context-owner rule, the last record class refused through a forged
 * record of another class.
 *
 * TS-only, owner learn contexts over `chainTrace()` (`p` sends `v`, `w`, self and variable
 * feedback): the producer's episode (inputs without the hidden `w`, output, coordinates) and
 * `notes` feedback entries (output, input with contributions, self and variable channels) with
 * `bigint` integers; the coordinates every entry carries when `p` has an execution scope with an
 * indexed position; an empty context for a key with no occurrences; the root fallback for a
 * root-owned key; one refusal per table, stack and scope check; the two `OwnerContextError`
 * constructor checks; context failures: a lone surrogate in `p`'s self feedback, which
 * `freezeFeedback` refuses (the TS message doesn't name `feedback[i]`), names `p`'s backward
 * context, and a synthetic source-free trace with run id `""` fails the root fallback's context
 * with `context === null`.
 *
 * Ported, after rewrite (the start is built with `prepareRunStarted` on the shared
 * `learnableStarted`, not the engine; the trace compares through the shared `traceSnapshot`):
 * `test_root_inventory_is_retained_without_running_capture_or_defaults` (START's params are
 * replaced with an empty array, which is readonly, where Python clears the list),
 * `test_terminal_trace_append_rejection_leaves_all_indexes_unchanged` and
 * `test_causal_route_trace_latch_is_installed_only_after_append` (the log's `append` patched on
 * the instance, its error an `Error`, Python's `OSError`).
 *
 * Ported, through the engine: `test_terminal_trace_keeps_actual_root_writer_until_quiescence`
 * (2 rows), `test_diamond_calls_shared_producer_once_with_two_ordered_messages` (the delivered
 * messages are frozen),
 * `test_entire_callback_result_is_validated_before_routing_any_input` (3 rows; each pins the
 * failure's message and cause class),
 * `test_empty_feedback_keeps_complete_history_without_invoking_callbacks` (2 rows; the `[null]`
 * refusal's full message is pinned), `test_root_default_feedback_survives_its_literal_route`,
 * `test_accepted_failure_uses_recorded_inputs_and_diagnostics` (2 rows; the missing parameter is
 * added by replacing `params`, the body's error is an `Error`, Python's `RuntimeError`) and
 * `test_failed_end_routes_text_without_fabricating_output_context`.
 *
 * `test_callback_failures_are_attributed_without_swallowing_host_faults`: `callback` and
 * `infrastructure` (the trace's `lookupSource` patched on the instance) are ported; the callback
 * row also pins the whole `BackwardFailure` and `BackwardComputeError`'s name, message and `cause`.
 * Changed: `host`: there is no host-control unwrapping, so a `CapabilityFault` a callback throws is
 * `backward.execute`, its `cause` the fault. Not ported: `cancel` and `memory`; TS has no
 * `_HostStop` or `MemoryError`.
 *
 * Changed: `test_callback_message_mutation_does_not_change_root_or_recorded_feedback`: the
 * callback's feedback is a deep-frozen copy, so its write throws a `TypeError` (asserted), and the
 * caller's message and the recorded feedback stay unchanged.
 *
 * TS-only, `computeBackward`: a result of another class is `backward.result` with a `TypeError`
 * cause; an async callback is awaited; a rejected promise is `backward.execute`; a thrown `"x"` or
 * `undefined` gives its text as the message, its type name as `causeType` and a `null` traceback;
 * an `Error` subclass's `name` is the `causeType` and its string `notes` follow the stack in the
 * traceback; a missing handler (the trace's handler map shadowed on the instance) rejects with its
 * own `Error`, not `backward.execute`; an error from result validation that is neither a
 * `TypeError` nor a `RangeError` (a forged result's throwing getter) escapes unchanged.
 *
 * Ported, through the engine, with routes, replay and template stops:
 * `test_terminal_trace_latches_causal_route_and_later_failure_wins` (2 rows; `router` and
 * `unrelated` patched on the instance, every prefix replayed over a fresh `copyGraph` of the
 * started flow with `noGrowth`) and
 * `test_opaque_expression_stops_are_recoverable_without_rerunning_functions` (the template function
 * registered with `registerTemplateFn` and restored in `afterEach`).
 *
 * Changed:
 * - `test_structural_ancestry_requires_exact_metadata_prescribed_transfer` (2 rows) and
 *   `test_canonical_file_input_feedback_keeps_literal_and_opaque_stops` (2 rows) run on
 *   `ownerEngine`'s hand-built root-only memory layout, not Python's resolved one; `middle`'s
 *   `captureBackwardContext` is patched on the instance.
 * - `test_preseeded_root_output_does_not_invent_an_executed_producer`: only the row where END is
 *   START, whose output is the run inputs' commit (source 0); a run start commits only the
 *   system values and the START inputs, so END can't be preseeded. The flow's readonly `endId` is
 *   assigned through a cast, and the engine grows with `noGrowth`.
 * - `test_backward_context_is_trace_owned_detached_and_survives_replay`: the context is frozen, so
 *   writing to its inputs throws a `TypeError` (asserted) where Python's write lands on a detached
 *   copy; the replay's recapture guard spies on `Node.prototype.captureBackwardContext`.
 *
 * TS-only: a backward context the trace can't build rejects with its own error before the callback
 * runs.
 *
 * Not ported: `test_pooled_reverse_walk_uses_outcomes_instead_of_start_sequence`. The Python test
 * blocks a template function on a thread event until the peer commits; a TS template function
 * binds synchronously at dispatch, so a consumer that started first can't block until a peer
 * commits and then bind its output.
 *
 * Ported, through CASE and AGENT: `test_case_control_and_unselected_branch_do_not_receive_feedback`
 * and `test_agent_inventory_is_recorded_and_empty_feedback_does_no_backward_work` (AGENT's
 * `forward` and `backward` are `spyOn` replacements on the prototype, restored in `finally`; the
 * variables are read with `parametersFromYaml` over an empty `readTypedefs({})`; the params and
 * variables are cleared by assigning empty arrays; the engine records no effective LLM map, as the
 * replaced `forward` calls no model).
 *
 * Ported, through CALL, MAP and LOOP growth (`engine` or `ownerEngine` with `growBuilder:
 * buildGrow`, Python's `_engine` default; `loadFlow` where Python loads YAML; Python's class-level
 * `CodeNode` patches are `spyOn` replacements on the prototype, restored in `finally`):
 * `test_diamond_crossing_call_uses_one_walk_and_preserves_child_default_projection`,
 * `test_loop_carried_routes_use_the_same_fold_after_live_pruning`,
 * `test_map_canonical_item_shape_preserves_its_projected_seed_stop` (2 rows; the template function
 * registered with `registerTemplateFn`) and `test_loop_passthrough_field_routes_feedback_to_its_seed`
 * (12 rows, shape × times 1, 2, 3, 6). The inline CODE bodies keep Python's text, which is also
 * valid TS; the feedback entries' integers are `bigint`s.
 *
 * Changed:
 * - `test_map_whole_list_context_and_item_order_survive_reversed_completion`: Python's thread
 *   event is a promise the `onEvent` observer resolves when `map#1`'s END succeeds; item 0's
 *   `forward` waits on it for at most 2 s (Python's 5 s, under Bun's 5 s test timeout), then
 *   rejects, so the run ends and the spies are restored.
 * - `test_empty_map_retains_synthetics_without_inventing_item_contributors`: the child body that
 *   must not run throws an `Error`, Python's `AssertionError`.
 * - `test_map_bounded_projected_file_lists_compose_before_stopping` (2 rows): `+ []` is ported;
 *   Python's `* 2` is list repetition, a type error in TS, so the second row adds the same 40-deep
 *   list to itself, with the same two-element feedback and the same single stop.
 *
 * Changed: an unfinished backward attempt fails `backward.unfinished`, where Python says
 * `backward.interrupted`.
 */

import { afterEach, expect, spyOn, test } from "bun:test";
import { isDeepStrictEqual } from "node:util";

import { buildGrow } from "../../src/compile/expand.ts";
import { CompiledFlow, copyGraph, END_ID, FlowOutput, START_ID } from "../../src/compile/model.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import { registerTemplateFn, TEMPLATE_FNS, type TemplateFn, type TemplateParam } from "../../src/expr/builtins.ts";
import { Failure } from "../../src/diagnostics.ts";
import { NodeSucceeded } from "../../src/events.ts";
import {
  type FeedbackValue,
  RouteContext,
  routeFeedback,
  SourceRef,
  sameSource,
  sourceKey,
  ValueRoute,
} from "../../src/expr/lineage.ts";
import {
  DeclSite,
  LearnContext,
  LearningFailure,
  LearningResult,
  MemoryDecl,
  MemoryEpisode,
  MemoryEpisodeStatus,
  MemoryMutation,
  MemoryOperationFailure,
  MemoryScope,
  MemorySpec,
  MemoryStack,
  PolicyCommitOutcome,
} from "../../src/memory/contracts.ts";
import {
  BackwardContext,
  BackwardResult,
  CapabilityFault,
  ELSE_HANDLE,
  Node,
  NodeKind,
  Output,
  Route,
} from "../../src/nodes/base.ts";
import { AgentNode } from "../../src/nodes/agent/node.ts";
import { CallNode } from "../../src/nodes/call/node.ts";
import { Case, CaseNode } from "../../src/nodes/case/node.ts";
import { CodeNode } from "../../src/nodes/code/node.ts";
import { Assertion, ExecutionPosition, ExecutionScope, Parameter } from "../../src/nodes/contracts.ts";
import { EndNode } from "../../src/nodes/end/node.ts";
import { MapNode } from "../../src/nodes/map/node.ts";
import { StartNode } from "../../src/nodes/start/node.ts";
import { BackwardComputeError, computeBackward } from "../../src/runtime/backward.ts";
import { FlowEngine } from "../../src/runtime/engine.ts";
import {
  AttributedBackwardResult,
  BackwardAttribution,
  BackwardFailure,
  BackwardOutcome,
  ForwardCapture,
  MemoryProposal,
} from "../../src/runtime/contracts.ts";
import {
  BackwardAttempt,
  BackwardMessage,
  type BackwardRecord,
  ContextCache,
  contextualTransfers,
  materializeContext,
  memoryCommitId,
  OwnerContextError,
  prefixLearning,
  projectOwnerContexts,
  proposalBatches,
  reduceBackwardRecord,
  RuntimeBackwardState,
  sameRuntimeBackwardState,
  seedBackward,
  takeMessages,
  validateAttributedBackward,
} from "../../src/runtime/learning.ts";
import {
  BackwardFinishedRecord,
  BackwardReadyRecord,
  BackwardStartedRecord,
  canonicalTyped,
  MemoryCommitFailedRecord,
  MemoryCommitRequestedRecord,
  MemoryCommitSucceededRecord,
  MemoryRunLog,
  NodeStartedRecord,
  NodeSucceededRecord,
  PoolCommitRecord,
  RecordEnvelope,
  routePayload,
  type RunRecord,
  RunStartedRecord,
  SCHEMA_VERSION,
} from "../../src/runtime/log.ts";
import {
  commitInitialPool,
  commitPrepared,
  type PreparedTransition,
  prepareNodeRouted,
  prepareNodeStarted,
  prepareRunStarted,
  replayRecords,
  RunState,
} from "../../src/runtime/state.ts";
import { type PrepareTraceOptions, prepareTrace, RetainedTrace, TraceTerminal } from "../../src/runtime/trace.ts";
import { FileRef, type Value, VariablePool } from "../../src/typesys/index.ts";
import { type JsonObject, logJson } from "../../src/typesys/json.ts";
import { readTypedefs } from "../../src/typesys/types.ts";
import { parametersFromYaml } from "../compose/declarations.ts";
import {
  commitStartSuccess,
  engine,
  feedbackEntries,
  forwardCapture,
  FuncNode,
  graph,
  learnableStarted,
  noGrowth,
  ownerEngine,
  POLICY,
  prepareOutput,
  RecordingBackwardNode,
  rejected,
  RUN_ID,
  terminalSkipRouteFlow,
  traceSnapshot,
} from "./run_state_fixtures.ts";

/** The error classes the refusals throw; a failing log's append throws an `Error`. */
type ErrorClass = typeof TypeError | typeof RangeError | typeof Error;

/** Assert that `fn` throws exactly `cls` with exactly `message`. */
function expectError(fn: () => unknown, cls: ErrorClass, message: string): void {
  let caught: unknown;
  try {
    fn();
  } catch (error) {
    caught = error;
  }
  expect((caught as Error | undefined)?.constructor).toBe(cls);
  expect((caught as Error).message).toBe(message);
}

/** A terminal trace of run `runId` with no occurrences and a synthetic source-free failure. */
function syntheticTrace(runId: string): RetainedTrace {
  const trace = new RetainedTrace(runId, START_ID, END_ID);
  trace.terminal = new TraceTerminal(
    false,
    null,
    null,
    new Failure({ code: "graph.terminal_unreachable", message: "no executing producer", nodeId: null }),
  );
  return trace;
}

test("an explicitly synthetic source-free failure keeps the direct root text", async () => {
  const trace = syntheticTrace("synthetic");
  const [root, results] = await computeBackward(trace, "original text");
  expect(() => validateAttributedBackward(trace, "original text", root, results)).not.toThrow();
  expect(root).toEqual(new BackwardResult({ selfFeedback: ["original text"] }));
  expect(results).toEqual([]);
  expectError(
    () => validateAttributedBackward(trace, "original text", new BackwardResult(), []),
    RangeError,
    "backward root result must match the accepted START and source-zero projection",
  );
});

// The run-state reducer's trace, on `learnableStarted`.

test("the root inventory is retained without running a capture or defaults", () => {
  const start = new StartNode(START_ID, { params: [new Parameter("topic", { hasDefault: true, default: [1n] })] });
  start.captureBackwardContext = () => {
    throw new Error("initialization captured an invocation");
  };
  const { state } = learnableStarted(graph([start], [[START_ID, END_ID]]));
  const trace = state.trace!;

  expect(trace.rootInputNames).toEqual(["topic"]);
  expect(trace.rootStartSource).toBeNull();
  expect(trace.lookupSource([...trace.sources.values()][0]!.source!)).toEqual({});
  start.params = [];
  expect(trace.rootInputNames).toEqual(["topic"]);
});

test("a terminal trace append rejection leaves every index unchanged", () => {
  const { state, log } = learnableStarted(graph([], [[START_ID, END_ID]]));
  let prepared: PreparedTransition | null = null;
  for (const [nodeId, value] of [
    [START_ID, {}],
    [END_ID, null],
  ] as const) {
    commitPrepared(log, state, prepareNodeStarted(state, nodeId));
    prepared = prepareOutput(log, state, nodeId, value, {
      forwardCapture: forwardCapture(state.flow.nodes.get(nodeId)!, {}),
    });
    if (nodeId === START_ID) commitPrepared(log, state, prepared);
  }
  const before = traceSnapshot(state.trace!);
  const append = log.append;

  log.append = () => {
    throw new Error("reject terminal");
  };
  expectError(() => commitPrepared(log, state, prepared!), Error, "reject terminal");
  expect(traceSnapshot(state.trace!)).toEqual(before);

  log.append = append;
  commitPrepared(log, state, prepared!);
  expect(state.trace!.terminal!.succeeded).toBe(true);
});

test("a causal route's trace latch is installed only after append", () => {
  const { state, log } = learnableStarted(terminalSkipRouteFlow({ sibling: true }));
  commitPrepared(log, state, prepareNodeStarted(state, START_ID));
  commitStartSuccess(state, log, {});
  for (const nodeId of ["router", "unrelated"]) commitPrepared(log, state, prepareNodeStarted(state, nodeId));
  const prepared = prepareNodeRouted(state, "router", "go", {
    forwardCapture: forwardCapture(state.flow.nodes.get("router")!, {}),
  });
  const before = traceSnapshot(state.trace!);
  const append = log.append;

  log.append = () => {
    throw new Error("reject causal route");
  };
  expectError(() => commitPrepared(log, state, prepared), Error, "reject causal route");
  expect(traceSnapshot(state.trace!)).toEqual(before);

  log.append = append;
  commitPrepared(log, state, prepared);
  expect(state.trace!.terminal).toBeNull();
  const causal = state.trace!.occurrences.at(-1)!.source;
  commitPrepared(
    log,
    state,
    prepareOutput(log, state, "unrelated", "done", {
      forwardCapture: forwardCapture(state.flow.nodes.get("unrelated")!, {}),
    }),
  );
  expect(state.trace!.terminal!.source).toEqual(causal);
});

// TS-only: the chain fixture.

const RUN = "run";

/** A CODE node with a settable memory scope and hidden inputs, recording its backward calls. */
class TraceNode extends Node {
  readonly kind = NodeKind.CODE;
  /** The inputs `modelHiddenInputs` reports. */
  hidden: ReadonlySet<string> = new Set();
  /** Each backward call's context and feedback, in call order. */
  readonly calls: [BackwardContext, readonly FeedbackValue[]][] = [];

  override modelHiddenInputs(): ReadonlySet<string> {
    return this.hidden;
  }

  override backward(context: BackwardContext, feedback: readonly FeedbackValue[]): BackwardResult {
    this.calls.push([context, feedback]);
    return new BackwardResult();
  }
}

/** A source route reading `path` of execution `seq`. */
function sourceRoute(seq: number, path: (string | number)[] = []): ValueRoute {
  return new ValueRoute("source", { source: new SourceRef(RUN, seq, path), expression: `\${source_${seq}}` });
}

/** The canonical typed wire of `value`. */
function wire(value: unknown) {
  return canonicalTyped(value, null)[0];
}

/** An envelope of `record` at `seq`. */
function at(seq: number, record: RunRecord): RecordEnvelope {
  return new RecordEnvelope(SCHEMA_VERSION, RUN, seq, record);
}

/** Prepare `envelope`, check the delta's origin and apply it. */
function step(
  trace: RetainedTrace | null,
  envelope: RecordEnvelope,
  node: Node | null = null,
  options: PrepareTraceOptions = {},
): RetainedTrace {
  const delta = prepareTrace(trace, envelope, node, options)!;
  delta.validateOrigin(trace);
  return delta.apply();
}

/** The chain's occurrences: START (seq 2), `p` (seq 5) and END (seq 8). */
const S = new SourceRef(RUN, 2);
const P = new SourceRef(RUN, 5);
const E = new SourceRef(RUN, 8);
const ZERO = new SourceRef(RUN, 0);

/** The options of `chainTrace`. */
interface ChainOptions {
  /** START's node-data `input_output_paths`; defaults to `{x: ["x"]}`. */
  startPaths?: Record<string, (string | number)[]>;
  /** The value `p` writes, and END and the run output; defaults to `2n`. */
  pOutput?: unknown;
  /** `p`'s node-data `input_output_paths`, or `null` (the default) for no node data. */
  pPaths?: Record<string, (string | number)[]> | null;
  /** `p`'s execution scope, set on the node and its capture; defaults to `null`. */
  pScope?: ExecutionScope | null;
}

/**
 * A terminal trace of a learnable run, driven through `prepareTrace` and `apply`: RunStarted (START
 * `__start__` writes `{x: 1n}`, root END `end`, root input names `["x"]`); START, whose capture
 * routes `x` to source 0 path `["x"]`; `p`, routing `v` from START's output `x` and `w` straight to
 * source 0 path `["x"]`, with variable `tone`, hidden input `w`, memory scope `["p"]` with key
 * `notes`, and execution scope `pScope`; END, routing `value` from `p`, with the successful
 * terminal. Returns the trace, the nodes and the envelopes.
 */
function chainTrace({ startPaths = { x: ["x"] }, pOutput = 2n, pPaths = null, pScope = null }: ChainOptions = {}) {
  const run = new RunStartedRecord({
    flowFingerprint: "a".repeat(64),
    startWiringOverride: {},
    effectiveLlmByStaticPath: {},
    numWorkers: 0,
    learnable: true,
  });
  const startCapture = new ForwardCapture({ x: 1n }, ValueRoute.record([["x", sourceRoute(0, ["x"])]]), {
    inputNames: ["x"],
    nodeData: { input_output_paths: startPaths },
  });
  const pCapture = new ForwardCapture(
    { v: 1n, w: 1n },
    ValueRoute.record([
      ["v", sourceRoute(2, ["x"])],
      ["w", sourceRoute(0, ["x"])],
    ]),
    {
      inputNames: ["v", "w"],
      variableNames: ["tone"],
      nodeData: pPaths === null ? null : { input_output_paths: pPaths },
      scope: pScope,
    },
  );
  const endCapture = new ForwardCapture({ value: pOutput }, ValueRoute.record([["value", sourceRoute(5)]]), {
    inputNames: ["value"],
  });
  const nodes = { start: new TraceNode(START_ID), p: new TraceNode("p"), end: new TraceNode(END_ID) };
  nodes.p.memoryScope = new MemoryScope(["p"], { memories: ["notes"] });
  nodes.p.hidden = new Set(["w"]);
  nodes.p.executionScope = pScope;
  const envelopes = [
    at(0, run),
    at(1, new PoolCommitRecord(START_ID, START_ID, wire({ x: 1n }))),
    at(2, new NodeStartedRecord(START_ID)),
    at(3, new PoolCommitRecord(START_ID, START_ID, wire({ x: 1n }))),
    at(4, new NodeSucceededRecord(START_ID, 3, { forwardCapture: startCapture })),
    at(5, new NodeStartedRecord("p")),
    at(6, new PoolCommitRecord("p", "p", wire(pOutput))),
    at(7, new NodeSucceededRecord("p", 6, { forwardCapture: pCapture })),
    at(8, new NodeStartedRecord(END_ID)),
    at(9, new PoolCommitRecord(END_ID, END_ID, wire(pOutput))),
    at(10, new NodeSucceededRecord(END_ID, 9, { forwardCapture: endCapture })),
  ];
  let trace = step(null, envelopes[0]!, null, { rootStartId: START_ID, rootEndId: END_ID, rootInputNames: ["x"] });
  trace = step(trace, envelopes[1]!, null, { initialCommit: true });
  const owners = [null, null, nodes.start, null, null, nodes.p, null, null, nodes.end];
  for (const [index, owner] of owners.entries()) {
    const envelope = envelopes[index + 2]!;
    const record = envelope.record;
    const options: PrepareTraceOptions = {};
    if (record instanceof NodeSucceededRecord) {
      const commit = envelopes[record.commit]!.record as PoolCommitRecord;
      options.committed = { key: commit.key, valueJson: logJson(commit.value) };
      if (record.nodeId === END_ID) {
        options.terminalSucceeded = true;
        options.terminalOutputJson = logJson(wire(pOutput));
      }
    }
    trace = step(trace, envelope, owner, options);
  }
  return { trace, nodes, envelopes };
}

const ROOT_SEED = new BackwardAttribution(null, null, 0, null, []);

/** An invoked occurrence's result. */
function invoked(
  source: SourceRef,
  feedback: FeedbackValue[],
  attribution: BackwardAttribution[],
  inputFeedback: Record<string, FeedbackValue[]>,
): AttributedBackwardResult {
  return new AttributedBackwardResult(source, feedback, attribution, new BackwardResult({ inputFeedback }), true);
}

/** An unreached occurrence's result. */
function unreached(source: SourceRef): AttributedBackwardResult {
  return new AttributedBackwardResult(source, [], [], new BackwardResult(), false);
}

/** The exact chain results for root feedback `Too low.`: `p` sends `v`, and `w` too when `withW`. */
function chainResults(withW = false): AttributedBackwardResult[] {
  const pFeedback: Record<string, FeedbackValue[]> = { v: ["Raise v."] };
  if (withW) pFeedback.w = ["Check w."];
  return [
    invoked(S, [{ x: "Raise v." }], [new BackwardAttribution(P, "v", 0, 0, [0])], { x: ["Raise v."] }),
    invoked(P, ["Too low."], [new BackwardAttribution(E, "value", 0, 0, [0])], pFeedback),
    invoked(E, ["Too low."], [ROOT_SEED], { value: ["Too low."] }),
  ];
}

test("chainTrace drives START, p and END to a terminal trace", () => {
  const { trace, nodes, envelopes } = chainTrace();
  expect(trace.occurrences.map((item) => item.source)).toEqual([S, P, E]);
  expect(trace.rootStartSource).toEqual(S);
  expect(trace.terminal!.source).toEqual(E);
  expect(trace.terminal!.output).toBe(2n);
  expect(trace.occurrence(P).memoryScope).toBe(nodes.p.memoryScope);
  expect(envelopes).toHaveLength(11);
});

test("validateAttributedBackward accepts the exact transfers END → p → START → source 0", () => {
  const { trace } = chainTrace();
  const root = new BackwardResult({ inputFeedback: { x: ["Raise v."] } });
  expect(() => validateAttributedBackward(trace, "Too low.", root, chainResults())).not.toThrow();
});

test("START's root inputs merge with another node's source-zero feedback", () => {
  const { trace } = chainTrace();
  const results = chainResults(true);
  const root = new BackwardResult({ inputFeedback: { x: ["Raise v.", "Check w."] } });
  expect(() => validateAttributedBackward(trace, "Too low.", root, results)).not.toThrow();
  expect(results[0]!.result.inputFeedback).toEqual({ x: ["Raise v."] });
  const startOnly = new BackwardResult({ inputFeedback: { x: ["Raise v."] } });
  expectError(
    () => validateAttributedBackward(trace, "Too low.", startOnly, results),
    RangeError,
    "backward root result must match the accepted START and source-zero projection",
  );
});

const TRANSFER_MISMATCH = "backward received feedback and attribution must match exact transfers";

test.each<[string, (results: AttributedBackwardResult[], trace: RetainedTrace) => unknown, ErrorClass, string]>([
  [
    "a wrong inventory order",
    (results, trace) =>
      validateAttributedBackward(trace, "Too low.", new BackwardResult(), [...results].reverse()),
    RangeError,
    "backward results must match the complete accepted occurrence order",
  ],
  [
    "a missing occurrence",
    (results, trace) => validateAttributedBackward(trace, "Too low.", new BackwardResult(), results.slice(1)),
    RangeError,
    "backward results must match the complete accepted occurrence order",
  ],
  [
    "wrong feedback",
    (results, trace) => {
      results[2] = invoked(E, ["Other."], [ROOT_SEED], { value: ["Too low."] });
      return validateAttributedBackward(trace, "Too low.", new BackwardResult(), results);
    },
    RangeError,
    TRANSFER_MISMATCH,
  ],
  [
    "a wrong attribution",
    (results, trace) => {
      results[1] = invoked(P, ["Too low."], [new BackwardAttribution(E, "value", 0, 1, [0])], { v: ["Raise v."] });
      return validateAttributedBackward(trace, "Too low.", new BackwardResult(), results);
    },
    RangeError,
    TRANSFER_MISMATCH,
  ],
  [
    // An AttributedBackwardResult ties `invoked` to having messages, so a result that isn't invoked
    // has no feedback either: this row fails on the feedback check before the invoked flag.
    "a wrong invoked flag",
    (results, trace) => {
      results[0] = unreached(S);
      return validateAttributedBackward(trace, "Too low.", new BackwardResult(), results);
    },
    RangeError,
    TRANSFER_MISMATCH,
  ],
  [
    "a local contract violation",
    (results, trace) => {
      results[1] = invoked(P, ["Too low."], [new BackwardAttribution(E, "value", 0, 0, [0])], { z: ["No."] });
      return validateAttributedBackward(trace, "Too low.", new BackwardResult(), results);
    },
    RangeError,
    "input_feedback selects undeclared input 'z'",
  ],
  [
    "a wrong root",
    (results, trace) => validateAttributedBackward(trace, "Too low.", new BackwardResult(), results),
    RangeError,
    "backward root result must match the accepted START and source-zero projection",
  ],
  [
    "feedback to an unavailable occurrence",
    (_results, trace) => {
      trace.terminal = new TraceTerminal(true, new SourceRef(RUN, 99), logJson(wire(2n)), null);
      return validateAttributedBackward(trace, "Too low.", new BackwardResult(), [S, P, E].map(unreached));
    },
    RangeError,
    "feedback targets an unavailable or later accepted occurrence",
  ],
  [
    "a non-terminal trace",
    (results, trace) => {
      trace.terminal = null;
      return validateAttributedBackward(trace, "Too low.", new BackwardResult(), results);
    },
    RangeError,
    "backward requires a retained terminal trace",
  ],
  [
    "a root result of another class",
    (results, trace) => validateAttributedBackward(trace, "Too low.", {} as BackwardResult, results),
    TypeError,
    "result must be an exact BackwardResult",
  ],
  [
    "results that aren't an array",
    (results, trace) => validateAttributedBackward(trace, "Too low.", new BackwardResult(), new Set(results) as never),
    TypeError,
    "attributed_results must be an array",
  ],
  [
    "a result of another class",
    (results, trace) =>
      validateAttributedBackward(trace, "Too low.", new BackwardResult(), [{ ...results[0]! }, ...results.slice(1)]),
    TypeError,
    "attributed_results require exact AttributedBackwardResult values",
  ],
])("validateAttributedBackward refuses %s", (_name, call, cls, message) => {
  const { trace } = chainTrace();
  expectError(() => call(chainResults(), trace), cls, message);
});

test("empty feedback seeds nothing", () => {
  const { trace } = chainTrace();
  const [pending, root] = seedBackward(trace, null);
  expect(pending.size).toBe(0);
  expect(root).toEqual(new BackwardResult());
  const results = [S, P, E].map(unreached);
  expect(() => validateAttributedBackward(trace, null, new BackwardResult(), results)).not.toThrow();
});

test("seedBackward sends copied root feedback to the terminal source", () => {
  const { trace } = chainTrace({ pOutput: { a: "x" } });
  const feedback = { a: "Wrong." };
  const [pending, root] = seedBackward(trace, feedback);
  expect(root).toEqual(new BackwardResult());
  const [message] = pending.get(sourceKey(E))!;
  expect(message!.feedback).toEqual(feedback);
  expect(message!.feedback).not.toBe(feedback);
  expect(Object.isFrozen(message!.feedback)).toBe(true);
  expect(message!.order).toEqual([0]);
  expect(message!.attribution).toEqual(ROOT_SEED);
  expectError(() => seedBackward(trace, { b: "Wrong." }), RangeError, "feedback selects unknown field 'b'");
});

test("a node-data path absent from the accepted output fails the structural walk", () => {
  const { trace } = chainTrace({ startPaths: { x: ["y"] } });
  const root = new BackwardResult({ inputFeedback: { x: ["Raise v."] } });
  expectError(
    () => validateAttributedBackward(trace, "Too low.", root, chainResults()),
    RangeError,
    "structural output path is absent from the accepted output",
  );
});

/** The messages `p` receives in a chain whose `p` writes `[1n]` with `v` at output path `[0]`. */
function listChain() {
  const { trace } = chainTrace({ pOutput: [1n], pPaths: { v: [0] } });
  const cache = new ContextCache(trace);
  const [pending] = seedBackward(trace, ["Raise."]);
  const endMessages = takeMessages(pending, E);
  const endContext = trace.backwardContext(E);
  const endResult = new BackwardResult({ inputFeedback: { value: [["Raise."]] } });
  const toP = contextualTransfers(trace, trace.occurrence(E), endContext, endMessages, endResult, cache);
  const pMessages = toP.map(([, message]) => message);
  return { trace, cache, pMessages };
}

test("an int-headed node-data path gives the output context positions", () => {
  const { trace, cache, pMessages } = listChain();
  const context = trace.backwardContext(P);
  const result = new BackwardResult({ inputFeedback: { v: ["Raise."] } });
  const transfers = contextualTransfers(trace, trace.occurrence(P), context, pMessages, result, cache);
  expect(transfers).toHaveLength(1);
  const [target, message] = transfers[0]!;
  expect(target).toEqual(S);
  expect(message.feedback).toEqual({ x: "Raise." });
  expect(message.attribution).toEqual(new BackwardAttribution(P, "v", 0, 0, [0]));
  expect(message.order).toEqual([0, 0, 0, 0, 0, 0, 0]);
  // The parent's contexts come first, then the output context, then the transfer's own.
  expect(message.context.slice(0, pMessages[0]!.context.length)).toEqual([...pMessages[0]!.context]);
  const output = message.context[pMessages[0]!.context.length]!;
  expect(output.role).toBe("output");
  expect(output.sources).toEqual([P]);
  expect(output.operand).toEqual([0]);
  expect(output.positions).toEqual([[0]]);
  const materialized = materializeContext(message, 0, cache).contexts as Record<string, unknown>[];
  const detail = materialized[pMessages[0]!.context.length]!;
  expect(detail.positions).toEqual([[0n]]);
  expect(detail.operand).toEqual([0n]);
  expect(detail.result).toEqual([1n]);
});

test("materializeContext describes a message with bigint integers", () => {
  const { trace } = chainTrace();
  const cache = new ContextCache(trace);
  const message = new BackwardMessage("Raise v.", new BackwardAttribution(P, "v", 0, 0, [0]), [0, 0], []);
  expect(materializeContext(message, 1, cache)).toEqual({
    message_index: 1n,
    sender: { run_id: RUN, execution_seq: 5n, path: [] },
    input_name: "v",
    sender_message_index: 0n,
    contribution_index: 0n,
    parents: [0n],
    contexts: [],
  });
});

test("materializeContext resolves a source context once and shares it", () => {
  const { trace } = chainTrace();
  const cache = new ContextCache(trace);
  const [pending] = seedBackward(trace, "Too low.");
  const endMessages = takeMessages(pending, E);
  const result = new BackwardResult({ inputFeedback: { value: ["Too low."] } });
  const context = trace.backwardContext(E);
  const [[, message]] = contextualTransfers(trace, trace.occurrence(E), context, endMessages, result, cache) as [
    [SourceRef, BackwardMessage],
  ];
  const materialized = materializeContext(message, 0, cache);
  expect(materialized).toEqual({
    message_index: 0n,
    sender: { run_id: RUN, execution_seq: 8n, path: [] },
    input_name: "value",
    sender_message_index: 0n,
    contribution_index: 0n,
    parents: [0n],
    contexts: [
      {
        expression: "${source_5}",
        result: 2n,
        sources: [{ source: { run_id: RUN, execution_seq: 5n, path: [] }, value: 2n }],
        role: "source",
        operand: null,
        positions: [],
        span: null,
        result_type: null,
        snapshot: null,
      },
    ],
  });
  const again = materializeContext(message, 0, cache);
  expect((again.contexts as unknown[])[0]).toBe((materialized.contexts as unknown[])[0]);
});

test("materializeContext keeps an evaluated result, bigint indexes and a snapshot's route payload", () => {
  const { trace } = chainTrace();
  const cache = new ContextCache(trace);
  const snapshot = ValueRoute.literal(1n);
  const context = new RouteContext({
    expression: "${a[0]}",
    resultJson: "[1]",
    sources: [ZERO],
    role: "item",
    operand: 2,
    positions: [[0], [1]],
    span: [0, 2],
    resultType: "list",
    snapshot,
  });
  const message = new BackwardMessage("x", ROOT_SEED, [0], [context]);
  const [detail] = materializeContext(message, 0, cache).contexts as Record<string, unknown>[];
  expect(detail).toEqual({
    expression: "${a[0]}",
    result: [1n],
    sources: [{ source: { run_id: RUN, execution_seq: 0n, path: [] }, value: { x: 1n } }],
    role: "item",
    operand: 2n,
    positions: [[0n], [1n]],
    span: [0n, 2n],
    result_type: "list",
    snapshot: routePayload(snapshot),
  });
  const projection = new RouteContext({
    expression: "${input.x}",
    resultJson: null,
    sources: [new SourceRef(RUN, 0, ["x"])],
    role: "projection",
    operand: ["x"],
  });
  const [projected] = materializeContext(new BackwardMessage("x", ROOT_SEED, [0], [projection]), 0, cache)
    .contexts as Record<string, unknown>[];
  expect(projected!.result).toBe(1n);
  expect(projected!.operand).toEqual(["x"]);
  expect(projected!.sources).toEqual([{ source: { run_id: RUN, execution_seq: 0n, path: ["x"] }, value: 1n }]);
});

test("ContextCache.lookup looks a source up once and doesn't cache a failure", () => {
  const { trace } = chainTrace();
  const cache = new ContextCache(trace);
  const looked: string[] = [];
  const original = trace.lookupSource.bind(trace);
  trace.lookupSource = (source) => {
    looked.push(sourceKey(source));
    return original(source);
  };
  const first = cache.lookup(new SourceRef(RUN, 0));
  expect(first).toEqual({ x: 1n });
  expect(cache.lookup(new SourceRef(RUN, 0))).toBe(first);
  expect(looked).toEqual([sourceKey(ZERO)]);
  const missing = new SourceRef(RUN, 99);
  const absent = `no accepted source for ${sourceKey(missing)}`;
  expectError(() => cache.lookup(missing), RangeError, absent);
  expectError(() => cache.lookup(missing), RangeError, absent);
  expect(looked).toEqual([sourceKey(ZERO), sourceKey(missing), sourceKey(missing)]);
});

test("takeMessages pops one source's messages sorted by order", () => {
  const message = (order: number[]) => new BackwardMessage(order.join(","), ROOT_SEED, order);
  const pending = new Map([
    [sourceKey(P), [message([0, 1]), message([0]), message([0, 0, 2]), message([1])]],
    [sourceKey(E), [message([0])]],
  ]);
  expect(takeMessages(pending, P).map((item) => item.feedback)).toEqual(["0", "0,0,2", "0,1", "1"]);
  expect([...pending.keys()]).toEqual([sourceKey(E)]);
  expect(takeMessages(pending, S)).toEqual([]);
  expect(Object.isFrozen(message([0]))).toBe(true);
});

// TS-only: backward attempt state.

const MUTATIONS = [new MemoryMutation("m", { note: "Raise v." })];
const OPERATION_FAILURE = new MemoryOperationFailure("commit", "memory.unavailable", "Down.", true);

/** An accepted start of attempt `id`. */
function started(id = 1): BackwardStartedRecord {
  return new BackwardStartedRecord(id, "Too low.");
}

/** A Ready of attempt `id` with an empty root result and one proposal per memory in `memories`. */
function ready(id = 1, memories: string[] = ["a", "b"]): BackwardReadyRecord {
  const proposals = memories.map((memory) => new MemoryProposal(memory, memory === "a" ? MUTATIONS : []));
  return new BackwardReadyRecord(id, new BackwardResult(), [], proposals);
}

/** A commit request for `memory` in attempt `id`. */
function requested(memory: string, id = 1): MemoryCommitRequestedRecord {
  return new MemoryCommitRequestedRecord(memory, `commit-${memory}`, [], id);
}

/** A succeeded commit for `memory` in attempt `id`. */
function succeeded(memory: string, id = 1): MemoryCommitSucceededRecord {
  return new MemoryCommitSucceededRecord(memory, `commit-${memory}`, id);
}

/** A failed commit for `memory` in attempt `id`. */
function failed(memory: string, id = 1): MemoryCommitFailedRecord {
  return new MemoryCommitFailedRecord(memory, `commit-${memory}`, OPERATION_FAILURE, id);
}

/** The failure of an unfinished attempt. */
const UNFINISHED = new BackwardFailure(
  "backward",
  "backward.unfinished",
  "backward attempt did not finish",
);

/** A Finished of attempt `id` that ended unfinished before Ready. */
function unfinishedFinish(id = 1): BackwardFinishedRecord {
  return new BackwardFinishedRecord(id, new BackwardOutcome("failed", { failure: UNFINISHED }));
}

test.each<[string, () => unknown, ErrorClass, string]>([
  [
    "a started record of another class",
    () => new BackwardAttempt({ ...started() } as BackwardStartedRecord),
    TypeError,
    "backward attempt requires exact accepted records",
  ],
  [
    "a Ready of another class",
    () => new BackwardAttempt(started(), { ready: started() as unknown as BackwardReadyRecord }),
    TypeError,
    "backward attempt requires exact accepted records",
  ],
  [
    "a Finished of another class",
    () => new BackwardAttempt(started(), { finished: ready() as unknown as BackwardFinishedRecord }),
    TypeError,
    "backward attempt requires exact accepted records",
  ],
  [
    "a Ready of another attempt",
    () => new BackwardAttempt(started(), { ready: ready(2) }),
    RangeError,
    "backward attempt record identities disagree",
  ],
  [
    "a Finished of another attempt",
    () => new BackwardAttempt(started(), { finished: unfinishedFinish(2) }),
    RangeError,
    "backward attempt record identities disagree",
  ],
  [
    "requests that aren't an array",
    () => new BackwardAttempt(started(), { requests: new Set([requested("a")]) as never }),
    TypeError,
    "backward commit records must be arrays",
  ],
  [
    "observations that aren't an array",
    () => new BackwardAttempt(started(), { observations: new Set([succeeded("a")]) as never }),
    TypeError,
    "backward commit records must be arrays",
  ],
  [
    "a request of another class",
    () => new BackwardAttempt(started(), { requests: [succeeded("a") as unknown as MemoryCommitRequestedRecord] }),
    TypeError,
    "backward commit requires an exact accepted record",
  ],
  [
    "an observation of another class",
    () => new BackwardAttempt(started(), { observations: [requested("a") as unknown as MemoryCommitFailedRecord] }),
    TypeError,
    "backward commit requires an exact accepted record",
  ],
  [
    "a request of another attempt",
    () => new BackwardAttempt(started(), { requests: [requested("a", 2)] }),
    RangeError,
    "backward commit attempt identities disagree",
  ],
  [
    "an observation of another attempt",
    () => new BackwardAttempt(started(), { observations: [failed("a", 2)] }),
    RangeError,
    "backward commit attempt identities disagree",
  ],
])("BackwardAttempt refuses %s", (_label, fn, cls, message) => {
  expectError(fn, cls, message);
});

test("BackwardAttempt holds its records by reference in frozen arrays", () => {
  const records = [requested("a")];
  const attempt = new BackwardAttempt(started(), { ready: ready(), requests: records });
  records.push(requested("b"));
  expect(attempt.requests).toHaveLength(1);
  expect(attempt.requests[0]).toBe(records[0]!);
  expect(attempt.finished).toBeNull();
  expect(attempt.observations).toEqual([]);
  expect(Object.isFrozen(attempt)).toBe(true);
  expect(Object.isFrozen(attempt.requests)).toBe(true);
});

test.each<[string, () => unknown, ErrorClass, string]>([
  [
    "a memory order that isn't an array",
    () => new RuntimeBackwardState(new Set(["a"]) as never),
    TypeError,
    "backward memory order must be an array of non-empty strings",
  ],
  [
    "an empty memory key",
    () => new RuntimeBackwardState([""]),
    TypeError,
    "backward memory order must be an array of non-empty strings",
  ],
  [
    "a memory key that isn't text",
    () => new RuntimeBackwardState([1 as unknown as string]),
    TypeError,
    "backward memory order must be an array of non-empty strings",
  ],
  [
    "a repeated memory key",
    () => new RuntimeBackwardState(["a", "a"]),
    RangeError,
    "backward memory order must be sorted and unique",
  ],
  [
    "a history that isn't an array",
    () => new RuntimeBackwardState([], new Set() as never),
    TypeError,
    "backward history must be an array",
  ],
  [
    "an attempt of another class",
    () => new RuntimeBackwardState([], [{ ...new BackwardAttempt(started()) } as BackwardAttempt]),
    TypeError,
    "backward history requires exact attempt states",
  ],
  [
    "an attempt id out of start order",
    () => new RuntimeBackwardState([], [new BackwardAttempt(started(2))]),
    RangeError,
    "backward attempt IDs must follow accepted start order",
  ],
  [
    "an unfinished attempt before another start",
    () => new RuntimeBackwardState([], [new BackwardAttempt(started(1)), new BackwardAttempt(started(2))]),
    RangeError,
    "an unfinished attempt cannot precede another start",
  ],
  [
    "requests out of memory order",
    () =>
      new RuntimeBackwardState(["a", "b"], [
        new BackwardAttempt(started(), { ready: ready(), requests: [requested("b")] }),
      ]),
    RangeError,
    "backward requests must follow recorded memory order",
  ],
  [
    "two requests without an observation",
    () =>
      new RuntimeBackwardState(["a", "b"], [
        new BackwardAttempt(started(), { ready: ready(), requests: [requested("a"), requested("b")] }),
      ]),
    RangeError,
    "backward requests and observations must alternate",
  ],
  [
    "an observation without a request",
    () => new RuntimeBackwardState(["a", "b"], [new BackwardAttempt(started(), { observations: [succeeded("a")] })]),
    RangeError,
    "backward requests and observations must alternate",
  ],
  [
    "an observation of another commit",
    () =>
      new RuntimeBackwardState(["a", "b"], [
        new BackwardAttempt(started(), {
          ready: ready(),
          requests: [requested("a")],
          observations: [new MemoryCommitSucceededRecord("a", "other", 1)],
        }),
      ]),
    RangeError,
    "backward observation must match its accepted request",
  ],
  [
    "a request after a failed commit",
    () =>
      new RuntimeBackwardState(["a", "b"], [
        new BackwardAttempt(started(), {
          ready: ready(),
          requests: [requested("a"), requested("b")],
          observations: [failed("a")],
        }),
      ]),
    RangeError,
    "a failed commit cannot be followed by another request",
  ],
  [
    "a request without Ready",
    () => new RuntimeBackwardState(["a", "b"], [new BackwardAttempt(started(), { requests: [requested("a")] })]),
    RangeError,
    "backward requests require accepted Ready proposals",
  ],
])("RuntimeBackwardState refuses %s", (_label, fn, cls, message) => {
  expectError(fn, cls, message);
});

test("an empty history starts nothing and leaves every memory unrequested", () => {
  const state = new RuntimeBackwardState(["a", "b"]);
  expect(state.memoryOrder).toEqual(["a", "b"]);
  expect(state.attempts).toEqual([]);
  expect(state.current).toBeNull();
  expect(state.started).toBeNull();
  expect(state.ready).toBeNull();
  expect(state.finished).toBeNull();
  expect(state.everRequested).toBe(false);
  expect(state.commitAttemptKey).toBeNull();
  expect(state.policyOutcomes).toEqual([
    new PolicyCommitOutcome("a", "unrequested"),
    new PolicyCommitOutcome("b", "unrequested"),
  ]);
  expect(state.canStart).toBe(true);
  expect(state.outcome).toBeNull();
  expect(prefixLearning(state)).toEqual(new LearningResult("incomplete", { policies: state.policyOutcomes }));
  expect(prefixLearning(new RuntimeBackwardState())).toBeNull();
  expect(Object.isFrozen(state)).toBe(true);
  expect(Object.isFrozen(state.memoryOrder)).toBe(true);
});

test("an unfinished attempt reads its records and derives the unfinished outcome with prefix learning", () => {
  const attempt = new BackwardAttempt(started(), {
    ready: ready(),
    requests: [requested("a"), requested("b")],
    observations: [succeeded("a")],
  });
  const state = new RuntimeBackwardState(["a", "b"], [attempt]);
  expect(state.current).toBe(attempt);
  expect(state.started).toBe(attempt.started);
  expect(state.ready).toBe(attempt.ready);
  expect(state.finished).toBeNull();
  expect(state.everRequested).toBe(true);
  expect(state.commitAttemptKey).toBe(1);
  expect(state.canStart).toBe(false);
  const policies = [
    new PolicyCommitOutcome("a", "succeeded", { commitId: "commit-a" }),
    new PolicyCommitOutcome("b", "unknown", { commitId: "commit-b" }),
  ];
  expect(state.policyOutcomes).toEqual(policies);
  const learning = new LearningResult("incomplete", { policies });
  const feedback = attempt.ready!.rootResult;
  expect(state.outcome).toEqual(new BackwardOutcome("failed", { feedback, failure: UNFINISHED, learning }));
  expect(state.outcome!.feedback).toBe(attempt.ready!.rootResult);
});

test("prefix learning succeeds only when asked and every memory is confirmed", () => {
  const attempt = new BackwardAttempt(started(), {
    ready: ready(1, ["a"]),
    requests: [requested("a")],
    observations: [succeeded("a")],
  });
  const state = new RuntimeBackwardState(["a"], [attempt]);
  const policies = [new PolicyCommitOutcome("a", "succeeded", { commitId: "commit-a" })];
  expect(prefixLearning(state, { succeeded: true })).toEqual(new LearningResult("succeeded", { policies }));
  expect(prefixLearning(state)).toEqual(new LearningResult("incomplete", { policies }));
});

test("an unfinished attempt before Ready has no feedback", () => {
  const state = new RuntimeBackwardState(["a"], [new BackwardAttempt(started())]);
  const outcome = state.outcome!;
  expect(outcome.feedback).toBeNull();
  expect(outcome.failure!.code).toBe("backward.unfinished");
  const policies = [new PolicyCommitOutcome("a", "unrequested")];
  expect(outcome.learning).toEqual(new LearningResult("incomplete", { policies }));
});

test("a failed commit fails the prefix learning, after a learner failure", () => {
  const attempt = new BackwardAttempt(started(), {
    ready: ready(),
    requests: [requested("a")],
    observations: [failed("a")],
  });
  const state = new RuntimeBackwardState(["a", "b"], [attempt]);
  const policies = [
    new PolicyCommitOutcome("a", "failed", { commitId: "commit-a", failure: OPERATION_FAILURE }),
    new PolicyCommitOutcome("b", "unrequested"),
  ];
  const commitFailure = new LearningFailure("a", "commit", OPERATION_FAILURE);
  expect(state.outcome!.learning).toEqual(new LearningResult("failed", { failures: [commitFailure], policies }));
  const learnerFailure = new LearningFailure("b", "learn", OPERATION_FAILURE);
  expect(prefixLearning(state, { learnerFailure })).toEqual(
    new LearningResult("failed", { failures: [learnerFailure, commitFailure], policies }),
  );
});

test("a finished history admits a start and keeps the accepted outcome", () => {
  const first = new BackwardAttempt(started(1), {
    ready: ready(1),
    requests: [requested("a")],
    observations: [failed("a")],
    finished: unfinishedFinish(1),
  });
  const second = new BackwardAttempt(started(2), { finished: unfinishedFinish(2) });
  const state = new RuntimeBackwardState(["a", "b"], [first, second]);
  expect(state.current).toBe(second);
  expect(state.finished).toBe(second.finished);
  expect(state.ready).toBeNull();
  expect(state.everRequested).toBe(true);
  expect(state.commitAttemptKey).toBe(2);
  expect(state.canStart).toBe(true);
  expect(state.outcome).toBe(second.finished!.outcome);
  expect(state.policyOutcomes.map((policy) => policy.status)).toEqual(["unrequested", "unrequested"]);
});

/**
 * A finished attempt 1 and an unfinished attempt 2 with `last` (by default `started(2)`) and
 * `options`.
 */
function twoAttempts(
  memories: string[] = ["a", "b"],
  last: BackwardStartedRecord = started(2),
  options: { ready?: BackwardReadyRecord | null } = { ready: ready(2) },
): RuntimeBackwardState {
  const first = new BackwardAttempt(started(1), { finished: unfinishedFinish(1) });
  return new RuntimeBackwardState(memories, [first, new BackwardAttempt(last, options)]);
}

test("sameRuntimeBackwardState is true for separately built equal states", () => {
  const state = twoAttempts();
  expect(sameRuntimeBackwardState(state, twoAttempts())).toBe(true);
  expect(sameRuntimeBackwardState(state, new RuntimeBackwardState(["a", "b"], state.attempts))).toBe(true);
});

test.each<[string, () => RuntimeBackwardState]>([
  ["a different memory order", () => twoAttempts(["a"])],
  ["a different last record", () => twoAttempts(["a", "b"], new BackwardStartedRecord(2, "Other."))],
  ["a null ready", () => twoAttempts(["a", "b"], started(2), {})],
])("sameRuntimeBackwardState is false for %s", (_label, other) => {
  expect(sameRuntimeBackwardState(twoAttempts(), other())).toBe(false);
  expect(sameRuntimeBackwardState(other(), twoAttempts())).toBe(false);
});

test("proposalBatches returns each memory's batch in table order and refuses another order", () => {
  const state = new RuntimeBackwardState(["a", "b"]);
  const batches = proposalBatches(state, ready());
  expect(batches).toHaveLength(2);
  expect(batches[0]).toEqual(MUTATIONS);
  expect(batches[1]).toEqual([]);
  for (const memories of [["b", "a"], ["a"], ["a", "b", "c"]]) {
    expectError(
      () => proposalBatches(state, ready(1, memories)),
      RangeError,
      "memory proposals must cover the complete recorded memory table in order",
    );
  }
});

// TS-only: the backward attempt reducer, over `chainTrace()` with memory key `notes`.

const NOTES = [new MemoryMutation("n1", { note: "Raise v." })];
const COMMIT_ID = memoryCommitId(RUN, "notes", NOTES, 1);
const CHAIN_ROOT = new BackwardResult({ inputFeedback: { x: ["Raise v."] } });
const UNREQUESTED = [new PolicyCommitOutcome("notes", "unrequested")];
const INCOMPLETE = new LearningResult("incomplete", { policies: UNREQUESTED });
const LEARN_CAUSE = new MemoryOperationFailure("learn", "memory.learn_failed", "Bad.", false);
const CONTEXT_CAUSE = new MemoryOperationFailure("learn", "memory.invalid_context", "Bad context.", false);
const COMMIT_FAILURE = new BackwardFailure("commit", OPERATION_FAILURE.code, OPERATION_FAILURE.message, {
  memory: "notes",
  cause: OPERATION_FAILURE,
});
const COMMIT_LEARNING = new LearningResult("failed", {
  failures: [new LearningFailure("notes", "commit", OPERATION_FAILURE)],
  policies: [new PolicyCommitOutcome("notes", "failed", { commitId: COMMIT_ID, failure: OPERATION_FAILURE })],
});
const SUCCEEDED_LEARNING = new LearningResult("succeeded", {
  policies: [new PolicyCommitOutcome("notes", "succeeded", { commitId: COMMIT_ID })],
});

/** The chain's Ready of attempt `id`, proposing `NOTES` for `notes` unless `proposals` is given. */
function chainReady(id = 1, proposals = [new MemoryProposal("notes", NOTES)]): BackwardReadyRecord {
  return new BackwardReadyRecord(id, CHAIN_ROOT, chainResults(), proposals);
}

const chainRequest = () => new MemoryCommitRequestedRecord("notes", COMMIT_ID, NOTES, 1);
const chainSucceeded = () => new MemoryCommitSucceededRecord("notes", COMMIT_ID, 1);
const chainFailed = () => new MemoryCommitFailedRecord("notes", COMMIT_ID, OPERATION_FAILURE, 1);

/** A Finished of attempt 1 with a failed outcome. */
function failedFinish(
  failure: BackwardFailure,
  options: { feedback?: BackwardResult | null; learning?: LearningResult | null } = {},
  attributed: AttributedBackwardResult[] = [],
): BackwardFinishedRecord {
  return new BackwardFinishedRecord(1, new BackwardOutcome("failed", { failure, ...options }), attributed);
}

/** A successful Finished of attempt 1 with the chain's root result. */
function succeededFinish(learning: LearningResult = SUCCEEDED_LEARNING): BackwardFinishedRecord {
  return new BackwardFinishedRecord(1, new BackwardOutcome("succeeded", { feedback: CHAIN_ROOT, learning }));
}

/** The unfinished Finished of attempt 1 before Ready. */
const chainUnfinished = () => failedFinish(UNFINISHED, { learning: INCOMPLETE });

/** The options of a `BackwardFailure`. */
type FailureOptions = ConstructorParameters<typeof BackwardFailure>[3];

/** A learner failure for `notes` from `cause`. */
function learnerFailure(cause = LEARN_CAUSE, options: FailureOptions = {}): BackwardFailure {
  return new BackwardFailure("learn", cause.code, cause.message, { memory: "notes", cause, ...options });
}

/** A context failure attributed to `p`'s occurrence. */
function contextFailure(options: FailureOptions = {}): BackwardFailure {
  return learnerFailure(CONTEXT_CAUSE, { runId: RUN, nodeId: "p", nodeStartedSeq: 5, ...options });
}

/** The learning of a learner failure from `cause` before any request. */
function learnerLearning(cause = LEARN_CAUSE): LearningResult {
  const failures = [new LearningFailure("notes", "learn", cause)];
  return new LearningResult("failed", { failures, policies: UNREQUESTED });
}

/** A learner's Finished of attempt 1: completed feedback, the chain's results and `learning`. */
function learnerFinish(failure: BackwardFailure, learning: LearningResult | null = null): BackwardFinishedRecord {
  return failedFinish(failure, { feedback: CHAIN_ROOT, learning }, chainResults());
}

/** A backward computation failure attributed to `p`'s occurrence. */
function computeFailure(options: FailureOptions = {}, code = "backward.execute", stage = "backward"): BackwardFailure {
  return new BackwardFailure(stage, code, "Broke.", { runId: RUN, nodeId: "p", nodeStartedSeq: 5, ...options });
}

/** Reduce `records` in order over `trace` from a state with memory key `notes`. */
function reduced(records: BackwardRecord[], trace: RetainedTrace = chainTrace().trace): RuntimeBackwardState {
  return records.reduce(
    (state, record) => reduceBackwardRecord(trace, state, record),
    new RuntimeBackwardState(["notes"]),
  );
}

test("the reducer accepts a full attempt Started → Ready → Requested → Succeeded → Finished", () => {
  const { trace } = chainTrace();
  const records = [started(), chainReady(), chainRequest(), chainSucceeded(), succeededFinish()];
  const empty = new RuntimeBackwardState(["notes"]);
  const states = [empty];
  for (const record of records) states.push(reduceBackwardRecord(trace, states.at(-1)!, record));
  const [, afterStart, afterReady, afterRequest, afterResult, afterFinish] = states as RuntimeBackwardState[];
  expect(empty.attempts).toEqual([]);
  expect(afterStart!.started).toBe(records[0] as BackwardStartedRecord);
  expect(afterStart!.canStart).toBe(false);
  expect(afterReady!.ready).toBe(records[1] as BackwardReadyRecord);
  expect(afterRequest!.policyOutcomes).toEqual([new PolicyCommitOutcome("notes", "unknown", { commitId: COMMIT_ID })]);
  expect(afterResult!.policyOutcomes).toEqual(SUCCEEDED_LEARNING.policies);
  expect(afterFinish!.finished).toBe(records[4] as BackwardFinishedRecord);
  expect(afterFinish!.outcome).toBe((records[4] as BackwardFinishedRecord).outcome);
  expect(afterFinish!.attempts).toHaveLength(1);
  expect(afterFinish!.canStart).toBe(true);
  expect(afterFinish!.everRequested).toBe(true);
  for (const state of states) {
    expect(Object.isFrozen(state)).toBe(true);
    expect(state.memoryOrder).toBe(empty.memoryOrder);
  }
  const next = reduceBackwardRecord(trace, afterFinish!, started(2));
  expect(next.attempts).toHaveLength(2);
  expect(next.commitAttemptKey).toBe(2);
  expect(next.attempts[0]).toBe(afterFinish!.attempts[0]!);
});

test.each<[string, () => BackwardRecord[]]>([
  ["an unfinished attempt before Ready", () => [started(), chainUnfinished()]],
  [
    "an unfinished attempt after Ready, with its results",
    () => [
      started(),
      chainReady(),
      failedFinish(UNFINISHED, { feedback: CHAIN_ROOT, learning: INCOMPLETE }, chainResults()),
    ],
  ],
  [
    "a commit failure",
    () => [
      started(),
      chainReady(),
      chainRequest(),
      chainFailed(),
      failedFinish(COMMIT_FAILURE, { feedback: CHAIN_ROOT, learning: COMMIT_LEARNING }),
    ],
  ],
  [
    "a learner failure",
    () => [started(), learnerFinish(learnerFailure(), learnerLearning())],
  ],
  [
    "a learner failure with an attributed context failure",
    () => [started(), learnerFinish(contextFailure(), learnerLearning(CONTEXT_CAUSE))],
  ],
  ["a computation failure", () => [started(), failedFinish(computeFailure(), { learning: INCOMPLETE })]],
])("the reducer accepts the Finished of %s", (_label, records) => {
  const list = records();
  const state = reduced(list);
  expect(state.finished).toBe(list.at(-1) as BackwardFinishedRecord);
  expect(state.canStart).toBe(true);
});

/** A record of another class with the matching attempt id. */
function forgedRecord(): BackwardRecord {
  return Object.assign(Object.create(NodeStartedRecord.prototype) as object, { attemptId: 1 }) as BackwardRecord;
}

test.each<[string, (trace: RetainedTrace) => unknown, ErrorClass, string]>([
  [
    "a state of another class",
    (trace) => reduceBackwardRecord(trace, {} as RuntimeBackwardState, started()),
    TypeError,
    "backward reduction requires RuntimeBackwardState",
  ],
  [
    "a Started on a non-terminal trace",
    (trace) => {
      trace.terminal = null;
      return reduced([started()], trace);
    },
    RangeError,
    "backward requires a retained terminal trace",
  ],
  [
    "a Started while an attempt is open",
    (trace) => reduced([started(1), started(2)], trace),
    RangeError,
    "backward attempt cannot start from the accepted prefix",
  ],
  [
    "a Started that skips an attempt id",
    (trace) => reduced([started(2)], trace),
    RangeError,
    "backward attempt_id must be the next accepted start",
  ],
  [
    "a Ready before any start",
    (trace) => reduced([chainReady()], trace),
    RangeError,
    "backward record requires its matching unfinished attempt",
  ],
  [
    "a Ready of another attempt",
    (trace) => reduced([started(1), chainReady(2)], trace),
    RangeError,
    "backward record requires its matching unfinished attempt",
  ],
  [
    "a record after Finished",
    (trace) => reduced([started(), chainUnfinished(), chainReady()], trace),
    RangeError,
    "backward record requires its matching unfinished attempt",
  ],
  [
    "a second Ready",
    (trace) => reduced([started(), chainReady(), chainReady()], trace),
    RangeError,
    "backward attempt already has a Ready",
  ],
  [
    "a Ready whose results disagree with the transfers",
    (trace) => {
      const proposals = [new MemoryProposal("notes", NOTES)];
      return reduced([started(), new BackwardReadyRecord(1, CHAIN_ROOT, [S, P, E].map(unreached), proposals)], trace);
    },
    RangeError,
    TRANSFER_MISMATCH,
  ],
  [
    "a Ready without the table's proposals",
    (trace) => reduced([started(), chainReady(1, [])], trace),
    RangeError,
    "memory proposals must cover the complete recorded memory table in order",
  ],
  [
    "a request before Ready",
    (trace) => reduced([started(), chainRequest()], trace),
    RangeError,
    "backward commit request requires Ready and no pending or failed memory",
  ],
  [
    "a request while one is pending",
    (trace) => reduced([started(), chainReady(), chainRequest(), chainRequest()], trace),
    RangeError,
    "backward commit request requires Ready and no pending or failed memory",
  ],
  [
    "a request after a failed commit",
    (trace) => reduced([started(), chainReady(), chainRequest(), chainFailed(), chainRequest()], trace),
    RangeError,
    "backward commit request requires Ready and no pending or failed memory",
  ],
  [
    "a request for another memory",
    (trace) => reduced([started(), chainReady(), new MemoryCommitRequestedRecord("other", COMMIT_ID, NOTES, 1)], trace),
    RangeError,
    "backward request must name the next recorded memory",
  ],
  [
    "a request after every memory",
    (trace) => reduced([started(), chainReady(), chainRequest(), chainSucceeded(), chainRequest()], trace),
    RangeError,
    "backward request must name the next recorded memory",
  ],
  [
    "a request with another batch",
    (trace) => reduced([started(), chainReady(), new MemoryCommitRequestedRecord("notes", COMMIT_ID, [], 1)], trace),
    RangeError,
    "backward commit batch and identity must match the ordered Ready proposals",
  ],
  [
    "a request with another commit id",
    (trace) => {
      const commitId = memoryCommitId(RUN, "notes", NOTES, 2);
      return reduced([started(), chainReady(), new MemoryCommitRequestedRecord("notes", commitId, NOTES, 1)], trace);
    },
    RangeError,
    "backward commit batch and identity must match the ordered Ready proposals",
  ],
  [
    "a commit result without a request",
    (trace) => reduced([started(), chainReady(), chainSucceeded()], trace),
    RangeError,
    "backward commit result requires a matching pending request",
  ],
  [
    "a commit result for another commit",
    (trace) =>
      reduced([started(), chainReady(), chainRequest(), new MemoryCommitSucceededRecord("notes", "other", 1)], trace),
    RangeError,
    "backward commit result must match its pending memory and identity",
  ],
  [
    "a record of another class",
    (trace) => reduced([started(), forgedRecord()], trace),
    RangeError,
    "backward commit requires a completed memory proposal",
  ],
  [
    "Finished feedback that differs from Ready",
    (trace) => {
      const finish = failedFinish(UNFINISHED, { feedback: new BackwardResult(), learning: INCOMPLETE });
      return reduced([started(), chainReady(), finish], trace);
    },
    RangeError,
    "Finished feedback must match accepted Ready",
  ],
  [
    "Finished attribution that differs from Ready",
    (trace) => {
      const unreachedResults = [S, P, E].map(unreached);
      const finish = failedFinish(UNFINISHED, { feedback: CHAIN_ROOT, learning: INCOMPLETE }, unreachedResults);
      return reduced([started(), chainReady(), finish], trace);
    },
    RangeError,
    "Finished attribution must match accepted Ready",
  ],
  [
    "a successful Finished before Ready",
    (trace) => reduced([started(), succeededFinish()], trace),
    RangeError,
    "successful Finished requires Ready and every memory confirmation",
  ],
  [
    "a successful Finished before the commit",
    (trace) => reduced([started(), chainReady(), succeededFinish()], trace),
    RangeError,
    "successful Finished requires Ready and every memory confirmation",
  ],
  [
    "a successful Finished with other learning",
    (trace) => {
      const finish = succeededFinish(new LearningResult("succeeded"));
      return reduced([started(), chainReady(), chainRequest(), chainSucceeded(), finish], trace);
    },
    RangeError,
    "successful Finished learning must match accepted memory confirmations",
  ],
  [
    "an unfinished attempt's Finished with other learning",
    (trace) => reduced([started(), failedFinish(UNFINISHED)], trace),
    RangeError,
    "the Finished of an unfinished attempt must exactly project the accepted prefix",
  ],
  [
    "an unfinished attempt's Finished before Ready with results",
    (trace) => reduced([started(), failedFinish(UNFINISHED, { learning: INCOMPLETE }, chainResults())], trace),
    RangeError,
    "the Finished of an unfinished attempt must exactly project the accepted prefix",
  ],
  [
    "a failed Finished after Ready without a commit failure",
    (trace) => {
      const finish = failedFinish(COMMIT_FAILURE, { feedback: CHAIN_ROOT, learning: COMMIT_LEARNING });
      return reduced([started(), chainReady(), finish], trace);
    },
    RangeError,
    "failed Finished after Ready requires a recorded commit failure",
  ],
  [
    "a commit Finished with another failure",
    (trace) => {
      const failure = new BackwardFailure("commit", OPERATION_FAILURE.code, "Other.", {
        memory: "notes",
        cause: OPERATION_FAILURE,
      });
      const finish = failedFinish(failure, { feedback: CHAIN_ROOT, learning: COMMIT_LEARNING });
      return reduced([started(), chainReady(), chainRequest(), chainFailed(), finish], trace);
    },
    RangeError,
    "commit Finished must match the exact accepted failure and memory states",
  ],
  [
    "a commit Finished with other learning",
    (trace) => {
      const finish = failedFinish(COMMIT_FAILURE, { feedback: CHAIN_ROOT, learning: learnerLearning() });
      return reduced([started(), chainReady(), chainRequest(), chainFailed(), finish], trace);
    },
    RangeError,
    "commit Finished must match the exact accepted failure and memory states",
  ],
  [
    "a learner failure with a Failure cause",
    (trace) => {
      const cause = new Failure({ code: "node.execute", message: "Bad.", nodeId: null });
      const failure = new BackwardFailure("learn", "node.execute", "Bad.", { memory: "notes", cause });
      return reduced([started(), learnerFinish(failure)], trace);
    },
    RangeError,
    "learner failure requires one recorded memory key and memory-operation cause",
  ],
  [
    "a learner failure for an unrecorded memory",
    (trace) => {
      const failure = learnerFailure(LEARN_CAUSE, { memory: "other" });
      return reduced([started(), learnerFinish(failure)], trace);
    },
    RangeError,
    "learner failure requires one recorded memory key and memory-operation cause",
  ],
  [
    "a learner failure without completed feedback",
    (trace) => {
      const finish = failedFinish(learnerFailure(), { learning: learnerLearning() }, chainResults());
      return reduced([started(), finish], trace);
    },
    RangeError,
    "learner Finished must retain the exact attributed failure and completed feedback",
  ],
  [
    "a learner failure whose code differs from its cause",
    (trace) => {
      const failure = new BackwardFailure("learn", "memory.other", "Bad.", { memory: "notes", cause: LEARN_CAUSE });
      return reduced([started(), learnerFinish(failure)], trace);
    },
    RangeError,
    "learner Finished must retain the exact attributed failure and completed feedback",
  ],
  [
    "learner learning without its failure",
    (trace) => reduced([started(), learnerFinish(learnerFailure(), INCOMPLETE)], trace),
    RangeError,
    "learner Finished learning must contain its exact known failure",
  ],
  [
    "an attributed learner failure that isn't a context failure",
    (trace) => {
      const failure = learnerFailure(LEARN_CAUSE, { runId: RUN, nodeId: "p", nodeStartedSeq: 5 });
      return reduced([started(), learnerFinish(failure)], trace);
    },
    RangeError,
    "an attributed learner failure must be a context failure of this run",
  ],
  [
    "a context failure of another run",
    (trace) => reduced([started(), learnerFinish(contextFailure({ runId: "other" }))], trace),
    RangeError,
    "an attributed learner failure must be a context failure of this run",
  ],
  [
    "a context failure naming no accepted occurrence",
    (trace) => reduced([started(), learnerFinish(contextFailure({ nodeStartedSeq: 9 }))], trace),
    RangeError,
    "a context failure must name an accepted occurrence",
  ],
  [
    "a context failure of another node",
    (trace) => reduced([started(), learnerFinish(contextFailure({ nodeId: "q" }))], trace),
    RangeError,
    "a context failure must match its recorded occurrence",
  ],
  [
    "a context failure of another scope",
    (trace) => {
      const failure = contextFailure({ scope: new ExecutionScope(["p"]) });
      return reduced([started(), learnerFinish(failure)], trace);
    },
    RangeError,
    "a context failure must match its recorded occurrence",
  ],
  [
    "a context failure of an occurrence without the memory",
    (trace) => {
      const failure = contextFailure({ nodeId: START_ID, nodeStartedSeq: 2 });
      return reduced([started(), learnerFinish(failure)], trace);
    },
    RangeError,
    "a context failure must name an occurrence that receives its memory",
  ],
  [
    "a computation failure with completed feedback",
    (trace) => {
      const finish = failedFinish(computeFailure(), { feedback: CHAIN_ROOT, learning: INCOMPLETE });
      return reduced([started(), finish], trace);
    },
    RangeError,
    "incomplete backward computation cannot carry completed or partial results",
  ],
  [
    "a computation failure with results",
    (trace) => reduced([started(), failedFinish(computeFailure(), { learning: INCOMPLETE }, chainResults())], trace),
    RangeError,
    "incomplete backward computation cannot carry completed or partial results",
  ],
  [
    "a failure of another stage before Ready",
    (trace) => {
      const finish = failedFinish(computeFailure({}, "backward.execute", "commit"), { learning: INCOMPLETE });
      return reduced([started(), finish], trace);
    },
    RangeError,
    "incomplete backward computation cannot carry completed or partial results",
  ],
  [
    "a computation failure with another code",
    (trace) =>
      reduced([started(), failedFinish(computeFailure({}, "backward.other"), { learning: INCOMPLETE })], trace),
    RangeError,
    "backward computation failure requires effective feedback and a driver failure code",
  ],
  [
    "a computation failure after empty feedback",
    (trace) =>
      reduced([new BackwardStartedRecord(1, null), failedFinish(computeFailure(), { learning: INCOMPLETE })], trace),
    RangeError,
    "backward computation failure requires effective feedback and a driver failure code",
  ],
  [
    "a computation failure without a run id",
    (trace) => reduced([started(), failedFinish(computeFailure({ runId: null }), { learning: INCOMPLETE })], trace),
    RangeError,
    "backward computation failure requires accepted occurrence attribution",
  ],
  [
    "a computation failure with a memory",
    (trace) => reduced([started(), failedFinish(computeFailure({ memory: "notes" }), { learning: INCOMPLETE })], trace),
    RangeError,
    "backward computation failure requires accepted occurrence attribution",
  ],
  [
    "a computation failure naming no accepted occurrence",
    (trace) =>
      reduced([started(), failedFinish(computeFailure({ nodeStartedSeq: 9 }), { learning: INCOMPLETE })], trace),
    RangeError,
    "backward computation failure requires an accepted occurrence",
  ],
  [
    "a computation failure of another node",
    (trace) => reduced([started(), failedFinish(computeFailure({ nodeId: "q" }), { learning: INCOMPLETE })], trace),
    RangeError,
    "backward computation failure must match its recorded invocation",
  ],
  [
    "a computation failure with other learning",
    (trace) => reduced([started(), failedFinish(computeFailure())], trace),
    RangeError,
    "failed computation learning must retain incomplete unrequested memories",
  ],
])("the reducer refuses %s", (_label, call, cls, message) => {
  expectError(() => call(chainTrace().trace), cls, message);
});

// TS-only: owner learn contexts, over `chainTrace()` with memory key `notes` on `p`.

const EMPTY_SPEC = new MemorySpec("empty", "empty", {}, [new DeclSite([], ["def", "unused"])]);
const NOTES_SPEC = new MemorySpec("notes", "notes", {}, [new DeclSite([], ["def", "p"])]);
const OUTER_SPEC = new MemorySpec("outer", "outer", {}, [new DeclSite([], ["root"])]);
const OWNER_TABLE = [EMPTY_SPEC, NOTES_SPEC, OUTER_SPEC];
const P_STACK = new MemoryStack(["p"], ["notes"]);
const OWNER_ROOT = new BackwardResult({ inputFeedback: { x: ["Raise v.", "Check w."] } });

/** The chain's results with `p` sending `v`, `w`, self feedback `self` and variable feedback. */
function ownerResults(self = "Be careful."): AttributedBackwardResult[] {
  const [start, , end] = chainResults(true);
  const result = new BackwardResult({
    inputFeedback: { v: ["Raise v."], w: ["Check w."] },
    selfFeedback: [self],
    variableFeedback: { tone: ["Softer."] },
  });
  const attribution = [new BackwardAttribution(E, "value", 0, 0, [0])];
  return [start!, new AttributedBackwardResult(P, ["Too low."], attribution, result, true), end!];
}

/** The options of `ownerContexts`. */
interface OwnerOptions {
  memories?: readonly MemorySpec[];
  stacks?: readonly MemoryStack[];
  root?: BackwardResult;
  results?: AttributedBackwardResult[];
}

/** Project the owner contexts of `trace` for root feedback `Too low.`. */
function ownerContexts(trace: RetainedTrace, options: OwnerOptions = {}): readonly LearnContext[] {
  const { memories = OWNER_TABLE, stacks = [P_STACK], root = OWNER_ROOT, results = ownerResults() } = options;
  return projectOwnerContexts(trace, memories, stacks, "Too low.", root, results);
}

/** Return what `fn` throws. */
function thrown(fn: () => unknown): unknown {
  try {
    fn();
  } catch (error) {
    return error;
  }
  throw new Error("expected an error");
}

/** A source reference's JSON form. */
function sourceJson(seq: number) {
  return { run_id: RUN, execution_seq: BigInt(seq), path: [] };
}

/** The materialized context of END's route reading `p`'s output. */
const P_OUTPUT_CONTEXT = {
  expression: "${source_5}",
  result: 2n,
  sources: [{ source: sourceJson(5), value: 2n }],
  role: "source",
  operand: null,
  positions: [],
  span: null,
  result_type: null,
  snapshot: null,
};

/** The materialized projection context of `p`'s route reading field `x` of execution `seq`. */
function fieldContext(seq: number) {
  return {
    expression: `\${source_${seq}}`,
    result: 1n,
    sources: [{ source: { ...sourceJson(seq), path: ["x"] }, value: 1n }],
    role: "projection",
    operand: ["x"],
    positions: [],
    span: null,
    result_type: null,
    snapshot: null,
  };
}

/** `p`'s execution coordinates: no execution scope. */
const P_COORDINATES = { execution_seq: 5n, owner_static_path: null, occurrence_path: null, execution_scope: null };

test("owner contexts project the producer's episode and feedback for its memory only", () => {
  const { trace } = chainTrace();
  const contexts = ownerContexts(trace);
  expect(contexts.map((context) => context.memory)).toEqual(["empty", "notes", "outer"]);
  const [empty, notes, outer] = contexts as LearnContext[];
  expect(empty).toEqual(new LearnContext(RUN, "empty", []));
  expect(notes!.runId).toBe(RUN);
  expect(notes!.episodes).toEqual([
    new MemoryEpisode(RUN, [], ["p"], MemoryEpisodeStatus.SUCCEEDED, { v: 1n }, { output: 2n, executionSeq: 5 }),
  ]);
  expect(notes!.feedback).toEqual([
    {
      ...P_COORDINATES,
      channel: "output",
      target_name: null,
      value: "Too low.",
      contribution_context: {
        message_index: 0n,
        sender: sourceJson(8),
        input_name: "value",
        sender_message_index: 0n,
        contribution_index: 0n,
        parents: [0n],
        contexts: [P_OUTPUT_CONTEXT],
      },
    },
    {
      ...P_COORDINATES,
      channel: "input",
      target_name: "v",
      value: "Raise v.",
      contribution_context: {
        message_index: 0n,
        parents: [0n],
        contexts: [P_OUTPUT_CONTEXT],
        contributions: [
          { source: sourceJson(2), value: { x: "Raise v." }, contribution_index: 0n, contexts: [fieldContext(2)] },
        ],
      },
    },
    {
      ...P_COORDINATES,
      channel: "input",
      target_name: "w",
      value: "Check w.",
      contribution_context: {
        message_index: 0n,
        parents: [0n],
        contexts: [P_OUTPUT_CONTEXT],
        contributions: [
          { source: sourceJson(0), value: { x: "Check w." }, contribution_index: 0n, contexts: [fieldContext(0)] },
        ],
      },
    },
    {
      ...P_COORDINATES,
      channel: "self",
      target_name: null,
      value: "Be careful.",
      message_index: 0n,
      contribution_context: {},
    },
    {
      ...P_COORDINATES,
      channel: "variable",
      target_name: "tone",
      value: "Softer.",
      message_index: 0n,
      contribution_context: {},
    },
  ]);
  // The root fallback skips START's source-zero message and keeps `p`'s.
  expect(outer!.episodes).toEqual([]);
  expect(outer!.feedback).toEqual([
    {
      execution_seq: null,
      owner_static_path: null,
      occurrence_path: null,
      execution_scope: null,
      root: true,
      channel: "input",
      target_name: "x",
      value: "Check w.",
      contribution_context: { message_index: 0n },
    },
  ]);
  for (const context of contexts) {
    expect(Object.isFrozen(context)).toBe(true);
    expect(Object.isFrozen(context.feedback)).toBe(true);
  }
});

test("owner context entries carry p's execution scope coordinates", () => {
  const position = new ExecutionPosition("map_item", ["outer"], 2);
  const scope = new ExecutionScope(["outer", "p"], { flowPath: ["outer"], occurrencePath: [position] });
  const { trace } = chainTrace({ pScope: scope });
  const notes = ownerContexts(trace)[1]!;
  const occurrencePath = [{ kind: "map_item", static_path: ["outer"], index: 2n }];
  const coordinates = {
    execution_seq: 5n,
    owner_static_path: ["outer", "p"],
    occurrence_path: occurrencePath,
    execution_scope: {
      static_path: ["outer", "p"],
      flow_path: ["outer"],
      occurrence_path: occurrencePath,
      boundary: null,
    },
  };
  expect(notes.feedback).toHaveLength(5);
  for (const entry of notes.feedback as Record<string, unknown>[]) {
    expect({
      execution_seq: entry.execution_seq,
      owner_static_path: entry.owner_static_path,
      occurrence_path: entry.occurrence_path,
      execution_scope: entry.execution_scope,
    }).toEqual(coordinates);
    expect(typeof (entry.occurrence_path as { index: unknown }[])[0]!.index).toBe("bigint");
  }
});

test.each<[string, OwnerOptions, ErrorClass, string]>([
  [
    "a table that isn't an array",
    { memories: new Set(OWNER_TABLE) as never },
    TypeError,
    "owner contexts require the complete recorded memory table",
  ],
  [
    "a table entry of another class",
    { memories: [{ ...EMPTY_SPEC } as MemorySpec] },
    TypeError,
    "owner contexts require the complete recorded memory table",
  ],
  [
    "stacks that aren't an array",
    { stacks: new Set([P_STACK]) as never },
    TypeError,
    "owner contexts require the recorded memory stacks",
  ],
  [
    "a stack of another class",
    { stacks: [{ ...P_STACK } as MemoryStack] },
    TypeError,
    "owner contexts require the recorded memory stacks",
  ],
  [
    "an unsorted table",
    { memories: [NOTES_SPEC, EMPTY_SPEC] },
    RangeError,
    "owner contexts require a memory table sorted by unique key",
  ],
  [
    "a repeated key",
    { memories: [NOTES_SPEC, NOTES_SPEC] },
    RangeError,
    "owner contexts require a memory table sorted by unique key",
  ],
  [
    "repeated stack locators",
    { stacks: [P_STACK, new MemoryStack(["p"], [])] },
    RangeError,
    "recorded memory stacks must have unique locators",
  ],
  [
    "a root result that disagrees with the transfers",
    { root: CHAIN_ROOT },
    RangeError,
    "backward root result must match the accepted START and source-zero projection",
  ],
  [
    "a scope without its recorded stack",
    { stacks: [] },
    RangeError,
    "historical memory scope disagrees with its recorded memory stack",
  ],
  [
    "a scope whose recorded stack differs",
    { stacks: [new MemoryStack(["p"], ["other"])] },
    RangeError,
    "historical memory scope disagrees with its recorded memory stack",
  ],
  [
    "a scope key absent from the table",
    { memories: [EMPTY_SPEC] },
    RangeError,
    "historical memory scope names a key absent from the recorded table",
  ],
])("owner contexts refuse %s", (_label, options, cls, message) => {
  expectError(() => ownerContexts(chainTrace().trace, options), cls, message);
});

test("OwnerContextError keeps the memory, the original error and the context", () => {
  const { trace } = chainTrace();
  const original = new RangeError("Bad.");
  const context = trace.backwardContext(P);
  const error = new OwnerContextError("notes", original, context);
  expect(error).toBeInstanceOf(Error);
  expect(error.name).toBe("OwnerContextError");
  expect(error.message).toBe("Bad.");
  expect(error.cause).toBe(original);
  expect(error.memory).toBe("notes");
  expect(error.original).toBe(original);
  expect(error.context).toBe(context);
  expect(new OwnerContextError("notes", original).context).toBeNull();
});

test.each<[string, () => unknown, string]>([
  [
    "an empty memory key",
    () => new OwnerContextError("", new RangeError("Bad.")),
    "owner context failure requires a memory key string",
  ],
  [
    "a memory key that isn't text",
    () => new OwnerContextError(1 as unknown as string, new RangeError("Bad.")),
    "owner context failure requires a memory key string",
  ],
  [
    "a context of another class",
    () => new OwnerContextError("notes", new RangeError("Bad."), {} as BackwardContext),
    "owner context failure context must be a BackwardContext or null",
  ],
])("OwnerContextError refuses %s", (_label, fn, message) => {
  expectError(fn, TypeError, message);
});

test("a feedback value freezeFeedback refuses fails the owner's context, naming its occurrence", () => {
  const { trace } = chainTrace();
  const error = thrown(() => ownerContexts(trace, { results: ownerResults("\uD800") })) as OwnerContextError;
  expect(error.constructor).toBe(OwnerContextError);
  expect(error.memory).toBe("notes");
  expect(error.original.constructor).toBe(RangeError);
  expect(error.original.message).toBe("Unicode surrogate is not valid canonical JSON");
  expect(error.message).toBe(error.original.message);
  expect(error.cause).toBe(error.original);
  expect(error.context).toEqual(trace.backwardContext(P));
});

test("a root fallback context that can't be built names no occurrence", () => {
  const trace = syntheticTrace("");
  const root = new BackwardResult({ selfFeedback: ["original text"] });
  const error = thrown(() =>
    projectOwnerContexts(trace, [OUTER_SPEC], [], "original text", root, []),
  ) as OwnerContextError;
  expect(error.constructor).toBe(OwnerContextError);
  expect(error.memory).toBe("outer");
  expect(error.context).toBeNull();
  expect(error.original.constructor).toBe(RangeError);
  expect(error.original.message).toBe("run_id must be non-empty");
});

test.each<[string, { answer: string[] } | null]>([
  ["null", null],
  ["record", { answer: ["stable"] }],
])(
  "the terminal trace keeps the actual root writer until quiescence: %s",
  async (_, value) => {
    const observations: (TraceTerminal | null)[] = [];
    const late = new FuncNode("late", () => {
      observations.push(built.state.trace!.terminal);
      return "last outcome";
    });
    const flow = graph([EndNode.value(END_ID, { outputName: "answer" }), late], [
      [START_ID, END_ID],
      [START_ID, "late"],
    ], { wiring: { [END_ID]: { answer: value }, late: {} } });
    const built = engine(flow, { learnable: true });

    await built.run();

    const trace = built.state.trace!;
    const endOccurrence = trace.occurrences.find((item) => item.nodeId === END_ID)!;
    expect(observations).toEqual([null]);
    expect(trace.occurrences.at(-1)!.nodeId).toBe("late");
    expect(trace.endId).toBe(END_ID);
    expect(trace.terminal!.succeeded).toBe(true);
    expect(trace.terminal!.source).toEqual(endOccurrence.source);
    expect(trace.terminal!.output).toEqual(value);
    expect(trace.terminal!.failure).toBeNull();
  },
);

// computeBackward, through the engine.

/** A backward callback that sends its feedback to input `x`. */
const passthrough = (_context: BackwardContext, feedback: readonly FeedbackValue[]) =>
  new BackwardResult({ inputFeedback: { x: feedback } });

/** A callback that must not run. */
function unreachable(label: string) {
  return (): BackwardResult => {
    throw new Error(label);
  };
}

/** Return the run's engine over `START -> work -> END`, END reporting `work`'s output as `result`. */
function workEngine(node: Node, inputs: Readonly<Record<string, unknown>> = {}) {
  return engine(graph([node], [[START_ID, "work"]], { outputs: [new FlowOutput("result", "${work.output}")] }), {
    inputs,
    learnable: true,
  });
}

/** Run `work` and return the `BackwardComputeError` computing `"review"` rejects with. */
async function computeError(node: Node): Promise<BackwardComputeError> {
  const built = workEngine(node);
  await built.run();
  const error = await rejected(computeBackward(built.state.trace!, "review"));
  expect(error).toBeInstanceOf(BackwardComputeError);
  return error as BackwardComputeError;
}

test("a diamond calls the shared producer once with two ordered messages", async () => {
  const producer = new RecordingBackwardNode("producer", (inputs) => inputs.x, passthrough, {
    reads: { x: "${input.topic}" },
  });
  const left = new RecordingBackwardNode("left", (inputs) => inputs.x, passthrough, {
    reads: { x: "${producer.output}" },
  });
  const right = new RecordingBackwardNode("right", (inputs) => inputs.x, passthrough, {
    reads: { x: "${producer.output}" },
  });
  const flow = graph([producer, left, right], [
    [START_ID, "producer"],
    ["producer", "left"],
    ["producer", "right"],
  ], { outputs: [new FlowOutput("left", "${left.output}"), new FlowOutput("right", "${right.output}")] });
  const built = engine(flow, { inputs: { topic: "reefs" }, learnable: true });
  await built.run();
  const trace = built.state.trace!;

  const [root, results] = await computeBackward(trace, { left: "L", right: "R" });

  validateAttributedBackward(trace, { left: "L", right: "R" }, root, results);
  expect([producer.calls.length, left.calls.length, right.calls.length]).toEqual([1, 1, 1]);
  expect(producer.calls[0]![1]).toEqual(["L", "R"]);
  expect(Object.isFrozen(producer.calls[0]![1])).toBe(true);
  expect(root.inputFeedback).toEqual({ topic: ["L", "R"] });
  expect(root.selfFeedback).toEqual([]);
  expect(root.variableFeedback).toEqual({});
  expect(results.map((item) => item.source)).toEqual(trace.occurrences.map((item) => item.source));
  expect(results.every((item) => item.invoked)).toBe(true);
});

test.each<[string, () => unknown, ErrorClass, string]>([
  [
    "an undeclared input",
    () => new BackwardResult({ inputFeedback: { x: ["valid"], unknown: ["bad"] } }),
    RangeError,
    "input_feedback selects undeclared input 'unknown'",
  ],
  [
    "a foreign variable",
    () => new BackwardResult({ inputFeedback: { x: ["valid"] }, variableFeedback: { foreign: ["bad"] } }),
    RangeError,
    "variable_feedback selects undeclared variable 'foreign'",
  ],
  [
    "a short positional message",
    () => new BackwardResult({ inputFeedback: { x: [[null]] } }),
    RangeError,
    'feedback["x"] must have exactly 2 items; got 1',
  ],
  ["another class (TS-only)", () => ({}), TypeError, "result must be a BackwardResult"],
])("the entire callback result is validated before routing any input: %s", async (_, bad, cls, message) => {
  const producer = new RecordingBackwardNode("producer", () => [1n, 2n], unreachable("routed partial result"));
  const consumer = new RecordingBackwardNode("consumer", (inputs) => inputs.x, () => bad() as BackwardResult, {
    reads: { x: "${producer.output}" },
  });
  const built = engine(graph([producer, consumer], [
    [START_ID, "producer"],
    ["producer", "consumer"],
  ], { outputs: [new FlowOutput("result", "${consumer.output}")] }), { learnable: true });
  await built.run();

  const error = (await rejected(computeBackward(built.state.trace!, "review"))) as BackwardComputeError;

  expect(error).toBeInstanceOf(BackwardComputeError);
  expect(error.failure.code).toBe("backward.result");
  expect(error.failure.nodeId).toBe("consumer");
  expect(error.failure.message).toBe(message);
  expect(error.failure.causeType).toBe(cls.name);
  expect((error.cause as Error).constructor).toBe(cls);
  expect(producer.calls).toEqual([]);
});

test.each<[string, FeedbackValue]>([
  ["null", null],
  ["positional nulls", [null, null]],
])("empty feedback keeps the complete history without invoking callbacks: %s", async (_, feedback) => {
  const node = new RecordingBackwardNode("work", () => [1n, 2n], unreachable("empty callback"));
  const built = workEngine(node);
  await built.run();
  const trace = built.state.trace!;

  const [root, results] = await computeBackward(trace, feedback);

  validateAttributedBackward(trace, feedback, root, results);
  expect(root).toEqual(new BackwardResult());
  expect(results.length).toBe(trace.occurrences.length);
  for (const result of results) {
    expect(result.invoked).toBe(false);
    expect(result.feedback).toEqual([]);
    expect(result.result).toEqual(new BackwardResult());
  }
  const error = await rejected(computeBackward(trace, [null]));
  expect((error as Error).constructor).toBe(RangeError);
  expect((error as Error).message).toBe("feedback must have exactly 2 items; got 1");
  expect(node.calls).toEqual([]);
});

test("root default feedback survives its literal route", async () => {
  const start = new StartNode(START_ID, {
    params: [
      new Parameter("topic", { hasDefault: true, default: "reefs" }),
      new Parameter("absent", { required: false }),
    ],
  });
  const built = engine(graph([start], [], { outputs: [new FlowOutput("result", "${input.topic}")] }), {
    learnable: true,
  });
  await built.run();
  const trace = built.state.trace!;

  const [root, results] = await computeBackward(trace, "");

  validateAttributedBackward(trace, "", root, results);
  expect(root.inputFeedback).toEqual({ topic: [""] });
});

test.each<[string, boolean]>([
  ["body failure", false],
  ["binding failure", true],
])("an accepted failure uses the recorded inputs and diagnostics: %s", async (_, early) => {
  const node = new RecordingBackwardNode(
    "work",
    () => {
      throw new Error("body broke");
    },
    (context, feedback) =>
      context.inputs === null
        ? new BackwardResult({ selfFeedback: ["binding unavailable"] })
        : new BackwardResult({ inputFeedback: { x: feedback }, selfFeedback: ["body failure"] }),
    { reads: { x: "${input.topic}" } },
  );
  if (early) node.params = [...node.params!, new Parameter("missing", { required: true })];
  const built = workEngine(node, { topic: "reefs" });
  await built.run();
  const trace = built.state.trace!;

  const [root, results] = await computeBackward(trace, "review failure");

  validateAttributedBackward(trace, "review failure", root, results);
  const [context, received] = node.calls[0]!;
  expect(received).toEqual(["review failure"]);
  expect(context.output).toBeNull();
  expect(context.failure).not.toBeNull();
  expect(context.inputs === null).toBe(early);
  expect(root.inputFeedback).toEqual(early ? {} : { topic: ["review failure"] });
  expect(root.selfFeedback).toEqual([]);
  expect(results.find((item) => sourceKey(item.source) === sourceKey(trace.terminal!.source!))!.invoked).toBe(true);
  expect((await rejected(computeBackward(trace, {})) as Error).constructor).toBe(RangeError);
});

test("a callback failure is attributed to its occurrence, with the original as the cause", async () => {
  const original = new Error("broken");
  const node = new RecordingBackwardNode("work", () => "value", () => {
    throw original;
  });

  const error = await computeError(node);

  const work = node.calls[0]![0];
  expect(error.name).toBe("BackwardComputeError");
  expect(error.message).toBe("broken");
  expect(error.cause).toBe(original);
  expect(error.failure).toEqual(new BackwardFailure("backward", "backward.execute", "broken", {
    scope: work.scope,
    runId: RUN_ID,
    nodeId: "work",
    nodeStartedSeq: work.nodeStartedSeq,
    causeType: "Error",
    traceback: original.stack!,
  }));
});

test("a host capability fault from a callback is attributed too, with the fault as the cause", async () => {
  // No host-control unwrapping: a fault a callback throws is the callback's failure.
  const fault = new CapabilityFault(new Error("broken"));
  const error = await computeError(new RecordingBackwardNode("work", () => "value", () => {
    throw fault;
  }));

  expect(error.failure.code).toBe("backward.execute");
  expect(error.failure.nodeId).toBe("work");
  expect(error.failure.causeType).toBe("CapabilityFault");
  expect(error.cause).toBe(fault);
});

test("an infrastructure fault reading the trace escapes unchanged", async () => {
  const original = new Error("broken");
  const node = new RecordingBackwardNode("work", () => "value", unreachable("callback"));
  const built = workEngine(node);
  await built.run();
  const trace = built.state.trace!;
  trace.lookupSource = () => {
    throw original;
  };

  expect(await rejected(computeBackward(trace, "review"))).toBe(original);
  expect(node.calls).toEqual([]);
});

test("a failed END routes text without fabricating an output context", async () => {
  const producer = new RecordingBackwardNode("producer", () => [1n, 2n], () => new BackwardResult());
  const built = engine(graph([producer], [[START_ID, "producer"]], {
    outputs: [new FlowOutput("result", "${producer.output}")],
  }), { learnable: true });
  built.state.flow.nodes.get(END_ID)!.preAsserts = [new Assertion("false", "fail END")];
  await built.run();

  const [root] = await computeBackward(built.state.trace!, "review failed END");

  expect(root).toEqual(new BackwardResult());
  const [context, messages] = producer.calls[0]!;
  expect(messages).toEqual(["review failed END"]);
  const details = context.downstreamContext.flatMap((entry) => entry.contexts as { role: string }[]);
  expect(details.some((detail) => detail.role === "output")).toBe(false);
});

test("a callback's message mutation changes neither the root feedback nor the recorded feedback", async () => {
  // The callback gets deep-frozen copies, so its write throws where Python's lands on a copy.
  let caught: unknown = null;
  const end = new RecordingBackwardNode(END_ID, () => [1n, 2n], (_context, feedback) => {
    try {
      (feedback[0] as FeedbackValue[])[0] = "mutated by callback";
    } catch (error) {
      caught = error;
    }
    return new BackwardResult();
  });
  const built = engine(graph([end], [[START_ID, END_ID]]), { learnable: true });
  await built.run();
  const message = [null, "review second"];

  const [, results] = await computeBackward(built.state.trace!, message);

  expect(caught).toBeInstanceOf(TypeError);
  expect(message).toEqual([null, "review second"]);
  expect(results.find((item) => item.invoked)!.feedback).toEqual([[null, "review second"]]);
});

// TS-only: promises, thrown values and the handler lookup.

test("an async callback is awaited", async () => {
  const node = new RecordingBackwardNode("work", (inputs) => inputs.x, async (_context, feedback) => {
    await Promise.resolve();
    return new BackwardResult({ inputFeedback: { x: feedback } });
  }, { reads: { x: "${input.topic}" } });
  const built = workEngine(node, { topic: "reefs" });
  await built.run();

  const [root] = await computeBackward(built.state.trace!, "review");

  expect(root.inputFeedback).toEqual({ topic: ["review"] });
});

test("a rejected callback promise is backward.execute", async () => {
  const original = new Error("async broken");
  const error = await computeError(new RecordingBackwardNode("work", () => "value", async () => {
    await Promise.resolve();
    throw original;
  }));

  expect(error.failure.code).toBe("backward.execute");
  expect(error.failure.message).toBe("async broken");
  expect(error.cause).toBe(original);
});

test.each<[string, unknown, string, string]>([
  ["a string", "x", "x", "string"],
  ["undefined", undefined, "undefined", "undefined"],
])("a thrown non-Error value gives its text, its type name and no traceback: %s", async (_, thrown, message, type) => {
  const error = await computeError(new RecordingBackwardNode("work", () => "value", () => {
    throw thrown;
  }));

  expect(error.failure.code).toBe("backward.execute");
  expect(error.failure.message).toBe(message);
  expect(error.failure.causeType).toBe(type);
  expect(error.failure.traceback).toBeNull();
  expect(error.cause).toBe(thrown);
});

test("an Error subclass's name is the cause type, and its notes follow the stack", async () => {
  class BrokenHook extends Error {
    override name = "BrokenHook";
  }
  const original = Object.assign(new BrokenHook("hook broke"), { notes: ["first note", 7, "second note"] });
  const error = await computeError(new RecordingBackwardNode("work", () => "value", () => {
    throw original;
  }));

  expect(error.failure.causeType).toBe("BrokenHook");
  expect(error.failure.traceback).toBe(`${original.stack}\nfirst note\nsecond note`);
});

test("a missing handler rejects as itself, not as backward.execute", async () => {
  const node = new RecordingBackwardNode("work", () => "value", unreachable("callback"));
  const built = workEngine(node);
  await built.run();
  const trace = built.state.trace!;
  Object.defineProperty(trace, "backwardHandlers", { value: new Map() });

  const error = await rejected(computeBackward(trace, "review"));

  expect((error as Error).constructor).toBe(Error);
  expect((error as Error).message).toBe("backward occurrence has no retained handler");
});

test("an error from result validation that isn't a TypeError or RangeError escapes", async () => {
  const original = new Error("forged result");
  const forged = Object.defineProperty(Object.create(BackwardResult.prototype), "inputFeedback", {
    get() {
      throw original;
    },
  }) as BackwardResult;

  const built = workEngine(new RecordingBackwardNode("work", () => "value", () => forged));
  await built.run();

  expect(await rejected(computeBackward(built.state.trace!, "review"))).toBe(original);
});

// computeBackward over routes, growth replay and template stops.

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

/** Return the result of `results` at `source`. */
function resultAt(results: readonly AttributedBackwardResult[], source: SourceRef): AttributedBackwardResult {
  return results.find((item) => sameSource(item.source, source))!;
}

test.each([false, true])(
  "a terminal trace latches the causal route and a later failure wins (explicit failure: %p)",
  async (explicit) => {
    const flow = terminalSkipRouteFlow({ sibling: true });
    const router = flow.nodes.get("router")!;
    router.forward = () => new Route("go");
    const calls: BackwardContext[] = [];
    router.backward = (context) => {
      calls.push(context);
      return new BackwardResult({ selfFeedback: ["route"] });
    };
    const observed: (TraceTerminal | null)[] = [];
    flow.nodes.get("unrelated")!.forward = () => {
      observed.push(built.state.trace!.terminal);
      if (explicit) throw new Error("later failure");
      return new Output("done");
    };
    const built = engine(flow, { learnable: true });
    const original = copyGraph(built.state.flow);
    await built.run();
    const trace = built.state.trace!;
    const chosen = trace.occurrences.find((item) => item.nodeId === (explicit ? "unrelated" : "router"))!;

    expect(observed).toEqual([null]);
    expect(trace.terminal!.source).toEqual(chosen.source);
    expect(trace.terminal!.succeeded).toBe(false);
    const [root, results] = await computeBackward(trace, "explain failure");
    validateAttributedBackward(trace, "explain failure", root, results);
    expect(root).toEqual(new BackwardResult());
    expect(calls.length).toBe(explicit ? 0 : 1);
    if (calls.length > 0) expect(calls[0]!.failure).toEqual(trace.terminal!.failure);
    const records = built.log.records();
    for (let length = 1; length <= records.length; length++) {
      const replay = replayRecords(new RunState(copyGraph(original)), records.slice(0, length), {
        growBuilder: noGrowth,
      });
      expect(replay.trace!.terminal).toEqual(length === records.length ? trace.terminal : null);
    }
  },
);

test.each([false, true])(
  "structural ancestry requires the exact metadata-prescribed transfer (transform: %p)",
  async (transform) => {
    const producer = new RecordingBackwardNode("producer", () => "value", () => new BackwardResult());
    const middle = new RecordingBackwardNode(
      "middle",
      (inputs) => inputs.x,
      (_context, feedback) =>
        new BackwardResult({ inputFeedback: { x: transform ? feedback.map((message) => `${message}!`) : feedback } }),
      { reads: { x: "${producer.output}" } },
    );
    middle.captureBackwardContext = () => ({ input_output_paths: { x: [] } });
    const left = new RecordingBackwardNode("left", (inputs) => inputs.x, passthrough, {
      reads: { x: "${middle.output}" },
    });
    const right = new RecordingBackwardNode("right", (inputs) => inputs.x, passthrough, {
      reads: { x: "${middle.output}" },
    });
    const { engine: built, memories, stacks } = ownerEngine(graph([producer, middle, left, right], [
      [START_ID, "producer"],
      ["producer", "middle"],
      ["middle", "left"],
      ["middle", "right"],
    ], { outputs: [new FlowOutput("left", "${left.output}"), new FlowOutput("right", "${right.output}")] }));
    await built.run();
    const trace = built.state.trace!;
    const feedback = { left: "same", right: "same" };

    const [root, results] = await computeBackward(trace, feedback);

    validateAttributedBackward(trace, feedback, root, results);
    const occurrence = trace.occurrences.find((item) => item.nodeId === "producer")!;
    const result = resultAt(results, occurrence.source);
    expect(result.feedback).toEqual(transform ? ["same!", "same!"] : ["same", "same"]);
    expect(result.attribution.map((item) => item.parents)).toEqual(transform ? [[0, 1], [0, 1]] : [[0], [1]]);
    const [context] = projectOwnerContexts(trace, memories, stacks, feedback, root, results);
    const received = feedbackEntries(context!).filter(
      (entry) => entry.channel === "output" && entry.execution_seq === BigInt(occurrence.source.executionSeq),
    );
    expect(received.map((entry) => entry.value)).toEqual([...result.feedback]);
    expect(received.map((entry) => (entry.contribution_context as { parents: unknown }).parents)).toEqual(
      result.attribution.map((item) => item.parents.map(BigInt)),
    );
    expect(received.map((entry) => entry.contribution_context)).toEqual([...producer.calls[0]![0].downstreamContext]);
  },
);

test("a root output that is the run inputs doesn't invent an executed producer", async () => {
  const start = new StartNode(START_ID, {
    params: [new Parameter("topic"), new Parameter("omitted", { hasDefault: true, default: "unused" })],
  });
  (start as Node).forward = () => new Route(ELSE_HANDLE);
  const flow = CompiledFlow.fromParts(new Map([[START_ID, start]]), []);
  // `endId` is readonly and fixed by the constructor.
  (flow as { endId: string }).endId = START_ID;
  const initialPool = new VariablePool(START_ID);
  initialPool.set(START_ID, { topic: "reefs", undeclared: "extra" });
  initialPool.addSystem("run_id", "preseed");
  const state = new RunState(flow);
  const log = new MemoryRunLog("preseed");
  commitPrepared(log, state, prepareRunStarted(state, {
    runId: "preseed",
    flowFingerprint: "f".repeat(64),
    startWiringOverride: {},
    effectiveLlmByStaticPath: {},
    numWorkers: 0,
    learnable: true,
  }));
  commitInitialPool(log, state, initialPool);
  await new FlowEngine(state, log, { growBuilder: noGrowth, numWorkers: 0 }).run();
  const trace = state.trace!;

  const [root, results] = await computeBackward(trace, "review");

  validateAttributedBackward(trace, "review", root, results);
  expect(root.inputFeedback).toEqual({ topic: ["review"] });
  expect(root.selfFeedback).toEqual([]);
  expect(results.every((result) => !result.invoked)).toBe(true);
  expect(trace.terminal!.source!.executionSeq).toBe(0);
});

test("opaque expression stops are recoverable without rerunning functions", async () => {
  const calls: unknown[] = [];
  standIn("backward_opaque", (value) => {
    calls.push(value);
    return value;
  }, [{ name: "value" }]);
  const node = new RecordingBackwardNode("work", (inputs) => inputs.x, passthrough, {
    reads: { x: "${backward_opaque(input.topic)}" },
  });
  const built = workEngine(node, { topic: "reefs" });
  await built.run();
  const trace = built.state.trace!;

  const [root, results] = await computeBackward(trace, "review opaque");

  validateAttributedBackward(trace, "review opaque", root, results);
  const occurrence = trace.occurrences.find((item) => item.nodeId === "work")!;
  const result = resultAt(results, occurrence.source).result;
  const route = occurrence.inputRoute.children[occurrence.inputRoute.names.indexOf("x")]!;
  const recovered = routeFeedback(route, result.inputFeedback.x![0]!, (source) => trace.lookupSource(source));
  expect(recovered.contributions).toEqual([]);
  expect(recovered.stopped[0]!.feedback).toBe("review opaque");
  expect(recovered.stopped[0]!.context.at(-1)!.result).toBe("reefs");
  expect(root.inputFeedback).toEqual({});
  expect(calls).toEqual(["reefs"]);
});

test("a backward context is trace-owned, detached and survives replay", async () => {
  const node = new RecordingBackwardNode("work", (inputs) => inputs.x, passthrough, { reads: { x: "${input.topic}" } });
  const built = workEngine(node, { topic: [1n, 2n] });
  const original = copyGraph(built.state.flow);
  await built.run();
  const trace = built.state.trace!;
  const occurrence = trace.occurrences.find((item) => item.nodeId === "work")!;
  const context = trace.backwardContext(occurrence.source);

  // The context is frozen, so the write throws where Python's lands on a detached copy.
  expect(() => (context.inputs!.x as unknown[]).push(3n)).toThrow(TypeError);
  expect(trace.backwardContext(occurrence.source).inputs).toEqual({ x: [1n, 2n] });
  const before = built.log.records();
  const live = await computeBackward(trace, ["one", null]);
  const recapture = spyOn(Node.prototype, "captureBackwardContext").mockImplementation(() => {
    throw new Error("replay recaptured");
  });
  try {
    const replay = replayRecords(new RunState(copyGraph(original)), built.log.records(), { growBuilder: noGrowth });
    expect(replay.trace!.backwardContext(occurrence.source)).toEqual(context);
    expect(await computeBackward(replay.trace!, ["one", null])).toEqual(live);
  } finally {
    recapture.mockRestore();
  }
  expect(built.log.records()).toEqual(before);
});

test.each([false, true])(
  "canonical file input feedback keeps literal and opaque stops (opaque: %p)",
  async (opaque) => {
    const value = { files: [new FileRef({ uri: "file:///tmp/example.txt" })] };
    const calls: null[] = [];
    standIn("backward_file", () => {
      calls.push(null);
      return value;
    });
    const feedback = { files: [{ uri: "Check the URI." }] };
    const node = new RecordingBackwardNode("work", () => "done", () => new BackwardResult({
      inputFeedback: { file: [feedback] },
    }), { reads: { file: opaque ? "${backward_file()}" : value } });
    const { engine: built, memories, stacks } = ownerEngine(graph([node], [[START_ID, "work"]], {
      outputs: [new FlowOutput("result", "${work.output}")],
    }));
    await built.run();
    const trace = built.state.trace!;

    const [root, results] = await computeBackward(trace, "review");

    const occurrence = trace.occurrences.find((item) => item.nodeId === "work")!;
    const result = resultAt(results, occurrence.source).result;
    const capture = occurrence.record.forwardCapture!;
    const recovered = routeFeedback(
      occurrence.inputRoute.children[0]!,
      result.inputFeedback.file![0]!,
      (source) => trace.lookupSource(source),
      { recordedValue: capture.boundInputs!.file },
    );
    expect(root).toEqual(new BackwardResult());
    expect(recovered.contributions).toEqual([]);
    expect(recovered.stopped.map((stop) => stop.feedback)).toEqual(opaque ? [feedback] : []);
    const [context] = projectOwnerContexts(trace, memories, stacks, "review", root, results);
    const entries = feedbackEntries(context!);
    const returned = entries.filter((entry) => entry.target_name === "file" && entry.channel === "input");
    expect(returned.map((entry) => entry.value)).toEqual([feedback]);
    const stopped = entries.filter((entry) => entry.channel === "stopped").map((entry) => entry.value);
    expect(stopped).toEqual(opaque ? [feedback] : []);
    expect(calls).toEqual(opaque ? [null] : []);
  },
);

test("a backward context the trace can't build escapes unchanged before the callback (TS-only)", async () => {
  // Context building sits outside the callback's `try`, so its fault isn't the callback's.
  const original = new Error("context broke");
  const node = new RecordingBackwardNode("work", () => "value", unreachable("callback"));
  const built = workEngine(node);
  await built.run();
  const trace = built.state.trace!;
  trace.backwardContext = () => {
    throw original;
  };

  expect(await rejected(computeBackward(trace, "review"))).toBe(original);
  expect(node.calls).toEqual([]);
});

// computeBackward through CASE, AGENT, and CALL, MAP and LOOP growth.

/** Return whether `value` is a record holding `key`. */
function holds(value: unknown, key: string): value is Readonly<Record<string, FeedbackValue>> {
  return typeof value === "object" && value !== null && !Array.isArray(value) && key in value;
}

test("a MAP's whole-list context and item order survive reversed completion", async () => {
  const secondDone = Promise.withResolvers<void>();
  const originalForward = CodeNode.prototype.forward;
  const calls = new Map<number | null, [BackwardContext, readonly FeedbackValue[]]>();
  const forward = spyOn(CodeNode.prototype, "forward").mockImplementation(async function (this: CodeNode, inputs) {
    if (this.executionScope!.occurrencePath.at(-1)!.index === 0) {
      // Bounded, so a run where map#1 never finishes still ends and `finally` restores the spies.
      let timer: ReturnType<typeof setTimeout> | undefined;
      const timeout = new Promise<never>((_resolve, reject) => {
        timer = setTimeout(() => reject(new Error("map#1 never finished")), 2000);
      });
      try {
        await Promise.race([secondDone.promise, timeout]);
      } finally {
        clearTimeout(timer);
      }
    }
    return originalForward.call(this, inputs);
  });
  const backward = spyOn(CodeNode.prototype, "backward").mockImplementation((context, feedback) => {
    calls.set(context.nodeStartedSeq, [context, feedback]);
    return new BackwardResult({ inputFeedback: { x: feedback } });
  });
  try {
    const work = new CodeNode("work", { code: "return inputs['x']" });
    work.params = [new Parameter("x")];
    const child = graph([work], [[START_ID, "work"]], {
      outputs: [new FlowOutput("result", "${work.output}")],
      wiring: { work: { x: "${input.x}" } },
    });
    const mapper = new MapNode("map", { flowId: "child", child });
    mapper.params = [new Parameter("x")];
    const producer = new RecordingBackwardNode(
      "producer",
      (inputs) => inputs.items,
      (_context, feedback) => new BackwardResult({ inputFeedback: { items: feedback } }),
      { reads: { items: "${input.items}" } },
    );
    const flow = graph([producer, mapper], [[START_ID, "producer"], ["producer", "map"]], {
      outputs: [new FlowOutput("result", "${map.output}")],
      wiring: { producer: { items: "${input.items}" }, map: { over: "${producer.output}", x: "${item}" } },
    });
    const { engine: built, memories, stacks } = ownerEngine(flow, {
      inputs: { items: [7n, 7n] },
      numWorkers: 2,
      growBuilder: buildGrow,
      onEvent: (event) => {
        if (event instanceof NodeSucceeded && event.nodeId === `map#1/${END_ID}`) secondDone.resolve();
      },
    });
    await built.run();
    const trace = built.state.trace!;
    const ends = trace.occurrences
      .map((item) => item.nodeId)
      .filter((nodeId) => nodeId === `map#0/${END_ID}` || nodeId === `map#1/${END_ID}`);
    expect(ends).toEqual([`map#1/${END_ID}`, `map#0/${END_ID}`]);

    const [root, results] = await computeBackward(trace, "review both items");

    validateAttributedBackward(trace, "review both items", root, results);
    const both = [["review both items", null], [null, "review both items"]];
    expect(producer.calls[0]![1]).toEqual(both);
    expect(root.inputFeedback).toEqual({ items: both });
    const itemContext = (index: number) => (detail: Readonly<Record<string, unknown>>) =>
      detail.role === "output" && isDeepStrictEqual(detail.result, [7n, 7n]) &&
      isDeepStrictEqual(detail.positions, [[BigInt(index)]]);
    for (const index of [0, 1]) {
      const occurrence = trace.occurrences.find((item) => item.nodeId === `map#${index}/work`)!;
      const [context, feedback] = calls.get(occurrence.source.executionSeq)!;
      expect(feedback).toEqual(["review both items"]);
      const details = context.downstreamContext.flatMap(
        (entry) => entry.contexts as readonly Readonly<Record<string, unknown>>[],
      );
      expect(details.some(itemContext(index))).toBe(true);
    }
    expect(results).toHaveLength(trace.occurrences.length);
    const ownerContexts = projectOwnerContexts(trace, memories, stacks, "review both items", root, results);
    expect(ownerContexts).toHaveLength(1);
    const [ownerContext] = ownerContexts;
    expect(ownerContext!.episodes).toHaveLength(trace.occurrences.length);
    const entries = feedbackEntries(ownerContext!);
    const producerReceived = entries.filter(
      (entry) => entry.channel === "output" && isDeepStrictEqual(entry.owner_static_path, ["producer"]),
    );
    expect(producerReceived.map((entry) => entry.value)).toEqual(both);
    for (const index of [0, 1]) {
      const occurrence = trace.occurrences.find((item) => item.nodeId === `map#${index}/work`)!;
      const received = entries.filter(
        (entry) => entry.channel === "output" && entry.execution_seq === BigInt(occurrence.source.executionSeq),
      );
      expect(received).toHaveLength(1);
      const path = [{ kind: "map_item", static_path: ["map"], index: BigInt(index) }];
      expect(received[0]!.occurrence_path).toEqual(path);
      expect((received[0]!.execution_scope as Readonly<Record<string, unknown>>).occurrence_path).toEqual(path);
      const contribution = received[0]!.contribution_context as Readonly<Record<string, unknown>>;
      expect(contribution).toEqual(calls.get(occurrence.source.executionSeq)![0].downstreamContext[0]!);
      const contexts = contribution.contexts as readonly Readonly<Record<string, unknown>>[];
      expect(contexts.some(itemContext(index))).toBe(true);
    }
  } finally {
    forward.mockRestore();
    backward.mockRestore();
  }
});

test("a diamond crossing a CALL uses one walk and preserves the child's default projection", async () => {
  const calls: BackwardContext[] = [];
  const backward = spyOn(CodeNode.prototype, "backward").mockImplementation((context, feedback) => {
    calls.push(context);
    return new BackwardResult({
      inputFeedback: { x: feedback, literal: ["child default"] },
      selfFeedback: ["child only"],
    });
  });
  try {
    const start = new StartNode(START_ID, {
      params: [new Parameter("x"), new Parameter("literal", { hasDefault: true, default: "default" })],
    });
    const work = new CodeNode("work", { code: "return inputs['x']" });
    work.params = [new Parameter("x"), new Parameter("literal")];
    const child = graph([start, work], [[START_ID, "work"]], {
      outputs: [new FlowOutput("result", "${work.output}")],
      wiring: { work: { x: "${input.x}", literal: "${input.literal}" } },
    });
    child.memories = [new MemoryDecl("child_notes", "child_notes", {}, new DeclSite([], ["def", "child"], null))];
    const call = new CallNode("call", { flowId: "child", child });
    call.params = [new Parameter("x")];
    const producer = new RecordingBackwardNode("producer", (inputs) => inputs.x, passthrough, {
      reads: { x: "${input.topic}" },
    });
    const right = new RecordingBackwardNode("right", (inputs) => inputs.x, passthrough, {
      reads: { x: "${producer.output}" },
    });
    const flow = graph([producer, call, right], [[START_ID, "producer"], ["producer", "call"], ["producer", "right"]], {
      outputs: [new FlowOutput("left", "${call.output}"), new FlowOutput("right", "${right.output}")],
      wiring: {
        producer: { x: "${input.topic}" },
        call: { x: "${producer.output}" },
        right: { x: "${producer.output}" },
      },
    });
    const { engine: built, memories, stacks } = ownerEngine(flow, {
      inputs: { topic: "reefs" },
      growBuilder: buildGrow,
    });
    await built.run();
    const trace = built.state.trace!;
    const feedback = { left: "L", right: "R" };

    const [root, results] = await computeBackward(trace, feedback);

    validateAttributedBackward(trace, feedback, root, results);
    expect(calls).toHaveLength(1);
    expect(producer.calls).toHaveLength(1);
    expect(producer.calls[0]![1]).toEqual(["L", "R"]);
    expect(root.inputFeedback).toEqual({ topic: ["L", "R"] });
    expect(trace.boundaries).toHaveLength(1);
    const [boundary] = trace.boundaries;
    expect(resultAt(results, boundary!.startSource!).result.inputFeedback).toEqual({
      x: ["L"],
      literal: ["child default"],
    });
    expect(resultAt(results, boundary!.owner).invoked).toBe(false);
    expect(root.selfFeedback).toEqual([]);
    expect(root.variableFeedback).toEqual({});
    const contexts = projectOwnerContexts(trace, memories, stacks, feedback, root, results);
    expect(contexts).toHaveLength(2);
    for (const context of contexts) {
      const entries = feedbackEntries(context);
      const defaults = entries.filter(
        (entry) =>
          entry.execution_seq === BigInt(boundary!.startSource!.executionSeq) &&
          entry.channel === "input" &&
          entry.target_name === "literal",
      );
      expect(defaults).toHaveLength(1);
      expect(defaults[0]!.value).toBe("child default");
      expect((defaults[0]!.contribution_context as { contributions: unknown }).contributions).toEqual([]);
      const selfEntries = entries.filter((entry) => entry.channel === "self");
      expect(selfEntries).toHaveLength(1);
      expect(selfEntries[0]!.value).toBe("child only");
      expect(selfEntries[0]!.message_index).toBe(0n);
      expect(selfEntries[0]!.owner_static_path).toEqual(["call", "work"]);
      expect(selfEntries[0]!.root).toBeFalsy();
    }
  } finally {
    backward.mockRestore();
  }
});

test("CASE control and an unselected branch don't receive feedback", async () => {
  const control = new RecordingBackwardNode("control", () => true, unreachable("control feedback"));
  const value = new RecordingBackwardNode("value", (inputs) => inputs.x, passthrough, {
    reads: { x: "${input.topic}" },
  });
  const branch = new CaseNode("case", [new Case("go", { when: "${__r0}" })]);
  branch.params = [new Parameter("__r0")];
  const selected = new RecordingBackwardNode("selected", (inputs) => inputs.x, passthrough, {
    reads: { x: "${value.output}" },
  });
  const unused = new RecordingBackwardNode("unused", () => {
    throw new Error("unselected forward");
  }, unreachable("unselected backward"));
  const flow = graph([control, value, branch, selected, unused], [
    [START_ID, "control"],
    [START_ID, "value"],
    ["control", "case"],
    ["case", "selected", "go"],
    ["case", "unused", ELSE_HANDLE],
    ["value", "selected"],
  ], {
    outputs: [new FlowOutput("result", "${selected.output | unused.output}")],
    wiring: {
      value: { x: "${input.topic}" },
      case: { __r0: "${control.output}" },
      selected: { x: "${value.output}" },
    },
  });
  const built = engine(flow, { inputs: { topic: "reefs" }, learnable: true });
  await built.run();
  const trace = built.state.trace!;

  const [root, results] = await computeBackward(trace, "review selected");

  validateAttributedBackward(trace, "review selected", root, results);
  expect(root.inputFeedback).toEqual({ topic: ["review selected"] });
  expect(control.calls).toEqual([]);
  expect(unused.calls).toEqual([]);
  const byNode = new Map(trace.occurrences.map((item) => [item.nodeId, item]));
  expect(byNode.has("unused")).toBe(false);
  expect(byNode.get("case")!.record.forwardCapture!.inputNames).toEqual([]);
  expect(resultAt(results, byNode.get("case")!.source).invoked).toBe(false);
});

test("an AGENT's inventory is recorded and empty feedback does no backward work", async () => {
  const calls: BackwardContext[] = [];
  const forward = spyOn(AgentNode.prototype, "forward").mockImplementation(
    async (inputs) => new Output(inputs.topic),
  );
  // AGENT's own `backward` is async, so the spy goes through `Node`'s signature, which may answer
  // synchronously.
  const backward = spyOn(AgentNode.prototype as Node, "backward").mockImplementation((context, feedback) => {
    calls.push(context);
    return new BackwardResult({
      inputFeedback: { topic: feedback },
      selfFeedback: ["agent note"],
      variableFeedback: { instruction: ["local note"] },
    });
  });
  try {
    const variables = parametersFromYaml("variables:\n  instruction: !str concise\n", readTypedefs({}), {
      field: "variables",
      requireDefault: true,
    });
    const node = new AgentNode("agent", { prompt: "${topic}", variables });
    node.params = [new Parameter("topic")];
    const built = engine(graph([node], [[START_ID, "agent"]], {
      outputs: [new FlowOutput("result", "${agent.output}")],
      wiring: { agent: { topic: "${input.topic}" } },
    }), { inputs: { topic: "reefs" }, learnable: true });
    await built.run();
    const trace = built.state.trace!;
    node.params = [];
    node.variables = [];
    node.captureBackwardContext = () => {
      throw new Error("recaptured invocation");
    };

    const [, inactive] = await computeBackward(trace, null);

    expect(calls).toEqual([]);
    expect(inactive).toHaveLength(trace.occurrences.length);
    const [root, active] = await computeBackward(trace, "review");
    expect(calls[0]!.inputNames).toEqual(["topic"]);
    expect(calls[0]!.variableNames).toEqual(["instruction"]);
    expect(root.inputFeedback).toEqual({ topic: ["review"] });
    expect(root.selfFeedback).toEqual([]);
    expect(root.variableFeedback).toEqual({});
    expect(active.some((result) => isDeepStrictEqual(result.result.variableFeedback, { instruction: ["local note"] })))
      .toBe(true);
  } finally {
    forward.mockRestore();
    backward.mockRestore();
  }
});

test("an empty MAP retains its synthetics without inventing item contributors", async () => {
  const work = new CodeNode("work", { code: "throw new Error('empty map ran a child');" });
  work.params = [new Parameter("x")];
  const child = graph([work], [[START_ID, "work"]], {
    outputs: [new FlowOutput("result", "${work.output}")],
    wiring: { work: { x: "${input.x}" } },
  });
  const mapper = new MapNode("map", { flowId: "child", child });
  mapper.params = [new Parameter("x")];
  const built = engine(graph([mapper], [[START_ID, "map"]], {
    outputs: [new FlowOutput("result", "${map.output}")],
    wiring: { map: { over: [], x: "${item}" } },
  }), { learnable: true, growBuilder: buildGrow });
  await built.run();
  const trace = built.state.trace!;

  const [root, results] = await computeBackward(trace, "review empty list");

  validateAttributedBackward(trace, "review empty list", root, results);
  expect(root).toEqual(new BackwardResult());
  expect(trace.boundaries).toEqual([]);
  expect(trace.occurrences).toHaveLength(5);
  expect(results).toHaveLength(5);
});

test("LOOP carried routes use the same fold after live pruning", async () => {
  const calls: (number | null)[] = [];
  const backward = spyOn(CodeNode.prototype, "backward").mockImplementation((context, feedback) => {
    calls.push(context.nodeStartedSeq);
    return new BackwardResult({
      inputFeedback: {
        n: feedback.map((message) => (typeof message === "string" ? message : (message as { n: FeedbackValue }).n)),
      },
    });
  });
  try {
    const flow = (await loadFlow(`id: carried
name: carried
input:
  n: !int
defs:
  body:
    long_term_memory: body_notes
    input:
      n: !int
    nodes:
      work:
        kind: code
        input: {n: '\${input.n}'}
        code: "return {'n': inputs['n'] - 1}"
        output: {n: int}
    output: \${work.output}
nodes:
  loop:
    kind: loop
    call: body
    times: 2
    input: {n: '\${input.n}'}
output: \${loop.output}
`)).compiled;
    const { engine: built, memories, stacks } = ownerEngine(flow, { inputs: { n: 3n }, growBuilder: buildGrow });
    const original = copyGraph(built.state.flow);
    await built.run();
    const trace = built.state.trace!;

    const [root, results] = await computeBackward(trace, "review carried value");

    validateAttributedBackward(trace, "review carried value", root, results);
    expect(root.inputFeedback).toEqual({ n: ["review carried value"] });
    expect(calls).toHaveLength(2);
    expect(new Set(calls).size).toBe(2);
    const drivers = trace.occurrences.filter((item) =>
      isDeepStrictEqual(item.record.forwardCapture!.scope!.staticPath, ["loop"]),
    );
    expect(drivers).toHaveLength(3);
    const reached = new Set(results.filter((result) => result.invoked).map((result) => sourceKey(result.source)));
    expect(reached.has(sourceKey(drivers.at(-1)!.source))).toBe(true);
    expect(drivers.slice(0, -1).every((item) => !reached.has(sourceKey(item.source)))).toBe(true);
    const contexts = projectOwnerContexts(trace, memories, stacks, "review carried value", root, results);
    expect(contexts).toHaveLength(2);
    const byKey = new Map(contexts.map((context) => [context.memory, context]));
    const policy = byKey.get(POLICY)!;
    expect(policy.episodes).toHaveLength(trace.occurrences.length);
    expect(byKey.get("body_notes")!.episodes).toHaveLength(6);
    expect(new Set(policy.episodes.map((episode) => episode.executionSeq)).size).toBe(policy.episodes.length);
    expect(trace.occurrences.some((item) => !built.state.flow.nodes.has(item.nodeId))).toBe(true);
    for (const ownerContext of contexts) {
      for (const episode of ownerContext.episodes) {
        const occurrence = trace.occurrences.find((item) => item.source.executionSeq === episode.executionSeq)!;
        expect(episode.ownerStaticPath).toEqual(occurrence.memoryScope!.staticPath);
        expect(episode.occurrencePath).toEqual(occurrence.memoryScope!.occurrencePath);
      }
    }
    for (const driver of drivers.slice(0, -1)) {
      const episode = policy.episodes.find((item) => item.executionSeq === driver.source.executionSeq)!;
      expect((episode.output as { kind: unknown }).kind).toBe("grow");
      expect(
        feedbackEntries(policy).some((entry) => entry.execution_seq === BigInt(driver.source.executionSeq)),
      ).toBe(false);
    }
    backward.mockImplementation(unreachable("projection replay invoked callback"));
    const replay = replayRecords(new RunState(original), built.log.records(), { growBuilder: buildGrow });
    expect(projectOwnerContexts(replay.trace!, memories, stacks, "review carried value", root, results)).toEqual(
      contexts,
    );
  } finally {
    backward.mockRestore();
  }
});

test.each([false, true])(
  "a MAP's canonical item shape preserves its projected seed stop (opaque: %p)",
  async (opaque) => {
    const files = [new FileRef({ uri: "file:///tmp/same.txt" }), new FileRef({ uri: "file:///tmp/same.txt" })];
    const calls: null[] = [];
    standIn("backward_files", () => {
      calls.push(null);
      return files;
    });
    const child = graph([new StartNode(START_ID, { params: [new Parameter("file")] })], [], {
      outputs: [new FlowOutput("uri", "${input.file.uri}")],
    });
    const mapper = new MapNode("map", { flowId: "child", child });
    mapper.params = [new Parameter("file")];
    const flow = graph([mapper], [[START_ID, "map"]], {
      outputs: [new FlowOutput("files", "${map.output}")],
      wiring: { map: { over: opaque ? "${backward_files()}" : files, file: "${item}" } },
    });
    const { engine: built, memories, stacks } = ownerEngine(flow, { growBuilder: buildGrow });
    await built.run();
    const trace = built.state.trace!;
    const feedback = [null, "Check second URI."];

    const [root, results] = await computeBackward(trace, feedback);

    const start = trace.occurrence(trace.boundaries[1]!.startSource!);
    const result = resultAt(results, start.source).result;
    expect(result.inputFeedback).toEqual({ file: [{ uri: "Check second URI." }] });
    const routed = routeFeedback(
      start.inputRoute.children[0]!,
      result.inputFeedback.file![0]!,
      (source) => trace.lookupSource(source),
      { recordedValue: start.record.forwardCapture!.boundInputs!.file },
    );
    expect(root).toEqual(new BackwardResult());
    expect(routed.contributions).toEqual([]);
    const stops = opaque ? [[null, { uri: "Check second URI." }]] : [];
    expect(routed.stopped.map((stop) => stop.feedback)).toEqual(stops);
    const [context] = projectOwnerContexts(trace, memories, stacks, feedback, root, results);
    const stopped = feedbackEntries(context!).filter((entry) => entry.channel === "stopped");
    expect(stopped.map((entry) => entry.value)).toEqual(stops);
    // The expansion checks its growth against the rebuild from its commits' `from`, which runs the
    // MAP's binding, and so the builtin, once more; backward reevaluates nothing.
    expect(calls).toEqual(opaque ? [null, null] : []);
  },
);

const DEEP_FILES = "item." + "nested.".repeat(40) + "files";

test.each([
  ["+ []", "+ []"],
  ["+ the same list", `+ ${DEEP_FILES}`],
])(
  "a MAP's bounded projected file lists compose before stopping (operator: %s)",
  async (_label, operator) => {
    let raw: Record<string, unknown> = { files: [new FileRef({ uri: "file:///tmp/x" })] };
    for (let i = 0; i < 40; i++) raw = { nested: raw };
    const calls: null[] = [];
    standIn("deep_map_files", () => {
      calls.push(null);
      return [raw];
    });
    const child = graph([new StartNode(START_ID, { params: [new Parameter("files")] })], [], {
      outputs: [new FlowOutput("files", "${input.files}")],
    });
    const mapper = new MapNode("map", { flowId: "child", child });
    mapper.params = [new Parameter("files")];
    const flow = graph([mapper], [[START_ID, "map"]], {
      outputs: [new FlowOutput("files", "${map.output}")],
      wiring: { map: { over: "${deep_map_files()}", files: `\${${DEEP_FILES} ${operator}}` } },
    });
    const built = engine(flow, { learnable: true, growBuilder: buildGrow });
    await built.run();
    const trace = built.state.trace!;
    expect(trace.terminal!.succeeded).toBe(true);
    const feedback: FeedbackValue[] = [{ uri: "Review." }];
    if (operator !== "+ []") feedback.push(null);

    const [root, results] = await computeBackward(trace, [feedback]);

    const start = trace.occurrence(trace.boundaries[0]!.startSource!);
    const result = resultAt(results, start.source).result;
    const routed = routeFeedback(
      start.inputRoute.children[0]!,
      result.inputFeedback.files![0]!,
      (source) => trace.lookupSource(source),
      { recordedValue: start.record.forwardCapture!.boundInputs!.files },
    );
    expect(root).toEqual(new BackwardResult());
    expect(routed.contributions).toEqual([]);
    expect(routed.stopped.map((stop) => stop.feedback)).toEqual([[{ uri: "Review." }]]);
    // Once live and once in the expansion's check against its rebuild from `from`.
    expect(calls).toEqual([null, null]);
  },
);

/** One LOOP pass-through flow: its source for `times`, the root feedback and the expected root input feedback. */
interface PassthroughFlow {
  source: (times: number) => string;
  feedback: FeedbackValue;
  expected: Readonly<Record<string, readonly FeedbackValue[]>>;
}

const PASSTHROUGH_FLOWS: Readonly<Record<string, PassthroughFlow>> = {
  "chat-shaped": {
    source: (times) => `id: passthrough
name: passthrough
input:
  n: !int
  tag: !str
defs:
  body:
    input:
      n: !int
      tag: !str
    nodes:
      work:
        kind: code
        input: {n: '\${input.n}', tag: '\${input.tag}'}
        code: "return {'n': inputs['n'] - 1, 'seen': inputs['tag']}"
        output: {n: int, seen: str}
    output: \${work.output}
nodes:
  loop:
    kind: loop
    call: body
    times: ${times}
    input: {n: '\${input.n}', tag: '\${input.tag}', seen: ''}
output: \${loop.output}
`,
    feedback: { seen: "review seen" },
    expected: { tag: ["review seen"] },
  },
  "loop-output": {
    source: (times) => `id: passthrough
name: passthrough
input:
  n: !int
  tag: !str
defs:
  body:
    input:
      n: !int
    nodes:
      work:
        kind: code
        input: {n: '\${input.n}'}
        code: "return {'n': inputs['n'] - 1}"
        output: {n: int}
    output: \${work.output}
nodes:
  loop:
    kind: loop
    call: body
    times: ${times}
    input: {n: '\${input.n}', tag: '\${input.tag}'}
output: \${loop.output}
`,
    feedback: { tag: "review tag" },
    expected: { tag: ["review tag"] },
  },
  "nested-call": {
    source: (times) => `id: passthrough
name: passthrough
input:
  n: !int
  tag: !str
defs:
  dec:
    input:
      n: !int
    nodes:
      work:
        kind: code
        input: {n: '\${input.n}'}
        code: "return {'n': inputs['n'] - 1}"
        output: {n: int}
    output: \${work.output}
  body:
    input:
      n: !int
    nodes:
      inner:
        kind: call
        call: dec
        input: {n: '\${input.n}'}
    output: \${inner.output}
nodes:
  loop:
    kind: loop
    call: body
    times: ${times}
    input: {n: '\${input.n}', tag: '\${input.tag}'}
output: \${loop.output}
`,
    feedback: { tag: "review tag" },
    expected: { tag: ["review tag"] },
  },
};

test.each(Object.keys(PASSTHROUGH_FLOWS).flatMap((shape) => [1, 2, 3, 6].map((times) => [shape, times] as const)))(
  "a LOOP pass-through field routes feedback to its seed (%s, times %d)",
  async (shape, times) => {
    // The body maps its output `seen` back to the `tag` input it echoed.
    const backward = spyOn(CodeNode.prototype, "backward").mockImplementation((_context, feedback) => {
      const routed: Record<string, FeedbackValue[]> = {};
      for (const message of feedback) {
        for (const [output, target] of [["n", "n"], ["seen", "tag"]] as const) {
          if (holds(message, output)) (routed[target] ??= []).push(message[output]!);
        }
      }
      return new BackwardResult({ inputFeedback: routed });
    });
    try {
      const { source, feedback, expected } = PASSTHROUGH_FLOWS[shape]!;
      const flow = (await loadFlow(source(times))).compiled;
      const { engine: built } = ownerEngine(flow, { inputs: { n: 10n, tag: "red" }, growBuilder: buildGrow });
      const original = copyGraph(built.state.flow);
      await built.run();
      const trace = built.state.trace!;
      const expectedOutput: JsonObject = { n: BigInt(10 - times), tag: "red" };
      if (shape === "chat-shaped") expectedOutput.seen = "red";
      expect(trace.terminal!.output).toEqual(expectedOutput);
      if (shape === "nested-call" && times >= 3) {
        const ids = [...built.state.flow.nodes.keys()];
        expect(ids.some((id) => id.startsWith("loop#0/"))).toBe(false);
        expect(ids.some((id) => id.startsWith(`loop#${times - 1}/`))).toBe(true);
      }

      const [root, results] = await computeBackward(trace, feedback);

      validateAttributedBackward(trace, feedback, root, results);
      expect(root.inputFeedback).toEqual(expected);
      const replay = replayRecords(new RunState(original), built.log.records(), { growBuilder: buildGrow });
      expect(await computeBackward(replay.trace!, feedback)).toEqual([root, results]);
    } finally {
      backward.mockRestore();
    }
  },
);
