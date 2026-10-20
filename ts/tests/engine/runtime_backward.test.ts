/**
 * The accepted reverse walk: a port of `tests/engine/test_runtime_backward.py`. Ported, after
 * rewrite: the `validate_attributed_backward` half of
 * `test_explicitly_synthetic_source_free_failure_keeps_direct_root_text`, with the root result
 * the test pins (`BackwardResult(self_feedback=("original text",))`) and the empty results built
 * by hand instead of computed.
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
 * rule; the getters on empty, unfinished and finished histories; the interrupted outcome an
 * unfinished attempt derives, with `prefixLearning`'s statuses and failure order;
 * `proposalBatches`' table order; and `sameRuntimeBackwardState`, true for separately built equal
 * states and false for a different memory order, a different last record and a `null` versus
 * non-`null` `ready`.
 *
 * TS-only, the reducer over `chainTrace()` with memory key `notes` and ids from `memoryCommitId`:
 * a full attempt Started → Ready → Requested → Succeeded → Finished, then a second start; the
 * accepted Finished of each branch (interrupted before and after Ready, commit failure, learner
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
 * Later (T4-6, with the engine): `test_terminal_trace_keeps_actual_root_writer_until_quiescence`.
 *
 * Later (T4-7, with `computeBackward`): the `compute_backward` half of
 * `test_explicitly_synthetic_source_free_failure_keeps_direct_root_text` and the file's other 24
 * tests, from `test_diamond_calls_shared_producer_once_with_two_ordered_messages` to
 * `test_loop_passthrough_field_routes_feedback_to_its_seed` (among them
 * `test_terminal_trace_latches_causal_route_and_later_failure_wins`, which also needs the engine
 * (T4-6)).
 */

import { expect, test } from "bun:test";

import { END_ID, START_ID } from "../../src/compile/model.ts";
import { Failure } from "../../src/diagnostics.ts";
import { type FeedbackValue, RouteContext, SourceRef, sourceKey, ValueRoute } from "../../src/expr/lineage.ts";
import {
  DeclSite,
  LearnContext,
  LearningFailure,
  LearningResult,
  MemoryEpisode,
  MemoryEpisodeStatus,
  MemoryMutation,
  MemoryOperationFailure,
  MemoryScope,
  MemorySpec,
  MemoryStack,
  PolicyCommitOutcome,
} from "../../src/memory/contracts.ts";
import { BackwardContext, BackwardResult, Node, NodeKind } from "../../src/nodes/base.ts";
import { ExecutionPosition, ExecutionScope, Parameter } from "../../src/nodes/contracts.ts";
import { StartNode } from "../../src/nodes/start/node.ts";
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
  encodePool,
  MemoryCommitFailedRecord,
  MemoryCommitRequestedRecord,
  MemoryCommitSucceededRecord,
  NodeStartedRecord,
  NodeSucceededRecord,
  RecordEnvelope,
  routePayload,
  type RunRecord,
  RunStartedRecord,
  SCHEMA_VERSION,
} from "../../src/runtime/log.ts";
import {
  commitPrepared,
  type PreparedTransition,
  prepareNodeRouted,
  prepareNodeStarted,
  prepareNodeSucceeded,
} from "../../src/runtime/state.ts";
import { type PrepareTraceOptions, prepareTrace, RetainedTrace, TraceTerminal } from "../../src/runtime/trace.ts";
import { VariablePool } from "../../src/typesys/index.ts";
import { logJson } from "../../src/typesys/json.ts";
import {
  commitStartSuccess,
  forwardCapture,
  graph,
  learnableStarted,
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

test("an explicitly synthetic source-free failure keeps the direct root text", () => {
  const trace = syntheticTrace("synthetic");
  const root = new BackwardResult({ selfFeedback: ["original text"] });
  expect(() => validateAttributedBackward(trace, "original text", root, [])).not.toThrow();
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
    prepared = prepareNodeSucceeded(state, nodeId, value, {
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
    prepareNodeSucceeded(state, "unrelated", "done", {
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

/** The chain's occurrences: START (seq 1), `p` (seq 3) and END (seq 5). */
const S = new SourceRef(RUN, 1);
const P = new SourceRef(RUN, 3);
const E = new SourceRef(RUN, 5);
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
  const pool = new VariablePool(START_ID);
  pool.set(START_ID, { x: 1n });
  const run = new RunStartedRecord({
    flowFingerprint: "a".repeat(64),
    initialPool: encodePool(pool),
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
      ["v", sourceRoute(1, ["x"])],
      ["w", sourceRoute(0, ["x"])],
    ]),
    {
      inputNames: ["v", "w"],
      variableNames: ["tone"],
      nodeData: pPaths === null ? null : { input_output_paths: pPaths },
      scope: pScope,
    },
  );
  const endCapture = new ForwardCapture({ value: pOutput }, ValueRoute.record([["value", sourceRoute(3)]]), {
    inputNames: ["value"],
  });
  const nodes = { start: new TraceNode(START_ID), p: new TraceNode("p"), end: new TraceNode(END_ID) };
  nodes.p.memoryScope = new MemoryScope(["p"], { memories: ["notes"] });
  nodes.p.hidden = new Set(["w"]);
  nodes.p.executionScope = pScope;
  const envelopes = [
    at(0, run),
    at(1, new NodeStartedRecord(START_ID)),
    at(2, new NodeSucceededRecord(START_ID, START_ID, wire({ x: 1n }), { forwardCapture: startCapture })),
    at(3, new NodeStartedRecord("p")),
    at(4, new NodeSucceededRecord("p", "p", wire(pOutput), { forwardCapture: pCapture })),
    at(5, new NodeStartedRecord(END_ID)),
    at(6, new NodeSucceededRecord(END_ID, END_ID, wire(pOutput), { forwardCapture: endCapture })),
  ];
  let trace = step(null, envelopes[0]!, null, { rootEndId: END_ID, rootInputNames: ["x"] });
  const owners = [null, nodes.start, null, nodes.p, null, nodes.end];
  for (const [index, owner] of owners.entries()) {
    const options = index === 5 ? { terminalSucceeded: true, terminalOutputJson: logJson(wire(pOutput)) } : {};
    trace = step(trace, envelopes[index + 1]!, owner, options);
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
  expect(envelopes).toHaveLength(7);
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
      trace.terminal = new TraceTerminal(true, new SourceRef(RUN, 9), logJson(wire(2n)), null);
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
    sender: { run_id: RUN, execution_seq: 3n, path: [] },
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
    sender: { run_id: RUN, execution_seq: 5n, path: [] },
    input_name: "value",
    sender_message_index: 0n,
    contribution_index: 0n,
    parents: [0n],
    contexts: [
      {
        expression: "${source_3}",
        result: 2n,
        sources: [{ source: { run_id: RUN, execution_seq: 3n, path: [] }, value: 2n }],
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
  const missing = new SourceRef(RUN, 9);
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

/** The failure of an interrupted attempt. */
const INTERRUPTED = new BackwardFailure(
  "backward",
  "backward.interrupted",
  "backward attempt interrupted before completion",
);

/** A Finished of attempt `id` that interrupted before Ready. */
function interruptedFinish(id = 1): BackwardFinishedRecord {
  return new BackwardFinishedRecord(id, new BackwardOutcome("failed", { failure: INTERRUPTED }));
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
    () => new BackwardAttempt(started(), { finished: interruptedFinish(2) }),
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

test("an unfinished attempt reads its records and derives the interrupted outcome with prefix learning", () => {
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
  expect(state.outcome).toEqual(new BackwardOutcome("failed", { feedback, failure: INTERRUPTED, learning }));
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

test("an interrupted attempt before Ready has no feedback", () => {
  const state = new RuntimeBackwardState(["a"], [new BackwardAttempt(started())]);
  const outcome = state.outcome!;
  expect(outcome.feedback).toBeNull();
  expect(outcome.failure!.code).toBe("backward.interrupted");
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
    finished: interruptedFinish(1),
  });
  const second = new BackwardAttempt(started(2), { finished: interruptedFinish(2) });
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

/** A finished attempt 1 and an unfinished attempt 2 with `last` (by default `started(2)`) and `options`. */
function twoAttempts(
  memories: string[] = ["a", "b"],
  last: BackwardStartedRecord = started(2),
  options: { ready?: BackwardReadyRecord | null } = { ready: ready(2) },
): RuntimeBackwardState {
  const first = new BackwardAttempt(started(1), { finished: interruptedFinish(1) });
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

/** The interrupted Finished of attempt 1 before Ready. */
const chainInterrupted = () => failedFinish(INTERRUPTED, { learning: INCOMPLETE });

/** The options of a `BackwardFailure`. */
type FailureOptions = ConstructorParameters<typeof BackwardFailure>[3];

/** A learner failure for `notes` from `cause`. */
function learnerFailure(cause = LEARN_CAUSE, options: FailureOptions = {}): BackwardFailure {
  return new BackwardFailure("learn", cause.code, cause.message, { memory: "notes", cause, ...options });
}

/** A context failure attributed to `p`'s occurrence. */
function contextFailure(options: FailureOptions = {}): BackwardFailure {
  return learnerFailure(CONTEXT_CAUSE, { runId: RUN, nodeId: "p", nodeStartedSeq: 3, ...options });
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
  return new BackwardFailure(stage, code, "Broke.", { runId: RUN, nodeId: "p", nodeStartedSeq: 3, ...options });
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
  ["an interrupted attempt before Ready", () => [started(), chainInterrupted()]],
  [
    "an interrupted attempt after Ready, with its results",
    () => [
      started(),
      chainReady(),
      failedFinish(INTERRUPTED, { feedback: CHAIN_ROOT, learning: INCOMPLETE }, chainResults()),
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
    (trace) => reduced([started(), chainInterrupted(), chainReady()], trace),
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
      const finish = failedFinish(INTERRUPTED, { feedback: new BackwardResult(), learning: INCOMPLETE });
      return reduced([started(), chainReady(), finish], trace);
    },
    RangeError,
    "Finished feedback must match accepted Ready",
  ],
  [
    "Finished attribution that differs from Ready",
    (trace) => {
      const unreachedResults = [S, P, E].map(unreached);
      const finish = failedFinish(INTERRUPTED, { feedback: CHAIN_ROOT, learning: INCOMPLETE }, unreachedResults);
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
    "an interrupted Finished with other learning",
    (trace) => reduced([started(), failedFinish(INTERRUPTED)], trace),
    RangeError,
    "interrupted Finished must exactly project the accepted prefix",
  ],
  [
    "an interrupted Finished before Ready with results",
    (trace) => reduced([started(), failedFinish(INTERRUPTED, { learning: INCOMPLETE }, chainResults())], trace),
    RangeError,
    "interrupted Finished must exactly project the accepted prefix",
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
      const failure = learnerFailure(LEARN_CAUSE, { runId: RUN, nodeId: "p", nodeStartedSeq: 3 });
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
      const failure = contextFailure({ nodeId: START_ID, nodeStartedSeq: 1 });
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

const EMPTY_SPEC = new MemorySpec("empty", "empty", {}, [new DeclSite([], ["def", "unused"])], false);
const NOTES_SPEC = new MemorySpec("notes", "notes", {}, [new DeclSite([], ["def", "p"])], false);
const OUTER_SPEC = new MemorySpec("outer", "outer", {}, [new DeclSite([], ["root"])], false);
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
  expression: "${source_3}",
  result: 2n,
  sources: [{ source: sourceJson(3), value: 2n }],
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
const P_COORDINATES = { execution_seq: 3n, owner_static_path: null, occurrence_path: null, execution_scope: null };

test("owner contexts project the producer's episode and feedback for its memory only", () => {
  const { trace } = chainTrace();
  const contexts = ownerContexts(trace);
  expect(contexts.map((context) => context.memory)).toEqual(["empty", "notes", "outer"]);
  const [empty, notes, outer] = contexts as LearnContext[];
  expect(empty).toEqual(new LearnContext(RUN, "empty", []));
  expect(notes!.runId).toBe(RUN);
  expect(notes!.episodes).toEqual([
    new MemoryEpisode(RUN, [], ["p"], MemoryEpisodeStatus.SUCCEEDED, { v: 1n }, { output: 2n, executionSeq: 3 }),
  ]);
  expect(notes!.feedback).toEqual([
    {
      ...P_COORDINATES,
      channel: "output",
      target_name: null,
      value: "Too low.",
      contribution_context: {
        message_index: 0n,
        sender: sourceJson(5),
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
          { source: sourceJson(1), value: { x: "Raise v." }, contribution_index: 0n, contexts: [fieldContext(1)] },
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
    execution_seq: 3n,
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
