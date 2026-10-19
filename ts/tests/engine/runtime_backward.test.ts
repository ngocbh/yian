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
 * unfinished attempt derives, with `prefixLearning`'s statuses and failure order; and
 * `proposalBatches`' table order.
 *
 * TS-only, the reducer over `chainTrace()` with memory key `notes` and ids from `memoryCommitId`:
 * a full attempt Started → Ready → Requested → Succeeded → Finished, then a second start; the
 * accepted Finished of each branch (interrupted before and after Ready, commit failure, learner
 * failure with and without an attributed context failure, computation failure); and one refusal
 * per reducer, Finished and context-owner rule, the last record class refused through a forged
 * record of another class.
 *
 * Later (T4-6, with the engine): `test_terminal_trace_keeps_actual_root_writer_until_quiescence`,
 * `test_root_inventory_is_retained_without_running_capture_or_defaults`,
 * `test_terminal_trace_append_rejection_leaves_all_indexes_unchanged` and
 * `test_causal_route_trace_latch_is_installed_only_after_append`.
 *
 * Later (T4-7, with `computeBackward`): the `compute_backward` half of
 * `test_explicitly_synthetic_source_free_failure_keeps_direct_root_text` and the file's other 24
 * tests, from `test_diamond_calls_shared_producer_once_with_two_ordered_messages` to
 * `test_loop_passthrough_field_routes_feedback_to_its_seed`.
 */

import { expect, test } from "bun:test";

import { END_ID, START_ID } from "../../src/compile/model.ts";
import { Failure } from "../../src/diagnostics.ts";
import { type FeedbackValue, RouteContext, SourceRef, sourceKey, ValueRoute } from "../../src/expr/lineage.ts";
import {
  LearningFailure,
  LearningResult,
  MemoryMutation,
  MemoryOperationFailure,
  MemoryScope,
  PolicyCommitOutcome,
} from "../../src/memory/contracts.ts";
import { BackwardContext, BackwardResult, Node, NodeKind } from "../../src/nodes/base.ts";
import { ExecutionScope } from "../../src/nodes/contracts.ts";
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
  prefixLearning,
  proposalBatches,
  reduceBackwardRecord,
  RuntimeBackwardState,
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
import { type PrepareTraceOptions, prepareTrace, RetainedTrace, TraceTerminal } from "../../src/runtime/trace.ts";
import { VariablePool } from "../../src/typesys/index.ts";
import { logJson } from "../../src/typesys/json.ts";

/** The two error classes the refusals throw. */
type ErrorClass = typeof TypeError | typeof RangeError;

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

test("an explicitly synthetic source-free failure keeps the direct root text", () => {
  const trace = new RetainedTrace("synthetic", START_ID, END_ID);
  trace.terminal = new TraceTerminal(
    false,
    null,
    null,
    new Failure({ code: "graph.terminal_unreachable", message: "no executing producer", nodeId: null }),
  );
  const root = new BackwardResult({ selfFeedback: ["original text"] });
  expect(() => validateAttributedBackward(trace, "original text", root, [])).not.toThrow();
  expectError(
    () => validateAttributedBackward(trace, "original text", new BackwardResult(), []),
    RangeError,
    "backward root result must match the accepted START and source-zero projection",
  );
});

// TS-only: the chain fixture.

const RUN = "run";

/** A CODE node with a settable memory scope, recording its backward calls. */
class TraceNode extends Node {
  readonly kind = NodeKind.CODE;
  /** Each backward call's context and feedback, in call order. */
  readonly calls: [BackwardContext, readonly FeedbackValue[]][] = [];

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
}

/**
 * A terminal trace of a learnable run, driven through `prepareTrace` and `apply`: RunStarted (START
 * `__start__` writes `{x: 1n}`, root END `end`, root input names `["x"]`); START, whose capture
 * routes `x` to source 0 path `["x"]`; `p`, routing `v` from START's output `x` and `w` straight to
 * source 0 path `["x"]`, with memory scope `["p"]` and key `notes`; END, routing `value` from `p`,
 * with the successful terminal. Returns the trace, the nodes and the envelopes.
 */
function chainTrace({ startPaths = { x: ["x"] }, pOutput = 2n, pPaths = null }: ChainOptions = {}) {
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
    { inputNames: ["v", "w"], nodeData: pPaths === null ? null : { input_output_paths: pPaths } },
  );
  const endCapture = new ForwardCapture({ value: pOutput }, ValueRoute.record([["value", sourceRoute(3)]]), {
    inputNames: ["value"],
  });
  const nodes = { start: new TraceNode(START_ID), p: new TraceNode("p"), end: new TraceNode(END_ID) };
  nodes.p.memoryScope = new MemoryScope(["p"], { memories: ["notes"] });
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
