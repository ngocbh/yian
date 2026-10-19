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
 * (inventory order, feedback, attribution, invoked, a local contract violation, the root, feedback
 * to an unavailable occurrence, a non-terminal trace); empty feedback seeds nothing; a node-data
 * path absent from the accepted output; an int-headed path's output context and its materialized
 * form; `materializeContext`'s shape and `bigint` integers; `ContextCache.lookup` looking a source
 * up once and not caching a failure; `takeMessages` order.
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
import { MemoryScope } from "../../src/memory/contracts.ts";
import { BackwardContext, BackwardResult, Node, NodeKind } from "../../src/nodes/base.ts";
import { AttributedBackwardResult, BackwardAttribution, ForwardCapture } from "../../src/runtime/contracts.ts";
import {
  BackwardMessage,
  ContextCache,
  contextualTransfers,
  materializeContext,
  seedBackward,
  takeMessages,
  validateAttributedBackward,
} from "../../src/runtime/learning.ts";
import {
  canonicalTyped,
  encodePool,
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
  expect(() => validateAttributedBackward(trace, "original text", new BackwardResult(), [])).toThrow(
    new RangeError("backward root result must match the accepted START and source-zero projection"),
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
  expect(() => validateAttributedBackward(trace, "Too low.", startOnly, results)).toThrow(
    new RangeError("backward root result must match the accepted START and source-zero projection"),
  );
});

const TRANSFER_MISMATCH = "backward received feedback and attribution must match exact transfers";

test.each<[string, (results: AttributedBackwardResult[], trace: RetainedTrace) => unknown, ErrorConstructor, string]>([
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
  expect(() => call(chainResults(), trace)).toThrow(new cls(message));
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
  expect(() => seedBackward(trace, { b: "Wrong." })).toThrow(new RangeError("feedback selects unknown field 'b'"));
});

test("a node-data path absent from the accepted output fails the structural walk", () => {
  const { trace } = chainTrace({ startPaths: { x: ["y"] } });
  const root = new BackwardResult({ inputFeedback: { x: ["Raise v."] } });
  expect(() => validateAttributedBackward(trace, "Too low.", root, chainResults())).toThrow(
    new RangeError("structural output path is absent from the accepted output"),
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
  const materialized = materializeContext(trace, message, 0, cache).contexts as Record<string, unknown>[];
  const detail = materialized[pMessages[0]!.context.length]!;
  expect(detail.positions).toEqual([[0n]]);
  expect(detail.operand).toEqual([0n]);
  expect(detail.result).toEqual([1n]);
});

test("materializeContext describes a message with bigint integers", () => {
  const { trace } = chainTrace();
  const cache = new ContextCache(trace);
  const message = new BackwardMessage("Raise v.", new BackwardAttribution(P, "v", 0, 0, [0]), [0, 0], []);
  expect(materializeContext(trace, message, 1, cache)).toEqual({
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
  const materialized = materializeContext(trace, message, 0, cache);
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
  const again = materializeContext(trace, message, 0, cache);
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
  const [detail] = materializeContext(trace, message, 0, cache).contexts as Record<string, unknown>[];
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
  const [projected] = materializeContext(trace, new BackwardMessage("x", ROOT_SEED, [0], [projection]), 0, cache)
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
  expect(() => cache.lookup(missing)).toThrow(RangeError);
  expect(() => cache.lookup(missing)).toThrow(RangeError);
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
