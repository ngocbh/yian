/**
 * Builders shared by the run-state tests (Python's `_fakes.py`, `_graph_builder.py` and the
 * helpers `test_run_state_initialization.py` and `test_run_state_growth.py` lend to other files).
 *
 * A `*.test.ts` imported by another test file registers its tests twice, so a helper that two or
 * more run-state test files use lives here.
 */

import { expect } from "bun:test";

import { applyExecutionLayout, resolveExecutionLayout } from "../../src/compile/execution_layout.ts";
import { CompiledFlow, Edge, END_ID, Flow, FlowOutput, START_ID } from "../../src/compile/model.ts";
import { Failure, type OriginRef } from "../../src/diagnostics.ts";
import { ValueRoute } from "../../src/expr/lineage.ts";
import { bindingCoSkips, exprRefsOf } from "../../src/expr/template.ts";
import { Node, type NodeCaps, NodeKind, type NodeResult, Output, Route } from "../../src/nodes/base.ts";
import { CallNode } from "../../src/nodes/call/node.ts";
import { comparePaths, Parameter } from "../../src/nodes/contracts.ts";
import {
  DeclSite,
  type MemoryScope,
  MemorySnapshotEnvelope,
  MemorySpec,
  MemoryStack,
} from "../../src/memory/index.ts";
import { EndNode } from "../../src/nodes/end/node.ts";
import { StartNode } from "../../src/nodes/start/node.ts";
import { ForwardCapture } from "../../src/runtime/contracts.ts";
import { FlowEngine, type FlowEngineOptions } from "../../src/runtime/engine.ts";
import { type EvalNodeOptions, evalNode, type WorkerOutcome } from "../../src/runtime/eval_node.ts";
import { type RunEvent, StreamChunk } from "../../src/events.ts";
import {
  decodePool,
  encodePool,
  MemoryRunLog,
  type RecordEnvelope,
  type RunLog,
  type RunRecord,
  RunStartedRecord,
} from "../../src/runtime/log.ts";
import type { RetainedTrace } from "../../src/runtime/trace.ts";
import {
  commitPrepared,
  type GrowBuilder,
  prepareNodeStarted,
  prepareNodeSucceeded,
  prepareRecord,
  prepareRunStarted,
  type RunStartData,
  RunState,
} from "../../src/runtime/state.ts";
import { VariablePool } from "../../src/typesys/pool.ts";

/** The run id the run-state tests record. */
export const RUN_ID = "run-123";
/** The flow fingerprint the run-state tests record. */
export const FINGERPRINT = "a".repeat(64);

/**
 * A deterministic node: `fn(inputs)` is its single value. `reads` (`{param: source}`) declares its
 * inputs: it stamps `params`, and `graph` collects it into the flow's wiring.
 */
export class FuncNode extends Node {
  readonly kind = NodeKind.CODE;
  /** The body, from the bound input record to the value. */
  readonly fn: (inputs: Record<string, unknown>) => unknown;
  /** The declared input sources by parameter name. */
  readonly reads: Readonly<Record<string, unknown>>;

  constructor(
    nodeId: string,
    fn: (inputs: Record<string, unknown>) => unknown,
    options: { reads?: Readonly<Record<string, unknown>> } & ConstructorParameters<typeof Node>[1] = {},
  ) {
    const { reads = {}, ...rest } = options;
    super(nodeId, rest);
    this.fn = fn;
    this.reads = { ...reads };
    this.params = Object.keys(reads).map((name) => new Parameter(name));
  }

  override forward(inputs: Record<string, unknown>): NodeResult {
    return new Output(this.fn(inputs));
  }
}

/** A `FuncNode` that appends its id to `log` when it runs and returns `{output}`. */
export class RecordNode extends FuncNode {
  constructor(
    nodeId: string,
    log: string[],
    output: unknown = "",
    options: ConstructorParameters<typeof FuncNode>[2] = {},
  ) {
    super(
      nodeId,
      () => {
        log.push(nodeId);
        return { output };
      },
      options,
    );
  }
}

/** Throws a plain `Error` with `message` from its body. */
export class FailNode extends Node {
  readonly kind = NodeKind.CODE;
  /** The thrown error's message. */
  readonly message: string;

  constructor(nodeId: string, message = "boom", options: ConstructorParameters<typeof Node>[1] = {}) {
    super(nodeId, options);
    this.message = message;
  }

  override forward(): NodeResult {
    throw new Error(this.message);
  }
}

/** A CASE fake that routes to one handle. */
export class BranchNode extends Node {
  readonly kind = NodeKind.CASE;
  /** The chosen handle (which downstream edge to take). */
  readonly handle: string;

  constructor(nodeId: string, handle: string, options: ConstructorParameters<typeof Node>[1] = {}) {
    super(nodeId, options);
    this.handle = handle;
  }

  override forward(): NodeResult {
    return new Route(this.handle);
  }
}

/**
 * Streams its chunks through `caps.emitStream` under key `output`, then returns
 * `{output: <the chunks joined>}` (the AGENT streaming shape).
 */
export class StreamNode extends Node {
  readonly kind = NodeKind.AGENT;
  override readonly needsLiveEvents = true;
  /** The chunks to stream, in order. */
  readonly chunks: readonly string[];

  constructor(nodeId: string, chunks: readonly string[], options: ConstructorParameters<typeof Node>[1] = {}) {
    super(nodeId, options);
    this.chunks = [...chunks];
  }

  override forward(_inputs: Record<string, unknown>, caps: NodeCaps): NodeResult {
    let text = "";
    for (const chunk of this.chunks) {
      text += chunk;
      caps.emitStream?.(new StreamChunk({ nodeId: this.id, key: "output", chunk }));
    }
    return new Output({ output: text });
  }
}

/** Return a promise that rejects with `signal`'s reason once it aborts, at once if it has. */
export function aborted(signal: AbortSignal): Promise<never> {
  return new Promise((_resolve, reject) => {
    if (signal.aborted) reject(signal.reason);
    else signal.addEventListener("abort", () => reject(signal.reason), { once: true });
  });
}

/**
 * Blocks on `gate`, a promise the test settles, standing in for a node that waits on a human: its
 * value is what `gate` resolves to, and when its signal aborts first it rejects with the signal's
 * reason. `entered` resolves once it blocks; `released` turns true once it stops blocking.
 */
export class BlockingNode extends Node {
  readonly kind = NodeKind.CODE;
  /** The promise the node waits on. */
  readonly gate: Promise<unknown>;
  /** Resolves once the node blocks. */
  readonly entered: Promise<void>;
  /** Whether the node has stopped blocking. */
  released = false;
  readonly #enter: () => void;

  constructor(nodeId: string, gate: Promise<unknown>, options: ConstructorParameters<typeof Node>[1] = {}) {
    super(nodeId, options);
    this.gate = gate;
    const { promise, resolve } = Promise.withResolvers<void>();
    this.entered = promise;
    this.#enter = resolve;
  }

  override async forward(_inputs: Record<string, unknown>, caps: NodeCaps): Promise<NodeResult> {
    this.#enter();
    try {
      return new Output(await Promise.race([this.gate, aborted(caps.signal!)]));
    } finally {
      this.released = true;
    }
  }
}

/** Return `START -> work -> END`, with a real START and an output-less END. */
export function flow(): CompiledFlow {
  const nodes = new Map<string, Node>([
    [START_ID, new StartNode(START_ID)],
    ["work", new FuncNode("work", (inputs) => inputs)],
    [END_ID, EndNode.value(END_ID, { outputName: null })],
  ]);
  return CompiledFlow.fromParts(nodes, [
    new Edge("start-work", START_ID, "work"),
    new Edge("work-end", "work", END_ID),
  ]);
}

/** Return a complete graph whose START success makes `first` and `second` ready in order. */
export function fanoutFlow(): CompiledFlow {
  const nodes = new Map<string, Node>([
    [START_ID, new StartNode(START_ID)],
    ["first", new FuncNode("first", (inputs) => inputs)],
    ["second", new FuncNode("second", (inputs) => inputs)],
    [END_ID, EndNode.value(END_ID, { outputName: null })],
  ]);
  return CompiledFlow.fromParts(nodes, [
    new Edge("start-first", START_ID, "first"),
    new Edge("start-second", START_ID, "second"),
  ]);
}

/** Return a route whose unselected `no` sibling has a two-node skip chain and `yes` leads to `chosen`. */
export function routeFlow(): CompiledFlow {
  const nodes = new Map<string, Node>([
    [START_ID, new StartNode(START_ID)],
    ...["router", "chosen", "dead-parent", "dead-child"].map(
      (id) => [id, new FuncNode(id, (inputs) => inputs)] as const,
    ),
    [END_ID, EndNode.value(END_ID, { outputName: null })],
  ]);
  return CompiledFlow.fromParts(nodes, [
    new Edge("start-router", START_ID, "router"),
    new Edge("route-dead", "router", "dead-parent", { sourceHandle: "no" }),
    new Edge("route-chosen", "router", "chosen", { sourceHandle: "yes" }),
    new Edge("dead-child", "dead-parent", "dead-child"),
  ]);
}

/**
 * Return a route whose selected `go` control into END still co-skips two of END's groups: `zeta`
 * (through `dead-z`) and then `alpha` (through `dead-a`), after an optional group. END is a record
 * of `zeta` and `alpha`. `routerOrigin` is the router's origin; `sibling` adds a START successor
 * `unrelated`.
 */
export function terminalSkipRouteFlow(
  options: { routerOrigin?: OriginRef | null; sibling?: boolean } = {},
): CompiledFlow {
  const nodes = new Map<string, Node>([
    [START_ID, new StartNode(START_ID)],
    ["router", new FuncNode("router", (inputs) => inputs, { origin: options.routerOrigin ?? null })],
    ...["dead-optional", "dead-z", "dead-a"].map((id) => [id, new FuncNode(id, (inputs) => inputs)] as const),
    [END_ID, EndNode.record(END_ID, { outputNames: ["zeta", "alpha"] })],
  ]);
  const edges = [
    new Edge("start-router", START_ID, "router"),
    new Edge("router-end", "router", END_ID, { sourceHandle: "go" }),
    new Edge("router-dead-optional", "router", "dead-optional", { sourceHandle: "skip-optional" }),
    new Edge("dead-optional-end", "dead-optional", END_ID, { inputGroup: "optional-first", optional: true }),
    new Edge("router-dead-z", "router", "dead-z", { sourceHandle: "skip-z" }),
    new Edge("dead-z-end", "dead-z", END_ID, { inputGroup: "zeta" }),
    new Edge("router-dead-a", "router", "dead-a", { sourceHandle: "skip-a" }),
    new Edge("dead-a-end", "dead-a", END_ID, { inputGroup: "alpha" }),
  ];
  if (options.sibling) {
    nodes.set("unrelated", new FuncNode("unrelated", (inputs) => inputs));
    edges.push(new Edge("start-unrelated", START_ID, "unrelated"));
  }
  return CompiledFlow.fromParts(nodes, edges);
}

/** The id of `callRedirectFlow`'s child END. */
export const CHILD_END = "call/__end__";

/**
 * Return a CALL-shaped graph: the router's `child` branch reaches the child END, which redirects
 * its output to the logical parent `call`, whose value END requires. With `fallback`, a START
 * successor `fallback` also feeds END's `result` group.
 */
export function callRedirectFlow(options: { fallback?: boolean } = {}): CompiledFlow {
  const childEnd = EndNode.value(CHILD_END, { outputName: null });
  childEnd.commitAs = "call";
  const nodes = new Map<string, Node>([
    [START_ID, new StartNode(START_ID)],
    ["router", new FuncNode("router", (inputs) => inputs)],
    [CHILD_END, childEnd],
    ["call", new CallNode("call", { flowId: "child" })],
    [END_ID, EndNode.value(END_ID, { outputName: null })],
  ]);
  const edges = [
    new Edge("start-router", START_ID, "router"),
    new Edge("router-child-end", "router", CHILD_END, { sourceHandle: "child" }),
    new Edge("call-end", "call", END_ID, { inputGroup: "result" }),
  ];
  if (options.fallback) {
    nodes.set("fallback", new FuncNode("fallback", (inputs) => inputs));
    edges.splice(1, 0, new Edge("start-fallback", START_ID, "fallback"));
    edges.push(new Edge("fallback-end", "fallback", END_ID, { inputGroup: "result" }));
  }
  return CompiledFlow.fromParts(nodes, edges);
}

/** Return the complete pool a fresh run of `flow` records under `runId`. */
export function pool(flow: CompiledFlow, runId: string = RUN_ID): VariablePool {
  const built = new VariablePool(flow.startId);
  built.set(flow.startId, { name: "Ada", count: 2n, aliases: ["A", "B"] });
  built.set("preseeded", { retained: true });
  built.addSystem("run_id", runId);
  built.addSystem("today", "2026-08-13");
  built.addSystem("now", "2026-08-13T12:34:56+00:00");
  built.addSystem("tenant", "tests");
  return built;
}

/** Return representative fresh-run data for `flow`. */
export function startData(
  flow: CompiledFlow,
  options: { runId?: string; pool?: VariablePool; numWorkers?: number } = {},
): RunStartData {
  const { runId = RUN_ID, numWorkers = 3 } = options;
  return {
    runId,
    flowFingerprint: FINGERPRINT,
    initialPool: options.pool ?? pool(flow, runId),
    startWiringOverride: { name: "${input.name}", metadata: { labels: ["one", "two"] } },
    effectiveLlmByStaticPath: { agent: { provider: "openai", model: "gpt-test", options: { temperature: 0.2 } } },
    numWorkers,
  };
}

/** The flow wiring the nodes declare through `FuncNode.reads`; another node declares none. */
function deriveWiring(nodes: ReadonlyMap<string, Node>): Record<string, Readonly<Record<string, unknown>>> {
  return Object.fromEntries([...nodes].map(([id, node]) => [id, node instanceof FuncNode ? node.reads : {}]));
}

/** Return a flow holding only `node`, wired from `wiring` (by default its `FuncNode.reads`). */
export function nodeFlow(
  node: Node,
  wiring: Readonly<Record<string, unknown>> = node instanceof FuncNode ? node.reads : {},
): CompiledFlow {
  return CompiledFlow.fromParts(new Map([[node.id, node]]), [], {
    wiring: new Map([[node.id, new Map(Object.entries(wiring))]]),
  });
}

/**
 * Evaluate `node` through `evalNode` and return its raw outcome (Python's `_fakes.drive`); the
 * other options go to `evalNode`.
 *
 * @param options.pool The pool the sources read; a fresh one by default.
 * @param options.flow The flow whose wiring holds the sources; `nodeFlow(node)` by default.
 */
export async function drive(
  node: Node,
  options: { pool?: VariablePool; flow?: CompiledFlow | null } & EvalNodeOptions = {},
): Promise<WorkerOutcome> {
  const { pool = new VariablePool(), flow = nodeFlow(node), ...rest } = options;
  return (await evalNode(node, flow, pool, rest)).outcome;
}

/** Return the value `promise` rejects with, or fail when it resolves. */
export async function rejected(promise: unknown): Promise<unknown> {
  try {
    await promise;
  } catch (error) {
    return error;
  }
  throw new Error("expected a rejection");
}

/** The text references of `source`, or none for a literal. */
function refsOf(source: unknown): string[] {
  return typeof source === "string" ? exprRefsOf(source) : [];
}

/** The producer node id of one reference: START for `input.<key>`, `<id>` for `<id>.output...`. */
function refProducer(path: string): string | null {
  const parts = path.split(".");
  if (parts.length >= 2 && parts[0] === "input") return START_ID;
  if (parts.length >= 2 && parts[1] === "output" && parts[0] !== "system" && parts[0] !== "item") return parts[0]!;
  return null;
}

/**
 * Build a compiled flow from `nodes` and `[from, to, handle?]` edges, adding a real START (with
 * one parameter per `${input.X}` the wiring reads) and END. END reads the declared `outputs` (one
 * as a value, more as a record) through edges minted from their bindings, and the manual edges
 * into END are dropped; with no outputs END returns `null`. An END with no incoming edge gets one
 * from START. `wiring` defaults to the `FuncNode.reads` of the nodes.
 */
export function graph(
  nodes: readonly Node[],
  rawEdges: readonly (readonly [string, string, string?])[],
  options: {
    outputs?: readonly FlowOutput[];
    wiring?: Readonly<Record<string, Readonly<Record<string, unknown>>>>;
  } = {},
): CompiledFlow {
  const nodeMap = new Map(nodes.map((node) => [node.id, node]));
  const sources = options.wiring ?? deriveWiring(nodeMap);
  if (!nodeMap.has(START_ID)) {
    const names = new Set<string>();
    for (const nodeWiring of Object.values(sources)) {
      for (const source of Object.values(nodeWiring)) {
        for (const ref of refsOf(source)) {
          const name = ref.startsWith("input.") ? ref.slice("input.".length).split(".")[0]! : "";
          if (name) names.add(name);
        }
      }
    }
    nodeMap.set(START_ID, new StartNode(START_ID, { params: [...names].map((name) => new Parameter(name)) }));
  }
  const outputs = options.outputs ?? [];
  if (!nodeMap.has(END_ID)) {
    nodeMap.set(
      END_ID,
      outputs.length <= 1
        ? EndNode.value(END_ID, { outputName: outputs[0]?.name ?? null })
        : EndNode.record(END_ID, { outputNames: outputs.map((output) => output.name) }),
    );
  }
  let manual = rawEdges;
  const endEdges: Edge[] = [];
  if (outputs.length > 0) {
    manual = rawEdges.filter(([, to]) => to !== END_ID);
    const counts = new Map<string, number>();
    for (const output of outputs) {
      const optional = !bindingCoSkips(output.binding.source);
      for (const ref of refsOf(output.binding.source)) {
        const producer = refProducer(ref);
        if (producer === null || !nodeMap.has(producer)) continue;
        const index = counts.get(producer) ?? 0;
        counts.set(producer, index + 1);
        const id = `${producer}->${END_ID}#${index}`;
        endEdges.push(new Edge(id, producer, END_ID, { inputGroup: output.name, optional }));
      }
    }
  }
  const edges = manual.map(([from, to, handle], i) => new Edge(`e${i}`, from, to, { sourceHandle: handle ?? null }));
  edges.push(...endEdges);
  if (!edges.some((edge) => edge.to === END_ID)) edges.push(new Edge(`${START_ID}->${END_ID}`, START_ID, END_ID));
  const finalSources = options.wiring ?? deriveWiring(nodeMap);
  const wiring = new Map(
    Object.entries(finalSources).map(([id, nodeWiring]) => [id, new Map(Object.entries(nodeWiring))]),
  );
  if (outputs.length > 0) {
    wiring.set(END_ID, new Map(outputs.map((output) => [output.name, output.binding])));
  }
  return CompiledFlow.fromParts(nodeMap, edges, { outputs, wiring });
}

/**
 * Return a learnable state of `flow` after its committed RunStarted, and its memory log; Python's
 * `_engine(flow, inputs, learnable=True, ...)` where a test only needs the started state.
 *
 * The flow gets its execution layout; the pool primes START with `inputs`, and START's wiring
 * reads each input as `${input.<name>}`. The run records no agent configuration.
 */
export function learnableStarted(
  flow: CompiledFlow,
  options: {
    inputs?: Readonly<Record<string, unknown>>;
    memories?: readonly MemorySpec[] | null;
    memoryStacks?: readonly MemoryStack[] | null;
    memorySnapshots?: readonly MemorySnapshotEnvelope[] | null;
  } = {},
): { state: RunState; log: MemoryRunLog } {
  const inputs = options.inputs ?? {};
  applyExecutionLayout(flow, resolveExecutionLayout(flow));
  const state = new RunState(flow);
  const initialPool = new VariablePool(flow.startId);
  initialPool.set(flow.startId, { ...inputs });
  initialPool.addSystem("run_id", RUN_ID);
  const log = new MemoryRunLog(RUN_ID);
  const prepared = prepareRunStarted(state, {
    runId: RUN_ID,
    flowFingerprint: FINGERPRINT,
    initialPool,
    startWiringOverride: Object.fromEntries(Object.keys(inputs).map((name) => [name, `\${input.${name}}`])),
    effectiveLlmByStaticPath: {},
    numWorkers: 0,
    learnable: true,
    memories: options.memories ?? null,
    memoryStacks: options.memoryStacks ?? null,
    memorySnapshots: options.memorySnapshots ?? null,
  });
  commitPrepared(log, state, prepared);
  return { state, log };
}

/** A growth builder for flows that never grow; a call fails the test. */
export const noGrowth: GrowBuilder = () => {
  throw new Error("this flow does not grow");
};

/**
 * Return a direct engine over a fresh in-memory authority for `flow` (Python's
 * `_graph_builder._engine`): `RunStarted` is prepared and committed on a `MemoryRunLog`, so START
 * is ready. The pool primes START with `inputs`, START's wiring reads each as
 * `${input.<name>}`, and the run records `effectiveLlmByStaticPath` (none by default) and the
 * memory configuration given. A learnable flow gets its execution layout first. The other options
 * go to `FlowEngine`.
 */
export function engine(
  flow: CompiledFlow,
  options: {
    inputs?: Readonly<Record<string, unknown>>;
    learnable?: boolean;
    effectiveLlmByStaticPath?: RunStartData["effectiveLlmByStaticPath"];
    memories?: readonly MemorySpec[] | null;
    memoryStacks?: readonly MemoryStack[] | null;
    memorySnapshots?: readonly MemorySnapshotEnvelope[] | null;
  } & Partial<FlowEngineOptions> = {},
): FlowEngine {
  const {
    inputs = {},
    learnable = false,
    numWorkers = 0,
    growBuilder = noGrowth,
    effectiveLlmByStaticPath = {},
    memories = null,
    memoryStacks = null,
    memorySnapshots = null,
    ...rest
  } = options;
  if (learnable) applyExecutionLayout(flow, resolveExecutionLayout(flow));
  const state = new RunState(flow);
  const initialPool = new VariablePool(flow.startId);
  initialPool.set(flow.startId, { ...inputs });
  initialPool.addSystem("run_id", RUN_ID);
  const log = new MemoryRunLog(RUN_ID);
  const prepared = prepareRunStarted(state, {
    runId: RUN_ID,
    flowFingerprint: FINGERPRINT,
    initialPool,
    startWiringOverride: Object.fromEntries(Object.keys(inputs).map((name) => [name, `\${input.${name}}`])),
    effectiveLlmByStaticPath,
    numWorkers,
    learnable,
    memories,
    memoryStacks,
    memorySnapshots,
  });
  commitPrepared(log, state, prepared);
  return new FlowEngine(state, log, { growBuilder, numWorkers, ...rest });
}

/**
 * Return every reducer-owned value of `state` in a form `toEqual` compares: the pool as its wire
 * object, maps and sets as entry arrays, JSON values as copies, and the flow and result by
 * reference.
 */
export function stateSnapshot(state: RunState): Record<string, unknown> {
  return {
    flow: state.flow,
    pool: encodePool(state.pool),
    nodeState: [...state.nodeState],
    edgeState: [...state.edgeState],
    ready: [...state.ready],
    inFlight: [...state.inFlight],
    acceptedStarts: [...state.acceptedStarts],
    result: state.result,
    depth: [...state.depth],
    growthOwner: [...state.growthOwner],
    deferredPrune: [...state.deferredPrune],
    wiringOverrides: structuredClone([...state.wiringOverrides]),
    runId: state.runId,
    flowFingerprint: state.flowFingerprint,
    nextSequence: state.nextSequence,
    numWorkers: state.numWorkers,
    input: structuredClone(state.input),
    effectiveLlmByStaticPath: structuredClone(state.effectiveLlmByStaticPath),
  };
}

/** The root containers whose identity a commit keeps. */
export function roots(state: RunState): unknown[] {
  return [
    state.pool,
    state.pool.store,
    state.nodeState,
    state.edgeState,
    state.ready,
    state.inFlight,
    state.acceptedStarts,
    state.interruptedAttempts,
    state.depth,
    state.wiringOverrides,
    state.input,
    state.effectiveLlmByStaticPath,
  ];
}

/** Every reducer coordinate in a detached form, plus the root containers by reference. */
export interface Snapshot {
  values: Record<string, unknown>;
  roots: unknown[];
}

/**
 * Return `state`'s snapshot: `stateSnapshot` with the latch, start counts, steps,
 * interrupted-attempt counts and last stop sequence as well.
 */
export function snapshot(state: RunState): Snapshot {
  return {
    values: {
      ...stateSnapshot(state),
      terminalUnreachableFailure: state.terminalUnreachableFailure,
      nodeStarts: [...state.nodeStarts],
      nodeSteps: [...state.nodeSteps],
      interruptedAttempts: [...state.interruptedAttempts],
      lastStopSequence: state.lastStopSequence,
    },
    roots: roots(state),
  };
}

/** Assert `state` still has `before`'s values and root containers. */
export function expectUnchanged(state: RunState, before: Snapshot): void {
  const after = snapshot(state);
  expect(after.values).toEqual(before.values);
  after.roots.forEach((root, index) => expect(root).toBe(before.roots[index]));
}

/** Return `state`'s values without the flow, for comparing independent replays. */
export function reducerSnapshot(state: RunState): Record<string, unknown> {
  const { flow: _flow, ...values } = snapshot(state).values;
  return values;
}

/**
 * Return a state and log of `flow` after a replay-valid RunStarted (START primed with
 * `{items: ["seed"]}`, two workers) and START's dispatch.
 */
export function initializedAndStarted(
  flow: CompiledFlow,
  options: { learnable?: boolean } = {},
): { state: RunState; log: MemoryRunLog } {
  const initialPool = new VariablePool(flow.startId);
  initialPool.set(flow.startId, { items: ["seed"] });
  initialPool.addSystem("run_id", RUN_ID);
  const state = new RunState(flow);
  const log = new MemoryRunLog(RUN_ID);
  commitPrepared(
    log,
    state,
    prepareRunStarted(state, {
      runId: RUN_ID,
      flowFingerprint: FINGERPRINT,
      initialPool,
      startWiringOverride: {},
      effectiveLlmByStaticPath: {},
      numWorkers: 2,
      learnable: options.learnable ?? false,
    }),
  );
  commitPrepared(log, state, prepareNodeStarted(state, START_ID));
  return { state, log };
}

/**
 * Commit START's success (`output`, by default `{items: ["canonical"]}`), so its successors enter
 * the ready FIFO. A learnable state records an empty binding capture.
 */
export function commitStartSuccess(state: RunState, log: MemoryRunLog, output?: unknown): void {
  const start = state.flow.nodes.get(START_ID)!;
  commitPrepared(
    log,
    state,
    prepareNodeSucceeded(state, START_ID, output === undefined ? { items: ["canonical"] } : output, {
      forwardCapture: state.learnable
        ? new ForwardCapture({}, ValueRoute.record([]), { scope: start.executionScope })
        : null,
    }),
  );
}

/** Return one canonical node-execution failure of `nodeId`. */
export function nodeFailure(
  nodeId: string,
  message = "boom",
  options: { causeType?: string | null; traceback?: string | null } = {},
): Failure {
  return new Failure({ code: "node.execute", message, nodeId, ...options });
}

/**
 * Return a state and log of `flow` (by default `routeFlow()`) with START succeeded and its
 * `router` in flight.
 */
export function routeInFlight(flow: CompiledFlow = routeFlow()): { state: RunState; log: MemoryRunLog } {
  const { state, log } = initializedAndStarted(flow);
  commitStartSuccess(state, log);
  commitPrepared(log, state, prepareNodeStarted(state, "router"));
  return { state, log };
}

/**
 * Return a binding capture of `inputs` for `node`: each input a literal route, under the node's
 * execution scope (Python's `_graph_builder._forward_capture`).
 */
export function forwardCapture(node: Node, inputs: Readonly<Record<string, unknown>> | null): ForwardCapture {
  const route = ValueRoute.record(
    Object.entries(inputs ?? {}).map(([name, value]) => [name, ValueRoute.literal(value)] as const),
  );
  return new ForwardCapture(inputs, route, { scope: node.executionScope });
}

/** The root flow's declaration site. */
const ROOT_SITE = new DeclSite([], ["root"]);

/** Return the sorted key table for `keys`, all owned by the root flow. */
function table(keys: readonly string[]): MemorySpec[] {
  return [...keys].sort().map((key) => new MemorySpec(key, key, { limit: 3n }, [ROOT_SITE], false));
}

/** Return the recorded stacks of `scopes`, one per static path, sorted by locator. */
export function stacks(...scopes: readonly MemoryScope[]): MemoryStack[] {
  const byLocator = new Map(scopes.map((scope) => [JSON.stringify(scope.staticPath), scope] as const));
  return [...byLocator.values()]
    .sort((a, b) => comparePaths(a.staticPath, b.staticPath))
    .map((scope) => new MemoryStack(scope.staticPath, scope.memories));
}

/** The options of `newState` and `startTraceNode`. */
interface TraceNodeOptions {
  keys: readonly string[] | null;
  learnable: boolean;
  routeHandle?: string;
  stacks?: readonly MemoryStack[];
}

/**
 * Return a state of `START -> node -> END` after its committed RunStarted, with START ready, and
 * its log. `keys: null` disables memory; otherwise the table holds `keys`, each snapshot is
 * `{epoch: 4}`, and the stacks default to the node's own scope (none without one). `routeHandle`
 * is the handle of the edge into END.
 */
function newState(node: Node, options: TraceNodeOptions): { state: RunState; log: MemoryRunLog } {
  const built = graph([node], [
    [START_ID, node.id],
    [node.id, END_ID, options.routeHandle],
  ]);
  const state = new RunState(built);
  const initialPool = new VariablePool(built.startId);
  initialPool.set(built.startId, {});
  initialPool.addSystem("run_id", RUN_ID);
  let memories: MemorySpec[] | null = null;
  let snapshots: MemorySnapshotEnvelope[] | null = null;
  let recordedStacks: readonly MemoryStack[] | null = null;
  if (options.keys !== null) {
    memories = table(options.keys);
    snapshots = memories.map((spec) => new MemorySnapshotEnvelope(spec.key, { epoch: 4n }));
    recordedStacks = options.stacks ?? (node.memoryScope === null ? [] : stacks(node.memoryScope));
  }
  const log = new MemoryRunLog(RUN_ID);
  commitPrepared(
    log,
    state,
    prepareRunStarted(state, {
      runId: RUN_ID,
      flowFingerprint: FINGERPRINT,
      initialPool,
      startWiringOverride: {},
      effectiveLlmByStaticPath: {},
      numWorkers: 0,
      memorySnapshots: snapshots,
      learnable: options.learnable,
      memories,
      memoryStacks: recordedStacks,
    }),
  );
  return { state, log };
}

/**
 * Return a state of `START -> node -> END` (as `newState` builds it) with START succeeded and
 * `node` the sole in-flight node, and its log.
 */
export function startTraceNode(node: Node, options: TraceNodeOptions): { state: RunState; log: MemoryRunLog } {
  const { state, log } = newState(node, options);
  commitPrepared(log, state, prepareNodeStarted(state, START_ID));
  commitStartSuccess(state, log, {});
  commitPrepared(log, state, prepareNodeStarted(state, node.id));
  return { state, log };
}

/**
 * Return every retained-trace field in a form `toEqual` compares: entries, occurrences and
 * boundaries as arrays, and the backward handlers by reference.
 */
export function traceSnapshot(trace: RetainedTrace): Record<string, unknown> {
  return {
    endId: trace.endId,
    rootInputNames: [...trace.rootInputNames],
    rootStartSource: trace.rootStartSource,
    terminal: trace.terminal,
    rootWriter: trace.rootWriter,
    causalRouteSource: trace.causalRouteSource,
    occurrences: [...trace.occurrences],
    sources: [...trace.sources],
    currentWriters: [...trace.currentWriters],
    started: [...trace.started],
    nextSequence: trace.nextSequence,
    boundaries: [...trace.boundaries],
    pendingConnections: [...trace.pendingConnections],
    backwardHandlers: [...trace.backwardHandlers],
  };
}

/**
 * Return `traceSnapshot` for comparing a replay with its live run: the backward handlers by key,
 * since each run binds its own nodes' handlers and a bound function can't name its node.
 */
export function replayTraceSnapshot(trace: RetainedTrace): Record<string, unknown> {
  return { ...traceSnapshot(trace), backwardHandlers: [...trace.backwardHandlers.keys()] };
}

/**
 * Replay `source`'s records over a fresh state of `flow` through `prepareRunStarted` (from the
 * recorded RunStarted fields) and `prepareRecord`; return the state, the new log and each
 * transition's observers.
 */
export function replayOutcomeLog(
  flow: CompiledFlow,
  source: MemoryRunLog,
): { state: RunState; log: MemoryRunLog; observers: (readonly RunEvent[])[] } {
  const records = source.records().map((envelope) => envelope.record);
  const started = records[0];
  if (!(started instanceof RunStartedRecord)) throw new Error("the log must start with RunStarted");
  const state = new RunState(flow);
  const log = new MemoryRunLog(RUN_ID);
  const observers = [
    commitPrepared(
      log,
      state,
      prepareRunStarted(state, {
        runId: RUN_ID,
        flowFingerprint: started.flowFingerprint,
        initialPool: decodePool(started.initialPool),
        startWiringOverride: started.startWiringOverride,
        effectiveLlmByStaticPath: started.effectiveLlmByStaticPath,
        numWorkers: started.numWorkers,
      }),
    ),
  ];
  for (const record of records.slice(1)) observers.push(commitPrepared(log, state, prepareRecord(state, record)));
  return { state, log, observers };
}

// The growth builders (Python's `test_run_state_growth.py` helpers).

/** A minimal kind-blind spawner used to isolate reducer behavior. */
export class Spawner extends Node {
  readonly kind = NodeKind.CALL;
  override readonly isSpawner = true;
  override readonly growDepthDelta = 1;
}

/** Return a complete static graph whose one spawner can be put in flight. */
export function parentFlow(options: { spawner?: Node; sibling?: boolean } = {}): CompiledFlow {
  const spawner = options.spawner ?? new Spawner("spawn");
  const nodes = new Map<string, Node>([
    [START_ID, new StartNode(START_ID, { params: [] })],
    [spawner.id, spawner],
    [END_ID, EndNode.value(END_ID, { outputName: null })],
  ]);
  if (options.sibling) nodes.set("side", new FuncNode("side", (inputs) => inputs));
  const edges = [new Edge("start-spawn", START_ID, spawner.id), new Edge("spawn-end", spawner.id, END_ID)];
  if (options.sibling) edges.push(new Edge("start-side", START_ID, "side"));
  return CompiledFlow.fromParts(nodes, edges, { wiring: new Map([...nodes.keys()].map((id) => [id, new Map()])) });
}

/** Return a replay-valid state and log with the static spawner in flight. */
export function initializedSpawner(flow: CompiledFlow = parentFlow()): { state: RunState; log: MemoryRunLog } {
  const initialPool = new VariablePool(flow.startId);
  initialPool.set(flow.startId, {});
  initialPool.addSystem("run_id", RUN_ID);
  const state = new RunState(flow);
  const log = new MemoryRunLog(RUN_ID);
  commitPrepared(
    log,
    state,
    prepareRunStarted(state, {
      runId: RUN_ID,
      flowFingerprint: FINGERPRINT,
      initialPool,
      startWiringOverride: {},
      effectiveLlmByStaticPath: {},
      numWorkers: 0,
    }),
  );
  commitPrepared(log, state, prepareNodeStarted(state, START_ID));
  commitPrepared(log, state, prepareNodeSucceeded(state, START_ID, {}));
  commitPrepared(log, state, prepareNodeStarted(state, "spawn"));
  return { state, log };
}

/** Return a manifestable two-boundary fragment committing under `spawnerId`. */
export function fragment(spawnerId: string, suffix: string = "child"): Flow {
  const startId = `${spawnerId}/${suffix}/start`;
  const endId = `${spawnerId}/${suffix}/end`;
  const end = EndNode.value(endId, { outputName: null });
  end.commitAs = spawnerId;
  return new Flow(
    new Map<string, Node>([
      [startId, new StartNode(startId, { params: [] })],
      [endId, end],
    ]),
    [new Edge(`${startId}->${endId}`, startId, endId, { ordering: true })],
    {
      wiring: new Map([
        [startId, new Map()],
        [endId, new Map()],
      ]),
      startId,
      endId,
    },
  );
}

/** A builder returning `fragment(spawner.id)`. */
export const fragmentBuilder: GrowBuilder = (spawner) => fragment(spawner.id);

/** A memory log, holding `prior`'s records, whose append fails before it retains the record. */
export class FailingLog extends MemoryRunLog {
  constructor(runId: string, prior: Iterable<RecordEnvelope> = []) {
    super(runId);
    for (const envelope of prior) super.append(envelope.record);
  }

  override append(_record: RunRecord): RecordEnvelope {
    throw new Error("disk full");
  }
}

/**
 * A minimal custom log that returns a caller-selected append result and never advances. It has
 * run id `RUN_ID` and next sequence 0 unless `options` says otherwise.
 */
export class ReturningLog implements RunLog {
  readonly runId: string;
  readonly path = null;
  readonly nextSequence: number;
  readonly #result: (record: RunRecord) => RecordEnvelope;

  constructor(
    result: (record: RunRecord) => RecordEnvelope,
    options: { runId?: string; nextSequence?: number } = {},
  ) {
    this.#result = result;
    this.runId = options.runId ?? RUN_ID;
    this.nextSequence = options.nextSequence ?? 0;
  }

  [Symbol.iterator](): Iterator<RecordEnvelope> {
    return [][Symbol.iterator]();
  }

  records(): readonly RecordEnvelope[] {
    return [];
  }

  append(record: RunRecord): RecordEnvelope {
    return this.#result(record);
  }

  close(): void {}

  [Symbol.dispose](): void {}
}
