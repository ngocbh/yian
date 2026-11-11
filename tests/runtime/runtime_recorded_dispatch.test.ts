/**
 * Bounded dispatch over the append-only run-state reducer: real reducer and log transitions pin
 * durability order, unfinished starts, observer order and the drain. Each test reads a failure from
 * the run's `NodeFailedRecord`; ints are `bigint`s.
 */

import { describe, expect, test } from "bun:test";

import { CompiledFlow, Edge, END_ID, Flow, GrowthRejected, START_ID } from "../../src/compile/model.ts";
import { type Failure, OriginRef } from "../../src/diagnostics.ts";
import {
  NodeExpanded,
  NodeFailed,
  NodeRouted,
  NodeStarted,
  NodeSucceeded,
  RunFailed,
  RunSucceeded,
  StreamChunk,
} from "../../src/events.ts";
import { Grow, Node, type NodeCaps, NodeKind, type NodeResult, Output, Route } from "../../src/nodes/base.ts";
import { ContractOrigins, type OutputContract, PartialRecordContract, ValueContract } from "../../src/nodes/contracts.ts";
import { EndNode } from "../../src/nodes/end/node.ts";
import { StartNode } from "../../src/nodes/start/node.ts";
import { FlowEngine, type FlowEngineOptions } from "../../src/runtime/engine.ts";
import {
  MemoryRunLog,
  NodeExpandedRecord,
  NodeFailedRecord,
  NodeStartedRecord,
  NodeSucceededRecord,
  RecordEnvelope,
  type RunRecord,
  RunStoppedRecord,
  SCHEMA_VERSION,
} from "../../src/runtime/log.ts";
import { commitInitialPool, commitPrepared, type GrowBuilder, NodeDataError, prepareRunStarted, RunState } from "../../src/runtime/state.ts";
import { VariablePool } from "../../src/typesys/pool.ts";
import { Type, ValueKind } from "../../src/typesys/values.ts";
import { aborted, BlockingNode, fragment, fragmentBuilder, noGrowth, rejected } from "../helpers/run_state_fixtures.ts";
import { YianRangeError, YianTypeError } from "../../src/typesys/errors.ts";

const RUN_ID = "recorded-dispatch";
const FINGERPRINT = "d".repeat(64);

/** Each worker mode: serial and one pooled worker. */
const MODES = [
  ["serial", 0],
  ["pooled", 1],
] as const;

/** Runs one injected action and returns its value (awaited, so a test may decide when). */
class ActionNode extends Node {
  readonly kind = NodeKind.CODE;
  readonly action: () => unknown;

  constructor(
    nodeId: string,
    action: () => unknown,
    options: { outputContract?: OutputContract | null; origin?: OriginRef | null } = {},
  ) {
    super(nodeId, options);
    this.action = action;
  }

  override async forward(): Promise<NodeResult> {
    return new Output(await this.action());
  }
}

/** Returns one fixed route handle. */
class RouteNode extends Node {
  readonly kind = NodeKind.CASE;

  override forward(): NodeResult {
    return new Route("chosen");
  }
}

/** Returns a handle absent from this node's outgoing edges. */
class InvalidRouteNode extends Node {
  readonly kind = NodeKind.CASE;

  override forward(): NodeResult {
    return new Route("absent");
  }
}

/** Returns malformed node-produced success metadata. */
class InvalidCommitTargetNode extends Node {
  readonly kind = NodeKind.CODE;

  override forward(): NodeResult {
    return new Output("done", { commitAs: 42 as never });
  }
}

/** Returns one value with a result-owned commit target. */
class ExplicitCommitNode extends ActionNode {
  readonly target: string;

  constructor(nodeId: string, action: () => unknown, target: string) {
    super(nodeId, action);
    this.target = target;
  }

  override async forward(): Promise<NodeResult> {
    return new Output(await this.action(), { commitAs: this.target });
  }
}

/** A spawner returning one seed-first expansion once `entered` (if given) resolves. */
class GrowNode extends Node {
  readonly kind = NodeKind.CALL;
  override readonly isSpawner = true;
  // A growth is rebuilt from its binding (live as a check, and on replay), so the double is
  // derived: its forward is synchronous unless it waits on `entered`.
  override readonly derived = true;
  override readonly growDepthDelta: number | null = 1;
  readonly seed: unknown;
  readonly entered: Promise<void> | null;

  constructor(
    nodeId: string,
    options: { seed?: unknown; entered?: Promise<void> | null; origin?: OriginRef | null } = {},
  ) {
    super(nodeId, { origin: options.origin ?? null });
    this.seed = "seed" in options ? options.seed : { canonical: [1n, 2n] };
    this.entered = options.entered ?? null;
  }

  override forward(): NodeResult | Promise<NodeResult> {
    return this.entered === null ? new Grow(this.seed) : this.entered.then(() => new Grow(this.seed));
  }
}

/**
 * Resolves `entered` once it runs, waits for its signal to abort, then rejects with an error of its
 * own rather than the signal's reason.
 */
class DrainFaultNode extends Node {
  readonly kind = NodeKind.CODE;
  /** Resolves once the node runs. */
  readonly entered: Promise<void>;
  /** Whether the node has stopped waiting. */
  released = false;
  readonly #enter: () => void;

  constructor(nodeId: string) {
    super(nodeId);
    const { promise, resolve } = Promise.withResolvers<void>();
    this.entered = promise;
    this.#enter = resolve;
  }

  override async forward(_inputs: Record<string, unknown>, caps: NodeCaps): Promise<NodeResult> {
    this.#enter();
    await aborted(caps.signal!).catch(() => {});
    this.released = true;
    throw new Error("sibling failed during the drain");
  }
}

/** Streams `early`, calls `streamed`, waits for its signal to abort, streams `late`, then returns. */
class LateStreamNode extends Node {
  readonly kind = NodeKind.CODE;
  override readonly needsLiveEvents = true;
  readonly streamed: () => void;

  constructor(nodeId: string, streamed: () => void = () => {}) {
    super(nodeId);
    this.streamed = streamed;
  }

  override async forward(_inputs: Record<string, unknown>, caps: NodeCaps): Promise<NodeResult> {
    caps.emitStream?.(new StreamChunk({ nodeId: this.id, key: "output", chunk: "early" }));
    this.streamed();
    await aborted(caps.signal!).catch(() => {});
    caps.emitStream?.(new StreamChunk({ nodeId: this.id, key: "output", chunk: "late" }));
    return new Output("discarded");
  }
}

/** Return START fanning out into `nodes`, each joined into one output-less END by ordering edges. */
function fanFlow(nodes: readonly Node[]): CompiledFlow {
  const all = new Map<string, Node>([
    [START_ID, new StartNode(START_ID, { params: [] })],
    ...nodes.map((node) => [node.id, node] as [string, Node]),
    [END_ID, EndNode.value(END_ID, { outputName: null })],
  ]);
  const edges = [
    ...nodes.map((node) => new Edge(`start-${node.id}`, START_ID, node.id, { optional: true, ordering: true })),
    ...nodes.map((node) => new Edge(`${node.id}-end`, node.id, END_ID, { optional: true, ordering: true })),
  ];
  return new CompiledFlow(all, edges);
}

/** A memory log that runs `guard` before each append and counts `close` calls. */
class GuardedLog extends MemoryRunLog {
  guard: ((record: RunRecord) => void) | null = null;
  closeCalls = 0;

  override append(record: RunRecord): RecordEnvelope {
    this.guard?.(record);
    return super.append(record);
  }

  override close(): void {
    this.closeCalls += 1;
    super.close();
  }
}

/** Commit one authoritative RunStarted for `flow` and return its state and log, START ready. */
function initialized<L extends MemoryRunLog = MemoryRunLog>(
  flow: CompiledFlow,
  numWorkers: number,
  log?: L,
): { state: RunState; log: L } {
  const state = new RunState(flow);
  const pool = new VariablePool(flow.startId);
  pool.set(flow.startId, {});
  pool.addSystem("run_id", RUN_ID);
  const target = log ?? (new MemoryRunLog(RUN_ID) as L);
  commitPrepared(
    target,
    state,
    prepareRunStarted(state, {
      runId: RUN_ID,
      flowFingerprint: FINGERPRINT,
      startWiringOverride: {},
      effectiveLlmByStaticPath: {},
      numWorkers,
    }),
  );
  commitInitialPool(target, state, pool);
  return { state, log: target };
}

/** Return an engine over `state` and `log` whose growth builder is unused unless given. */
function dispatcher(
  state: RunState,
  log: MemoryRunLog,
  numWorkers: number,
  options: Partial<FlowEngineOptions> = {},
): FlowEngine {
  return new FlowEngine(state, log, { growBuilder: noGrowth, numWorkers, ...options });
}

/** Return the failures the run recorded, in log order. */
function failures(log: MemoryRunLog): Failure[] {
  return log
    .records()
    .map((envelope) => envelope.record)
    .filter((record) => record instanceof NodeFailedRecord)
    .map((record) => (record as NodeFailedRecord).failure);
}

/** Whether the log holds a `NodeFailedRecord`. */
function anyFailure(log: MemoryRunLog): boolean {
  return failures(log).length > 0;
}

/** Whether the log holds a record of class `kind` about `nodeId`. */
function hasRecord(log: MemoryRunLog, kind: abstract new (...args: never[]) => unknown, nodeId: string): boolean {
  return log
    .records()
    .some((envelope) => envelope.record instanceof kind && (envelope.record as { nodeId?: string }).nodeId === nodeId);
}

/** The ids of the in-flight nodes, in dispatch order. */
function inFlight(state: RunState): string[] {
  return [...state.inFlight.keys()];
}

describe("durability and host faults", () => {
  test("started record is durable before node code runs", async () => {
    const executed: string[] = [];
    const { state, log } = initialized(
      fanFlow([new ActionNode("work", () => executed.push("work") && "done")]),
      0,
      new GuardedLog(RUN_ID),
    );
    log.guard = (record) => {
      if (record instanceof NodeStartedRecord && record.nodeId === "work") expect(executed).toEqual([]);
    };
    await dispatcher(state, log, 0).run();

    const records = log.records().map((envelope) => envelope.record);
    const startIndex = records.findIndex((record) => record instanceof NodeStartedRecord && record.nodeId === "work");
    const successIndex = records.findIndex(
      (record) => record instanceof NodeSucceededRecord && record.nodeId === "work",
    );
    expect(executed).toEqual(["work"]);
    expect(startIndex).toBeGreaterThan(-1);
    expect(startIndex).toBeLessThan(successIndex);
  });

  test("start append failure prevents node execution and propagates exactly", async () => {
    const executed: string[] = [];
    const expected = new Error("start fsync failed");
    const { state, log } = initialized(
      fanFlow([new ActionNode("work", () => executed.push("work") && "done")]),
      0,
      new GuardedLog(RUN_ID),
    );
    log.guard = (record) => {
      if (record instanceof NodeStartedRecord && record.nodeId === "work") throw expected;
    };

    expect(await rejected(dispatcher(state, log, 0).run())).toBe(expected);
    expect(executed).toEqual([]);
    expect(state.inFlight.has("work")).toBe(false);
  });

  test("outcome append failure leaves one unfinished durable start", async () => {
    const executed: string[] = [];
    const expected = new Error("outcome fsync failed");
    const { state, log } = initialized(
      fanFlow([new ActionNode("work", () => executed.push("work") && "done")]),
      0,
      new GuardedLog(RUN_ID),
    );
    log.guard = (record) => {
      if (record instanceof NodeSucceededRecord && record.nodeId === "work") throw expected;
    };

    expect(await rejected(dispatcher(state, log, 0).run())).toBe(expected);
    expect(executed).toEqual(["work"]);
    expect(inFlight(state)).toEqual(["work"]);
    expect(hasRecord(log, NodeStartedRecord, "work")).toBe(true);
    expect(hasRecord(log, NodeSucceededRecord, "work")).toBe(false);
  });

  test("a log returning a mismatched outcome envelope is not rewritten as a node failure", async () => {
    const { state, log } = initialized(fanFlow([new ActionNode("work", () => "done")]), 0);
    const append = log.append.bind(log);
    log.append = (record) => {
      if (record instanceof NodeSucceededRecord && record.nodeId === "work") {
        return new RecordEnvelope(SCHEMA_VERSION, RUN_ID, 99, record);
      }
      return append(record);
    };

    const error = await rejected(dispatcher(state, log, 0).run());
    expect(error).toEqual(new YianRangeError("run-log append result does not match prepared transition"));
    expect(inFlight(state)).toEqual(["work"]);
    expect(anyFailure(log)).toBe(false);
  });

  test("serial yields and callbacks follow post durable observer order", async () => {
    const { state, log } = initialized(fanFlow([new ActionNode("work", () => "done")]), 0);
    const callbacks: unknown[] = [];

    const yielded = await dispatcher(state, log, 0, { onEvent: (event) => callbacks.push(event) }).run();

    expect(callbacks).toEqual([...yielded]);
    expect(yielded.map((event) => event.constructor)).toEqual([
      NodeStarted,
      NodeSucceeded,
      NodeStarted,
      NodeSucceeded,
      NodeStarted,
      NodeSucceeded,
      RunSucceeded,
    ]);
    expect(yielded.flatMap((event) => ("nodeId" in event ? [event.nodeId] : []))).toEqual([
      START_ID,
      START_ID,
      "work",
      "work",
      END_ID,
      END_ID,
    ]);
  });

  test("authority mismatch is rejected before any node executes", async () => {
    const executed: string[] = [];
    const { state } = initialized(fanFlow([new ActionNode("work", () => executed.push("work"))]), 0);
    const otherLog = new MemoryRunLog("other-run");

    const error = await rejected(dispatcher(state, otherLog, 0).run());
    expect(error).toBeInstanceOf(RangeError);
    expect((error as Error).message).toContain("run_id mismatch");
    expect(executed).toEqual([]);
    expect(otherLog.nextSequence).toBe(0);
  });

  test("dispatcher borrows and never closes its log", async () => {
    const { state, log } = initialized(fanFlow([new ActionNode("work", () => "done")]), 0, new GuardedLog(RUN_ID));

    await dispatcher(state, log, 0).run();

    expect(log.closeCalls).toBe(0);
    expect(state.result).toBeInstanceOf(RunSucceeded);
    log.close();
    expect(log.closeCalls).toBe(1);
  });
});

describe("output contracts", () => {
  test("invalid raw output constructs contract failure with operation origin", async () => {
    const outerOrigin = new OriginRef("nodes/outer");
    const nodeOrigin = new OriginRef("nodes/work");
    const contractOrigin = new OriginRef("nodes/work/output");
    const node = new ActionNode("work", () => "not-an-int", {
      outputContract: new ValueContract(
        Type.scalar(ValueKind.INTEGER),
        new ContractOrigins({ default: contractOrigin }),
      ),
      origin: nodeOrigin,
    });
    node.originContext = [outerOrigin];
    const { state, log } = initialized(fanFlow([node]), 0);

    const events = await dispatcher(state, log, 0).run();

    const recorded = failures(log);
    expect(recorded.length).toBe(1);
    expect(recorded[0]!.code).toBe("contract.output");
    expect(recorded[0]!.nodeId).toBe("work");
    expect(recorded[0]!.trace).toEqual([outerOrigin, contractOrigin]);
    expect(recorded[0]!.causeType).toBe("TypeCheckError");
    expect(recorded[0]!.traceback).not.toBeNull();
    expect(events.at(-2)).toBeInstanceOf(NodeFailed);
    expect(events.at(-1)).toBeInstanceOf(RunFailed);
    expect((events.at(-1) as RunFailed).failure.causeType).toBe("TypeCheckError");
    expect(state.result).toBeInstanceOf(RunFailed);
    expect(log.records().at(-1)!.record).toBeInstanceOf(NodeFailedRecord);
  });

  /** A record type holding a required int `score`. */
  const scoreItem = (): Type =>
    new Type({
      kind: ValueKind.OBJECT,
      fields: new Map([["score", Type.scalar(ValueKind.INTEGER)]]),
      required: new Set(["score"]),
    });

  test("nested output failure uses the exact contract origin", async () => {
    const rootOrigin = new OriginRef("nodes/work/output");
    const scoreOrigin = new OriginRef("nodes/work/output/items/0/score");
    const node = new ActionNode("work", () => [{ score: "bad" }], {
      outputContract: new ValueContract(
        new Type({ kind: ValueKind.LIST_OBJECT, element: scoreItem() }),
        new ContractOrigins({ default: rootOrigin, entries: [[[0, "score"], scoreOrigin]] }),
      ),
    });
    const { state, log } = initialized(fanFlow([node]), 0);

    await dispatcher(state, log, 0).run();

    const recorded = failures(log);
    expect(recorded.length).toBe(1);
    expect(recorded[0]!.code).toBe("contract.output");
    expect(recorded[0]!.trace).toEqual([scoreOrigin]);
  });

  test("partial output failure uses the known field origin", async () => {
    const countOrigin = new OriginRef("defs/body/output/count");
    const node = new ActionNode("work", () => ({ count: "bad", note: "opaque" }), {
      outputContract: new PartialRecordContract([
        ["count", new ValueContract(Type.scalar(ValueKind.INTEGER), new ContractOrigins({ default: countOrigin }))],
        ["note", null],
      ]),
      origin: new OriginRef("nodes/work"),
    });
    const { state, log } = initialized(fanFlow([node]), 0);

    await dispatcher(state, log, 0).run();

    const recorded = failures(log);
    expect(recorded.length).toBe(1);
    expect(recorded[0]!.code).toBe("contract.output");
    expect(recorded[0]!.trace).toEqual([countOrigin]);
    expect(recorded[0]!.causeType).toBe("TypeCheckError");
  });

  test("partial output shape failure falls back to node origin", async () => {
    // Unknown keys are refused although no field contract owns their origin.
    const nodeOrigin = new OriginRef("nodes/work");
    const node = new ActionNode("work", () => ({ count: 1n, note: "opaque", extra: true }), {
      outputContract: new PartialRecordContract([
        ["count", new ValueContract(Type.scalar(ValueKind.INTEGER))],
        ["note", null],
      ]),
      origin: nodeOrigin,
    });
    const { state, log } = initialized(fanFlow([node]), 0);

    await dispatcher(state, log, 0).run();

    const recorded = failures(log);
    expect(recorded.length).toBe(1);
    expect(recorded[0]!.code).toBe("contract.output");
    expect(recorded[0]!.trace).toEqual([nodeOrigin]);
    expect(recorded[0]!.message).toContain("unknown fields");
  });

  test("nonfinite output at the root uses the contract default origin", async () => {
    const rootOrigin = new OriginRef("nodes/work/output");
    const node = new ActionNode("work", () => NaN, {
      outputContract: new ValueContract(
        Type.scalar(ValueKind.NUMBER),
        new ContractOrigins({ default: rootOrigin, entries: [[["unrelated"], new OriginRef("nodes/work/output/unrelated")]] }),
      ),
    });
    const { state, log } = initialized(fanFlow([node]), 0);

    await dispatcher(state, log, 0).run();

    const recorded = failures(log);
    expect(recorded.length).toBe(1);
    expect(recorded[0]!.code).toBe("contract.output");
    expect(recorded[0]!.causeType).toBe("TypeCheckError");
    expect(recorded[0]!.message).toBe("NaN is not a yian value");
    expect(recorded[0]!.trace).toEqual([rootOrigin]);
  });

  test("non type output failure uses the contract default origin", async () => {
    // A canonical-JSON refusal has no value path and keeps the root fallback.
    const rootOrigin = new OriginRef("nodes/work/output");
    const node = new ActionNode("work", () => "\ud800", {
      outputContract: new ValueContract(
        Type.scalar(ValueKind.STRING),
        new ContractOrigins({ default: rootOrigin, entries: [[["unrelated"], new OriginRef("nodes/work/output/unrelated")]] }),
      ),
    });
    const { state, log } = initialized(fanFlow([node]), 0);

    await dispatcher(state, log, 0).run();

    const recorded = failures(log);
    expect(recorded.length).toBe(1);
    expect(recorded[0]!.code).toBe("contract.output");
    expect(recorded[0]!.causeType).toBe("TypeCheckError");
    expect(recorded[0]!.message).toBe("text holding a lone surrogate is not a yian value");
    expect(recorded[0]!.trace).toEqual([rootOrigin]);
  });

  test.each([
    ["baked", false],
    ["result", true],
  ])("redirected output uses effective target contract origin (%s)", async (_, explicit) => {
    const producerOrigin = new OriginRef("nodes/producer");
    const contractOrigin = new OriginRef("nodes/target/output");
    const itemsOrigin = new OriginRef("nodes/target/output/items");
    const action = (): unknown => ({ items: [{ score: 1n }, { score: "bad" }] });
    const producer = explicit
      ? new ExplicitCommitNode("producer", action, "target")
      : new ActionNode("producer", action, { origin: producerOrigin });
    producer.origin = producerOrigin;
    producer.commitAs = explicit ? "absent" : "target";
    const flow = fanFlow([producer]);
    const outputType = new Type({
      kind: ValueKind.OBJECT,
      fields: new Map([["items", new Type({ kind: ValueKind.LIST_OBJECT, element: scoreItem() })]]),
      required: new Set(["items"]),
    });
    flow.nodes.set(
      "target",
      new ActionNode("target", () => ({ items: [{ score: 1n }] }), {
        outputContract: new ValueContract(
          outputType,
          new ContractOrigins({ default: contractOrigin, entries: [[["items"], itemsOrigin]] }),
        ),
      }),
    );
    const { state, log } = initialized(flow, 0);

    await dispatcher(state, log, 0).run();

    const recorded = failures(log);
    expect(recorded.length).toBe(1);
    expect(recorded[0]!.code).toBe("contract.output");
    expect(recorded[0]!.nodeId).toBe("producer");
    expect(recorded[0]!.trace).toEqual([itemsOrigin]);
  });

  test("malformed compiled output contract remains a host failure", async () => {
    // The dispatcher must not blame a node for corrupt compiled IR.
    const { state, log } = initialized(fanFlow([new ActionNode("work", () => "done")]), 0);
    state.flow.nodes.get("work")!.outputContract = Type.scalar(ValueKind.STRING) as unknown as OutputContract;

    const error = await rejected(dispatcher(state, log, 0).run());
    expect(error).toBeInstanceOf(TypeError);
    expect((error as Error).message).toContain("exact ValueContract");
    expect(error).not.toBeInstanceOf(NodeDataError);
    expect(inFlight(state)).toEqual(["work"]);
    expect(anyFailure(log)).toBe(false);
  });

  test.each(MODES)("malformed nested output type remains a host failure (%s)", async (_, numWorkers) => {
    // Corrupt contract metadata must not pass for a refused node output.
    const nested = new Type({
      kind: ValueKind.OBJECT,
      fields: new Map([["value", Type.scalar(ValueKind.INTEGER)]]),
      required: new Set([1 as never, "value"]),
    });
    const outputType = new Type({
      kind: ValueKind.OBJECT,
      fields: new Map([["nested", nested]]),
      required: new Set(["nested"]),
    });
    const node = new ActionNode("work", () => ({ nested: {} }), { outputContract: new ValueContract(outputType) });
    const { state, log } = initialized(fanFlow([node]), numWorkers);

    const error = await rejected(dispatcher(state, log, numWorkers).run());
    expect(error).toBeInstanceOf(TypeError);
    expect((error as Error).message).toContain("required fields");
    expect(error).not.toBeInstanceOf(NodeDataError);
    expect(inFlight(state)).toEqual(["work"]);
    expect(anyFailure(log)).toBe(false);
  });

  test.each(MODES)("noncanonical output type metadata remains a host failure (%s)", async (_, numWorkers) => {
    // A contract-owned lone surrogate fails before the output is classified.
    const field = "bad\ud800";
    const outputType = new Type({
      kind: ValueKind.OBJECT,
      fields: new Map([[field, Type.scalar(ValueKind.STRING)]]),
      required: new Set([field]),
    });
    const node = new ActionNode("work", () => ({ [field]: "value" }), { outputContract: new ValueContract(outputType) });
    const { state, log } = initialized(fanFlow([node]), numWorkers);

    const error = await rejected(dispatcher(state, log, numWorkers).run());
    expect(error).toBeInstanceOf(RangeError);
    expect((error as Error).message).toContain("surrogate");
    expect(error).not.toBeInstanceOf(NodeDataError);
    expect(inFlight(state)).toEqual(["work"]);
    expect(anyFailure(log)).toBe(false);
  });

  test.each(MODES)("excessively nested output constructs contract failure (%s)", async (_, numWorkers) => {
    let value: unknown = "leaf";
    for (let depth = 0; depth < 1_200; depth += 1) value = [value];
    const { state, log } = initialized(fanFlow([new ActionNode("work", () => value)]), numWorkers);

    const events = await dispatcher(state, log, numWorkers).run();

    const recorded = failures(log);
    expect(recorded.length).toBe(1);
    expect(recorded[0]!.code).toBe("contract.output");
    expect(recorded[0]!.causeType).toBe("TypeCheckError");
    expect(recorded[0]!.message).toContain("deeper than 100 levels");
    expect(events.at(-2)).toBeInstanceOf(NodeFailed);
    expect(events.at(-1)).toBeInstanceOf(RunFailed);
    expect(log.records().at(-1)!.record).toBeInstanceOf(NodeFailedRecord);
  });

  test.each(MODES)("invalid output metadata constructs contract failure with node origin (%s)", async (_, numWorkers) => {
    const nodeOrigin = new OriginRef("nodes/work");
    const { state, log } = initialized(fanFlow([new InvalidCommitTargetNode("work", { origin: nodeOrigin })]), numWorkers);

    const events = await dispatcher(state, log, numWorkers).run();

    const recorded = failures(log);
    expect(recorded.length).toBe(1);
    expect(recorded[0]!.code).toBe("contract.output");
    expect(recorded[0]!.nodeId).toBe("work");
    expect(recorded[0]!.trace).toEqual([nodeOrigin]);
    expect(recorded[0]!.causeType).toBe("TypeError");
    const failed = events.at(-2) as NodeFailed;
    expect(failed).toBeInstanceOf(NodeFailed);
    expect(failed.failure.nodeId).toBe("work");
    expect(failed.failure.message).toBe("commit_as must be a string or null");
    expect(failed.failure.causeType).toBe("TypeError");
    expect(events.at(-1)).toBeInstanceOf(RunFailed);
    expect(log.records().at(-1)!.record).toBeInstanceOf(NodeFailedRecord);
  });
});

describe("routes", () => {
  test("route raw outcome is prepared committed and reduced", async () => {
    const ran: string[] = [];
    const router = new RouteNode("route");
    const nodes = new Map<string, Node>([
      [START_ID, new StartNode(START_ID, { params: [] })],
      ["route", router],
      ["chosen", new ActionNode("chosen", () => ran.push("chosen") && "yes")],
      ["skipped", new ActionNode("skipped", () => ran.push("skipped") && "no")],
      [END_ID, EndNode.value(END_ID, { outputName: null })],
    ]);
    const flow = new CompiledFlow(nodes, [
      new Edge("start-route", START_ID, "route", { optional: true, ordering: true }),
      new Edge("route-chosen", "route", "chosen", { sourceHandle: "chosen" }),
      new Edge("route-skipped", "route", "skipped", { sourceHandle: "other" }),
      new Edge("chosen-end", "chosen", END_ID, { optional: true, ordering: true }),
      new Edge("skipped-end", "skipped", END_ID, { optional: true, ordering: true }),
    ]);
    const { state, log } = initialized(flow, 0);

    const events = await dispatcher(state, log, 0).run();

    expect(ran).toEqual(["chosen"]);
    expect(events.some((event) => event instanceof NodeRouted)).toBe(true);
    expect(state.result).toBeInstanceOf(RunSucceeded);
  });

  test("invalid route constructs contract failure with node origin", async () => {
    const outerOrigin = new OriginRef("nodes/outer");
    const nodeOrigin = new OriginRef("nodes/route");
    const node = new InvalidRouteNode("route", { origin: nodeOrigin });
    node.originContext = [outerOrigin];
    const { state, log } = initialized(fanFlow([node]), 0);

    const events = await dispatcher(state, log, 0).run();

    const recorded = failures(log);
    expect(recorded.length).toBe(1);
    expect(recorded[0]!.code).toBe("contract.route");
    expect(recorded[0]!.nodeId).toBe("route");
    expect(recorded[0]!.trace).toEqual([outerOrigin, nodeOrigin]);
    expect(recorded[0]!.causeType).toBe("RangeError");
    const failed = events.at(-2) as NodeFailed;
    expect(failed).toBeInstanceOf(NodeFailed);
    expect(failed.failure.nodeId).toBe("route");
    expect(failed.failure.causeType).toBe("RangeError");
    expect(failed.failure.message).toContain("matches no outgoing edge");
    expect(events.at(-1)).toBeInstanceOf(RunFailed);
    expect(log.records().at(-1)!.record).toBeInstanceOf(NodeFailedRecord);
  });

  test("source free route rejection has an empty trace", async () => {
    // A node built in code doesn't invent source provenance.
    const { state, log } = initialized(fanFlow([new InvalidRouteNode("route")]), 0);

    await dispatcher(state, log, 0).run();

    const recorded = failures(log);
    expect(recorded.length).toBe(1);
    expect(recorded[0]!.code).toBe("contract.route");
    expect(recorded[0]!.trace).toEqual([]);
  });
});

describe("growth", () => {
  test("grow raw outcome uses injected builder and flat record", async () => {
    const { state, log } = initialized(fanFlow([new GrowNode("spawn")]), 0);

    const events = await dispatcher(state, log, 0, { growBuilder: fragmentBuilder }).run();

    expect(events.some((event) => event instanceof NodeExpanded)).toBe(true);
    const expanded = log
      .records()
      .map((envelope) => envelope.record)
      .filter((record) => record instanceof NodeExpandedRecord) as NodeExpandedRecord[];
    expect(expanded.length).toBe(1);
    expect(expanded[0]!.spawnerId).toBe("spawn");
    // The record holds no seed; the observer carries it.
    expect((events.find((event) => event instanceof NodeExpanded) as NodeExpanded).grow.seed).toEqual({ canonical: [1n, 2n] });
    expect(state.result).toBeInstanceOf(RunSucceeded);
  });

  test("invalid live grow depth trait remains a host failure", async () => {
    const spawner = new GrowNode("spawn");
    (spawner as { growDepthDelta: unknown }).growDepthDelta = "invalid";
    const { state, log } = initialized(fanFlow([spawner]), 0);

    const error = await rejected(dispatcher(state, log, 0, { growBuilder: fragmentBuilder }).run());
    expect((error as Error).constructor).toBe(YianRangeError);
    expect((error as Error).message).toContain("grow_depth_delta");
    expect(inFlight(state)).toEqual(["spawn"]);
    expect(anyFailure(log)).toBe(false);
  });

  for (const [mode, numWorkers] of MODES) {
    test.each([Error, TypeError, RangeError])(`injected grow builder fault remains exact host failure (${mode}, %p)`, async (kind) => {
      const expected = new kind("builder programming fault");
      const { state, log } = initialized(fanFlow([new GrowNode("spawn")]), numWorkers);
      const rejectBuilder: GrowBuilder = () => {
        throw expected;
      };

      expect(await rejected(dispatcher(state, log, numWorkers, { growBuilder: rejectBuilder }).run())).toBe(expected);
      expect(inFlight(state)).toEqual(["spawn"]);
      expect(anyFailure(log)).toBe(false);
    });
  }

  test.each(MODES)("invalid grow builder return remains a host failure (%s)", async (_, numWorkers) => {
    const { state, log } = initialized(fanFlow([new GrowNode("spawn")]), numWorkers);

    const error = await rejected(dispatcher(state, log, numWorkers, { growBuilder: () => ({}) as Flow }).run());
    expect(error).toEqual(new YianTypeError("grow_builder must return a Flow"));
    expect((error as Error).constructor).toBe(YianTypeError);
    expect(inFlight(state)).toEqual(["spawn"]);
    expect(anyFailure(log)).toBe(false);
  });

  test.each([
    ["authored", new OriginRef("nodes/spawn"), [new OriginRef("nodes/outer")], [new OriginRef("nodes/outer"), new OriginRef("nodes/spawn")]],
    ["source-free", null, [], []],
  ] as const)("legal kind malformed seed constructs contract growth failure (%s)", async (_, origin, context, trace) => {
    const node = new GrowNode("spawn", { seed: [], origin });
    node.originContext = context;
    const { state, log } = initialized(fanFlow([node]), 0);
    // Stands in for the compile-owned CALL builder, which refuses a seed that isn't a record.
    const callBuilder: GrowBuilder = (spawner, seed) => {
      if (typeof seed !== "object" || seed === null || Array.isArray(seed)) {
        throw new GrowthRejected(new TypeError("CALL seed must be a record"));
      }
      return fragment(spawner.id);
    };

    const events = await dispatcher(state, log, 0, { growBuilder: callBuilder }).run();

    const recorded = failures(log);
    expect(recorded.length).toBe(1);
    expect(recorded[0]!.code).toBe("contract.growth");
    expect(recorded[0]!.nodeId).toBe("spawn");
    expect(recorded[0]!.trace).toEqual([...trace]);
    expect(recorded[0]!.causeType).toBe("TypeError");
    expect(events.at(-2)).toBeInstanceOf(NodeFailed);
    expect(events.at(-1)).toBeInstanceOf(RunFailed);
  });

  test("grow builder fault drains pooled sibling and preserves identity", async () => {
    const blocked = new BlockingNode("blocked", new Promise(() => {}));
    const expected = new Error("builder programming fault");
    const { state, log } = initialized(fanFlow([new GrowNode("spawn", { entered: blocked.entered }), blocked]), 2);
    const rejectBuilder: GrowBuilder = () => {
      throw expected;
    };

    expect(await rejected(dispatcher(state, log, 2, { growBuilder: rejectBuilder }).run())).toBe(expected);
    expect(blocked.released).toBe(true);
    expect(inFlight(state)).toEqual(["spawn", "blocked"]);
    expect(anyFailure(log)).toBe(false);
  });
});

describe("pooled dispatch", () => {
  test("positive worker count never exceeds the submission bound", async () => {
    const { state, log } = initialized(
      fanFlow(Array.from({ length: 5 }, (_, index) => new ActionNode(`work-${index}`, () => "done"))),
      2,
    );
    const inFlightAtStart: number[] = [];
    const onEvent = (event: unknown): void => {
      if (event instanceof NodeStarted && event.nodeId.startsWith("work-")) inFlightAtStart.push(state.inFlight.size);
    };

    await dispatcher(state, log, 2, { onEvent }).run();

    expect(inFlightAtStart.length).toBe(5);
    expect(Math.max(...inFlightAtStart)).toBe(2);
    expect(state.result).toBeInstanceOf(RunSucceeded);
  });

  test("parallel outcomes commit in completion message order", async () => {
    const fastCommitted = Promise.withResolvers<void>();
    const { state, log } = initialized(
      fanFlow([
        new ActionNode("slow", async () => {
          await fastCommitted.promise;
          return "slow";
        }),
        new ActionNode("fast", () => "fast"),
      ]),
      2,
    );
    const onEvent = (event: unknown): void => {
      if (event instanceof NodeSucceeded && event.nodeId === "fast") fastCommitted.resolve();
    };

    await dispatcher(state, log, 2, { onEvent }).run();

    const completed = log
      .records()
      .map((envelope) => envelope.record)
      .filter((record) => record instanceof NodeSucceededRecord && ["slow", "fast"].includes(record.nodeId))
      .map((record) => (record as NodeSucceededRecord).nodeId);
    expect(completed).toEqual(["fast", "slow"]);
  });
});

describe("host callbacks and drains", () => {
  test("event callback failure signals drains and reraises same exception", async () => {
    const expected = new Error("observer failed");
    const blocked = new DrainFaultNode("blocked");
    const { state, log } = initialized(
      fanFlow([new BlockingNode("trigger", blocked.entered.then(() => "triggered")), blocked]),
      2,
    );
    const onEvent = (event: unknown): void => {
      if (event instanceof NodeSucceeded && event.nodeId === "trigger") throw expected;
    };
    const built = dispatcher(state, log, 2, { onEvent });

    expect(await rejected(built.run())).toBe(expected);
    expect(blocked.released).toBe(true);
    expect(built.signal.reason).toBe(expected);
    expect(inFlight(state)).toEqual(["blocked"]);
    expect(state.result).toBeNull();
    // Only the stop is appended after the trigger's success, whose observer threw.
    const [succeeded, stopped] = log.records().slice(-2).map((envelope) => envelope.record);
    expect(succeeded).toBeInstanceOf(NodeSucceededRecord);
    expect((succeeded as NodeSucceededRecord).nodeId).toBe("trigger");
    expect(stopped).toEqual(new RunStoppedRecord("callback", "observer failed", ["blocked"]));
  });

  test("started observer failure prevents submission of that node", async () => {
    const executed: string[] = [];
    const expected = new Error("started observer failed");
    const { state, log } = initialized(fanFlow([new ActionNode("work", () => executed.push("work") && "done")]), 1);
    const onEvent = (event: unknown): void => {
      if (event instanceof NodeStarted && event.nodeId === "work") throw expected;
    };

    expect(await rejected(dispatcher(state, log, 1, { onEvent }).run())).toBe(expected);
    expect(executed).toEqual([]);
    expect(inFlight(state)).toEqual(["work"]);
    expect(log.records().at(-1)!.record).toEqual(new RunStoppedRecord("callback", "started observer failed", ["work"]));
  });

  test("node failure drains sibling and discards late stream chunks", async () => {
    const streamed = Promise.withResolvers<void>();
    const chunks: string[] = [];
    const fail = new ActionNode("fail", async () => {
      await streamed.promise;
      throw new Error("boom");
    });
    const { state, log } = initialized(fanFlow([new LateStreamNode("stream", streamed.resolve), fail]), 2);

    const events = await dispatcher(state, log, 2, { onStream: (chunk) => chunks.push(chunk.chunk) }).run();

    expect(chunks).toEqual(["early"]);
    expect(events.at(-2)).toBeInstanceOf(NodeFailed);
    expect((events.at(-2) as NodeFailed).failure.nodeId).toBe("fail");
    expect(events.at(-1)).toBeInstanceOf(RunFailed);
    expect(inFlight(state)).toEqual(["stream"]);
    expect(hasRecord(log, NodeSucceededRecord, "stream")).toBe(false);
  });

  test("primary host interruption is preserved after full sibling drain", async () => {
    const host = new AbortController();
    const primary = new Error("terminal EOF");
    const blocked = new BlockingNode("blocked", new Promise(() => {}));
    const interrupt = new ActionNode("interrupt", async () => {
      await blocked.entered;
      host.abort(primary);
      throw new Error("interrupted");
    });
    const { state, log } = initialized(fanFlow([blocked, interrupt]), 2);

    expect(await rejected(dispatcher(state, log, 2, { signal: host.signal }).run())).toBe(primary);
    expect(blocked.released).toBe(true);
    expect(state.result).toBeNull();
    expect(log.records().at(-1)!.record).toEqual(new RunStoppedRecord("signal", "terminal EOF", ["blocked", "interrupt"]));
  });

  test.each(MODES)("stream callback failure is host error not durable node failure (%s)", async (_, numWorkers) => {
    const expected = new Error("stream callback failed");
    const { state, log } = initialized(fanFlow([new LateStreamNode("stream")]), numWorkers);
    const onStream = (): void => {
      throw expected;
    };

    expect(await rejected(dispatcher(state, log, numWorkers, { onStream }).run())).toBe(expected);
    expect(inFlight(state)).toEqual(["stream"]);
    expect(anyFailure(log)).toBe(false);
    expect(log.records().at(-1)!.record).toEqual(new RunStoppedRecord("callback", "stream callback failed", ["stream"]));
  });
});
