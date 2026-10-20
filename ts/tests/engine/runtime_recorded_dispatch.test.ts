/**
 * Bounded dispatch over the append-only run-state reducer: a port of
 * `tests/engine/test_runtime_recorded_dispatch.py`. Real reducer and log transitions pin durability
 * order, unfinished starts, observer order and the drain.
 *
 * Python's `_capture_semantic_failures` spies on `FlowEngine._prepare_failure`; here each test reads
 * the failure from the run's `NodeFailedRecord`. Python's `ValueError` is a `RangeError` here, and
 * a contract's type mismatch the value layer's `TypeCheckError`; ints are `bigint`s.
 *
 * Ported: `test_started_record_is_durable_before_node_code_runs`,
 * `test_start_append_failure_prevents_node_execution_and_propagates_exactly`,
 * `test_outcome_append_failure_leaves_one_unfinished_durable_start`,
 * `test_serial_yields_and_callbacks_follow_post_durable_observer_order`,
 * `test_invalid_raw_output_constructs_contract_failure_with_operation_origin`,
 * `test_nested_output_failure_uses_the_exact_contract_origin`,
 * `test_partial_output_failure_uses_the_known_field_origin`,
 * `test_partial_output_shape_failure_falls_back_to_node_origin`,
 * `test_redirected_output_uses_effective_target_contract_origin` (2 rows),
 * `test_malformed_compiled_output_contract_remains_a_host_failure`,
 * `test_malformed_nested_output_type_remains_a_host_failure`,
 * `test_noncanonical_output_type_metadata_remains_a_host_failure`,
 * `test_excessively_nested_output_constructs_contract_failure`,
 * `test_invalid_output_metadata_constructs_contract_failure_with_node_origin` (the last four
 * serial and pooled), `test_route_raw_outcome_is_prepared_committed_and_reduced`,
 * `test_invalid_route_constructs_contract_failure_with_node_origin`,
 * `test_source_free_route_rejection_has_an_empty_trace`,
 * `test_grow_raw_outcome_uses_injected_builder_and_flat_record`,
 * `test_invalid_live_grow_depth_trait_remains_a_host_failure`,
 * `test_injected_grow_builder_fault_remains_exact_host_failure` (3 error classes, serial and
 * pooled), `test_invalid_grow_builder_return_remains_a_host_failure` (serial and pooled),
 * `test_parallel_outcomes_commit_in_completion_message_order`,
 * `test_authority_mismatch_is_rejected_before_any_node_executes` and
 * `test_dispatcher_borrows_and_never_closes_its_log`.
 *
 * Ported (adapted):
 * - `test_legal_kind_malformed_seed_constructs_contract_growth_failure` (2 rows): TS has no
 *   compile-owned CALL builder yet, so a test builder refuses the list seed with
 *   `GrowthRejected(TypeError)`, as Python's `build_grow` does.
 * - `test_grow_builder_fault_drains_pooled_sibling_and_preserves_identity`: the blocked sibling
 *   waits for its signal to abort, then rejects with the signal's reason; the spawner waits until
 *   the sibling has entered.
 * - `test_positive_worker_count_never_exceeds_the_submission_bound`: the bound is read as the
 *   in-flight count at each start, with no timing.
 * - `test_nonfinite_output_at_the_root_uses_the_contract_default_origin`: the message is the value
 *   layer's `NaN is not a yian value`.
 * - `test_non_type_output_failure_uses_the_contract_default_origin`: a lone surrogate is refused by
 *   the value layer, so the cause is a `TypeCheckError` (Python's canonical-JSON `ValueError`), still
 *   without a value path.
 *
 * Changed (the monkeypatch rows, which replace private engine or reducer functions):
 * - `test_reducer_programming_failure_is_not_rewritten_as_node_failure`: the log returns a
 *   mismatched envelope for the outcome, so `commitPrepared` throws; the run rejects with that
 *   error and no failure is recorded.
 * - `test_success_preparation_runtime_error_remains_exact_host_failure` and the 12 rows of
 *   `test_transition_preparation_data_shaped_programming_error_remains_exact_host_failure`: covered
 *   by the malformed contract rows (success preparation, serial and pooled) and the grow builder
 *   rows (growth preparation, serial and pooled); a route preparation host fault is the
 *   reducer's (`run_state_outcomes.test.ts`).
 * - `test_growth_digest_fault_remains_exact_host_failure` and
 *   `test_growth_materialization_fault_remains_exact_host_failure`: covered by the grow builder
 *   fault rows, a host fault inside growth preparation.
 *
 * Not ported: `test_submit_failure_is_exact_and_leaves_the_started_node_recoverable`: TS has no
 * executor, so nothing can refuse a submission.
 *
 * Later (the engine's stop rule): `test_event_callback_failure_signals_drains_and_reraises_same_exception`,
 * `test_cleanup_failure_never_replaces_callback_error_and_drain_is_attempted` (replaced: a worker
 * rejecting during the drain never replaces the callback error),
 * `test_started_observer_failure_prevents_submission_of_that_node`,
 * `test_primary_host_interruption_is_preserved_after_full_sibling_drain`,
 * `test_node_failure_drains_sibling_and_discards_late_stream_chunks` and
 * `test_stream_callback_failure_is_host_error_not_durable_node_failure`.
 */

import { describe, expect, test } from "bun:test";

import { CompiledFlow, Edge, END_ID, Flow, GrowthRejected, START_ID } from "../../src/compile/model.ts";
import { type Failure, OriginRef } from "../../src/diagnostics.ts";
import { NodeExpanded, NodeFailed, NodeRouted, NodeStarted, NodeSucceeded, RunFailed, RunSucceeded } from "../../src/events.ts";
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
  SCHEMA_VERSION,
} from "../../src/runtime/log.ts";
import { commitPrepared, type GrowBuilder, NodeDataError, prepareRunStarted, RunState } from "../../src/runtime/state.ts";
import { VariablePool } from "../../src/typesys/pool.ts";
import { Type, ValueKind } from "../../src/typesys/values.ts";
import { fragment, fragmentBuilder, rejected } from "./run_state_fixtures.ts";

const RUN_ID = "recorded-dispatch";
const FINGERPRINT = "d".repeat(64);

/** Each worker mode Python parametrizes: serial and one pooled worker. */
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

  override async forward(): Promise<NodeResult> {
    if (this.entered !== null) await this.entered;
    return new Grow(this.seed);
  }
}

/** Signals `entered`, waits for its signal to abort, signals `released`, then rejects with the reason. */
class StopAwareNode extends Node {
  readonly kind = NodeKind.CODE;
  readonly entered: () => void;
  readonly released: () => void;

  constructor(nodeId: string, entered: () => void, released: () => void) {
    super(nodeId);
    this.entered = entered;
    this.released = released;
  }

  override async forward(_inputs: Record<string, unknown>, caps: NodeCaps): Promise<NodeResult> {
    const signal = caps.signal!;
    this.entered();
    if (!signal.aborted) await new Promise((resolve) => signal.addEventListener("abort", resolve, { once: true }));
    this.released();
    throw signal.reason;
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
  return CompiledFlow.fromParts(all, edges);
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
      initialPool: pool,
      startWiringOverride: {},
      effectiveLlmByStaticPath: {},
      numWorkers,
    }),
  );
  return { state, log: target };
}

/** Return an engine over `state` and `log` whose growth builder is unused unless given. */
function dispatcher(
  state: RunState,
  log: MemoryRunLog,
  numWorkers: number,
  options: Partial<FlowEngineOptions> = {},
): FlowEngine {
  return new FlowEngine(state, log, { growBuilder: () => null as never, numWorkers, ...options });
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
    expect(error).toEqual(new RangeError("run-log append result does not match prepared transition"));
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
    const flow = CompiledFlow.fromParts(nodes, [
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
    expect(expanded[0]!.seed).toEqual({ canonical: [1n, 2n] });
    expect(state.result).toBeInstanceOf(RunSucceeded);
  });

  test("invalid live grow depth trait remains a host failure", async () => {
    const spawner = new GrowNode("spawn");
    (spawner as { growDepthDelta: unknown }).growDepthDelta = "invalid";
    const { state, log } = initialized(fanFlow([spawner]), 0);

    const error = await rejected(dispatcher(state, log, 0, { growBuilder: fragmentBuilder }).run());
    expect((error as Error).constructor).toBe(RangeError);
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
    expect(error).toEqual(new TypeError("grow_builder must return a Flow"));
    expect((error as Error).constructor).toBe(TypeError);
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
    const entered = Promise.withResolvers<void>();
    let released = false;
    const expected = new Error("builder programming fault");
    const flow = fanFlow([
      new GrowNode("spawn", { entered: entered.promise }),
      new StopAwareNode("blocked", entered.resolve, () => {
        released = true;
      }),
    ]);
    const { state, log } = initialized(flow, 2);
    const rejectBuilder: GrowBuilder = () => {
      throw expected;
    };

    expect(await rejected(dispatcher(state, log, 2, { growBuilder: rejectBuilder }).run())).toBe(expected);
    expect(released).toBe(true);
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
