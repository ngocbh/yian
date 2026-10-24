/**
 * Engine-level fakes for waits and wakes: TOOL nodes whose `take`, `sleep` or host call the engine
 * decides, a CODE node held until the test releases it, a flow whose START declares inboxes, and a
 * clock the test drives. The reducer recognises a TOOL node by `hostToolCall()`, so these run only
 * through `engine()`/`inboxEngine()`, which never build a manifest.
 */
import { applyExecutionLayout, resolveExecutionLayout } from "../../src/compile/execution_layout.ts";
import { CompiledFlow, Edge, END_ID, START_ID } from "../../src/compile/model.ts";
import { type CompleteResult, Node, NodeKind, type NodeCaps, type NodeResult, Output, Pending } from "../../src/nodes/base.ts";
import { type CallView, HostToolCall, Parameter, ValueContract } from "../../src/nodes/contracts.ts";
import { EndNode } from "../../src/nodes/end/node.ts";
import { StartNode } from "../../src/nodes/start/node.ts";
import type { Scheduler } from "../../src/platform.ts";
import type { ComposedFlow } from "../../src/compose/build/pipeline.ts";
import { replayRunLog, startWiring } from "../../src/run.ts";
import { FlowEngine } from "../../src/runtime/engine.ts";
import { formatRecordTime, JsonlRunLog, MemoryRunLog, type RunLog } from "../../src/runtime/log.ts";
import { commitInitialPool, commitPrepared, prepareComplete, prepareRunStarted, RunState } from "../../src/runtime/state.ts";
import { toolCallId } from "../../src/tools/context.ts";
import { FunctionTool } from "../../src/tools/function_tools.ts";
import type { JsonObject } from "../../src/typesys/json.ts";
import { VariablePool } from "../../src/typesys/pool.ts";
import { inboxType, Type, type ValueRecord, ValueKind } from "../../src/typesys/values.ts";
import { type engine, FINGERPRINT, FIXED_CLOCK, noGrowth, RUN_ID } from "./run_state_fixtures.ts";

/** A TOOL `take` on `inbox`: the engine decides it at commit. */
export class FakeTake extends Node {
  readonly kind = NodeKind.TOOL;
  override readonly needsSteps = true;
  readonly inbox: string;
  readonly mode: "one" | "all";
  readonly timeoutSeconds: number | null;

  constructor(id: string, inbox: string, mode: "one" | "all" = "one", timeoutSeconds: number | null = null) {
    super(id);
    this.inbox = inbox;
    this.mode = mode;
    this.timeoutSeconds = timeoutSeconds;
  }

  override hostToolCall(): HostToolCall {
    return new HostToolCall("take", [], new Map(), null);
  }

  override forward(): NodeResult {
    return new Pending([], { wait: { tool: "take", inbox: this.inbox, mode: this.mode, timeoutSeconds: this.timeoutSeconds } });
  }
}

/** A TOOL `sleep` for `seconds`. */
export class FakeSleep extends Node {
  readonly kind = NodeKind.TOOL;
  override readonly needsSteps = true;
  readonly seconds: number;

  constructor(id: string, seconds: number) {
    super(id);
    this.seconds = seconds;
  }

  override hostToolCall(): HostToolCall {
    return new HostToolCall("sleep", [], new Map(), null);
  }

  override forward(): NodeResult {
    return new Pending([], { wait: { tool: "sleep", until: null, forSeconds: this.seconds } });
  }
}

/**
 * A TOOL host call `tool` in run `runId`, `deadlineAt` (epoch ms) stamped on an `ask`'s pending step.
 * - No call held: count a handler call, record the `pending` step, return Pending([id]).
 * - An open call: return Pending([id]) without calling the handler.
 * - An answered call: return its output.
 * - A closed or timed-out call: return null.
 */
export class FakeHost extends Node {
  readonly kind = NodeKind.TOOL;
  override readonly needsSteps = true;
  readonly tool: string;
  readonly runId: string;
  readonly deadlineAt: number | null;
  handlerCalls = 0;
  /** The `maxEventBytes` each evaluation saw. */
  limits: number[] = [];

  constructor(id: string, options: { tool?: string; runId?: string; deadlineAt?: number | null } = {}) {
    super(id);
    this.tool = options.tool ?? "lookup";
    this.runId = options.runId ?? RUN_ID;
    this.deadlineAt = options.deadlineAt ?? null;
  }

  get callId(): string {
    return toolCallId(this.runId, this.id);
  }

  override hostToolCall(): HostToolCall {
    return new HostToolCall(this.tool, [], new Map(), null);
  }

  override forward(_inputs: Record<string, unknown>, caps: NodeCaps): NodeResult {
    const steps = caps.steps!;
    this.limits.push(steps.maxEventBytes);
    const held = steps.calls.find((call) => call.callId === this.callId);
    if (held?.state === "answered") return new Output((held.result as { output: { value: unknown } }).output.value);
    if (held?.state === "closed" || held?.state === "timed_out") return new Output(null);
    if (held === undefined) {
      this.handlerCalls += 1;
      const step: ValueRecord = { kind: "pending", call_id: this.callId, tool: this.tool, arguments: {} };
      if (this.deadlineAt !== null) step["deadline"] = formatRecordTime(this.deadlineAt);
      steps.record(step);
    }
    return new Pending([this.callId]);
  }

  override waitResultStep(call: CallView, result: CompleteResult): JsonObject {
    return "output" in result
      ? { kind: "tool_result", call_id: call.callId, output: { kind: "str", value: String(result.output) } }
      : { kind: "tool_result", call_id: call.callId, error: { type: "CompletedError", message: result.error } };
  }
}

/** A CODE node that blocks until `release()`, or until its signal aborts; `started` resolves when it begins. */
export class GateNode extends Node {
  readonly kind = NodeKind.CODE;
  readonly #started = Promise.withResolvers<void>();
  readonly #gate = Promise.withResolvers<unknown>();
  aborted = false;

  get started(): Promise<void> {
    return this.#started.promise;
  }

  release(value: unknown = null): void {
    this.#gate.resolve(value);
  }

  override async forward(_inputs: Record<string, unknown>, caps: NodeCaps): Promise<NodeResult> {
    this.#started.resolve();
    const signal = caps.signal!;
    const stop = new Promise<never>((_, reject) =>
      signal.addEventListener(
        "abort",
        () => {
          this.aborted = true;
          reject(signal.reason);
        },
        { once: true },
      ),
    );
    return new Output(await Promise.race([this.#gate.promise, stop]));
  }
}

/** START with an `Inbox(str)` param per name in `inboxes`, the nodes, END, and `graph`-style `[from, to, handle?]` edges. */
export function inboxFlow(
  inboxes: readonly string[],
  nodes: readonly Node[],
  edges: readonly (readonly [string, string, string?])[],
): CompiledFlow {
  const str = Type.scalar(ValueKind.STRING);
  const params = inboxes.map((name) => new Parameter(name, { contract: new ValueContract(inboxType(str)) }));
  return CompiledFlow.fromParts(
    new Map<string, Node>([
      [START_ID, new StartNode(START_ID, { params })],
      ...nodes.map((node) => [node.id, node] as const),
      [END_ID, EndNode.value(END_ID, { outputName: null })],
    ]),
    edges.map(([from, to, handle], i) => new Edge(`e${i}`, from, to, { sourceHandle: handle ?? null })),
  );
}

/**
 * As `engine()` (run_state_fixtures.ts), with START wired by `startWiring(params, inputs)`, so each
 * inbox binds its handle. The run records the constant `FINGERPRINT`, so any node class runs.
 */
export function inboxEngine(flow: CompiledFlow, options: Parameters<typeof engine>[1] = {}): FlowEngine {
  const {
    inputs = {},
    runId = RUN_ID,
    learnable = false,
    numWorkers = 0,
    growBuilder = noGrowth,
    defaults = {},
    effectiveLlmByStaticPath = {},
    memories: _memories,
    memoryStacks: _memoryStacks,
    memorySnapshots: _memorySnapshots,
    ...rest
  } = options;
  if (learnable) applyExecutionLayout(flow, resolveExecutionLayout(flow));
  const state = new RunState(flow);
  const initialPool = new VariablePool(flow.startId);
  initialPool.set(flow.startId, { ...inputs });
  initialPool.addSystem("run_id", runId);
  const log = new MemoryRunLog(runId, { clock: FIXED_CLOCK });
  const params = flow.nodes.get(flow.startId)!.params ?? [];
  const prepared = prepareRunStarted(state, {
    runId,
    flowFingerprint: FINGERPRINT,
    startWiringOverride: startWiring(params, inputs),
    effectiveLlmByStaticPath,
    numWorkers,
    learnable,
    defaults,
  });
  commitPrepared(log, state, prepared);
  commitInitialPool(log, state, initialPool);
  return new FlowEngine(state, log, { ...rest, growBuilder, numWorkers });
}

/** A `Scheduler` the test drives, with `set` to move its time and `pending` to count live sleepers. */
export type HandClock = Scheduler & { set(ms: number): void; pending(): number };

/**
 * A clock the test drives, honouring the `Scheduler` contract:
 * - `now()` is what the test last set;
 * - `sleep(ms, signal)` rejects at once on an aborted signal, resolves at once for `ms <= 0`, and
 *   otherwise registers a sleeper due at now+ms, calls `onSleep(deadline)`, resolves when `set`
 *   reaches it, and rejects on abort (removing itself);
 * - `set(ms)` resolves every sleeper due by `ms`, in deadline order;
 * - `pending()` counts live sleepers.
 */
export function handClock(start: number, onSleep?: (deadline: number) => void): HandClock {
  let now = start;
  const sleepers: { readonly at: number; readonly wake: () => void }[] = [];
  return {
    now: () => now,
    sleep(ms: number, signal?: AbortSignal): Promise<void> {
      if (signal?.aborted) return Promise.reject(signal.reason);
      if (!Number.isFinite(ms)) return Promise.reject(new RangeError(`sleep ms must be finite, got ${ms}`));
      if (ms <= 0) return Promise.resolve();
      return new Promise<void>((resolve, reject) => {
        const onAbort = (): void => {
          sleepers.splice(sleepers.indexOf(sleeper), 1);
          reject(signal!.reason);
        };
        const sleeper = {
          at: now + ms,
          wake: () => {
            signal?.removeEventListener("abort", onAbort);
            resolve();
          },
        };
        signal?.addEventListener("abort", onAbort, { once: true });
        sleepers.push(sleeper);
        onSleep?.(sleeper.at);
      });
    },
    set(ms: number): void {
      now = ms;
      const due = sleepers.filter((sleeper) => sleeper.at <= ms).sort((a, b) => a.at - b.at);
      for (const sleeper of due) sleepers.splice(sleepers.indexOf(sleeper), 1);
      for (const sleeper of due) sleeper.wake();
    },
    pending: () => sleepers.length,
  };
}

/** The record class names of `log`, in order. */
export const recordNames = (log: RunLog): string[] => log.records().map((entry) => entry.record.constructor.name);

/**
 * Answer call `callId` of the waiting run at `log` as an outside writer does: lock and replay the
 * log through `flow`, append the answer, and close. Returns the answer's seq, or `null` for a
 * repeat.
 */
export function completeByHand(log: string, flow: ComposedFlow, callId: string, result: CompleteResult): number | null {
  const authority = JsonlRunLog.resume(log);
  try {
    const state = replayRunLog(flow, authority);
    const prepared = prepareComplete(state, callId, result);
    if (prepared === null) return null;
    commitPrepared(authority, state, prepared);
    return prepared.expectedSequence;
  } finally {
    authority.close();
  }
}

/**
 * A host tool `gate` whose handler resolves `entered`, then waits until `release()` (answering
 * with what it was given) or until its call's signal aborts (rejecting with the reason).
 */
export function gateTool(): { readonly tool: FunctionTool; readonly entered: Promise<void>; release(value?: unknown): void } {
  const entered = Promise.withResolvers<void>();
  const gate = Promise.withResolvers<unknown>();
  const tool = new FunctionTool({
    name: "gate",
    description: "waits for the test",
    parametersJsonSchema: { type: "object", properties: {} },
    execute: async (_args, ctx) => {
      entered.resolve();
      const stopped = new Promise<never>((_, reject) => {
        if (ctx.signal.aborted) reject(ctx.signal.reason);
        ctx.signal.addEventListener("abort", () => reject(ctx.signal.reason), { once: true });
      });
      return await Promise.race([gate.promise, stopped]);
    },
  });
  return { tool, entered: entered.promise, release: (value: unknown = "opened") => gate.resolve(value) };
}
