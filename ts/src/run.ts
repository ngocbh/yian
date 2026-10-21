/**
 * The package-root run boundary over one authoritative append-only log.
 *
 * `runFlow` initializes a private copy of a loaded flow, commits its complete canonical start
 * state, and drives only post-durable observer notifications.
 *
 * The runtime borrows an already-open log. This module owns log creation and closure after the
 * engine's structured drain. It also owns the compile/runtime inversion seam by injecting
 * `compile/expand.ts`'s `buildGrow` into the kind-blind engine.
 *
 * Knows about:   compile, compose, defaults, diagnostics, events, llm_clients, memory contracts,
 *                nodes, platform, runtime, skills, tools, typesys.
 * Never imports: the CLI.
 */

import { resolve } from "node:path";
import { types } from "node:util";

import { buildGrow } from "./compile/expand.ts";
import { flowFingerprint } from "./compile/manifest.ts";
import { copyGraph } from "./compile/model.ts";
import { ComposedFlow } from "./compose/build/pipeline.ts";
import { childSignature } from "./compose/nodes/resolver.ts";
import { current, type Defaults, type Env, nonDefaults, processEnv } from "./defaults.ts";
import { type Failure, reprText } from "./diagnostics.ts";
import { type RunEvent, RunFailed, RunSucceeded, type StreamChunk } from "./events.ts";
import { defaultModelAccess, type ModelAccess } from "./llm_clients/factory.ts";
import type { LearningResult, Memory } from "./memory/contracts.ts";
import type { Steering } from "./nodes/base.ts";
import { checkInputNames, coerceInputs, copyValue } from "./nodes/binding.ts";
import type { Parameter } from "./nodes/contracts.ts";
import { realScheduler, type Scheduler } from "./platform.ts";
import type { BackwardOutcome } from "./runtime/contracts.ts";
import { FlowEngine } from "./runtime/engine.ts";
import { addNote, requireSteering } from "./runtime/eval_node.ts";
import { JsonlRunLog, MemoryRunLog, type RunLog } from "./runtime/log.ts";
import { commitPrepared, prepareRunStarted, RunState } from "./runtime/state.ts";
import type { Skill } from "./skills/catalog.ts";
import { isMcpServer } from "./tools/mcp.ts";
import { type ToolEntry, ToolSet } from "./tools/registry.ts";
import { VariablePool } from "./typesys/pool.ts";
import { defaultRunId, seedSystemClock } from "./typesys/seeding.ts";
import {
  fromHost,
  jsTypeName,
  rejectDeepNesting,
  setField,
  TypeCheckError,
  type Value,
  type ValueRecord,
} from "./typesys/values.ts";

/** Sees each durable observer, and each in-loop model and tool event. */
export type EventHook = (event: RunEvent) => unknown;
/** Sees each transient stream chunk. */
export type StreamHook = (chunk: StreamChunk) => unknown;

/**
 * The terminal, surface-independent result of a fresh run.
 *
 * `events` holds only the observers committed during this call.
 */
export class RunResult {
  /** The canonical caller-supplied inputs `RunStarted` retained. */
  readonly input: Record<string, unknown>;
  /** Either `"succeeded"` or `"failed"`. */
  readonly status: "succeeded" | "failed";
  /** The durable identity shared by the state and its run log. */
  readonly runId: string;
  /** The JSONL log path as given to `runFlow`, or `null` for a process-local run. */
  readonly log: string | null;
  /** The terminal flow value on success; otherwise `null`. */
  readonly output: unknown;
  /** The canonical terminal failure on failure; otherwise `null`. */
  readonly failure: Failure | null;
  /** The post-durable observers of this call only. */
  readonly events: readonly RunEvent[];
  /**
   * The number of accepted starts of each node that a resume retried with no recorded stop, by
   * node id; empty for a run never resumed.
   */
  readonly interruptedAttempts: ReadonlyMap<string, number>;
  #backwardOutcome: BackwardOutcome | null;

  /** Hold the given fields; `output`, `failure` and `backwardOutcome` default to `null`. */
  constructor(fields: {
    input: Record<string, unknown>;
    status: "succeeded" | "failed";
    runId: string;
    log: string | null;
    output?: unknown;
    failure?: Failure | null;
    events?: readonly RunEvent[];
    interruptedAttempts?: ReadonlyMap<string, number>;
    backwardOutcome?: BackwardOutcome | null;
  }) {
    this.input = fields.input;
    this.status = fields.status;
    this.runId = fields.runId;
    this.log = fields.log;
    this.output = fields.output ?? null;
    this.failure = fields.failure ?? null;
    this.events = fields.events ?? [];
    this.interruptedAttempts = fields.interruptedAttempts ?? new Map();
    this.#backwardOutcome = fields.backwardOutcome ?? null;
  }

  /**
   * The learning projection of the last observed backward attempt, initially `null`. It never
   * changes the forward result.
   */
  get learning(): LearningResult | null {
    return this.#backwardOutcome?.learning ?? null;
  }

  /** The last observed accepted backward disposition, initially `null`. */
  get backwardOutcome(): BackwardOutcome | null {
    return this.#backwardOutcome;
  }
}

/** The options of one `runFlow` call. */
export interface RunFlowOptions {
  /**
   * A new JSONL log path. `null` (the default) uses a process-local `MemoryRunLog`; an existing
   * path is never overwritten.
   */
  log?: string | null;
  /**
   * The non-empty identity exposed as `system.run_id` and retained by every record. `null` (the
   * default) mints a fresh one.
   */
  runId?: string | null;
  /** The non-negative worker bound recorded at initialization (default 8); zero runs serially. */
  numWorkers?: number;
  /**
   * Sees each observer after its record is appended and reduced, plus the transient in-loop AGENT
   * events. Called synchronously; a throw rejects the run with that error.
   */
  onEvent?: EventHook | null;
  /** Sees each transient stream chunk; called as `onEvent` is. */
  onStream?: StreamHook | null;
  /** The per-run outer model configuration layer. */
  llmConfig?: Readonly<Record<string, unknown>> | null;
  /** The backend of each declared memory key, typically a `MemoryRegistry`. */
  memory?: ReadonlyMap<string, Memory> | null;
  /** Retain accepted execution for a later explicit backward call (default `false`). */
  learnable?: boolean;
  /**
   * The tools this run sees, keyed by tool id; `null` (the default) uses the registered tools.
   * A record, even an empty one, is the whole set.
   */
  tools?: Readonly<Record<string, ToolEntry>> | null;
  /** The skills this run can grant, keyed by `Skill.name`. */
  skills?: ReadonlyMap<string, Skill> | null;
  /** The host inbox for AGENT nodes authored `steerable: true`. */
  steering?: Steering | null;
  /** The host's app object a tool receives as `ctx.context`; never recorded. */
  context?: unknown;
  /**
   * The host's stop: aborting it stops the run, which rejects with its reason. An already-aborted
   * signal rejects before anything is created.
   */
  signal?: AbortSignal | null;
  /** The environment for connection selection, credentials and model building (default `process.env`). */
  env?: Env | null;
  /** For tests: the clock the system clock is seeded from and node waits run on (default `realScheduler`). */
  scheduler?: Scheduler | null;
  /** For tests: a draw in [0, 1) for backoff jitter (default `Math.random`). */
  random?: (() => number) | null;
  /** For tests: the model builder AGENT nodes use (default `defaultModelAccess(env)`). */
  models?: ModelAccess | null;
}

/**
 * Start and drive one flow through an in-memory or durable run log.
 *
 * Flow failures resolve as a `RunResult` with `status` `"failed"`. Host, callback, log and
 * reducer errors reject after any running nodes have settled and the log has been closed.
 *
 * @param flow The loaded flow to copy and execute.
 * @param inputs Caller values keyed by declared input name; any other key is an error. Each is
 *   taken through host intake (an own data property, `{n: 3}` becomes the int `3n`), and
 *   supplied text is coerced to a declared scalar; START applies the declared defaults and the
 *   full boundary type checks as its recorded execution.
 * @param options See `RunFlowOptions`.
 * @returns The terminal success or failure, with this call's post-durable observers.
 * @throws TypeError when `flow` isn't a `ComposedFlow`, `learnable` isn't a boolean, `signal`
 *   isn't an `AbortSignal`, `inputs` isn't a record (a Proxy included), a `tools` entry is
 *   malformed, or `steering` lacks its methods.
 * @throws TypeCheckError when an input holds an accessor, NaN, a lone surrogate, a cycle or
 *   another non-yian value, nests deeper than `YIAN_MAX_VALUE_NESTING` levels, or has text that
 *   can't be read.
 * @throws RangeError when an input key names no declared input, or the identity, worker count or
 *   prepared start state breaks its contract.
 * @throws DefaultsError when a `YIAN_<NAME>` variable holds a value its entry doesn't accept.
 * @throws The signal's reason when it aborts.
 * @throws Error (code `EEXIST`) when `log` names an existing path, and the `fs` error when the log
 *   can't be created, locked, written, synced or closed.
 */
export async function runFlow(
  flow: ComposedFlow,
  inputs: Readonly<Record<string, unknown>>,
  options: RunFlowOptions = {},
): Promise<RunResult> {
  // A bad YIAN_<NAME> value fails here, before any work.
  const runDefaults = current();
  const { learnable = false, tools = null, steering = null, context, signal = null } = options;
  if (!(flow instanceof ComposedFlow)) throw new TypeError("runFlow expects a ComposedFlow");
  if (typeof learnable !== "boolean") throw new TypeError("learnable must be a boolean");
  if (signal !== null && !(signal instanceof AbortSignal)) throw new TypeError("signal must be an AbortSignal or null");
  signal?.throwIfAborted();
  // A server the run opens and closes itself.
  if (tools !== null && Object.values(tools).some((entry) => isMcpServer(entry) && !entry.isOpen)) {
    throw new Error("run-owned MCP servers are not ported yet");
  }
  new ToolSet(tools, { context });
  requireSteering(steering);
  return startRun(flow, inputs, options, runDefaults);
}

/**
 * Take each supplied input through host intake, keyed as given.
 *
 * @throws TypeError when `inputs` isn't a record or is a Proxy.
 * @throws RangeError when a key names no declared input.
 * @throws TypeCheckError when an input is an accessor property or isn't a yian value.
 */
function hostInputs(params: readonly Parameter[], inputs: unknown): Record<string, Value> {
  // Checked first: reading a Proxy's prototype or keys would run its traps.
  if (types.isProxy(inputs)) throw new TypeError("inputs must be a mapping of input names to values, got Proxy");
  checkInputNames(params, inputs);
  const record = inputs as Readonly<Record<string, unknown>>;
  const taken: ValueRecord = {};
  for (const name of Object.keys(record)) {
    const where = `input ${reprText(name)}`;
    const descriptor = Object.getOwnPropertyDescriptor(record, name)!;
    if (!("value" in descriptor)) throw new TypeCheckError(`${where}: accessor property is not a yian value`);
    setField(taken, name, fromHost(descriptor.value, where));
  }
  return taken;
}

/** Record and drive a fresh run whose checks have passed. */
async function startRun(
  flow: ComposedFlow,
  inputs: Readonly<Record<string, unknown>>,
  options: RunFlowOptions,
  runDefaults: Defaults,
): Promise<RunResult> {
  const { log = null, runId = null, numWorkers = 8, onEvent = null, onStream = null, memory = null } = options;
  const { learnable = false, tools = null, skills = null, steering = null, context, signal = null } = options;
  const random = options.random ?? null;
  const scheduler = options.scheduler ?? realScheduler;
  const models = options.models ?? defaultModelAccess(options.env ?? processEnv);
  const runTools = new ToolSet(tools, { context });

  // Identity belongs to the authored program, before cascade resolution or any per-run copy can
  // change object topology.
  const authoredFingerprint = flowFingerprint(flow.compiled);
  const privateFlow = copyGraph(flow.compiled);
  // Held for a later explicit backward call.
  const retainedProgram = learnable ? copyGraph(privateFlow) : null;

  // Convert supplied boundary values before RunStarted persists its input. This is the only
  // string conversion: a value crossing a CALL/MAP/LOOP edge is validated as is, and START stays
  // the authority for required/default handling and contract validation.
  const coerced = coerceInputs(flow.input, hostInputs(flow.input, inputs));
  for (const [name, value] of Object.entries(coerced)) rejectDeepNesting(value, `input ${reprText(name)}`);

  // The run-start LLM chain (host credentials, the connection snapshot, the cascade and the
  // call-time agents) is not ported yet, so only flows without AGENT nodes run.
  const effectiveLlm: Record<string, Record<string, unknown>> = {};
  const callTimeLlm = null;

  const canonicalRunId = runId ?? defaultRunId();
  const pool = new VariablePool(privateFlow.startId);
  pool.set(privateFlow.startId, coerced);
  seedSystemClock(pool, scheduler.now());
  pool.addSystem("run_id", canonicalRunId);
  const startWiringOverride = Object.fromEntries(Object.keys(coerced).map((name) => [name, `\${input.${name}}`]));

  if (memory !== null) throw new Error("memory is not ported yet");

  let authority: RunLog | null = null;
  let primary: { error: unknown } | null = null;
  try {
    // The result reads its files through the contract a CALL of this flow gets.
    const state = new RunState(privateFlow, { resultContract: childSignature(flow).output });
    const preparedStart = prepareRunStarted(state, {
      runId: canonicalRunId,
      flowFingerprint: authoredFingerprint,
      initialPool: pool,
      startWiringOverride,
      effectiveLlmByStaticPath: effectiveLlm,
      callTimeLlm,
      numWorkers,
      learnable,
      defaults: nonDefaults(runDefaults),
    });

    // Preparation precedes durable-file creation, so invalid initialization cannot leave an empty
    // path behind.
    authority = log === null ? new MemoryRunLog(canonicalRunId) : JsonlRunLog.create(resolve(log), canonicalRunId);
    const events: RunEvent[] = [];
    publishBoundaryObservers(commitPrepared(authority, state, preparedStart), events, onEvent);

    const engine = new FlowEngine(state, authority, {
      growBuilder: buildGrow,
      numWorkers,
      onEvent,
      onStream,
      tools: runTools,
      skills,
      steering,
      models,
      signal,
      scheduler,
      random,
    });
    events.push(...(await engine.run()));
    return resultFromState(state, log, events);
  } catch (error) {
    primary = { error };
    throw error;
  } finally {
    closeRunResources(authority, primary);
  }
}

/**
 * Publish an already-applied segment lead before retaining it in the result. Nothing is in flight
 * yet, so a throwing callback rejects the run with no stop record.
 */
function publishBoundaryObservers(observers: readonly RunEvent[], events: RunEvent[], onEvent: EventHook | null): void {
  for (const observer of observers) {
    onEvent?.(observer);
    events.push(observer);
  }
}

/**
 * Project one terminal reducer state into the public result.
 *
 * @throws Error when the state has no terminal result.
 */
function resultFromState(state: RunState, path: string | null, events: readonly RunEvent[]): RunResult {
  const terminal = state.result;
  const common = {
    input: copyValue(state.input) as Record<string, unknown>,
    runId: requiredStateRunId(state),
    log: path,
    events,
    interruptedAttempts: new Map(state.interruptedAttempts),
    backwardOutcome: state.backwardState.outcome,
  };
  if (terminal instanceof RunSucceeded) {
    return new RunResult({ ...common, status: "succeeded", output: copyValue(terminal.output as Value) });
  }
  if (terminal instanceof RunFailed) return new RunResult({ ...common, status: "failed", failure: terminal.failure });
  throw new Error("engine returned without deriving a terminal run result");
}

/**
 * Return the initialized non-empty run identity a `RunResult` carries.
 *
 * @throws RangeError when the state has no valid run id.
 */
function requiredStateRunId(state: RunState): string {
  if (typeof state.runId !== "string" || state.runId === "") {
    throw new RangeError("terminal run state has no valid run_id");
  }
  return state.runId;
}

/**
 * Close one boundary-owned log while preserving an initiating error: with no `primary`, a close
 * error is thrown; with one, it is noted on it.
 */
function closeOwnedLog(log: RunLog, primary: { error: unknown } | null): void {
  try {
    log.close();
  } catch (cleanupError) {
    if (primary === null) throw cleanupError;
    noteCleanupError(primary.error, "run-log close", cleanupError);
  }
}

/** Close the run's log, once created, while preserving one initiating error. */
function closeRunResources(log: RunLog | null, primary: { error: unknown } | null): void {
  if (log !== null) closeOwnedLog(log, primary);
}

/** Note a cleanup error on the initiating error, when it can take a note, without changing control. */
function noteCleanupError(primaryError: unknown, operation: string, cleanupError: unknown): void {
  addNote(primaryError, () => {
    const [name, message] =
      cleanupError instanceof Error
        ? [cleanupError.name, cleanupError.message]
        : [jsTypeName(cleanupError), String(cleanupError)];
    return `${operation} also raised ${name}: ${message}`;
  });
}
