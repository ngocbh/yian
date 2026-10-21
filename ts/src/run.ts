/**
 * The package-root run boundary over one authoritative append-only log.
 *
 * `runFlow` initializes a private copy of a loaded flow, commits its complete canonical start
 * state, and drives only post-durable observer notifications. `resumeFlow` replays a durable log
 * silently onto a private copy and drives the unfinished rest the same way.
 *
 * The runtime borrows an already-open log. This module owns log creation and closure after the
 * engine's structured drain. It also owns the compile/runtime inversion seam by injecting
 * `compile/expand.ts`'s `buildGrow` into the kind-blind engine.
 *
 * This is the `yian/run` module, as Python's `yian.run`: it exports `runFlow`, `resumeFlow` and
 * `RunResult`, and re-exports `BackwardOutcome` (`runtime/contracts.ts`) and `Steering`
 * (`nodes/base.ts`). It also exports `inspectRun`, which reads where a run log stands.
 *
 * Knows about:   compile, compose, defaults, diagnostics, events, expr, llm_clients, memory contracts,
 *                memory_host (the registry type check only), nodes, platform, runtime, skills, tools,
 *                typesys.
 * Never imports: the CLI.
 */

import { randomUUID } from "node:crypto";
import { resolve } from "node:path";
import { types } from "node:util";

import { applyExecutionLayout, resolveExecutionLayout } from "./compile/execution_layout.ts";
import { buildGrow } from "./compile/expand.ts";
import {
  applyEffectiveLlmMap,
  bindHostCredentials,
  callTimeSelection,
  hostKeyProviders,
  inFlightLlmConfigs,
  knownCallTimeProviders,
  resolveRunLlm,
  type RunLLM,
  staticAgentLabels,
  valueConfigSubjects,
} from "./compile/llm_cascade.ts";
import { flowFingerprint } from "./compile/manifest.ts";
import {
  applyMemoryLayout,
  checkSessionCompatibility,
  memoryStacks,
  type ResolvedMemoryLayout,
  resolveMemoryLayout,
} from "./compile/memory_layout.ts";
import { type CompiledFlow, copyGraph, END_ID, type Flow } from "./compile/model.ts";
import { typeRelation } from "./compile/validation.ts";
import { ComposedFlow } from "./compose/build/pipeline.ts";
import { CLOSE_MATCH_CUTOFF } from "./compose/helpers/tags.ts";
import { childSignature } from "./compose/nodes/resolver.ts";
import { checkRecorded, current, type Defaults, type Env, nonDefaults, processEnv, withEnv } from "./defaults.ts";
import { compareCodeUnits, type Failure, getCloseMatches, type OriginRef, reprText } from "./diagnostics.ts";
import { type RunEvent, RunFailed, RunSucceeded, type StreamChunk } from "./events.ts";
import { ExpressionError } from "./expr/expressions.ts";
import type { FeedbackValue } from "./expr/lineage.ts";
import { checkTemplateCalls } from "./expr/template.ts";
import { LLMConfig, popHostCredentials } from "./llm_clients/config.ts";
import {
  type AgentConfig,
  blockProblem,
  checkModelConfig,
  type ConnectionSnapshot,
  connectionSnapshot,
  HEADER_NAMES_KEY,
  HOST_KEY_MARKER,
  headerNameDiff,
  legacyConnectionConfig,
  legacyConnectionMap,
  refusedBlock,
  type SelectionRecord,
  splitModelConfig,
} from "./llm_clients/connection.ts";
import { validateHeaders } from "./llm_clients/connection_settings.ts";
import { defaultModelAccess, type ModelAccess } from "./llm_clients/factory.ts";
import {
  type LearningResult,
  type Memory,
  MemorySetupError,
  type MemorySpec,
  type MemoryStack,
  sameMemorySpec,
} from "./memory/contracts.ts";
import { MemoryRegistry } from "./memory_host.ts";
import type { Node, Steering } from "./nodes/base.ts";
import { applyDefaults, BindingError, checkInputNames, coerceInputs, copyValue } from "./nodes/binding.ts";
import {
  type HostToolCall,
  isExactly,
  iterSourceStrings,
  type OutputContract,
  type Parameter,
  sameItems,
  sameOptional,
} from "./nodes/contracts.ts";
import { realScheduler, type Scheduler } from "./platform.ts";
import { driveBackward } from "./runtime/backward.ts";
import type { BackwardOutcome } from "./runtime/contracts.ts";
import { FlowEngine, resumedAttemptEvents } from "./runtime/engine.ts";
import { addNote, requireModels, requireSteering } from "./runtime/eval_node.ts";
import {
  decodeRecord,
  encodeRecord,
  isRunLogLocked,
  JsonlRunLog,
  MemoryRunLog,
  NodeExpandedRecord,
  NodeFailedRecord,
  NodeRoutedRecord,
  NodeStartedRecord,
  NodeSucceededRecord,
  readRunLog,
  type RecordEnvelope,
  type RunLog,
  RunResumedRecord,
  RunStartedRecord,
  RunStoppedRecord,
} from "./runtime/log.ts";
import { MemoryRuntime } from "./runtime/memory_runtime.ts";
import {
  addInterruptedAttempts,
  commitPrepared,
  prepareRunResumed,
  prepareRunStarted,
  RunState,
  replayRecords,
} from "./runtime/state.ts";
import { Skill } from "./skills/catalog.ts";
import { AlreadyOpenError, isMcpServer, type McpServer } from "./tools/mcp.ts";
import { type ToolEntry, ToolSet } from "./tools/registry.ts";
import { toolTypes } from "./tools/types.ts";
import { VariablePool } from "./typesys/pool.ts";
import { defaultRunId, seedSystemClock } from "./typesys/seeding.ts";
import {
  canonicalTypeLabel,
  fromHost,
  isRecord,
  jsTypeName,
  rejectDeepNesting,
  setField,
  type Type,
  TypeCheckError,
  type Value,
  ValueKind,
  type ValueRecord,
  writeAsText,
} from "./typesys/values.ts";

export type { Steering } from "./nodes/base.ts";
export { BackwardOutcome } from "./runtime/contracts.ts";

/** Sees each durable observer, and each in-loop model and tool event. */
export type EventHook = (event: RunEvent) => unknown;
/** Sees each transient stream chunk. */
export type StreamHook = (chunk: StreamChunk) => unknown;

/**
 * The terminal, surface-independent result of a run.
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
  /** The JSONL log path as given to `runFlow` or `resumeFlow`, or `null` for a process-local run. */
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

  /**
   * Run one new explicit backward attempt over the retained accepted execution.
   *
   * Each call appends a new attempt, whatever earlier attempts did; an unfinished earlier attempt
   * (from a killed process, say) is first recorded as interrupted. The forward fields never change.
   *
   * @param feedback Ordinary text or structured feedback on the flow's output, taken through host
   *   intake (`{n: 3}` holds the int `3n`).
   * @param options.apiKey The provider key of a run started with `llmConfig.api_key`, for the same
   *   agents as in `runFlow`, never recorded; `null` (the default) uses the key this result was
   *   created with, if any. A run that marked no agent ignores it.
   * @param options.headers The header values sent with that key; their names must be the recorded
   *   ones, compared without case. `null` uses the headers this result was created with, if any.
   * @returns The accepted attempt's outcome, separate from the forward status, output, failure and
   *   events; `learning` is its projection.
   * @throws TypeError when the feedback has an invalid message carrier.
   * @throws TypeCheckError when the feedback isn't a yian value or nests deeper than
   *   `YIAN_MAX_VALUE_NESTING` levels.
   * @throws RangeError when the run wasn't retained (`learnable: true`), the feedback doesn't match
   *   the output's shape, or `headers` has a name or value that can't be sent or names other than
   *   the recorded ones (the message never echoes a value).
   * @throws MemorySetupError when the run's memories can't restore their recorded snapshots.
   * @throws Error when another backward call on this result is in progress, or an earlier call
   *   couldn't retain its accepted prefix; host and log faults also reject, keeping the accepted
   *   prefix.
   */
  async backward(
    feedback: unknown = null,
    options: { apiKey?: string | null; headers?: Readonly<Record<string, string>> | null } = {},
  ): Promise<BackwardOutcome> {
    const handle = BACKWARD_HANDLES.get(this);
    if (handle === undefined) throw new RangeError("backward requires a run retained with learnable: true");
    // Taken and bounded before anything walks the feedback recursively.
    const taken = fromHost(feedback, "feedback");
    rejectDeepNesting(taken, "feedback");
    if (handle.busy) throw new Error("a backward call on this result is already in progress");
    const { apiKey = null, headers = null } = options;
    handle.busy = true;
    try {
      return await handle.backward(taken as FeedbackValue, {
        apiKey: apiKey === null ? null : secretKey(apiKey),
        headers: headers === null ? null : secretHeaders(headers),
      });
    } finally {
      this.#backwardOutcome = handle.outcome;
      handle.busy = false;
    }
  }
}

/** The retained authority behind each learnable `RunResult`, found by its result. */
const BACKWARD_HANDLES = new WeakMap<RunResult, BackwardHandle>();

/** What a `BackwardHandle` keeps. */
type BackwardHandleFields = {
  readonly program: CompiledFlow;
  readonly runId: string;
  /** The absolute log path captured at the run, or `null` for a process-local log. */
  readonly path: string | null;
  /** A process-local log's records; empty for a durable log. */
  records: readonly RecordEnvelope[];
  readonly memory: ReadonlyMap<string, Memory> | null;
  /** The last accepted backward outcome this handle observed. */
  outcome: BackwardOutcome | null;
  /** The host key, held for this result's own backward calls and never recorded. */
  readonly apiKey: string | null;
  /** The header values sent with the host key, held and never recorded as the key is. */
  readonly headers: Readonly<Record<string, string>> | null;
  /** The run's environment: the replay checks the recorded `YIAN_*` values against it. */
  readonly env: Env;
  readonly models: ModelAccess;
};

// The fields are declared once, in `BackwardHandleFields`, and merged into the class.
interface BackwardHandle extends BackwardHandleFields {}

/**
 * The private original program and closed append authority a learnable result keeps for later
 * feedback: the log's path, or a process-local log's records.
 */
class BackwardHandle {
  /** Whether a backward call on the result is running. */
  busy = false;
  #unusable = false;

  constructor(fields: BackwardHandleFields) {
    Object.assign(this, fields);
  }

  /**
   * Reopen and replay the log, then drive one whole backward attempt over it.
   *
   * @throws Error when an earlier call couldn't retain its accepted prefix, and what the replay or
   *   `driveBackward` throws.
   */
  async backward(
    feedback: FeedbackValue,
    options: { apiKey: string | null; headers: Readonly<Record<string, string>> | null },
  ): Promise<BackwardOutcome> {
    if (this.#unusable) {
      throw new Error("backward handle is unavailable after an accepted-prefix refresh failure");
    }
    // The call comes after the run's scope ended, so the replay's `checkRecorded` needs its own.
    return withEnv(this.env, async () => {
      const authority = this.path === null ? MemoryRunLog.fromRecords(this.records) : JsonlRunLog.resume(this.path);
      let state: RunState | null = null;
      let replaySequence: number | null = null;
      let primary: { error: unknown } | null = null;
      try {
        state = replayCompiledRunLog(this.program, authority, {
          apiKey: options.apiKey ?? this.apiKey,
          headers: options.headers ?? this.headers,
        });
        if (state.runId !== this.runId) throw new RangeError("backward log identity differs from the retained run");
        const callTime = state.callTimeLlm as CallTimeLlm | null;
        if (options.headers !== null && usesHostKey(state.effectiveLlmByStaticPath, callTime)) {
          // Before the attempt is appended; kept headers were checked when kept.
          checkHostHeaderNames(state.effectiveLlmByStaticPath, options.headers, callTime);
        }
        replaySequence = state.nextSequence;
        const replayed = state;
        const openMemory = (): Promise<MemoryRuntime> => {
          if (this.memory === null) throw new MemorySetupError("backward requires this run's enabled memory mapping");
          // `driveBackward` opens memory only when the run recorded a memory table.
          const memories = replayed.memories!;
          checkRegistryTypes(this.memory, memories);
          const program = copyGraph(this.program);
          applyEffectiveLlmMap(program, legacyConnectionMap(replayed.effectiveLlmByStaticPath), replayed.callTimeLlm);
          const layout = resolveRuntimeMemoryLayout(program);
          checkRecordedMemoryTable(memories, replayed.memoryStacks, layout.memories, layout.stacks, "retained program");
          return MemoryRuntime.open({
            memories,
            origins: layout.origins,
            memory: this.memory,
            snapshots: replayed.memorySnapshots,
          });
        };
        return await driveBackward(state, authority, feedback, { openMemory, models: this.models });
      } catch (error) {
        primary = { error };
        throw error;
      } finally {
        try {
          if (replaySequence !== null) {
            try {
              this.#refresh(authority, state!, replaySequence);
            } catch (refreshError) {
              this.#unusable = true;
              if (primary === null) {
                primary = { error: refreshError };
                // With no error in flight, the refresh error is the call's error.
                throw refreshError;
              }
              noteCleanupError(primary.error, "backward prefix refresh", refreshError);
            }
          }
        } finally {
          closeOwnedLog(authority, primary);
        }
      }
    });
  }

  /**
   * Retain the accepted facts, replaying an append that outpaced the local reduction.
   *
   * @throws RangeError when the log's sequence went back or the log changed while it was read, or
   *   the log holds another run.
   */
  #refresh(authority: RunLog, state: RunState, replaySequence: number): void {
    const sequence = authority.nextSequence;
    if (!Number.isSafeInteger(sequence) || sequence < state.nextSequence) {
      throw new RangeError("backward log sequence regressed during the call");
    }
    // Merely reading another handle's work must keep this result's observation when its own call
    // accepted no new record.
    if (sequence === replaySequence) return;
    if (state.nextSequence > replaySequence) this.outcome = state.backwardState.outcome;
    const records = authority.records();
    if (authority.nextSequence !== sequence || records.length !== sequence) {
      throw new RangeError("backward log changed during prefix refresh");
    }
    if (this.path === null) this.records = records;
    if (sequence > state.nextSequence) {
      const accepted = replayCompiledRunLog(this.program, authority);
      if (accepted.runId !== this.runId) throw new RangeError("backward log identity differs from the retained run");
      this.outcome = accepted.backwardState.outcome;
    }
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
   * events. Called synchronously; a throw, or a rejection of the promise it returns, rejects the run
   * with that error.
   */
  onEvent?: EventHook | null;
  /** Sees each transient stream chunk; called as `onEvent` is. */
  onStream?: StreamHook | null;
  /**
   * The per-run outer model configuration layer, taken through host intake (`{max_tokens: 1000}`
   * holds the int `1000n`). Its `api_key` goes only to the agents of its `provider`, with its
   * `base_url`, `organization`, `project` and `headers`; none of them is recorded.
   */
  llmConfig?: Readonly<Record<string, unknown>> | null;
  /**
   * The backend of each declared memory key, a `Map`, typically a `MemoryRegistry`. Each key's
   * session opens before the log and closes when the run ends.
   */
  memory?: ReadonlyMap<string, Memory> | null;
  /** Retain accepted execution for a later explicit backward call (default `false`). */
  learnable?: boolean;
  /**
   * The tools this run sees, keyed by tool id; `null` (the default) uses the registered tools.
   * A record, even an empty one, is the whole set. An `McpServer` that isn't open is opened for
   * this run and closed when it ends; one the host opened stays open.
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
  /**
   * The environment for connection selection, credentials, model building and the run's `YIAN_*`
   * values. Without one, those read `process.env`, and the `YIAN_*` values are the ones already in
   * effect: an enclosing `withEnv` scope's, else this process's.
   */
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
 *   isn't an `AbortSignal`, `inputs` isn't a record (a Proxy included), `llmConfig` is a primitive
 *   or an array, a `tools` entry is malformed, `skills` isn't a `Map` of `Skill`s each keyed by its
 *   name, `steering` lacks its methods, `scheduler` lacks `now` or `sleep`, `random` isn't a
 *   function, `models` lacks `env` or `createModel`, or `memory` isn't a `Map` of memories.
 * @throws ExpressionError when a condition, binding or prompt set in code can't run.
 * @throws TypeCheckError when an input holds an accessor, NaN, a lone surrogate, a cycle or
 *   another non-yian value, nests deeper than `YIAN_MAX_VALUE_NESTING` levels, or has text that
 *   can't be read; or when `llmConfig` is or holds a non-yian value (a `Map`, a class instance, a
 *   Proxy).
 * @throws RangeError when an input key names no declared input; an AGENT grants a tool, group or
 *   skill this run lacks; a TOOL node names a tool this run lacks, or its inputs or output can't
 *   fit that tool's types; `llmConfig` or an agent's resolved model config is invalid, an agent
 *   has no provider or model, or `llmConfig.api_key` has no `provider`; or the identity, worker
 *   count or prepared start state breaks its contract; or a `tools` server was opened by
 *   another run in progress. A message never echoes the key.
 * @throws MemorySetupError when the flow's memory layout breaks a rule, `memory` lacks a declared
 *   key or is a `MemoryRegistry` built for another flow, or a memory fails to open its session.
 * @throws McpConnectError when a `tools` server doesn't start or doesn't list its tools.
 * @throws AuthError when the credential store can't be read.
 * @throws DefaultsError when a `YIAN_<NAME>` variable in `env` holds a value its entry doesn't accept.
 * @throws The signal's reason when it aborts.
 * @throws Error (code `EEXIST`) when `log` names an existing path, and the `fs` error when the log
 *   can't be created, locked, written, synced or closed.
 */
export async function runFlow(
  flow: ComposedFlow,
  inputs: Readonly<Record<string, unknown>>,
  options: RunFlowOptions = {},
): Promise<RunResult> {
  // The scope makes everything the run reads, the engine's work included, see the run's values;
  // a bad YIAN_<NAME> value fails here, before any work.
  const env = options.env ?? processEnv;
  return withEnv(env, async () => {
    const runDefaults = current();
    const { learnable = false, tools = null, steering = null, context, signal = null } = options;
    if (!(flow instanceof ComposedFlow)) throw new TypeError("runFlow expects a ComposedFlow");
    if (typeof learnable !== "boolean") throw new TypeError("learnable must be a boolean");
    if (signal !== null && !(signal instanceof AbortSignal)) {
      throw new TypeError("signal must be an AbortSignal or null");
    }
    signal?.throwIfAborted();
    const [otherTools, unopened] = splitRunServers(tools);
    // The other entries' errors, and the checks that need no tools, come before any server starts.
    new ToolSet(otherTools, { context });
    checkConditions(flow.compiled);
    checkTemplates(flow.compiled);
    const skills = checkSkills(flow.compiled, options.skills ?? null);
    requireSteering(steering);
    checkTestOptions(options);
    const runServers = await openRunServers(unopened, signal);
    try {
      return await startRun(flow, inputs, { ...options, skills }, runDefaults, env);
    } finally {
      await closeRunServers(runServers);
    }
  });
}

/**
 * Check the options for tests before anything is created, as the engine checks them.
 *
 * @throws TypeError when `scheduler` lacks `now` or `sleep`, `random` isn't a function, or
 *   `models` lacks `env` or `createModel`.
 */
function checkTestOptions(options: Pick<RunFlowOptions, "scheduler" | "random" | "models">): void {
  const { scheduler = null, random = null, models = null } = options;
  if (
    scheduler !== null &&
    (typeof scheduler !== "object" || typeof scheduler.now !== "function" || typeof scheduler.sleep !== "function")
  ) {
    throw new TypeError("scheduler must be a Scheduler or null");
  }
  if (random !== null && typeof random !== "function") throw new TypeError("random must be a function or null");
  requireModels(models);
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

/** A run's `RunStarted.call_time_llm`: the call-time agents' bases and the selection they bind from. */
type CallTimeLlm = {
  selection: SelectionRecord;
  agents: Record<string, AgentConfig>;
};

/**
 * Return host header values once every name and value may be sent, in their given order.
 *
 * @throws RangeError when a name or value is refused, or there are too many headers; the message
 *   never echoes a value.
 */
function secretHeaders(raw: unknown): Record<string, string> {
  return validateHeaders(raw, null);
}

/**
 * Return whether a run's recorded agents may use the host key: a static agent is marked for it, or
 * the call-time agents' selection names its provider.
 *
 * @param effective The recorded effective model map, keyed by static agent path.
 * @param callTime The recorded `RunStarted.call_time_llm`, or `null`.
 */
function usesHostKey(
  effective: Readonly<Record<string, Readonly<AgentConfig>>>,
  callTime: Readonly<CallTimeLlm> | null,
): boolean {
  return hostKeyProviders(effective).size > 0 || (callTime !== null && callTime.selection.host !== null);
}

/**
 * Return the host key the call-time agents' selection names, for a resume check.
 *
 * A call-time agent that has not started may still resolve to that key's provider, so a resume
 * that can't supply the key is refused up front, as for a marked agent.
 *
 * @param callTime The recorded `RunStarted.call_time_llm`, or `null`.
 * @returns `{provider, header_names}` (header names only when recorded), or `null` when the
 *   selection names no host key.
 */
function selectionHost(callTime: Readonly<CallTimeLlm> | null): Record<string, unknown> | null {
  const host = callTime === null ? null : callTime.selection.host;
  if (host === null) return null;
  const selected: Record<string, unknown> = { provider: host["provider"] };
  const names = host["header_names"] as readonly string[] | undefined;
  if (names !== undefined && names.length > 0) selected["header_names"] = [...names];
  return selected;
}

/**
 * Check that the host's headers have the names recorded for its key's agents; names compare
 * without case, as HTTP compares them.
 *
 * @param effective The recorded effective model map, keyed by static agent path.
 * @param headers The headers given for the host key; `null` gives none.
 * @param callTime The recorded `RunStarted.call_time_llm`; the header names its selection records
 *   for the host key are expected too.
 * @throws RangeError when the names differ; the message names the missing and unrecorded header
 *   names, never a value.
 */
function checkHostHeaderNames(
  effective: Readonly<Record<string, Readonly<AgentConfig>>>,
  headers: Readonly<Record<string, unknown>> | null,
  callTime: Readonly<CallTimeLlm> | null = null,
): void {
  const expected: string[] = [];
  for (const config of Object.values(effective)) {
    if (config[HOST_KEY_MARKER] === true) expected.push(...((config[HEADER_NAMES_KEY] as string[] | undefined) ?? []));
  }
  const host = callTime === null ? null : callTime.selection.host;
  if (host !== null) expected.push(...((host["header_names"] as string[] | undefined) ?? []));
  const diff = headerNameDiff(expected, Object.keys(headers ?? {}));
  if (diff !== null) {
    throw new RangeError(`headers must have the header names this run recorded for its host key: ${diff}`);
  }
}

/** Validate a host key without echoing it in errors. */
function secretKey(raw: string): string {
  return new LLMConfig({ api_key: raw }).apiKey!;
}

/**
 * Reject any agent config its model call would reject, before the run starts.
 *
 * @param effectiveLlm The effective model map, keyed by static agent path.
 * @param apiKey The host key, supplied to each config marked for it.
 * @param headers The header values sent with the host key, supplied beside it.
 * @param labels The authored site of each synthesized agent's `llm_config`, keyed by static agent
 *   path; a message names it instead of the path.
 * @param located The `llm_config` subject of each agent whose layers carry values
 *   (`valueConfigSubjects`), keyed by static agent path; its message names the subject, and the
 *   provider's block when the block is what is refused.
 * @throws RangeError when a config would be rejected by its model call; the message names its agent
 *   path, or the authored site of a synthesized agent.
 */
function checkEffectiveLlm(
  effectiveLlm: Readonly<Record<string, Readonly<AgentConfig>>>,
  apiKey: string | null,
  headers: Readonly<Record<string, string>> | null,
  labels: ReadonlyMap<string, string>,
  located: ReadonlyMap<string, string> | null = null,
): void {
  for (const [path, config] of Object.entries(effectiveLlm)) {
    // The host key and header values stay out of the map; supply them where the marker expects
    // them.
    let probe = config;
    if (HOST_KEY_MARKER in config) {
      const supplied: AgentConfig = { ...config, api_key: apiKey };
      if (headers !== null && Object.keys(headers).length > 0) supplied["headers"] = headers;
      probe = supplied;
    }
    try {
      checkModelConfig(probe);
    } catch (error) {
      if (!(error instanceof RangeError)) throw error;
      const subject = located?.get(path);
      if (subject !== undefined) {
        throw new RangeError(blockProblem(subject, refusedBlock(probe).slice(1), error.message), { cause: error });
      }
      const where = labels.get(path) ?? `Agent ${reprText(path)}`;
      throw new RangeError(`${where}: ${error.message}`, { cause: error });
    }
  }
}

/**
 * @internal Return the run's inputs as run start resolves against them: the supplied inputs plus
 * each omitted input's valid default.
 *
 * START still binds and checks the inputs, so a missing required input or a default that fails its
 * check stays START's logged failure: such a default is left out here rather than thrown before the
 * log exists.
 *
 * @param params The flow's declared inputs.
 * @param coerced The supplied inputs, already coerced at the run boundary.
 */
export function runStartInputs(
  params: readonly Parameter[],
  coerced: Readonly<Record<string, unknown>>,
): Record<string, unknown> {
  let runInputs: Record<string, unknown> = { ...coerced };
  for (const parameter of params) {
    try {
      runInputs = applyDefaults([parameter], runInputs);
    } catch (error) {
      if (!(error instanceof BindingError)) throw error;
    }
  }
  return runInputs;
}

/**
 * @internal Resolve a fresh run's llm cascade over `compiled`, in place, with its inputs.
 *
 * Values that read only the run's inputs resolve against the supplied inputs, with omitted inputs'
 * defaults filled.
 *
 * @param compiled The run's private flow copy; its agents get their effective configs.
 * @param coerced The supplied inputs, already coerced at the run boundary.
 * @param llmConfig The host layer, without the host key and its settings.
 * @param snapshot The one selection snapshot that determines every agent's defaults.
 * @returns The resolved configurations, before any connection is bound.
 * @throws RangeError as `resolveRunLlm` throws it.
 */
export function runStartLlm(
  compiled: CompiledFlow,
  coerced: Readonly<Record<string, unknown>>,
  llmConfig: Readonly<Record<string, unknown>>,
  snapshot: ConnectionSnapshot,
): RunLLM {
  return resolveRunLlm(compiled, llmConfig, {
    defaults: snapshot.defaults,
    providerModels: snapshot.providerModels,
    strayModel: snapshot.strayModel,
    runInputs: runStartInputs(compiled.nodes.get(compiled.startId)!.params ?? [], coerced),
  });
}

/** Check the tools, then record and drive a fresh run whose MCP servers are open. */
async function startRun(
  flow: ComposedFlow,
  inputs: Readonly<Record<string, unknown>>,
  options: RunFlowOptions,
  runDefaults: Defaults,
  env: Env,
): Promise<RunResult> {
  const { log = null, runId = null, numWorkers = 8, onEvent = null, onStream = null, memory = null } = options;
  const { learnable = false, tools = null, skills = null, steering = null, context, signal = null } = options;
  const random = options.random ?? null;
  const scheduler = options.scheduler ?? realScheduler;
  const models = options.models ?? defaultModelAccess(env);
  const runTools = new ToolSet(tools, { context });
  checkToolGrants(flow.compiled, runTools);
  checkToolTypes(flow.compiled, runTools);

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

  // The key stays out of the cascade so RunStarted never records it. Its base_url and connection
  // settings stay out too, so no authored layer can pair the key with another. Host intake makes
  // the copy they are taken from, with an integral number as an int.
  const runLlmConfig = fromHost(options.llmConfig ?? {}, "llm_config");
  if (!isRecord(runLlmConfig)) throw new TypeError("llmConfig must be a record or null");
  const host = popHostCredentials(runLlmConfig);
  let apiKey = host.apiKey;
  let headers = host.headers;

  // One selection snapshot determines every static agent before RunStarted, and what each agent
  // resolved only when it runs binds from.
  const snapshot = connectionSnapshot({ env });
  const hostKeyProvider = apiKey === null ? null : (runLlmConfig["provider"] as string);
  const runLlm = runStartLlm(privateFlow, coerced, runLlmConfig, snapshot);
  const effectiveLlm = snapshot.bind(runLlm.effective, {
    hostKeyProvider,
    hostBaseUrl: host.baseUrl,
    hostSettings: host,
  });
  // Each call-time agent's provider when run start already knows it: it limits the recorded
  // selection and the session check.
  const known = knownCallTimeProviders(privateFlow, runLlm.callTime, snapshot.defaults);
  let callTimeLlm: CallTimeLlm | null = null;
  if (Object.keys(runLlm.callTime).length > 0) {
    callTimeLlm = {
      selection: callTimeSelection(snapshot, { known, hostKeyProvider, hostBaseUrl: host.baseUrl, hostSettings: host }),
      agents: runLlm.callTime,
    };
  }
  applyEffectiveLlmMap(privateFlow, effectiveLlm, callTimeLlm);
  // A key that neither marks an agent nor is in the call-time agents' selection has nothing to
  // authenticate, so the run keeps neither it nor its headers.
  if (apiKey !== null && !usesHostKey(effectiveLlm, callTimeLlm)) {
    apiKey = null;
    headers = null;
  }
  if (apiKey !== null) bindHostCredentials(privateFlow, apiKey, headers);
  checkEffectiveLlm(effectiveLlm, apiKey, headers, staticAgentLabels(privateFlow), valueConfigSubjects(privateFlow));

  if (learnable) applyExecutionLayout(privateFlow, resolveExecutionLayout(privateFlow));

  const canonicalRunId = runId ?? defaultRunId();
  const pool = new VariablePool(privateFlow.startId);
  pool.set(privateFlow.startId, coerced);
  seedSystemClock(pool, scheduler.now());
  pool.addSystem("run_id", canonicalRunId);
  const startWiringOverride = Object.fromEntries(Object.keys(coerced).map((name) => [name, `\${input.${name}}`]));

  let memoryRuntime: MemoryRuntime | null = null;
  let layout: MemoryLayoutParts | null = null;
  if (memory !== null) {
    layout = resolveRuntimeMemoryLayout(privateFlow, effectiveLlm, known);
    checkRegistryTypes(memory, layout.memories);
    memoryRuntime = await MemoryRuntime.open({
      memories: layout.memories,
      origins: layout.origins,
      memory,
      snapshots: null,
    });
  }

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
      memorySnapshots: memoryRuntime?.snapshots ?? null,
      learnable,
      memories: layout?.memories ?? null,
      memoryStacks: layout?.stacks ?? null,
      defaults: nonDefaults(runDefaults),
    });

    // Taken once, so a later directory change can't move the log.
    const locator = log === null ? null : resolve(log);
    // Preparation precedes durable-file creation, so invalid initialization cannot leave an empty
    // path behind.
    authority = locator === null ? new MemoryRunLog(canonicalRunId) : JsonlRunLog.create(locator, canonicalRunId);
    const observers = commitPrepared(authority, state, preparedStart);
    const runtime = memoryRuntime;
    // Built before publishing so a callback's rejected promise can stop it.
    const engine = new FlowEngine(state, authority, {
      growBuilder: buildGrow,
      numWorkers,
      onEvent,
      onStream,
      // A bare `runtime.openContext` would lose its `this`.
      memoryContextOpener:
        runtime === null ? null : (scope, request, opts) => runtime.openContext(scope, request, opts),
      tools: runTools,
      skills,
      steering,
      models,
      signal,
      scheduler,
      random,
    });
    const events: RunEvent[] = [];
    publishBoundaryObservers(observers, events, onEvent, (error) => engine.stopForCallback(error));
    events.push(...(await engine.run()));
    const handle = retainedHandle(retainedProgram, state, authority, memory, {
      locator,
      apiKey,
      headers,
      env,
      models,
    });
    return resultFromState(state, log, events, handle);
  } catch (error) {
    primary = { error };
    throw error;
  } finally {
    await closeRunResources(memoryRuntime, authority, primary);
  }
}

/**
 * @internal A check of the recorded model configs before a resume appends anything (the CLI's
 * credential check). It gets a copy of the recorded effective map plus each started call-time
 * agent's recorded config keyed by its node id (old OpenAI records adapted to their legacy
 * connection), and as `callTimeHost` the host key the call-time agents' selection names. A throw,
 * or a rejection of the promise it returns, stops the resume and appends nothing.
 */
export type LlmConfigsCheck = (
  configs: Record<string, AgentConfig>,
  options: { callTimeHost: Record<string, unknown> | null },
) => unknown;

/** The options of one `resumeFlow` call. */
export interface ResumeFlowOptions {
  /** The existing JSONL run log to lock, repair to its last complete line, check and replay. */
  log: string;
  /**
   * The non-negative worker bound of the new segment; `null` (the default) keeps the last recorded
   * one.
   */
  numWorkers?: number | null;
  /**
   * Sees each observer committed by this call only, plus the transient in-loop AGENT events; the
   * replayed history and a finished log publish nothing. Called as `runFlow`'s is.
   */
  onEvent?: EventHook | null;
  /** Sees each new transient stream chunk; chunks are never replayed. */
  onStream?: StreamHook | null;
  /**
   * The backend of each declared memory key, a `Map`, typically a `MemoryRegistry`. Required when
   * the run recorded memory and has unfinished work; a finished log opens no memory.
   */
  memory?: ReadonlyMap<string, Memory> | null;
  /**
   * The tools the resumed segment sees, with `runFlow`'s rules. Tools are not recorded, so pass the
   * ones the run used (`ask_user` included). An unopened `McpServer` is opened only when the run is
   * unfinished, after the log is replayed, and closed when the resume ends; a finished log opens
   * none.
   */
  tools?: Readonly<Record<string, ToolEntry>> | null;
  /** The skills the run was given; an unfinished run of a flow that grants skills needs them. */
  skills?: ReadonlyMap<string, Skill> | null;
  /** The host inbox for AGENT nodes authored `steerable: true`. */
  steering?: Steering | null;
  /**
   * The provider key of a run started with `llmConfig.api_key`, for the agents the log marks for it
   * (and a call-time agent resolving to its provider); never recorded. A log that marks no agent
   * ignores it.
   */
  apiKey?: string | null;
  /**
   * The header values a run started with `llmConfig.headers` sent with its key; their names must
   * be the recorded ones, compared without case. Never recorded; a log that marks no agent ignores
   * them.
   */
  headers?: Readonly<Record<string, string>> | null;
  /** The host's app object a tool receives as `ctx.context`; the log doesn't hold it. */
  context?: unknown;
  /**
   * The host's stop: aborting it stops the resumed segment, which rejects with its reason. An
   * already-aborted signal rejects before the log is locked.
   */
  signal?: AbortSignal | null;
  /** The environment, as `runFlow`'s `env`; the run's recorded `YIAN_*` values must match it. */
  env?: Env | null;
  /** For tests: as `runFlow`'s `scheduler`. */
  scheduler?: Scheduler | null;
  /** For tests: as `runFlow`'s `random`. */
  random?: (() => number) | null;
  /** For tests: as `runFlow`'s `models`. */
  models?: ModelAccess | null;
  /** @internal See `LlmConfigsCheck`; not called for a finished log. */
  checkLlmConfigs?: LlmConfigsCheck | null;
}

/**
 * Recover a durable run, or read a finished one, without backward work.
 *
 * The recorded history is replayed silently on a private copy of `flow`; recovery never reads the
 * ambient model defaults, mints a new identity or takes new inputs or configuration. A finished
 * log returns its result with no events, opening no server and no memory. Otherwise the starts
 * the log left unfinished are retried first, in their recorded order, after one `RunResumed`
 * observer and one `NodeAttemptResumed` per retried node.
 *
 * @param flow The loaded flow whose fingerprint must match the log's `RunStarted`.
 * @param options See `ResumeFlowOptions`.
 * @returns The recorded terminal result, or the terminal result of the resumed segment with its
 *   post-durable observers.
 * @throws TypeError when `flow` isn't a `ComposedFlow`, `signal` isn't an `AbortSignal`, a `tools`
 *   entry is malformed, `skills`, `steering`, `scheduler`, `random` or `models` is malformed, a
 *   worker count isn't an integer, or a recorded carrier has the wrong type.
 * @throws ExpressionError as `runFlow`, for an unfinished run.
 * @throws RangeError when the log's prefix is invalid, the flow fingerprint differs, the worker
 *   count is negative, `apiKey` or `headers` is refused or their header names aren't the recorded
 *   ones (the message never echoes a value), a recorded agent config has an invalid field, the
 *   recorded memory table doesn't match the flow, or, for an unfinished run, a tool or skill
 *   check of `runFlow` fails, a `tools` server was opened by another run in progress, or a started
 *   call-time agent resolves to another config than the one it recorded (`node '<path>'
 *   llm_config: recorded config differs from the recomputed one; ...`), before its model call.
 * @throws AuthError when a call-time agent resolves to the provider of the run's host key and
 *   `apiKey` (or a recorded header) wasn't given, before its model call, or the credential store
 *   can't be read.
 * @throws MemorySetupError when the run recorded memory and has unfinished work, and `memory` is
 *   missing, lacks a key, is a `MemoryRegistry` built for another flow, or can't reproduce the
 *   recorded snapshots.
 * @throws McpConnectError when a `tools` server doesn't start or doesn't list its tools.
 * @throws DefaultsError when a `YIAN_<NAME>` variable in `env` holds a value its entry doesn't accept.
 * @throws DefaultsMismatchError when the run recorded `YIAN_*` values that differ from `env`'s.
 * @throws The signal's reason when it aborts.
 * @throws Error (code `ENOENT`) when `log` doesn't exist, and the lock, repair, append, sync or
 *   close error of the log.
 */
export async function resumeFlow(flow: ComposedFlow, options: ResumeFlowOptions): Promise<RunResult> {
  // As in `runFlow`, the scope covers everything the resume reads; a bad YIAN_<NAME> value fails
  // here, before the log is opened.
  const env = options.env ?? processEnv;
  return withEnv(env, async () => {
    current();
    const { log, tools = null, steering = null, context, signal = null } = options;
    if (!(flow instanceof ComposedFlow)) throw new TypeError("resumeFlow expects a ComposedFlow");
    if (signal !== null && !(signal instanceof AbortSignal)) {
      throw new TypeError("signal must be an AbortSignal or null");
    }
    signal?.throwIfAborted();
    const [otherTools, unopened] = splitRunServers(tools);
    // The other entries' errors come before the log is opened.
    new ToolSet(otherTools, { context });
    const { apiKey: rawKey = null, headers: rawHeaders = null } = options;
    let apiKey = rawKey === null ? null : secretKey(rawKey);
    let headers = rawHeaders === null ? null : secretHeaders(rawHeaders);
    requireSteering(steering);
    checkTestOptions(options);
    // The absolute path is taken once, so a later directory change can't move the log.
    const locator = resolve(log);
    const authority = JsonlRunLog.resume(locator);
    let memoryRuntime: MemoryRuntime | null = null;
    let runServers: McpServer[] = [];
    let primary: { error: unknown } | null = null;
    try {
      const state = replayRunLog(flow, authority, { apiKey, headers });
      const callTime = state.callTimeLlm as CallTimeLlm | null;
      if (!usesHostKey(state.effectiveLlmByStaticPath, callTime)) {
        apiKey = null;
        headers = null;
      } else if (apiKey !== null || headers !== null) {
        // Checked before the terminal return too, since a later backward call sends them.
        checkHostHeaderNames(state.effectiveLlmByStaticPath, headers, callTime);
      }
      const models = options.models ?? defaultModelAccess(env);
      const retainedProgram = state.learnable ? copyGraph(flow.compiled) : null;
      const retain = (): BackwardHandle | null =>
        retainedHandle(retainedProgram, state, authority, options.memory ?? null, {
          locator,
          apiKey,
          headers,
          env,
          models,
        });
      if (state.result !== null) return resultFromState(state, log, [], retain());
      // The checks that need no tools come before any server starts.
      checkConditions(flow.compiled);
      checkTemplates(flow.compiled);
      const skills = checkSkills(flow.compiled, options.skills ?? null);
      // A started call-time agent resumes with the config it recorded; one not yet started is
      // checked when it resolves.
      const inFlight = inFlightLlmConfigs(state.flow.nodes, state.nodeSteps);
      const recordedLlm = { ...state.effectiveLlmByStaticPath, ...inFlight };
      checkRecordedLlmFields(
        recordedLlm,
        staticAgentLabels(flow.compiled),
        staticPathKeys(state.flow.nodes, Object.keys(inFlight)),
      );
      if (options.checkLlmConfigs != null) {
        await options.checkLlmConfigs(legacyConnectionMap(recordedLlm), { callTimeHost: selectionHost(callTime) });
      }
      const memory = options.memory ?? null;
      if (state.memories !== null && memory === null) {
        throw new MemorySetupError("resuming this run requires its enabled memory mapping");
      }
      runServers = await openRunServers(unopened, signal);
      const runTools = new ToolSet(tools, { context });
      checkToolGrants(flow.compiled, runTools);
      checkToolTypes(flow.compiled, runTools);

      if (state.memories !== null) {
        checkRegistryTypes(memory!, state.memories);
        const authored = resolveAuthoredMemoryLayout(flow);
        const { memories, stacks } = authored;
        checkRecordedMemoryTable(state.memories, state.memoryStacks, memories, stacks, "supplied flow");
        memoryRuntime = await MemoryRuntime.open({
          memories: state.memories,
          origins: authored.origins,
          memory: memory!,
          snapshots: state.memorySnapshots,
        });
      }

      const numWorkers = options.numWorkers ?? state.numWorkers;
      const prepared = prepareRunResumed(state, { numWorkers });
      const observers = commitPrepared(authority, state, prepared);
      const runtime = memoryRuntime;
      // Built before publishing so a callback's rejected promise can stop it.
      const engine = new FlowEngine(state, authority, {
        growBuilder: buildGrow,
        numWorkers,
        onEvent: options.onEvent ?? null,
        onStream: options.onStream ?? null,
        // A bare `runtime.openContext` would lose its `this`.
        memoryContextOpener:
          runtime === null ? null : (scope, request, opts) => runtime.openContext(scope, request, opts),
        tools: runTools,
        skills,
        steering,
        models,
        signal,
        scheduler: options.scheduler ?? realScheduler,
        random: options.random ?? null,
      });
      const events: RunEvent[] = [];
      const lead = [...observers, ...resumedAttemptEvents(state, prepared.record as RunResumedRecord)];
      publishBoundaryObservers(lead, events, options.onEvent ?? null, (error) => engine.stopForCallback(error));
      events.push(...(await engine.run()));
      return resultFromState(state, log, events, retain());
    } catch (error) {
      primary = { error };
      throw error;
    } finally {
      await closeRunServers(runServers);
      await closeRunResources(memoryRuntime, authority, primary);
    }
  });
}

/** Where a run log stands, as `inspectRun` reads it. */
export type RunStatus = "running" | "succeeded" | "failed" | "stopped" | "interrupted";

/** What `inspectRun` reads from one run log. */
export interface RunInspection {
  /**
   * `running` while a writer holds the log's lock; `succeeded` or `failed` once the run finished;
   * `stopped` when the host stopped the last segment; otherwise `interrupted` (the process ended
   * without a stop). A run that isn't running or finished can be resumed.
   */
  readonly status: RunStatus;
  /** The nodes started and not settled, in start order; `[]` for a finished run. */
  readonly inFlight: readonly string[];
  /**
   * By node id, the starts a resume retried with no stop covering them, plus, for a run that is
   * neither running nor finished, one for each node in flight that no stop covers: the attempts a
   * resume would retry as interrupted.
   */
  readonly interruptedAttempts: ReadonlyMap<string, number>;
}

/**
 * Read where the run in `log` stands without changing the file or taking its lock.
 *
 * In-flight nodes and interrupted counts come from the records. Without `flow`, the status is
 * read from the records too: `failed` after a node failure; `succeeded` after the top-level END
 * succeeded, and `failed` after a route proved END unreachable, each with nothing in flight. The
 * records don't hold the ready nodes, so a run whose process ended after END succeeded (or after
 * that proof) but before a node already ready started reads as finished, though `resumeFlow` would
 * still run that node. With `flow`, the log is replayed as `resumeFlow` replays it, so the status
 * is exact; pass the flow when you have it. The lock is probed before and after the records are
 * read, so a writer that holds it at either probe makes the run `running`.
 *
 * @param log The JSONL run log's path.
 * @param options.flow The loaded flow the run was started with, for an exact status.
 * @param options.env The environment whose `YIAN_*` values the replay with `flow` checks the
 *   recorded ones against, as `resumeFlow`'s `env`; without one, the values already in effect.
 *   Reading without `flow` doesn't use it.
 * @returns The status, the nodes in flight and the interrupted counts.
 * @throws Error (code `ENOENT`) when `log` doesn't exist, and the `fs` error of a failed read.
 * @throws RangeError when a complete line isn't a valid record, the log holds no record and no
 *   writer, or, with `flow`, the replay refuses the log (`replayRunLog`).
 * @throws DefaultsError when a `YIAN_<NAME>` variable in `env` holds a value its entry doesn't accept.
 * @throws DefaultsMismatchError with `flow`, when the run recorded `YIAN_*` values that differ
 *   from `env`'s.
 */
export function inspectRun(
  log: string,
  options: { flow?: ComposedFlow | null; env?: Env | null } = {},
): RunInspection {
  const lockedBefore = isRunLogLocked(log);
  const envelopes = readRunLog(log);
  // A writer that took or released the lock while the records were read is caught by one probe.
  const running = isRunLogLocked(log) || lockedBefore;
  if (envelopes.length === 0 && !running) throw new RangeError(`cannot inspect an empty run log: ${log}`);
  const flow = options.flow ?? null;
  // A running log's status needs no replay.
  const source =
    running || flow === null
      ? scanRunRecords(envelopes)
      : withEnv(options.env ?? processEnv, () => replayedScan(flow, envelopes));
  const { starts, interrupted, lastStop, finished } = source;
  let status: RunStatus;
  if (running) status = "running";
  else if (finished !== null) status = finished;
  else status = lastStop === null ? "interrupted" : "stopped";
  // A running run's attempts are still running, and a finished run's are abandoned.
  if (status === "interrupted" || status === "stopped") addInterruptedAttempts(interrupted, starts, lastStop);
  const inFlight = status === "succeeded" || status === "failed" ? [] : [...starts.keys()];
  return { status, inFlight, interruptedAttempts: interrupted };
}

/**
 * Prove every static AGENT grant and TOOL `tool_id` names a tool this run supplies.
 *
 * An AGENT grant may name a tool group; a TOOL node must name one tool. Every static node is
 * checked, including nodes that may never run, so an unresolvable id rejects the run before any
 * record is written instead of failing its node when the node starts.
 *
 * @throws RangeError when an AGENT grants a tool or group this run lacks, or `ask_user` with no
 *   handler; or a TOOL node names a tool this run lacks, a tool group, or `ask_user` with no
 *   handler.
 */
function checkToolGrants(flow: CompiledFlow, tools: ToolSet): void {
  for (const [nodeId, node] of staticNodes(flow)) {
    for (const toolId of tools.expand(node.grantedTools())) {
      try {
        tools.resolve([toolId]);
      } catch (error) {
        if (!(error instanceof RangeError)) throw error;
        const reason =
          toolId === "ask_user"
            ? `'ask_user': ${error.message}`
            : `unknown tool or group ${reprText(toolId)}; tool groups in this run: ${writeAsText([...tools.groups])}`;
        throw new RangeError(`AGENT node ${reprText(nodeId)} grants ${reason}`);
      }
    }
    const call = node.hostToolCall();
    if (call === null) continue;
    try {
      tools.resolve([call.toolId]);
    } catch (error) {
      if (!(error instanceof RangeError)) throw error;
      let reason: string;
      if (call.toolId === "ask_user" || tools.groups.includes(call.toolId)) {
        reason = `${reprText(call.toolId)}: ${error.message}`;
      } else {
        const known = tools.toolIds;
        reason =
          `tool ${reprText(call.toolId)}, which isn't in this run's tools; ` +
          `known tools: ${writeAsText([...known])}`;
        const [close] = getCloseMatches(call.toolId, known, 1, CLOSE_MATCH_CUTOFF);
        if (close !== undefined) reason += `; did you mean ${reprText(close)}?`;
      }
      throw new RangeError(`TOOL node ${reprText(nodeId)} names ${reason}`);
    }
  }
}

/**
 * Reject a condition that doesn't parse on any static node, before any record is written.
 *
 * A condition is an assertion, a CASE `when`, or a LOOP predicate. A YAML flow can't hold one that
 * doesn't parse (Compose rejects it at load); this catches conditions set in code.
 *
 * @throws ExpressionError when a condition doesn't parse; the message names the node's path.
 */
function checkConditions(flow: CompiledFlow): void {
  for (const [nodePath, node] of staticNodes(flow)) node.validateConditions({ path: nodePath });
}

/**
 * Reject a binding or prompt span that can't run, before any record is written.
 *
 * Checks every static flow level: each template string of a wiring source (END's is the flow's
 * output) must parse, pass each builtin arguments it takes, and name only constructors in the
 * type table the binder evaluates it with (the reading node's; none for a reserved key such as
 * WAIT `until` or MAP `over`); each node checks its prompts. A YAML flow can't hold such a span
 * (Compose rejects it at load); this catches values set in code.
 *
 * @param flow A compiled flow or a static child flow.
 * @param path The static path of the node that owns `flow`; empty for the top level.
 * @throws ExpressionError when a span can't run; the message names the node and binding, the flow
 *   output, or the prompt field.
 */
function checkTemplates(flow: Flow, path: readonly string[] = []): void {
  const owner = path.length > 0 ? `node ${reprText(path.join("/"))} ` : "";
  for (const [nodeId, node] of flow.nodes) {
    const nodePath = [...path, nodeId].join("/");
    const reserved = node.reservedWiringKeys();
    for (const [key, binding] of flow.wiring.get(nodeId) ?? []) {
      let subject: string;
      if (nodeId === flow.endId) {
        subject = `${owner}${node.bindingLabels.get(key) ?? `flow output ${reprText(key)}`}`;
      } else if (node.bindingLabels.has(key) && !node.wiringLabels.has(key)) {
        // A desugared param (an inline call's argument, a value entry's input) reads as its
        // authored site; a CASE param keeps its wiring site name.
        subject = `${owner}${node.bindingLabels.get(key)}`;
      } else if (reserved.has(key)) {
        subject = `node ${reprText(nodePath)} ${node.reservedKeyLabel(key)}`;
      } else if (node.bindsPerItem) {
        subject = `node ${reprText(nodePath)} map input ${reprText(key)}`;
      } else {
        subject = `node ${reprText(nodePath)} ${node.wiringRefLabel(key).replace(/:+$/, "")}`;
      }
      // The binder resolves a reserved key with no type table.
      checkTemplateSource(binding, subject, reserved.has(key) ? new Map() : node.types);
    }
    node.validatePrompts({ path: nodePath });
    for (const child of node.originChildren()) checkTemplates(child as Flow, [...path, nodeId]);
  }
}

/**
 * Check each template string of one wiring source, prefixing an error with `subject`.
 *
 * @throws ExpressionError when a string's span can't run.
 */
function checkTemplateSource(binding: unknown, subject: string, types: ReadonlyMap<string, Type>): void {
  for (const [, text] of iterSourceStrings(binding)) {
    try {
      checkTemplateCalls(text, types);
    } catch (error) {
      if (!(error instanceof ExpressionError)) throw error;
      throw new ExpressionError(`${subject}: ${error.message}`, { cause: error });
    }
  }
}

/**
 * Prove every static TOOL node's inputs and output agree with its tool's types. Every static node
 * is checked, including nodes that may never run, before any record is written.
 *
 * @throws RangeError when a TOOL node passes an input its tool doesn't take, omits one the tool
 *   requires, binds an input whose type can never fit the tool's, or declares an output the tool
 *   can never return.
 */
function checkToolTypes(flow: CompiledFlow, tools: ToolSet): void {
  for (const [nodeId, node] of staticNodes(flow)) {
    const call = node.hostToolCall();
    if (call === null) continue;
    const declared = toolTypes(tools.resolve([call.toolId])[0]!);
    if (declared.inputs !== null) checkToolInputs(nodeId, call, declared.inputs, declared.required);
    if (declared.output !== null) checkToolOutput(nodeId, call, declared.output);
  }
}

/**
 * Prove one TOOL node's inputs match the names and types its tool declares.
 *
 * @param nodeId The node's path, for the message.
 * @param call The node's tool call.
 * @param inputs The tool's declared input types by name; `null` leaves an input unchecked.
 * @param required The names of the inputs the tool requires.
 * @throws RangeError when an input isn't one the tool takes, a required input is missing (all
 *   missing inputs are named at once), or an input's binding type is disjoint from the tool's.
 */
function checkToolInputs(
  nodeId: string,
  call: HostToolCall,
  inputs: ReadonlyMap<string, Type | null>,
  required: ReadonlySet<string>,
): void {
  for (const name of call.inputs) {
    if (!inputs.has(name)) {
      const taken = [...inputs.keys()].sort(compareCodeUnits);
      throw new RangeError(
        `TOOL node ${reprText(nodeId)} input ${reprText(name)}: tool ${reprText(call.toolId)} takes no such input; ` +
          `it takes ${writeAsText(taken)}`,
      );
    }
  }
  const missing = [...required].filter((name) => !call.inputs.includes(name)).sort(compareCodeUnits);
  if (missing.length > 0) {
    const named = missing.length === 1 ? `input ${reprText(missing[0]!)}` : `inputs ${writeAsText(missing)}`;
    throw new RangeError(
      `TOOL node ${reprText(nodeId)} is missing ${named} that tool ${reprText(call.toolId)} requires`,
    );
  }
  for (const [name, inferred] of call.inputTypes) {
    const expected = inputs.get(name) ?? null;
    // Disjoint is yian's non-coercing relation, so a binding a tool's validator would coerce (a
    // string of digits into an int) is rejected too.
    if (expected !== null && typeRelation(inferred, expected) === "disjoint") {
      throw new RangeError(
        `TOOL node ${reprText(nodeId)} input ${reprText(name)}: tool ${reprText(call.toolId)} takes ` +
          `${canonicalTypeLabel(expected)}, but the binding is ${bindingLabel(inferred)}`,
      );
    }
  }
}

/**
 * Prove one TOOL node's declared output can hold a value its tool returns; a call with no output
 * type isn't checked.
 *
 * @throws RangeError when the tool's return type is disjoint from the node's output type.
 */
function checkToolOutput(nodeId: string, call: HostToolCall, declaredOutput: Type): void {
  if (call.outputType === null) return;
  if (typeRelation(declaredOutput, call.outputType) === "disjoint") {
    throw new RangeError(
      `TOOL node ${reprText(nodeId)} output: declared ${canonicalTypeLabel(call.outputType)}, ` +
        `but tool ${reprText(call.toolId)} returns ${canonicalTypeLabel(declaredOutput)}`,
    );
  }
}

/** The most characters of a one-value string `bindingLabel` shows, `...` included. */
const LITERAL_LABEL_WIDTH = 40;

/** Return a binding type's label, showing a one-value string or int by its value. */
function bindingLabel(typ: Type): string {
  const [low, high] = typ.bounds ?? [null, null];
  if (typ.kind === ValueKind.INTEGER && low !== null && low === high && !typ.nullable) return `${low} (int)`;
  if (typ.kind === ValueKind.STRING && typ.tags?.size === 1 && typ.variants === undefined && !typ.nullable) {
    // Code points, so a cut never splits a surrogate pair.
    let chars = [...[...typ.tags][0]!];
    if (chars.length > LITERAL_LABEL_WIDTH) chars = [...chars.slice(0, LITERAL_LABEL_WIDTH - 3), "..."];
    return `"${chars.join("")}" (str)`;
  }
  return canonicalTypeLabel(typ);
}

/**
 * Map each runtime node id to its encoded static path, read from its execution scope.
 *
 * @param nodes The run's live nodes, keyed by node id.
 * @param nodeIds The runtime ids to map.
 * @returns The `/`-joined static path of each id whose node has an execution scope.
 */
function staticPathKeys(nodes: ReadonlyMap<string, Node>, nodeIds: Iterable<string>): Map<string, string> {
  const keys = new Map<string, string>();
  for (const nodeId of nodeIds) {
    const scope = nodes.get(nodeId)!.executionScope;
    if (scope !== null) keys.set(nodeId, scope.staticPath.join("/"));
  }
  return keys;
}

/**
 * @internal Check the public fields of each recorded agent config before a resume appends.
 *
 * A record written before a key was retired may carry it set; resuming would then run with a config
 * the current carrier refuses. No key is required here, so the host-key marker is left out, and
 * with it the base_url and connection settings bound to that key (replay has already checked them;
 * the model call checks the URL against its connection and the header names against the headers
 * given).
 *
 * @param effective The recorded configs: the effective model map keyed by static agent path, and
 *   the config of each started call-time agent keyed by its runtime id.
 * @param labels The authored site of each synthesized agent's `llm_config`, keyed by static agent
 *   path; a message names it instead of the path.
 * @param staticPaths The static path of each runtime id in `effective`, under which its label is
 *   looked up; other keys are static paths already.
 * @throws RangeError when a config has an invalid or retired field; the message names its agent
 *   path or runtime id, or the authored site of a synthesized agent, and never echoes values.
 */
export function checkRecordedLlmFields(
  effective: Readonly<Record<string, Readonly<AgentConfig>>>,
  labels: ReadonlyMap<string, string>,
  staticPaths: ReadonlyMap<string, string> = new Map(),
): void {
  for (const [path, config] of Object.entries(effective)) {
    const hostOnly: readonly string[] =
      HOST_KEY_MARKER in config ? [HOST_KEY_MARKER, "base_url", "organization", "project", HEADER_NAMES_KEY] : [];
    const publicFields = Object.fromEntries(Object.entries(config).filter(([key]) => !hostOnly.includes(key)));
    try {
      splitModelConfig(legacyConnectionConfig(publicFields));
    } catch (error) {
      if (!(error instanceof RangeError)) throw error;
      const where = labels.get(staticPaths.get(path) ?? path) ?? `Agent ${reprText(path)}`;
      throw new RangeError(`${where}: ${error.message}`, { cause: error });
    }
  }
}

/**
 * Copy the host's skills and prove every AGENT grant can be served.
 *
 * Walks top-level and static child-flow agents before any record is written. A flow that grants
 * skills needs a `skills` map (an empty one grants none), so a run never silently drops them. A
 * listed skill must be in the map.
 *
 * @returns A copy of `skills`, or `null` when the host passed none.
 * @throws TypeError when `skills` isn't a `Map` of `Skill`s each keyed by its name.
 * @throws RangeError when a flow grants skills and no map was passed, or a listed skill is missing.
 */
function checkSkills(flow: CompiledFlow, skills: ReadonlyMap<string, Skill> | null): Map<string, Skill> | null {
  let copied: Map<string, Skill> | null = null;
  if (skills !== null) {
    if (!(skills instanceof Map)) throw new TypeError("skills must be a Map of name to Skill, or null");
    for (const [name, skill] of skills as ReadonlyMap<unknown, unknown>) {
      if (!isExactly(skill, Skill) || skill.name !== name) {
        const shown = reprText(String(name));
        throw new TypeError(`skills[${shown}] must be a Skill named ${shown}`);
      }
    }
    copied = new Map(skills);
  }
  for (const [nodeId, grants] of skillGrants(flow)) {
    if (copied === null) {
      throw new RangeError(
        `AGENT node ${reprText(nodeId)} grants skills, but no skills= mapping was passed; ` +
          "pass the run's skills (or an empty Map for none)",
      );
    }
    const missing = grants === "all" ? [] : grants.filter((name) => !copied.has(name));
    if (missing.length > 0) {
      throw new RangeError(
        `AGENT node ${reprText(nodeId)} grants skills not supplied to this run: ${writeAsText(missing)}`,
      );
    }
  }
  return copied;
}

/**
 * @internal Return each static node's non-empty skill grant, keyed by its node path.
 */
export function skillGrants(flow: Flow): [string, readonly string[] | "all"][] {
  const found: [string, readonly string[] | "all"][] = [];
  for (const [nodeId, node] of staticNodes(flow)) {
    // Read kind-blind: only a node that declares `needsSkills` carries a grant.
    const grants = (node as { skills?: readonly string[] | "all" }).skills;
    if (node.needsSkills && grants !== undefined && grants.length > 0) found.push([nodeId, grants]);
  }
  return found;
}

/**
 * @internal Return every static node of `flow` and its child flows, keyed by node path.
 *
 * Kind-blind: child flows come from `Node.originChildren()`, so the run boundary imports no node
 * class.
 */
export function staticNodes(flow: Flow, path: readonly string[] = []): [string, Node][] {
  const found: [string, Node][] = [];
  for (const [nodeId, node] of flow.nodes) {
    const nodePath = [...path, nodeId];
    found.push([nodePath.join("/"), node]);
    for (const child of node.originChildren()) found.push(...staticNodes(child as Flow, nodePath));
  }
  return found;
}

/**
 * @internal Silently rebuild the state of one borrowed, locked log prefix.
 *
 * The authored fingerprint is checked before copying. The private graph gets only the complete
 * effective model map and the call-time agents `RunStarted` recorded; recovery never reads the
 * current model defaults. Recorded transitions go through the live reducer and growth builder.
 * A host `apiKey`, and the `headers` sent with it, go to the agents the map marks for it and to
 * the call-time agents. The run result reads its files through the flow's output contract, as
 * `runFlow`'s does.
 *
 * @throws TypeError when `flow` isn't a `ComposedFlow`.
 * @throws RangeError, and the other errors of `replayCompiledRunLog`.
 */
export function replayRunLog(
  flow: ComposedFlow,
  log: RunLog,
  options: { apiKey?: string | null; headers?: Readonly<Record<string, string>> | null } = {},
): RunState {
  if (!(flow instanceof ComposedFlow)) throw new TypeError("replayRunLog expects a ComposedFlow");
  return replayCompiledRunLog(flow.compiled, log, { ...options, resultContract: childSignature(flow).output });
}

/**
 * @internal Replay a log prefix over a private copy of an authored compiled program.
 *
 * `resultContract` is the contract the terminal output is read through to restore its files
 * (`RunState.resultContract`); a finished log produces its terminal during replay, so it is set
 * before any record is folded.
 *
 * @throws RangeError when the log's run id or sequence is malformed or changes while it is read,
 *   the prefix is empty, an envelope isn't in canonical form or breaks the run id or the sequence,
 *   the first record isn't `RunStarted`, the flow fingerprint differs, the recorded memory table
 *   doesn't match the flow, or a record isn't the exact next transition.
 * @throws DefaultsMismatchError when the run recorded `YIAN_*` values that differ from the ones in
 *   effect.
 */
export function replayCompiledRunLog(
  flow: CompiledFlow,
  log: RunLog,
  options: {
    apiKey?: string | null;
    headers?: Readonly<Record<string, string>> | null;
    resultContract?: OutputContract | null;
  } = {},
): RunState {
  const { apiKey = null, headers = null, resultContract = null } = options;
  const sourceRunId = log.runId;
  const sourceSequence = log.nextSequence;
  if (typeof sourceRunId !== "string" || sourceRunId === "") {
    throw new RangeError("run log run_id must be a non-empty string");
  }
  if (!Number.isSafeInteger(sourceSequence) || sourceSequence < 0) {
    throw new RangeError("run log next_sequence must be a non-negative integer");
  }

  const supplied = [...log];
  if (log.runId !== sourceRunId || log.nextSequence !== sourceSequence) {
    throw new RangeError("run log changed while its historical prefix was read");
  }
  if (sourceSequence !== supplied.length) {
    throw new RangeError(
      `run log sequence mismatch: next_sequence is ${sourceSequence}, prefix length is ${supplied.length}`,
    );
  }
  if (supplied.length === 0) throw new RangeError("cannot replay an empty run log");

  const canonical: RecordEnvelope[] = [];
  for (const [expectedSequence, suppliedEnvelope] of supplied.entries()) {
    const line = encodeRecord(suppliedEnvelope);
    const envelope = decodeRecord(line);
    // Envelopes are frozen and compared by their canonical lines, as Python compares values.
    if (encodeRecord(envelope) !== line) {
      throw new RangeError(`run log envelope ${expectedSequence} is not in canonical wire form`);
    }
    if (envelope.seq !== expectedSequence) {
      throw new RangeError(`run log sequence mismatch: expected ${expectedSequence}, found ${envelope.seq}`);
    }
    if (envelope.runId !== sourceRunId) {
      throw new RangeError(
        `run log run_id mismatch: expected ${reprText(sourceRunId)}, found ${reprText(envelope.runId)}`,
      );
    }
    canonical.push(envelope);
  }

  const first = canonical[0]!.record;
  if (!(first instanceof RunStartedRecord)) throw new RangeError("the first run log record must be RunStarted");
  // Values like AGENT_RETRIES change the loaded flow, so a mismatch checked after the fingerprint
  // would surface as a fingerprint error instead of naming them.
  checkRecorded(first.defaults);
  const actualFingerprint = flowFingerprint(flow);
  if (actualFingerprint !== first.flowFingerprint) {
    throw new RangeError(
      `flow fingerprint mismatch: recorded ${reprText(first.flowFingerprint)}, supplied ${reprText(actualFingerprint)}`,
    );
  }

  const privateFlow = copyGraph(flow);
  applyEffectiveLlmMap(privateFlow, legacyConnectionMap(first.effectiveLlmByStaticPath), first.callTimeLlm);
  if (apiKey !== null) {
    // A run that marked no agent ignores the key and headers: no static agent is marked, and a
    // call-time agent gets them only when its selection names the key's provider, and uses them
    // only when its resolved config is marked.
    bindHostCredentials(privateFlow, apiKey, headers);
  }
  if (first.learnable) applyExecutionLayout(privateFlow, resolveExecutionLayout(privateFlow));
  if (first.memories !== null) {
    const replayed = resolveRuntimeMemoryLayout(privateFlow);
    checkRecordedMemoryTable(first.memories, first.memoryStacks, replayed.memories, replayed.stacks, "supplied flow");
  }
  const state = replayRecords(new RunState(privateFlow, { resultContract }), canonical, { growBuilder: buildGrow });
  if (log.runId !== sourceRunId || log.nextSequence !== sourceSequence) {
    throw new RangeError("run log changed while its historical prefix was replayed");
  }
  return state;
}

/** What `scanRunRecords` reads from a log's records. */
type RecordScan = {
  /** The start sequence of each node in flight, in start order. */
  starts: Map<string, number>;
  /** The interrupted counts the run's resumes recorded, as `RunState.interruptedAttempts`. */
  interrupted: Map<string, number>;
  /** The sequence of the last `RunStopped` since the last `RunResumed`, or `null`. */
  lastStop: number | null;
  /** How the records show the run finished, or `null` when they don't. */
  finished: "succeeded" | "failed" | null;
};

/**
 * Read the in-flight nodes, interrupted counts and terminal status from a log's records, as the
 * reducer derives them from starts, outcomes, stops and resumes.
 *
 * Finished means a node failed, or, with nothing in flight, the top-level END succeeded or a route
 * proved it unreachable; the records don't hold the ready nodes, so a node still ready is missed.
 */
function scanRunRecords(envelopes: readonly RecordEnvelope[]): RecordScan {
  const starts = new Map<string, number>();
  const interrupted = new Map<string, number>();
  let lastStop: number | null = null;
  let failed = false;
  let endSucceeded = false;
  let unreachable = false;
  for (const { seq, record } of envelopes) {
    if (record instanceof NodeStartedRecord) {
      starts.set(record.nodeId, seq);
    } else if (record instanceof NodeSucceededRecord) {
      starts.delete(record.nodeId);
      if (record.nodeId === END_ID) endSucceeded = true;
    } else if (record instanceof NodeRoutedRecord) {
      starts.delete(record.nodeId);
      if (record.terminalFailure !== null) unreachable = true;
    } else if (record instanceof NodeExpandedRecord) {
      starts.delete(record.spawnerId);
    } else if (record instanceof NodeFailedRecord) {
      starts.delete(record.failure.nodeId!);
      failed = true;
    } else if (record instanceof RunStoppedRecord) {
      lastStop = seq;
    } else if (record instanceof RunResumedRecord) {
      // The resume retries exactly the nodes in flight.
      addInterruptedAttempts(interrupted, starts, lastStop);
      starts.clear();
      lastStop = null;
    }
  }
  let finished: RecordScan["finished"] = null;
  if (failed || (starts.size === 0 && unreachable)) finished = "failed";
  else if (starts.size === 0 && endSucceeded) finished = "succeeded";
  return { starts, interrupted, lastStop, finished };
}

/**
 * Replay `envelopes` read-only over a private copy of `flow`, as `resumeFlow` replays its log, and
 * read the same parts as `scanRunRecords` from the state.
 *
 * @throws RangeError, and the other errors of `replayRunLog`.
 */
function replayedScan(flow: ComposedFlow, envelopes: readonly RecordEnvelope[]): RecordScan {
  const state = replayRunLog(flow, MemoryRunLog.fromRecords(envelopes));
  let finished: RecordScan["finished"] = null;
  if (state.result !== null) finished = state.result instanceof RunSucceeded ? "succeeded" : "failed";
  return {
    starts: new Map(state.acceptedStarts),
    interrupted: new Map(state.interruptedAttempts),
    lastStop: state.lastStopSequence,
    finished,
  };
}

/** A layout's record-facing key table, per-key origins and stacks. */
type MemoryLayoutParts = {
  memories: readonly MemorySpec[];
  origins: Map<string, OriginRef | null>;
  stacks: readonly MemoryStack[];
};

/**
 * Derive the key table, origins and stacks, and stamp one private graph.
 *
 * With `effectiveLlm`, first check that each session key resolves one conversation compatibility
 * across its placements; `callTimeKnown` gives the provider of each placement resolved only when
 * it runs, when run start knows it (`checkSessionCompatibility`).
 *
 * @param flow The run's private flow copy; its nodes get their memory scopes.
 * @throws MemorySetupError when the layout breaks a rule or two placements of a session key
 *   resolve different providers.
 */
function resolveRuntimeMemoryLayout(
  flow: CompiledFlow,
  effectiveLlm: Readonly<Record<string, Readonly<AgentConfig>>> | null = null,
  callTimeKnown: ReadonlyMap<string, string | null> = new Map(),
): MemoryLayoutParts {
  const resolved = resolveMemoryLayout(flow);
  if (effectiveLlm !== null) {
    checkSessionCompatibility(resolved, effectiveLlm, { callTimeKnown: Object.fromEntries(callTimeKnown) });
  }
  applyMemoryLayout(flow, resolved);
  return memoryLayoutParts(resolved);
}

/** Return one layout's record-facing key table, per-key origins and stacks. */
function memoryLayoutParts(resolved: ResolvedMemoryLayout): MemoryLayoutParts {
  return {
    memories: resolved.memories.map((memory) => memory.spec),
    origins: new Map(resolved.memories.map((memory) => [memory.key, memory.origin])),
    stacks: memoryStacks(resolved),
  };
}

/**
 * Reject a `MemoryRegistry` whose declared types differ from this flow's table; any other `Map` is
 * not checked.
 *
 * @throws MemorySetupError when a key's registry type differs from its declared type.
 */
function checkRegistryTypes(memory: ReadonlyMap<string, Memory>, specs: readonly MemorySpec[]): void {
  if (!(memory instanceof MemoryRegistry)) return;
  for (const spec of specs) {
    const built = memory.types.get(spec.key);
    if (built !== undefined && built !== spec.type) {
      throw new MemorySetupError(
        `memory ${reprText(spec.key)} is declared as ${reprText(spec.type)} but the registry ` +
          `built it as ${reprText(built)}; build the registry from the flow you run`,
      );
    }
  }
}

/**
 * Resolve memory on the authored static program, never on replayed growth.
 *
 * Nothing is stamped, so a def called from several nodes needs no de-aliasing; the table, origins
 * and stacks equal those a fresh run records.
 *
 * @throws MemorySetupError when the layout breaks a rule.
 */
function resolveAuthoredMemoryLayout(flow: ComposedFlow): MemoryLayoutParts {
  return memoryLayoutParts(resolveMemoryLayout(flow.compiled));
}

/**
 * Reject a recorded key table or stack set that the flow doesn't resolve.
 *
 * @param subject What resolved `memories` and `stacks`, for the message.
 * @throws RangeError when the table or the stacks differ.
 */
function checkRecordedMemoryTable(
  recordedMemories: readonly MemorySpec[] | null,
  recordedStacks: readonly MemoryStack[] | null,
  memories: readonly MemorySpec[],
  stacks: readonly MemoryStack[],
  subject: string,
): void {
  const sameStack = (a: MemoryStack, b: MemoryStack): boolean =>
    sameItems(a.locator, b.locator) && sameItems(a.memories, b.memories);
  if (
    !sameOptional(recordedMemories, memories, (a, b) => sameItems(a, b, sameMemorySpec)) ||
    !sameOptional(recordedStacks, stacks, (a, b) => sameItems(a, b, sameStack))
  ) {
    throw new RangeError(`recorded memory table or stacks do not match the ${subject}`);
  }
}

/**
 * Publish an already-applied segment lead before retaining it in the result. Nothing is in flight
 * yet, so a throwing callback rejects the run with no stop record; a promise the callback returns
 * that rejects goes to `stop`, which stops the run with its error.
 */
function publishBoundaryObservers(
  observers: readonly RunEvent[],
  events: RunEvent[],
  onEvent: EventHook | null,
  stop: (error: unknown) => void,
): void {
  for (const observer of observers) {
    const returned = onEvent?.(observer);
    if (returned instanceof Promise) returned.then(undefined, stop);
    events.push(observer);
  }
}

/**
 * Project one terminal reducer state into the public result.
 *
 * @throws Error when the state has no terminal result.
 */
function resultFromState(
  state: RunState,
  path: string | null,
  events: readonly RunEvent[],
  handle: BackwardHandle | null = null,
): RunResult {
  const terminal = state.result;
  const common = {
    input: copyValue(state.input) as Record<string, unknown>,
    runId: requiredStateRunId(state),
    log: path,
    events,
    interruptedAttempts: new Map(state.interruptedAttempts),
    backwardOutcome: handle === null ? state.backwardState.outcome : handle.outcome,
  };
  let result: RunResult;
  if (terminal instanceof RunSucceeded) {
    result = new RunResult({ ...common, status: "succeeded", output: copyValue(terminal.output as Value) });
  } else if (terminal instanceof RunFailed) {
    result = new RunResult({ ...common, status: "failed", failure: terminal.failure });
  } else {
    throw new Error("engine returned without deriving a terminal run result");
  }
  if (handle !== null) BACKWARD_HANDLES.set(result, handle);
  return result;
}

/**
 * Detach the authority a later `RunResult.backward` needs, keeping no open session or log: the
 * log's path (or a process-local log's records), the retained program and memory, the host key
 * and headers, the run's environment and model access. A plain `Map` of memories is copied, so a
 * later change to it doesn't reach the handle; a `MemoryRegistry` is kept as is.
 *
 * @param program The retained original program, or `null` for a run that isn't learnable.
 * @param locator The log's absolute path, or `null` for a process-local log.
 * @returns The handle, or `null` when `program` is `null`.
 */
function retainedHandle(
  program: CompiledFlow | null,
  state: RunState,
  log: RunLog,
  memory: ReadonlyMap<string, Memory> | null,
  options: Pick<BackwardHandleFields, "apiKey" | "headers" | "env" | "models"> & { locator: string | null },
): BackwardHandle | null {
  if (program === null) return null;
  return new BackwardHandle({
    program,
    runId: requiredStateRunId(state),
    path: options.locator,
    records: log.path === null ? log.records() : [],
    memory: memory !== null && Object.getPrototypeOf(memory) === Map.prototype ? new Map(memory) : memory,
    outcome: state.backwardState.outcome,
    apiKey: options.apiKey,
    headers: options.headers,
    env: options.env,
    models: options.models,
  });
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

// Python's `_set_stop_preserving` has no counterpart: a run's stop is the host's signal, which the
// run only reads, and a callback's error stops the engine through `stopForCallback`.

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

/** Return the error for the server under `key`, which another run has open or is opening. */
function otherRunError(key: string, options?: ErrorOptions): RangeError {
  return new RangeError(
    `tools[${JSON.stringify(key)}]: this MCP server was opened by another run; ` +
      'open it with "await server.open()" to share it between runs',
    options,
  );
}

/**
 * Split the run's tools into the entries that need no opening and the unopened servers.
 *
 * An unopened server under a key `ToolSet` rejects (empty, or `ask_user`) stays with the other
 * entries, so `ToolSet` throws its usual error before any server starts.
 *
 * @param tools The `tools` option of `runFlow`. Anything other than a record is returned as is,
 *   for `ToolSet` to reject.
 * @returns `tools` without its unopened `McpServer` entries, and those servers in key order with
 *   the first key each sits under, each once even when it sits under several keys.
 * @throws RangeError when a server entry is open because another run in progress opened it.
 */
function splitRunServers(
  tools: Readonly<Record<string, ToolEntry>> | null,
): [Readonly<Record<string, ToolEntry>> | null, [string, McpServer][]] {
  if (!isRecord(tools)) return [tools, []];
  // No prototype, so a `__proto__` key stays an entry.
  const others: Record<string, ToolEntry> = Object.create(null);
  const unopened: [string, McpServer][] = [];
  for (const key of Object.keys(tools)) {
    const entry = tools[key]!;
    if (!isMcpServer(entry) || key === "" || key === "ask_user") {
      others[key] = entry;
    } else if (entry.isOpen) {
      if (entry.openedByRun) throw otherRunError(key);
      others[key] = entry;
    } else if (!unopened.some(([, seen]) => seen === entry)) {
      unopened.push([key, entry]);
    }
  }
  return [others, unopened];
}

/**
 * Open the servers for one run, in order; if one fails, close the ones already open.
 *
 * Once `signal` aborts, no further server opens: the ones already open close and the run rejects
 * with its reason, before anything is recorded.
 *
 * @param servers Unopened servers, each listed once with the `tools` key it sits under.
 * @param signal The run's stop.
 * @returns The opened servers, owned by a token unique to this run.
 * @throws RangeError when another caller opened or is opening a server since it was found
 *   unopened.
 * @throws Error when the MCP SDK is not installed.
 * @throws McpConnectError when a server doesn't start or doesn't list its tools.
 * @throws The signal's reason when it aborts before every server is open.
 */
async function openRunServers(
  servers: readonly [string, McpServer][],
  signal: AbortSignal | null,
): Promise<McpServer[]> {
  const token = `run-${randomUUID().replaceAll("-", "")}`;
  const opened: McpServer[] = [];
  try {
    for (const [key, server] of servers) {
      signal?.throwIfAborted();
      try {
        await server.openForRun(token, signal === null ? {} : { signal });
      } catch (error) {
        if (error instanceof AlreadyOpenError) throw otherRunError(key, { cause: error });
        throw error;
      }
      opened.push(server);
    }
  } catch (error) {
    await closeRunServers(opened);
    throw error;
  }
  return opened;
}

/**
 * Close the run's servers in order.
 *
 * Python's `_close_run_servers` also guards each close and notes its error on the run's; here
 * `McpServer.close()` never rejects, so there is nothing to note.
 */
async function closeRunServers(servers: readonly McpServer[]): Promise<void> {
  for (const server of servers) await server.close();
}

/**
 * Close memory then the log while preserving one initiating error: with no `primary`, the first
 * close error is thrown, later ones noted on it; with one, each is noted on it.
 */
async function closeRunResources(
  memoryRuntime: MemoryRuntime | null,
  log: RunLog | null,
  primary: { error: unknown } | null,
): Promise<void> {
  let pending = primary;
  if (memoryRuntime !== null) {
    try {
      await memoryRuntime.close();
    } catch (cleanupError) {
      if (pending === null) pending = { error: cleanupError };
      else noteCleanupError(pending.error, "memory-session close", cleanupError);
    }
  }
  if (log !== null) closeOwnedLog(log, pending);
  if (primary === null && pending !== null) throw pending.error;
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
