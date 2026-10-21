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
 * Knows about:   compile, compose, defaults, diagnostics, events, expr, llm_clients, memory contracts,
 *                nodes, platform, runtime, skills, tools, typesys.
 * Never imports: the CLI.
 */

import { resolve } from "node:path";
import { types } from "node:util";

import { applyExecutionLayout, resolveExecutionLayout } from "./compile/execution_layout.ts";
import { buildGrow } from "./compile/expand.ts";
import {
  applyEffectiveLlmMap,
  bindHostCredentials,
  callTimeSelection,
  hostKeyProviders,
  knownCallTimeProviders,
  resolveRunLlm,
  type RunLLM,
  staticAgentLabels,
  valueConfigSubjects,
} from "./compile/llm_cascade.ts";
import { flowFingerprint } from "./compile/manifest.ts";
import { type CompiledFlow, copyGraph, type Flow } from "./compile/model.ts";
import { typeRelation } from "./compile/validation.ts";
import { ComposedFlow } from "./compose/build/pipeline.ts";
import { CLOSE_MATCH_CUTOFF } from "./compose/helpers/tags.ts";
import { childSignature } from "./compose/nodes/resolver.ts";
import { current, type Defaults, type Env, nonDefaults, processEnv, withEnv } from "./defaults.ts";
import { compareCodeUnits, type Failure, getCloseMatches, reprText } from "./diagnostics.ts";
import { type RunEvent, RunFailed, RunSucceeded, type StreamChunk } from "./events.ts";
import { ExpressionError } from "./expr/expressions.ts";
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
  refusedBlock,
  type SelectionRecord,
  splitModelConfig,
} from "./llm_clients/connection.ts";
import { validateHeaders } from "./llm_clients/connection_settings.ts";
import { defaultModelAccess, type ModelAccess } from "./llm_clients/factory.ts";
import type { LearningResult, Memory } from "./memory/contracts.ts";
import type { Node, Steering } from "./nodes/base.ts";
import { applyDefaults, BindingError, checkInputNames, coerceInputs, copyValue } from "./nodes/binding.ts";
import { type HostToolCall, isExactly, iterSourceStrings, type Parameter } from "./nodes/contracts.ts";
import { realScheduler, type Scheduler } from "./platform.ts";
import type { BackwardOutcome } from "./runtime/contracts.ts";
import { FlowEngine } from "./runtime/engine.ts";
import { addNote, requireSteering } from "./runtime/eval_node.ts";
import { JsonlRunLog, MemoryRunLog, type RunLog } from "./runtime/log.ts";
import { commitPrepared, prepareRunStarted, RunState } from "./runtime/state.ts";
import { Skill } from "./skills/catalog.ts";
import { isMcpServer } from "./tools/mcp.ts";
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
  /**
   * The environment for connection selection, credentials, model building and the run's `YIAN_*`
   * values (default `process.env`).
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
 *   isn't an `AbortSignal`, `inputs` isn't a record (a Proxy included), `llmConfig` isn't a record,
 *   a `tools` entry is malformed, `skills` isn't a `Map` of `Skill`s each keyed by its name, or
 *   `steering` lacks its methods.
 * @throws ExpressionError when a condition, binding or prompt set in code can't run.
 * @throws TypeCheckError when an input holds an accessor, NaN, a lone surrogate, a cycle or
 *   another non-yian value, nests deeper than `YIAN_MAX_VALUE_NESTING` levels, or has text that
 *   can't be read.
 * @throws RangeError when an input key names no declared input; an AGENT grants a tool, group or
 *   skill this run lacks; a TOOL node names a tool this run lacks, or its inputs or output can't
 *   fit that tool's types; `llmConfig` or an agent's resolved model config is invalid, an agent
 *   has no provider or model, or `llmConfig.api_key` has no `provider`; or the identity, worker
 *   count or prepared start state breaks its contract. A message never echoes the key.
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
  return withEnv(options.env ?? processEnv, () => {
    const runDefaults = current();
    const { learnable = false, tools = null, steering = null, context, signal = null } = options;
    if (!(flow instanceof ComposedFlow)) throw new TypeError("runFlow expects a ComposedFlow");
    if (typeof learnable !== "boolean") throw new TypeError("learnable must be a boolean");
    if (signal !== null && !(signal instanceof AbortSignal)) {
      throw new TypeError("signal must be an AbortSignal or null");
    }
    signal?.throwIfAborted();
    // A server the run opens and closes itself.
    if (tools !== null && Object.values(tools).some((entry) => isMcpServer(entry) && !entry.isOpen)) {
      throw new Error("run-owned MCP servers are not ported yet");
    }
    // The tool entries' errors, and the checks that need no tools, come before any server starts.
    new ToolSet(tools, { context });
    checkConditions(flow.compiled);
    checkTemplates(flow.compiled);
    const skills = checkSkills(flow.compiled, options.skills ?? null);
    requireSteering(steering);
    return startRun(flow, inputs, { ...options, skills }, runDefaults);
  });
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

/** Check the tools, then record and drive a fresh run. */
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
  const env = options.env ?? processEnv;
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
  // selection.
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
    const observers = commitPrepared(authority, state, preparedStart);
    // Built before publishing so a callback's rejected promise can stop it.
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
    const events: RunEvent[] = [];
    publishBoundaryObservers(observers, events, onEvent, (error) => engine.stopForCallback(error));
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
