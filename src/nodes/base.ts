/**
 * The node base contract: a node is a pure function of its bound input record.
 *
 * The base's async template method `run(inputs, caps)` fills defaults, checks the pre-asserts,
 * calls the kind's body, checks the post-asserts, and resolves to ONE of the closed sum
 * `Output | Route | Grow | Pending` (`NodeResult`). A kind supplies its body as `forward(inputs, caps)`,
 * which returns a result or a promise of one. The node never receives the pool: the engine's
 * evaluator binds its inputs (the read boundary) and hands it a record. Narrow effects enter
 * through one capabilities object, `caps`: the node's own abort `signal` and, when the host gives
 * them, the `scheduler` its waits run on and the `random` draw for backoff jitter, plus what its
 * traits ask for (`bindItem` for MAP, `steps` for a node that records durable steps). Failure is
 * not a variant: a failing node rejects, and the evaluator turns an expected rejection into a
 * canonical failure retained by the accepted failure record.
 *
 * Invariant: a node never writes the pool. It describes its one output value as
 * `new Output(value)`; the engine performs the write under the node's id. This keeps nodes pure.
 *
 * Knows about:   diagnostics, events, expr, llm_clients (types), platform (the `Scheduler` type),
 *                skills, tools, typesys, nodes/binding, nodes/contracts.
 * Never imports: compile, compose, runtime, the package-root run boundary, or the CLI.
 */

import { current } from "../defaults.ts";
import { Failure, type FailureCode, OriginRef, reprText } from "../diagnostics.ts";
import type {
  ModelCallFinished,
  ModelCallStarted,
  ReplayDegraded,
  StreamChunk,
  ToolCallFinished,
  ToolCallStarted,
} from "../events.ts";
import { ExpressionError, parseCondition, promptRefs } from "../expr/index.ts";
import { type DeclaredTypes, evaluateWhenRecord, localDeclared } from "../expr/expressions.ts";
import type { ModelAccess } from "../llm_clients/factory.ts";
import type { Request, ToolSpec } from "../llm/messages.ts";
import type { MemorySession } from "../memory/base.ts";
import type { Scheduler } from "../platform.ts";
import type { Skill } from "../skills/catalog.ts";
import type { MemoryGrant, ToolGrant } from "../tools/grants.ts";
import type { ToolSet } from "../tools/registry.ts";
import type { JsonObject } from "../typesys/json.ts";
import type { VariablePool } from "../typesys/pool.ts";
import {
  cutText,
  fromHost,
  isRecord,
  jsTypeName,
  MemoryRef,
  setField,
  type Type,
  TypeCheckError,
  type TypedValue,
  type Value,
  type ValueRecord,
} from "../typesys/values.ts";
import {
  applyDefaults,
  BindingError,
  copyValue,
  type ReservedBinder,
  resolveSource,
  sourceOrigin,
  validateParameters,
} from "./binding.ts";
import {
  type Assertion,
  type CallView,
  type HostToolCall,
  isExactly,
  iterOriginCarrier,
  mapOrigin,
  type NodeSteps,
  type OriginMapper,
  type OutputContract,
  outputContractOrigin,
  Parameter,
  PartialRecordContract,
  rewriteOriginCarrier,
  validateOptionalOrigin,
  ValueContract,
} from "./contracts.ts";

/** The closed vocabulary of node kinds. Dispatch is an explicit `switch`, never a registry. */
export const NodeKind = {
  AGENT: "agent",
  CODE: "code",
  TOOL: "tool",
  CASE: "case",
  /** Authorable: iterates a callable to a fixpoint (`('a -> 'a) -> 'a -> 'a`). */
  LOOP: "loop",
  /** Internal-only: the loader-synthesized input boundary (parameter binding). */
  START: "start",
  /** Internal-only: the loader-synthesized return boundary (record and list modes). */
  END: "end",
  /** Authorable: consults another flow once (`kind: call`). */
  CALL: "call",
  /** Authorable: maps a callable over a list (`kind: map` with `over:`). */
  MAP: "map",
} as const;

export type NodeKind = (typeof NodeKind)[keyof typeof NodeKind];

/**
 * A produced value that the dispatcher records before the reducer binds it. Frozen.
 *
 * `commitAs` is the node-chosen commit redirect: when set, the reducer writes `value` under
 * `commitAs` instead of the node's own id and fires that target's out-edges; `null` (the common
 * case) commits under the node's own id. LOOP uses it when its final driver commits under the
 * compiled origin; the baked `Node.commitAs` field covers child-flow terminal redirects.
 */
export class Output {
  /** The produced value. */
  readonly value: unknown;
  /** The target id to commit under instead of the node's own id, or `null`. */
  readonly commitAs: string | null;

  /** Describe one produced value; `value` and `commitAs` default to `null`. */
  constructor(value: unknown = null, options: { commitAs?: string | null } = {}) {
    this.value = value;
    this.commitAs = options.commitAs ?? null;
    Object.freeze(this);
  }
}

/**
 * A routing-only outcome (a terminal) carrying only the chosen case handle: no value, no
 * post-asserts. Frozen. A router (CASE) returns it; the dispatcher records the handle and the
 * reducer takes its out-edge while skip-flooding siblings.
 */
export class Route {
  /** The chosen case handle. */
  readonly handle: string;

  /** Describe one routing choice. */
  constructor(handle: string) {
    this.handle = handle;
    Object.freeze(this);
  }
}

// Labeled branches are named by their target node id, and no authored node id may hold `#`, so
// no labeled branch can share the else route.
/**
 * The handle a router takes when no labeled branch matches. Unlike any other handle it may have
 * no outgoing edge.
 */
export const ELSE_HANDLE = "#else";

/** The start of each reserved wiring key a node routes a memory grant through. */
export const MEMORY_KEY_PREFIX = "__memory_";

/** Return the reserved wiring key of a grant of memory input `input`: `__memory_<input>`. */
export function memoryKey(input: string): string {
  return `${MEMORY_KEY_PREFIX}${input}`;
}

/**
 * Resolve the reserved binding of each memory input `grants` name, once per input, to the memory
 * value it holds (`null` for an Optional memory left out), keyed by its reserved key.
 *
 * @param field The authored field the grants are in (`tools` or `tool_id`), for the message.
 * @param bind Resolves a named source and applies the check, as `bindReserved`'s `bind`; `null`
 *   resolves against `pool`.
 * @throws Error when the wiring lacks a grant's reserved binding.
 * @throws BindingError (`contract.input`) when a binding holds something other than a memory, or
 *   two inputs hold one memory, which the node would reach twice.
 */
export function bindMemoryGrants(
  nodeId: string,
  grants: readonly MemoryGrant[],
  field: string,
  nodeWiring: ReadonlyMap<string, unknown>,
  pool: VariablePool,
  bind: ReservedBinder | null,
): Record<string, unknown> {
  const reserved: Record<string, unknown> = {};
  // The input each bound memory identity came through: one node reaches a memory by one input.
  const granted = new Map<string, string>();
  for (const input of new Set(grants.map((grant) => grant.input))) {
    const key = memoryKey(input);
    if (!nodeWiring.has(key)) throw new Error(`node ${reprText(nodeId)} has no \`${key}\` wiring`);
    const source = nodeWiring.get(key);
    const normalize = (value: unknown): MemoryRef | null => {
      if (value === null || MemoryRef.isMemoryRef(value)) return value;
      throw new BindingError(`node ${reprText(nodeId)}: \`${input}\` in \`${field}:\` must be a memory`, {
        failureCode: "contract.input",
        origin: sourceOrigin(source),
      });
    };
    const value = bind === null ? normalize(resolveSource(source, pool)) : bind(key, source, normalize);
    reserved[key] = value;
    if (!MemoryRef.isMemoryRef(value)) continue;
    const other = granted.get(value.identity);
    if (other !== undefined) {
      throw new BindingError(`node ${reprText(nodeId)}: \`${input}\` and \`${other}\` are the same memory; grant one`, {
        failureCode: "contract.input",
        origin: sourceOrigin(source),
      });
    }
    granted.set(value.identity, input);
  }
  return reserved;
}

/**
 * One deterministic graph expansion a spawner describes for the runtime to record. Frozen.
 *
 * `seed` is the pure input to the compile-owned growth builder: kind-shaped but opaque to the
 * engine (a CALL record, MAP records list, or LOOP record and index). The same canonical value
 * is persisted, so durable replay rebuilds identical topology without re-running the node that
 * produced it. `prune` holds the ids to retire in the same step (empty for CALL and MAP; a
 * self-respawning LOOP's finished iteration ids). The reducer retires a pruned id with its whole
 * grown subtree once every node in it is settled; until then it stays live.
 */
export class Grow {
  /** The pure input to the compile-owned growth builder. */
  readonly seed: unknown;
  /** The ids to retire in the same step; a private copy, read-only by contract. */
  readonly prune: ReadonlySet<string>;

  /** Describe one expansion; `prune` is copied into a new set and defaults to empty. */
  constructor(seed: unknown, prune: Iterable<string> = []) {
    this.seed = seed;
    this.prune = new Set(prune);
    Object.freeze(this);
  }
}

/** A take or sleep the engine decides when it commits the node's outcome. */
export type WaitRequest =
  | { readonly tool: "take"; readonly inbox: string; readonly mode: "one" | "all"; readonly timeoutSeconds: number | null }
  | { readonly tool: "sleep"; readonly until: string | null; readonly forSeconds: number | null };

/** A host's answer to a pending call. */
export type CompleteResult = { readonly output: unknown } | { readonly error: string };

/**
 * "I wait": on host calls the node registered (`callIds`), or on a take or sleep the engine
 * decides when it commits the outcome (`wait`). Exactly one is given. Frozen.
 */
export class Pending {
  /** The ids of the host calls the node waits on; empty when `wait` is set. */
  readonly callIds: readonly string[];
  /** The take or sleep the engine decides, or null when the node waits on host calls. */
  readonly wait: WaitRequest | null;

  /**
   * Describe one wait; `callIds` and `wait` are copied.
   *
   * @throws TypeError when `callIds` is a single string, a call id isn't a string, or `wait`
   *   names another tool.
   * @throws RangeError when an id is empty or repeated, both or neither of `callIds` and `wait`
   *   are given, or `wait`'s fields break their rules.
   */
  constructor(callIds: Iterable<string>, options: { wait?: WaitRequest | null } = {}) {
    // A string is an Iterable<string>, so it would typecheck and wait on its characters.
    if (typeof callIds === "string") throw new TypeError("pending call ids must be a collection of ids, not one string");
    const ids = [...callIds];
    for (const id of ids) {
      if (typeof id !== "string") throw new TypeError(`pending call ids must be strings, got ${jsTypeName(id)}`);
      if (id === "") throw new RangeError("pending call ids must be non-empty");
    }
    if (new Set(ids).size !== ids.length) throw new RangeError("pending call ids must be unique");
    const wait = options.wait ?? null;
    if (wait !== null) checkWaitRequest(wait);
    if ((ids.length > 0) === (wait !== null)) throw new RangeError("Pending takes call ids or a wait, exactly one");
    this.callIds = Object.freeze(ids);
    this.wait = wait === null ? null : Object.freeze({ ...wait });
    Object.freeze(this);
  }
}

/** Require a take with an inbox, a known mode and an optional finite timeout, or a sleep with exactly one of `until` and `forSeconds`. */
function checkWaitRequest(wait: WaitRequest): void {
  const tool: unknown = isRecord(wait) ? wait.tool : undefined;
  if (tool === "take") {
    const take = wait as Extract<WaitRequest, { tool: "take" }>;
    if (typeof take.inbox !== "string" || take.inbox === "") throw new RangeError("a take's inbox must be a non-empty string");
    if (take.mode !== "one" && take.mode !== "all") throw new RangeError(`a take's mode must be "one" or "all", got ${reprText(String(take.mode))}`);
    if (take.timeoutSeconds !== null && !isFiniteNumber(take.timeoutSeconds)) {
      throw new RangeError("a take's timeoutSeconds must be null or a finite number");
    }
  } else if (tool === "sleep") {
    const sleep = wait as Extract<WaitRequest, { tool: "sleep" }>;
    const hasUntil = sleep.until !== null;
    const hasFor = sleep.forSeconds !== null;
    if (hasUntil === hasFor) throw new RangeError("a sleep takes until or forSeconds, exactly one");
    if (hasUntil && (typeof sleep.until !== "string" || sleep.until === "")) {
      throw new RangeError("a sleep's until must be a non-empty string");
    }
    if (hasFor && !isFiniteNumber(sleep.forSeconds)) throw new RangeError("a sleep's forSeconds must be a finite number");
  } else {
    throw new TypeError(`a wait's tool must be "take" or "sleep", got ${isRecord(wait) ? reprText(String(tool)) : jsTypeName(wait)}`);
  }
}

/** Whether `value` is a finite JS number. */
function isFiniteNumber(value: unknown): boolean {
  return typeof value === "number" && Number.isFinite(value);
}

/** The closed sum a node's `run` resolves to. */
export type NodeResult = Output | Route | Grow | Pending;

/** Return whether `value` can key a `WeakMap` or be held in a `WeakSet`. CASE and LOOP use it. */
export function isWeakKey(value: unknown): value is object {
  return (typeof value === "object" && value !== null) || typeof value === "function";
}

/**
 * Parse a condition expression, prefixing its parse error with `subject`. The node kinds with
 * predicates use it.
 *
 * A `T(...)` constructor must name a type in `types`, the node's type table; with no table,
 * every constructor is refused.
 *
 * @throws ExpressionError `<subject> is not a valid boolean expression: <error>`, with the parse
 *   error as its `cause`.
 */
export function parseNamedCondition(
  expression: string,
  subject: string,
  types: ReadonlyMap<string, Type> | null = null,
): void {
  try {
    // An omitted table is an empty one, so every constructor is refused.
    parseCondition(expression, types ?? new Map());
  } catch (error) {
    if (!(error instanceof ExpressionError)) throw error;
    throw new ExpressionError(`${subject} is not a valid boolean expression: ${error.message}`, { cause: error });
  }
}

/**
 * Check one prompt template's `${...}` spans, prefixing an error with `subject`. AGENT uses it.
 *
 * @throws ExpressionError `<subject>: <error>`, with the template error as its `cause`.
 */
export function checkNamedPrompt(text: string, subject: string): void {
  try {
    promptRefs(text);
  } catch (error) {
    if (!(error instanceof ExpressionError)) throw error;
    throw new ExpressionError(`${subject}: ${error.message}`, { cause: error });
  }
}

/**
 * Return a frozen copy of `failure`, checked by `Failure`'s constructor, whose trace holds plain
 * `OriginRef`s. The run log and the run state use it.
 *
 * @throws TypeError or RangeError when a field breaks `Failure`'s contract, such as a trace that
 *   isn't an array of `OriginRef`s.
 */
export function retainFailure(failure: Failure): Failure {
  // Rebuilt once to check the trace with Failure's own errors,
  // before its items are copied without subclass state.
  const checked = new Failure(failure);
  const trace = checked.trace.map((origin) => Object.freeze(new OriginRef(origin.key)));
  const retained = new Failure({ ...checked, trace });
  Object.freeze(retained.trace);
  return Object.freeze(retained);
}

/** The requests one accepted invocation sent, rebuilt from its recorded steps. */
export interface RebuiltRequests {
  /**
   * Each `model_call` step's rebuilt request, by the step's index in the steps given; a step the
   * rebuild didn't reach (it stopped at a reply that failed the node, or never started) has none.
   */
  readonly requests: ReadonlyMap<number, RebuiltRequest>;
  /** Whether every recorded model call's `request_sha256` matched its rebuilt request. */
  readonly digestOk: boolean;
}

/** One `model_call` step's request, rebuilt from the invocation's render and the steps before it. */
export interface RebuiltRequest {
  /**
   * The request as the call sent it; `null` when it can't be rebuilt (a tool the recorded specs
   * lack, a structured model that can't be built).
   */
  readonly request: Request | null;
  /** Whether the step recorded the purpose and the `request_sha256` of the rebuilt request. */
  readonly digestOk: boolean;
}

/** What a node rebuilding its requests (`rebuildRequests`) reads beyond its inputs and steps. */
export interface RebuildCaps {
  /** The skills the run recorded, by name. */
  readonly skills: ReadonlyMap<string, Skill>;
  /**
   * The request's tool list as the run bound it: the specs of the tools `grants` grant, as the run
   * recorded them, each under the name the model saw, then `loadSkill` when given; `null` when a
   * grant can't be resolved.
   */
  readonly toolSpecs: (
    grants: readonly ToolGrant[],
    loadSkill: ToolSpec | null,
  ) => readonly ToolSpec[] | null;
  /** Builds the invocation's model to read its request settings off; never called. */
  readonly models: ModelAccess | null;
  /** The declared type of each of the node's locals, as `NodeCaps.localTypes` gives it. */
  readonly localTypes?: ReadonlyMap<string, Type | null>;
}

/**
 * A node-local `asserts:` clause was false, or couldn't be evaluated: thrown by a node's own
 * `run` template around its kind body (START included, after binding).
 *
 * `assertion` is the exact carrier the template checked, or `null` when the error was thrown
 * outside the template. Only the template's instance, marked as an assertion stage, gets the
 * failure code `failureCode`; the same class thrown by node code stays `node.execute`.
 */
export class NodeAssertFailed extends Error {
  override name = "NodeAssertFailed";
  /** The assertion that failed, or `null` outside the run template. */
  readonly assertion: Assertion | null;
  /** `assertion.failed` for a false condition, `condition.evaluate` for one that threw. */
  readonly failureCode: "assertion.failed" | "condition.evaluate";

  /**
   * Retain the assertion that classifies this template-owned failure; `assertion` defaults to
   * `null` and `failureCode` to `assertion.failed`.
   */
  constructor(
    message: string,
    options: {
      assertion?: Assertion | null;
      failureCode?: "assertion.failed" | "condition.evaluate";
      cause?: unknown;
    } = {},
  ) {
    const { assertion = null, failureCode = "assertion.failed", cause } = options;
    super(message, cause === undefined ? undefined : { cause });
    this.assertion = assertion;
    this.failureCode = failureCode;
  }
}

/**
 * @internal Return a stage marker's message: an `Error`'s own message, else the thrown value as
 * text, else its type name when it has no text. The evaluator uses it for a failure's message.
 */
export function markerMessage(original: unknown): string {
  if (original instanceof Error) return original.message;
  // A body may throw a value with no text (a null-prototype object, a throwing `toString`); the
  // marker must still wrap it.
  try {
    return String(original);
  } catch {
    return jsTypeName(original);
  }
}

/**
 * @internal Return a thrown value's `markerMessage` cut to `YIAN_MAX_FAILURE_CHARS` code points by
 * `cutText`. Used for a failure's message, which the run log records.
 */
export function failureMessage(error: unknown): string {
  return cutText(markerMessage(error), current().maxFailureChars);
}

/**
 * @internal Return an `Error`'s stack followed by its string `notes`, one per line, else `null`;
 * the notes in the text keep a cleanup note durable. Used for a failure's traceback.
 *
 * The stack's header (the error's name and message, before its first `at` frame), its frames
 * and each note are each cut to `YIAN_MAX_FAILURE_CHARS` code points by `cutText`, so a huge
 * message can't push the frames and notes out.
 */
export function tracebackText(error: unknown): string | null {
  if (!(error instanceof Error)) return null;
  const cap = current().maxFailureChars;
  const notes: unknown = (error as { notes?: unknown }).notes;
  const lines: string[] = [];
  if (typeof error.stack === "string") {
    const frames = /\n[ \t]+at /.exec(error.stack)?.index ?? error.stack.length;
    lines.push(cutText(error.stack.slice(0, frames), cap) + cutText(error.stack.slice(frames), cap));
  }
  for (const note of Array.isArray(notes) ? notes : []) {
    if (typeof note === "string") lines.push(cutText(note, cap));
  }
  return lines.length > 0 ? lines.join("\n") : null;
}

/** A stage marker retaining one value thrown by a node's body. Used by the evaluator. */
export class NodeBodyError extends Error {
  override name = "NodeBodyError";
  /** The exact value the body threw. */
  readonly original: unknown;

  /** Keep the exact thrown value, also as the `cause`. */
  constructor(original: unknown) {
    super(markerMessage(original), { cause: original });
    this.original = original;
  }
}

/** A stage marker for a failure of an evaluator-owned binding stage. Used by the evaluator. */
export class BindingStageError extends Error {
  override name = "BindingStageError";
  /** The binder's error. */
  readonly original: BindingError;

  /** Keep the binder's error apart from any error a node body throws, also as the `cause`. */
  constructor(original: BindingError) {
    super(markerMessage(original), { cause: original });
    this.original = original;
  }
}

/**
 * A carrier for an unexpected error from the deferred per-item binder, which must escape the
 * node. Used by the evaluator.
 */
export class DeferredBindingFault extends Error {
  override name = "DeferredBindingFault";
  /** The exact value the injected binder threw. */
  readonly original: unknown;

  /** Keep the exact host or setup fault, also as the `cause`. */
  constructor(original: unknown) {
    super(markerMessage(original), { cause: original });
    this.original = original;
  }
}

/** A carrier for an unexpected host fault from an injected capability. Used by the evaluator. */
export class CapabilityFault extends Error {
  override name = "CapabilityFault";
  /** The exact value the capability threw. */
  readonly original: unknown;

  /** Keep the exact host fault, so the evaluator can pass it on as it was, also as the `cause`. */
  constructor(original: unknown) {
    super(markerMessage(original), { cause: original });
    this.original = original;
  }
}

/** One read of a node's steer inbox: the `steer` step's `n`, the sends it consumed (their seqs) and their values, oldest first. */
export interface SteerRead {
  readonly n: number;
  readonly from: readonly number[];
  readonly values: readonly Value[];
}

/**
 * A node's steer inbox for one evaluation, which the engine opens (`NodeCaps.openSteerInbox`).
 * `read` is zero-wait: it consumes every event queued now, records them as one `steer` step, and
 * returns them; with nothing queued it records nothing and returns `null`.
 *
 * A host's `interrupt(run, inbox)` of this inbox aborts `interrupt`. The signal is separate from
 * the node's own `caps.signal`, so an interrupt never aborts a running tool.
 */
export interface SteerInbox {
  /** The top-level inbox read. */
  readonly inbox: string;
  /** Whether a `read` now would consume anything. */
  queued(): boolean;
  /** @throws CapabilityFault when the run refuses the step; the node signal's reason once it stopped. */
  read(): SteerRead | null;
  /** Aborted once a host interrupts this inbox's readers; already aborted when the node opens with one pending. */
  readonly interrupt: AbortSignal;
  /** The seq of the `interrupted` record naming this node that it hasn't honored yet, else `null`. */
  interruptedBy(): number | null;
}

/**
 * A host value given as a node's output (a tool's result, a human's answer) that isn't a yian
 * value; the node's failure is `contract.output`.
 */
export class HostOutputError extends TypeCheckError {}

/**
 * Take a host value given as a node's output in as a yian value.
 *
 * @throws HostOutputError when the value isn't a yian value, at the refused path.
 */
export function hostOutput(value: unknown): Value {
  try {
    return fromHost(value, "output");
  } catch (error) {
    if (!(error instanceof TypeCheckError)) throw error;
    throw new HostOutputError(error.message, { path: error.path, cause: error });
  }
}

/** A stage marker for an assertion failure of the base run template. Used by the evaluator. */
export class AssertionStageError extends Error {
  override name = "AssertionStageError";
  /** The template's assertion failure. */
  readonly original: NodeAssertFailed;

  /** Keep the template's assertion failure apart from node-body errors, also as the `cause`. */
  constructor(original: NodeAssertFailed) {
    super(markerMessage(original), { cause: original });
    this.original = original;
  }
}

/** A run's sessions on its memories, which every node bound to a memory shares. */
export interface NodeMemories {
  /**
   * Return the session on the memory `identity`.
   *
   * @throws Error when the run has none: a node bound to a memory the run didn't bind.
   */
  session(identity: string): MemorySession;
}

/**
 * Return the run's session on the memory `identity`, through `caps.memories`.
 *
 * @throws CapabilityFault carrying the error when the evaluator gave no memory sessions, or none on
 *   that memory: the run didn't bind a memory its nodes were bound to.
 */
export function memorySession(caps: NodeCaps, identity: string): MemorySession {
  if (caps.memories === undefined) throw new CapabilityFault(new TypeError("the evaluator gave no memory sessions"));
  try {
    return caps.memories.session(identity);
  } catch (error) {
    throw new CapabilityFault(error);
  }
}

/**
 * The capabilities a node's `run` and `forward` receive, as one object. The evaluator passes
 * every node its own `signal`, and the `scheduler` and `random` the host gave it, plus only what
 * the node's traits ask for.
 */
export interface NodeCaps {
  /** The node's own abort signal; a node that blocks cooperatively stops when it aborts. */
  readonly signal?: AbortSignal;
  /** The clock and cancellable sleep the node's waits run on; `realScheduler` when absent. */
  readonly scheduler?: Scheduler;
  /** A draw in [0, 1) for backoff jitter; `Math.random` when absent. */
  readonly random?: () => number;
  /** The run's folder (its log's folder), where the run's files go; absent when the run keeps no log. */
  readonly runDir?: string | null;
  /** Binds one element's input record; supplied to a node that binds per item (MAP). */
  readonly bindItem?: (element: Value, index: number) => ValueRecord;
  /** The node's durable steps; supplied to a node that records them (`needsSteps`). */
  readonly steps?: NodeSteps;
  /** The run's tool set; supplied to a node that declares `needsTools`. */
  readonly tools?: ToolSet;
  /** The run's memory sessions; supplied to a node that declares `needsMemories` in a run that has them. */
  readonly memories?: NodeMemories;
  /** The run's host-supplied skills by name; supplied to a node that declares `needsSkills`. */
  readonly skills?: ReadonlyMap<string, Skill>;
  /** The transient stream-chunk sink; supplied to a node that declares `needsLiveEvents`. */
  readonly emitStream?: (chunk: StreamChunk) => void;
  /** The transient in-loop observer sink (`IN_LOOP_EVENTS`); supplied with `needsLiveEvents`. */
  readonly emitEvent?: (
    event: ModelCallStarted | ModelCallFinished | ToolCallStarted | ToolCallFinished | ReplayDegraded,
  ) => void;
  /**
   * Opens the node's steer inbox; supplied to a node declaring `readsSteerInbox` when the engine
   * supplies steps. Opening registers the node as reading `inbox` until its outcome commits.
   */
  readonly openSteerInbox?: (inbox: string) => SteerInbox;
  /** The run's environment and model builder; supplied to a node that declares `needsModels`. */
  readonly models?: ModelAccess;
  /**
   * The declared type of each of the node's locals (its inputs, `output` for a post assert,
   * `event` for `render:`), so a field they declare reads null when it's absent; absent leaves
   * every local undeclared.
   */
  readonly localTypes?: ReadonlyMap<string, Type | null>;
}

/**
 * The base contract every node kind implements: a pure function of its input record.
 *
 * A node implements its `forward` body; the inherited `run` template resolves to one `NodeResult`
 * (`Output | Route | Grow | Pending`). The node never receives the pool (the engine's evaluator
 * binds its inputs and hands it a record) and never writes the pool: it describes its one output
 * as `new Output(value)` and the engine performs the write under the node id. Failure is a
 * rejection, not a variant.
 *
 * The traits are read-only fields a kind overrides (`override readonly isSpawner = true`). A
 * compile-owned node copy is shallow, so code that changes a node reassigns a field and never
 * mutates an array, record, map or carrier it read from one.
 */
export abstract class Node {
  /**
   * The closed-vocabulary tag used by compose/compile and node-owned diagnostics. Runtime never
   * dispatches on it. Set per kind.
   */
  abstract readonly kind: NodeKind;
  /**
   * Declares "I may grow the graph" (a spawner returns a `Grow`). Overridden by the spawner
   * kinds (CALL, MAP, LOOP); the evaluator gates the grow path on it.
   */
  readonly isSpawner: boolean = false;
  /**
   * Declares "my committed output is the run's input record" (START). That record is the
   * argument list around the input values, so it may go one level over the value nesting limit
   * while each input value stays within it.
   */
  readonly commitsInputRecord: boolean = false;
  /**
   * Declares "I bind my inputs per element through `caps.bindItem` rather than once up front"
   * (MAP). When set, the evaluator starts the record empty and supplies `caps.bindItem`;
   * otherwise it binds `params` once from the pool.
   */
  readonly bindsPerItem: boolean = false;
  /**
   * The reserved input key a per-item binder may project its local scope from; `null` when the
   * node has no recorded source collection for the local scope.
   */
  readonly itemSourceKey: string | null = null;
  /**
   * Nested-call depth policy for the engine's growth core: the depth stamped on a grow's subgraph
   * spawners and its terminal is `parentDepth + growDepthDelta`. `null` for a non-spawner (the
   * core does no depth work), `0` for LOOP (it inherits its parent's call depth), `1` for CALL
   * and MAP (each child is one level deeper).
   */
  readonly growDepthDelta: number | null = null;
  /**
   * Declares "I am the self-respawning fixpoint-iteration driver" (LOOP). The growth reducer
   * gates the origin-keyed single-live-record invariant on it: each iteration is a fresh driver
   * clone whose grow is attributed to `originId` and supersedes the prior live iteration.
   */
  readonly isLoop: boolean = false;
  /**
   * Declares "I can use host-supplied skills": the evaluator supplies the run's skills only when
   * the run boundary injected them and this is set. AGENT is the only built-in kind that opts in.
   */
  readonly needsSkills: boolean = false;
  /**
   * Declares "I look up or invoke tools": the evaluator supplies the run's own tool set, so
   * per-run tools and the per-run ask handler never leak between runs. TOOL and AGENT opt in.
   */
  readonly needsTools: boolean = false;
  /**
   * Declares "I call memory tools": the evaluator supplies the run's memory sessions, which the
   * run opened on the memories its host bound. TOOL and AGENT opt in.
   */
  readonly needsMemories: boolean = false;
  /**
   * Declares "I can publish transient in-loop observers": the evaluator supplies the stream and
   * event sinks the host has. AGENT is the only built-in kind that opts in.
   */
  readonly needsLiveEvents: boolean = false;
  /**
   * Declares "I record durable steps and read them back on a later start": the engine supplies
   * `caps.steps` only when this is set. The steps are opaque to the runtime.
   */
  readonly needsSteps: boolean = false;
  /**
   * Whether a woken evaluation of this node does no work a worker slot bounds: it returns from its
   * settled calls without calling anything, so the engine wakes it without a free slot.
   */
  readonly wakesWithoutWorker: boolean = false;
  /**
   * Declares "my success also commits the messages my evaluation added" (AGENT): the engine writes
   * a messages commit after my output commit, from `messageSteps`, and the reducer rebuilds its
   * value with `rebuildMessages`.
   */
  readonly producesMessages: boolean = false;
  /**
   * Declares "I read a steer inbox": the evaluator supplies `openSteerInbox`, and the host's
   * interrupt signal when the run boundary injected one. Writable: AGENT sets it from
   * `steer_inbox:`.
   */
  readsSteerInbox: boolean = false;
  /**
   * Declares "I call models": the evaluator supplies the run's model access only when the run
   * boundary injected it and this is set, so a run's environment and model builder never leak
   * into another run.
   */
  readonly needsModels: boolean = false;
  /**
   * Declares "my value only reads earlier pool values through my binding, and my `forward` is
   * pure and synchronous" (START, END, LOOP, CALL, MAP), so it can be rebuilt from the values its
   * binding read. The engine builds such a node's input route in every run.
   */
  readonly derived: boolean = false;

  /** The node's unique id within its (possibly namespaced) flow. */
  id: string;
  /** A human-friendly display title, or `null`. */
  title: string | null;
  /**
   * The canonical declared output contract: one exact value or a closed record with field-local
   * type evidence, shared by compiled and directly built nodes. `null` leaves the write
   * unenforced.
   */
  outputContract: OutputContract | null;
  /**
   * The node-side declared params, with no source: the flow owns the wiring. Stamped by the
   * compiler; the evaluator binds through `params` and the flow's wiring. `null` for a node that
   * declares no inputs.
   */
  params: readonly Parameter[] | null = null;
  /**
   * Ordered node-local contracts checked by the `run` template against the bound record before
   * `forward`. Stamped by the loader; empty for most nodes.
   */
  preAsserts: readonly Assertion[] = [];
  /** Ordered node-local contracts (reading `${output}`) checked after `forward`. */
  postAsserts: readonly Assertion[] = [];
  /**
   * Engine-baked commit redirect: write this node's output under `commitAs` instead of its own
   * id, and fire that target's out-edges. Set by the compile-owned growth builders on a
   * child-flow terminal (a CALL or MAP child END filler, or a MAP END-list filler) to point back
   * at the spawner id; `null` for an ordinary node.
   */
  commitAs: string | null = null;
  /**
   * "Attribute my grows and self-commit to this origin node": `null` on every ordinary node. Set
   * only on a self-respawning loop driver: the compiled loop `L` and each per-iteration clone
   * `L~k` set it to `L`, so the engine attributes every iteration's `Grow` and `Output` to the
   * one durable origin. A driver is the origin when `id === originId`, which gates "do not
   * self-prune".
   */
  originId: string | null = null;
  /**
   * Opaque authored-source identity for this node, for diagnostics; `null` for a source-free or
   * generated node. Unrelated to `originId`, which controls LOOP execution attribution.
   */
  origin: OriginRef | null;
  /**
   * Generated outer-to-inner authored ancestry for failures inside this node. Directly built
   * nodes start empty; compile-owned expansion derives it from enclosing spawners after the
   * authored origins are rewritten, so it stays outside local origin traversal.
   */
  originContext: readonly OriginRef[] = [];
  /**
   * Author-supplied static per-node config, baked at build time from the flow's `env:` default
   * merged with the node's own `env:` (the node wins). The engine never reads it; only a node's
   * own `forward` interprets the keys it understands (AGENT: `max_tool_iterations`), and unknown
   * keys are carried but ignored. `{}` when none is authored.
   */
  env: Readonly<Record<string, unknown>> = {};
  /**
   * The type table `isinstance(x, T)` looks `T` up in when this node evaluates its bindings,
   * asserts and predicates, keyed by type name. Filled at load from the names the node's
   * expressions use; empty when none are used.
   */
  types: ReadonlyMap<string, Type> = new Map();
  /**
   * Wiring param to the author-facing phrase for a reference error, for a synthesized param the
   * author never wrote (a CASE `__rN` from a `when:` ref, `__on` from `on:`), such as
   * `cases[1].when` or `on:`. Render-only and absent from run records; filled by the desugar that
   * mints the param. Empty for authored params, whose default label is `input '<param>'`.
   */
  wiringLabels: ReadonlyMap<string, string> = new Map();
  /**
   * Param to the full phrase a runtime binding error starts with in place of `input '<param>'`
   * or `binding '<param>'` (such as `node 'g' on`), where `wiringLabels` holds only the load-time
   * site phrase (`on:`). Not part of node identity or the manifest; filled by Compose for params
   * the author never wrote as inputs. Empty means the default text.
   */
  bindingLabels: ReadonlyMap<string, string> = new Map();

  /**
   * Build a node; each option defaults to `null`.
   *
   * @param nodeId The node's id within its flow.
   * @param options.title A display title.
   * @param options.outputContract The declared output contract: an exact `ValueContract` or
   *   `PartialRecordContract`.
   * @param options.origin The authored-source identity.
   * @throws TypeError for an output contract of another class, or an origin that isn't an
   *   `OriginRef`.
   */
  constructor(
    nodeId: string,
    options: { title?: string | null; outputContract?: OutputContract | null; origin?: OriginRef | null } = {},
  ) {
    const { title = null, outputContract = null, origin = null } = options;
    if (
      outputContract !== null &&
      !isExactly(outputContract, ValueContract) &&
      !isExactly(outputContract, PartialRecordContract)
    ) {
      throw new TypeError(
        "output_contract must be an exact ValueContract, PartialRecordContract, or null, " +
          `got ${jsTypeName(outputContract)}`,
      );
    }
    validateOptionalOrigin(origin, "node origin");
    this.id = nodeId;
    this.title = title;
    this.outputContract = outputContract;
    this.origin = origin;
  }

  /**
   * Return the generated ancestry plus the best local failure origin.
   *
   * `operationOrigin` wins when present; otherwise the node's `origin` is used. When neither
   * exists, only the ancestry is returned. Order and repeated origins are kept.
   *
   * @param operationOrigin A more precise authored identity for the failing operation.
   * @throws TypeError when the ancestry isn't an array of `OriginRef`s or the selected origin
   *   isn't an `OriginRef`.
   * @throws RangeError for an `OriginRef` whose key bypassed the constructor's check.
   */
  failureTrace(operationOrigin: OriginRef | null = null): readonly OriginRef[] {
    if (!Array.isArray(this.originContext)) throw new TypeError("node origin_context must be an array");
    for (const origin of this.originContext) {
      if (!(origin instanceof OriginRef)) throw new TypeError("node origin_context item must be an OriginRef");
      // Rebuilding checks the key, which an object made without the constructor never had checked.
      new OriginRef(origin.key);
    }
    const selected = operationOrigin ?? this.origin;
    validateOptionalOrigin(selected, "failure operation origin");
    if (selected === null) return this.originContext;
    new OriginRef(selected.key);
    return [...this.originContext, selected];
  }

  /**
   * Yield this node's source-neutral origins without following child flows: the node origin,
   * then the output contract's, the params', the pre-asserts' and the post-asserts', in
   * structural order.
   *
   * @param seen Carriers already visited by a graph traversal; a carrier is yielded once per
   *   `seen`. Omitted, a local traversal starts.
   * @throws TypeError when `origin` isn't an `OriginRef` or `null`.
   */
  *iterLocalOrigins(seen: Set<object> = new Set()): Generator<OriginRef> {
    validateOptionalOrigin(this.origin, "node origin");
    if (this.origin !== null) yield this.origin;
    if (this.outputContract !== null) yield* iterOriginCarrier(this.outputContract, seen);
    if (this.params !== null) {
      for (const parameter of this.params) yield* iterOriginCarrier(parameter, seen);
    }
    for (const assertion of this.preAsserts) yield* iterOriginCarrier(assertion, seen);
    for (const assertion of this.postAsserts) yield* iterOriginCarrier(assertion, seen);
  }

  /**
   * Rewrite this node's local origins, keeping shared carriers shared.
   *
   * This mutating hook is for a compile-owned private node copy, which may be shallow: it
   * assigns `origin`, `outputContract` and fresh `params`, `preAsserts` and `postAsserts` arrays
   * of rebuilt carriers, and never changes a carrier or an array in place.
   *
   * @param mapper A pure transform for each present origin; its errors propagate.
   * @param seen Carriers already rewritten in this graph, mapped to their rebuilt forms. Omitted,
   *   a local rewrite starts.
   * @throws TypeError when `origin` isn't an `OriginRef` or `null`, or `mapper` returns anything
   *   but an `OriginRef`.
   */
  rewriteLocalOrigins(mapper: OriginMapper, seen: Map<object, object> = new Map()): void {
    validateOptionalOrigin(this.origin, "node origin");
    this.origin = mapOrigin(this.origin, mapper);
    if (this.outputContract !== null) this.outputContract = rewriteOriginCarrier(this.outputContract, mapper, seen);
    if (this.params !== null) {
      this.params = this.params.map((parameter) => rewriteOriginCarrier(parameter, mapper, seen));
    }
    this.preAsserts = this.preAsserts.map((assertion) => rewriteOriginCarrier(assertion, mapper, seen));
    this.postAsserts = this.postAsserts.map((assertion) => rewriteOriginCarrier(assertion, mapper, seen));
  }

  /**
   * Return the static child flows whose origins belong to this node. The base has none; the
   * higher-order kinds override this without importing the compile layer that owns traversal.
   */
  originChildren(): readonly unknown[] {
    return [];
  }

  /**
   * Return the authored operation origin for one node-body error. The base returns the output
   * contract's origin at the error's path for a `HostOutputError` and the node's `origin` (`null`
   * for a source-free node) otherwise; a kind with finer source-neutral mapping overrides this and
   * falls back to `super` for a `HostOutputError`.
   */
  exceptionOrigin(error: unknown): OriginRef | null {
    if (error instanceof HostOutputError) return outputContractOrigin(this.outputContract, error.path);
    return this.origin;
  }

  /**
   * Return the failure code for one node-body error. The base returns `contract.output` for a
   * `HostOutputError` and `node.execute` otherwise; a kind with a finer source-neutral
   * classification overrides this and falls back to `super` for a `HostOutputError`.
   */
  exceptionCode(error: unknown): FailureCode {
    if (error instanceof HostOutputError) return "contract.output";
    return "node.execute";
  }

  /**
   * Return the node whose successors observe this node's absent output.
   *
   * A successful output commits under the baked `commitAs` redirect when present, else under the
   * node's own id. Route skip propagation follows the same boundary, so generated completion
   * nodes don't strand their parent's outgoing edges. A kind whose output target is chosen by
   * other node-owned state overrides this.
   */
  skippedOutputTarget(): string {
    return this.commitAs ?? this.id;
  }

  /**
   * Return the `n`s (from 1) of the steps, among `steps` (this node's recorded steps across its
   * starts, in order), that make up its new messages, ascending. Pure: the reducer calls it on
   * every append and replay.
   *
   * @throws TypeError for a node that doesn't produce messages.
   */
  messageSteps(_steps: readonly Readonly<JsonObject>[]): number[] {
    throw new TypeError(`node ${reprText(this.id)} produces no messages`);
  }

  /**
   * Rebuild this node's new messages from `steps`, the bodies `messageSteps` picked, in order, and
   * `output`, the value its output commit holds, as a typed value. Pure and deterministic.
   *
   * @param _localTypes The declared type of each of the node's locals, as `NodeCaps.localTypes`
   *   gives it.
   * @throws TypeError for a node that doesn't produce messages.
   */
  rebuildMessages(
    _steps: readonly Readonly<ValueRecord>[],
    _output: TypedValue | null,
    _localTypes: ReadonlyMap<string, Type | null> | null = null,
  ): TypedValue {
    throw new TypeError(`node ${reprText(this.id)} produces no messages`);
  }

  /**
   * Reject malformed node-owned execution setup before the body runs.
   *
   * The base requires a kind's own `forward`, then parses every condition through
   * `validateConditions`, so a malformed post-assert or predicate is reported before the body
   * has effects. A composite kind extends this for its own invariants, such as requiring its
   * compiled child flow.
   *
   * @throws Error when the node's class defines no `forward`.
   * @throws ExpressionError when a condition doesn't parse.
   */
  validateExecutionSetup(): void {
    if (this.forward === Node.prototype.forward) {
      throw new Error(`node ${reprText(this.id)} (${this.constructor.name}) defines no forward`);
    }
    this.validateConditions();
  }

  /**
   * Parse every condition this node evaluates. The base parses the node's assertions; a kind
   * that evaluates its own predicate (CASE `when`, LOOP `while`/`until`) extends this.
   *
   * @param options.path The static node path to name in the error instead of the node's id,
   *   used by a pre-run check over a whole flow.
   * @throws ExpressionError when a condition doesn't parse; the message names the node, the
   *   field and the expression.
   */
  validateConditions(options: { path?: string | null } = {}): void {
    this.validateAssertions(options);
  }

  /**
   * Parse every pre-assert and post-assert expression on this node.
   *
   * @param options.path The static node path to name in the error instead of the node's id,
   *   used by a pre-run check over a whole flow.
   * @throws ExpressionError when an assertion doesn't parse; the message names the node, the
   *   assertion list and the assertion.
   */
  validateAssertions(options: { path?: string | null } = {}): void {
    const { path = null } = options;
    for (const assertion of this.preAsserts) this.parseAssertion("pre-assert", assertion, path);
    for (const assertion of this.postAsserts) this.parseAssertion("post-assert", assertion, path);
  }

  /**
   * Check every prompt template this node renders. The base renders none; AGENT overrides
   * this. An override throws `ExpressionError` when a template's `${...}` span doesn't
   * parse, calls `isinstance` or a function that isn't a builtin, or calls a builtin with
   * arguments it doesn't take; the message names the node and the field.
   *
   * @param options.path The static node path to name in the error instead of the node's id,
   *   used by a pre-run check over a whole flow.
   */
  validatePrompts(options: { path?: string | null } = {}): void {}

  /** Parse one assertion's expression, naming the node and the assertion when it doesn't parse. */
  private parseAssertion(label: string, assertion: Assertion, where: string | null = null): void {
    // `||`, not `??`: an empty path names the node by its id.
    parseNamedCondition(
      assertion.expression,
      `node ${reprText(where || this.id)} ${label} ${this.shownAssertion(assertion)}`,
      this.types,
    );
  }

  /** Show an assertion in a message: its display, then its expression when that differs. */
  private shownAssertion(assertion: Assertion): string {
    const shown = reprText(assertion.display);
    return assertion.expression === assertion.display ? shown : `${shown} (\`${assertion.expression}\`)`;
  }

  /**
   * Run the node's complete pure-execution template.
   *
   * Declared defaults are filled first, then node-owned setup is validated (including that every
   * assertion and kind predicate parses), the pre-asserts run against the bound record, `forward`
   * runs, and the post-asserts check an `Output` against the inputs as bound. The promise resolves
   * to exactly one `NodeResult` (`Output | Route | Grow | Pending`). Every failure rejects it, with
   * the original error; the checks and the synchronous part of `forward` (up to its first `await`)
   * still run before the promise is returned.
   *
   * The evaluator runs the same template through `runForEvaluator`, so a kind changes what it
   * does through `forward`, never by overriding this.
   *
   * @param inputs The bound input record; the template fills defaults into a fresh copy.
   * @param caps The capabilities the node's traits ask for, passed to `forward`.
   */
  run(inputs: Readonly<Record<string, unknown>>, caps: NodeCaps = {}): Promise<NodeResult> {
    return this.runTemplate(inputs, false, null, caps);
  }

  /**
   * Run the base template with the stage markers and input capture. Used by the evaluator.
   *
   * A default's `BindingError` rejects as a `BindingStageError`, an assertion failure as an
   * `AssertionStageError`, and anything else the body throws as a `NodeBodyError`, except four
   * markers, which pass through as they are: `BindingStageError`, `DeferredBindingFault`,
   * `CapabilityFault` and `NodeBodyError`. An `AssertionStageError` thrown by a body is wrapped.
   *
   * @param inputs The bound input record.
   * @param caps The capabilities the node's traits ask for, passed to `forward`.
   * @param captureInputs Receives the completed input record after defaults and before the
   *   assertions or the body, before this returns its promise; never passed to `forward`. The
   *   checks and the synchronous part of `forward` (up to its first `await`) also run before the
   *   promise is returned.
   */
  runForEvaluator(
    inputs: Readonly<Record<string, unknown>>,
    caps: NodeCaps = {},
    captureInputs: ((inputs: Record<string, unknown>) => void) | null = null,
  ): Promise<NodeResult> {
    return this.runTemplate(inputs, true, captureInputs, caps);
  }

  /**
   * The template: fill declared defaults for omitted inputs, check the pre-asserts, run the
   * kind's `forward`, then check the post-asserts against `{...inputs, output}`.
   *
   * Defaults come first, so a pre-assert may read an input the caller omitted but the node
   * declares a default for; a present key, `null` included, shadows its default. START uses this
   * same template, so the parameter seam is its only default authority. The assertions read only
   * the bound record, never the pool: every assert reaching a node is node-local after reference
   * lowering, so the evaluator binds what it reads into the record.
   *
   * @param classify Whether to wrap the stage failures in the evaluator's markers.
   */
  private async runTemplate(
    inputs: Readonly<Record<string, unknown>>,
    classify: boolean,
    captureInputs: ((inputs: Record<string, unknown>) => void) | null,
    caps: NodeCaps,
  ): Promise<NodeResult> {
    // Everything before the first `await` runs inside the caller's call, the synchronous part of
    // `forward` included, so `captureInputs` has run by the time `runForEvaluator` returns;
    // `async` still turns a throw here into a rejection.
    const params = this.params ?? [];
    let record: Record<string, unknown>;
    try {
      record = applyDefaults(params, inputs, `node ${reprText(this.id)}`);
    } catch (error) {
      if (classify && error instanceof BindingError) throw new BindingStageError(error);
      throw error;
    }
    if (captureInputs !== null) captureInputs(record);
    this.validateExecutionSetup();
    const declared = localDeclared(caps.localTypes ?? null);
    const preFailure = this.failingAssertion("pre-assert", this.preAsserts, record, declared);
    if (preFailure !== null) throw classify ? new AssertionStageError(preFailure) : preFailure;
    // A snapshot for the post-asserts, taken before `forward`: a kind's `forward` may change the
    // record it receives, and the post-asserts must see the inputs as bound. Only paid when the
    // node has post-asserts.
    const postRecord = this.postAsserts.length > 0 ? (copyValue(record as ValueRecord) as ValueRecord) : null;
    let result: NodeResult;
    try {
      result = await this.forward(record, caps);
    } catch (error) {
      // The evaluator's own capabilities and deferred binder raise these four inside the body; an
      // assertion marker comes only from this template, so one a body throws is node code.
      if (
        !classify ||
        error instanceof BindingStageError ||
        error instanceof DeferredBindingFault ||
        error instanceof CapabilityFault ||
        error instanceof NodeBodyError
      ) {
        throw error;
      }
      throw new NodeBodyError(error);
    }
    // Only an `Output` has post-asserts checked: a spawner's grow path owns its own.
    if (result instanceof Output && postRecord !== null) {
      // `setField`, so an input named `__proto__` stays data; the snapshot is this run's own.
      setField(postRecord, "output", result.value as Value);
      const postFailure = this.failingAssertion("post-assert", this.postAsserts, postRecord, declared);
      if (postFailure !== null) throw classify ? new AssertionStageError(postFailure) : postFailure;
    }
    return result;
  }

  /**
   * Run the node as a pure function of its bound input record: the kind's body, which the `run`
   * template wraps.
   *
   * Every concrete kind defines it; the evaluator drives the `run` template, which checks the
   * assertions around this body exactly once. A kind that omits it gets this loud error (the setup
   * check reports it first). The evaluator gives a body error the code the node's `exceptionCode`
   * names (default `node.execute`).
   *
   * @param inputs The bound input record; the body may change it.
   * @param caps The capabilities the node's traits ask for.
   * @returns One `NodeResult` (`Output | Route | Grow | Pending`), or a promise of one.
   * @throws Error always, on the base.
   */
  forward(inputs: Record<string, unknown>, caps: NodeCaps): NodeResult | Promise<NodeResult> {
    throw new Error(`node ${reprText(this.id)} (${this.constructor.name}) defines no forward`);
  }

  /** Return this node's local variable declarations; the base declares none. */
  declaredVariables(): readonly Parameter[] {
    return [];
  }

  /**
   * Return the tool grants a model inside this node may call: tools and tool groups by name, and
   * some tools of a group. The run boundary checks them against the run's tools before any record
   * is written. The base grants none.
   */
  grantedTools(): readonly ToolGrant[] {
    return [];
  }

  /**
   * Return the grants of memory tools a model inside this node, or the node itself, may call, by
   * the memory input of the node's own flow. Never among `grantedTools()` or `hostToolCall()`, so
   * the run-start tool checks never see one. The base grants none.
   */
  memoryGrants(): readonly MemoryGrant[] {
    return [];
  }

  /**
   * Return the tool call this node makes directly, without a model. The run boundary checks it
   * against the run's tools before any record is written. The base makes none.
   */
  hostToolCall(): HostToolCall | null {
    return null;
  }

  /**
   * Shape the `tool_result` step for `complete` on one of this node's calls. Pure. The base node
   * registers no host calls, so it throws; ToolNode overrides it.
   *
   * @throws TypeError on the base. An override throws TypeCheckError when the output fails the
   *   tool's output type or ask's answer schema.
   */
  waitResultStep(call: CallView, _result: CompleteResult): JsonObject {
    throw new TypeError(`node ${reprText(this.id)} registers no host calls; it can't take ${reprText(call.callId)}'s result`);
  }

  /**
   * Return the node's declared inputs, without reserved wiring, in declaration order. Kinds with
   * control-only params override this.
   *
   * @throws TypeError when `params` isn't an exact array of `Parameter`s.
   */
  dataInputNames(): readonly string[] {
    const parameters = this.params ?? [];
    validateParameters(parameters);
    return parameters.map((parameter) => parameter.name);
  }

  /**
   * Rebuild the model requests one accepted invocation of this node sent, from its recorded steps.
   *
   * @param inputs The invocation's bound inputs as recorded, or `null` when none were.
   * @param steps The steps the node recorded across its starts, in order.
   * @param caps The run's recorded skills and tool specs, and model access.
   * @returns `null` by default: the node calls no model.
   */
  rebuildRequests(
    inputs: Readonly<Record<string, unknown>> | null,
    steps: readonly ValueRecord[],
    caps: RebuildCaps,
  ): Promise<RebuiltRequests | null> {
    return Promise.resolve(null);
  }

  /**
   * Return the reserved input keys the evaluator pre-resolves from `nodeWiring` and `pool`
   * before `run`, merged into the bound record. WAIT returns `{until: <ISO deadline>}`; MAP
   * returns `{over: <list>}`; the base reserves none and returns `{}`.
   *
   * @param nodeWiring The flow-owned wiring for this node.
   * @param pool The live pool the evaluator reads sources from.
   * @param options.bind Resolves a named source and applies the node-owned check before
   *   retaining its value; it takes the name, the source and the normalizing callable. Omitted
   *   for direct read-hook calls.
   */
  bindReserved(
    nodeWiring: ReadonlyMap<string, unknown>,
    pool: VariablePool,
    options: { bind?: ReservedBinder | null } = {},
  ): Record<string, unknown> {
    return {};
  }

  /**
   * Return the reserved author-wiring key names this node owns beyond its declared `params`: the
   * keys the wiring parity check allows. They are the static names of the keys `bindReserved`
   * resolves (WAIT: `until`; MAP: `over`), read at load time. The base reserves none.
   */
  reservedWiringKeys(): ReadonlySet<string> {
    return new Set();
  }

  /**
   * Return the author-facing phrase naming a reserved wiring key in a message, without the node:
   * it follows the node, as in `node 'w' wait until`. Default: `<kind> <key>`.
   *
   * @param key One of `reservedWiringKeys()`.
   */
  reservedKeyLabel(key: string): string {
    return `${this.kind} ${key}`;
  }

  /**
   * Return the author-facing phrase for a wiring `param` in a reference error: the label the
   * desugar recorded in `wiringLabels` for a synthesized param (`cases[1].when`, `on:`), else
   * `input '<param>'`.
   */
  wiringRefLabel(param: string): string {
    return this.wiringLabels.get(param) ?? `input ${reprText(param)}`;
  }

  /**
   * Return the first assertion that doesn't hold on `record`, as the error to throw, or `null`.
   *
   * Each assertion is evaluated with this node's type table and its locals' `declared` types. A
   * false condition fails with `assertion.failed`; one that throws an `ExpressionError` while
   * evaluating fails with `condition.evaluate`, that error as the `cause`. Callers run
   * `validateExecutionSetup` first, which has parsed every assertion and named the node in any
   * parse error.
   */
  private failingAssertion(
    label: string,
    assertions: readonly Assertion[],
    record: Record<string, unknown>,
    declared: DeclaredTypes,
  ): NodeAssertFailed | null {
    const subject = `node ${reprText(this.id)} ${label}`;
    for (const assertion of assertions) {
      let holds: boolean;
      try {
        holds = evaluateWhenRecord(assertion.expression, record as ValueRecord, this.types, declared);
      } catch (error) {
        if (!(error instanceof ExpressionError)) throw error;
        return new NodeAssertFailed(
          `${subject} ${this.shownAssertion(assertion)} could not be evaluated: ${error.message}`,
          { assertion, failureCode: "condition.evaluate", cause: error },
        );
      }
      if (!holds) return new NodeAssertFailed(`${subject} failed: ${assertion.display}`, { assertion });
    }
    return null;
  }
}
