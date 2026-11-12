/**
 * TOOL: a direct, deterministic invocation of one tool from the run's `ToolSet`.
 *
 * No LLM in the loop (distinct from an AGENT node's tool loop). The node's `input:` entries are
 * bindings like any other kind's, resolved by the engine's bind seam, so the tool sees only its
 * own record, never the live pool. The tool is looked up in the run's `ToolSet` (the global
 * registry when the run passed none) and invoked through its public schema. The result is taken
 * in as a yian value, and an optional `output:` declaration sets the node's output contract, which
 * the engine checks against it on every call.
 *
 * With `retries:` the node calls the tool again after a failed attempt, waiting an exponential
 * backoff with jitter on its scheduler, and records each retried attempt's failure and start as
 * node steps in the run log, so a resumed run continues the attempt count.
 *
 * A memory's tool (`tool_id: {lessons: read}`) is looked up in the memory the node's flow binds to
 * that input instead, and called on the run's session on it.
 *
 * Knows about: nodes/base, nodes/binding, nodes/contracts, nodes/ids, nodes/tool/questions,
 *              defaults, diagnostics, memory (memory tools), platform, tools and typesys.
 */

import { current } from "../../defaults.ts";
import { type FailureCode, reprText } from "../../diagnostics.ts";
import { MEMORY_SCOPE_LABEL, memoryTools } from "../../memory/tools.ts";
import { realScheduler } from "../../platform.ts";
import { ASK, BUILTIN_TOOL_IDS, type BuiltinToolId, isPending, SLEEP, TAKE, toolCallId } from "../../tools/context.ts";
import { isPermanentToolError, isToolCheckError, isTransientToolError, ToolCheckError } from "../../tools/errors.ts";
import { type MemoryGrant, visibleToolName } from "../../tools/grants.ts";
import { type ToolRef, ToolSet } from "../../tools/registry.ts";
import { YianRangeError, YianTypeError } from "../../typesys/errors.ts";
import { eventBytes, type JsonObject } from "../../typesys/json.ts";
import type { VariablePool } from "../../typesys/pool.ts";
import { deadlineUtc, formatRecordTime, parseDuration } from "../../typesys/temporal.ts";
import {
  buildValue,
  buildValueAs,
  fromHost,
  inboxName,
  jsonPart,
  MemoryRef,
  jsonSafeText,
  jsTypeName,
  rejectDeepNesting,
  renderValuePath,
  shownText,
  type Type,
  TypeCheckError,
  TypedValue,
  type ValueRecord,
  valueRepr,
} from "../../typesys/values.ts";
import {
  bindMemoryGrants,
  CapabilityFault,
  hostOutput,
  MEMORY_KEY_PREFIX,
  memoryKey,
  memorySession,
  Node,
  type NodeCaps,
  NodeKind,
  type CompleteResult,
  Output,
  Pending,
} from "../base.ts";
import { BindingError, type ReservedBinder } from "../binding.ts";
import { acceptsNull, type CallView, HostToolCall, outputContractType } from "../contracts.ts";
import { staticPathOf } from "../ids.ts";
import { dropRepeatedLabels, parseQuestions, type QuestionSpec } from "./questions.ts";

/**
 * A TOOL call failed after at least one retry; the message lists every attempt.
 *
 * Its `cause` is the last attempt's error. An attempt that started but has no recorded failure
 * (the run stopped or crashed inside it) shows as `unfinished`.
 */
export class ToolAttemptsFailed extends Error {
  override name = "ToolAttemptsFailed";
}

/** A host completed the node's pending call with an error; final, never retried. */
export class CompletedError extends Error {
  override name = "CompletedError";
}

/** A TOOL `ask` timed out with no answer and its declared output doesn't allow null (`tool.timeout`). */
export class ToolTimeoutError extends Error {
  override name = "ToolTimeoutError";
}

/**
 * @internal Return the seconds to wait before retrying after failure number `failure` (1-based).
 *
 * The delay doubles from `current().toolRetryDelaySeconds` up to
 * `current().toolRetryMaxDelaySeconds` and is drawn uniformly from its upper half with `random`.
 * An error `isTransientToolError` accepts whose `retryAfter` is a finite number of seconds, at
 * least 0, waits that long instead, up to the cap, and takes no draw.
 */
export function backoff(failure: number, error: unknown, random: () => number): number {
  const { toolRetryDelaySeconds: base, toolRetryMaxDelaySeconds: cap } = current();
  if (isTransientToolError(error)) {
    let after: unknown = null;
    // A foreign branded error may hold anything here, or throw from a getter.
    try {
      after = (error as { retryAfter?: unknown }).retryAfter;
    } catch {}
    if (typeof after === "number" && Number.isFinite(after) && after >= 0) return Math.min(cap, after);
  }
  // The cap is a safe integer, below 2^53, so exponent 53 reaches it for any base of at least 1;
  // capping the exponent keeps a huge `failure` from giving 0 * Infinity (NaN) for a zero base.
  const delay = Math.min(cap, base * 2 ** Math.min(failure - 1, 53));
  return delay / 2 + (random() * delay) / 2;
}

/** The `kind` of each step a TOOL records. */
const STEP_KIND = "tool_attempt";

/** Return the recorded form of a failed attempt's error: its type name and message. */
function errorStep(error: unknown): { type: string; message: string } {
  let name: string;
  // A thrown value may be anything, its `name` or `message` a throwing getter.
  try {
    name = error instanceof Error ? String(error.name) : jsTypeName(error);
  } catch {
    name = jsTypeName(error);
  }
  const type = shownText(jsonSafeText(name));
  let text: string;
  try {
    text = error instanceof Error ? String(error.message) : String(error);
  } catch {
    text = `<${type} object>`;
  }
  return { type, message: shownText(jsonSafeText(text)) };
}

/**
 * Return the final failure's message, naming every attempt's error up to `attempt`.
 *
 * Each error's text is capped by `shownText`, so the message grows only with the number of
 * attempts.
 */
function attemptsMessage(toolId: string, history: readonly ValueRecord[], attempt: number, error: unknown): string {
  const failed = new Map<number, unknown>();
  for (const step of history) if (step.event === "failed") failed.set(Number(step.attempt), step.error);
  const parts: string[] = [];
  for (let number = 1; number < attempt; number++) {
    const recorded = failed.get(number) as ValueRecord | null | undefined;
    const shown = recorded == null ? "unfinished" : `${recorded.type}: ${recorded.message}`;
    parts.push(`attempt ${number}: ${shown}`);
  }
  const last = errorStep(error);
  parts.push(`attempt ${attempt}: ${last.type}: ${last.message}`);
  return `tool ${reprText(toolId)} failed after ${attempt} attempts: ` + parts.join("; ");
}

/** Return a `contract.input` failure of a built-in's input. */
function inputError(message: string, cause?: unknown): BindingError {
  return new BindingError(message, { failureCode: "contract.input", cause });
}

/**
 * Return the seconds of a built-in's duration input `name`, or null when it is absent or null.
 *
 * A literal duration was checked at load and arrives as integer seconds; a `${...}` arrives as
 * text and is parsed here.
 *
 * @throws BindingError classified `contract.input` for anything else, `0s` and a plain number
 *   text included.
 */
function durationSeconds(value: unknown, tool: string, name: string): number | null {
  if (value === null || value === undefined) return null;
  if (typeof value === "string") {
    try {
      return parseDuration(value);
    } catch (error) {
      if (!(error instanceof RangeError)) throw error;
      throw inputError(`${tool} \`${name}\` ${valueRepr(value)}: ${error.message}`, error);
    }
  }
  const seconds = typeof value === "bigint" ? Number(value) : value;
  if (typeof seconds === "number" && Number.isSafeInteger(seconds) && seconds >= 1) return seconds;
  throw inputError(`${tool} \`${name}\` must be a duration, got ${valueRepr(value)}`);
}

/**
 * Return the record time `seconds` after `caps`' clock (`realScheduler` when absent) for the
 * duration input `name`, whose bound value is `value`.
 *
 * @throws BindingError classified `contract.input` when that time falls after the year 9999.
 */
function deadlineAfter(caps: NodeCaps, seconds: number, tool: string, name: string, value: unknown): string {
  try {
    return formatRecordTime((caps.scheduler ?? realScheduler).now() + seconds * 1000);
  } catch (error) {
    if (!(error instanceof RangeError)) throw error;
    throw inputError(`${tool} \`${name}\` ${valueRepr(value)}: the deadline falls after the year 9999`, error);
  }
}

/**
 * Return an ask's question as a `pending` step records it: `question`, `questions` (each with
 * `question`, `header`, `options` and `multi_select`), `answer_schema` and `title`, the null ones
 * left out.
 *
 * @throws TypeCheckError when a part isn't a yian value.
 */
function askRequestData(request: {
  question: unknown;
  questions: readonly QuestionSpec[] | null;
  answerSchema: readonly Readonly<Record<string, unknown>>[] | null;
  title: string | null;
}): JsonObject {
  const { question, questions, answerSchema, title } = request;
  const data = {
    question,
    ...(questions === null
      ? {}
      : {
          questions: questions.map((spec) => ({
            question: spec.question,
            header: spec.header,
            options: spec.options.map((option) => ({ label: option.label, description: option.description })),
            multi_select: spec.multiSelect,
          })),
        }),
    ...(answerSchema === null ? {} : { answer_schema: answerSchema }),
    ...(title === null ? {} : { title }),
  };
  return jsonPart(fromHost(data, "ask request"), [], "mapping") as JsonObject;
}

/**
 * A direct, deterministic invocation of one tool from the run's `ToolSet` (no LLM in the loop).
 *
 * The node's bound input record is checked by `ToolSet.validate` and passed to `ToolSet.invoke`;
 * the tool sees only its own record, never the live pool. The tool's `ToolContext` names this
 * node, with the call id `"<runId>/<node id>"` on a run's tool set, which a resumed re-run of the
 * node sees again. The result is taken in as a yian value; when `outputContract` is set (from an
 * optional authored `output:`), the engine validates every returned value against it.
 *
 * The arguments are checked once, before the first attempt. A failed call is retried up to
 * `retries` times unless the tool threw a permanent error or a check yian runs failed (a
 * `ToolCheckError`); argument and output checks are never retried. Before a retry the node waits
 * an exponential backoff (`current().toolRetryDelaySeconds` doubling up to
 * `current().toolRetryMaxDelaySeconds`, drawn from its upper half), or a transient error's
 * `retryAfter` up to that cap. Each attempt sees `ToolContext.attempt`; the call id stays the same.
 *
 * Through the engine's node steps the node records `{kind: "tool_attempt", event: "failed", ...}`
 * after each failure it will retry and `{kind: "tool_attempt", event: "started", ...}` before the
 * retry, so a resumed run continues the attempt count; an unfinished attempt (a crash or stop
 * inside it) uses up no retry. A failure without a retry propagates unchanged; after a retry the
 * node rejects with `ToolAttemptsFailed`, whose message lists every attempt's error.
 *
 * `take` and `sleep` are engine built-ins, not tools: the node binds and checks their inputs and
 * returns `Pending` with the wait request, and the engine decides, when it commits the outcome,
 * whether the call completes at once or the node waits.
 *
 * A host tool, or the run's `ask` handler, may return `pending()`: the node records a
 * `{kind: "pending", call_id, tool, arguments, deadline?, request?}` step and returns
 * `Pending([call_id])`; an `ask`'s `request` is the question its handler got (`askRequestData`),
 * and the host answers later through `complete`. The call id is the same on every attempt and
 * start, so a handler must be idempotent by `callId`: a crash before it returned calls it again.
 * Every evaluation reads `caps.steps.calls` first: an open call waits again without calling the
 * handler, an answered one returns its output or fails with its error (never retried), and an
 * `ask` closed `timed_out` returns null when the declared output is `Optional(...)`, else (no
 * `output:` included) fails with `tool.timeout`. `ask` sends yian's base surface (prompt,
 * questions, a `timeout` as a deadline) plus the host's declared extras.
 *
 * `inputTypes` maps each input name to the type the loader proved its binding has (a literal's
 * exact type, a typed binding's declared type, or a reference's inferred type); an input whose
 * type is not proven is absent.
 */
export class ToolNode extends Node {
  readonly kind = NodeKind.TOOL;
  /** Looks up and invokes its tool in the run's tool set. */
  override readonly needsTools: boolean = true;
  /** Calls a memory's tool on the run's session on that memory. */
  override readonly needsMemories: boolean = true;
  /** Records each retried attempt as a durable step, so a resumed run continues the count. */
  override readonly needsSteps: boolean = true;
  /**
   * A woken TOOL node returns from its settled call (or decides its take or sleep inline) without
   * calling its tool, so it takes no worker slot.
   */
  override readonly wakesWithoutWorker: boolean = true;
  /**
   * The tool's name as messages and the pending step show it: its id, or `<group>_<tool>` for a
   * group's tool.
   */
  toolId: string;
  /** The tool, looked up in the run's `ToolSet`: a bare id, or a group's tool. */
  toolRef: ToolRef;
  /**
   * Whether `toolRef` names a memory's tool, `{<memory input>: <tool>}`, which rides the reserved
   * `__memory_<input>` wiring instead of the run's `ToolSet`.
   */
  readonly memoryTool: boolean;
  /** How many times a failed tool call may be retried, at least 0. */
  retries: number;
  /** The proven type of each input's binding; an input whose type is not proven is absent. */
  inputTypes: ReadonlyMap<string, Type>;
  /** `ask`: the answer schema derived from `output:`, one entry per field; `[]` otherwise. */
  answerSchema: readonly Readonly<Record<string, unknown>>[];

  /**
   * Build a TOOL node.
   *
   * @param nodeId The node's unique id.
   * @param options.toolId The tool id, or a group's tool as a `ToolRef`, looked up in the run's
   *   `ToolSet`.
   * @param options.title A display title.
   * @param options.retries How many times a failed call may be retried, at least 0; `null` or
   *   omitted means `current().toolRetries`.
   * @param options.answerSchema For `ask`, the answer schema; `null` or omitted means `[]`.
   * @param options.memoryTool Whether `toolId` names a memory's tool by its input; defaults to
   *   `false`.
   * @throws TypeError when `retries` isn't a safe-integer number or `null`.
   * @throws RangeError when `retries` is negative, or a memory's tool names no memory input.
   */
  constructor(
    nodeId: string,
    options: {
      toolId: string | ToolRef;
      title?: string | null;
      retries?: number | null;
      answerSchema?: readonly Readonly<Record<string, unknown>>[] | null;
      memoryTool?: boolean;
    },
  ) {
    const { toolId, title = null, retries = null, answerSchema = null, memoryTool = false } = options;
    super(nodeId, { title });
    if (retries !== null) {
      const message = `node ${reprText(nodeId)}: retries must be a non-negative int, got ${valueRepr(retries)}`;
      if (!Number.isSafeInteger(retries)) throw new YianTypeError(message);
      if (retries < 0) throw new YianRangeError(message);
    }
    const ref = typeof toolId === "string" ? { group: null, name: toolId } : { ...toolId };
    this.toolRef = Object.freeze(ref);
    if (memoryTool && ref.group === null) {
      throw new YianRangeError(`node ${reprText(nodeId)}: a memory's tool is named \`{<memory>: <tool>}\``);
    }
    this.memoryTool = memoryTool;
    this.toolId = visibleToolName(this.toolRef.group, this.toolRef.name);
    this.retries = retries ?? current().toolRetries;
    this.inputTypes = new Map();
    this.answerSchema = answerSchema ?? [];
  }

  /**
   * The built-in `toolId` names (`take`, `sleep` or `ask`), or `null` for a host tool. A getter, so
   * it follows a reassigned `toolId` and is never an own field the manifest reads.
   */
  get builtin(): BuiltinToolId | null {
    return BUILTIN_TOOL_IDS.has(this.toolId as BuiltinToolId) ? (this.toolId as BuiltinToolId) : null;
  }

  /**
   * Return this node's tool id and group, input names, proven input types, and output type; `null`
   * for a memory's tool, which the run's tools never hold.
   */
  override hostToolCall(): HostToolCall | null {
    if (this.memoryTool) return null;
    return new HostToolCall(
      this.toolRef.name,
      this.toolRef.group,
      (this.params ?? []).map((parameter) => parameter.name),
      this.inputTypes,
      outputContractType(this.outputContract),
    );
  }

  /** Return the grant of the memory tool this node calls, if it names one. */
  override memoryGrants(): readonly MemoryGrant[] {
    if (!this.memoryTool) return [];
    return [Object.freeze({ input: this.toolRef.group!, tools: Object.freeze([this.toolRef.name]) })];
  }

  /** Declare `__memory_<input>` for the memory whose tool this node calls. */
  override reservedWiringKeys(): ReadonlySet<string> {
    return new Set(this.memoryGrants().map((grant) => memoryKey(grant.input)));
  }

  /**
   * Resolve the binding of the memory whose tool this node calls (`__memory_<input>`) to its
   * memory value; `{}` for a host tool.
   *
   * @throws Error when the wiring lacks the binding.
   * @throws BindingError (`contract.input`) when it holds something other than a memory.
   */
  override bindReserved(
    nodeWiring: ReadonlyMap<string, unknown>,
    pool: VariablePool,
    options: { bind?: ReservedBinder | null } = {},
  ): Record<string, unknown> {
    return bindMemoryGrants(this.id, this.memoryGrants(), "tool_id", nodeWiring, pool, options.bind ?? null);
  }

  /** Name the reserved `__memory_<input>` key as the authored `tool_id`. */
  override reservedKeyLabel(key: string): string {
    return key.startsWith(MEMORY_KEY_PREFIX) ? "tool_id" : super.reservedKeyLabel(key);
  }

  /**
   * Return `tool.timeout` for an `ask` nobody answered in time, a built-in's bad input's own code
   * (`contract.input`), and the base's code otherwise.
   */
  override exceptionCode(error: unknown): FailureCode {
    if (error instanceof ToolTimeoutError) return "tool.timeout";
    if (error instanceof BindingError) return error.failureCode;
    return super.exceptionCode(error);
  }

  /**
   * Call the tool, retrying failed attempts, and return its result, or `Pending` when the node
   * waits: on a host call that returned `pending()`, or on the wait a `take` or `sleep` asks the
   * engine to decide. A call held from an earlier evaluation is reused, never called again.
   *
   * `caps.tools` is the run's tool set (`new ToolSet()`, the global registry, when absent); the
   * backoff waits on `caps.scheduler` (`realScheduler` when absent) with `caps.signal`, and draws
   * its jitter from `caps.random` (`Math.random` when absent). `caps.steps` is this start's
   * recorded steps; absent, the node retries in memory, recording nothing, as the first start.
   *
   * @throws RangeError, TypeCheckError or ToolCheckError from `ToolSet.validate` when the tool is
   *   unknown or an argument doesn't fit, before any attempt.
   * @throws ToolAttemptsFailed when the call fails for good after at least one retry. When no
   *   retry happened, the tool's own error propagates unchanged.
   * @throws HostOutputError when the result isn't a yian value; never retried.
   * @throws The reason of `caps.signal` once it aborts during a wait; a tool's error after a stop
   *   propagates unchanged, unretried and unrecorded.
   * @throws BindingError classified `contract.input` when a `take`, `sleep` or `ask` duration or
   *   a `take` or `sleep` input is malformed.
   * @throws QuestionSpecError when an `ask`'s question list is invalid, before the handler runs.
   * @throws ToolCheckError when a pending call's arguments are over the run's event size limit.
   * @throws CompletedError when the host completed the held call with an error.
   * @throws ToolTimeoutError when an `ask` timed out and the declared output doesn't allow null.
   */
  override async forward(inputs: Record<string, unknown>, caps: NodeCaps): Promise<Output | Pending> {
    if (this.builtin === TAKE) return this.takeRequest(inputs, caps);
    if (this.builtin === SLEEP) return this.sleepRequest(inputs, caps);
    const tools = caps.tools ?? new ToolSet();
    // Outside a run each call gets a fresh id, so there is no call to hold.
    const callId = tools.runId === "" ? null : toolCallId(tools.runId, this.id);
    const held = callId === null ? undefined : caps.steps?.calls.find((call) => call.callId === callId);
    if (held !== undefined) return this.fromHeldCall(held);
    if (this.builtin === ASK) return await this.ask(inputs, caps, tools, callId);
    if (this.memoryTool) return await this.callTool(this.toolArguments(inputs), caps, this.memoryToolSet(inputs, caps, tools), callId);
    return await this.callTool(inputs, caps, tools, callId);
  }

  /**
   * Return `tools` with the memory tool this node calls as a local group, over the run's session on
   * the memory its input is bound to, so `toolRef` reaches it and never a run group by that name.
   *
   * @throws CapabilityFault carrying the error when the evaluator gave no memory sessions, or none
   *   on that memory.
   * @throws TypeError when the input holds no memory: a TOOL node can't name an Optional memory.
   */
  private memoryToolSet(inputs: Readonly<Record<string, unknown>>, caps: NodeCaps, tools: ToolSet): ToolSet {
    const [grant] = this.memoryGrants();
    const { input } = grant!;
    const value = inputs[memoryKey(input)];
    if (!MemoryRef.isMemoryRef(value)) throw new YianTypeError(`TOOL node ${reprText(this.id)}: \`${input}\` holds no memory`);
    const session = memorySession(caps, value.identity);
    const local = memoryTools(value, session, grant!, { kind: "tool", name: staticPathOf(this.id) });
    return tools.withLocalGroups(new Map([[input, local]]), MEMORY_SCOPE_LABEL);
  }

  /** Return the inputs a memory tool takes: the bound record without its reserved memory key. */
  private toolArguments(inputs: Readonly<Record<string, unknown>>): Record<string, unknown> {
    return Object.fromEntries(Object.entries(inputs).filter(([name]) => !name.startsWith(MEMORY_KEY_PREFIX)));
  }

  /**
   * Return the result of the call this node holds from an earlier evaluation: wait again on an
   * open call, return an answer or fail with a completed error, or return null for an `ask` that
   * timed out (see `timedOut`).
   *
   * @throws CompletedError when the host completed the call with an error.
   * @throws ToolTimeoutError when an `ask` timed out and the output doesn't allow null.
   */
  private fromHeldCall(held: CallView): Output | Pending {
    if (held.state === "open") return new Pending([held.callId]);
    if (held.state === "answered" && held.result !== null) {
      const { result } = held;
      if (Object.hasOwn(result, "error")) {
        throw new CompletedError(String((result.error as Readonly<JsonObject>).message));
      }
      return new Output(TypedValue.fromWire(result.output).value);
    }
    if (held.state === "closed" && held.closedReason === "timed_out") return this.timedOut(held);
    // TOOL calls aren't closed on a restart, and the terminal records end the run.
    throw new Error(`call ${reprText(held.callId)} is ${held.state}; a TOOL node never re-enters it`);
  }

  /**
   * Return null for an `ask` nobody answered in time when the declared output is `Optional(...)`.
   *
   * @throws ToolTimeoutError otherwise, no `output:` included.
   */
  private timedOut(held: CallView): Output {
    const typ = outputContractType(this.outputContract);
    // No `output:` reads as `str`, which a timeout can't give.
    if (typ !== null && acceptsNull(typ)) return new Output(null);
    throw new ToolTimeoutError(
      `TOOL node ${reprText(this.id)} (ask): nobody answered by ${held.deadline}; ` +
        "declare `output: !Optional(...)` to route on a timeout",
    );
  }

  /**
   * Record the `pending` step of a call whose handler returned `pending()`, and wait on it.
   *
   * @throws Error outside a durable run (no steps, no run id, or no log).
   * @throws ToolCheckError when the arguments are over the run's event size limit; nothing is
   *   recorded.
   */
  private registerPending(
    caps: NodeCaps,
    callId: string | null,
    tool: string,
    inputs: Record<string, unknown>,
    deadline: string | null,
    request: JsonObject | null = null,
  ): Pending {
    const steps = caps.steps;
    if (steps === undefined || callId === null || !steps.durable) {
      throw new Error(`tool ${reprText(tool)} returned pending() outside a durable run; give the run a log`);
    }
    const args = jsonPart(fromHost(inputs, `tool ${reprText(tool)} arguments`), [], "mapping") as ValueRecord;
    const step = {
      kind: "pending",
      call_id: callId,
      tool,
      arguments: args,
      ...(deadline === null ? {} : { deadline }),
      ...(request === null ? {} : { request }),
    };
    // The cap is on the whole step, measured as the run log measures an event.
    const size = eventBytes(step);
    if (size > steps.maxEventBytes) {
      throw new ToolCheckError(
        `tool ${reprText(tool)} pending call step is ${size} bytes, over the run's limit of ` +
          `${steps.maxEventBytes} (YIAN_MAX_EVENT_BYTES)`,
      );
    }
    steps.record(step);
    return new Pending([callId]);
  }

  /**
   * Ask through the run's ask handler: yian's base surface (`prompt`, `questions` or
   * `__questions`, parsed first, the latter without a composer's repeated option labels; `timeout`
   * as a deadline) plus the host's extras. An inline answer
   * is the output; `pending()` records the call with its deadline and waits.
   *
   * @throws QuestionSpecError when the question list is invalid, before the handler runs.
   * @throws BindingError classified `contract.input` when `timeout` isn't a duration or its
   *   deadline falls after the year 9999.
   */
  private async ask(
    inputs: Record<string, unknown>,
    caps: NodeCaps,
    tools: ToolSet,
    callId: string | null,
  ): Promise<Output | Pending> {
    const { prompt = null, questions = null, __questions = null, timeout = null, ...extra } = inputs;
    // Authored questions must be right as written; a composer model's repeated labels are dropped
    // instead, so a model's slip doesn't fail the run.
    const raw = questions ?? (__questions === null ? null : dropRepeatedLabels(__questions));
    const parsed = raw === null ? null : parseQuestions(raw);
    const seconds = durationSeconds(timeout, ASK, "timeout");
    // Computed before the handler runs so the request carries it; recorded only on pending().
    const deadline = seconds === null ? null : deadlineAfter(caps, seconds, ASK, "timeout", timeout);
    const answerSchema = this.answerSchema.length > 0 ? this.answerSchema : null;
    const answer = await tools.invokeTrusted(
      ASK,
      { prompt, questions: parsed, ...extra, answerSchema, title: this.title, nodeId: this.id, deadline },
      { node: this.id, ...(callId === null ? {} : { callId }), attempt: caps.steps?.starts ?? 1, signal: caps.signal },
    );
    if (isPending(answer)) {
      // The handler's request as the ask tool builds it, so a host can show the question again.
      const request = askRequestData({ question: prompt ?? "", questions: parsed, answerSchema, title: this.title });
      return this.registerPending(caps, callId, ASK, inputs, deadline, request);
    }
    return new Output(hostOutput(answer));
  }

  /**
   * Shape the `tool_result` step for a host's `complete` on this node's call: the value checked
   * against the declared output (for `ask`, the answer type without null), or the error.
   *
   * @throws TypeCheckError when the output isn't a yian value, nests deeper than
   *   `MAX_VALUE_NESTING`, or fails that type, or is null for an `ask` (null means nobody answered
   *   in time).
   */
  override waitResultStep(call: CallView, result: CompleteResult): JsonObject {
    if (Object.hasOwn(result, "error")) {
      const message = shownText(jsonSafeText(String((result as { error: unknown }).error)));
      return { kind: "tool_result", call_id: call.callId, error: { type: "CompletedError", message } };
    }
    const declared = outputContractType(this.outputContract);
    // The host's answer schema is the declared type without null: null means nobody answered.
    const typ = declared !== null && this.builtin === ASK ? declared.replace({ nullable: false }) : declared;
    const where = `complete(${reprText(call.callId)}) output`;
    const output = fromHost((result as { output: unknown }).output, where);
    // Here, not at the next advance, which would fail the node's output contract on it.
    rejectDeepNesting(output, where);
    if (this.builtin === ASK && output === null) {
      throw new TypeCheckError(`${where}: an ask's answer can't be null; null means nobody answered in time`);
    }
    let value: TypedValue;
    try {
      value = typ === null ? buildValue(output) : buildValueAs(typ, output);
    } catch (error) {
      // The check's reason is bare; name the answer, and where in it the check failed, as host
      // intake does.
      if (!(error instanceof TypeCheckError)) throw error;
      const at = error.path.length > 0 ? ` at ${renderValuePath(error.path)}` : "";
      throw new TypeCheckError(`${where}${at}: ${error.message}`, {
        path: error.path,
        cause: error,
      });
    }
    return { kind: "tool_result", call_id: call.callId, output: value.toWire() as JsonObject };
  }

  /**
   * Return a take's wait request from its bound `from`, `take` and `timeout`. The engine decides
   * whether it commits at once or waits, so nothing is recorded here.
   *
   * @throws BindingError classified `contract.input` when `from` isn't an inbox, `take` isn't
   *   `one` or `all`, or `timeout` isn't a duration or ends after the year 9999.
   */
  private takeRequest(inputs: Record<string, unknown>, caps: NodeCaps): Pending {
    const inbox = inboxName(inputs.from);
    if (inbox === null) throw inputError(`take \`from\` must be an inbox, got ${valueRepr(inputs.from ?? null)}`);
    const mode = inputs.take ?? "one";
    if (mode !== "one" && mode !== "all") throw inputError(`take \`take\` must be one or all, got ${valueRepr(mode)}`);
    const timeoutSeconds = durationSeconds(inputs.timeout, TAKE, "timeout");
    // The engine turns the timeout into a record-time deadline; one it can't write is the input's fault.
    if (timeoutSeconds !== null) deadlineAfter(caps, timeoutSeconds, TAKE, "timeout", inputs.timeout);
    return new Pending([], { wait: { tool: TAKE, inbox, mode, timeoutSeconds } });
  }

  /**
   * Return a sleep's wait request: `until` as a UTC record time, or `for` in seconds. A deadline
   * already past is still a request, which the engine commits at once.
   *
   * @throws BindingError classified `contract.input` unless exactly one of `until` and `for` is
   *   given, or when `until` isn't a date or datetime or `for` isn't a duration or ends after the
   *   year 9999.
   */
  private sleepRequest(inputs: Record<string, unknown>, caps: NodeCaps): Pending {
    const until = inputs.until ?? null;
    const forValue = inputs.for ?? null;
    if ((until === null) === (forValue === null)) throw inputError("sleep takes exactly one of `until` and `for`");
    if (until === null) {
      const forSeconds = durationSeconds(forValue, SLEEP, "for")!;
      // The engine turns the seconds into a record-time deadline; one it can't write is the input's fault.
      deadlineAfter(caps, forSeconds, SLEEP, "for", forValue);
      return new Pending([], { wait: { tool: SLEEP, until: null, forSeconds } });
    }
    if (typeof until !== "string") {
      throw inputError(`sleep \`until\` did not resolve to a date/datetime: ${valueRepr(until)}`);
    }
    let text: string;
    try {
      text = formatRecordTime(deadlineUtc(until));
    } catch (error) {
      if (!(error instanceof RangeError)) throw error;
      throw inputError(`sleep \`until\` ${valueRepr(until)}: ${error.message}`, error);
    }
    return new Pending([], { wait: { tool: SLEEP, until: text, forSeconds: null } });
  }

  /**
   * Call the host tool, retrying failed attempts, and return its result; a `pending()` result
   * records the call and waits.
   */
  private async callTool(
    inputs: Record<string, unknown>,
    caps: NodeCaps,
    tools: ToolSet,
    callId: string | null,
  ): Promise<Output | Pending> {
    const scheduler = caps.scheduler ?? realScheduler;
    const random = caps.random ?? Math.random;
    const { signal, steps } = caps;
    const history: ValueRecord[] = steps === undefined ? [] : [...steps.prior];
    // The first attempt of each start is the node's start itself; only in-evaluation retries
    // record a `started` step.
    let attempt = (steps?.starts ?? 1) + history.filter((step) => step.event === "started").length;
    let failures = history.filter((step) => step.event === "failed").length;
    const record = (step: ValueRecord): void => {
      history.push(step);
      steps?.record(step);
    };

    const call = await tools.validate(this.toolRef, inputs, { signal });
    const last = history.at(-1);
    // The previous start recorded a failure and stopped before the retry finished: wait again.
    if (last?.event === "failed") await scheduler.sleep(Number(last.retry_in) * 1000, signal);
    let result: unknown;
    for (;;) {
      try {
        result = await tools.invoke(call, { node: this.id, ...(callId === null ? {} : { callId }), attempt, signal });
        break;
      } catch (error) {
        // A stop or a host fault is neither retried nor recorded.
        if (signal?.aborted || error instanceof CapabilityFault) throw error;
        const retryable = !isPermanentToolError(error) && !isToolCheckError(error);
        if (retryable) failures += 1;
        if (!retryable || failures > this.retries) {
          if (!history.some((step) => step.event === "failed")) throw error;
          throw new ToolAttemptsFailed(attemptsMessage(this.toolId, history, attempt, error), { cause: error });
        }
        const delay = backoff(failures, error, random);
        record({ kind: STEP_KIND, event: "failed", attempt: BigInt(attempt), error: errorStep(error), retry_in: delay });
        await scheduler.sleep(delay * 1000, signal);
        attempt += 1;
        record({ kind: STEP_KIND, event: "started", attempt: BigInt(attempt) });
      }
    }
    // Outside the loop's `try`, so a pending call or a result that isn't a yian value is never retried.
    if (isPending(result)) return this.registerPending(caps, callId, this.toolId, inputs, null);
    return new Output(hostOutput(result));
  }
}
