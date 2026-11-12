/**
 * Process-local context for one AGENT invocation, the steps it records, and the run files its
 * replies' media go to.
 *
 * Knows about: diagnostics, events, llm errors, llm messages, llm_clients factory, nodes/base,
 *              nodes/contracts, platform, skills, tools, typesys, peer AGENT modules, node's
 *              crypto and node's fs.
 */

import { createHash, randomBytes } from "node:crypto";
import fs from "node:fs";
import { readFile, realpath } from "node:fs/promises";
import { join, resolve, sep } from "node:path";

import { reprText } from "../../diagnostics.ts";
import type { ModelCallPurpose } from "../../events.ts";
import { ProviderError } from "../../llm/errors.ts";
import { MediaPart, Message, NamedTool, type Part, Request, Response, Usage } from "../../llm/messages.ts";
import { isStorageError } from "../../llm_clients/auth/store.ts";
import type { ModelAccess } from "../../llm_clients/factory.ts";
import { type Scheduler, syncFolder, warn } from "../../platform.ts";
import type { Skill } from "../../skills/catalog.ts";
import type { ToolGrant } from "../../tools/grants.ts";
import type { ToolSet } from "../../tools/registry.ts";
import { type JsonObject, logJson } from "../../typesys/json.ts";
import { FileRef, jsonSafeText, type Type, type ValueRecord } from "../../typesys/values.ts";
import { CapabilityFault, type NodeCaps, type SteerInbox } from "../base.ts";
import type { NodeSteps } from "../contracts.ts";
import type { CloserReason } from "./results.ts";
import { foldSteps, type NumberedStep, type SteerRender, type StepExchange, steerValues } from "./messages.ts";

/** The system prompt of an AGENT that authors none. */
export const DEFAULT_SYSTEM =
  "You are an analysis agent inside a larger workflow. Follow the instructions " +
  "precisely, use the available tools to gather any facts you need, and return " +
  "a clear, self-contained answer. Ask the user only through an explicitly granted tool.";

/** The AGENT tool loop could not go on: a tool returned `pending()` outside a durable run. */
export class AgentLoopError extends Error {
  override name = "AgentLoopError";
}

/** What the AGENT loop receives to run one invocation. */
export interface AgentRunContext {
  readonly nodeId: string;
  /** The rendered prompt's messages, after the system message; the last is the user's. */
  readonly messages: readonly Message[];
  /**
   * The tool grants, a text read as a name grant; the model sees a group's tools as
   * `<group>_<tool>`.
   */
  readonly tools: readonly (string | ToolGrant)[];
  /** The effective model config, or `null`. */
  readonly llmConfig: Readonly<Record<string, unknown>> | null;
  /** The node's declared output type; `null` passes the reply's text through. */
  readonly outputType: Type | null;
  /** The structured-reply corrections allowed after the first call. */
  readonly retries: number;
  /**
   * The model turns with tool calls that run; -1 means no cap. The reply at the cap runs none of
   * its calls, and one more model call answers.
   */
  readonly maxToolIterations: number;
  /** The host's stop; absent for direct loop tests. */
  readonly signal?: AbortSignal;
  /** The run's tools and groups, which `tools` grants from. */
  readonly toolset: ToolSet | null;
  /** The transient text sink; absent means no streaming. */
  readonly emitStream?: NodeCaps["emitStream"];
  /** The transient model and tool call observer sink. */
  readonly emitEvent?: NodeCaps["emitEvent"];
  /** The authored or default system text plus the skill listing. */
  readonly systemPrompt: string;
  /** The granted skills served by `load_skill`, sorted by name. */
  readonly skills: readonly Skill[];
  /** The steer inbox this invocation reads, and its interrupt signal; absent without `steer_inbox`. */
  readonly steerInbox?: SteerInbox;
  /** Renders each steer event to text (the agent's `render:`); absent for the default rendering. */
  readonly renderSteer?: SteerRender;
  /** The run's environment and model builder. */
  readonly models: ModelAccess;
  /** The credentials the built model's route sends, for scrubbing; `() => []` until it is built. */
  sent: () => readonly string[];
  /** This start's durable steps; absent for direct loop tests, which record nothing. */
  readonly steps?: NodeSteps;
  /** The clock model-retry waits run on; the real one when absent. */
  readonly scheduler?: Scheduler;
  /** A draw in [0, 1) for model-retry backoff jitter; `Math.random` when absent. */
  readonly random?: () => number;
  /** The run's folder, where its media files go (`writeRunFile`); `null` or absent without a log. */
  readonly runDir?: string | null;
}

/** Return the credentials `ctx`'s model sends, or none when the read fails. */
export function sentCredentials(ctx: AgentRunContext): readonly string[] {
  try {
    return ctx.sent();
  } catch {
    // A failing read must not replace the error being scrubbed.
    return [];
  }
}

/** How one model call retries: the clock and jitter of its waits. */
export interface RetrySettings {
  readonly scheduler?: Scheduler;
  readonly random?: () => number;
}

/** The token counts a model call step records. */
type UsageCounts = Pick<Usage, "inputTokens" | "outputTokens" | "cacheReadTokens" | "cacheCreationTokens" | "reasoningTokens">;

/** What every `model_call` step records. Times are epoch milliseconds. */
interface CallFields {
  readonly purpose: ModelCallPurpose;
  readonly model: string;
  /** The request's digest (`requestSha256`). */
  readonly requestSha256: string;
  /** When the attempt started. */
  readonly startedAt: number;
  /** When the attempt's first event arrived; `null` before one did. */
  readonly firstTokenAt: number | null;
}

/** What a `model_call` step records of a call whose reply arrived. */
export interface ModelCallFields extends CallFields {
  /** The reply, its media in file form (`inlineToFile`). */
  readonly output: Message;
  readonly responseId: string | null;
  /** The provider's finish reason, or `interrupted` for a reply a host interrupt cut. */
  readonly finish: string | null;
  readonly usage: UsageCounts | null;
}

/** What a `model_call` step records of a failed attempt. */
export interface FailedCallFields extends CallFields {
  /** What arrived before the failure, or `null` before anything did. */
  readonly output: Message | null;
  /** The usage the provider reported before the failure, or `null`. */
  readonly usage: UsageCounts | null;
  /** Seconds until the retry, or `null` when the call is not retried. */
  readonly retryIn: number | null;
}

/** What a `tool_result` step records of one tool call. */
export interface ToolResultFields {
  /** The id of the call's `tool_call` part. */
  readonly callId: string;
  /** When the call started (epoch milliseconds); `null` for a call that never ran. */
  readonly startedAt: number | null;
  /** The text the model reads, exactly as its `ToolResultPart` holds it. */
  readonly output: string;
  readonly isError: boolean;
  /**
   * Set only on a result yian wrote in place of running the call: `skipped` for one a host
   * interrupt skipped, `crashed` for one a resume closed because its run was cut off before its
   * result, `timed_out` for an `ask_user` whose deadline passed, `interrupted` for a pending call a
   * host interrupt closed.
   */
  readonly closed?: "skipped" | CloserReason;
}

/** A tool call's result as an earlier start recorded it. */
export interface RecordedResult {
  readonly output: string;
  readonly isError: boolean;
  /** `skipped`, `crashed`, `timed_out` or `interrupted` for a result yian wrote in place of running the call, else `null`. */
  readonly closed: string | null;
  /** The `n` of its `tool_result` step. */
  readonly n: number;
}

/**
 * A model call an earlier start recorded, as `takeRecorded` gives it back: its reply, or the
 * error (`{kind, message}`, as recorded) it failed with, and the `n` of that step.
 */
export type RecordedCall =
  | { readonly n: number; readonly response: Response }
  | { readonly n: number; readonly failure: { readonly kind: string; readonly message: string } };

/** A steer read an earlier start recorded, as `takeRecordedSteers` gives it back: its step's `n` and the values it read. */
export interface RecordedSteer {
  readonly n: number;
  readonly values: readonly unknown[];
}

/** Attempts of a model call that an earlier start made and lost: their count, and the last one's wait. */
export interface PriorAttempts {
  readonly attempts: number;
  /** Seconds to wait before the next attempt. */
  readonly retryIn: number;
}

/**
 * Writes one AGENT start's steps through the engine's `NodeSteps`, each as a JSON object with a
 * `kind`: `model_call` for a model call that returned or an attempt that failed, `tool_result`
 * for a tool call's result. Each write returns the step's `n`, as the engine committed it. Without
 * engine steps (direct loop tests) it records nothing and numbers the steps itself.
 *
 * It also replays what earlier starts recorded (`NodeSteps.prior`), in order, grouped as
 * `foldSteps` groups them: each logical call is taken back (`takeRecorded`) in place of
 * calling the model, and the retried attempts after the last one are the next live call's
 * (`priorAttempts`). Each `tool_result` is found by its reply and its call id (`recordedResult`).
 * The `steer` steps before a logical call are handed back (`takeRecordedSteers`), in order, before
 * it, with the values the engine attached to each copy; those after the last call are handed back
 * after it. An `interrupted` step, the honor of a host interrupt, ends the replay where it was
 * written (`honored`). Steps of any other kind (`llm_config`, or a kind this version doesn't know)
 * are not the loop's and are skipped.
 */
export class AgentSteps {
  /** The run's folder, where reply media is stored; `null` without a log. */
  readonly runDir: string | null;
  readonly #steps: NodeSteps | undefined;
  readonly #clock: () => number;
  readonly #scrub: (error: unknown) => unknown;
  readonly #nodeId: string;
  #count = 0;
  /** The `n` of the last step this start wrote, or `null`. */
  #lastWritten: number | null = null;
  /** The recorded logical calls, each its failed attempts then its reply or its last failure. */
  readonly #calls: readonly (readonly NumberedStep[])[];
  /** How many of `#calls` were taken. */
  #taken = 0;
  /** The retried failed attempts recorded after the last logical call. */
  #trailing: readonly NumberedStep[];
  /** The recorded tool results, by the `n` of the reply they answer, then by call id. */
  readonly #results: ReadonlyMap<number, StepExchange["results"]>;
  /** The `n` of each recorded `pending` step, by its yian call id. */
  readonly #pending: ReadonlyMap<string, number>;
  /** How many flow calls the node recorded, by flow: the next one's scope number. */
  readonly #flowCalls = new Map<string, number>();
  /** The recorded steer steps, by the index in `#calls` of the logical call they preceded. */
  readonly #steers: Map<number, readonly NumberedStep[]>;
  /** The `by` of a recorded `interrupted` step: the interrupt an earlier start honored, else `null`. */
  readonly #honored: number | null;

  /**
   * @param steps This start's engine steps, or `undefined` to record nothing.
   * @param clock Epoch milliseconds, read for each recorded time.
   * @param runDir The run's folder, or `null`.
   * @param scrub Returns an error with every credential the call may have sent redacted.
   * @param nodeId The node, which a replay mismatch names.
   */
  constructor(
    steps: NodeSteps | undefined,
    clock: () => number,
    runDir: string | null,
    scrub: (error: unknown) => unknown,
    nodeId: string,
  ) {
    this.#steps = steps;
    this.#clock = clock;
    this.runDir = runDir;
    this.#scrub = scrub;
    this.#nodeId = nodeId;
    // Steps are numbered densely from 1 across the node's starts, so a step's `n` is its place.
    const { exchanges, open, honored } = foldSteps(steps?.prior ?? []);
    this.#calls = exchanges.map((exchange) => exchange.attempts);
    this.#trailing = open.attempts;
    this.#results = new Map(exchanges.map((exchange) => [exchange.attempts.at(-1)!.n, exchange.results]));
    this.#pending = new Map(
      (steps?.prior ?? []).flatMap((step, index) =>
        step["kind"] === "pending" ? [[step["call_id"] as string, index + 1] as const] : [],
      ),
    );
    for (const step of steps?.prior ?? []) {
      if (step["kind"] !== "pending" || step["flow"] !== true) continue;
      const tool = step["tool"] as string;
      this.#flowCalls.set(tool, (this.#flowCalls.get(tool) ?? 0) + 1);
    }
    this.#steers = new Map(
      [...exchanges, open].flatMap((exchange, index) => (exchange.steers.length > 0 ? [[index, exchange.steers]] : [])),
    );
    this.#honored = honored === null ? null : Number(honored.step["by"]);
  }

  /**
   * Take the steer reads recorded before the next logical call, oldest first, once.
   *
   * @returns Each one's step `n` and the values the engine attached to the step's copy; empty when
   *   none was recorded there, or they were taken.
   * @throws CapabilityFault carrying a RangeError naming the node and the step when a step's copy
   *   has no values (`steerValues`): the steps came from somewhere other than the engine.
   */
  takeRecordedSteers(): RecordedSteer[] {
    const recorded = this.#steers.get(this.#taken) ?? [];
    this.#steers.delete(this.#taken);
    return recorded.map(({ n, step }) => {
      try {
        return { n, values: steerValues(step) };
      } catch (error) {
        const message = `node ${reprText(this.#nodeId)} step ${n}: ${(error as RangeError).message}; resume through the engine`;
        throw new CapabilityFault(new RangeError(message, { cause: error }));
      }
    });
  }

  /** Whether a steer read was recorded before the next logical call, which `takeRecordedSteers` would hand back. */
  hasRecordedSteer(): boolean {
    return this.#steers.has(this.#taken);
  }

  /**
   * Whether replay is over and recorded failed attempts continue the next call, so it retries a
   * recorded request and no steer joins it.
   */
  continuesCall(): boolean {
    return !this.replaying() && this.#trailing.length > 0;
  }

  /** Whether a recorded call is still to be taken, so the loop is replaying. */
  replaying(): boolean {
    return this.#taken < this.#calls.length;
  }

  /** The `by` of the `interrupted` step an earlier start wrote, or `null`: the loop ends there once replay is over. */
  honored(): number | null {
    return this.#honored;
  }

  /**
   * Take the next recorded call, in place of a model call for `purpose` whose request has digest
   * `digest`.
   *
   * @returns The reply as recorded (media in file form) or the recorded error of a call that
   *   failed, and its step's `n`; `null` once every recorded call was taken.
   * @throws CapabilityFault carrying a RangeError when the reply or an attempt before it recorded
   *   another purpose or digest: the host or the yian version rebuilds a different request, so
   *   the recorded reply doesn't answer it. Nothing is taken or written; the run stays resumable.
   */
  takeRecorded(purpose: string, digest: string): RecordedCall | null {
    const call = this.#calls[this.#taken];
    if (call === undefined) return null;
    this.#check(call, purpose, digest);
    this.#taken += 1;
    const { n, step } = call.at(-1)!;
    if (!Object.hasOwn(step, "error")) return { n, response: recordedResponse(step) };
    const { kind, message } = step["error"] as { kind: string; message: string };
    return { n, failure: { kind, message } };
  }

  /**
   * Take the failed attempts recorded after the last logical call, each to be retried, for the
   * first live call (`purpose`, digest `digest`) to continue: their count, and the last one's
   * `retry_in`.
   *
   * @returns `null` while replaying, without such attempts, or once they were taken.
   * @throws CapabilityFault as `takeRecorded`.
   */
  priorAttempts(purpose: string, digest: string): PriorAttempts | null {
    if (this.replaying() || this.#trailing.length === 0) return null;
    const trailing = this.#trailing;
    this.#check(trailing, purpose, digest);
    this.#trailing = [];
    // A whole number of seconds reads back from the log as an int.
    return { attempts: trailing.length, retryIn: Number(trailing.at(-1)!.step["retry_in"]) };
  }

  /**
   * Return the result an earlier start recorded for tool call `callId` of the reply whose
   * `model_call` step is `n`, or `null`.
   */
  recordedResult(n: number, callId: string): RecordedResult | null {
    const recorded = this.#results.get(n)?.get(callId);
    if (recorded === undefined) return null;
    const { step } = recorded;
    const closed = step["closed"];
    return {
      output: step["output"] as string,
      isError: step["is_error"] === true,
      closed: typeof closed === "string" ? closed : null,
      n: recorded.n,
    };
  }

  /** Return the `n` of the `pending` step an earlier start recorded for yian call `callId`, or `null`. */
  recordedPending(callId: string): number | null {
    return this.#pending.get(callId) ?? null;
  }

  /** Throw the replay mismatch fault unless each of `steps` recorded `purpose` and `digest`. */
  #check(steps: readonly NumberedStep[], purpose: string, digest: string): void {
    for (const { n, step } of steps) {
      for (const [field, rebuilt] of [
        ["purpose", purpose],
        ["request_sha256", digest],
      ] as const) {
        const recorded = step[field];
        if (recorded === rebuilt) continue;
        throw new CapabilityFault(
          new RangeError(
            `node ${reprText(this.#nodeId)} step ${n}: recorded ${field} ${String(recorded)}, rebuilt ${rebuilt}; ` +
              "resume with the host that started the run",
          ),
        );
      }
    }
  }

  /**
   * Return the `n` the next step this start writes gets, so a call in flight can name its step
   * before writing it.
   */
  nextN(): number {
    return this.#steps === undefined ? this.#count + 1 : this.#steps.next;
  }

  /** Return the clock's reading, in epoch milliseconds. */
  now(): number {
    return this.#clock();
  }

  /**
   * Write a call whose reply arrived as a `model_call` step: `purpose`, `model`,
   * `request_sha256`, `started_at`, `first_token_at`, `output` (`yian.message.v1`), `response_id`,
   * `finish` and `usage`.
   *
   * @returns The step's `n`.
   * @throws Whatever `NodeSteps.record` throws.
   */
  modelCall(call: ModelCallFields, options?: { settled?: boolean }): number {
    // The reducer rebuilds new_messages purely, so it can't warn; the drop is reported here, live.
    // Without steps the engine writes no messages commit, so nothing is dropped.
    if (this.#steps !== undefined && call.purpose === "chat") {
      const media = call.output.parts.filter((part): part is MediaPart => part instanceof MediaPart && part.file === null);
      if (media.some((part) => part.data !== null)) {
        warn(
          `node ${reprText(this.#nodeId)} replied with media given inline, not as a run-folder file ` +
            "(a run without a log keeps a reply's media inline); new_messages leaves it out",
        );
      }
      if (media.some((part) => part.url !== null)) {
        warn(
          `node ${reprText(this.#nodeId)} replied with media given by URL, not as a run-folder file; ` +
            "new_messages leaves it out",
        );
      }
    }
    return this.#record(
      {
        ...this.#callStep(call),
        output: call.output.toJSON(),
        response_id: call.responseId,
        finish: call.finish,
        usage: usageStep(call.usage),
      },
      options,
    );
  }

  /**
   * Write a failed attempt as a `model_call` step: the fields `modelCall` writes up to
   * `first_token_at`, `output` (what arrived, media stored as run files; `null` when nothing did
   * or it can't be stored), `usage`, `error` (`{kind, message}`: a `ProviderError`'s kind, else
   * the error's name; the message scrubbed) and `retry_in`.
   *
   * @returns The step's `n`.
   * @throws Whatever `NodeSteps.record` throws.
   */
  failedModelCall(call: FailedCallFields, error: unknown): number {
    let output: JsonObject | null = null;
    try {
      output = call.output === null ? null : inlineToFile(call.output, this.runDir).toJSON();
    } catch {
      // Media that can't be stored must not replace the call's own error.
    }
    return this.#record({
      ...this.#callStep(call),
      output,
      usage: usageStep(call.usage),
      error: errorStep(this.#scrub(error)),
      retry_in: call.retryIn,
    });
  }

  /**
   * Write a tool call's result as a `tool_result` step: `call_id`, `started_at`, `output`,
   * `is_error`, and `closed` when given.
   *
   * @returns The step's `n`.
   * @throws Whatever `NodeSteps.record` throws.
   */
  toolResult(result: ToolResultFields, options?: { settled?: boolean }): number {
    const step: ValueRecord = {
      kind: "tool_result",
      call_id: result.callId,
      started_at: result.startedAt === null ? null : recordTime(result.startedAt),
      output: result.output,
      is_error: result.isError,
    };
    if (result.closed !== undefined) step["closed"] = result.closed;
    return this.#record(step, options);
  }

  /**
   * Write a call whose handler returned `pending()` as a `pending` step: `call_id` (the yian call
   * id), `tool`, `tool_call_id`, `reply` (the reply's `model_call` step `n`), `index` (the call's
   * place among the reply's tool calls), and `deadline` (a record time) for an `ask_user` with a
   * timeout. A flow tool's call adds `flow: true`, its `arguments`, which its scope starts with,
   * and that `scope`'s callsite, `<node>/<flow>#<k>`, `k` counting the node's calls of the flow.
   *
   * @returns The step's `n`.
   * @throws Whatever `NodeSteps.record` throws.
   */
  pending(call: {
    callId: string;
    tool: string;
    toolCallId: string;
    reply: number;
    index: number;
    deadline: string | null;
    flowArguments?: Readonly<Record<string, unknown>> | null;
  }): number {
    const { callId, tool, toolCallId, reply, index, deadline, flowArguments = null } = call;
    const step: ValueRecord = {
      kind: "pending",
      call_id: callId,
      tool,
      tool_call_id: toolCallId,
      reply: BigInt(reply),
      index: BigInt(index),
    };
    if (deadline !== null) step["deadline"] = deadline;
    if (flowArguments === null) return this.#record(step, { settled: true });
    const k = this.#flowCalls.get(tool) ?? 0;
    step["flow"] = true;
    step["arguments"] = flowArguments as ValueRecord;
    step["scope"] = `${this.#nodeId}/${tool}#${k}`;
    const n = this.#record(step, { settled: true });
    this.#flowCalls.set(tool, k + 1);
    return n;
  }

  /**
   * Write the honor of a host interrupt as an `interrupted` step: `by`, the seq of the
   * `interrupted` record.
   *
   * @returns The step's `n`.
   * @throws Whatever `NodeSteps.record` throws.
   */
  interrupted(by: number): number {
    return this.#record({ kind: "interrupted", by: BigInt(by) });
  }

  /** The fields every `model_call` step starts with. */
  #callStep(call: CallFields): ValueRecord {
    return {
      kind: "model_call",
      purpose: call.purpose,
      model: call.model,
      request_sha256: call.requestSha256,
      started_at: recordTime(call.startedAt),
      first_token_at: call.firstTokenAt === null ? null : recordTime(call.firstTokenAt),
    };
  }

  #record(step: ValueRecord, options?: { settled?: boolean }): number {
    this.#lastWritten = this.#steps === undefined ? ++this.#count : this.#steps.record(step, options);
    return this.#lastWritten;
  }

  /** The `n` of the last step this start wrote, or `null` before any. */
  lastWritten(): number | null {
    return this.#lastWritten;
  }
}

/** Return epoch milliseconds `ms` as a record time, such as `2026-10-20T14:02:11.123Z`. */
function recordTime(ms: number): string {
  return new Date(ms).toISOString();
}

/** Return the recorded token counts of `usage` (ints; `null` where unreported), or `null`. */
function usageStep(usage: UsageCounts | null): ValueRecord | null {
  if (usage === null) return null;
  const count = (value: number | null) => (value === null ? null : BigInt(value));
  return {
    input_tokens: count(usage.inputTokens),
    output_tokens: count(usage.outputTokens),
    cache_read_tokens: count(usage.cacheReadTokens),
    cache_creation_tokens: count(usage.cacheCreationTokens),
    reasoning_tokens: count(usage.reasoningTokens),
  };
}

/** The recorded kind of an auth-store fault, whose name is `AuthError` (`StorageError`). */
export const STORAGE_ERROR_KIND = "StorageError";

/**
 * Return a failed call's recorded error: its kind (a `ProviderError`'s kind, `StorageError` for an
 * auth-store fault, else the error's name) and message, lone surrogates as `?`.
 */
function errorStep(error: unknown): { kind: string; message: string } {
  let kind: string;
  if (error instanceof ProviderError) kind = error.kind;
  else if (isStorageError(error)) kind = STORAGE_ERROR_KIND;
  else kind = error instanceof Error ? error.name : typeof error;
  const message = error instanceof Error ? error.message : String(error);
  return { kind: jsonSafeText(kind), message: jsonSafeText(message) };
}

/** Return the reply a successful `model_call` step recorded, as a `Response`. */
export function recordedResponse(step: ValueRecord): Response {
  const usage = step["usage"] as ValueRecord | null;
  const count = (value: unknown) => (value == null ? null : Number(value));
  return new Response({
    message: Message.fromJSON(step["output"]),
    usage:
      usage === null
        ? null
        : new Usage({
            inputTokens: Number(usage["input_tokens"]),
            outputTokens: Number(usage["output_tokens"]),
            // The step doesn't record the total; the provider's is the sum of these two.
            totalTokens: Number(usage["input_tokens"]) + Number(usage["output_tokens"]),
            cacheReadTokens: count(usage["cache_read_tokens"]),
            // Absent from a step an older version wrote.
            cacheCreationTokens: count(usage["cache_creation_tokens"]),
            reasoningTokens: count(usage["reasoning_tokens"]),
          }),
    finishReason: step["finish"] as string | null,
    responseId: step["response_id"] as string | null,
  });
}

/** Unwind before starting another external effect once the host stopped the invocation. */
export function checkStopped(ctx: Pick<AgentRunContext, "signal">): void {
  ctx.signal?.throwIfAborted();
}

// --------------------------------------------------------------------------------------- //
// The host interrupt on replay. A replaying loop never reads the interrupt: the recorded calls
// already happened. A recorded honor step ends it at the same point.
// --------------------------------------------------------------------------------------- //

/**
 * Whether the loop is to honor a host interrupt now, as it reads it between effects: never while
 * `steps` is replaying, since the recorded calls already happened; once replay is over, when an
 * earlier start recorded its honor or the interrupt signal aborted.
 */
export function interruptRequested(interrupt: AbortSignal | undefined, steps: AgentSteps): boolean {
  return !steps.replaying() && (steps.honored() !== null || interrupt?.aborted === true);
}

/**
 * Return the digest a `model_call` step records for `request`: the SHA-256 of the `logJson` text
 * (canonical layout; ints of any size, so a long int argument hashes) of its messages, tools,
 * tool choice, parallel tool calls and output format. Media is hashed in the form the request
 * holds it, the file form for a reply.
 */
export function requestSha256(request: Request): string {
  return new RequestDigest().digest(request);
}

// The log JSON of a request opens with its messages: `logJson` sorts keys, and the other keys sort
// after `messages`. A message object has three keys or more, so none is written escaped.
const MESSAGES_OPEN = '{"messages":[';

/**
 * Takes the digests of a series of requests, each the one `requestSha256` gives.
 *
 * A request whose messages start with the last request's messages, the same objects, hashes only
 * the messages it adds, so a growing tool loop serializes each message once; any other request
 * hashes all of its messages.
 */
export class RequestDigest {
  /** The messages hashed so far, the last request's. */
  #messages: readonly Message[] = [];
  /** The hash of `MESSAGES_OPEN` and the log JSON of `#messages`, comma separated. */
  #hash = createHash("sha256").update(MESSAGES_OPEN, "utf8");

  /** Return the digest a `model_call` step records for `request`. */
  digest(request: Request): string {
    const { messages } = request;
    const extending =
      messages.length >= this.#messages.length && this.#messages.every((message, i) => messages[i] === message);
    const from = extending ? this.#messages.length : 0;
    // All new messages are serialized before any is hashed, so one that can't be leaves the state as it was.
    const texts = messages.slice(from).map((message, i) => {
      const text = logJson(message.toJSON());
      return from + i === 0 ? text : `,${text}`;
    });
    if (!extending) this.#hash = createHash("sha256").update(MESSAGES_OPEN, "utf8");
    for (const text of texts) this.#hash.update(text, "utf8");
    this.#messages = messages;
    const choice = request.toolChoice;
    const format = request.outputFormat;
    const rest = logJson({
      tools: request.tools.map((tool) => ({
        name: tool.name,
        description: tool.description,
        parameters: tool.parameters,
        strict: tool.strict,
      })),
      tool_choice: choice instanceof NamedTool ? { name: choice.name } : choice,
      parallel_tool_calls: request.parallelToolCalls,
      output_format: format === null ? null : { name: format.name, schema: format.schema, strict: format.strict },
    });
    // `rest` without its opening brace continues the request's object after its messages.
    return this.#hash.copy().update(`],${rest.slice(1)}`, "utf8").digest("hex");
  }
}

// --------------------------------------------------------------------------------------- //
// Media files
// --------------------------------------------------------------------------------------- //

// The extensions of the media types models send; any other type is stored as `.bin`.
const MEDIA_EXTENSIONS: ReadonlyMap<string, string> = new Map([
  ["image/png", "png"],
  ["image/jpeg", "jpg"],
  ["image/gif", "gif"],
  ["image/webp", "webp"],
  ["image/svg+xml", "svg"],
  ["audio/wav", "wav"],
  ["audio/mpeg", "mp3"],
  ["application/pdf", "pdf"],
  ["text/plain", "txt"],
]);

/**
 * Store `bytes` as a file of the run in folder `dir` and return its reference.
 *
 * The file is `<dir>/media/<sha256>.<ext>`, mode 0600, in a `media` folder made with mode 0700
 * (`dir` must exist); `ext` follows `mime` (`.bin` when unknown). The same bytes of the same type
 * keep one file. The file and the folder are synced before this returns, so a record naming the
 * file can be written next.
 *
 * @returns The file's reference: its path relative to the run folder (`media/<sha256>.<ext>`) as
 *   `uri`, so it still holds when the folder is copied elsewhere; `mime`; and its file name as `name`.
 * @throws Error, writing no file, when `media` exists but isn't a folder (a symlink included), or,
 *   on POSIX, is another user's or its group or others can write it.
 */
export function writeRunFile(dir: string, bytes: Uint8Array, mime: string): FileRef {
  const media = resolve(dir, "media");
  try {
    fs.mkdirSync(media, { mode: 0o700 });
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== "EEXIST") throw error;
  }
  // A `media` already there may not be ours (a log in a shared folder such as /tmp): a symlink
  // would lead writes out, which `fileToInline` then refuses to read, and another writer could
  // swap the bytes a step names.
  const info = fs.lstatSync(media);
  const refused = (why: string) => new Error(`the run's media folder ${media} ${why}; give the run a folder of its own`);
  if (!info.isDirectory()) throw refused("is not a folder");
  // Owner and mode are checked only on POSIX: Windows has no `getuid`, and its modes read 0o666.
  const uid = process.getuid?.();
  if (uid !== undefined && (info.uid !== uid || (info.mode & 0o022) !== 0)) {
    throw refused("is another user's or others can write it");
  }
  // A new `media` entry is durable only once the run folder holding it is synced. Synced every
  // time: a crash after an earlier call made `media` may have left the folder unsynced.
  syncFolder(resolve(dir));
  const ext = MEDIA_EXTENSIONS.get(mime.split(";")[0]!.trim().toLowerCase()) ?? "bin";
  const name = `${createHash("sha256").update(bytes).digest("hex")}.${ext}`;
  const path = join(media, name);
  // Two writers of the same bytes are harmless: each renames its own complete copy into place.
  if (!fs.existsSync(path)) {
    // Written under a temporary name and renamed, so a crash never leaves a short file at `path`.
    const temporary = join(media, `.${name}.${randomBytes(8).toString("hex")}`);
    try {
      const fd = fs.openSync(temporary, "wx", 0o600);
      try {
        let offset = 0;
        while (offset < bytes.length) offset += fs.writeSync(fd, bytes, offset);
        fs.fsyncSync(fd);
      } finally {
        fs.closeSync(fd);
      }
      fs.renameSync(temporary, path);
    } finally {
      if (fs.existsSync(temporary)) fs.unlinkSync(temporary);
    }
  }
  syncFolder(media);
  return new FileRef({ uri: `media/${name}`, mime, name });
}

/**
 * Return the bytes of a media part's base64 `data`.
 *
 * @throws RangeError when `data` isn't base64 (padding may be left out; url-safe letters,
 *   whitespace and line breaks are refused), since decoding it anyway would store, and later
 *   send, different bytes.
 */
function base64Bytes(data: string, mime: string): Buffer {
  const bytes = Buffer.from(data, "base64");
  if (bytes.toString("base64").replace(/=+$/, "") !== data.replace(/=+$/, "")) {
    throw new RangeError(`${mime} media data isn't base64`);
  }
  return bytes;
}

/** Return `message` with `parts` in place of its parts, everything else kept. */
function withParts(message: Message, parts: readonly Part[]): Message {
  return new Message({
    role: message.role,
    parts,
    author: message.author,
    protocol: message.protocol,
    providerPayload: message.providerPayload,
  });
}

/**
 * Return `message` with each inline (`data`) media part stored as a run file under `dir` and
 * given in its file form; its type, name and payload stay.
 *
 * A message without inline media, or any message when `dir` is `null` (a run without a log), is
 * returned as is.
 *
 * @throws RangeError when a part's `data` isn't base64; nothing is stored then.
 */
export function inlineToFile(message: Message, dir: string | null): Message {
  if (dir === null || !message.parts.some((part) => part instanceof MediaPart && part.data !== null)) return message;
  // Every part is decoded before any file is written.
  const decoded = message.parts.map((part) =>
    part instanceof MediaPart && part.data !== null ? base64Bytes(part.data, part.mime) : null,
  );
  const parts = message.parts.map((part, index) => {
    const bytes = decoded[index];
    if (!(part instanceof MediaPart) || bytes == null) return part;
    return new MediaPart({
      mime: part.mime,
      file: writeRunFile(dir, bytes, part.mime),
      name: part.name,
      providerPayload: part.providerPayload,
    });
  });
  return withParts(message, parts);
}

/**
 * Return `request` with each file media part given its bytes inline (`data`, read from the
 * file's `uri` resolved against the run folder `runDir`); its type, name and payload stay. This
 * is the form sent to a provider, so provider adapters never see a file reference.
 *
 * A request without file media is returned as is.
 *
 * @throws RangeError (as a rejection) when a `uri`, as written or with symlinks followed, names
 *   no path inside the run's `media` folder (an absolute path elsewhere, a `..` escape, the log, a
 *   symlink out, or a `media` folder that is itself a symlink).
 * @throws Error (as a rejection) naming the path when a file can't be read.
 */
export async function fileToInline(request: Request, runDir: string): Promise<Request> {
  const filed = (part: Part): part is MediaPart => part instanceof MediaPart && part.file !== null;
  if (!request.messages.some((message) => message.parts.some(filed))) return request;
  const messages: Message[] = [];
  for (const message of request.messages) {
    if (!message.parts.some(filed)) {
      messages.push(message);
      continue;
    }
    const parts: Part[] = [];
    for (const part of message.parts) {
      if (!filed(part)) {
        parts.push(part);
        continue;
      }
      const uri = part.file!.uri;
      const path = resolve(runDir, uri);
      const outside = new RangeError(`media file ${uri} is not inside the run's media folder`);
      // Checked on the path as written first, so an escaping uri is refused whether or not it exists.
      if (!path.startsWith(resolve(runDir, "media") + sep)) throw outside;
      let bytes: Buffer;
      try {
        // Then on real paths, with only the run folder resolved, so neither a symlinked `media`
        // folder nor a symlink inside it can lead a read out of the folder.
        const media = join(await realpath(runDir), "media");
        const real = await realpath(path);
        if (!real.startsWith(media + sep)) throw outside;
        bytes = await readFile(real);
      } catch (error) {
        if (error instanceof RangeError) throw error;
        throw new Error(`media file ${path} can't be read: ${error instanceof Error ? error.message : String(error)}`, {
          cause: error,
        });
      }
      parts.push(
        new MediaPart({
          mime: part.mime,
          data: bytes.toString("base64"),
          name: part.name,
          providerPayload: part.providerPayload,
        }),
      );
    }
    messages.push(withParts(message, parts));
  }
  return new Request({
    messages,
    tools: request.tools,
    toolChoice: request.toolChoice,
    parallelToolCalls: request.parallelToolCalls,
    outputFormat: request.outputFormat,
  });
}
