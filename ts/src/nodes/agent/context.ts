/**
 * Process-local context for one AGENT invocation.
 *
 * Knows about:   llm executor types, llm_clients (connection, factory), memory contracts, nodes/base,
 *                nodes/contracts, platform, skills, tools and typesys.
 * Never imports: compile, compose, runtime, the package-root run boundary, or the CLI.
 */

import type { RetryInfo } from "../../llm/executor.ts";
import { scrubCallCredentials } from "../../llm_clients/connection.ts";
import type { ModelAccess } from "../../llm_clients/factory.ts";
import type { EvaluationTrace, MemoryContext } from "../../memory/contracts.ts";
import type { Scheduler } from "../../platform.ts";
import type { Skill } from "../../skills/catalog.ts";
import type { ToolSet } from "../../tools/registry.ts";
import type { Type } from "../../typesys/values.ts";
import type { NodeCaps, Steering } from "../base.ts";
import type { NodeSteps } from "../contracts.ts";

/** The system prompt of an AGENT that authors none. */
export const DEFAULT_SYSTEM =
  "You are an analysis agent inside a larger workflow. Follow the instructions " +
  "precisely, use the available tools to gather any facts you need, and return " +
  "a clear, self-contained answer. Ask the user only through an explicitly granted tool.";

/** The AGENT tool loop could not produce a final answer. */
export class AgentLoopError extends Error {
  override name = "AgentLoopError";
}

/** The AGENT tool loop hit `max_tool_iterations` without a final answer. */
export class ToolIterationLimitError extends AgentLoopError {
  override name = "ToolIterationLimitError";
}

/** What the AGENT loop receives to run one invocation. */
export interface AgentRunContext {
  readonly nodeId: string;
  /** The prompt, already rendered. */
  readonly prompt: string;
  /** The granted tool and group ids. */
  readonly tools: readonly string[];
  /** The effective model config, or `null`. */
  readonly llmConfig: Readonly<Record<string, unknown>> | null;
  /** The node's declared output type; `null` passes the reply's text through. */
  readonly outputType: Type | null;
  /** The structured-reply corrections allowed after the first call. */
  readonly retries: number;
  /** The model turns before `ToolIterationLimitError`; -1 means no cap. */
  readonly maxToolIterations: number;
  /** The host's stop; absent for direct loop tests. */
  readonly signal?: AbortSignal;
  /** The write-only visible evidence sink. */
  readonly evaluationTrace?: EvaluationTrace;
  readonly memoryContext: MemoryContext;
  /** The run's tools and groups, which `tools` grants from. */
  readonly toolset: ToolSet | null;
  /** The transient text sink; absent means no streaming. */
  readonly emitStream?: NodeCaps["emitStream"];
  /** The transient model and tool call observer sink. */
  readonly emitEvent?: NodeCaps["emitEvent"];
  /** The authored or default system text plus the skill listing; memory follows it. */
  readonly baseSystemPrompt: string;
  /** The granted skills served by `load_skill`, sorted by name. */
  readonly skills: readonly Skill[];
  /** The host inbox; present only for a steerable node. */
  readonly steering?: Steering;
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

/** How one model call retries: the clock and jitter of its waits, and what runs before each wait. */
export interface RetrySettings {
  readonly scheduler?: Scheduler;
  readonly random?: () => number;
  readonly onRetry?: (retry: RetryInfo) => void;
}

/**
 * Return the retry settings of model call `call` (1-based within this start) of an invocation.
 *
 * With steps, `onRetry` records `{model_attempt: {call, attempt, status: "failed", error: {kind,
 * message}, partial, retry_in}}` for the failed attempt, with the error scrubbed of every
 * credential the call may have sent. `partial` is the `yian.message.v1` JSON of what arrived, or
 * `null` before the first event; `retry_in` is the wait in seconds. A resumed AGENT restarts and
 * never reads these steps.
 */
export function retrySettings(ctx: AgentRunContext, call: number): RetrySettings {
  const { steps, scheduler, random } = ctx;
  if (steps === undefined) return { scheduler, random };
  const onRetry = (retry: RetryInfo): void => {
    const error = scrubCallCredentials(retry.error, ctx.llmConfig, { env: ctx.models.env, sent: sentCredentials(ctx) });
    steps.record({
      model_attempt: {
        call: BigInt(call),
        attempt: BigInt(retry.attempt),
        status: "failed",
        error: { kind: error.kind, message: error.message },
        partial: retry.partial?.toJSON() ?? null,
        retry_in: retry.retryIn,
      },
    });
  };
  return { scheduler, random, onRetry };
}

/** Unwind before starting another external effect once the host stopped the invocation. */
export function checkStopped(ctx: Pick<AgentRunContext, "signal">): void {
  ctx.signal?.throwIfAborted();
}

/**
 * Take the host's queued steer messages for this node, awaiting a promised answer; none without
 * an inbox.
 */
export async function drainSteers(ctx: Pick<AgentRunContext, "nodeId" | "steering">): Promise<string[]> {
  return ctx.steering === undefined ? [] : [...(await ctx.steering.drain(ctx.nodeId))];
}

/** Return the signal the host aborts to interrupt this node; `undefined` without an inbox. */
export function interruptSignal(ctx: Pick<AgentRunContext, "nodeId" | "steering">): AbortSignal | undefined {
  return ctx.steering?.interruptSignal(ctx.nodeId);
}
