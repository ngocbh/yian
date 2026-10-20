/**
 * Process-local context for one AGENT invocation.
 *
 * Knows about:   llm_clients factory, memory contracts, nodes/base, skills, tools and typesys.
 * Never imports: compile, compose, runtime, the package-root run boundary, or the CLI.
 */

import type { ModelAccess } from "../../llm_clients/factory.ts";
import type { EvaluationTrace, MemoryContext } from "../../memory/contracts.ts";
import type { Skill } from "../../skills/catalog.ts";
import type { ToolSet } from "../../tools/registry.ts";
import type { Type } from "../../typesys/values.ts";
import type { NodeCaps, Steering } from "../base.ts";

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
}

/** Unwind before starting another external effect once the host stopped the invocation. */
export function checkStopped(ctx: Pick<AgentRunContext, "signal">): void {
  ctx.signal?.throwIfAborted();
}
