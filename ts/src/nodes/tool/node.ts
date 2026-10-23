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
 * Knows about:   nodes/base, nodes/contracts, defaults, diagnostics, platform, tools and typesys.
 * Never imports: compile, compose, runtime, the package-root run boundary, or the CLI.
 */

import { current } from "../../defaults.ts";
import { reprText } from "../../diagnostics.ts";
import { realScheduler } from "../../platform.ts";
import { isPermanentToolError, isToolCheckError, isTransientToolError } from "../../tools/errors.ts";
import { ToolSet } from "../../tools/registry.ts";
import { jsTypeName, shownText, type Type, type ValueRecord, valueRepr } from "../../typesys/values.ts";
import { CapabilityFault, hostOutput, Node, type NodeCaps, NodeKind, Output } from "../base.ts";
import { HostToolCall, outputContractType } from "../contracts.ts";

/**
 * A TOOL call failed after at least one retry; the message lists every attempt.
 *
 * Its `cause` is the last attempt's error. An attempt that started but has no recorded failure
 * (the run stopped or crashed inside it) shows as `interrupted`.
 */
export class ToolAttemptsFailed extends Error {
  override name = "ToolAttemptsFailed";
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
    } catch {
      after = null;
    }
    if (typeof after === "number" && Number.isFinite(after) && after >= 0) return Math.min(cap, after);
  }
  // The cap is a safe integer, below 2^53, so exponent 53 reaches it for any base of at least 1;
  // capping the exponent keeps a huge `failure` from giving 0 * Infinity (NaN) for a zero base.
  const delay = Math.min(cap, base * 2 ** Math.min(failure - 1, 53));
  return delay / 2 + (random() * delay) / 2;
}

/** Return `text` with each lone surrogate replaced by `?`, so it can be written as JSON. */
function clean(text: string): string {
  return text.replace(/\p{Cs}/gu, "?");
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
  const type = shownText(clean(name));
  let text: string;
  try {
    text = error instanceof Error ? String(error.message) : String(error);
  } catch {
    text = `<${type} object>`;
  }
  return { type, message: shownText(clean(text)) };
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
    const shown = recorded == null ? "interrupted" : `${recorded.type}: ${recorded.message}`;
    parts.push(`attempt ${number}: ${shown}`);
  }
  const last = errorStep(error);
  parts.push(`attempt ${attempt}: ${last.type}: ${last.message}`);
  return `tool ${reprText(toolId)} failed after ${attempt} attempts: ` + parts.join("; ");
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
 * retry, so a resumed run continues the attempt
 * count; an attempt interrupted by a crash uses up no retry. A failure without a retry propagates
 * unchanged; after a retry the node rejects with `ToolAttemptsFailed`, whose message lists every
 * attempt's error.
 *
 * `inputTypes` maps each input name to the type the loader proved its binding has (a literal's
 * exact type, a typed binding's declared type, or a reference's inferred type); an input whose
 * type is not proven is absent.
 */
export class ToolNode extends Node {
  readonly kind = NodeKind.TOOL;
  /** Looks up and invokes its tool in the run's tool set. */
  override readonly needsTools: boolean = true;
  /** Records each retried attempt as a durable step, so a resumed run continues the count. */
  override readonly needsSteps: boolean = true;
  /** The tool id, looked up in the run's `ToolSet`; an unknown id rejects at run time. */
  toolId: string;
  /** How many times a failed tool call may be retried, at least 0. */
  retries: number;
  /** The proven type of each input's binding; an input whose type is not proven is absent. */
  inputTypes: ReadonlyMap<string, Type>;

  /**
   * Build a TOOL node.
   *
   * @param nodeId The node's unique id.
   * @param options.toolId The tool id, looked up in the run's `ToolSet`.
   * @param options.title A display title.
   * @param options.retries How many times a failed call may be retried, at least 0; `null` or
   *   omitted means `current().toolRetries`.
   * @throws TypeError when `retries` isn't a safe-integer number or `null`.
   * @throws RangeError when `retries` is negative.
   */
  constructor(nodeId: string, options: { toolId: string; title?: string | null; retries?: number | null }) {
    const { toolId, title = null, retries = null } = options;
    super(nodeId, { title });
    if (retries !== null) {
      const message = `node ${reprText(nodeId)}: retries must be a non-negative int, got ${valueRepr(retries)}`;
      if (!Number.isSafeInteger(retries)) throw new TypeError(message);
      if (retries < 0) throw new RangeError(message);
    }
    this.toolId = toolId;
    this.retries = retries ?? current().toolRetries;
    this.inputTypes = new Map();
  }

  /** Return this node's tool id, input names, proven input types, and output type. */
  override hostToolCall(): HostToolCall {
    return new HostToolCall(
      this.toolId,
      (this.params ?? []).map((parameter) => parameter.name),
      this.inputTypes,
      outputContractType(this.outputContract),
    );
  }

  /**
   * Call the tool, retrying failed attempts, and return its result.
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
   */
  override async forward(inputs: Record<string, unknown>, caps: NodeCaps): Promise<Output> {
    const tools = caps.tools ?? new ToolSet();
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

    const call = await tools.validate(this.toolId, inputs, { signal });
    const last = history.at(-1);
    // The previous start recorded a failure and stopped before the retry finished: wait again.
    if (last?.event === "failed") await scheduler.sleep(Number(last.retry_in) * 1000, signal);
    let result: unknown;
    for (;;) {
      try {
        result = await tools.invoke(call, { node: this.id, attempt, signal });
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
    // Outside the loop's `try`, so a result that isn't a yian value is never retried.
    return new Output(hostOutput(result));
  }
}
