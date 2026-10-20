/**
 * TOOL: a direct, deterministic invocation of one tool from the run's `ToolSet`.
 *
 * No LLM in the loop (distinct from an AGENT node's tool loop). The node's `input:` entries are
 * bindings like any other kind's, resolved by the engine's bind seam, so the tool sees only its
 * own record, never the live pool. An optional `output:` declaration sets the node's output
 * contract, which the engine checks against the tool's return value on every call. `retries:`
 * is how many times a failed call may be tried again.
 *
 * Only the node's shell is here. Its body (the call, its retries with backoff, and each retried
 * attempt recorded as a node step) is not ported yet and comes with `forward`.
 *
 * Knows about:   nodes/base, nodes/contracts, defaults, diagnostics and typesys.
 * Never imports: compile, compose, runtime, the package-root run boundary, or the CLI.
 */

import { current } from "../../defaults.ts";
import { reprText } from "../../diagnostics.ts";
import { isTransientToolError } from "../../tools/errors.ts";
import { jsTypeName, shownText, type Type, type ValueRecord, valueRepr } from "../../typesys/values.ts";
import { Node, NodeKind } from "../base.ts";
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
 * A failed call is retried up to `retries` times. `inputTypes` maps each input name to the type
 * the loader proved its binding has (a literal's exact type, a typed binding's declared type, or
 * a reference's inferred type); an input whose type is not proven is absent.
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

  /** The tool call and its retries; not ported yet. */
  override forward(): never {
    throw new Error(`node ${reprText(this.id)} (ToolNode) forward is not ported yet (T4-11)`);
  }
}
