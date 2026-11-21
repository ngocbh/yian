/**
 * The host's check before each tool call an AGENT's model makes (`BeforeToolCall`): the call it
 * sees, where the call's tool comes from (`ToolSource`), the decisions it may give, and the bound
 * on the texts a decision carries. A run with a hook records that it is gated, and a resume of it
 * needs one.
 *
 * Until hosts can pass the hook as an option, it reaches a run only through `withGate`, which this
 * package's entry points don't export.
 *
 * Knows about: tools/context and typesys.
 */

import { YianRangeError, YianTypeError } from "../typesys/errors.ts";
import type { JsonObject } from "../typesys/json.ts";
import { isRecord } from "../typesys/values.ts";
import type { PendingMarker } from "./context.ts";

/** Where a gated call's tool comes from. */
export type ToolSource =
  | { readonly kind: "host"; readonly group: string | null; readonly tool: string }
  | { readonly kind: "mcp"; readonly server: string; readonly tool: string }
  | { readonly kind: "module"; readonly alias: string; readonly ref: string; readonly export: string }
  | { readonly kind: "memory"; readonly input: string; readonly tool: string }
  | { readonly kind: "flow"; readonly flow: string; readonly library: string | null };

/** A flow call enclosing the calling agent: its scope `<agent>/<flow>#<k>` and its call id. */
export interface FlowCallScope {
  readonly scope: string;
  readonly callId: string;
}

/** One model-made tool call before it runs. Deep-frozen. */
export interface GatedCall {
  /** `<run>/<node>/<n>/<index>`, the id `complete` takes. */
  readonly callId: string;
  /** The agent's expanded node id. */
  readonly node: string;
  /** The name the model sees. */
  readonly tool: string;
  readonly source: ToolSource;
  /** The model's parsed arguments, as the recorded reply holds them; never a validator's output. */
  readonly arguments: Readonly<JsonObject>;
  /** The flow calls enclosing the agent, outermost first; empty at the top level. */
  readonly scopes: readonly FlowCallScope[];
  /** 1, or more for a call sent again after a crash cut it. */
  readonly attempt: number;
}

/** What the hook gets beside the call. */
export interface GateContext {
  readonly runId: string;
  /**
   * Aborts when the decision is no longer wanted: the node stops (a run stop, a failing sibling,
   * the end of an enclosing flow call's scope) or the agent takes an interrupt. Its reason says which.
   */
  readonly signal: AbortSignal;
}

/**
 * The hook's decision: `undefined` or `{run: true}` runs the call, `{run: true, note}` runs it and
 * keeps the note in the run log for audit, `{refuse}` gives the model the text as the call's
 * error, and `pending()` holds the call until the host answers it.
 */
export type GateDecision =
  | undefined
  | { readonly run: true; readonly note?: string }
  | { readonly refuse: string }
  | PendingMarker;

/** The host's check before each tool call an AGENT's model makes; it may answer later through a Promise. */
export type BeforeToolCall = (call: GatedCall, ctx: GateContext) => GateDecision | PromiseLike<GateDecision>;

/** The longest note, refusal text or release note, in UTF-8 bytes. */
export const GATE_TEXT_MAX_BYTES = 4096;

/** What the model reads when the hook fails or gives no valid decision. */
export const GATE_FAILED_TEXT = "The host's permission check failed; the call didn't run.";

/** The fields of each kind of tool source after `kind`, and whether each may be `null`. */
const SOURCE_FIELDS: Readonly<Record<ToolSource["kind"], Readonly<Record<string, boolean>>>> = {
  host: { group: true, tool: false },
  mcp: { server: false, tool: false },
  module: { alias: false, ref: false, export: false },
  memory: { input: false, tool: false },
  flow: { flow: false, library: true },
};

/**
 * Whether `value` is a `ToolSource`: exactly the fields of its kind, each a non-empty string, or
 * `null` where its kind allows. Never throws.
 */
export function isToolSource(value: unknown): value is ToolSource {
  if (!isRecord(value) || typeof value["kind"] !== "string" || !Object.hasOwn(SOURCE_FIELDS, value["kind"])) return false;
  const fields = SOURCE_FIELDS[value["kind"] as ToolSource["kind"]];
  const names = Object.keys(fields);
  if (Object.keys(value).length !== names.length + 1) return false;
  return names.every((name) => {
    if (!Object.hasOwn(value, name)) return false;
    const field = value[name];
    return (field === null && fields[name]) || (typeof field === "string" && field !== "");
  });
}

/**
 * Return `value` when it is a text a gate decision may carry: a non-empty, well-formed string of
 * at most `GATE_TEXT_MAX_BYTES` UTF-8 bytes; `what` names it in errors.
 *
 * @throws TypeError when it isn't a string.
 * @throws RangeError when it is empty, has a lone surrogate, or is over the bound.
 */
export function checkGateText(value: unknown, what: string): string {
  if (typeof value !== "string") throw new YianTypeError(`${what} must be a string`);
  if (value === "") throw new YianRangeError(`${what} must be a non-empty string`);
  if (!value.isWellFormed()) throw new YianRangeError(`${what} has a lone surrogate`);
  const size = new TextEncoder().encode(value).length;
  if (size > GATE_TEXT_MAX_BYTES) {
    throw new YianRangeError(`${what} is ${size} UTF-8 bytes, over the limit of ${GATE_TEXT_MAX_BYTES}`);
  }
  return value;
}

/** The hook each registered options object carries. */
const GATES = new WeakMap<object, BeforeToolCall | null>();

/**
 * @internal Register `hook` as the `beforeToolCall` of the `runFlow` or `advance` options object
 * `options`, and return the same object; a copy of it carries no hook.
 *
 * @throws TypeError when `hook` isn't a function or `null`.
 */
export function withGate<T extends object>(options: T, hook: BeforeToolCall | null): T {
  if (hook !== null && typeof hook !== "function") throw new YianTypeError("beforeToolCall must be a function or null");
  GATES.set(options, hook);
  return options;
}

/** @internal The hook `withGate` registered on `options`, or `null`. */
export function gateOf(options: object): BeforeToolCall | null {
  return GATES.get(options) ?? null;
}
