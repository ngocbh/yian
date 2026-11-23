/**
 * The host's check before each tool call an AGENT's model makes (`BeforeToolCall`, the
 * `beforeToolCall` option of `runFlow` and `advance`): the call it sees, where the call's tool
 * comes from (`ToolSource`), what the call acts on as its tool says (`GatedCall.target`), the
 * decisions it may give, and the bound on the texts a decision carries. A run with a hook records
 * that it is gated, and a resume of it needs one.
 *
 * Knows about: tools/context, llm/messages (the JSON check) and typesys.
 */

import { checkJsonValue, MessageFormatError } from "../llm/messages.ts";
import { YianRangeError, YianTypeError } from "../typesys/errors.ts";
import { freezeParsed, type JsonObject } from "../typesys/json.ts";
import { isRecord, jsTypeName } from "../typesys/values.ts";
import type { PendingMarker } from "./context.ts";

/**
 * Where a gated call's tool comes from:
 * - `host`: a host tool, by its own name in its `group` (`null` for a top-level or registry tool);
 * - `mcp`: an MCP server's tool, by the server's key in the run's tools and its name there;
 * - `module`: a tool module's `export`, by its `uses:` alias and the alias's `ref` as written;
 * - `memory`: a memory's tool, by the agent's memory `input` and the tool's name;
 * - `flow`: a flow the agent grants, by its `defs:` entry or `uses:` alias, or a library's def,
 *   by the def's name and the library's alias.
 */
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
  /** `<run>/<node>/<n>/<index>`, the id `complete` and `release` take. */
  readonly callId: string;
  /** The agent's expanded node id. */
  readonly node: string;
  /** The name the model sees. */
  readonly tool: string;
  readonly source: ToolSource;
  /** The model's parsed arguments, as the recorded reply holds them; never a validator's output. */
  readonly arguments: Readonly<JsonObject>;
  /**
   * What the call acts on, as its tool's `gateTarget` gives it (`checkGateTarget`), never read from
   * the model's arguments and never cut short; `null` when the tool declares none or gives `null`.
   */
  readonly target: Readonly<JsonObject> | null;
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
 * keeps the note in the run log for audit (a `note` that is `undefined` is none), `{refuse}` gives
 * the model the text as the call's error, and `pending()` holds the call until the host answers it.
 */
export type GateDecision =
  | undefined
  | { readonly run: true; readonly note?: string }
  | { readonly refuse: string }
  | PendingMarker;

/**
 * The host's check before each tool call an AGENT's model makes, including the agents inside a
 * flow an agent calls as a tool; it may answer later through a Promise. It is asked once the call
 * is checked: its arguments' schema, then its validator and `isConcurrencySafe` (a module tool's
 * on its module's thread), which must have no effects, since a checked call may never run. It
 * reads the model's arguments, never a validator's output, and the call's target, which its tool's
 * `gateTarget` gives from the validator's output; a target that fails, or whose hold would be over
 * the run's event bound, refuses the call without asking. A refused flow call never starts its
 * flow. A held call's tool doesn't run: the host answers it with `complete`, whose result the model
 * reads, or runs it once with `release`. The reply's calls before it and in its concurrency group
 * run; its later calls wait until every hold is answered, then are asked about. A call sent again
 * after a cut can't be held, since it may have run: it is closed `crashed`.
 */
export type BeforeToolCall = (call: GatedCall, ctx: GateContext) => GateDecision | PromiseLike<GateDecision>;

/** The longest note, refusal text or release note, in UTF-8 bytes. */
export const GATE_TEXT_MAX_BYTES = 4096;

/**
 * What the model reads when the hook fails or gives no valid decision, or when a call's target
 * fails and its tool has no `refuseUngated`.
 */
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

/**
 * Return `value` when it is a target a gated call may carry (`GatedCall.target`): `null`, or a
 * deep-frozen copy of a plain JSON object; `what` names it in errors. Its size is the run's to
 * bound, by the hold step it would make.
 *
 * @throws TypeError when it is neither, a thenable from any realm among them, or holds a value
 *   that isn't JSON.
 */
export function checkGateTarget(value: unknown, what: string): Readonly<JsonObject> | null {
  if (value === null) return null;
  if (isThenable(value)) {
    // Not a target; handle its rejection so it can't crash the process.
    Promise.resolve(value).catch(() => {});
    throw new YianTypeError(`${what} must be a JSON object or null, got ${jsTypeName(value)}`);
  }
  if (!isRecord(value)) throw new YianTypeError(`${what} must be a JSON object or null, got ${jsTypeName(value)}`);
  try {
    return freezeParsed(checkJsonValue(value, what) as JsonObject);
  } catch (error) {
    if (error instanceof MessageFormatError) throw new YianTypeError(error.message);
    throw error;
  }
}

/** @internal Whether `value` has a `then` method, as a Promise of any realm does. Reading it may throw. */
export function isThenable(value: unknown): value is PromiseLike<unknown> {
  const object = (typeof value === "object" && value !== null) || typeof value === "function";
  return object && typeof (value as { then?: unknown }).then === "function";
}
