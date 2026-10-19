/**
 * Errors a host tool throws to report a failure or steer retries, and yian's own check error.
 *
 * A tool throws `ReportedToolError` for a failure an AGENT's model should read and act on; in a
 * TOOL node it fails the node like any other error. A TOOL node with `retries:` retries a failed
 * call unless the tool threw a permanent error or a check yian runs failed (`ToolCheckError`). A
 * tool throws `TransientToolError` with `retryAfter` to say how long to wait before the next
 * attempt.
 *
 * yian classifies these errors by their `TOOL_ERROR` brand, never by `instanceof`: a host and a
 * tool package can load two copies of yian, and a permanent error from the other copy must still
 * stop retries, since a retried call repeats any side effect the failed attempt already had.
 */

import { jsTypeName } from "../typesys/values.ts";

/** The brand every tool error carries: the list of kinds it belongs to. */
export const TOOL_ERROR = Symbol.for("yian.tool_error");

/** A kind a tool error belongs to; an error can belong to several. */
export type ToolErrorKind = "permanent" | "transient" | "reported" | "check";

const KINDS: ReadonlySet<unknown> = new Set<ToolErrorKind>(["permanent", "transient", "reported", "check"]);

/** Return whether `value` is a known tool error kind. */
function isKind(value: unknown): value is ToolErrorKind {
  return KINDS.has(value);
}

const NONE: readonly ToolErrorKind[] = Object.freeze([]);
const PERMANENT: readonly ToolErrorKind[] = Object.freeze(["permanent"]);
const TRANSIENT: readonly ToolErrorKind[] = Object.freeze(["transient"]);
const REPORTED: readonly ToolErrorKind[] = Object.freeze(["reported"]);
const REPORTED_PERMANENT: readonly ToolErrorKind[] = Object.freeze(["reported", "permanent"]);
const CHECK: readonly ToolErrorKind[] = Object.freeze(["check"]);

/**
 * A tool call failed in a way another attempt can't fix; TOOL retries stop.
 *
 * @example
 * function place(order: { qty: number }): string {
 *   if (order.qty <= 0) throw new PermanentToolError("quantity must be positive");
 *   ...
 * }
 */
export class PermanentToolError extends Error {
  override name = "PermanentToolError";
  readonly [TOOL_ERROR]: readonly ToolErrorKind[] = PERMANENT;
}

/**
 * A tool call failed in a way the calling model should read and act on.
 *
 * In an AGENT's tool loop, the model reads the message as the call's result (an error tool
 * message, or the error's name when the message is empty) and the loop continues. In a TOOL node
 * it is an ordinary failure: the node fails, and `retries:` retries it unless it is also
 * permanent. The message goes to the model provider and the session transcript, so it must not
 * hold secrets.
 *
 * @example
 * function readFile(path: string): string {
 *   if (!existsSync(path)) throw new ReportedToolError(`no such file: ${path}`);
 *   ...
 * }
 */
export class ReportedToolError extends Error {
  override name = "ReportedToolError";
  readonly [TOOL_ERROR]: readonly ToolErrorKind[];

  /**
   * @param message The text the model reads.
   * @param options `cause`, and `permanent: true` to also stop TOOL retries.
   * @throws {TypeError} If `permanent` is neither a boolean nor `undefined`.
   */
  constructor(message?: string, options?: ErrorOptions & { permanent?: boolean }) {
    const permanent: unknown = options?.permanent;
    if (permanent !== undefined && typeof permanent !== "boolean") {
      throw new TypeError(`permanent must be a boolean, not ${jsTypeName(permanent)}`);
    }
    super(message, options);
    this[TOOL_ERROR] = permanent === true ? REPORTED_PERMANENT : REPORTED;
  }
}

/**
 * A tool call failed in a way a later attempt may not; TOOL retries apply.
 *
 * @example
 * function quote(symbol: string): number {
 *   const response = fetchQuote(symbol);
 *   if (response.status === 429) throw new TransientToolError("rate limited", { retryAfter: 5 });
 *   ...
 * }
 */
export class TransientToolError extends Error {
  override name = "TransientToolError";
  readonly [TOOL_ERROR]: readonly ToolErrorKind[] = TRANSIENT;
  /**
   * Seconds to wait before the next attempt, finite and at least 0. It replaces the computed
   * backoff, up to the retry delay cap. `null` uses the computed backoff.
   */
  readonly retryAfter: number | null;

  /**
   * @param message The error's message.
   * @param options `cause`, and `retryAfter` (default `null`).
   * @throws {TypeError} If `retryAfter` is not a number, `null` or `undefined`.
   * @throws {RangeError} If `retryAfter` is negative or not finite.
   */
  constructor(message?: string, options?: ErrorOptions & { retryAfter?: number | null }) {
    const retryAfter: unknown = options?.retryAfter ?? null;
    if (retryAfter !== null) {
      if (typeof retryAfter !== "number") {
        throw new TypeError(`retryAfter must be a number of seconds or null, not ${jsTypeName(retryAfter)}`);
      }
      if (!Number.isFinite(retryAfter)) throw new RangeError(`retryAfter must be finite, not ${retryAfter}`);
      if (retryAfter < 0) throw new RangeError(`retryAfter must be at least 0, not ${retryAfter}`);
    }
    super(message, options);
    this.retryAfter = retryAfter;
  }
}

/** A check yian runs on a tool call's arguments or result failed; TOOL retries never retry it. */
export class ToolCheckError extends TypeError {
  override name = "ToolCheckError";
  readonly [TOOL_ERROR]: readonly ToolErrorKind[] = CHECK;
}

/**
 * Return the known kinds in `error`'s `TOOL_ERROR` array, frozen, or none when it has no such
 * array. Never throws, so a hostile object (a throwing getter or proxy trap) has no kinds.
 */
export function toolErrorKinds(error: unknown): readonly ToolErrorKind[] {
  if (error === null || (typeof error !== "object" && typeof error !== "function")) return NONE;
  try {
    const kinds: unknown = (error as { [TOOL_ERROR]?: unknown })[TOOL_ERROR];
    if (!Array.isArray(kinds)) return NONE;
    // Read the foreign array by length and index only; its methods, iterator and species run its code.
    const known: ToolErrorKind[] = [];
    for (let i = 0; i < kinds.length; i++) {
      const kind: unknown = kinds[i];
      if (isKind(kind)) known.push(kind);
    }
    return Object.freeze(known);
  } catch {
    return NONE;
  }
}

/** Return whether `error` is branded permanent, from any copy of yian. */
export function isPermanentToolError(error: unknown): boolean {
  return toolErrorKinds(error).includes("permanent");
}

/** Return whether `error` is branded transient, from any copy of yian. */
export function isTransientToolError(error: unknown): boolean {
  return toolErrorKinds(error).includes("transient");
}

/** Return whether `error` is branded reported, from any copy of yian. */
export function isReportedToolError(error: unknown): boolean {
  return toolErrorKinds(error).includes("reported");
}

/** Return whether `error` is a failed check yian ran, from any copy of yian. */
export function isToolCheckError(error: unknown): boolean {
  return toolErrorKinds(error).includes("check");
}
