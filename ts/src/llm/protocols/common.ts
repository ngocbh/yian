/** Helpers the protocol adapters share. */

import { JSONDecodeError, type JsonValue, parseJson } from "../../typesys/json.ts";
import { ProviderError } from "../errors.ts";
import type { Part } from "../messages.ts";

/** Parse one event's JSON, throwing a `ProviderError` that does not echo it. */
export function eventData(data: string, provider: string): JsonValue {
  try {
    return parseJson(data, "plain");
  } catch (error) {
    if (error instanceof JSONDecodeError) throw new ProviderError("protocol", `invalid JSON in ${provider} stream event`);
    throw error;
  }
}

/** Return the error message for a usage count that is not an int. */
export function notIntMessage(provider: string): string {
  return `${provider} response usage has a count that is not an int`;
}

/**
 * Return a usage count as a number, `null` when it is `null` or absent; throw `notInt` when it
 * is not an int within `Number.MAX_SAFE_INTEGER`.
 */
export function tokenCount(value: JsonValue | undefined, notInt: string): number | null {
  if (value === null || value === undefined) return null;
  if (typeof value !== "bigint" || !Number.isSafeInteger(Number(value))) throw new ProviderError("protocol", notInt);
  return Number(value);
}

/**
 * Return a call's id: its own unless missing or in `taken`, else `yian_call_<n>` with `_<k>`
 * appended while that is in `taken`. The id it gets is added to `taken`.
 *
 * @param n The call's position among the reply's calls.
 * @param taken The ids in the reply so far: the server's and those generated.
 */
export function callId(id: string | null, n: number, taken: Set<string>): string {
  let out = id;
  if (out === null || taken.has(out)) {
    out = `yian_call_${n}`;
    for (let k = 1; taken.has(out); k++) out = `yian_call_${n}_${k}`;
  }
  taken.add(out);
  return out;
}

/** Return the error text for a part a message of `role` cannot carry on route `protocol`. */
export function wrongPart(protocol: string, part: Part, role: string): string {
  const article = role === "assistant" ? "an" : "a";
  return `the ${protocol} route cannot send a ${part.constructor.name} in ${article} ${role} message`;
}
