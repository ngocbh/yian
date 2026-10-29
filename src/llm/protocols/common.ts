/** Helpers the protocol adapters share. */

import { JSONDecodeError, type JsonValue, parseJson } from "../../typesys/json.ts";
import { encodeBody } from "../body.ts";
import { type ErrorKind, kindForStatus, ProviderError } from "../errors.ts";
import { MessageFormatError, type Part, Usage } from "../messages.ts";

const BAD_REQUEST_CODES = [
  "invalid_prompt",
  "invalid_request_error",
  "invalid_image",
  "invalid_image_format",
  "invalid_base64_image",
  "invalid_image_url",
  "invalid_image_mode",
  "image_too_large",
  "image_too_small",
  "image_parse_error",
  "image_file_too_large",
  "image_file_not_found",
  "empty_image_file",
  "unsupported_image_media_type",
  "failed_to_download_image",
];
/** OpenAI's in-stream error codes, for Responses and chat; any other code, or none, is a server error. */
export const OPENAI_STREAM_ERROR_KINDS: ReadonlyMap<string, ErrorKind> = new Map<string, ErrorKind>([
  ["rate_limit_exceeded", "rate_limit"],
  ["server_error", "server"],
  ["vector_store_timeout", "server"],
  ["context_length_exceeded", "context_length"],
  ...BAD_REQUEST_CODES.map((code): [string, ErrorKind] => [code, "bad_request"]),
  ["image_content_policy_violation", "refused"],
  ["insufficient_quota", "quota"],
  ["usage_limit_reached", "quota"],
  ["usage_not_included", "quota"],
]);

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
 * Return a reply's `Usage`, throwing `notInt` where `Usage` refuses its counts.
 *
 * @throws {ProviderError} A sum is past `Number.MAX_SAFE_INTEGER` (kind `protocol`).
 */
export function usageOrProtocolError(fields: ConstructorParameters<typeof Usage>[0], notInt: string): Usage {
  try {
    return new Usage(fields);
  } catch (error) {
    // `Usage` refuses a sum past `Number.MAX_SAFE_INTEGER`, which `Number` arithmetic rounds.
    if (error instanceof MessageFormatError) throw new ProviderError("protocol", notInt);
    throw error;
  }
}

/**
 * Return the kind for an in-stream error's `code` when it is an HTTP error status, with `body`
 * classified as the HTTP path classifies a body; else `null`.
 */
export function httpCodeKind(code: JsonValue | undefined, body: JsonValue): ErrorKind | null {
  // A 1xx-3xx code would map to kinds that don't fit an error inside a 200 stream.
  if (typeof code !== "bigint" || code < 400n || code > 599n) return null;
  return kindForStatus(Number(code), encodeBody(body, "stream error"));
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
