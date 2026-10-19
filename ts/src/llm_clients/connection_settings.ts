/**
 * Checks for connection settings: an OpenAI organization or project id and extra headers.
 *
 * The settings travel with a provider key, so every caller that accepts them (the `llm_config`
 * carrier, the auth store, `yian auth`, resuming a run) checks them here. This module imports
 * nothing from `config.ts` or `auth/`, so each of those can use it. Every check throws
 * `RangeError` with a message that never contains a header value or an id, since header values
 * are secrets; header names are not secret and are named.
 */

import { current } from "../defaults.ts";
import { isRecord } from "../typesys/index.ts";

const CREDENTIAL = "yian sends the credential itself";
const TRANSPORT = "yian's client sets it";
const OPENAI_IDS = "use llm_config.organization / project";

/**
 * Header names refused for every provider, lowercase name to reason.
 *
 * Credential headers would replace or duplicate the key yian sends; framing headers would corrupt
 * the request.
 */
export const REFUSED_HEADER_NAMES: Readonly<Record<string, string>> = Object.freeze(
  Object.assign(Object.create(null) as Record<string, string>, {
    authorization: CREDENTIAL,
    "proxy-authorization": CREDENTIAL,
    "x-api-key": CREDENTIAL,
    "api-key": CREDENTIAL,
    "x-goog-api-key": CREDENTIAL,
    host: TRANSPORT,
    "content-length": TRANSPORT,
    "transfer-encoding": TRANSPORT,
    connection: TRANSPORT,
    "content-type": TRANSPORT,
  }),
);

/**
 * Header names refused only for `provider: openai`, which has its own fields for them.
 *
 * Other providers (an OpenAI-compatible gateway, say) may send them as ordinary headers.
 */
export const OPENAI_REFUSED_HEADER_NAMES: Readonly<Record<string, string>> = Object.freeze(
  Object.assign(Object.create(null) as Record<string, string>, {
    "openai-organization": OPENAI_IDS,
    "openai-project": OPENAI_IDS,
  }),
);

// RFC 9110 token characters.
const TOKEN = /^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$/;
// RFC 9110 field-value restricted to visible ASCII: no surrounding whitespace, controls or
// non-ASCII, which an HTTP client would otherwise reject at request time with the value in the
// error.
const FIELD_VALUE = /^[\x21-\x7e]+(?:[ \t]+[\x21-\x7e]+)*$/;
const VISIBLE_ASCII = /^[\x21-\x7e]+$/;

/** Return `table`'s own entry for `key`, so inherited names such as `constructor` are absent. */
function own(table: Readonly<Record<string, string>>, key: string): string | undefined {
  return Object.hasOwn(table, key) ? table[key] : undefined;
}

/** Return the number of code points in `text`, as Python's `len` counts. */
function codePoints(text: string): number {
  return [...text].length;
}

/**
 * Return `name` unchanged if it may be sent as a connection header.
 *
 * `provider` is the provider the header goes to; `"openai"` also refuses
 * `OPENAI_REFUSED_HEADER_NAMES`, and `null` checks only the names refused for every provider.
 * Throws `RangeError` if `name` is longer than `maxHeaderValueChars` characters, is not an HTTP
 * token, or is a refused name.
 */
export function validateHeaderName(name: unknown, provider: string | null): string {
  const limit = current().maxHeaderValueChars;
  if (typeof name === "string" && codePoints(name) > limit) {
    throw new RangeError(`header names must be at most ${limit} characters`);
  }
  if (typeof name !== "string" || !TOKEN.test(name)) {
    throw new RangeError("header names must be HTTP tokens: letters, digits and !#$%&'*+-.^_`|~");
  }
  const lower = name.toLowerCase();
  const reason =
    own(REFUSED_HEADER_NAMES, lower) ?? (provider === "openai" ? own(OPENAI_REFUSED_HEADER_NAMES, lower) : undefined);
  if (reason !== undefined) throw new RangeError(`header ${name} is refused: ${reason}`);
  return name;
}

/**
 * Return a header value unchanged if it may be sent.
 *
 * `name` is the header's name, already checked; messages name it. Throws `RangeError` if the
 * value is not a string, is longer than `maxHeaderValueChars` characters, or is not non-empty
 * visible ASCII with spaces or tabs only between characters.
 */
export function validateHeaderValue(name: string, value: unknown): string {
  if (typeof value !== "string") throw new RangeError(`header ${name} value must be a string`);
  const limit = current().maxHeaderValueChars;
  if (codePoints(value) > limit) {
    throw new RangeError(`header ${name} value must be at most ${limit} characters`);
  }
  if (!FIELD_VALUE.test(value)) {
    throw new RangeError(
      `header ${name} value must be non-empty visible ASCII, with spaces or tabs only between characters`,
    );
  }
  return value;
}

/**
 * Return connection headers as a new record if every name and value may be sent.
 *
 * Names compare case-insensitively, so two spellings of one name are refused. An `undefined`
 * entry counts as absent. The result keeps the given order. Throws `RangeError` if `headers` is
 * not a plain record, has more than `maxConnectionHeaders` entries, repeats a name, or has a name
 * or value refused by `validateHeaderName` or `validateHeaderValue`.
 */
export function validateHeaders(headers: unknown, provider: string | null): Record<string, string> {
  if (!isRecord(headers)) throw new RangeError("headers must be a mapping of header name to value");
  const entries = Object.entries(headers).filter(([, value]) => value !== undefined);
  const limit = current().maxConnectionHeaders;
  if (entries.length > limit) throw new RangeError(`headers must have at most ${limit} headers`);
  const seen = new Set<string>();
  const checked: [string, string][] = [];
  for (const [name, value] of entries) {
    validateHeaderName(name, provider);
    const lower = name.toLowerCase();
    if (seen.has(lower)) throw new RangeError(`header ${name} is given more than once (names ignore case)`);
    seen.add(lower);
    checked.push([name, validateHeaderValue(name, value)]);
  }
  // `Object.fromEntries` defines own keys, so a `__proto__` header stays a header.
  return Object.fromEntries(checked);
}

/**
 * Return `value` unchanged if it is an allowed OpenAI organization or project id.
 *
 * `field` is `"organization"` or `"project"`; messages name it. Throws `RangeError` if `value`
 * is not a string, is longer than `maxOrganizationChars` characters, or is not non-empty visible
 * ASCII.
 */
export function validateOrganization(value: unknown, field: string): string {
  if (typeof value !== "string") throw new RangeError(`${field} must be a string`);
  const limit = current().maxOrganizationChars;
  if (codePoints(value) > limit) throw new RangeError(`${field} must be at most ${limit} characters`);
  if (!VISIBLE_ASCII.test(value)) throw new RangeError(`${field} must be non-empty visible ASCII, without spaces`);
  return value;
}

/**
 * Return the request headers that carry OpenAI organization and project ids.
 *
 * Gives `OpenAI-Organization` and `OpenAI-Project`, in that order, for the ids that are not
 * `null` or `undefined`. Throws `RangeError` if a given id is refused by `validateOrganization`.
 */
export function openaiIdHeaders(organization: unknown, project: unknown): Record<string, string> {
  const headers: Record<string, string> = {};
  if (organization !== null && organization !== undefined) {
    headers["OpenAI-Organization"] = validateOrganization(organization, "organization");
  }
  if (project !== null && project !== undefined) {
    headers["OpenAI-Project"] = validateOrganization(project, "project");
  }
  return headers;
}
