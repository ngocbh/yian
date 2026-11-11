/**
 * Auth objects: the credential headers for one request and the secrets they sent, and the
 * scrub that replaces those secrets in text.
 *
 * Each object resolves its headers per request, so a key source that rotates or refreshes is
 * read every time. `credentials()` lists every secret the object holds or has resolved, so the
 * executor can replace those exact values in error text without calling a source again.
 * Secrets and their sources live in `#private` fields, so inspecting an auth object, or a model
 * that holds one, never prints them.
 */

import { current } from "../defaults.ts";

/** A key, or a function that returns it on each request. */
export type Secret = string | (() => string | Promise<string>);

/** Credential headers for a model call. */
export interface Auth {
  /**
   * Resolve the credential headers for one request.
   *
   * @returns Header names mapped to their values.
   * @throws TypeError If a resolved value is not valid header text.
   */
  headers(): Promise<Record<string, string>>;
  /**
   * List the static secrets and every value resolved so far; never calls a source.
   *
   * @returns The secrets, in first-seen order, without duplicates.
   */
  credentials(): readonly string[];
}

/**
 * Return a key or token if it is valid header text, never echoing it on failure.
 *
 * Valid text is a non-empty string of printable ASCII and inner spaces, so `Bearer x` passes
 * while tabs, CR, LF, other control characters, non-ASCII and leading or trailing spaces fail.
 * HTTP strips the padding from a header value, so a padded key would send something other than
 * the secret it was given. Connection headers use the looser `checkConnectionHeaderValue`.
 *
 * @param name What the value is, for the error message.
 * @param value The value to check.
 * @returns `value` unchanged.
 * @throws TypeError If `value` is not valid header text.
 */
export function checkHeaderValue(name: string, value: unknown): string {
  if (typeof value !== "string" || !/^[\x21-\x7e](?:[\x20-\x7e]*[\x21-\x7e])?$/.test(value)) {
    throw new TypeError(`${name} must be non-empty HTTP header text`);
  }
  return value;
}

// Visible ASCII with spaces or tabs only between characters: RFC 9110 field-value without
// surrounding whitespace, controls or non-ASCII, which an HTTP client would otherwise reject at
// request time with the value in the message.
const FIELD_VALUE = /^[\x21-\x7e]+(?:[ \t]+[\x21-\x7e]+)*$/;

/**
 * Return a connection header value unchanged if it may be sent, never echoing it on failure.
 *
 * Route headers, extra headers and connection settings all check their values here.
 *
 * @param name The header's name; messages name it.
 * @param value The value to check.
 * @returns `value` unchanged.
 * @throws TypeError If `value` is not a string, is longer than `maxHeaderValueChars`
 *   characters, or is not non-empty visible ASCII with spaces or tabs only between characters.
 */
export function checkConnectionHeaderValue(name: string, value: unknown): string {
  if (typeof value !== "string") throw new TypeError(`header ${name} value must be a string`);
  const limit = current().maxHeaderValueChars;
  // Spread counts code points, not UTF-16 units.
  if ([...value].length > limit) throw new TypeError(`header ${name} value must be at most ${limit} characters`);
  if (!FIELD_VALUE.test(value)) {
    throw new TypeError(
      `header ${name} value must be non-empty visible ASCII, with spaces or tabs only between characters`,
    );
  }
  return value;
}

/** An ordered set of the secrets an auth object has resolved. */
class Seen {
  readonly #values = new Set<string>();

  /** Record each non-empty string in `values`; anything else was never sent. */
  add(...values: unknown[]): void {
    for (const value of values) {
      if (typeof value === "string" && value) this.#values.add(value);
    }
  }

  /** Return the recorded values in first-seen order. */
  snapshot(): readonly string[] {
    return [...this.#values];
  }
}

/** Shared resolution for a key that is a string or a function returning one. */
abstract class KeyAuth implements Auth {
  readonly #key: Secret;
  readonly #seen = new Seen();

  /** @param key The key, or a function that returns it on each request. */
  constructor(key: Secret) {
    this.#key = key;
    this.#seen.add(key);
  }

  abstract headers(): Promise<Record<string, string>>;

  /**
   * Resolve the key for one request and record it for the scrub.
   *
   * @returns The key, checked as header text.
   * @throws TypeError If the key is not valid header text.
   */
  protected async resolve(): Promise<string> {
    const key: unknown = typeof this.#key === "function" ? await this.#key() : this.#key;
    // Record before checking: a value that fails the check may still reach error text
    // elsewhere, and listing it only widens the scrub.
    this.#seen.add(key);
    return checkHeaderValue("API key", key);
  }

  credentials(): readonly string[] {
    return this.#seen.snapshot();
  }
}

/** Send no credentials, for local servers such as Ollama. */
export class NoAuth implements Auth {
  async headers(): Promise<Record<string, string>> {
    return {};
  }

  credentials(): readonly string[] {
    return [];
  }
}

/** Send the key as `Authorization: Bearer <key>`. */
export class BearerAuth extends KeyAuth {
  async headers(): Promise<Record<string, string>> {
    return { Authorization: `Bearer ${await this.resolve()}` };
  }
}

/** Send the key as the value of a named header. */
export class HeaderKeyAuth extends KeyAuth {
  /** The header name, for example `"x-api-key"`, `"api-key"` or `"x-goog-api-key"`. */
  readonly header: string;

  /**
   * @param header The header name.
   * @param key The key, or a function that returns it on each request.
   */
  constructor(header: string, key: Secret) {
    super(key);
    this.header = header;
  }

  async headers(): Promise<Record<string, string>> {
    return { [this.header]: await this.resolve() };
  }
}

/** A ChatGPT subscription login's current access. */
export interface SubscriptionAccess {
  accessToken: string;
  accountId: string;
  /** The data residency region, if the account has one. */
  residency?: string | null;
}

/** Send a ChatGPT subscription login's access token and account id. */
export class SubscriptionAuth implements Auth {
  readonly #source: () => SubscriptionAccess | Promise<SubscriptionAccess>;
  readonly #seen = new Seen();

  /** @param source Called once per request; returns the current access, refreshed as it sees fit. */
  constructor(source: () => SubscriptionAccess | Promise<SubscriptionAccess>) {
    this.#source = source;
  }

  /**
   * Resolve the current access and build the subscription headers: `Authorization`,
   * `ChatGPT-Account-Id` and `originator`, plus `x-openai-internal-codex-residency` when the
   * access has a residency.
   *
   * @throws TypeError If a resolved value is not valid header text.
   */
  async headers(): Promise<Record<string, string>> {
    const access: Partial<SubscriptionAccess> | null | undefined = await this.#source();
    const token: unknown = access?.accessToken;
    const accountId: unknown = access?.accountId;
    const residency: unknown = access?.residency;
    this.#seen.add(token, accountId);
    const headers: Record<string, string> = {
      Authorization: `Bearer ${checkHeaderValue("Subscription accessToken", token)}`,
      "ChatGPT-Account-Id": checkHeaderValue("Subscription accountId", accountId),
      originator: "yian",
    };
    if (residency !== undefined && residency !== null) {
      headers["x-openai-internal-codex-residency"] = checkHeaderValue("Subscription residency", residency);
    }
    return headers;
  }

  credentials(): readonly string[] {
    return this.#seen.snapshot();
  }
}

/** Send a user name and password as `Authorization: Basic <base64 of user:password>`. */
export class BasicAuth implements Auth {
  readonly #token: string;
  readonly #seen = new Seen();

  /**
   * @param username The user name.
   * @param password The password; may be empty.
   */
  constructor(username: string, password: string) {
    this.#token = Buffer.from(`${username}:${password}`, "utf8").toString("base64");
    this.#seen.add(password, this.#token);
  }

  async headers(): Promise<Record<string, string>> {
    return { Authorization: `Basic ${this.#token}` };
  }

  credentials(): readonly string[] {
    return this.#seen.snapshot();
  }
}

/**
 * Send secret extra headers alongside another auth object's, so the scrub knows their values.
 *
 * The inner auth's headers win a name clash.
 */
export class ExtraHeadersAuth implements Auth {
  readonly #auth: Auth;
  readonly #extra: Readonly<Record<string, string>>;

  /**
   * @param auth The auth object whose headers are sent too.
   * @param headers The extra header names mapped to their values; copied.
   * @throws TypeError If a value may not be sent; the message names the header, never the value.
   */
  constructor(auth: Auth, headers: Record<string, string>) {
    this.#auth = auth;
    // Copied so a caller mutating its own object cannot change what is sent.
    this.#extra = Object.freeze({ ...headers });
    for (const [name, value] of Object.entries(this.#extra)) checkConnectionHeaderValue(name, value);
  }

  async headers(): Promise<Record<string, string>> {
    return { ...this.#extra, ...(await this.#auth.headers()) };
  }

  credentials(): readonly string[] {
    const seen = new Seen();
    seen.add(...this.#auth.credentials(), ...Object.values(this.#extra));
    return seen.snapshot();
  }
}

/** The marker `scrub` puts in place of a credential. */
export const REDACTED = "[REDACTED]";

/** One form a secret may take in text: its name, its text, and whether to compare it case-insensitively. */
export interface SecretForm {
  form: string;
  text: string;
  ignoreCase?: boolean;
}

/** Percent-encode `value`: every character but `A-Za-z0-9_.-~`. */
function percentEncode(value: string): string {
  return encodeURIComponent(value).replace(
    /[!'()*]/g,
    (char) => `%${char.charCodeAt(0).toString(16).toUpperCase()}`,
  );
}

/** Return each UTF-16 code unit of `value` as `\uXXXX` in lowercase hex. */
function unicodeEscape(value: string): string {
  let out = "";
  for (let i = 0; i < value.length; i++) out += `\\u${value.charCodeAt(i).toString(16).padStart(4, "0")}`;
  return out;
}

/**
 * Return `text` lowercased, as long as it is: a character whose lowercase is longer (such as
 * "İ") stays as it is, so an index into the result is the same index into `text`.
 */
function foldCase(text: string): string {
  if (/^[\x00-\x7f]*$/.test(text)) return text.toLowerCase();
  let out = "";
  for (const char of text) {
    const lower = char.toLowerCase();
    out += lower.length === char.length ? lower : char;
  }
  return out;
}

/**
 * Return the forms a secret may take in text, each distinct text once under its first name.
 *
 * The forms are `plain`; `percent-encoded` (`%20` for a space, as in a path) and `plus-encoded`
 * (`+`, as in a query), both encoding all but `A-Za-z0-9_.-~`; `JSON-escaped`, as a JSON
 * string holds it with and without non-ASCII escaped and with and without `/` as `\/`; and
 * `\u-escaped`, every character as a `\uXXXX` escape, in lowercase hex. Every form is flagged
 * `ignoreCase`: a server may echo a secret lowered or raised, and an encoder may write either
 * hex case. Texts that differ only in letter case count as one.
 *
 * @param value The secret.
 * @returns The forms, plain first.
 */
export function secretForms(value: string): SecretForm[] {
  const candidates: SecretForm[] = [{ form: "plain", text: value, ignoreCase: true }];
  // encodeURIComponent throws on a lone surrogate, which no request could send encoded.
  if (value.isWellFormed()) {
    const percent = percentEncode(value);
    candidates.push({ form: "percent-encoded", text: percent, ignoreCase: true });
    candidates.push({ form: "plus-encoded", text: percent.replaceAll("%20", "+"), ignoreCase: true });
  }
  const json = JSON.stringify(value).slice(1, -1);
  // Encoders that escape everything outside printable ASCII write the first form; JSON.stringify
  // keeps it.
  const asciiJson = json.replace(/[^\x20-\x7e]/g, unicodeEscape);
  for (const escaped of [asciiJson, json]) {
    candidates.push({ form: "JSON-escaped", text: escaped, ignoreCase: true });
    candidates.push({ form: "JSON-escaped", text: escaped.replaceAll("/", "\\/"), ignoreCase: true });
  }
  candidates.push({ form: "\\u-escaped", text: unicodeEscape(value), ignoreCase: true });

  const forms = new Map<string, SecretForm>();
  for (const candidate of candidates) {
    const key = foldCase(candidate.text);
    if (!forms.has(key)) forms.set(key, candidate);
  }
  return [...forms.values()];
}

// Shorter base64 fragments would match too much unrelated text.
const MIN_BASE64_FRAGMENT = 8;

/**
 * Return the base64 forms of `secret`: for each of the three byte alignments, the characters of
 * its standard and URL-safe encoding that only the secret's bits decide, when there are
 * `MIN_BASE64_FRAGMENT` or more. They match in their own letter case only: base64 is
 * case-sensitive. A `Basic` credential holds one, whatever the user name before the secret.
 */
export function base64Forms(secret: string): SecretForm[] {
  const bytes = Buffer.from(secret, "utf8");
  const forms: SecretForm[] = [];
  for (let pad = 0; pad < 3; pad++) {
    const encoded = Buffer.concat([Buffer.alloc(pad), bytes]).toString("base64");
    // Character i holds bits [6i, 6i + 6); keep those wholly inside the secret's bits.
    const start = Math.ceil((8 * pad) / 6);
    const end = Math.floor((8 * (pad + bytes.length)) / 6);
    const fragment = encoded.slice(start, end);
    if (fragment.length < MIN_BASE64_FRAGMENT) continue;
    forms.push({ form: "base64", text: fragment });
    forms.push({ form: "URL-safe base64", text: fragment.replaceAll("+", "-").replaceAll("/", "_") });
  }
  return forms;
}

/** Return every form `scrub` replaces of each distinct non-empty credential: `secretForms`' and `base64Forms`'. */
function credentialForms(credentials: Iterable<string>): SecretForm[] {
  const forms: SecretForm[] = [];
  for (const value of new Set(credentials)) {
    if (value) forms.push(...secretForms(value), ...base64Forms(value));
  }
  return forms;
}

/**
 * Return a text cut at a size cap without any part of a credential the cut split.
 *
 * The cut can only split the last credential form, so the trailing run of non-whitespace is
 * dropped, but at most as many characters as the longest form of any credential; that removes
 * any escape mix. Then a trailing start of any form `scrub` replaces is dropped, in any letter
 * case but base64's, for a credential with whitespace inside, and trailing whitespace. Every
 * proper prefix counts, down to one character, so a little more than a credential may go. Empty
 * credentials are ignored.
 *
 * @param text The cut text.
 * @param credentials The secrets the cut may have split.
 * @returns `text` without those tails.
 */
export function dropCredentialTail(text: string, credentials: Iterable<string>): string {
  const forms = credentialForms(credentials);
  const longest = Math.max(0, ...forms.map((form) => form.text.length));
  // A loop, not /\S*$/, which backtracks quadratically over a long run before whitespace.
  let end = text.length;
  while (end > 0 && text.length - end < longest && !/\s/.test(text[end - 1]!)) end--;
  text = text.slice(0, end).trimEnd();
  let cut = text.length;
  let folded: string | null = null;
  for (const form of forms) {
    const max = Math.min(form.text.length - 1, text.length);
    // The fold keeps lengths, so the tail of the folded text is the folded tail.
    const tail = form.ignoreCase ? (folded ??= foldCase(text)).slice(text.length - max) : text.slice(text.length - max);
    const start = form.ignoreCase ? foldCase(form.text) : form.text;
    // Only a tail longer than the one already found can move the cut.
    for (let length = max; length > text.length - cut; length--) {
      if (tail.endsWith(start.slice(0, length))) {
        cut = text.length - length;
        break;
      }
    }
  }
  return text.slice(0, cut).trimEnd();
}

/**
 * Replace each match of `forms` in `text` with `marker`.
 *
 * Every match is found in `text` as given, an `ignoreCase` form's in any letter case; matches
 * that overlap are replaced as one, so a form inside a longer one goes with it, while matches
 * that only touch get a marker each. A plain search, not a regular expression, so a form of any
 * length can be looked for.
 *
 * @param text The text to scrub.
 * @param forms The forms to replace; empty texts are ignored.
 * @param marker What replaces each match, literally; defaults to `REDACTED`.
 * @returns The scrubbed text.
 */
export function replaceForms(text: string, forms: Iterable<SecretForm>, marker: string = REDACTED): string {
  const matches: [number, number][] = [];
  let folded: string | null = null;
  for (const form of forms) {
    if (!form.text) continue;
    const within = form.ignoreCase ? (folded ??= foldCase(text)) : text;
    const needle = form.ignoreCase ? foldCase(form.text) : form.text;
    for (let at = within.indexOf(needle); at !== -1; at = within.indexOf(needle, at + 1)) {
      matches.push([at, at + needle.length]);
    }
  }
  if (matches.length === 0) return text;
  matches.sort((a, b) => a[0] - b[0]);
  let out = "";
  // Where the text after the last marker written starts.
  let kept = 0;
  let [start, end] = matches[0]!;
  for (const [from, to] of matches.slice(1)) {
    if (from < end) {
      end = Math.max(end, to);
      continue;
    }
    out += text.slice(kept, start) + marker;
    kept = end;
    [start, end] = [from, to];
  }
  return out + text.slice(kept, start) + marker + text.slice(end);
}

/**
 * Replace each form of each credential in `text` with `marker`.
 *
 * The forms are `secretForms`', in any letter case, and `base64Forms`'; overlapping matches are
 * replaced as one, so a credential that contains a shorter one is replaced whole. Empty
 * credentials are ignored.
 *
 * @param text The text to scrub.
 * @param credentials The secrets to replace.
 * @param marker What replaces each one, literally; defaults to `REDACTED`.
 * @returns The scrubbed text.
 */
export function scrub(text: string, credentials: Iterable<string>, marker: string = REDACTED): string {
  return replaceForms(text, credentialForms(credentials), marker);
}
