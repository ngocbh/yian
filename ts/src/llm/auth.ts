/**
 * Auth objects: the credential headers for one request and the secrets they sent, and the
 * scrub that replaces those secrets in text.
 *
 * Each object resolves its headers per request, so a key source that rotates or refreshes is
 * read every time. `credentials()` lists every secret the object holds or has resolved, so the
 * executor can replace those exact values in error text without calling a source again.
 */

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
 * Return `value` if it is valid header text, never echoing it on failure.
 *
 * Valid text is a non-empty string of printable ASCII and inner spaces, so `Bearer x` passes
 * while CR, LF, other control characters, non-ASCII and leading or trailing spaces fail. HTTP
 * strips the padding from a header value, so a padded key would send something other than the
 * secret it was given.
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

/** An ordered set of the secrets an auth object has resolved. */
class Seen {
  private readonly values = new Set<string>();

  /** Record each non-empty string in `values`; anything else was never sent. */
  add(...values: unknown[]): void {
    for (const value of values) {
      if (typeof value === "string" && value) this.values.add(value);
    }
  }

  /** Return the recorded values in first-seen order. */
  snapshot(): readonly string[] {
    return [...this.values];
  }
}

/** Shared resolution for a key that is a string or a function returning one. */
abstract class KeyAuth implements Auth {
  private readonly key: Secret;
  private readonly seen = new Seen();

  /** @param key The key, or a function that returns it on each request. */
  constructor(key: Secret) {
    this.key = key;
    this.seen.add(key);
  }

  abstract headers(): Promise<Record<string, string>>;

  /**
   * Resolve the key for one request and record it for the scrub.
   *
   * @returns The key, checked as header text.
   * @throws TypeError If the key is not valid header text.
   */
  protected async resolve(): Promise<string> {
    const key: unknown = typeof this.key === "function" ? await this.key() : this.key;
    // Record before checking: a value that fails the check may still reach error text
    // elsewhere, and listing it only widens the scrub.
    this.seen.add(key);
    return checkHeaderValue("API key", key);
  }

  credentials(): readonly string[] {
    return this.seen.snapshot();
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
  private readonly source: () => SubscriptionAccess | Promise<SubscriptionAccess>;
  private readonly seen = new Seen();

  /** @param source Called once per request; returns the current access, refreshed as it sees fit. */
  constructor(source: () => SubscriptionAccess | Promise<SubscriptionAccess>) {
    this.source = source;
  }

  /**
   * Resolve the current access and build the subscription headers: `Authorization`,
   * `ChatGPT-Account-Id` and `originator`, plus `x-openai-internal-codex-residency` when the
   * access has a residency.
   *
   * @throws TypeError If a resolved value is not valid header text.
   */
  async headers(): Promise<Record<string, string>> {
    const access: Partial<SubscriptionAccess> | null | undefined = await this.source();
    const token: unknown = access?.accessToken;
    const accountId: unknown = access?.accountId;
    const residency: unknown = access?.residency;
    this.seen.add(token, accountId);
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
    return this.seen.snapshot();
  }
}

/** The marker `scrub` puts in place of a credential. */
export const REDACTED = "[REDACTED]";

/** Percent-encode `value` as Python's `quote(value, safe="")` does: all but `A-Za-z0-9_.-~`. */
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
 * Return the forms a secret may take in text, each distinct text once under its first name.
 *
 * The forms are `plain`; `percent-encoded` (`%20` for a space, as in a path) and `plus-encoded`
 * (`+`, as in a query), both encoding all but `A-Za-z0-9_.-~`; `JSON-escaped`, as a JSON
 * string holds it with and without non-ASCII escaped and with and without `/` as `\/`; and
 * `\u-escaped`, every character as a `\uXXXX` escape, in lowercase hex and flagged
 * `ignoreCase`, since a writer may use either case. A form whose text is also the
 * `\u-escaped` text keeps the flag.
 *
 * @param value The secret.
 * @returns The forms, plain first.
 */
export function secretForms(value: string): { form: string; text: string; ignoreCase?: boolean }[] {
  const candidates: { form: string; text: string; ignoreCase?: boolean }[] = [{ form: "plain", text: value }];
  // encodeURIComponent throws on a lone surrogate, which no request could send encoded.
  if (value.isWellFormed()) {
    const percent = percentEncode(value);
    candidates.push({ form: "percent-encoded", text: percent });
    candidates.push({ form: "plus-encoded", text: percent.replaceAll("%20", "+") });
  }
  const json = JSON.stringify(value).slice(1, -1);
  // Python's json.dumps escapes everything outside printable ASCII; JSON.stringify keeps it.
  const asciiJson = json.replace(/[^\x20-\x7e]/g, unicodeEscape);
  for (const escaped of [asciiJson, json]) {
    candidates.push({ form: "JSON-escaped", text: escaped });
    candidates.push({ form: "JSON-escaped", text: escaped.replaceAll("/", "\\/") });
  }
  candidates.push({ form: "\\u-escaped", text: unicodeEscape(value), ignoreCase: true });

  const forms = new Map<string, { form: string; text: string; ignoreCase?: boolean }>();
  for (const candidate of candidates) {
    const first = forms.get(candidate.text);
    if (!first) forms.set(candidate.text, { ...candidate });
    else if (candidate.ignoreCase) first.ignoreCase = true;
  }
  return [...forms.values()];
}

/**
 * Replace each form of each credential in `text` with `marker`.
 *
 * Forms come from `secretForms` and are replaced longest first across all credentials, so a
 * credential that contains a shorter one is replaced whole. Empty credentials are ignored.
 *
 * @param text The text to scrub.
 * @param credentials The secrets to replace.
 * @param marker What replaces each one, literally; defaults to `REDACTED`.
 * @returns The scrubbed text.
 */
export function scrub(text: string, credentials: Iterable<string>, marker: string = REDACTED): string {
  const forms: { text: string; ignoreCase?: boolean }[] = [];
  for (const value of credentials) {
    if (value) forms.push(...secretForms(value));
  }
  forms.sort((a, b) => b.text.length - a.text.length);
  for (const form of forms) {
    // split/join and a replacer function keep `$` in the marker literal.
    text = form.ignoreCase
      ? text.replace(new RegExp(form.text.replaceAll("\\", "\\\\"), "gi"), () => marker)
      : text.split(form.text).join(marker);
  }
  return text;
}
