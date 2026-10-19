/** Immutable credential and selection values shared by authentication services. */

import { current } from "../../defaults.ts";
import { compareCodeUnits } from "../../diagnostics.ts";
import { isRecord } from "../../typesys/index.ts";
import { validateHeaderName, validateHeaders, validateOrganization } from "../connection_settings.ts";

/** An authentication or credential-storage failure safe to show to a user. */
export class AuthError extends Error {
  override name = "AuthError";
}

/** Return `link`'s `name` property, or `undefined` if reading it throws. */
function read(link: object, name: string): unknown {
  try {
    return (link as Record<string, unknown>)[name];
  } catch {
    return undefined;
  }
}

/**
 * Whether a model call's error means its credentials were refused or missing.
 *
 * Missing means yian couldn't supply them, which it reports as an `AuthError`. Walks `error` and
 * its `cause` chain, since a wrapping error keeps the provider's as its cause; the walk stops at a
 * non-object, a repeated link, or after 16 links, so a cyclic or runaway chain ends. A link counts
 * when it is an `AuthError`, a link whose `kind` is `"auth"` (such as a `ProviderError` of kind
 * `auth`), or carries an integer `status` or `code` of 401 or 403, or of 400 with
 * `api_key_invalid` in its text: Gemini reports an invalid key as a 400 whose only distinguishing
 * mark is that reason, so the substring match is deliberate. A link whose `kind` is `"refused"`
 * doesn't count by its status: yian read that reply as a content-filter refusal, such as
 * OpenRouter's 403 content-policy reply, and the refusal is the root cause. Callers run this inside a `catch`,
 * so it never throws: a link whose property or text can't be read doesn't count, and a `cause`
 * that can't be read ends the walk.
 */
export function isAuthFailure(error: unknown): boolean {
  const seen = new Set<object>();
  let link: unknown = error;
  while (typeof link === "object" && link !== null && !seen.has(link) && seen.size < 16) {
    seen.add(link);
    // A Proxy's getPrototypeOf trap can make `instanceof` throw; such a link isn't an AuthError.
    try {
      if (link instanceof AuthError) return true;
    } catch {}
    const kind = read(link, "kind");
    if (kind === "auth") return true;
    for (const name of kind === "refused" ? [] : ["status", "code"]) {
      const status = read(link, name);
      if (typeof status !== "number" || !Number.isInteger(status)) continue;
      if (status === 401 || status === 403) return true;
      if (status === 400) {
        let text: string;
        try {
          text = String(link).toLowerCase();
        } catch {
          text = "";
        }
        if (text.includes("api_key_invalid")) return true;
      }
    }
    try {
      link = (link as { cause?: unknown }).cause;
    } catch {
      return false;
    }
  }
  return false;
}

/** Whether `value` is a non-empty string of visible ASCII (U+0021 to U+007E). */
export function isVisibleText(value: unknown): value is string {
  return typeof value === "string" && /^[\x21-\x7e]+$/.test(value);
}

const HEADERS_SHAPE = "Invalid connection headers: give a mapping of name to value";

/**
 * An API key staged for validation or read from the private store.
 *
 * `headers` are extra request headers sent beside the key, such as a gateway token. They are
 * secrets like the key. Given as a plain record (`undefined` entries skipped) or as an iterable of
 * name/value pairs such as a `Map`, they are kept as frozen pairs sorted by name. Throws
 * `AuthError` if the key is not visible text or a header cannot be sent; messages never echo a
 * value.
 */
export class ApiKeyCredentials {
  // Private, so JSON.stringify, inspect and spreads never show the secrets.
  readonly #apiKey: string;
  readonly #headers: readonly (readonly [string, string])[];

  constructor(apiKey: unknown, headers: unknown = []) {
    if (!isVisibleText(apiKey)) throw new AuthError("Invalid API key credential");
    let pairs: unknown[];
    if (isRecord(headers)) {
      pairs = Object.entries(headers).filter(([, value]) => value !== undefined);
    } else if (typeof headers === "object" && headers !== null && Symbol.iterator in headers) {
      pairs = Array.from(headers as Iterable<unknown>);
    } else {
      throw new AuthError(HEADERS_SHAPE);
    }
    if (
      !pairs.every(
        (pair): pair is [string, unknown] => Array.isArray(pair) && pair.length === 2 && typeof pair[0] === "string",
      )
    ) {
      throw new AuthError(HEADERS_SHAPE);
    }
    // A record keeps only one of two pairs with the same name, so count exact repeats before
    // building it.
    if (new Set(pairs.map(([name]) => name)).size !== pairs.length) {
      throw new AuthError("Invalid connection headers: a header name is given more than once");
    }
    let checked: Record<string, string>;
    try {
      // validateHeaders skips `undefined` values, but only a record's are absent; a pair's
      // `undefined` is a value, refused as `null` is.
      checked = validateHeaders(Object.fromEntries(pairs.map(([name, value]) => [name, value ?? null])), null);
    } catch (error) {
      if (error instanceof RangeError) throw new AuthError(`Invalid connection headers: ${error.message}`);
      throw error;
    }
    this.#apiKey = apiKey;
    this.#headers = Object.freeze(
      Object.entries(checked)
        .sort(([a], [b]) => compareCodeUnits(a, b))
        .map((pair) => Object.freeze(pair)),
    );
    Object.freeze(this);
  }

  /** The login method these credentials are for. */
  get method(): "api_key" {
    return "api_key";
  }

  /** The API key. */
  get apiKey(): string {
    return this.#apiKey;
  }

  /** The extra headers as frozen name/value pairs, sorted by name. */
  get headers(): readonly (readonly [string, string])[] {
    return this.#headers;
  }
}

/**
 * Subscription tokens and routing metadata returned by authentication.
 *
 * Throws `AuthError` unless the tokens and account id are visible text, `residency` is absent or
 * visible text, and `expiresAt` (epoch seconds) is a positive finite number.
 */
export class OAuthCredentials {
  // Private, so JSON.stringify, inspect and spreads never show the tokens.
  readonly #accessToken: string;
  readonly #refreshToken: string;
  /** When the access token expires, in epoch seconds. */
  readonly expiresAt: number;
  /** The subscription account id. */
  readonly accountId: string;
  /** The account's data residency region, if any. */
  readonly residency: string | null;

  constructor(fields: {
    accessToken: unknown;
    refreshToken: unknown;
    expiresAt: unknown;
    accountId: unknown;
    residency?: unknown;
  }) {
    const { accessToken, refreshToken, expiresAt, accountId } = fields;
    const residency = fields.residency ?? null;
    if (
      !isVisibleText(accessToken) ||
      !isVisibleText(refreshToken) ||
      !isVisibleText(accountId) ||
      (residency !== null && !isVisibleText(residency)) ||
      typeof expiresAt !== "number" ||
      !(expiresAt > 0 && expiresAt <= Number.MAX_VALUE)
    ) {
      throw new AuthError("Invalid subscription credentials");
    }
    this.#accessToken = accessToken;
    this.#refreshToken = refreshToken;
    this.expiresAt = expiresAt;
    this.accountId = accountId;
    this.residency = residency;
    Object.freeze(this);
  }

  /** The login method these credentials are for. */
  get method(): "subscription" {
    return "subscription";
  }

  /** The access token. */
  get accessToken(): string {
    return this.#accessToken;
  }

  /** The refresh token. */
  get refreshToken(): string {
    return this.#refreshToken;
  }
}

/** Credentials for one provider login. */
export type Credentials = ApiKeyCredentials | OAuthCredentials;

/** How a provider is logged in to; `"none"` is a provider saved without a credential. */
export type LoginMethod = "api_key" | "subscription" | "none";

/** Whether `value` is a `LoginMethod`. */
function isLoginMethod(value: unknown): value is LoginMethod {
  return value === "api_key" || value === "subscription" || value === "none";
}

/**
 * The nonsecret provider choice that a run may pin to one connection.
 *
 * `method` is `"none"` for a provider saved without a credential, such as a local server.
 * `endpoint` is the provider's base URL when it needs one, such as an Azure resource, and `null`
 * otherwise. `organization` and `project` are the OpenAI ids an `openai` API-key login sends.
 * `headerNames` names, sorted, the extra headers its API-key credential sends; their values stay
 * with the credential. Throws `AuthError` if a field is invalid.
 */
export class ProviderSelection {
  readonly provider: string;
  readonly method: LoginMethod;
  readonly model: string;
  readonly credentialId: string;
  readonly endpoint: string | null;
  readonly organization: string | null;
  readonly project: string | null;
  readonly headerNames: readonly string[];

  constructor(fields: {
    provider: unknown;
    method: unknown;
    model: unknown;
    credentialId: unknown;
    endpoint?: unknown;
    organization?: unknown;
    project?: unknown;
    headerNames?: unknown;
  }) {
    const { provider, method, model, credentialId } = fields;
    const endpoint = fields.endpoint ?? null;
    const organization = fields.organization ?? null;
    const project = fields.project ?? null;
    const headerNames = fields.headerNames ?? [];
    if (
      !isVisibleText(provider) ||
      !isVisibleText(model) ||
      !isVisibleText(credentialId) ||
      !isLoginMethod(method) ||
      (endpoint !== null && !isVisibleText(endpoint)) ||
      !Array.isArray(headerNames)
    ) {
      throw new AuthError("Invalid provider selection");
    }
    const openaiKey = provider === "openai" && method === "api_key";
    if (!openaiKey && (organization !== null || project !== null)) {
      throw new AuthError("organization and project are only for an openai API-key login");
    }
    if (method !== "api_key" && headerNames.length > 0) {
      throw new AuthError("Connection headers are only for an API-key login");
    }
    if (provider === "ollama" && headerNames.length > 0) {
      throw new AuthError("Connection headers are not sent to provider ollama");
    }
    let checkedOrganization: string | null;
    let checkedProject: string | null;
    let names: string[];
    try {
      checkedOrganization = organization === null ? null : validateOrganization(organization, "organization");
      checkedProject = project === null ? null : validateOrganization(project, "project");
      names = headerNames.map((name: unknown) => validateHeaderName(name, provider));
      if (new Set(names.map((name) => name.toLowerCase())).size !== names.length) {
        throw new RangeError("a header name is given more than once (names ignore case)");
      }
      const limit = current().maxConnectionHeaders;
      if (names.length > limit) throw new RangeError(`headers must have at most ${limit} headers`);
    } catch (error) {
      if (error instanceof RangeError) throw new AuthError(`Invalid provider selection: ${error.message}`);
      throw error;
    }
    this.provider = provider;
    this.method = method;
    this.model = model;
    this.credentialId = credentialId;
    this.endpoint = endpoint;
    this.organization = checkedOrganization;
    this.project = checkedProject;
    this.headerNames = Object.freeze(names.sort(compareCodeUnits));
    Object.freeze(this);
  }
}

/** A detached, nonsecret snapshot of saved provider choices. */
export interface AuthState {
  readonly defaultProvider: string | null;
  readonly selections: readonly ProviderSelection[];
}

/** A decoded HTTP result; the body may contain credentials, so it stays out of inspection. */
export class HttpResponse {
  /** The HTTP status. */
  readonly status: number;
  // Private, so JSON.stringify, inspect and spreads never show the body.
  readonly #data: unknown;

  constructor(status: number, data: unknown) {
    this.status = status;
    this.#data = data;
    Object.freeze(this);
  }

  /** The decoded JSON body, or `null` when the body wasn't JSON or the reply was a redirect. */
  get data(): unknown {
    return this.#data;
  }
}
